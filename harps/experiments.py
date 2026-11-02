from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTModel
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL

from .config import *
from .dataset import *
from .noise import *
from .features import *
from .metrics import *
from .baselines import *
from .deep_svdd import *
from .reporting import *

def job_belongs_to_worker(
    job_index: int,
    worker_id: int,
    num_experiment_workers: int,
) -> bool:
    if worker_id < 0:
        return True

    if not 0 <= worker_id < num_experiment_workers:
        raise ValueError(f"worker_id must be -1 or in [0, {num_experiment_workers - 1}]")

    return job_index % num_experiment_workers == worker_id


def run_all_baselines(
    cfg: ExperimentConfig,
    manifest: pd.DataFrame,
    feature_manager: FeatureManager,
) -> pd.DataFrame:
    classes = sorted(manifest["class_name"].unique())
    jobs = []

    for seed in cfg.seeds:
        for anomaly_class in classes:
            for feature_type in FEATURE_NAMES:
                for baseline_name in BASELINE_NAMES:
                    if cfg.only_new_baselines and baseline_name not in ("knn", "mahalanobis"):
                        continue
                    jobs.append((seed, anomaly_class, feature_type, baseline_name, "clean", None))

                    for snr_db in cfg.snrs:
                        jobs.append(
                            (
                                seed,
                                anomaly_class,
                                feature_type,
                                baseline_name,
                                "noisy",
                                snr_db,))
                        # [NEW - R2-4] Mismatched pink-noise evaluation.
                        if cfg.run_pink:
                            jobs.append((seed, anomaly_class, feature_type,
                                         baseline_name, "pink", snr_db))

    rows = []

    for job_index, job in enumerate(jobs):
        if not job_belongs_to_worker(
            job_index,
            cfg.worker_id,
            cfg.num_experiment_workers,):
            continue

        (
            seed,
            anomaly_class,
            feature_type,
            baseline_name,
            eval_condition,
            eval_snr_db,
        ) = job

        condition_name = evaluation_condition_name(eval_condition, eval_snr_db)

        output_dir = (
            Path(cfg.output_dir)
            / "baselines"
            / feature_type
            / baseline_name
            / f"seed_{seed}"
            / f"anomaly_{anomaly_class}"
            / condition_name)

        metrics_path = output_dir / "metrics.csv"

        if cfg.skip_completed and metrics_path.exists():
            print(f"[SKIP] {metrics_path}")
            rows.append(pd.read_csv(metrics_path).iloc[0].to_dict())
            continue

        print(
            f"\n[BASELINE {job_index + 1}/{len(jobs)}] "
            f"seed={seed} anomaly={anomaly_class} feature={feature_type} "
            f"model={baseline_name} eval={condition_name}")

        try:
            row = run_baseline_loco(
                cfg=cfg,
                manifest=manifest,
                feature_manager=feature_manager,
                anomaly_class=anomaly_class,
                feature_type=feature_type,
                baseline_name=baseline_name,
                seed=seed,
                eval_condition=eval_condition,
                eval_snr_db=eval_snr_db,
                output_dir=output_dir,)
            rows.append(row)

        except Exception as exc:
            warnings.warn(f"Baseline failed: {job}\n{type(exc).__name__}: {exc}")

    return pd.DataFrame(rows)


def run_all_deep_svdd(
    cfg: ExperimentConfig,
    manifest: pd.DataFrame,
    feature_manager: FeatureManager,
    device: torch.device,
) -> pd.DataFrame:
    classes = sorted(manifest["class_name"].unique())
    jobs = []

    for seed in cfg.seeds:
        for anomaly_class in classes:
            jobs.append((seed, anomaly_class, "clean", None, "ast"))

            for train_snr_db in cfg.train_snrs:
                jobs.append((seed, anomaly_class, "noisy", train_snr_db, "ast"))
                jobs.append((seed, anomaly_class, "paired_no_consistency", train_snr_db, "ast"))
                jobs.append((seed, anomaly_class, "paired_consistency", train_snr_db, "ast"))
                # [NEW - R2-2] Added only when --run_asymmetric is supplied.
                if cfg.run_geometry_comparison:
                    for mode in ("noisy_sb_center_noisy_radius_clean", "noisy_sb_center_clean_radius_noisy", "noisy_sb_center_noisy_radius_noisy"):
                        jobs.append((seed, anomaly_class, mode, train_snr_db, "ast"))
                if cfg.run_asymmetric:
                    jobs.append((seed, anomaly_class, "paired_asymmetric_no_consistency", train_snr_db, "ast"))
                    jobs.append((seed, anomaly_class, "paired_asymmetric", train_snr_db, "ast"))
            # [NEW - R1-4] MFCC+SVDD is an additional feature ablation.
            if cfg.run_mfcc_svdd:
                jobs.append((seed, anomaly_class, "clean", None, "mfcc"))
                for train_snr_db in cfg.train_snrs:
                    jobs.append((seed, anomaly_class, "paired_consistency", train_snr_db, "mfcc"))
                    if cfg.run_asymmetric:
                        jobs.append((seed, anomaly_class, "paired_asymmetric", train_snr_db, "mfcc"))

    rows = []

    for job_index, job in enumerate(jobs):
        if not job_belongs_to_worker(
            job_index,
            cfg.worker_id,
            cfg.num_experiment_workers,):
            continue

        seed, anomaly_class, training_mode, train_snr_db, feature_type = job

        train_name = (
            "clean"
            if training_mode == "clean"
            else f"{training_mode}_snr{train_snr_db:g}")

        model_dir = (
            Path(cfg.output_dir)
            / "deep_svdd"
            / ("mfcc" if feature_type == "mfcc" else "ast")
            / train_name
            / f"seed_{seed}"
            / f"anomaly_{anomaly_class}")

        success_path = model_dir / "_SUCCESS.json"

        if cfg.skip_completed and success_path.exists():
            expected = ["clean"] + [f"snr{x:g}" for x in cfg.snrs]
            if cfg.run_pink:
                expected += [f"pink_snr{x:g}" for x in cfg.snrs]
            missing = [x for x in expected if not (model_dir / "evaluation" / x / "metrics.csv").exists()]
            if not missing:
                print(f"[SKIP] {success_path}")
                for metrics_path in sorted((model_dir / "evaluation").glob("*/metrics.csv")):
                    rows.append(pd.read_csv(metrics_path).iloc[0].to_dict())
                continue
            # [NEW - R2-4] Evaluate newly requested conditions from saved model;
            # do NOT retrain and overwrite a completed model.
            print(f"[EVALUATE SAVED MODEL] {model_dir} missing={missing}")
            checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
            dim = int(checkpoint["projection"]["input_dim"])
            out_dim = int(checkpoint["projection"]["output_dim"])
            model = ProjectionHead(input_dim=dim, rep_dim=out_dim).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            center = checkpoint["center"].to(device)
            feature_scaler = (joblib.load(model_dir / "feature_scaler.joblib")
                              if feature_type == "mfcc" else None)
            evaluate_deep_svdd(cfg, manifest, feature_manager, model, center,
                               anomaly_class, seed, training_mode, train_snr_db,
                               device, model_dir, feature_type, feature_scaler)
            if cfg.run_eigen:
                write_eigen_analysis(model, model_dir / "projection_eigenvalues.csv")
            for metrics_path in sorted((model_dir / "evaluation").glob("*/metrics.csv")):
                rows.append(pd.read_csv(metrics_path).iloc[0].to_dict())
            continue

        print(
            f"\n[DEEP SVDD {job_index + 1}/{len(jobs)}] "
            f"seed={seed} anomaly={anomaly_class} train={train_name}")

        model, center, radius, _ = train_deep_svdd(
            cfg=cfg,
            manifest=manifest,
            feature_manager=feature_manager,
            anomaly_class=anomaly_class,
            seed=seed,
            training_mode=training_mode,
            train_snr_db=train_snr_db,
            device=device,
            output_dir=model_dir,
            feature_type=feature_type,)

        feature_scaler = (joblib.load(model_dir / "feature_scaler.joblib")
                          if feature_type == "mfcc" else None)
        rows.extend(
            evaluate_deep_svdd(
                cfg=cfg,
                manifest=manifest,
                feature_manager=feature_manager,
                model=model,
                center=center,
                anomaly_class=anomaly_class,
                seed=seed,
                training_mode=training_mode,
                train_snr_db=train_snr_db,
                device=device,
                model_dir=model_dir,
                feature_type=feature_type,
                feature_scaler=feature_scaler,))

        if cfg.run_eigen:
            write_eigen_analysis(model, model_dir / "projection_eigenvalues.csv")
        with open(success_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "held_out_class": anomaly_class,
                    "training_mode": training_mode,
                    "train_snr_db": train_snr_db,
                    "radius": radius,},
                f,
                indent=2,
                ensure_ascii=False,)

    return pd.DataFrame(rows)


# [NEW - R2-3] Separate one-factor-at-a-time sensitivity grid.
# Defaults are predeclared, not chosen using held-out anomaly performance.
def run_sensitivity_grid(cfg: ExperimentConfig, manifest: pd.DataFrame,
                         fm: FeatureManager, device: torch.device) -> None:
    from dataclasses import replace
    classes = list(cfg.sensitivity_classes) or sorted(manifest.class_name.unique())
    unknown = set(classes) - set(manifest.class_name.unique())
    if unknown:
        raise ValueError(f"Unknown sensitivity classes: {unknown}")
    cases = [("reference", "default", cfg)]
    for key, values in (("lambda_noisy", (0.25, 0.5, 2.0)),
                        ("lambda_cons", (0.0, 0.01, 1.0)),
                        ("rep_dim", (64, 256)),
                        ("nu", (0.05, 0.2)),
                        ("radius_update_epochs", ((6, 10, 15, 20, 25, 30), (15, 20, 25, 30)))):
        for value in values:
            cases.append((key, str(value).replace(" ", ""), replace(cfg, **{key: value})))
    jobs = [(seed, cls, param, val, scfg)
            for seed in cfg.sensitivity_seeds for cls in classes
            for param, val, scfg in cases]
    for job_idx, (seed, cls, param, val, scfg) in enumerate(jobs):
        if not job_belongs_to_worker(job_idx, cfg.worker_id, cfg.num_experiment_workers):
            continue
        root = Path(cfg.output_dir) / "sensitivity" / param / val / f"seed_{seed}" / f"anomaly_{cls}"
        success = root / "_SUCCESS.json"
        if cfg.skip_completed and success.exists():
            continue
        model, center, radius, _ = train_deep_svdd(
            scfg, manifest, fm, cls, seed, "paired_asymmetric",
            cfg.sensitivity_snr, device, root)
        # Keep sensitivity evaluation on original white-noise grid, not pink.
        scfg_eval = replace(scfg, run_pink=False)
        evaluate_deep_svdd(scfg_eval, manifest, fm, model, center, cls, seed,
                           "paired_asymmetric", cfg.sensitivity_snr, device, root)
        for csv_path in (root / "evaluation").glob("*/metrics.csv"):
            metrics_df = pd.read_csv(csv_path)
            metrics_df["sensitivity_parameter"] = param
            metrics_df["sensitivity_value"] = val
            metrics_df.to_csv(csv_path, index=False)
        success.write_text(json.dumps({"parameter": param, "value": val,
                                       "seed": seed, "held_out_class": cls}), encoding="utf-8")


