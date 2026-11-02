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
from .features import *
from .baselines import *
from .deep_svdd import *
from .experiments import *
from .reporting import *
from .benchmarking import *

def parse_args():
    parser = argparse.ArgumentParser(description="Final ShipsEar LOCO anomaly-detection pipeline.")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ast_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--snrs", type=float, nargs="+", default=list(DEFAULT_SNRS))
    parser.add_argument("--train_snrs",type=float,nargs="+",default=list(DEFAULT_SNRS),)

    parser.add_argument("--worker_id", type=int, default=-1)
    parser.add_argument("--num_experiment_workers", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ast_batch_size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_noisy", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=0.1)

    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--run_deep_svdd", action="store_true")
    parser.add_argument("--run_all", action="store_true")
    # [NEW - R2-2] Enables two additional asymmetric objectives.
    parser.add_argument("--run_asymmetric", action="store_true")
    parser.add_argument("--run_geometry_comparison", action="store_true",
                        help="Compare noisy-only clean/noisy center and radius in a 2x2 factorial design.")
    # [NEW - REVIEWER] Opt-in additions. Original run grid unchanged unless enabled.
    parser.add_argument("--run_mfcc_svdd", action="store_true")
    parser.add_argument("--only_new_baselines", action="store_true")
    parser.add_argument("--run_pink", action="store_true")
    parser.add_argument("--run_sensitivity", action="store_true")
    parser.add_argument("--run_eigen", action="store_true")
    parser.add_argument("--benchmark_cost", action="store_true")
    parser.add_argument("--sensitivity_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sensitivity_classes", type=str, nargs="*", default=[])
    parser.add_argument("--sensitivity_snr", type=float, default=10.0)
    parser.add_argument("--rep_dim", type=int, default=128)
    parser.add_argument("--radius_update_epochs", type=int, nargs="+", default=[10, 15, 20, 25, 30])

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_skip_completed", action="store_true")
    parser.add_argument("--allow_non_12_classes", action="store_true")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    cfg = ExperimentConfig(
        data_root=args.data_root,
        ast_model_path=args.ast_model_path,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        snrs=tuple(args.snrs),
        train_snrs=tuple(args.train_snrs),
        device=args.device,
        worker_id=args.worker_id,
        num_experiment_workers=args.num_experiment_workers,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        ast_batch_size=args.ast_batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        nu=args.nu,
        lr=args.lr,
        lambda_noisy=args.lambda_noisy,
        lambda_cons=args.lambda_cons,
        amp=not args.no_amp,
        skip_completed=not args.no_skip_completed,
        run_asymmetric=args.run_asymmetric,
        run_geometry_comparison=args.run_geometry_comparison,
        run_mfcc_svdd=args.run_mfcc_svdd,
        only_new_baselines=args.only_new_baselines,
        run_pink=args.run_pink,
        run_sensitivity=args.run_sensitivity,
        run_eigen=args.run_eigen,
        sensitivity_seeds=tuple(args.sensitivity_seeds),
        sensitivity_classes=tuple(args.sensitivity_classes),
        sensitivity_snr=args.sensitivity_snr,
        rep_dim=args.rep_dim,
        radius_update_epochs=tuple(args.radius_update_epochs),)

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    manifest = build_manifest(Path(cfg.data_root))
    manifest.to_csv(output_root / "dataset_manifest.csv", index=False)

    loco_counts = build_loco_counts(manifest)
    loco_counts.to_csv(output_root / "loco_sample_counts.csv", index=False)
    # [NEW - R2-3] Segment-level statistics; no change to LOCO split.
    if cfg.worker_id in (-1, 0):
        write_dataset_statistics(manifest, output_root, cfg.target_sr)

    n_classes = manifest["class_name"].nunique()

    print("\nDataset class counts:")
    print(manifest["class_name"].value_counts().sort_index().to_string())
    print(f"\nTotal clips: {len(manifest)}")
    print(f"Classes: {n_classes}")

    if n_classes != 12 and not args.allow_non_12_classes:
        raise RuntimeError(
            f"Expected 12 classes for the final ShipsEar experiment, found {n_classes}. "
            "Use --allow_non_12_classes only for debugging.")

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(cfg.device)

    print("\nLoading frozen AST...")
    ast_encoder = FrozenASTEncoder(cfg.ast_model_path,device=device,local_files_only=cfg.local_files_only,)

    print(f"AST hidden size: {ast_encoder.hidden_size}")

    if ast_encoder.hidden_size != 768:
        warnings.warn(
            f"AST checkpoint hidden size is {ast_encoder.hidden_size}, not 768. "
            "The projection will use the actual checkpoint hidden size.")

    with open(output_root / "ast_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ast_encoder.metadata(),f,indent=2,ensure_ascii=False,default=str,)

    feature_manager = FeatureManager(cfg,manifest,ast_encoder,)
    if args.benchmark_cost and cfg.worker_id in (-1, 0):
        benchmark_feature_extraction(cfg, manifest, feature_manager, device, output_root)

    run_baselines = args.run_baselines or args.run_all
    run_deep_svdd_flag = args.run_deep_svdd or args.run_all

    if not run_baselines and not run_deep_svdd_flag and not cfg.run_sensitivity:
        raise RuntimeError("Nothing selected. Use --run_baselines, --run_deep_svdd, or --run_all.")

    if run_baselines:
        run_all_baselines(cfg,manifest,feature_manager,)

    if run_deep_svdd_flag:
        run_all_deep_svdd(cfg,manifest,feature_manager,device,)
    if cfg.run_sensitivity:
        run_sensitivity_grid(cfg, manifest, feature_manager, device)

    # Rebuild summaries from all completed jobs so results generated by
    # multiple worker processes can be merged after completion.
    baseline_metrics = collect_existing_metrics(output_root / "baselines")

    if not baseline_metrics.empty:
        summarize_results(baseline_metrics,output_root / "summary" / "baselines", cfg.bootstrap_repeats)
    svdd_metrics = collect_existing_metrics(output_root / "deep_svdd")

    if not svdd_metrics.empty:
        summarize_results(svdd_metrics,output_root / "summary" / "deep_svdd", cfg.bootstrap_repeats)

    sensitivity_metrics = collect_existing_metrics(output_root / "sensitivity")
    if not sensitivity_metrics.empty:
        summarize_results(sensitivity_metrics, output_root / "summary" / "sensitivity")
    print("\nFinished.")



