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
from .metrics import *

# ============================================================
# Summaries
# ============================================================

METRIC_COLUMNS = [
    "auroc", "auprc", "precision_at_5pct", "recall_at_5pct",
    "precision_at_10pct", "recall_at_10pct",
    "prevalence_1pct_weighted_auprc",
    "prevalence_1pct_precision_at_5pct", "prevalence_1pct_recall_at_5pct",
    "prevalence_1pct_precision_at_10pct", "prevalence_1pct_recall_at_10pct",
    "precision_at_10",
    "recall_at_10",
    "precision_at_30",
    "recall_at_30",
    "precision_at_50",
    "recall_at_50",
    "precision_at_100",
    "recall_at_100",]


def summarize_results(df: pd.DataFrame, output_dir: Path, bootstrap_repeats: int = 2000) -> None:
    if df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_task_metrics.csv", index=False)
    # [MODIFIED - R1-1] Legacy metrics remain, new metrics added when present.
    available_metrics = [m for m in METRIC_COLUMNS if m in df.columns]

    group_cols = [
        "method", "feature", "training_mode", "train_snr_db",
        "eval_condition", "eval_snr_db"]
    # [NEW - R2-3] Keep sensitivity configurations separate in aggregation.
    if "sensitivity_parameter" in df.columns:
        group_cols += ["sensitivity_parameter", "sensitivity_value"]

    # Variation across seeds for each held-out class.
    per_class = (
        df.groupby(group_cols + ["held_out_class"],dropna=False,)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index())
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    # Variation across LOCO classes for each seed.
    per_seed = (
        df.groupby(group_cols + ["seed"],dropna=False,)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index())
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)

    # Main summary:
    # 1) average the five seeds within each held-out class;
    # 2) compute mean/std across the held-out classes.
    class_means = (df.groupby(group_cols + ["held_out_class"],dropna=False,)[available_metrics]
        .mean()
        .reset_index())

    overall = (
        class_means.groupby(group_cols,dropna=False,)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index())
    overall.to_csv(output_dir / "overall_metrics_across_loco.csv",index=False,)
    if "sensitivity_parameter" not in df.columns:
        summarize_uncertainty(df, output_dir, bootstrap_repeats)
        if ("method" in df.columns and (df["method"] == "deep_svdd").any()):
            summarize_wilcoxon(df, output_dir)


# [NEW - R2-4] Class-cluster bootstrap and seed-level variability.
def summarize_uncertainty(df: pd.DataFrame, output_dir: Path, bootstrap_repeats: int = 2000) -> None:
    if df.empty:
        return
    group_cols = ["method", "feature", "training_mode", "train_snr_db", "eval_condition", "eval_snr_db"]
    metrics = [m for m in ("auroc", "auprc", "precision_at_5pct", "recall_at_5pct",
                           "precision_at_10pct", "recall_at_10pct") if m in df]
    seed_means = df.groupby(group_cols + ["seed"], dropna=False)[metrics].mean().reset_index()
    seed_sd = seed_means.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    seed_sd.to_csv(output_dir / "seed_uncertainty.csv", index=False)
    class_means = df.groupby(group_cols + ["held_out_class"], dropna=False)[metrics].mean().reset_index()
    rng = np.random.default_rng(2026)
    records = []
    for key, g in class_means.groupby(group_cols, dropna=False):
        for metric in metrics:
            values = g[metric].dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            indices = rng.integers(0, len(values), size=(bootstrap_repeats, len(values)))
            boot = values[indices].mean(axis=1)
            records.append(dict(zip(group_cols, key), metric=metric, n_classes=len(values),
                                mean=float(values.mean()),
                                ci_lower=float(np.quantile(boot, .025)),
                                ci_upper=float(np.quantile(boot, .975)),
                                ci_method="class_cluster_bootstrap"))
    pd.DataFrame(records).to_csv(output_dir / "confidence_intervals.csv", index=False)


# [NEW - R2-4] Wilcoxon uses paired class means, never 60 seeds as independent.
def summarize_wilcoxon(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty or "training_mode" not in df:
        return
    cases = [("paired_consistency", "paired_no_consistency"),
             ("paired_asymmetric", "paired_consistency"),
             ("paired_asymmetric", "paired_asymmetric_no_consistency")]
    keys = ["feature", "train_snr_db", "eval_condition", "eval_snr_db"]
    means = df[df.method == "deep_svdd"].groupby(keys + ["training_mode", "held_out_class"], dropna=False)["auroc"].mean().reset_index()
    rows = []
    for key, g in means.groupby(keys, dropna=False):
        for a, b in cases:
            ga = g[g.training_mode == a].set_index("held_out_class")["auroc"]
            gb = g[g.training_mode == b].set_index("held_out_class")["auroc"]
            shared = ga.index.intersection(gb.index)
            if len(shared) < 3:
                continue
            delta = ga.loc[shared].to_numpy() - gb.loc[shared].to_numpy()
            p = 1.0 if np.allclose(delta, 0) else float(wilcoxon(delta).pvalue)
            rows.append(dict(zip(keys, key), method_a=a, method_b=b,
                             n_classes=len(shared), mean_difference=float(delta.mean()), p_value=p))
    result = pd.DataFrame(rows)
    if not result.empty:
        # Holm adjustment within each evaluation/feature/SNR family.
        result["p_holm"] = np.nan
        for _, idx in result.groupby(keys, dropna=False).groups.items():
            ids = list(idx)
            ordered = sorted(ids, key=lambda i: result.at[i, "p_value"])
            running = 0.0
            for rank, i in enumerate(ordered):
                running = max(running, min(1.0, (len(ordered) - rank) * result.at[i, "p_value"]))
                result.at[i, "p_holm"] = running
    result.to_csv(output_dir / "paired_wilcoxon_tests.csv", index=False)


# [NEW - R2-3] Dataset counts; recording IDs are NOT inferred from segment filenames.
def write_dataset_statistics(manifest: pd.DataFrame, output_root: Path, target_sr: int) -> None:
    counts = manifest.groupby("class_name").size().rename("n_segments").reset_index()
    counts["percentage"] = counts.n_segments / len(manifest) * 100
    counts["loco_reference_count"] = len(manifest) - counts.n_segments
    counts["loco_anomaly_count"] = counts.n_segments
    counts["recording_count"] = np.nan
    counts["recording_id_status"] = "not_available_in_manifest"
    counts.to_csv(output_root / "dataset_statistics.csv", index=False)
    # Duration from audio header without decoding whole waveform when soundfile supports it.
    try:
        import soundfile as sf
        rows = []
        for row in manifest.itertuples(index=False):
            info = sf.info(row.file_path)
            rows.append({"sample_id": row.sample_id, "class_name": row.class_name,
                         "duration_seconds": float(info.duration),
                         "original_sample_rate": int(info.samplerate)})
        pd.DataFrame(rows).to_csv(output_root / "audio_duration_statistics.csv", index=False)
    except Exception as exc:
        warnings.warn(f"Audio duration statistics unavailable: {exc}")


# [NEW - R2-6] Spectral properties of W.T @ W; no refitting required.
def write_eigen_analysis(model: nn.Module, path: Path) -> None:
    w = model.projection.weight.detach().cpu().numpy().astype(np.float64)
    # Nonzero eigenvalues of W.T W equal those of W W.T.
    eigenvalues = np.linalg.eigvalsh(w @ w.T)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    mass = eigenvalues.sum()
    proportions = eigenvalues / mass if mass > 0 else np.zeros_like(eigenvalues)
    pd.DataFrame({"rank": np.arange(1, len(eigenvalues) + 1),
                  "eigenvalue": eigenvalues,
                  "explained_metric_weight": proportions,
                  "cumulative_metric_weight": np.cumsum(proportions)}).to_csv(path, index=False)
    # [NEW - R2-6] Top right singular vectors are the actual eigen-directions
    # of W.T @ W in the original feature space (sign is arbitrary).
    _, _, vh = np.linalg.svd(w, full_matrices=False)
    top = min(10, len(vh))
    pd.DataFrame(vh[:top].T,
                 columns=[f"direction_{i+1}" for i in range(top)]).to_csv(
                     path.with_name("projection_top_eigendirections.csv"), index_label="feature_index")


def collect_existing_metrics(root: Path) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()

    rows = []

    for path in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(path)
            if len(df) == 1 and "auroc" in df.columns:
                rows.append(df.iloc[0].to_dict())
        except Exception:
            pass

    return pd.DataFrame(rows)


# [NEW - R2-5] End-to-end frozen-feature timing on a fixed clean subset.
# Includes waveform loading + AST frontend + AST forward; separate from head timing.
