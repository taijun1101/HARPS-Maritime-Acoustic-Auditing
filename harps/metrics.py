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

def precision_recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    k_eff = min(int(k), len(scores))
    order = np.argsort(-scores, kind="stable")
    top = order[:k_eff]
    tp = int(y_true[top].sum())

    precision = tp / k_eff if k_eff else np.nan
    n_anomalies = int(y_true.sum())
    recall = tp / n_anomalies if n_anomalies else np.nan
    return float(precision), float(recall)


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Evaluation requires both labels 0 and 1.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite anomaly scores detected.")

    result = {"auroc": float(roc_auc_score(y_true, scores)),
              "auprc": float(average_precision_score(y_true, scores)),
              "anomaly_prevalence": float(y_true.mean())}
    for percent in (5, 10):
        budget = max(1, math.ceil(len(scores) * percent / 100.0))
        p, r = precision_recall_at_k(y_true, scores, budget)
        result[f"precision_at_{percent}pct"] = p
        result[f"recall_at_{percent}pct"] = r
        result[f"budget_at_{percent}pct"] = budget

    for k in ks:
        precision, recall = precision_recall_at_k(y_true, scores, k)
        result[f"precision_at_{k}"] = precision
        result[f"recall_at_{k}"] = recall

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    result.update({
        "mean_normal_score": float(np.mean(normal_scores)),
        "mean_anomaly_score": float(np.mean(anomaly_scores)),
        "median_normal_score": float(np.median(normal_scores)),
        "median_anomaly_score": float(np.median(anomaly_scores)),
        "reverse_auroc_diagnostic": float(roc_auc_score(y_true, -scores)),})
    return result

def prevalence_metrics(y_true: np.ndarray, scores: np.ndarray, prevalence: float = 0.01) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if not (0 < prevalence < 1) or min(n0, n1) == 0:
        raise ValueError("Prevalence analysis requires both classes and 0<prevalence<1")
    weights = np.where(y == 1, prevalence / n1, (1.0 - prevalence) / n0)
    order = np.argsort(-scores, kind="stable")
    cumulative_mass = np.cumsum(weights[order])
    out = {"target_prevalence": prevalence,
           "weighted_auprc": float(average_precision_score(y, scores, sample_weight=weights))}
    for pct in (5, 10):
        budget = pct / 100.0
        # Partial final item: exactly 5%/10% weighted archive mass.
        prev = np.r_[0.0, cumulative_mass[:-1]]
        included = np.clip((budget - prev) / weights[order], 0.0, 1.0)
        taken = weights[order] * included
        tp = float(np.sum(taken * y[order]))
        out[f"precision_at_{pct}pct"] = tp / budget
        out[f"recall_at_{pct}pct"] = tp / prevalence
    return out


def make_prediction_dataframe(
    manifest: pd.DataFrame,
    anomaly_class: str,
    scores: np.ndarray,
) -> pd.DataFrame:
    df = manifest.copy()
    df["binary_label"] = (df["class_name"] == anomaly_class).astype(int)
    df["anomaly_score"] = np.asarray(scores, dtype=float)

    order = np.argsort(-df["anomaly_score"].to_numpy(), kind="stable")
    ranks = np.empty(len(df), dtype=int)
    ranks[order] = np.arange(1, len(df) + 1)
    df["rank"] = ranks

    for k in DEFAULT_KS:
        df[f"is_top{k}"] = (df["rank"] <= k).astype(int)

    return df


def evaluation_condition_name(condition: str, snr_db: Optional[float]) -> str:
    if condition == "clean":
        return "clean"
    return f"{'pink_' if condition == 'pink' else ''}snr{snr_db:g}"


# ============================================================
# Baseline experiment
# ============================================================

