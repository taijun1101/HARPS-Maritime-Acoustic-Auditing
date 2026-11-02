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

def _supported_kwargs(cls, kwargs: Dict) -> Dict:
    signature = inspect.signature(cls.__init__)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_baseline(name: str, seed: int, input_dim: int):
    if name == "knn":
        return KNNReferenceScorer(k=5), {"k": 5, "self_match": "excluded"}
    if name == "mahalanobis":
        return ShrinkageMahalanobisScorer(), {"covariance": "LedoitWolf"}
    if name == "iforest":
        kwargs = _supported_kwargs(IForest,{"contamination": 0.1, "random_state": seed},)
        return IForest(**kwargs), kwargs

    if name == "ocsvm":
        kwargs = _supported_kwargs(OCSVM,{"kernel": "rbf", "nu": 0.1, "contamination": 0.1},)
        return OCSVM(**kwargs), kwargs

    if name == "ae":
        if AutoEncoder is None:
            raise RuntimeError("PyOD AutoEncoder is unavailable.")

        h1 = max(32, input_dim // 2)
        h2 = max(16, input_dim // 4)
        kwargs = _supported_kwargs(
            AutoEncoder,
            {
                "hidden_neuron_list": [input_dim, h1, h2, h1, input_dim],
                "hidden_neurons": [h1, h2, h1],
                "epoch_num": 30,
                "epochs": 30,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "contamination": 0.1,
                "verbose": 0,
                "random_state": seed,},)
        return AutoEncoder(**kwargs), kwargs

    if name == "vae":
        if VAE is None:
            raise RuntimeError("PyOD VAE is unavailable.")

        h1 = max(32, input_dim // 2)
        h2 = max(16, input_dim // 4)
        kwargs = _supported_kwargs(
            VAE,
            {
                "encoder_neuron_list": [h1, h2],
                "decoder_neuron_list": [h2, h1],
                "encoder_neurons": [h1, h2],
                "decoder_neurons": [h2, h1],
                "latent_dim": 16,
                "epoch_num": 30,
                "epochs": 30,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "contamination": 0.1,
                "verbose": 0,
                "random_state": seed,},)
        return VAE(**kwargs), kwargs

    if name == "so_gaal":
        if SO_GAAL is None:
            raise RuntimeError("SO_GAAL is unavailable in the installed PyOD version.")

        kwargs = _supported_kwargs(SO_GAAL,{"contamination": 0.1, "stop_epochs": 20},)
        return SO_GAAL(**kwargs), kwargs

    if name == "mo_gaal":
        if MO_GAAL is None:
            raise RuntimeError("MO_GAAL is unavailable in the installed PyOD version.")

        kwargs = _supported_kwargs(MO_GAAL,{"contamination": 0.1, "stop_epochs": 20},)
        return MO_GAAL(**kwargs), kwargs

    raise ValueError(f"Unknown baseline: {name}")


class KNNReferenceScorer:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X):
        self.X = np.asarray(X, dtype=np.float64)
        self.nn = NearestNeighbors(n_neighbors=min(len(self.X), self.k + 1), metric="euclidean").fit(self.X)
        return self

    def decision_function(self, X, query_reference_indices=None):
        X = np.asarray(X, dtype=np.float64)
        distances, indices = self.nn.kneighbors(X)
        result = np.empty(len(X), dtype=float)
        for i, (ds, ids) in enumerate(zip(distances, indices)):
            if query_reference_indices is not None and query_reference_indices[i] >= 0:
                ds = ds[ids != query_reference_indices[i]]
            result[i] = float(np.mean(ds[:self.k])) if len(ds) else np.nan
        return result


class ShrinkageMahalanobisScorer:
    def fit(self, X):
        self.estimator = LedoitWolf().fit(np.asarray(X, dtype=np.float64))
        return self

    def decision_function(self, X):
        delta = np.asarray(X, dtype=np.float64) - self.estimator.location_
        return np.einsum("ij,jk,ik->i", delta, self.estimator.precision_, delta)


# ============================================================
# Metrics
# ============================================================

def run_baseline_loco(
    cfg: ExperimentConfig,
    manifest: pd.DataFrame,
    feature_manager: FeatureManager,
    anomaly_class: str,
    feature_type: str,
    baseline_name: str,
    seed: int,
    eval_condition: str,
    eval_snr_db: Optional[float],
    output_dir: Path,
) -> Dict:
    set_global_seed(seed)

    reference_mask = (manifest["class_name"] != anomaly_class).to_numpy()
    y_eval = (manifest["class_name"] == anomaly_class).astype(int).to_numpy()

    # All conventional baselines are trained on CLEAN reference normals.
    X_clean = feature_manager.get_eval_features(feature_type, "clean", seed)
    X_ref = X_clean[reference_mask]

    X_eval = feature_manager.get_eval_features(feature_type,eval_condition,seed,eval_snr_db,)

    # Fit preprocessing ONLY on reference normals.
    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)
    X_eval_scaled = scaler.transform(X_eval)

    model, model_kwargs = build_baseline(baseline_name,seed=seed,input_dim=X_ref_scaled.shape[1],)

    start = time.time()
    model.fit(X_ref_scaled)
    train_seconds = time.time() - start

    start = time.time()
    # PyOD convention: larger decision_function score = more abnormal.
    if baseline_name == "knn":
        # [MODIFIED - R2-2] Exclude identity match for enrolled references.
        ref_ids = np.full(len(manifest), -1, dtype=int)
        ref_ids[np.flatnonzero(reference_mask)] = np.arange(reference_mask.sum())
        scores = np.asarray(model.decision_function(X_eval_scaled,
                             query_reference_indices=ref_ids), dtype=np.float64)
    else:
        scores = np.asarray(model.decision_function(X_eval_scaled), dtype=np.float64)
    inference_seconds = time.time() - start

    metrics = evaluate_scores(y_eval, scores)
    # [NEW - R1-1] Fixed 1% simulated anomaly prevalence.
    prevalence = prevalence_metrics(y_eval, scores)
    metrics.update({f"prevalence_1pct_{k}": v for k, v in prevalence.items()})
    metrics.update({
        "method": baseline_name,
        "feature": feature_type,
        "training_mode": "clean",
        "train_snr_db": np.nan,
        "seed": seed,
        "held_out_class": anomaly_class,
        "eval_condition": eval_condition,
        "eval_snr_db": eval_snr_db,
        "n_reference": int(reference_mask.sum()),
        "n_eval_normal": int((y_eval == 0).sum()),
        "n_eval_anomaly": int((y_eval == 1).sum()),
        "n_eval_total": len(y_eval),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_segment": inference_seconds / len(X_eval_scaled),
        "cost_scope": "feature-space only; excludes feature extraction",})

    output_dir.mkdir(parents=True, exist_ok=True)
    make_prediction_dataframe(manifest, anomaly_class, scores).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": baseline_name,
                "feature": feature_type,
                "training_condition": "clean_reference_only",
                "score_convention": "larger_is_more_anomalous",
                "pyod_constructor_kwargs": model_kwargs,},
            f,indent=2,ensure_ascii=False,default=str,)

    return metrics


# ============================================================
# Deep SVDD: original 768 -> 128 projection
# ============================================================

