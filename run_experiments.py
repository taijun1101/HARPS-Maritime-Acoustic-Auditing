# -*- coding: utf-8 -*-
"""
ShipsEar Final LOCO One-Class Anomaly Detection
================================================

Protocol
--------
For each held-out acoustic class c_a:
    D_ref  = {x_i | y_i != c_a}
    D_anom = {x_i | y_i == c_a}
    D_eval = D_ref U D_anom = complete dataset

There is NO 80/20 train/test split.

Baselines
---------
Each baseline is evaluated with BOTH MFCC and frozen AST features:
    - Isolation Forest
    - OCSVM
    - AutoEncoder
    - Variational AutoEncoder
    - SO-GAAL
    - MO-GAAL

KNN and LOF are intentionally excluded.

Proposed method
---------------
Frozen AST (768-D)
    -> single bias-free Linear projection (768 -> 128)
    -> soft-boundary Deep SVDD

Training modes:
    clean
    noisy
    paired_no_consistency
    paired_consistency

Evaluation:
    clean, 20 dB, 10 dB, 5 dB

Labels / score convention:
    normal = 0
    anomaly = 1
    larger score = more anomalous
"""

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



AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_SNRS = (20.0, 10.0, 5.0)
DEFAULT_KS = (10, 30, 50, 100)
# [MODIFIED - R2-2] Add AST kNN and shrinkage Mahalanobis; MFCC also available.
BASELINE_NAMES = ("iforest", "ocsvm", "ae", "vae", "so_gaal", "mo_gaal", "knn", "mahalanobis")
FEATURE_NAMES = ("mfcc", "ast")
# [MODIFIED - R2-2] Preserve all original modes; add explicit asymmetric controls.
DEEP_SVDD_MODES = ("clean", "noisy", "paired_no_consistency", "paired_consistency",
                   "paired_asymmetric_no_consistency", "paired_asymmetric",
                    "noisy_sb_center_noisy_radius_clean", "noisy_sb_center_clean_radius_noisy", "noisy_sb_center_noisy_radius_noisy")


@dataclass
class ExperimentConfig:
    data_root: str
    ast_model_path: str
    output_dir: str
    target_sr: int = 16000
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    snrs: Tuple[float, ...] = DEFAULT_SNRS
    train_snrs: Tuple[float, ...] = DEFAULT_SNRS

    # MFCC: 40 coefficient means + 40 standard deviations = 80-D.
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 128

    # Original-paper projection: AST 768 -> 128 only.
    rep_dim: int = 128

    # Deep SVDD.
    nu: float = 0.1
    epochs: int = 30
    warmup_epochs: int = 5
    radius_update_epochs: Tuple[int, ...] = (10, 15, 20, 25, 30)
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lambda_noisy: float = 1.0
    lambda_cons: float = 0.1

    batch_size: int = 16
    ast_batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    local_files_only: bool = True

    worker_id: int = -1
    num_experiment_workers: int = 1
    skip_completed: bool = True
    # [NEW - R2-2] Opt-in extra modes; original --run_all grid stays identical.
    run_asymmetric: bool = False
    run_geometry_comparison: bool = False
    # [NEW - R1-4/R2-2/R2-3/R2-4] All additions are opt-in.
    run_mfcc_svdd: bool = False
    run_pink: bool = False
    run_sensitivity: bool = False
    run_eigen: bool = False
    only_new_baselines: bool = False
    sensitivity_seeds: Tuple[int, ...] = (0,)
    sensitivity_classes: Tuple[str, ...] = ()
    sensitivity_snr: float = 10.0
    prevalence_repeats: int = 100
    bootstrap_repeats: int = 2000
    knn_k: int = 5



# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_int_seed(*items) -> int:
    text = "|".join(str(x) for x in items)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def training_noise_seed(seed: int, epoch: int, sample_id: str, snr_db: float) -> int:
    return stable_int_seed("TRAIN_NOISE", seed, epoch, sample_id, snr_db)


def evaluation_noise_seed(seed: int, sample_id: str, snr_db: float) -> int:
    return stable_int_seed("EVAL_NOISE", seed, sample_id, snr_db)

def atomic_save_npz(path: Path, **arrays) -> None:
    """
    Safely write an npz cache.

    Multiple experiment workers may request the same cache simultaneously.
    Each process writes to its own temporary file and then atomically replaces
    the final cache file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")

    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)
# ============================================================
# Dataset
# ============================================================

def build_manifest(data_root: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(data_root.rglob("*.wav")):
        rel = path.relative_to(data_root)

        if len(rel.parts) < 3:
            raise ValueError(
                f"Cannot infer class from {path}. Expected structure: "
                "data_root/group/class_name/file.wav")

        class_name = rel.parts[1]

        rows.append({
            "sample_id": str(rel).replace("\\", "/"),
            "file_path": str(path),
            "class_name": class_name,})

    df = pd.DataFrame(rows)
    df = df.sort_values(["class_name", "sample_id"]).reset_index(drop=True)
    df["global_index"] = np.arange(len(df), dtype=int)

    return df


def build_loco_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    total = len(manifest)
    rows = []

    for anomaly_class in sorted(manifest["class_name"].unique()):
        n_anomaly = int((manifest["class_name"] == anomaly_class).sum())
        n_ref = total - n_anomaly
        rows.append({
            "held_out_class": anomaly_class,
            "reference_samples": n_ref,
            "evaluation_normal": n_ref,
            "evaluation_anomaly": n_anomaly,
            "evaluation_total": total,})

    return pd.DataFrame(rows)


# ============================================================
# Audio / Gaussian noise
# ============================================================

def load_audio(file_path: str, target_sr: int) -> np.ndarray:
    waveform, _ = librosa.load(file_path, sr=target_sr, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def add_gaussian_noise(waveform: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    x = waveform.astype(np.float32, copy=False)
    signal_power = float(np.mean(x ** 2))

    if signal_power <= 1e-12:
        return x.copy()

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    current_noise_power = float(np.mean(noise ** 2))
    scale = math.sqrt(noise_power / max(current_noise_power, 1e-12))
    return (x + noise * scale).astype(np.float32)


# [NEW - R2-4] Pink (1/f) noise, independently seeded and power-normalized.
def add_pink_noise(waveform: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32)
    if len(x) < 2 or float(np.mean(x.astype(np.float64)**2)) <= 1e-12:
        return x.copy()
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(len(x)))
    freq = np.fft.rfftfreq(len(x))
    weights = np.zeros_like(freq)
    weights[1:] = 1.0 / np.sqrt(freq[1:])
    spectrum *= weights
    noise = np.fft.irfft(spectrum, n=len(x)).astype(np.float32)
    noise -= noise.mean()
    power = float(np.mean(noise.astype(np.float64)**2))
    signal_power = float(np.mean(x.astype(np.float64)**2))
    noise *= np.sqrt((signal_power / (10.0**(snr_db / 10.0))) / max(power, 1e-20))
    return (x + noise).astype(np.float32)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# MFCC
# ============================================================

def extract_mfcc_feature(waveform: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=cfg.target_sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


# ============================================================
# Frozen AST
# ============================================================
class FrozenASTEncoder:
    """
    Frozen Hugging Face AST.

    Representation:
        mean of the first two special-token representations:
        h = (hidden[:, 0, :] + hidden[:, 1, :]) / 2
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        local_files_only: bool = True,
    ):
        self.device = device
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path,local_files_only=local_files_only,)
        self.model = ASTModel.from_pretrained(model_path,local_files_only=local_files_only,).to(device)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: Sequence[np.ndarray],
        sampling_rate: int,
    ) -> np.ndarray:

        inputs = self.feature_extractor(
            list(waveforms),
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,)

        input_values = inputs["input_values"].to(self.device,non_blocking=True,)

        outputs = self.model(input_values=input_values)

        hidden = outputs.last_hidden_state

        representation = (hidden[:, 0, :] + hidden[:, 1, :]) / 2.0

        return (
            representation
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32))

    def metadata(self) -> Dict:
        return {
            "model_config": self.model.config.to_dict(),
            "feature_extractor": self.feature_extractor.to_dict(),
            "hidden_size": self.hidden_size,
            "frozen": True,
            "pooling": "mean_of_first_two_special_tokens",}

# ============================================================
# Feature manager
# ============================================================
class FeatureManager:
    """Global feature caches shared across LOCO folds and training modes."""

    def __init__(self, cfg: ExperimentConfig, manifest: pd.DataFrame,
                 ast_encoder: FrozenASTEncoder):
        self.cfg = cfg
        self.manifest = manifest.reset_index(drop=True)
        self.ast_encoder = ast_encoder
        self.cache_root = Path(cfg.output_dir) / "feature_cache"
        self.eval_cache_root = self.cache_root / "evaluation"
        self.eval_cache_root.mkdir(parents=True, exist_ok=True)
        self._waveform_cache: Dict[str, np.ndarray] = {}

    def get_waveform(self, file_path: str) -> np.ndarray:
        if file_path not in self._waveform_cache:
            self._waveform_cache[file_path] = load_audio(file_path, self.cfg.target_sr)
        return self._waveform_cache[file_path]

    def _eval_cache_path(self, feature_type: str, condition: str,
                         seed: int, snr_db: Optional[float]) -> Path:
        if condition == "clean":
            return self.eval_cache_root / f"{feature_type}_clean.npz"
        prefix = "pink_" if condition == "pink" else ""
        return (self.eval_cache_root /
                f"{feature_type}_eval_{prefix}snr{snr_db:g}_seed{seed}.npz")

    def _training_cache_path(self, feature_type: str, seed: int,
                             epoch: int, snr_db: float) -> Path:
        return (self.cache_root / f"training_noisy_{feature_type}" /
                f"seed_{seed}" / f"snr_{snr_db:g}" /
                f"epoch_{epoch:03d}.npz")

    def _extract_features(self, feature_type: str, condition: str,
                          seed: int, snr_db: Optional[float] = None,
                          epoch: Optional[int] = None) -> np.ndarray:
        """Extract the complete dataset in manifest order, without LOCO filtering."""
        chunks, batch = [], []
        for row in self.manifest.itertuples(index=False):
            waveform = self.get_waveform(row.file_path)
            if condition != "clean":
                # Training and evaluation intentionally use different seed schemes.
                if epoch is not None:
                    nseed = training_noise_seed(seed, epoch, row.sample_id, snr_db)
                elif condition == "pink":
                    nseed = stable_int_seed("EVAL_PINK", seed, row.sample_id, float(snr_db))
                else:
                    nseed = evaluation_noise_seed(seed, row.sample_id, float(snr_db))
                noise_fn = add_pink_noise if condition == "pink" else add_gaussian_noise
                waveform = noise_fn(waveform, float(snr_db), nseed)

            if feature_type == "mfcc":
                chunks.append(extract_mfcc_feature(waveform, self.cfg))
            else:
                batch.append(waveform)
                if len(batch) == self.cfg.ast_batch_size:
                    chunks.append(self.ast_encoder.encode_waveforms(batch, self.cfg.target_sr))
                    batch = []

        if feature_type == "ast":
            if batch:
                chunks.append(self.ast_encoder.encode_waveforms(batch, self.cfg.target_sr))
            return np.concatenate(chunks, axis=0).astype(np.float32)
        return np.asarray(chunks, dtype=np.float32)

    @staticmethod
    def _load_or_build(path: Path, build, description: str) -> np.ndarray:
        if path.exists():
            return np.load(path, allow_pickle=False)["features"]
        print(f"[CACHE BUILD] {description}")
        features = build()
        atomic_save_npz(path, features=features)
        return features

    def get_eval_features(self, feature_type: str, condition: str,
                          seed: int, snr_db: Optional[float] = None) -> np.ndarray:
        path = self._eval_cache_path(feature_type, condition, seed, snr_db)
        return self._load_or_build(
            path,
            lambda: self._extract_features(feature_type, condition, seed, snr_db),
            f"Evaluation {feature_type} {condition} seed={seed} snr={snr_db}",)

    def _get_training_noisy_all(self, feature_type: str, seed: int,
                                epoch: int, snr_db: float) -> np.ndarray:
        path = self._training_cache_path(feature_type, seed, epoch, snr_db)
        return self._load_or_build(
            path,
            lambda: self._extract_features(feature_type, "noisy", seed, snr_db, epoch),
            f"Training noisy {feature_type.upper()} | seed={seed} epoch={epoch} SNR={snr_db:g}",)

    def get_training_noisy_ast_all(self, seed: int, epoch: int, snr_db: float) -> np.ndarray:
        return self._get_training_noisy_all("ast", seed, epoch, snr_db)

    def get_training_noisy_mfcc_all(self, seed: int, epoch: int, snr_db: float) -> np.ndarray:
        return self._get_training_noisy_all("mfcc", seed, epoch, snr_db)

    def get_training_noisy_ast_features(self, reference_indices: np.ndarray, seed: int, epoch: int, snr_db: float) -> np.ndarray:
        return self.get_training_noisy_ast_all(seed, epoch, snr_db)[reference_indices]


# ============================================================
# PyOD baselines
# ============================================================

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

class ProjectionHead(nn.Module):
    """
    Original IEEE-paper projection:
        AST embedding (768-D) -> Linear(768, 128, bias=False)

    No hidden layer.
    No activation.
    No LayerNorm.
    No dropout.
    """

    def __init__(self, input_dim: int = 768, rep_dim: int = 128):
        super().__init__()
        self.projection = nn.Linear(input_dim, rep_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PairedFeatureDataset(Dataset):
    def __init__(
        self,
        clean_features: np.ndarray,
        noisy_features: Optional[np.ndarray] = None,):
        self.clean = torch.as_tensor(clean_features, dtype=torch.float32)
        self.noisy = (
            torch.as_tensor(noisy_features, dtype=torch.float32)
            if noisy_features is not None
            else None)

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int):
        if self.noisy is None:
            return self.clean[index]
        return self.clean[index], self.noisy[index]


@torch.no_grad()
def initialize_center(
    model: nn.Module,
    clean_features: np.ndarray,
    device: torch.device,
    batch_size: int,
    eps: float = 0.1,
) -> torch.Tensor:
    model.eval()
    loader = DataLoader(torch.as_tensor(clean_features, dtype=torch.float32),batch_size=batch_size,shuffle=False,)
    total = None
    count = 0

    for x in loader:
        x = x.to(device)
        z = model(x)

        if total is None:
            total = torch.zeros(z.shape[1], device=device)

        total += z.sum(dim=0)
        count += len(z)

    center = total / count

    small_positive = (center.abs() < eps) & (center >= 0)
    small_negative = (center.abs() < eps) & (center < 0)
    center[small_positive] = eps
    center[small_negative] = -eps
    return center.detach()


def one_class_loss(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return torch.sum((z - center) ** 2, dim=1).mean()


def soft_boundary_loss(
    z: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    nu: float,
) -> torch.Tensor:
    dist_sq = torch.sum((z - center) ** 2, dim=1)
    return radius ** 2 + (1.0 / nu) * torch.relu(dist_sq - radius ** 2).mean()


def svdd_component(
    z: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    nu: float,
    use_soft_boundary: bool,
) -> torch.Tensor:
    if use_soft_boundary:
        return soft_boundary_loss(z, center, radius, nu)
    return one_class_loss(z, center)


@torch.no_grad()
def radius_from_clean_features(
    model: nn.Module,
    clean_features: np.ndarray,
    center: torch.Tensor,
    nu: float,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = DataLoader(torch.as_tensor(clean_features, dtype=torch.float32),batch_size=batch_size,shuffle=False,)
    distances = []

    for x in loader:
        x = x.to(device)
        z = model(x)
        d = torch.sqrt(torch.sum((z - center) ** 2, dim=1))
        distances.append(d.cpu().numpy())

    return float(np.quantile(np.concatenate(distances), 1.0 - nu))



def train_deep_svdd(
    cfg: ExperimentConfig,
    manifest: pd.DataFrame,
    feature_manager: FeatureManager,
    anomaly_class: str,
    seed: int,
    training_mode: str,
    train_snr_db: Optional[float],
    device: torch.device,
    output_dir: Path,
    feature_type: str = "ast",
) -> Tuple[ProjectionHead, torch.Tensor, float, pd.DataFrame]:
    if training_mode not in DEEP_SVDD_MODES:
        raise ValueError(f"Unknown training mode: {training_mode}")
    if training_mode != "clean" and train_snr_db is None:
        raise ValueError("train_snr_db is required for noisy/paired training.")

    set_global_seed(seed)

    reference_mask = (manifest["class_name"] != anomaly_class).to_numpy()

    reference_indices = np.flatnonzero(reference_mask)
    # [MODIFIED - R1-4] MFCC uses reference-only standardization.
    X_clean_all = feature_manager.get_eval_features(feature_type, "clean", seed)
    feature_scaler = None
    if feature_type == "mfcc":
        feature_scaler = StandardScaler().fit(X_clean_all[reference_mask])
        X_clean_all = feature_scaler.transform(X_clean_all).astype(np.float32)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_scaler, output_dir / "feature_scaler.joblib")
    X_clean_ref = X_clean_all[reference_mask]
    synchronize_if_cuda(device)
    train_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = ProjectionHead(input_dim=X_clean_ref.shape[1],rep_dim=cfg.rep_dim,).to(device)

    # [MODIFIED - R2-2] Name variants by loss and reference geometry, not by paper provenance.
    # All four noisy-only variants use soft-boundary (SB) after warm-up.
    # Legacy noisy: clean center + clean radius.
    # Noisy-reference geometry: first-epoch noisy center + current-epoch noisy radius.
    noisy_center_modes = {"noisy_sb_center_noisy_radius_clean", "noisy_sb_center_noisy_radius_noisy"}
    noisy_radius_modes = {"noisy_sb_center_clean_radius_noisy", "noisy_sb_center_noisy_radius_noisy"}
    if training_mode in noisy_center_modes:
        if feature_type == "ast":
            center_ref = feature_manager.get_training_noisy_ast_features(
                reference_indices, seed, 1, float(train_snr_db))
        else:
            center_ref = feature_manager.get_training_noisy_mfcc_all(
                seed, 1, float(train_snr_db))[reference_indices]
            center_ref = feature_scaler.transform(center_ref).astype(np.float32)
    else:
        center_ref = X_clean_ref
    center = initialize_center(model,center_ref,device=device,batch_size=cfg.batch_size,)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay,)

    radius = 0.0
    history = []
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, cfg.epochs + 1):
        if training_mode == "clean":
            X_noisy_ref = None
        else:
            if feature_type == "ast":
                X_noisy_ref = feature_manager.get_training_noisy_ast_features(
                    reference_indices=reference_indices, seed=seed, epoch=epoch,
                    snr_db=float(train_snr_db))
            else:
                X_noisy_ref = feature_manager.get_training_noisy_mfcc_all(
                    seed, epoch, float(train_snr_db))[reference_indices]
                X_noisy_ref = feature_scaler.transform(X_noisy_ref).astype(np.float32)

        dataset = PairedFeatureDataset(X_clean_ref, X_noisy_ref)
        generator = torch.Generator()
        generator.manual_seed(stable_int_seed("DATALOADER", seed, anomaly_class, training_mode, epoch))

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            generator=generator,)

        model.train()
        running_total = 0.0
        running_clean = 0.0
        running_noisy = 0.0
        running_cons = 0.0
        n_samples = 0
        use_soft_boundary = epoch > cfg.warmup_epochs

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            if training_mode == "clean":
                x_clean = batch.to(device, non_blocking=True)
                x_noisy = None
            else:
                x_clean, x_noisy = batch
                x_clean = x_clean.to(device, non_blocking=True)
                x_noisy = x_noisy.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,):
                z_clean = model(x_clean)
                clean_loss = svdd_component(
                    z_clean,
                    center,
                    radius,
                    cfg.nu,
                    use_soft_boundary,)

                noisy_loss = torch.zeros((), device=device)
                consistency_loss = torch.zeros((), device=device)

                if training_mode == "clean":
                    loss = clean_loss

                elif training_mode in ("noisy", "noisy_sb_center_noisy_radius_clean", "noisy_sb_center_clean_radius_noisy", "noisy_sb_center_noisy_radius_noisy"):
                    z_noisy = model(x_noisy)
                    noisy_loss = svdd_component(
                        z_noisy,
                        center,
                        radius,
                        cfg.nu,
                        use_soft_boundary,)
                    loss = noisy_loss

                elif training_mode == "paired_no_consistency":
                    z_noisy = model(x_noisy)
                    noisy_loss = svdd_component(
                        z_noisy,
                        center,
                        radius,
                        cfg.nu,
                        use_soft_boundary,)
                    loss = clean_loss + cfg.lambda_noisy * noisy_loss

                # [NEW - R2-2] Asymmetric: clean hinge (after warm-up),
                # noisy unhinged squared distance. Original modes are unchanged.
                elif training_mode in ("paired_asymmetric_no_consistency", "paired_asymmetric"):
                    z_noisy = model(x_noisy)
                    noisy_loss = one_class_loss(z_noisy, center)
                    if training_mode == "paired_asymmetric":
                        consistency_loss = torch.mean(torch.sum((z_clean - z_noisy) ** 2, dim=1))
                    loss = (clean_loss + cfg.lambda_noisy * noisy_loss
                            + cfg.lambda_cons * consistency_loss)

                else:
                    # [UNCHANGED - R2-2] Original symmetric paired_consistency.
                    z_noisy = model(x_noisy)
                    noisy_loss = svdd_component(
                        z_noisy,
                        center,
                        radius,
                        cfg.nu,
                        use_soft_boundary,)
                    consistency_loss = torch.mean(torch.sum((z_clean - z_noisy) ** 2, dim=1))
                    loss = (
                        clean_loss
                        + cfg.lambda_noisy * noisy_loss
                        + cfg.lambda_cons * consistency_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_n = len(x_clean)
            running_total += float(loss.detach().item()) * batch_n
            running_clean += float(clean_loss.detach().item()) * batch_n
            running_noisy += float(noisy_loss.detach().item()) * batch_n
            running_cons += float(consistency_loss.detach().item()) * batch_n
            n_samples += batch_n

        # Factorial geometry ablation: only the selected noisy-only modes use
        # the current epoch's corrupted normal reference for radius updates.
        # All legacy and paired modes preserve their clean-reference radius.
        if epoch in cfg.radius_update_epochs:
            radius_ref = X_noisy_ref if training_mode in noisy_radius_modes else X_clean_ref
            radius = radius_from_clean_features(
                model=model,
                clean_features=radius_ref,
                center=center,
                nu=cfg.nu,
                device=device,
                batch_size=cfg.batch_size,)

        n = max(n_samples, 1)
        row = {
            "epoch": epoch,
            "loss_total": running_total / n,
            "loss_clean": running_clean / n,
            "loss_noisy": running_noisy / n,
            "loss_consistency": running_cons / n,
            "radius": radius,}
        history.append(row)

        print(
            f"[SVDD {training_mode} {epoch:02d}/{cfg.epochs:02d}] "
            f"total={row['loss_total']:.6f} clean={row['loss_clean']:.6f} "
            f"noisy={row['loss_noisy']:.6f} cons={row['loss_consistency']:.6f} "
            f"R={radius:.6f}")

    synchronize_if_cuda(device)
    train_seconds = time.perf_counter() - train_start
    peak_memory_mb = (torch.cuda.max_memory_allocated(device) / 1048576
                      if device.type == "cuda" else np.nan)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_cost.json", "w", encoding="utf-8") as f:
        json.dump({"training_seconds_including_feature_cache_access": train_seconds,
                   "peak_gpu_allocated_mb": peak_memory_mb,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "feature_type": feature_type}, f, indent=2)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "center": center.detach().cpu(),
            "radius": radius,
            "training_mode": training_mode,
            "train_snr_db": train_snr_db,
            "seed": seed,
            "held_out_class": anomaly_class,
            "feature_type": feature_type,
            "center_reference": "noisy_epoch_1" if training_mode in noisy_center_modes else "clean",
            "radius_reference": "noisy_current_epoch" if training_mode in noisy_radius_modes else "clean",
            "noisy_loss": ("none" if training_mode == "clean" else
                           "unhinged_center_distance" if training_mode.startswith("paired_asymmetric") else
                           "soft_boundary_after_warmup"),
            "projection": {
                "input_dim": X_clean_ref.shape[1],
                "output_dim": cfg.rep_dim,
                "type": "single_linear",
                "bias": False,
                "hidden_layer": None,
                "activation": None,
                "normalization": None,
                "dropout": None,
            },},output_dir / "model.pt",)

    return model, center, radius, history_df


@torch.no_grad()
def deep_svdd_scores(
    model: nn.Module,
    features: np.ndarray,
    center: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.as_tensor(features, dtype=torch.float32),batch_size=batch_size,shuffle=False,)
    scores = []

    for x in loader:
        x = x.to(device)
        z = model(x)
        scores.append(torch.sum((z - center) ** 2, dim=1).cpu().numpy())

    return np.concatenate(scores).astype(np.float64)


def evaluate_deep_svdd(
    cfg: ExperimentConfig,
    manifest: pd.DataFrame,
    feature_manager: FeatureManager,
    model: nn.Module,
    center: torch.Tensor,
    anomaly_class: str,
    seed: int,
    training_mode: str,
    train_snr_db: Optional[float],
    device: torch.device,
    model_dir: Path,
    feature_type: str = "ast",
    feature_scaler=None,
) -> List[Dict]:
    y_eval = (manifest["class_name"] == anomaly_class).astype(int).to_numpy()
    rows = []
    eval_conditions = [("clean", None)] + [("noisy", snr) for snr in cfg.snrs]
    if cfg.run_pink:
        eval_conditions += [("pink", snr) for snr in cfg.snrs]

    for eval_condition, eval_snr_db in eval_conditions:
        X_eval = feature_manager.get_eval_features(
            feature_type, eval_condition, seed, eval_snr_db)
        if feature_scaler is not None:
            X_eval = feature_scaler.transform(X_eval).astype(np.float32)
        synchronize_if_cuda(device)
        inference_start = time.perf_counter()
        scores = deep_svdd_scores(
            model,
            X_eval,
            center,
            device=device,
            batch_size=cfg.batch_size,)
        synchronize_if_cuda(device)
        inference_seconds = time.perf_counter() - inference_start
        metrics = evaluate_scores(y_eval, scores)
        prevalence = prevalence_metrics(y_eval, scores)
        metrics.update({f"prevalence_1pct_{k}": v for k, v in prevalence.items()})
        metrics["inference_seconds"] = inference_seconds
        metrics["inference_seconds_per_segment"] = inference_seconds / len(X_eval)
        metrics["cost_scope"] = "projection and scoring only; excludes feature extraction"
        metrics.update({
            "method": "deep_svdd",
            "feature": feature_type,
            "training_mode": training_mode,
            "train_snr_db": train_snr_db,
            "seed": seed,
            "held_out_class": anomaly_class,
            "eval_condition": eval_condition,
            "eval_snr_db": eval_snr_db,
            "n_reference": int((y_eval == 0).sum()),
            "n_eval_normal": int((y_eval == 0).sum()),
            "n_eval_anomaly": int((y_eval == 1).sum()),
            "n_eval_total": len(y_eval),})

        condition_name = evaluation_condition_name(eval_condition, eval_snr_db)
        eval_dir = model_dir / "evaluation" / condition_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        make_prediction_dataframe(manifest, anomaly_class, scores).to_csv(eval_dir / "predictions.csv",index=False,)
        pd.DataFrame([metrics]).to_csv(eval_dir / "metrics.csv",index=False,)
        rows.append(metrics)

    return rows


# ============================================================
# Experiment grids
# ============================================================

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
def benchmark_feature_extraction(cfg: ExperimentConfig, manifest: pd.DataFrame,
                                 fm: FeatureManager, device: torch.device,
                                 output_root: Path, n_samples: int = 32) -> None:
    n = min(n_samples, len(manifest))
    if n == 0:
        return
    waveforms = [fm.get_waveform(p) for p in manifest.file_path.iloc[:n]]
    records = []
    for feature in FEATURE_NAMES:
        synchronize_if_cuda(device)
        start = time.perf_counter()
        if feature == "mfcc":
            for wav in waveforms:
                extract_mfcc_feature(wav, cfg)
        else:
            for j in range(0, n, cfg.ast_batch_size):
                fm.ast_encoder.encode_waveforms(waveforms[j:j+cfg.ast_batch_size], cfg.target_sr)
        synchronize_if_cuda(device)
        seconds = time.perf_counter() - start
        records.append({"feature": feature, "n_segments": n,
                        "feature_seconds": seconds,
                        "feature_seconds_per_segment": seconds / n,
                        "scope": "warm feature extraction; waveform loading excluded"})
    pd.DataFrame(records).to_csv(output_root / "feature_extraction_cost.csv", index=False)


# ============================================================
# CLI
# ============================================================

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


if __name__ == "__main__":
    main()
