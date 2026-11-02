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

