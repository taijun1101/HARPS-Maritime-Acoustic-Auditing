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

