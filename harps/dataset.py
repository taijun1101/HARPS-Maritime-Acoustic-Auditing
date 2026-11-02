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


