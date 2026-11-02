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
from .features import *
from .noise import *

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

