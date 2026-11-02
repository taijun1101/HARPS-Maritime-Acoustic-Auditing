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

