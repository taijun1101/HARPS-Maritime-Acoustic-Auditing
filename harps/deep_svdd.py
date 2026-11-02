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

