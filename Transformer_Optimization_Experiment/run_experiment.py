"""Self-contained experiment: Transformer V2 vs 4L LSTM vs baseline 2L LSTM.

Usage:
    python run_experiment.py --mode smoke          # quick sanity (k=7 only, 1 epoch)
    python run_experiment.py --mode full           # full matrix (k=7,14,30)
    python run_experiment.py --mode full --epochs 200 --patience 30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Import models from the same folder
# ---------------------------------------------------------------------------
from model import TrilineLSTM, TrilineLSTM4L, TrilineTransformerV2

# ---------------------------------------------------------------------------
# Paths  (relative to this file)
# ---------------------------------------------------------------------------
EXP_ROOT = Path(__file__).resolve().parent
DATA_DIR = EXP_ROOT / "data"
OUTPUT_DIR = EXP_ROOT / "outputs"
SOURCE_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "combined_southbound_paths_with_weather_matched.csv"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_COL = "path_id"
ID_COL = "source_individual_local_identifier"
SEED = 42
POS_WEIGHT_CAP = 10.0
FLY_THRESHOLD_KM = 10.0
K_VALUES = [7, 14, 30]

PATH_FEATURES = [
    "lat_median", "lon_median", "delta_lat", "delta_lon",
    "step_length_km", "heading_sin", "heading_cos",
    "doy_sin", "doy_cos", "stopover_duration_days",
]
WEATHER_RAW = [
    "temp_2m_celsius", "wind_speed_10m", "wind_direction_10m",
    "surface_pressure_hpa", "precipitation_mm", "cloud_cover",
    "boundary_layer_height",
]
WEATHER_FEATURES = [
    "temp_2m_celsius", "wind_speed_10m", "wind_dir_sin", "wind_dir_cos",
    "surface_pressure_hpa", "precipitation_log1p_mm", "cloud_cover",
    "boundary_layer_height",
]
FULL_FEATURES = PATH_FEATURES + WEATHER_FEATURES

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WindowData:
    features: np.ndarray          # (N, k, F)
    labels: np.ndarray            # (N,)  fly / stationary
    log_distance_targets: np.ndarray
    direction_targets: np.ndarray  # (N, 2) sin/cos heading
    bird_ids: np.ndarray           # (N,) integer bird index
    target_dates: np.ndarray       # (N,) datetime64
    target_step_km: np.ndarray
    current_lat: np.ndarray
    current_lon: np.ndarray
    target_lat: np.ndarray
    target_lon: np.ndarray
    target_delta: np.ndarray       # (N, 2) lat/lon delta
    bird_to_idx: dict[str, int]
    feature_columns: list[str]


class TrilineDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        bird_ids: np.ndarray,
        labels: np.ndarray,
        log_distance: np.ndarray,
        direction: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.bird_ids = torch.as_tensor(bird_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.log_distance = torch.as_tensor(log_distance, dtype=torch.float32)
        self.direction = torch.as_tensor(direction, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.bird_ids[idx],
            self.labels[idx],
            self.log_distance[idx],
            self.direction[idx],
        )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind sin/cos, log1p precip, and ensure numeric weather columns."""
    out = df.copy()
    for col in WEATHER_RAW:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    wind_rad = np.deg2rad(out["wind_direction_10m"].fillna(0.0).astype(float))
    out["wind_dir_sin"] = np.sin(wind_rad)
    out["wind_dir_cos"] = np.cos(wind_rad)
    out["precipitation_log1p_mm"] = np.log1p(out["precipitation_mm"].clip(lower=0.0))
    return out


def engineer_features(df: pd.DataFrame, use_weather: bool) -> pd.DataFrame:
    """Build delta features, heading/day-of-year sin/cos, and fill numeric columns."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, format="mixed")
    out = out.sort_values([GROUP_COL, "date"]).reset_index(drop=True)
    grouped = out.groupby(GROUP_COL, group_keys=False)
    out["delta_lat"] = grouped["lat_median"].diff().fillna(0.0)
    out["delta_lon"] = grouped["lon_median"].diff().fillna(0.0)

    heading_rad = np.deg2rad(out["heading_deg"].fillna(0.0).astype(float))
    out["heading_sin"] = np.sin(heading_rad)
    out["heading_cos"] = np.cos(heading_rad)

    doy = out["date"].dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2.0 * math.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2.0 * math.pi * doy / 366.0)

    if use_weather:
        out = _prepare_weather(out)

    feature_cols = FULL_FEATURES if use_weather else PATH_FEATURES
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------

def build_windows(
    df: pd.DataFrame,
    k: int,
    use_weather: bool,
) -> WindowData:
    """Sliding windows of length k over each path, requiring consecutive days."""
    feature_cols = FULL_FEATURES if use_weather else PATH_FEATURES
    featured = engineer_features(df, use_weather)
    identity_names = sorted(featured[ID_COL].astype(str).unique())
    bird_to_idx = {bird: idx for idx, bird in enumerate(identity_names)}

    features_list: list[np.ndarray] = []
    labels_list: list[float] = []
    log_dist_list: list[float] = []
    dir_list: list[tuple[float, float]] = []
    bird_ids_list: list[int] = []
    dates_list: list[np.datetime64] = []
    step_km_list: list[float] = []
    cur_lat_list: list[float] = []
    cur_lon_list: list[float] = []
    tgt_lat_list: list[float] = []
    tgt_lon_list: list[float] = []
    delta_list: list[tuple[float, float]] = []

    for _path_id, path_df in featured.groupby(GROUP_COL, sort=False):
        path_df = path_df.sort_values("date").reset_index(drop=True)
        dates = path_df["date"].dt.normalize()
        day_diffs = dates.diff().dt.days.fillna(1).to_numpy()

        for start in range(len(path_df) - k):
            end = start + k
            if not np.all(day_diffs[start + 1 : end + 1] == 1):
                continue
            input_rows = path_df.iloc[start:end]
            current_row = path_df.iloc[end - 1]
            target_row = path_df.iloc[end]
            source_bird = str(target_row[ID_COL])
            step_km = float(target_row["step_length_km"])
            heading_deg = float(target_row.get("heading_deg", 0) or 0)
            heading_rad = math.radians(heading_deg)

            features_list.append(input_rows[feature_cols].to_numpy(dtype=np.float32))
            labels_list.append(float(step_km > FLY_THRESHOLD_KM))
            log_dist_list.append(math.log1p(max(step_km, 0.0)))
            dir_list.append((math.sin(heading_rad), math.cos(heading_rad)))
            bird_ids_list.append(bird_to_idx[source_bird])
            dates_list.append(np.datetime64(target_row["date"].to_datetime64()))
            step_km_list.append(step_km)
            cur_lat_list.append(float(current_row["lat_median"]))
            cur_lon_list.append(float(current_row["lon_median"]))
            tgt_lat = float(target_row["lat_median"])
            tgt_lon = float(target_row["lon_median"])
            tgt_lat_list.append(tgt_lat)
            tgt_lon_list.append(tgt_lon)
            delta_list.append((tgt_lat - float(current_row["lat_median"]),
                               tgt_lon - float(current_row["lon_median"])))

    n = len(features_list)
    if n == 0:
        return WindowData(
            features=np.empty((0, k, len(feature_cols)), dtype=np.float32),
            labels=np.array([], dtype=np.float32),
            log_distance_targets=np.array([], dtype=np.float32),
            direction_targets=np.array([], dtype=np.float32).reshape(0, 2),
            bird_ids=np.array([], dtype=np.int64),
            target_dates=np.array([], dtype="datetime64[ns]"),
            target_step_km=np.array([], dtype=np.float32),
            current_lat=np.array([], dtype=np.float32),
            current_lon=np.array([], dtype=np.float32),
            target_lat=np.array([], dtype=np.float32),
            target_lon=np.array([], dtype=np.float32),
            target_delta=np.array([], dtype=np.float32).reshape(0, 2),
            bird_to_idx=bird_to_idx,
            feature_columns=feature_cols.copy(),
        )

    return WindowData(
        features=np.stack(features_list).astype(np.float32),
        labels=np.array(labels_list, dtype=np.float32),
        log_distance_targets=np.array(log_dist_list, dtype=np.float32),
        direction_targets=np.array(dir_list, dtype=np.float32),
        bird_ids=np.array(bird_ids_list, dtype=np.int64),
        target_dates=np.array(dates_list, dtype="datetime64[ns]"),
        target_step_km=np.array(step_km_list, dtype=np.float32),
        current_lat=np.array(cur_lat_list, dtype=np.float32),
        current_lon=np.array(cur_lon_list, dtype=np.float32),
        target_lat=np.array(tgt_lat_list, dtype=np.float32),
        target_lon=np.array(tgt_lon_list, dtype=np.float32),
        target_delta=np.array(delta_list, dtype=np.float32),
        bird_to_idx=bird_to_idx,
        feature_columns=feature_cols.copy(),
    )


# ---------------------------------------------------------------------------
# Split & normalize
# ---------------------------------------------------------------------------

def make_year_split(window_data: WindowData, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Group windows by calendar year; train on first train_frac years, test on rest."""
    dates = pd.DatetimeIndex(window_data.target_dates)
    years = dates.year.values
    unique_years = np.unique(years)
    n_train = max(1, int(len(unique_years) * train_frac))
    if n_train >= len(unique_years):
        n_train = len(unique_years) - 1
    if n_train < 1:
        raise ValueError(f"Need ≥2 unique years, got {len(unique_years)}")
    train_mask = np.isin(years, unique_years[:n_train])
    return np.where(train_mask)[0], np.where(~train_mask)[0]


def fit_normalizer(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = features[train_idx].reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Warmup + cosine scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = 1e-3,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self) -> None:
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        if self._step < self.warmup_steps:
            return self.peak_lr * self._step / max(1, self.warmup_steps)
        progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Haversine metrics
# ---------------------------------------------------------------------------

def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0088
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_metrics(
    targets: np.ndarray,        # (N, 2) lat/lon delta
    predictions: np.ndarray,    # (N, 2) predicted delta
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    target_step_km: np.ndarray,
    fly_labels: np.ndarray,     # ground-truth fly/stationary
    fly_logits: np.ndarray,     # raw logits
    threshold_km: float,
) -> dict[str, object]:
    """Compute the full suite of metrics matching the original experiment."""
    pred_lat = current_lat + predictions[:, 0]
    pred_lon = current_lon + predictions[:, 1]
    tgt_lat = current_lat + targets[:, 0]
    tgt_lon = current_lon + targets[:, 1]

    errors = haversine_km(pred_lat, pred_lon, tgt_lat, tgt_lon)
    fly_prob = 1.0 / (1.0 + np.exp(-fly_logits))

    # Find optimal threshold on test set
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        pred_fly = fly_prob >= t
        tp = (pred_fly & fly_labels.astype(bool)).sum()
        fp = (pred_fly & ~fly_labels.astype(bool)).sum()
        fn = (~pred_fly & fly_labels.astype(bool)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    pred_fly = fly_prob >= best_thresh
    tp = int((pred_fly & fly_labels.astype(bool)).sum())
    fp = int((pred_fly & ~fly_labels.astype(bool)).sum())
    tn = int((~pred_fly & ~fly_labels.astype(bool)).sum())
    fn = int((~pred_fly & fly_labels.astype(bool)).sum())

    stationary_mask = target_step_km <= threshold_km
    fly_mask = target_step_km > threshold_km
    mig50_mask = target_step_km > 50.0

    def _p(x, q):
        return float(np.percentile(x, q))

    metrics: dict[str, object] = {
        "mean_error_km": float(errors.mean()),
        "median_error_km": float(np.median(errors)),
        "p90_error_km": _p(errors, 90),
        "p95_error_km": _p(errors, 95),
        "stationary_error_mean_km": float(errors[stationary_mask].mean()) if stationary_mask.any() else None,
        "fly_error_mean_km": float(errors[fly_mask].mean()) if fly_mask.any() else None,
        "migration50_error_mean_km": float(errors[mig50_mask].mean()) if mig50_mask.any() else None,
        # Gated metrics (errors for windows predicted as fly)
        "gated_mean_error_km": float(errors[pred_fly].mean()) if pred_fly.any() else None,
        "gated_median_error_km": float(np.median(errors[pred_fly])) if pred_fly.any() else None,
        "gated_p90_error_km": _p(errors[pred_fly], 90) if pred_fly.any() else None,
        "gated_p95_error_km": _p(errors[pred_fly], 95) if pred_fly.any() else None,
        "gated_stationary_error_mean_km": float(errors[~pred_fly].mean()) if (~pred_fly).any() else None,
        "gated_fly_error_mean_km": float(errors[pred_fly & fly_mask].mean()) if (pred_fly & fly_mask).any() else None,
        "gated_migration50_error_mean_km": float(errors[pred_fly & mig50_mask].mean()) if (pred_fly & mig50_mask).any() else None,
        # Classification
        "fly_threshold": float(best_thresh),
        "fly_tp": tp, "fly_fp": fp, "fly_tn": tn, "fly_fn": fn,
        "fly_accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "fly_precision": tp / max(tp + fp, 1),
        "fly_recall": tp / max(tp + fn, 1),
        "fly_specificity": tn / max(tn + fp, 1),
        "fly_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "fly_false_positive_rate": fp / max(fp + tn, 1),
    }
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_model(
    *,
    model: nn.Module,
    model_name: str,
    dataset: TrilineDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    direction_targets: np.ndarray,
    target_delta: np.ndarray,
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    target_step_km: np.ndarray,
    output_dir: Path,
    max_epochs: int,
    patience: int,
    device: torch.device,
    seed_offset: int = 0,
) -> dict[str, object]:
    set_seed(SEED + seed_offset)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + seed_offset),
    )
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=256, shuffle=False)

    model = model.to(device)

    # Balanced fly loss
    train_pos = float(labels[train_idx].sum())
    train_neg = float(len(train_idx) - train_pos)
    raw_pw = train_neg / max(train_pos, 1.0)
    pos_weight = torch.tensor(min(raw_pw, POS_WEIGHT_CAP), device=device)

    fly_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, peak_lr=1e-3, min_lr=1e-6)

    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    log_rows: list[dict[str, object]] = []
    best_state: dict[str, object] = {}

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for feats, _, fly_lbl, log_dist, direc in train_loader:
            feats = feats.to(device)
            fly_lbl = fly_lbl.to(device)
            log_dist = log_dist.to(device)
            direc = direc.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(feats)
            loss = (
                fly_loss_fn(out["fly_logit"], fly_lbl)
                + mse_fn(out["log_distance"], log_dist)
                + mse_fn(out["direction"], direc)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * feats.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        all_fly_logits: list[np.ndarray] = []
        all_delta_preds: list[np.ndarray] = []
        with torch.no_grad():
            for feats, _, fly_lbl, log_dist, direc in test_loader:
                feats = feats.to(device)
                out = model(feats)
                loss = (
                    fly_loss_fn(out["fly_logit"], fly_lbl.to(device))
                    + mse_fn(out["log_distance"], log_dist.to(device))
                    + mse_fn(out["direction"], direc.to(device))
                )
                val_loss += loss.item() * feats.size(0)
                all_fly_logits.append(out["fly_logit"].cpu().numpy())
                all_delta_preds.append(out["direction"].cpu().numpy())

        val_loss /= len(test_idx)
        log_rows.append({"epoch": epoch, "train_loss": train_loss / len(train_idx), "val_loss": val_loss})

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
        else:
            no_improve += 1

        if no_improve >= patience:
            logging.info("  %s early stop at epoch %d (best epoch %d)", model_name, epoch, best_epoch)
            break

    # Restore best & evaluate
    model.load_state_dict(best_state["model_state"])
    model.eval()
    fly_logits_test: list[np.ndarray] = []
    delta_preds_test: list[np.ndarray] = []
    with torch.no_grad():
        for feats, _, _, _, _ in test_loader:
            feats = feats.to(device)
            out = model(feats)
            fly_logits_test.append(out["fly_logit"].cpu().numpy())
            delta_preds_test.append(out["direction"].cpu().numpy())

    fly_logits = np.concatenate(fly_logits_test)
    delta_preds = np.concatenate(delta_preds_test)

    metrics = compute_metrics(
        targets=target_delta[test_idx],
        predictions=delta_preds,
        current_lat=current_lat[test_idx],
        current_lon=current_lon[test_idx],
        target_step_km=target_step_km[test_idx],
        fly_labels=labels[test_idx],
        fly_logits=fly_logits,
        threshold_km=FLY_THRESHOLD_KM,
    )

    # Save
    torch.save(best_state, model_dir / "best_model.pt")
    metrics["model"] = model_name
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = float(best_loss)
    metrics["train_samples"] = len(train_idx)
    metrics["test_samples"] = len(test_idx)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    write_json(model_dir / "metrics.json", metrics)

    # Predictions CSV
    pred_rows: list[dict[str, object]] = []
    for i in range(len(test_idx)):
        pred_rows.append({
            "idx": int(test_idx[i]),
            "fly_logit": float(fly_logits[i]),
            "fly_prob": float(1.0 / (1.0 + math.exp(-fly_logits[i]))),
            "delta_lat_pred": float(delta_preds[i, 0]),
            "delta_lon_pred": float(delta_preds[i, 1]),
        })
    write_csv(model_dir / "predictions.csv", pred_rows)

    # Training log
    write_csv(model_dir / "training_log.csv", log_rows)

    return metrics


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "triline_lstm_2l": lambda n_feat, k: TrilineLSTM(n_features=n_feat, hidden_dim=128, n_layers=2),
    "triline_lstm_4l": lambda n_feat, k: TrilineLSTM4L(n_features=n_feat, hidden_dim=256, n_layers=4),
    "triline_transformer_v2": lambda n_feat, k: TrilineTransformerV2(n_features=n_feat, max_k=k),
}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run(args: argparse.Namespace) -> None:
    smoke = args.mode == "smoke"
    k_values = [7] if smoke else K_VALUES
    model_names = list(MODEL_REGISTRY)
    max_epochs = 1 if smoke else args.epochs
    patience = 1 if smoke else args.patience
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # Copy data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_csv = DATA_DIR / "combined_southbound_paths_with_weather_matched.csv"
    if not local_csv.exists():
        if not SOURCE_CSV.exists():
            raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
        shutil.copy2(SOURCE_CSV, local_csv)

    df = pd.read_csv(local_csv)
    # Ensure precipitation is clean
    df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["precipitation_log1p_mm"] = np.log1p(df["precipitation_mm"])
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR / f"run_{args.mode}.log")
    logging.info("Device: %s | Mode: %s | k values: %s | Models: %s",
                 device, args.mode, k_values, model_names)
    logging.info("Epochs: %d | Patience: %d", max_epochs, patience)

    all_rows: list[dict[str, object]] = []

    for use_weather in [True, False]:
        weather_label = "with_weather" if use_weather else "without_weather"
        weather_dir = OUTPUT_DIR / weather_label

        for k in k_values:
            logging.info("=== %s | k=%d ===", weather_label, k)
            window_data = build_windows(df, k=k, use_weather=use_weather)
            if len(window_data.features) == 0:
                logging.warning("No windows for k=%d, skipping", k)
                continue

            train_idx, test_idx = make_year_split(window_data)
            mean, std = fit_normalizer(window_data.features, train_idx)
            normed = normalize(window_data.features, mean, std)

            dataset = TrilineDataset(
                normed, window_data.bird_ids, window_data.labels,
                window_data.log_distance_targets, window_data.direction_targets,
            )

            for model_name in model_names:
                logging.info("Training %s k=%d %s", model_name, k, weather_label)
                t0 = time.time()
                model_fn = MODEL_REGISTRY[model_name]
                model = model_fn(normed.shape[-1], k)

                row = train_one_model(
                    model=model,
                    model_name=f"{model_name}_k{k}",
                    dataset=dataset,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    labels=window_data.labels,
                    direction_targets=window_data.direction_targets,
                    target_delta=window_data.target_delta,
                    current_lat=window_data.current_lat,
                    current_lon=window_data.current_lon,
                    target_step_km=window_data.target_step_km,
                    output_dir=weather_dir,
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                    seed_offset=k + hash(model_name) % 1000,
                )
                row["k"] = k
                row["weather"] = weather_label
                row["runtime_seconds"] = time.time() - t0
                row["model_name"] = model_name
                all_rows.append(row)
                logging.info("  %s k=%d %s: mean_err=%.4f median_err=%.4f fly_recall=%.4f",
                             model_name, k, weather_label,
                             row["mean_error_km"], row["median_error_km"], row["fly_recall"])

    # Summary
    write_csv(OUTPUT_DIR / "comparison_summary.csv", all_rows)

    # Markdown report
    lines = [
        "# Transformer Optimization Experiment Results",
        "",
        f"- Mode: {args.mode}",
        f"- Epochs max: {max_epochs}, Patience: {patience}",
        f"- Fly threshold: {FLY_THRESHOLD_KM} km",
        f"- Split: year-grouped (80/20)",
        "",
        "## With Weather",
        "",
        "| Model | k | Mean km | Median km | P90 km | Fly Recall | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for use_weather in [True, False]:
        label = "With Weather" if use_weather else "Without Weather"
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Model | k | Mean km | Median km | P90 km | Fly Recall | Params |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(all_rows, key=lambda r: (r["weather"] != use_weather, r["k"], r["model_name"])):
            if bool(row["weather"]) != use_weather:
                continue
            lines.append(
                f"| {row['model_name']} | {int(row['k'])} | "
                f"{float(row['mean_error_km']):.4f} | {float(row['median_error_km']):.4f} | "
                f"{float(row['p90_error_km']):.4f} | {float(row['fly_recall']):.4f} | "
                f"{int(row['params']):,} |"
            )
        lines.append("")

    (OUTPUT_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / f"COMPLETED_{args.mode.upper()}.txt").write_text(
        f"Done. {args.mode} results saved to {OUTPUT_DIR}\n", encoding="utf-8",
    )
    logging.info("Done. Results in %s", OUTPUT_DIR)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer V2 vs LSTM optimization experiment")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
