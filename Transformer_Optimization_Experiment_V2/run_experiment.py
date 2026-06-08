"""Self-contained experiment V2: V3 Transformer vs strong LSTM baselines.

Models tested:
- triline_lstm_2l      (2L, 128 dim)  — baseline
- triline_lstm_4l      (4L, 256 dim)  — previous winner
- triline_lstm_6l      (6L, 512 dim)  — push LSTM to limit
- triline_transformer_v2       (CLS,         4L, 256 dim) — V1 baseline
- triline_transformer_v3       (CLS+last,    4L, 256 dim) — hybrid pool
- triline_transformer_v3_alibi (CLS+last+ALiBi,4L, 256 dim) —recency bias

Usage:
    python run_experiment.py --mode smoke
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
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model import (
    TrilineLSTM,
    TrilineLSTM4L,
    TrilineLSTM6L,
    TrilineTransformerV2,
    TrilineTransformerV3,
    TrilineTransformerV3_ALiBi,
)

# ── Paths ──────────────────────────────────────────────────────────────────
EXP_ROOT = Path(__file__).resolve().parent
DATA_DIR = EXP_ROOT / "data"
OUTPUT_DIR = EXP_ROOT / "outputs"
SOURCE_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "combined_southbound_paths_with_weather_matched.csv"
)

# ── Constants ──────────────────────────────────────────────────────────────
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

# ── Model registry ─────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Any] = {
    "triline_lstm_2l":             lambda n_feat, k: TrilineLSTM(n_features=n_feat, hidden_dim=128, n_layers=2),
    "triline_lstm_4l":             lambda n_feat, k: TrilineLSTM4L(n_features=n_feat, hidden_dim=256, n_layers=4),
    "triline_lstm_6l":             lambda n_feat, k: TrilineLSTM6L(n_features=n_feat, hidden_dim=512, n_layers=6),
    "triline_transformer_v2":      lambda n_feat, k: TrilineTransformerV2(n_features=n_feat, max_k=k),
    "triline_transformer_v3":      lambda n_feat, k: TrilineTransformerV3(n_features=n_feat, max_k=k),
    "triline_transformer_v3_alibi":lambda n_feat, k: TrilineTransformerV3_ALiBi(n_features=n_feat, max_k=k),
}


# ═══════════════════════════════════════════════════════════════════════════
# Everything below is the same as V1 runner
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class WindowData:
    features: np.ndarray
    labels: np.ndarray
    log_distance_targets: np.ndarray
    direction_targets: np.ndarray
    bird_ids: np.ndarray
    target_dates: np.ndarray
    target_step_km: np.ndarray
    current_lat: np.ndarray
    current_lon: np.ndarray
    target_lat: np.ndarray
    target_lon: np.ndarray
    target_delta: np.ndarray
    bird_to_idx: dict[str, int]
    feature_columns: list[str]


class TrilineDataset(Dataset):
    def __init__(self, features, bird_ids, labels, log_distance, direction):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.bird_ids = torch.as_tensor(bird_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.log_distance = torch.as_tensor(log_distance, dtype=torch.float32)
        self.direction = torch.as_tensor(direction, dtype=torch.float32)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.bird_ids[idx], self.labels[idx],
                self.log_distance[idx], self.direction[idx])


def _prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in WEATHER_RAW:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    wind_rad = np.deg2rad(out["wind_direction_10m"].fillna(0.0).astype(float))
    out["wind_dir_sin"] = np.sin(wind_rad)
    out["wind_dir_cos"] = np.cos(wind_rad)
    out["precipitation_log1p_mm"] = np.log1p(out["precipitation_mm"].clip(lower=0.0))
    return out


def engineer_features(df: pd.DataFrame, use_weather: bool) -> pd.DataFrame:
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


def build_windows(df: pd.DataFrame, k: int, use_weather: bool) -> WindowData:
    feature_cols = FULL_FEATURES if use_weather else PATH_FEATURES
    featured = engineer_features(df, use_weather)
    identity_names = sorted(featured[ID_COL].astype(str).unique())
    bird_to_idx = {bird: i for i, bird in enumerate(identity_names)}

    features_l, labels_l, log_dist_l, dir_l, bird_l = [], [], [], [], []
    dates_l, step_l, cur_lat_l, cur_lon_l, tgt_lat_l, tgt_lon_l, delta_l = [], [], [], [], [], [], []

    for _pid, path_df in featured.groupby(GROUP_COL, sort=False):
        path_df = path_df.sort_values("date").reset_index(drop=True)
        dates = path_df["date"].dt.normalize()
        day_diffs = dates.diff().dt.days.fillna(1).to_numpy()
        for start in range(len(path_df) - k):
            end = start + k
            if not np.all(day_diffs[start + 1 : end + 1] == 1):
                continue
            in_rows = path_df.iloc[start:end]
            cur_row = path_df.iloc[end - 1]
            tgt_row = path_df.iloc[end]
            bird = str(tgt_row[ID_COL])
            step = float(tgt_row["step_length_km"])
            hdg = math.radians(float(tgt_row.get("heading_deg", 0) or 0))
            features_l.append(in_rows[feature_cols].to_numpy(dtype=np.float32))
            labels_l.append(float(step > FLY_THRESHOLD_KM))
            log_dist_l.append(math.log1p(max(step, 0.0)))
            dir_l.append((math.sin(hdg), math.cos(hdg)))
            bird_l.append(bird_to_idx[bird])
            dates_l.append(np.datetime64(tgt_row["date"].to_datetime64()))
            step_l.append(step)
            cur_lat_l.append(float(cur_row["lat_median"]))
            cur_lon_l.append(float(cur_row["lon_median"]))
            tlat, tlon = float(tgt_row["lat_median"]), float(tgt_row["lon_median"])
            tgt_lat_l.append(tlat); tgt_lon_l.append(tlon)
            delta_l.append((tlat - float(cur_row["lat_median"]), tlon - float(cur_row["lon_median"])))

    n = len(features_l)
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
            bird_to_idx=bird_to_idx, feature_columns=feature_cols.copy(),
        )
    return WindowData(
        features=np.stack(features_l).astype(np.float32),
        labels=np.array(labels_l, dtype=np.float32),
        log_distance_targets=np.array(log_dist_l, dtype=np.float32),
        direction_targets=np.array(dir_l, dtype=np.float32),
        bird_ids=np.array(bird_l, dtype=np.int64),
        target_dates=np.array(dates_l, dtype="datetime64[ns]"),
        target_step_km=np.array(step_l, dtype=np.float32),
        current_lat=np.array(cur_lat_l, dtype=np.float32),
        current_lon=np.array(cur_lon_l, dtype=np.float32),
        target_lat=np.array(tgt_lat_l, dtype=np.float32),
        target_lon=np.array(tgt_lon_l, dtype=np.float32),
        target_delta=np.array(delta_l, dtype=np.float32),
        bird_to_idx=bird_to_idx, feature_columns=feature_cols.copy(),
    )


def make_year_split(wd: WindowData, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    years = pd.DatetimeIndex(wd.target_dates).year.values
    uniq = np.unique(years)
    n_tr = max(1, int(len(uniq) * frac))
    if n_tr >= len(uniq): n_tr = len(uniq) - 1
    if n_tr < 1: raise ValueError(f"Need ≥2 years, got {len(uniq)}")
    mask = np.isin(years, uniq[:n_tr])
    return np.where(mask)[0], np.where(~mask)[0]


def fit_normalizer(features, train_idx):
    flat = features[train_idx].reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize(features, mean, std):
    return ((features - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr=1e-3, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self):
        if self._step < self.warmup_steps:
            return self.peak_lr * self._step / max(1, self.warmup_steps)
        p = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * p))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_metrics(targets, predictions, cur_lat, cur_lon, step_km, fly_labels, fly_logits, threshold_km):
    pred_lat = cur_lat + predictions[:, 0]
    pred_lon = cur_lon + predictions[:, 1]
    tgt_lat = cur_lat + targets[:, 0]
    tgt_lon = cur_lon + targets[:, 1]
    errors = haversine_km(pred_lat, pred_lon, tgt_lat, tgt_lon)
    fly_prob = 1.0 / (1.0 + np.exp(-fly_logits))

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        pf = fly_prob >= t
        tp = (pf & fly_labels.astype(bool)).sum()
        fp = (pf & ~fly_labels.astype(bool)).sum()
        fn = (~pf & fly_labels.astype(bool)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    pf = fly_prob >= best_t
    tp = int((pf & fly_labels.astype(bool)).sum())
    fp = int((pf & ~fly_labels.astype(bool)).sum())
    tn = int((~pf & ~fly_labels.astype(bool)).sum())
    fn = int((~pf & fly_labels.astype(bool)).sum())

    stat = step_km <= threshold_km
    fly_m = step_km > threshold_km
    mig50 = step_km > 50.0

    def _p(x, q): return float(np.percentile(x, q))

    return {
        "mean_error_km": float(errors.mean()),
        "median_error_km": float(np.median(errors)),
        "p90_error_km": _p(errors, 90), "p95_error_km": _p(errors, 95),
        "stationary_error_mean_km": float(errors[stat].mean()) if stat.any() else None,
        "fly_error_mean_km": float(errors[fly_m].mean()) if fly_m.any() else None,
        "migration50_error_mean_km": float(errors[mig50].mean()) if mig50.any() else None,
        "gated_mean_error_km": float(errors[pf].mean()) if pf.any() else None,
        "gated_median_error_km": float(np.median(errors[pf])) if pf.any() else None,
        "gated_p90_error_km": _p(errors[pf], 90) if pf.any() else None,
        "gated_p95_error_km": _p(errors[pf], 95) if pf.any() else None,
        "gated_stationary_error_mean_km": float(errors[~pf].mean()) if (~pf).any() else None,
        "gated_fly_error_mean_km": float(errors[pf & fly_m].mean()) if (pf & fly_m).any() else None,
        "gated_migration50_error_mean_km": float(errors[pf & mig50].mean()) if (pf & mig50).any() else None,
        "fly_threshold": float(best_t),
        "fly_tp": tp, "fly_fp": fp, "fly_tn": tn, "fly_fn": fn,
        "fly_accuracy": (tp+tn)/max(tp+fp+tn+fn,1),
        "fly_precision": tp/max(tp+fp,1), "fly_recall": tp/max(tp+fn,1),
        "fly_specificity": tn/max(tn+fp,1),
        "fly_f1": 2*tp/max(2*tp+fp+fn,1),
        "fly_false_positive_rate": fp/max(fp+tn,1),
    }


def train_one_model(*, model, model_name, dataset, train_idx, test_idx,
                    labels, direction_targets, target_delta, current_lat,
                    current_lon, target_step_km, output_dir, max_epochs,
                    patience, device, seed_offset=0):
    set_seed(SEED + seed_offset)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=128,
                              shuffle=True, generator=torch.Generator().manual_seed(SEED+seed_offset))
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=256, shuffle=False)

    model = model.to(device)
    train_pos = float(labels[train_idx].sum())
    train_neg = float(len(train_idx) - train_pos)
    pw = torch.tensor(min(train_neg / max(train_pos, 1.0), POS_WEIGHT_CAP), device=device)

    fly_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    mse_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_steps = max_epochs * len(train_loader)
    warmup = int(total_steps * 0.1)
    sched = WarmupCosineScheduler(opt, warmup, total_steps)

    best_loss = float("inf"); best_epoch = 0; no_imp = 0
    best_state = {}
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train(); tr_loss = 0.0
        for feats, _, fly_l, ld, direc in train_loader:
            feats = feats.to(device); fly_l = fly_l.to(device)
            ld = ld.to(device); direc = direc.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(feats)
            loss = (fly_loss_fn(out["fly_logit"], fly_l) +
                    mse_fn(out["log_distance"], ld) +
                    mse_fn(out["direction"], direc))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tr_loss += loss.item() * feats.size(0)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for feats, _, fly_l, ld, direc in test_loader:
                feats = feats.to(device)
                out = model(feats)
                loss = (fly_loss_fn(out["fly_logit"], fly_l.to(device)) +
                        mse_fn(out["log_distance"], ld.to(device)) +
                        mse_fn(out["direction"], direc.to(device)))
                val_loss += loss.item() * feats.size(0)
        val_loss /= len(test_idx)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss; best_epoch = epoch; no_imp = 0
            best_state = {"model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(best_state["model_state"])
    model.eval()
    fly_ls, delta_ls = [], []
    with torch.no_grad():
        for feats, _, _, _, _ in test_loader:
            out = model(feats.to(device))
            fly_ls.append(out["fly_logit"].cpu().numpy())
            delta_ls.append(out["direction"].cpu().numpy())

    fly_logits = np.concatenate(fly_ls)
    delta_preds = np.concatenate(delta_ls)

    metrics = compute_metrics(
        targets=target_delta[test_idx], predictions=delta_preds,
        cur_lat=current_lat[test_idx], cur_lon=current_lon[test_idx],
        step_km=target_step_km[test_idx], fly_labels=labels[test_idx],
        fly_logits=fly_logits, threshold_km=FLY_THRESHOLD_KM,
    )
    metrics["model"] = model_name
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = float(best_loss)
    metrics["train_samples"] = len(train_idx)
    metrics["test_samples"] = len(test_idx)
    metrics["params"] = sum(p.numel() for p in model.parameters())

    torch.save(best_state, model_dir / "best_model.pt")
    write_json(model_dir / "metrics.json", metrics)

    pred_rows = [{"idx": int(test_idx[i]), "fly_logit": float(fly_logits[i]),
                  "fly_prob": float(1.0/(1.0+math.exp(-fly_logits[i]))),
                  "delta_lat_pred": float(delta_preds[i,0]),
                  "delta_lon_pred": float(delta_preds[i,1])}
                 for i in range(len(test_idx))]
    write_csv(model_dir / "predictions.csv", pred_rows)

    return metrics


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def run(args):
    smoke = args.mode == "smoke"
    k_values = [7] if smoke else K_VALUES
    model_names = list(MODEL_REGISTRY)
    max_epochs = 1 if smoke else args.epochs
    patience = 1 if smoke else args.patience
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_csv = DATA_DIR / "combined_southbound_paths_with_weather_matched.csv"
    if not local_csv.exists():
        if not SOURCE_CSV.exists():
            raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
        shutil.copy2(SOURCE_CSV, local_csv)

    df = pd.read_csv(local_csv)
    df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["precipitation_log1p_mm"] = np.log1p(df["precipitation_mm"])
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTPUT_DIR / f"run_{args.mode}.log")
    logging.info("Device: %s | Mode: %s | k: %s | Models: %s", device, args.mode, k_values, model_names)

    all_rows = []
    for use_weather in [True, False]:
        wlabel = "with_weather" if use_weather else "without_weather"
        wdir = OUTPUT_DIR / wlabel
        for k in k_values:
            logging.info("=== %s k=%d ===", wlabel, k)
            wd = build_windows(df, k=k, use_weather=use_weather)
            if len(wd.features) == 0:
                logging.warning("No windows for k=%d", k); continue
            train_idx, test_idx = make_year_split(wd)
            mean, std = fit_normalizer(wd.features, train_idx)
            normed = normalize(wd.features, mean, std)
            ds = TrilineDataset(normed, wd.bird_ids, wd.labels, wd.log_distance_targets, wd.direction_targets)

            for mname in model_names:
                logging.info("Training %s k=%d %s", mname, k, wlabel)
                t0 = time.time()
                model = MODEL_REGISTRY[mname](normed.shape[-1], k)
                row = train_one_model(
                    model=model, model_name=f"{mname}_k{k}", dataset=ds,
                    train_idx=train_idx, test_idx=test_idx, labels=wd.labels,
                    direction_targets=wd.direction_targets, target_delta=wd.target_delta,
                    current_lat=wd.current_lat, current_lon=wd.current_lon,
                    target_step_km=wd.target_step_km, output_dir=wdir,
                    max_epochs=max_epochs, patience=patience, device=device,
                    seed_offset=k + hash(mname) % 1000,
                )
                row["k"] = k; row["weather"] = wlabel
                row["runtime_seconds"] = time.time() - t0
                row["model_name"] = mname
                all_rows.append(row)
                logging.info("  %s k=%d %s: mean=%.4f med=%.4f fly_rec=%.4f",
                             mname, k, wlabel, row["mean_error_km"],
                             row["median_error_km"], row["fly_recall"])

    write_csv(OUTPUT_DIR / "comparison_summary.csv", all_rows)

    # Markdown report
    lines = ["# Transformer Optimization Experiment V2 Results", "",
             f"- Mode: {args.mode}  |  Epochs: {max_epochs}  |  Patience: {patience}",
             f"- Fly threshold: {FLY_THRESHOLD_KM} km  |  Split: year-grouped 80/20", ""]
    for use_weather in [True, False]:
        label = "With Weather" if use_weather else "Without Weather"
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Model | k | Mean km | Median km | P90 km | Fly Recall | Params |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(all_rows, key=lambda r: (r["weather"] != str(use_weather), r["k"], r["model_name"])):
            if str(row["weather"]) != str(use_weather): continue
            lines.append(f"| {row['model_name']} | {int(row['k'])} | "
                         f"{float(row['mean_error_km']):.4f} | {float(row['median_error_km']):.4f} | "
                         f"{float(row['p90_error_km']):.4f} | {float(row['fly_recall']):.4f} | "
                         f"{int(row['params']):,} |")
        lines.append("")
    (OUTPUT_DIR / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / f"COMPLETED_{args.mode.upper()}.txt").write_text(
        f"Done. {args.mode} results saved to {OUTPUT_DIR}\n", encoding="utf-8")
    logging.info("Done. Results → %s", OUTPUT_DIR)


def setup_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(); root.handlers.clear(); root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="w", encoding="utf-8")]:
        h.setFormatter(fmt); root.addHandler(h)


def parse_args():
    p = argparse.ArgumentParser(description="Transformer V3 vs LSTM — Experiment V2")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
