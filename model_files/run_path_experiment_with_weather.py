from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model import TrilineTransformer


ROOT = Path(__file__).resolve().parents[1]
SOUTHBOUND_CSV = (
    ROOT / "data" / "filtered" / "dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv"
)
WEATHER_ROOT = ROOT / "data" / "weather"
NOWEATHER_OUTPUT_DIR = ROOT / "Path_Experiment_Noweather"
GROUP_COLUMN = "path_id"
IDENTITY_COLUMN = "source_individual_local_identifier"
SEED = 42
POS_WEIGHT_CAP = 10.0

PATH_FEATURE_COLUMNS = [
    "lat_median",
    "lon_median",
    "delta_lat",
    "delta_lon",
    "step_length_km",
    "heading_sin",
    "heading_cos",
    "doy_sin",
    "doy_cos",
    "stopover_duration_days",
]
WEATHER_RAW_COLUMNS = [
    "temp_2m_celsius",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure_hpa",
    "precipitation_mm",
    "cloud_cover",
    "boundary_layer_height",
]
WEATHER_FEATURE_COLUMNS = [
    "temp_2m_celsius",
    "wind_speed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "surface_pressure_hpa",
    "precipitation_log1p_mm",
    "cloud_cover",
    "boundary_layer_height",
]
COMPACT_FEATURE_COLUMNS = PATH_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS
DELTA_FEATURE_COLUMNS = ["delta_lat", "delta_lon"]

FULL_BASELINE_K = [7, 14, 30]
FULL_DIRECT_TRANSFORMER_K = [7, 14, 30]
FULL_TRILINE_TRANSFORMER_2L_K = [7, 14, 30]
FULL_CONTEXT_K = [7, 14, 30]
ROLLOUT_CONTEXT_K = 30
ROLLOUT_MIN_DISPLACEMENT_KM = 50.0
ROLLOUT_LAT_MIN = 30.0
ROLLOUT_LAT_MAX = 55.0


@dataclass(frozen=True)
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
    target_path_ids: np.ndarray
    target_source_birds: np.ndarray
    bird_to_idx: dict[str, int]
    feature_columns: list[str]


class DirectDeltaDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        bird_ids: np.ndarray,
        target_delta_norm: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.bird_ids = torch.as_tensor(bird_ids, dtype=torch.long)
        self.target_delta_norm = torch.as_tensor(target_delta_norm, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return self.features[idx], self.bird_ids[idx], self.target_delta_norm[idx]


class TrilineDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        bird_ids: np.ndarray,
        labels: np.ndarray,
        log_distance_targets: np.ndarray,
        direction_targets: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.bird_ids = torch.as_tensor(bird_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.log_distance_targets = torch.as_tensor(log_distance_targets, dtype=torch.float32)
        self.direction_targets = torch.as_tensor(direction_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.features[idx],
            self.bird_ids[idx],
            self.labels[idx],
            self.log_distance_targets[idx],
            self.direction_targets[idx],
        )


class DirectTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_birds: int,
        max_k: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        bird_embedding_dim: int = 2,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, d_model)
        self.bird_embedding = nn.Embedding(n_birds, bird_embedding_dim)
        self.bird_projection = nn.Linear(bird_embedding_dim, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_k, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[1]
        x = self.feature_projection(features)
        x = x + self.positional_embedding[:, :seq_len, :]
        x = x + self.bird_projection(self.bird_embedding(bird_ids)).unsqueeze(1)
        encoded = self.encoder(x)
        return self.head(encoded[:, -1, :])


class DirectMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_birds: int,
        k: int,
        last_day_only: bool,
        hidden_dim: int = 128,
        bird_embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.last_day_only = last_day_only
        input_dim = n_features if last_day_only else k * n_features
        self.bird_embedding = nn.Embedding(n_birds, bird_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + bird_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor) -> torch.Tensor:
        if self.last_day_only:
            x = features[:, -1, :]
        else:
            x = features.flatten(start_dim=1)
        x = torch.cat([x, self.bird_embedding(bird_ids)], dim=1)
        return self.net(x)


class TrilineLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_birds: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        bird_embedding_dim: int = 2,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(n_features, hidden_dim)
        self.bird_embedding = nn.Embedding(n_birds, bird_embedding_dim)
        self.bird_projection = nn.Linear(bird_embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fly_head = triline_head(hidden_dim, 1, dropout)
        self.distance_head = triline_head(hidden_dim, 1, dropout)
        self.direction_head = triline_head(hidden_dim, 2, dropout)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.feature_projection(features)
        x = x + self.bird_projection(self.bird_embedding(bird_ids)).unsqueeze(1)
        encoded, _ = self.lstm(x)
        pooled = encoded[:, -1, :]
        return {
            "fly_logit": self.fly_head(pooled).squeeze(-1),
            "log_distance": self.distance_head(pooled).squeeze(-1),
            "direction": self.direction_head(pooled),
        }


class TrilineLinearAR(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_birds: int,
        k: int,
        hidden_dim: int = 128,
        bird_embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bird_embedding = nn.Embedding(n_birds, bird_embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(k * n_features + bird_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fly_head = triline_head(hidden_dim, 1, dropout)
        self.distance_head = triline_head(hidden_dim, 1, dropout)
        self.direction_head = triline_head(hidden_dim, 2, dropout)

    def forward(self, features: torch.Tensor, bird_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([features.flatten(start_dim=1), self.bird_embedding(bird_ids)], dim=1)
        rep = self.projection(x)
        return {
            "fly_logit": self.fly_head(rep).squeeze(-1),
            "log_distance": self.distance_head(rep).squeeze(-1),
            "direction": self.direction_head(rep),
        }


def triline_head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def load_weather_data(weather_root: Path = WEATHER_ROOT) -> pd.DataFrame:
    files = sorted(weather_root.glob("weather_*/dataset2_daily_weather_*.csv"))
    if not files:
        raise FileNotFoundError(f"No weather CSV files found under {weather_root}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["weather_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    weather["weather_bird"] = weather["individual_local_identifier"].astype(str)
    weather["weather_date"] = pd.to_datetime(weather["date"], format="mixed").dt.normalize()
    for column in WEATHER_RAW_COLUMNS:
        weather[column] = (
            pd.to_numeric(weather[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
    keep_columns = ["weather_bird", "weather_date", "weather_file", *WEATHER_RAW_COLUMNS]
    return weather[keep_columns].sort_values(["weather_bird", "weather_date"]).reset_index(drop=True)


def nearest_same_bird_weather_row(
    weather_by_bird: dict[str, pd.DataFrame],
    bird: str,
    date_value: pd.Timestamp,
) -> tuple[pd.Series | None, int | None]:
    bird_weather = weather_by_bird.get(bird)
    if bird_weather is None or bird_weather.empty:
        return None, None

    dates = bird_weather["weather_date"].to_numpy(dtype="datetime64[ns]")
    target = np.datetime64(date_value.to_datetime64())
    insert_at = int(np.searchsorted(dates, target))
    candidates = []
    if insert_at < len(dates):
        candidates.append(insert_at)
    if insert_at > 0:
        candidates.append(insert_at - 1)
    if not candidates:
        return None, None

    best_idx = min(candidates, key=lambda idx: abs((dates[idx] - target).astype("timedelta64[D]").astype(int)))
    match_days = int(abs((dates[best_idx] - target).astype("timedelta64[D]").astype(int)))
    return bird_weather.iloc[best_idx], match_days


def add_weather_to_paths(df: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = df.copy()
    paths["_row_id"] = np.arange(len(paths))
    paths["_path_bird"] = paths[IDENTITY_COLUMN].astype(str)
    paths["_path_date"] = pd.to_datetime(paths["date"], utc=True, format="mixed").dt.tz_convert(None).dt.normalize()

    merged = paths.merge(
        weather,
        left_on=["_path_bird", "_path_date"],
        right_on=["weather_bird", "weather_date"],
        how="left",
        sort=False,
    )
    merged["weather_match_type"] = np.where(merged["weather_date"].notna(), "exact", "nearest_same_bird")
    merged["weather_match_days"] = np.where(merged["weather_date"].notna(), 0, np.nan)

    weather_by_bird = {
        bird: bird_weather.reset_index(drop=True)
        for bird, bird_weather in weather.groupby("weather_bird", sort=False)
    }
    missing_mask = merged["weather_date"].isna()
    for idx in merged.index[missing_mask]:
        match_row, match_days = nearest_same_bird_weather_row(
            weather_by_bird,
            str(merged.at[idx, "_path_bird"]),
            pd.Timestamp(merged.at[idx, "_path_date"]),
        )
        if match_row is None:
            merged.at[idx, "weather_match_type"] = "missing_same_bird"
            continue
        for column in ["weather_bird", "weather_date", "weather_file", *WEATHER_RAW_COLUMNS]:
            merged.at[idx, column] = match_row[column]
        merged.at[idx, "weather_match_days"] = int(match_days or 0)

    for column in WEATHER_RAW_COLUMNS:
        merged[column] = (
            pd.to_numeric(merged[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
    missing_weather_values = {
        column: int(merged[column].isna().sum())
        for column in WEATHER_RAW_COLUMNS
    }
    merged[WEATHER_RAW_COLUMNS] = merged[WEATHER_RAW_COLUMNS].fillna(0.0)

    wind_rad = np.deg2rad(merged["wind_direction_10m"].fillna(0.0).astype(float))
    merged["wind_dir_sin"] = np.sin(wind_rad)
    merged["wind_dir_cos"] = np.cos(wind_rad)
    merged["precipitation_log1p_mm"] = np.log1p(merged["precipitation_mm"].clip(lower=0.0))

    exact_count = int((merged["weather_match_type"] == "exact").sum())
    nearest_count = int((merged["weather_match_type"] == "nearest_same_bird").sum())
    missing_count = int((merged["weather_match_type"] == "missing_same_bird").sum())
    summary = {
        "weather_root": str(weather_root_path(weather)),
        "weather_rows": int(len(weather)),
        "weather_birds": int(weather["weather_bird"].nunique()),
        "weather_date_min": str(weather["weather_date"].min().date()),
        "weather_date_max": str(weather["weather_date"].max().date()),
        "path_rows": int(len(paths)),
        "exact_matches": exact_count,
        "nearest_same_bird_matches": nearest_count,
        "missing_same_bird_matches": missing_count,
        "max_weather_match_days": int(pd.to_numeric(merged["weather_match_days"], errors="coerce").fillna(0).max()),
        "missing_weather_values_after_match_before_fill": missing_weather_values,
        "weather_feature_columns": WEATHER_FEATURE_COLUMNS,
        "weather_raw_columns": WEATHER_RAW_COLUMNS,
    }

    drop_columns = ["_row_id", "_path_bird", "_path_date"]
    return merged.drop(columns=[column for column in drop_columns if column in merged.columns]), summary


def weather_root_path(weather: pd.DataFrame) -> Path:
    return WEATHER_ROOT


def input_label(input_csv: Path) -> str:
    try:
        return str(input_csv.relative_to(ROOT))
    except ValueError:
        return str(input_csv)


def use_prejoined_weather_paths(df: pd.DataFrame, input_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    label = input_label(input_csv)
    required_columns = [GROUP_COLUMN, IDENTITY_COLUMN, "date", *WEATHER_RAW_COLUMNS]
    missing_columns = [column for column in required_columns if column not in out.columns]
    if missing_columns:
        raise ValueError(f"Prejoined weather input is missing required columns: {missing_columns}")

    out["weather_bird"] = out.get("weather_bird", out[IDENTITY_COLUMN].astype(str))
    out["weather_bird"] = out["weather_bird"].fillna(out[IDENTITY_COLUMN].astype(str)).astype(str)
    out["weather_date"] = out.get("weather_date", out["date"])
    out["weather_date"] = out["weather_date"].fillna(out["date"])
    out["weather_date"] = pd.to_datetime(out["weather_date"], utc=True, format="mixed").dt.normalize()
    if "weather_file" not in out.columns:
        out["weather_file"] = label
    out["weather_file"] = out["weather_file"].fillna(label).astype(str)
    if "weather_match_type" not in out.columns:
        out["weather_match_type"] = "prejoined"
    out["weather_match_type"] = out["weather_match_type"].fillna("prejoined").astype(str)
    out["weather_match_days"] = pd.to_numeric(out.get("weather_match_days", 0), errors="coerce").fillna(0)

    for column in WEATHER_RAW_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_weather_values = {
        column: int(out[column].isna().sum())
        for column in WEATHER_RAW_COLUMNS
    }
    out[WEATHER_RAW_COLUMNS] = out[WEATHER_RAW_COLUMNS].fillna(0.0)

    wind_rad = np.deg2rad(out["wind_direction_10m"].fillna(0.0).astype(float))
    out["wind_dir_sin"] = np.sin(wind_rad)
    out["wind_dir_cos"] = np.cos(wind_rad)
    out["precipitation_log1p_mm"] = np.log1p(out["precipitation_mm"].clip(lower=0.0))

    match_counts = out["weather_match_type"].astype(str).value_counts().to_dict()
    summary = {
        "input_csv": str(input_csv),
        "prejoined_weather_data": True,
        "path_rows": int(len(out)),
        "paths": int(out[GROUP_COLUMN].nunique()),
        "source_birds": int(out[IDENTITY_COLUMN].nunique()),
        "weather_birds": int(out["weather_bird"].nunique()),
        "weather_date_min": str(pd.to_datetime(out["weather_date"], utc=True, format="mixed").min().date()),
        "weather_date_max": str(pd.to_datetime(out["weather_date"], utc=True, format="mixed").max().date()),
        "weather_match_type_counts": {str(key): int(value) for key, value in match_counts.items()},
        "max_weather_match_days": int(pd.to_numeric(out["weather_match_days"], errors="coerce").fillna(0).max()),
        "missing_weather_values_after_match_before_fill": missing_weather_values,
        "weather_feature_columns": WEATHER_FEATURE_COLUMNS,
        "weather_raw_columns": WEATHER_RAW_COLUMNS,
    }
    return out, summary


def add_southbound_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, format="mixed")
    out = out.sort_values([GROUP_COLUMN, "date"]).reset_index(drop=True)
    grouped = out.groupby(GROUP_COLUMN, group_keys=False)
    out["delta_lat"] = grouped["lat_median"].diff().fillna(0.0)
    out["delta_lon"] = grouped["lon_median"].diff().fillna(0.0)

    heading_rad = np.deg2rad(out["heading_deg"].fillna(0.0).astype(float))
    out["heading_sin"] = np.sin(heading_rad)
    out["heading_cos"] = np.cos(heading_rad)

    day_of_year = out["date"].dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2.0 * math.pi * day_of_year / 366.0)
    out["doy_cos"] = np.cos(2.0 * math.pi * day_of_year / 366.0)

    for column in COMPACT_FEATURE_COLUMNS:
        out[column] = (
            pd.to_numeric(out[column], errors="coerce")
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
    return out


def summarize_data(df: pd.DataFrame, input_csv: Path = SOUTHBOUND_CSV) -> dict[str, object]:
    return {
        "input_csv": str(input_csv),
        "rows": int(len(df)),
        "paths": int(df[GROUP_COLUMN].nunique()),
        "source_birds": int(df[IDENTITY_COLUMN].nunique()),
        "date_min": str(pd.to_datetime(df["date"], utc=True, format="mixed").min()),
        "date_max": str(pd.to_datetime(df["date"], utc=True, format="mixed").max()),
        "feature_columns": COMPACT_FEATURE_COLUMNS,
        "path_feature_columns": PATH_FEATURE_COLUMNS,
        "weather_feature_columns": WEATHER_FEATURE_COLUMNS,
        "n_features": len(COMPACT_FEATURE_COLUMNS),
        "weather_data": True,
        "group_column": GROUP_COLUMN,
        "identity_column": IDENTITY_COLUMN,
    }


def build_windows(df: pd.DataFrame, k: int, fly_threshold_km: float) -> WindowData:
    featured = add_southbound_compact_features(df)
    identity_names = sorted(featured[IDENTITY_COLUMN].astype(str).unique())
    bird_to_idx = {bird: idx for idx, bird in enumerate(identity_names)}

    features: list[np.ndarray] = []
    labels: list[float] = []
    log_distance_targets: list[float] = []
    direction_targets: list[tuple[float, float]] = []
    bird_ids: list[int] = []
    target_dates: list[np.datetime64] = []
    target_step_km: list[float] = []
    current_lat: list[float] = []
    current_lon: list[float] = []
    target_lat: list[float] = []
    target_lon: list[float] = []
    target_delta: list[tuple[float, float]] = []
    target_path_ids: list[str] = []
    target_source_birds: list[str] = []

    for path_id, path_df in featured.groupby(GROUP_COLUMN, sort=False):
        path_df = path_df.sort_values("date").reset_index(drop=True)
        dates = path_df["date"].dt.normalize()
        day_diffs = dates.diff().dt.days.fillna(1).to_numpy()

        for start in range(0, len(path_df) - k):
            end = start + k
            if not np.all(day_diffs[start + 1 : end + 1] == 1):
                continue

            input_rows = path_df.iloc[start:end]
            current_row = path_df.iloc[end - 1]
            target_row = path_df.iloc[end]
            source_bird = str(target_row[IDENTITY_COLUMN])
            step_km = float(target_row["step_length_km"])
            heading_deg = (
                float(target_row["heading_deg"]) if pd.notna(target_row["heading_deg"]) else 0.0
            )
            heading_rad = math.radians(heading_deg)

            features.append(input_rows[COMPACT_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            labels.append(float(step_km > fly_threshold_km))
            log_distance_targets.append(math.log1p(max(step_km, 0.0)))
            direction_targets.append((math.sin(heading_rad), math.cos(heading_rad)))
            bird_ids.append(bird_to_idx[source_bird])
            target_dates.append(np.datetime64(target_row["date"].to_datetime64()))
            target_step_km.append(step_km)
            current_lat.append(float(current_row["lat_median"]))
            current_lon.append(float(current_row["lon_median"]))
            target_lat_value = float(target_row["lat_median"])
            target_lon_value = float(target_row["lon_median"])
            target_lat.append(target_lat_value)
            target_lon.append(target_lon_value)
            target_delta.append(
                (
                    target_lat_value - float(current_row["lat_median"]),
                    target_lon_value - float(current_row["lon_median"]),
                )
            )
            target_path_ids.append(str(path_id))
            target_source_birds.append(source_bird)

    if not features:
        empty_features = np.empty((0, k, len(COMPACT_FEATURE_COLUMNS)), dtype=np.float32)
        return WindowData(
            empty_features,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            bird_to_idx,
            COMPACT_FEATURE_COLUMNS.copy(),
        )

    return WindowData(
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        log_distance_targets=np.asarray(log_distance_targets, dtype=np.float32),
        direction_targets=np.asarray(direction_targets, dtype=np.float32),
        bird_ids=np.asarray(bird_ids, dtype=np.int64),
        target_dates=np.asarray(target_dates, dtype="datetime64[ns]"),
        target_step_km=np.asarray(target_step_km, dtype=np.float32),
        current_lat=np.asarray(current_lat, dtype=np.float32),
        current_lon=np.asarray(current_lon, dtype=np.float32),
        target_lat=np.asarray(target_lat, dtype=np.float32),
        target_lon=np.asarray(target_lon, dtype=np.float32),
        target_delta=np.asarray(target_delta, dtype=np.float32),
        target_path_ids=np.asarray(target_path_ids, dtype=object),
        target_source_birds=np.asarray(target_source_birds, dtype=object),
        bird_to_idx=bird_to_idx,
        feature_columns=COMPACT_FEATURE_COLUMNS.copy(),
    )


def make_chronological_split(window_data: WindowData, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(window_data.target_dates, kind="stable")
    split_idx = int(len(order) * train_fraction)
    if split_idx <= 0 or split_idx >= len(order):
        raise ValueError(f"Need enough windows for split, got {len(order)}")
    return order[:split_idx], order[split_idx:]


def fit_normalizer(features: np.ndarray, train_idx: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    train_features = features[np.asarray(list(train_idx))]
    flat = train_features.reshape(-1, train_features.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def fit_target_delta_normalizer(target_delta: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_targets = target_delta[train_idx]
    mean = train_targets.mean(axis=0).astype(np.float32)
    std = train_targets.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.deg2rad(lat1.astype(float))
    lon1_rad = np.deg2rad(lon1.astype(float))
    lat2_rad = np.deg2rad(lat2.astype(float))
    lon2_rad = np.deg2rad(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return radius_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def reconstruct_from_distance_heading(
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    distance_km: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = direction / np.maximum(norm, 1e-6)
    heading_rad = np.arctan2(unit[:, 0], unit[:, 1])
    delta_lat = distance_km * np.cos(heading_rad) / 111.0
    cos_lat = np.cos(np.deg2rad(current_lat.astype(float)))
    delta_lon = distance_km * np.sin(heading_rad) / (111.0 * np.maximum(np.abs(cos_lat), 1e-6))
    return current_lat + delta_lat, current_lon + delta_lon


def gps_metrics(
    errors_km: np.ndarray,
    target_step_km: np.ndarray,
    fly_threshold_km: float,
) -> dict[str, float | None]:
    def subset_mean(mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(errors_km[mask].mean())

    if len(errors_km) == 0:
        return {
            "mean_error_km": None,
            "median_error_km": None,
            "p90_error_km": None,
            "p95_error_km": None,
            "stationary_error_mean_km": None,
            "fly_error_mean_km": None,
            "migration50_error_mean_km": None,
        }

    stationary_mask = target_step_km <= fly_threshold_km
    fly_mask = target_step_km > fly_threshold_km
    migration50_mask = target_step_km > 50.0
    return {
        "mean_error_km": float(errors_km.mean()),
        "median_error_km": float(np.median(errors_km)),
        "p90_error_km": float(np.percentile(errors_km, 90)),
        "p95_error_km": float(np.percentile(errors_km, 95)),
        "stationary_error_mean_km": subset_mean(stationary_mask),
        "fly_error_mean_km": subset_mean(fly_mask),
        "migration50_error_mean_km": subset_mean(migration50_mask),
    }


def fly_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    preds = (scores >= threshold).astype(np.int64)
    y = labels.astype(np.int64)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    metrics: dict[str, float | int | None] = {
        "fly_threshold": float(threshold),
        "fly_tp": tp,
        "fly_fp": fp,
        "fly_tn": tn,
        "fly_fn": fn,
        "fly_accuracy": float((tp + tn) / max(tp + fp + tn + fn, 1)),
        "fly_precision": float(precision),
        "fly_recall": float(recall),
        "fly_specificity": float(specificity),
        "fly_f1": float(f1),
        "fly_false_positive_rate": float(fp / max(fp + tn, 1)),
    }
    if len(np.unique(y)) == 2:
        metrics["fly_roc_auc"] = float(roc_auc_score(y, scores))
        metrics["fly_pr_auc"] = float(average_precision_score(y, scores))
    else:
        metrics["fly_roc_auc"] = None
        metrics["fly_pr_auc"] = None
    return metrics


def threshold_sweep(labels: np.ndarray, scores: np.ndarray, output_csv: Path) -> tuple[float, dict[str, object]]:
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        thresholds = np.array([0.5], dtype=np.float32)
    else:
        thresholds = np.unique(
            np.concatenate(
                [
                    np.linspace(float(finite_scores.min()), float(finite_scores.max()), 201),
                    np.quantile(finite_scores, np.linspace(0.0, 1.0, 101)),
                ]
            )
        )
    rows: list[dict[str, object]] = []
    best_threshold = float(thresholds[0])
    best_metrics = fly_metrics(labels, scores, best_threshold)
    for threshold in thresholds:
        metrics = fly_metrics(labels, scores, float(threshold))
        rows.append(metrics)
        if (
            float(metrics["fly_f1"]) > float(best_metrics["fly_f1"])
            or (
                float(metrics["fly_f1"]) == float(best_metrics["fly_f1"])
                and float(metrics["fly_precision"]) > float(best_metrics["fly_precision"])
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return best_threshold, best_metrics


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def release_torch_objects(*objects: object) -> None:
    for obj in objects:
        if isinstance(obj, nn.Module):
            obj.to("cpu")
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({key: json_safe(value) for key, value in payload.items()}, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    output_csv: Path,
    window_data: WindowData,
    test_idx: np.ndarray,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    error_km: np.ndarray,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    rows: dict[str, Any] = {
        "target_date": pd.to_datetime(window_data.target_dates[test_idx]).astype(str),
        "path_id": window_data.target_path_ids[test_idx],
        "source_bird_id": window_data.target_source_birds[test_idx],
        "current_lat": window_data.current_lat[test_idx],
        "current_lon": window_data.current_lon[test_idx],
        "target_lat": window_data.target_lat[test_idx],
        "target_lon": window_data.target_lon[test_idx],
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "error_km": error_km,
        "target_step_km": window_data.target_step_km[test_idx],
        "true_fly_label": window_data.labels[test_idx].astype(int),
    }
    if extras:
        rows.update(extras)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def train_direct_model(
    *,
    model_spec: dict[str, object],
    window_data: WindowData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    output_dir: Path,
    fly_threshold_km: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    start_time = time.time()
    model_dir = output_dir / str(model_spec["name"])
    model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED + int(model_spec["k"]))

    target_mean, target_std = fit_target_delta_normalizer(window_data.target_delta, train_idx)
    target_delta_norm = ((window_data.target_delta - target_mean) / target_std).astype(np.float32)
    dataset = DirectDeltaDataset(normalized_features, window_data.bird_ids, target_delta_norm)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + int(model_spec["k"])),
    )
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=256, shuffle=False)

    model_kind = str(model_spec["kind"])
    if model_kind == "direct_transformer":
        model = DirectTransformer(
            n_features=normalized_features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            max_k=int(model_spec["k"]),
            n_layers=int(model_spec.get("n_layers", 2)),
        )
    elif model_kind == "direct_mlp_last":
        model = DirectMLP(
            n_features=normalized_features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            k=int(model_spec["k"]),
            last_day_only=True,
        )
    elif model_kind == "direct_mlp_sequence":
        model = DirectMLP(
            n_features=normalized_features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            k=int(model_spec["k"]),
            last_day_only=False,
        )
    else:
        raise ValueError(f"Unknown direct model kind: {model_kind}")
    model = model.to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    log_rows: list[dict[str, object]] = []
    checkpoint_path = model_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, bird_ids, targets in train_loader:
            features = features.to(device)
            bird_ids = bird_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, bird_ids)
            loss = mse_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = int(targets.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_count += batch_size

        test_loss, _, _ = evaluate_direct(model, test_loader, target_mean, target_std, device)
        scheduler.step(test_loss)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "test_delta_mse_norm": test_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if test_loss < best_loss - 1e-5:
            best_loss = test_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": cpu_state_dict(model),
                    "model_spec": model_spec,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_delta_mean": target_mean,
                    "target_delta_std": target_std,
                    "feature_columns": window_data.feature_columns,
                    "model_feature_columns": window_data.feature_columns,
                    "model_n_features": int(normalized_features.shape[-1]),
                    "bird_to_idx": window_data.bird_to_idx,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    pd.DataFrame(log_rows).to_csv(model_dir / "training_log.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_loss, pred_delta, _ = evaluate_direct(model, test_loader, target_mean, target_std, device)
    pred_lat = window_data.current_lat[test_idx] + pred_delta[:, 0]
    pred_lon = window_data.current_lon[test_idx] + pred_delta[:, 1]
    errors = haversine_km(window_data.target_lat[test_idx], window_data.target_lon[test_idx], pred_lat, pred_lon)
    metrics = gps_metrics(errors, window_data.target_step_km[test_idx], fly_threshold_km)

    save_predictions(
        model_dir / "predictions.csv",
        window_data,
        test_idx,
        pred_lat,
        pred_lon,
        errors,
        extras={"pred_delta_lat": pred_delta[:, 0], "pred_delta_lon": pred_delta[:, 1]},
    )
    metrics_payload: dict[str, object] = {
        "setup_fly_threshold_km": fly_threshold_km,
        "model": model_spec["name"],
        "model_family": "direct",
        "kind": model_kind,
        "k": int(model_spec["k"]),
        "n_layers": int(model_spec.get("n_layers", 0)),
        "params": parameter_count(model),
        "best_epoch": best_epoch,
        "best_test_loss": best_loss,
        "final_delta_mse_norm": final_loss,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_fly_rate": float(window_data.labels[train_idx].mean()),
        "test_fly_rate": float(window_data.labels[test_idx].mean()),
        "runtime_seconds": float(time.time() - start_time),
        **metrics,
    }
    write_json(model_dir / "metrics.json", metrics_payload)
    release_torch_objects(model, optimizer, scheduler)
    return metrics_payload


def evaluate_direct(
    model: nn.Module,
    loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    predictions: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    with torch.no_grad():
        for features, bird_ids, targets in loader:
            features = features.to(device)
            bird_ids = bird_ids.to(device)
            targets = targets.to(device)
            outputs = model(features, bird_ids)
            total_loss += float(mse_loss(outputs, targets).item())
            total_count += int(targets.shape[0])
            predictions.append(outputs.cpu().numpy())
            targets_out.append(targets.cpu().numpy())
    pred_norm = np.concatenate(predictions)
    target_norm = np.concatenate(targets_out)
    pred_delta = pred_norm * target_std.reshape(1, -1) + target_mean.reshape(1, -1)
    target_delta = target_norm * target_std.reshape(1, -1) + target_mean.reshape(1, -1)
    return total_loss / max(total_count, 1), pred_delta.astype(np.float32), target_delta.astype(np.float32)


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    log_distance_targets: torch.Tensor,
    direction_targets: torch.Tensor,
    fly_loss_fn: nn.Module,
    mse_loss_fn: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    fly_loss = fly_loss_fn(outputs["fly_logit"], labels)
    distance_loss = mse_loss_fn(outputs["log_distance"], log_distance_targets)
    moving_weight = labels.view(-1, 1)
    direction_error = (outputs["direction"] - direction_targets).pow(2)
    direction_loss = (direction_error * moving_weight).sum() / moving_weight.sum().clamp_min(1.0)
    total = fly_loss + distance_loss + 0.5 * direction_loss
    return total, {
        "fly_loss": float(fly_loss.detach().cpu().item()),
        "distance_loss": float(distance_loss.detach().cpu().item()),
        "direction_loss": float(direction_loss.detach().cpu().item()),
    }


def train_triline_model(
    *,
    model_spec: dict[str, object],
    window_data: WindowData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    output_dir: Path,
    fly_threshold_km: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    start_time = time.time()
    model_dir = output_dir / str(model_spec["name"])
    model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED + int(model_spec["k"]) + int(model_spec.get("n_layers", 0)))

    dataset = TrilineDataset(
        normalized_features,
        window_data.bird_ids,
        window_data.labels,
        window_data.log_distance_targets,
        window_data.direction_targets,
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + int(model_spec["k"])),
    )
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=256, shuffle=False)

    model_kind = str(model_spec["kind"])
    if model_kind == "triline_transformer":
        model = TrilineTransformer(
            n_features=normalized_features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            max_k=int(model_spec["k"]),
            n_layers=int(model_spec.get("n_layers", 2)),
            use_bird_id=True,
        )
    elif model_kind == "triline_lstm":
        model = TrilineLSTM(
            n_features=normalized_features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            n_layers=int(model_spec.get("n_layers", 2)),
        )
    elif model_kind == "triline_linear_ar":
        model = TrilineLinearAR(
            n_features=normalized_features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            k=int(model_spec["k"]),
        )
    else:
        raise ValueError(f"Unknown triline model kind: {model_kind}")
    model = model.to(device)

    train_positive = float(window_data.labels[train_idx].sum())
    train_negative = float(len(train_idx) - train_positive)
    raw_pos_weight = train_negative / max(train_positive, 1.0)
    pos_weight_value = min(raw_pos_weight, POS_WEIGHT_CAP)
    fly_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    log_rows: list[dict[str, object]] = []
    checkpoint_path = model_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        component_sums = {"fly_loss": 0.0, "distance_loss": 0.0, "direction_loss": 0.0}
        for features, bird_ids, labels, log_distance_targets, direction_targets in train_loader:
            features = features.to(device)
            bird_ids = bird_ids.to(device)
            labels = labels.to(device)
            log_distance_targets = log_distance_targets.to(device)
            direction_targets = direction_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, bird_ids)
            loss, components = multitask_loss(
                outputs, labels, log_distance_targets, direction_targets, fly_loss_fn, mse_loss_fn
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = int(labels.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_count += batch_size
            for key, value in components.items():
                component_sums[key] += value * batch_size

        test_loss, labels_np, probabilities, _, _ = evaluate_triline(
            model, test_loader, fly_loss_fn, mse_loss_fn, device
        )
        fixed_fly_metrics = fly_metrics(labels_np, probabilities, 0.5)
        scheduler.step(test_loss)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "train_fly_loss": component_sums["fly_loss"] / max(train_count, 1),
                "train_distance_loss": component_sums["distance_loss"] / max(train_count, 1),
                "train_direction_loss": component_sums["direction_loss"] / max(train_count, 1),
                "test_loss": test_loss,
                "fixed_0_5_fly_f1": fixed_fly_metrics["fly_f1"],
                "fixed_0_5_fly_precision": fixed_fly_metrics["fly_precision"],
                "fixed_0_5_fly_recall": fixed_fly_metrics["fly_recall"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if test_loss < best_loss - 1e-5:
            best_loss = test_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": cpu_state_dict(model),
                    "model_spec": model_spec,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "feature_columns": window_data.feature_columns,
                    "model_feature_columns": (
                        DELTA_FEATURE_COLUMNS
                        if model_spec.get("feature_mode", "full") == "delta"
                        else window_data.feature_columns
                    ),
                    "model_n_features": int(normalized_features.shape[-1]),
                    "bird_to_idx": window_data.bird_to_idx,
                    "pos_weight": pos_weight_value,
                    "raw_pos_weight": raw_pos_weight,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    pd.DataFrame(log_rows).to_csv(model_dir / "training_log.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_loss, labels_np, probabilities, pred_distances, pred_directions = evaluate_triline(
        model, test_loader, fly_loss_fn, mse_loss_fn, device
    )
    selected_threshold, tuned_fly_metrics = threshold_sweep(
        labels_np, probabilities, model_dir / "threshold_sweep.csv"
    )
    fixed_fly_metrics = fly_metrics(labels_np, probabilities, 0.5)

    pred_lat, pred_lon = reconstruct_from_distance_heading(
        window_data.current_lat[test_idx],
        window_data.current_lon[test_idx],
        pred_distances,
        pred_directions,
    )
    errors = haversine_km(window_data.target_lat[test_idx], window_data.target_lon[test_idx], pred_lat, pred_lon)
    primary_metrics = gps_metrics(errors, window_data.target_step_km[test_idx], fly_threshold_km)

    gated_distances = pred_distances.copy()
    gated_distances[probabilities < selected_threshold] = 0.0
    gated_lat, gated_lon = reconstruct_from_distance_heading(
        window_data.current_lat[test_idx],
        window_data.current_lon[test_idx],
        gated_distances,
        pred_directions,
    )
    gated_errors = haversine_km(
        window_data.target_lat[test_idx],
        window_data.target_lon[test_idx],
        gated_lat,
        gated_lon,
    )
    gated_metrics = {
        f"gated_{key}": value
        for key, value in gps_metrics(gated_errors, window_data.target_step_km[test_idx], fly_threshold_km).items()
    }

    save_predictions(
        model_dir / "predictions.csv",
        window_data,
        test_idx,
        pred_lat,
        pred_lon,
        errors,
        extras={
            "fly_probability": probabilities,
            "selected_fly_threshold": np.full_like(probabilities, selected_threshold),
            "predicted_distance_km": pred_distances,
            "gated_pred_lat": gated_lat,
            "gated_pred_lon": gated_lon,
            "gated_error_km": gated_errors,
        },
    )
    metrics_payload: dict[str, object] = {
        "setup_fly_threshold_km": fly_threshold_km,
        "model": model_spec["name"],
        "model_family": "triline",
        "kind": model_kind,
        "k": int(model_spec["k"]),
        "n_layers": int(model_spec.get("n_layers", 0)),
        "feature_mode": model_spec.get("feature_mode", "full"),
        "params": parameter_count(model),
        "best_epoch": best_epoch,
        "best_test_loss": best_loss,
        "final_test_loss": final_loss,
        "raw_pos_weight": raw_pos_weight,
        "pos_weight": pos_weight_value,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_fly_rate": float(window_data.labels[train_idx].mean()),
        "test_fly_rate": float(window_data.labels[test_idx].mean()),
        "runtime_seconds": float(time.time() - start_time),
        **primary_metrics,
        **gated_metrics,
        **tuned_fly_metrics,
    }
    for key, value in fixed_fly_metrics.items():
        metrics_payload[f"fixed_0_5_{key}"] = value
    write_json(model_dir / "metrics.json", metrics_payload)
    release_torch_objects(model, optimizer, scheduler)
    return metrics_payload


def evaluate_triline(
    model: nn.Module,
    loader: DataLoader,
    fly_loss_fn: nn.Module,
    mse_loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    labels_out: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    pred_distances: list[np.ndarray] = []
    pred_directions: list[np.ndarray] = []
    with torch.no_grad():
        for features, bird_ids, labels, log_distance_targets, direction_targets in loader:
            features = features.to(device)
            bird_ids = bird_ids.to(device)
            labels = labels.to(device)
            log_distance_targets = log_distance_targets.to(device)
            direction_targets = direction_targets.to(device)
            outputs = model(features, bird_ids)
            loss, _ = multitask_loss(
                outputs, labels, log_distance_targets, direction_targets, fly_loss_fn, mse_loss_fn
            )
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_count += int(labels.shape[0])
            labels_out.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(outputs["fly_logit"]).cpu().numpy())
            pred_distances.append(torch.expm1(outputs["log_distance"]).clamp_min(0.0).cpu().numpy())
            pred_directions.append(outputs["direction"].cpu().numpy())
    return (
        total_loss / max(total_count, 1),
        np.concatenate(labels_out),
        np.concatenate(probabilities),
        np.concatenate(pred_distances),
        np.concatenate(pred_directions),
    )


def run_baseline(
    *,
    name: str,
    window_data: WindowData,
    test_idx: np.ndarray,
    output_dir: Path,
    fly_threshold_km: float,
) -> dict[str, object]:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    if name == "persistence":
        pred_lat = window_data.current_lat[test_idx]
        pred_lon = window_data.current_lon[test_idx]
    elif name == "const_velocity":
        last_delta = window_data.features[test_idx][:, -1, [2, 3]]
        pred_lat = window_data.current_lat[test_idx] + last_delta[:, 0]
        pred_lon = window_data.current_lon[test_idx] + last_delta[:, 1]
    else:
        raise ValueError(f"Unknown baseline: {name}")

    errors = haversine_km(window_data.target_lat[test_idx], window_data.target_lon[test_idx], pred_lat, pred_lon)
    save_predictions(model_dir / "predictions.csv", window_data, test_idx, pred_lat, pred_lon, errors)
    metrics_payload: dict[str, object] = {
        "setup_fly_threshold_km": fly_threshold_km,
        "model": name,
        "model_family": "baseline",
        "kind": name,
        "k": int(window_data.features.shape[1]),
        "params": 0,
        "best_epoch": None,
        "train_samples": int(len(window_data.features) - len(test_idx)),
        "test_samples": int(len(test_idx)),
        "test_fly_rate": float(window_data.labels[test_idx].mean()),
        **gps_metrics(errors, window_data.target_step_km[test_idx], fly_threshold_km),
    }
    write_json(model_dir / "metrics.json", metrics_payload)
    pd.DataFrame([{"epoch": 0, "note": "non_trainable_baseline"}]).to_csv(
        model_dir / "training_log.csv", index=False
    )
    torch.save({"model": name, "non_trainable_baseline": True}, model_dir / "best_model.pt")
    return metrics_payload


def full_model_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for k in FULL_DIRECT_TRANSFORMER_K:
        specs.append({"name": f"direct_transformer_2l_k{k}", "kind": "direct_transformer", "k": k, "n_layers": 2})
    specs.extend(
        [
            {"name": "direct_mlp_last_day_k1", "kind": "direct_mlp_last", "k": 1},
            {"name": "direct_mlp_sequence_k30", "kind": "direct_mlp_sequence", "k": 30},
        ]
    )
    for k in FULL_TRILINE_TRANSFORMER_2L_K:
        specs.append({"name": f"triline_transformer_2l_k{k}", "kind": "triline_transformer", "k": k, "n_layers": 2})
    specs.append({"name": "triline_transformer_3l_k30", "kind": "triline_transformer", "k": 30, "n_layers": 3})
    for k in FULL_CONTEXT_K:
        specs.append({"name": f"triline_transformer_4l_k{k}", "kind": "triline_transformer", "k": k, "n_layers": 4})
    for k in FULL_CONTEXT_K:
        specs.append({"name": f"triline_lstm_2l_k{k}", "kind": "triline_lstm", "k": k, "n_layers": 2})
    for k in FULL_CONTEXT_K:
        specs.append({"name": f"triline_linear_ar_full_k{k}", "kind": "triline_linear_ar", "k": k, "feature_mode": "full"})
    specs.append(
        {
            "name": "triline_linear_ar_delta_k30",
            "kind": "triline_linear_ar",
            "k": 30,
            "feature_mode": "delta",
        }
    )
    return specs


def smoke_model_specs() -> list[dict[str, object]]:
    return [
        {"name": "direct_transformer_2l_k7", "kind": "direct_transformer", "k": 7, "n_layers": 2},
        {"name": "direct_mlp_last_day_k1", "kind": "direct_mlp_last", "k": 1},
        {"name": "direct_mlp_sequence_k30", "kind": "direct_mlp_sequence", "k": 30},
        {"name": "triline_transformer_2l_k7", "kind": "triline_transformer", "k": 7, "n_layers": 2},
        {"name": "triline_transformer_3l_k30", "kind": "triline_transformer", "k": 30, "n_layers": 3},
        {"name": "triline_transformer_4l_k30", "kind": "triline_transformer", "k": 30, "n_layers": 4},
        {"name": "triline_lstm_2l_k7", "kind": "triline_lstm", "k": 7, "n_layers": 2},
        {"name": "triline_linear_ar_full_k7", "kind": "triline_linear_ar", "k": 7, "feature_mode": "full"},
        {"name": "triline_linear_ar_delta_k30", "kind": "triline_linear_ar", "k": 30, "feature_mode": "delta"},
    ]


def needed_k_values(model_specs: list[dict[str, object]], include_full_baselines: bool) -> list[int]:
    values = {int(spec["k"]) for spec in model_specs}
    if include_full_baselines:
        values.update(FULL_BASELINE_K)
    else:
        values.update({7, 30})
    return sorted(values)


def select_feature_mode(
    normalized_features: np.ndarray,
    feature_columns: list[str],
    feature_mode: object,
) -> np.ndarray:
    if feature_mode == "delta":
        indices = [feature_columns.index(column) for column in DELTA_FEATURE_COLUMNS]
        return normalized_features[:, :, indices]
    return normalized_features


def build_direct_model_from_spec(
    model_spec: dict[str, object],
    n_features: int,
    n_birds: int,
) -> nn.Module:
    model_kind = str(model_spec["kind"])
    if model_kind == "direct_transformer":
        return DirectTransformer(
            n_features=n_features,
            n_birds=n_birds,
            max_k=int(model_spec["k"]),
            n_layers=int(model_spec.get("n_layers", 2)),
        )
    if model_kind == "direct_mlp_last":
        return DirectMLP(
            n_features=n_features,
            n_birds=n_birds,
            k=int(model_spec["k"]),
            last_day_only=True,
        )
    if model_kind == "direct_mlp_sequence":
        return DirectMLP(
            n_features=n_features,
            n_birds=n_birds,
            k=int(model_spec["k"]),
            last_day_only=False,
        )
    raise ValueError(f"Unknown direct model kind: {model_kind}")


def build_triline_model_from_spec(
    model_spec: dict[str, object],
    n_features: int,
    n_birds: int,
) -> nn.Module:
    model_kind = str(model_spec["kind"])
    if model_kind == "triline_transformer":
        return TrilineTransformer(
            n_features=n_features,
            n_birds=n_birds,
            max_k=int(model_spec["k"]),
            n_layers=int(model_spec.get("n_layers", 2)),
            use_bird_id=True,
        )
    if model_kind == "triline_lstm":
        return TrilineLSTM(
            n_features=n_features,
            n_birds=n_birds,
            n_layers=int(model_spec.get("n_layers", 2)),
        )
    if model_kind == "triline_linear_ar":
        return TrilineLinearAR(
            n_features=n_features,
            n_birds=n_birds,
            k=int(model_spec["k"]),
        )
    raise ValueError(f"Unknown triline model kind: {model_kind}")


def normalize_model_window(
    feature_window: np.ndarray,
    checkpoint: dict[str, object],
) -> np.ndarray:
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    normalized = normalize_features(feature_window[None, :, :], feature_mean, feature_std)
    model_spec = checkpoint.get("model_spec", {})
    if isinstance(model_spec, dict):
        return select_feature_mode(normalized, COMPACT_FEATURE_COLUMNS, model_spec.get("feature_mode", "full"))
    return normalized


def load_checkpoint_model(
    checkpoint_path: Path,
    family: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_spec = dict(checkpoint["model_spec"])
    n_features = int(checkpoint.get("model_n_features", len(checkpoint.get("model_feature_columns", COMPACT_FEATURE_COLUMNS))))
    n_birds = len(checkpoint["bird_to_idx"])
    if family == "direct":
        model = build_direct_model_from_spec(model_spec, n_features, n_birds)
    elif family == "triline":
        model = build_triline_model_from_spec(model_spec, n_features, n_birds)
    else:
        raise ValueError(f"Unknown checkpoint family: {family}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def append_predicted_feature_row(
    context_rows: list[dict[str, float | pd.Timestamp]],
    *,
    date: pd.Timestamp,
    lat: float,
    lon: float,
    weather_values: dict[str, float],
    stationary_threshold_km: float = 1.0,
) -> None:
    prev = context_rows[-1]
    prev_lat = float(prev["lat_median"])
    prev_lon = float(prev["lon_median"])
    delta_lat = lat - prev_lat
    delta_lon = lon - prev_lon
    step = float(haversine_km(np.array([prev_lat]), np.array([prev_lon]), np.array([lat]), np.array([lon]))[0])
    heading_rad = math.atan2(delta_lon * math.cos(math.radians(prev_lat)), delta_lat)
    day_of_year = float(pd.Timestamp(date).dayofyear)
    previous_stopover = float(prev.get("stopover_duration_days", 0.0))
    next_row = {
            "date": pd.Timestamp(date),
            "lat_median": float(lat),
            "lon_median": float(lon),
            "delta_lat": float(delta_lat),
            "delta_lon": float(delta_lon),
            "step_length_km": step,
            "heading_sin": math.sin(heading_rad),
            "heading_cos": math.cos(heading_rad),
            "doy_sin": math.sin(2.0 * math.pi * day_of_year / 366.0),
            "doy_cos": math.cos(2.0 * math.pi * day_of_year / 366.0),
            "stopover_duration_days": previous_stopover + 1.0 if step <= stationary_threshold_km else 0.0,
    }
    for column in WEATHER_FEATURE_COLUMNS:
        next_row[column] = float(weather_values.get(column, 0.0))
    context_rows.append(next_row)


def weather_values_from_row(row: pd.Series) -> dict[str, float]:
    return {column: float(row.get(column, 0.0)) for column in WEATHER_FEATURE_COLUMNS}


def dataframe_context_rows(path_df: pd.DataFrame, n_rows: int) -> list[dict[str, float | pd.Timestamp]]:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for _, row in path_df.iloc[:n_rows].iterrows():
        payload: dict[str, float | pd.Timestamp] = {"date": pd.Timestamp(row["date"])}
        for column in COMPACT_FEATURE_COLUMNS:
            payload[column] = float(row[column])
        rows.append(payload)
    return rows


def context_feature_window(
    context_rows: list[dict[str, float | pd.Timestamp]],
    k: int,
) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in COMPACT_FEATURE_COLUMNS] for row in context_rows[-k:]],
        dtype=np.float32,
    )


def rollout_direct_model(
    model: nn.Module,
    checkpoint: dict[str, object],
    path_df: pd.DataFrame,
    rollout_steps: int,
    device: torch.device,
    context_days: int = ROLLOUT_CONTEXT_K,
) -> tuple[list[float], list[float]]:
    model_spec = dict(checkpoint["model_spec"])
    k = int(model_spec["k"])
    context_rows = dataframe_context_rows(path_df, context_days)
    pred_lat = [float(row["lat_median"]) for row in context_rows]
    pred_lon = [float(row["lon_median"]) for row in context_rows]
    bird_to_idx = checkpoint["bird_to_idx"]
    bird_name = str(path_df.iloc[0][IDENTITY_COLUMN])
    bird_id = int(bird_to_idx.get(bird_name, 0))
    target_mean = np.asarray(checkpoint["target_delta_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_delta_std"], dtype=np.float32)

    with torch.no_grad():
        for step in range(rollout_steps):
            target_row = path_df.iloc[context_days + step]
            features = normalize_model_window(context_feature_window(context_rows, k), checkpoint)
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            bird_tensor = torch.as_tensor([bird_id], dtype=torch.long, device=device)
            pred_norm = model(feature_tensor, bird_tensor).detach().cpu().numpy()[0]
            delta = pred_norm * target_std + target_mean
            next_lat = float(context_rows[-1]["lat_median"]) + float(delta[0])
            next_lon = float(context_rows[-1]["lon_median"]) + float(delta[1])
            append_predicted_feature_row(
                context_rows,
                date=pd.Timestamp(target_row["date"]),
                lat=next_lat,
                lon=next_lon,
                weather_values=weather_values_from_row(target_row),
            )
            pred_lat.append(next_lat)
            pred_lon.append(next_lon)
    return pred_lat, pred_lon


def rollout_triline_model(
    model: nn.Module,
    checkpoint: dict[str, object],
    path_df: pd.DataFrame,
    rollout_steps: int,
    device: torch.device,
    context_days: int = ROLLOUT_CONTEXT_K,
) -> tuple[list[float], list[float], list[float]]:
    model_spec = dict(checkpoint["model_spec"])
    k = int(model_spec["k"])
    context_rows = dataframe_context_rows(path_df, context_days)
    pred_lat = [float(row["lat_median"]) for row in context_rows]
    pred_lon = [float(row["lon_median"]) for row in context_rows]
    fly_probs = [float("nan")] * context_days
    bird_to_idx = checkpoint["bird_to_idx"]
    bird_name = str(path_df.iloc[0][IDENTITY_COLUMN])
    bird_id = int(bird_to_idx.get(bird_name, 0))

    with torch.no_grad():
        for step in range(rollout_steps):
            target_row = path_df.iloc[context_days + step]
            features = normalize_model_window(context_feature_window(context_rows, k), checkpoint)
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            bird_tensor = torch.as_tensor([bird_id], dtype=torch.long, device=device)
            outputs = model(feature_tensor, bird_tensor)
            fly_prob = float(torch.sigmoid(outputs["fly_logit"]).detach().cpu().numpy()[0])
            distance = float(torch.expm1(outputs["log_distance"]).clamp_min(0.0).detach().cpu().numpy()[0])
            direction = outputs["direction"].detach().cpu().numpy()
            next_lat_arr, next_lon_arr = reconstruct_from_distance_heading(
                np.array([float(context_rows[-1]["lat_median"])]),
                np.array([float(context_rows[-1]["lon_median"])]),
                np.array([distance]),
                direction,
            )
            next_lat = float(next_lat_arr[0])
            next_lon = float(next_lon_arr[0])
            append_predicted_feature_row(
                context_rows,
                date=pd.Timestamp(target_row["date"]),
                lat=next_lat,
                lon=next_lon,
                weather_values=weather_values_from_row(target_row),
            )
            pred_lat.append(next_lat)
            pred_lon.append(next_lon)
            fly_probs.append(fly_prob)
    return pred_lat, pred_lon, fly_probs


def rollout_baselines(
    path_df: pd.DataFrame,
    rollout_steps: int,
    context_days: int = ROLLOUT_CONTEXT_K,
) -> dict[str, tuple[list[float], list[float]]]:
    context = path_df.iloc[:context_days]
    persistence_lat = context["lat_median"].astype(float).tolist()
    persistence_lon = context["lon_median"].astype(float).tolist()
    constant_lat = persistence_lat.copy()
    constant_lon = persistence_lon.copy()
    last_delta_lat = float(context.iloc[-1]["delta_lat"])
    last_delta_lon = float(context.iloc[-1]["delta_lon"])
    for _ in range(rollout_steps):
        persistence_lat.append(persistence_lat[-1])
        persistence_lon.append(persistence_lon[-1])
        constant_lat.append(constant_lat[-1] + last_delta_lat)
        constant_lon.append(constant_lon[-1] + last_delta_lon)
    return {
        "persistence": (persistence_lat, persistence_lon),
        "const_velocity": (constant_lat, constant_lon),
    }


def path_total_displacement_km(path_df: pd.DataFrame) -> float:
    first = path_df.iloc[0]
    last = path_df.iloc[-1]
    return float(
        haversine_km(
            np.array([float(first["lat_median"])]),
            np.array([float(first["lon_median"])]),
            np.array([float(last["lat_median"])]),
            np.array([float(last["lon_median"])]),
        )[0]
    )


def select_rollout_path(
    featured: pd.DataFrame,
    window_data: WindowData,
    test_idx: np.ndarray,
    rollout_length: int,
    fly_threshold_km: float,
) -> tuple[str, pd.DataFrame, str]:
    test_start = pd.to_datetime(window_data.target_dates[test_idx].min(), utc=True)
    min_rows = ROLLOUT_CONTEXT_K + max(1, rollout_length)
    candidates: list[tuple[int, float, str, pd.DataFrame]] = []
    fallback: list[tuple[int, float, str, pd.DataFrame]] = []
    test_path_ids = set(str(path_id) for path_id in window_data.target_path_ids[test_idx])

    for path_id, path_df in featured.groupby(GROUP_COLUMN, sort=False):
        path_df = path_df.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(path_df["date"], utc=True, format="mixed").dt.normalize()
        if len(path_df) < ROLLOUT_CONTEXT_K + 1:
            continue
        max_consecutive_prefix = 1
        for diff in dates.diff().dt.days.fillna(1).iloc[1:]:
            if int(diff) != 1:
                break
            max_consecutive_prefix += 1
        valid_df = path_df.iloc[:max_consecutive_prefix].copy()
        if len(valid_df) < ROLLOUT_CONTEXT_K + 1:
            continue
        displacement = path_total_displacement_km(valid_df)
        stationary_rate = float((valid_df["step_length_km"].astype(float) <= fly_threshold_km).mean())
        score = (1 if displacement >= 50.0 else 0) + (1 if stationary_rate < 0.8 else 0)
        row = (score * 100000 + len(valid_df), displacement, str(path_id), valid_df)
        if len(valid_df) >= min_rows and pd.to_datetime(valid_df["date"], utc=True, format="mixed").min() >= test_start:
            candidates.append(row)
        if str(path_id) in test_path_ids:
            fallback.append(row)

    if candidates:
        _, _, path_id, path_df = max(candidates, key=lambda item: (item[0], item[1]))
        return path_id, path_df, "strict_test_path"
    if fallback:
        _, _, path_id, path_df = max(fallback, key=lambda item: (item[0], item[1]))
        return path_id, path_df, "fallback_path_with_test_windows"
    raise RuntimeError("No valid rollout path found for the chronological test split")


def best_summary_row(
    setup_dir: Path,
    family: str,
    kind: str | None = None,
) -> dict[str, object]:
    summary_path = setup_dir / f"{family}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run training before rollout visualization")
    summary = pd.read_csv(summary_path)
    summary = summary[pd.to_numeric(summary["mean_error_km"], errors="coerce").notna()].copy()
    summary = summary[summary["k"].astype(int) <= ROLLOUT_CONTEXT_K]
    if kind is not None:
        summary = summary[summary["kind"].astype(str) == kind].copy()
    if summary.empty:
        suffix = "" if kind is None else f" with kind={kind}"
        raise RuntimeError(f"No usable {family} rows{suffix} found in {summary_path}")
    row = summary.loc[summary["mean_error_km"].astype(float).idxmin()]
    return row.to_dict()


def model_checkpoint_path(setup_dir: Path, row: dict[str, object]) -> Path:
    return setup_dir / f"k_{int(row['k'])}" / str(row["model"]) / "best_model.pt"


def write_rollout_csv(
    output_csv: Path,
    path_df: pd.DataFrame,
    direct_lat: list[float],
    direct_lon: list[float],
    triline_lat: list[float],
    triline_lon: list[float],
    triline_fly_prob: list[float],
    baselines: dict[str, tuple[list[float], list[float]]],
) -> None:
    n_rows = len(direct_lat)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        row = path_df.iloc[i]
        payload: dict[str, object] = {
            "step_index": i,
            "date": str(pd.Timestamp(row["date"])),
            "phase": "context" if i < ROLLOUT_CONTEXT_K else "rollout",
            "true_lat": float(row["lat_median"]),
            "true_lon": float(row["lon_median"]),
            "direct_lat": direct_lat[i],
            "direct_lon": direct_lon[i],
            "triline_lat": triline_lat[i],
            "triline_lon": triline_lon[i],
            "triline_fly_probability": triline_fly_prob[i],
        }
        for name, (lat_values, lon_values) in baselines.items():
            payload[f"{name}_lat"] = lat_values[i]
            payload[f"{name}_lon"] = lon_values[i]
        rows.append(payload)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def svg_polyline(
    lon_values: list[float],
    lat_values: list[float],
    *,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    width: int,
    height: int,
    pad: int,
) -> str:
    def x(lon: float) -> float:
        return pad + (lon - min_lon) / max(max_lon - min_lon, 1e-9) * (width - 2 * pad)

    def y(lat: float) -> float:
        return height - pad - (lat - min_lat) / max(max_lat - min_lat, 1e-9) * (height - 2 * pad)

    return " ".join(f"{x(lon):.1f},{y(lat):.1f}" for lat, lon in zip(lat_values, lon_values))


def write_rollout_svg(
    output_svg: Path,
    path_id: str,
    path_df: pd.DataFrame,
    direct_lat: list[float],
    direct_lon: list[float],
    triline_lat: list[float],
    triline_lon: list[float],
    baselines: dict[str, tuple[list[float], list[float]]],
) -> None:
    n_rows = len(direct_lat)
    true_lat = path_df.iloc[:n_rows]["lat_median"].astype(float).tolist()
    true_lon = path_df.iloc[:n_rows]["lon_median"].astype(float).tolist()
    all_lats = true_lat + direct_lat + triline_lat
    all_lons = true_lon + direct_lon + triline_lon
    for lat_values, lon_values in baselines.values():
        all_lats.extend(lat_values)
        all_lons.extend(lon_values)
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    lat_pad = max((max_lat - min_lat) * 0.08, 0.05)
    lon_pad = max((max_lon - min_lon) * 0.08, 0.05)
    min_lat -= lat_pad
    max_lat += lat_pad
    min_lon -= lon_pad
    max_lon += lon_pad
    width, height, pad = 1000, 720, 64
    context_lat = true_lat[:ROLLOUT_CONTEXT_K]
    context_lon = true_lon[:ROLLOUT_CONTEXT_K]
    series = [
        ("Ground truth", true_lat, true_lon, "#111827", 3, ""),
        ("Observed context", context_lat, context_lon, "#16a34a", 5, ""),
        ("Best direct", direct_lat, direct_lon, "#2563eb", 3, "6 5"),
        ("Best triline", triline_lat, triline_lon, "#dc2626", 3, "3 5"),
        ("Persistence", baselines["persistence"][0], baselines["persistence"][1], "#6b7280", 2, "2 6"),
        ("Constant velocity", baselines["const_velocity"][0], baselines["const_velocity"][1], "#9333ea", 2, "8 6"),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Autoregressive rollout: {path_id}</text>',
        f'<text x="{pad}" y="58" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">First {ROLLOUT_CONTEXT_K} days are observed context; remaining days are model rollouts.</text>',
        f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]
    for label, lat_values, lon_values, color, stroke_width, dash in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = svg_polyline(
            lon_values,
            lat_values,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            width=width,
            height=height,
            pad=pad,
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
        )
    legend_x = width - 260
    legend_y = 92
    lines.append(f'<g font-family="Arial, sans-serif" font-size="13" fill="#111827">')
    for i, (label, _, _, color, stroke_width, dash) in enumerate(series):
        y = legend_y + i * 24
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 36}" y2="{y}" stroke="{color}" '
            f'stroke-width="{stroke_width}"{dash_attr}/>'
        )
        lines.append(f'<text x="{legend_x + 46}" y="{y + 4}">{label}</text>')
    lines.append("</g>")
    lines.append("</svg>")
    output_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rollout_visualizations(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    rollout_length: int,
    setup_names: list[str],
    device: torch.device,
) -> None:
    featured = add_southbound_compact_features(df)
    for setup_name in setup_names:
        fly_threshold_km = 30.0 if setup_name == "setup_30km" else 10.0
        setup_dir = output_dir / setup_name
        rollout_dir = setup_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        window_data = build_windows(df, k=ROLLOUT_CONTEXT_K, fly_threshold_km=fly_threshold_km)
        _, test_idx = make_chronological_split(window_data)
        path_id, path_df, selection_rule = select_rollout_path(
            featured,
            window_data,
            test_idx,
            rollout_length,
            fly_threshold_km,
        )
        actual_rollout_steps = min(int(rollout_length), len(path_df) - ROLLOUT_CONTEXT_K)
        path_df = path_df.iloc[: ROLLOUT_CONTEXT_K + actual_rollout_steps].reset_index(drop=True)

        direct_row = best_summary_row(setup_dir, "direct")
        triline_row = best_summary_row(setup_dir, "triline")
        direct_ckpt_path = model_checkpoint_path(setup_dir, direct_row)
        triline_ckpt_path = model_checkpoint_path(setup_dir, triline_row)
        direct_model, direct_checkpoint = load_checkpoint_model(direct_ckpt_path, "direct", device)
        triline_model, triline_checkpoint = load_checkpoint_model(triline_ckpt_path, "triline", device)

        direct_lat, direct_lon = rollout_direct_model(
            direct_model,
            direct_checkpoint,
            path_df,
            actual_rollout_steps,
            device,
        )
        triline_lat, triline_lon, triline_fly_prob = rollout_triline_model(
            triline_model,
            triline_checkpoint,
            path_df,
            actual_rollout_steps,
            device,
        )
        baselines = rollout_baselines(path_df, actual_rollout_steps)
        write_rollout_csv(
            rollout_dir / "rollout_predictions.csv",
            path_df,
            direct_lat,
            direct_lon,
            triline_lat,
            triline_lon,
            triline_fly_prob,
            baselines,
        )
        write_rollout_svg(
            rollout_dir / "rollout_plot.svg",
            path_id,
            path_df,
            direct_lat,
            direct_lon,
            triline_lat,
            triline_lon,
            baselines,
        )
        true_lat = path_df["lat_median"].astype(float).to_numpy()
        true_lon = path_df["lon_median"].astype(float).to_numpy()
        future = np.arange(ROLLOUT_CONTEXT_K, len(path_df))
        direct_error = haversine_km(
            true_lat[future],
            true_lon[future],
            np.asarray(direct_lat, dtype=float)[future],
            np.asarray(direct_lon, dtype=float)[future],
        )
        triline_error = haversine_km(
            true_lat[future],
            true_lon[future],
            np.asarray(triline_lat, dtype=float)[future],
            np.asarray(triline_lon, dtype=float)[future],
        )
        write_json(
            rollout_dir / "rollout_summary.json",
            {
                "setup": setup_name,
                "path_id": path_id,
                "selection_rule": selection_rule,
                "context_days": ROLLOUT_CONTEXT_K,
                "rollout_steps": actual_rollout_steps,
                "direct_model": direct_row["model"],
                "direct_k": int(direct_row["k"]),
                "triline_model": triline_row["model"],
                "triline_k": int(triline_row["k"]),
                "direct_mean_rollout_error_km": float(direct_error.mean()) if len(direct_error) else None,
                "direct_final_rollout_error_km": float(direct_error[-1]) if len(direct_error) else None,
                "triline_mean_rollout_error_km": float(triline_error.mean()) if len(triline_error) else None,
                "triline_final_rollout_error_km": float(triline_error[-1]) if len(triline_error) else None,
                "csv": str(rollout_dir / "rollout_predictions.csv"),
                "svg": str(rollout_dir / "rollout_plot.svg"),
            },
        )
        release_torch_objects(direct_model, triline_model)
        logging.info("%s rollout visualization saved to %s", setup_name, rollout_dir)


def observed_context_days_for_path(
    path_df: pd.DataFrame,
    min_context_days: int = ROLLOUT_CONTEXT_K,
    min_displacement_km: float = ROLLOUT_MIN_DISPLACEMENT_KM,
) -> tuple[int, float]:
    if len(path_df) <= min_context_days:
        return len(path_df), 0.0
    start = path_df.iloc[0]
    context_days = min_context_days
    displacement = 0.0
    while context_days < len(path_df):
        current = path_df.iloc[context_days - 1]
        displacement = float(
            haversine_km(
                np.array([float(start["lat_median"])]),
                np.array([float(start["lon_median"])]),
                np.array([float(current["lat_median"])]),
                np.array([float(current["lon_median"])]),
            )[0]
        )
        if displacement >= min_displacement_km:
            break
        context_days += 1
    return context_days, displacement


def rollout_error_summary(
    path_df: pd.DataFrame,
    lat_values: list[float],
    lon_values: list[float],
    context_days: int,
) -> tuple[float | None, float | None]:
    if len(path_df) <= context_days:
        return None, None
    future = np.arange(context_days, len(path_df))
    true_lat = path_df["lat_median"].astype(float).to_numpy()
    true_lon = path_df["lon_median"].astype(float).to_numpy()
    errors = haversine_km(
        true_lat[future],
        true_lon[future],
        np.asarray(lat_values, dtype=float)[future],
        np.asarray(lon_values, dtype=float)[future],
    )
    return float(errors.mean()), float(errors[-1])


def build_rollout_rows_for_path(
    *,
    setup_name: str,
    path_id: str,
    path_df: pd.DataFrame,
    context_days: int,
    direct_lat: list[float],
    direct_lon: list[float],
    lstm_lat: list[float],
    lstm_lon: list[float],
    lstm_fly_prob: list[float],
    transformer_lat: list[float],
    transformer_lon: list[float],
    transformer_fly_prob: list[float],
    baselines: dict[str, tuple[list[float], list[float]]],
    direct_model_name: str,
    lstm_model_name: str,
    transformer_model_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    true_lat = path_df["lat_median"].astype(float).to_numpy()
    true_lon = path_df["lon_median"].astype(float).to_numpy()
    direct_errors = haversine_km(true_lat, true_lon, np.asarray(direct_lat), np.asarray(direct_lon))
    lstm_errors = haversine_km(true_lat, true_lon, np.asarray(lstm_lat), np.asarray(lstm_lon))
    transformer_errors = haversine_km(
        true_lat,
        true_lon,
        np.asarray(transformer_lat),
        np.asarray(transformer_lon),
    )
    persistence_lat, persistence_lon = baselines["persistence"]
    velocity_lat, velocity_lon = baselines["const_velocity"]
    persistence_errors = haversine_km(true_lat, true_lon, np.asarray(persistence_lat), np.asarray(persistence_lon))
    velocity_errors = haversine_km(true_lat, true_lon, np.asarray(velocity_lat), np.asarray(velocity_lon))
    first = path_df.iloc[0]
    for i, row in path_df.reset_index(drop=True).iterrows():
        is_context = i < context_days
        rows.append(
            {
                "setup": setup_name,
                "path_id": path_id,
                "source_bird_id": str(row.get(IDENTITY_COLUMN, first.get(IDENTITY_COLUMN, ""))),
                "path_year": int(row.get("path_year", first.get("path_year", 0)) or 0),
                "path_copy_index": int(row.get("path_copy_index", first.get("path_copy_index", 0)) or 0),
                "step_index": int(i),
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "phase": "observed" if is_context else "rollout",
                "observed_days": int(context_days),
                "rollout_step": "" if is_context else int(i - context_days + 1),
                "true_lat": float(row["lat_median"]),
                "true_lon": float(row["lon_median"]),
                "target_step_km": float(row.get("step_length_km", 0.0)),
                "direct_model": direct_model_name,
                "direct_lat": float(direct_lat[i]),
                "direct_lon": float(direct_lon[i]),
                "direct_error_km": 0.0 if is_context else float(direct_errors[i]),
                "lstm_model": lstm_model_name,
                "lstm_lat": float(lstm_lat[i]),
                "lstm_lon": float(lstm_lon[i]),
                "lstm_error_km": 0.0 if is_context else float(lstm_errors[i]),
                "lstm_fly_probability": "" if not np.isfinite(lstm_fly_prob[i]) else float(lstm_fly_prob[i]),
                "transformer_model": transformer_model_name,
                "transformer_lat": float(transformer_lat[i]),
                "transformer_lon": float(transformer_lon[i]),
                "transformer_error_km": 0.0 if is_context else float(transformer_errors[i]),
                "transformer_fly_probability": (
                    "" if not np.isfinite(transformer_fly_prob[i]) else float(transformer_fly_prob[i])
                ),
                "persistence_lat": float(persistence_lat[i]),
                "persistence_lon": float(persistence_lon[i]),
                "persistence_error_km": 0.0 if is_context else float(persistence_errors[i]),
                "const_velocity_lat": float(velocity_lat[i]),
                "const_velocity_lon": float(velocity_lon[i]),
                "const_velocity_error_km": 0.0 if is_context else float(velocity_errors[i]),
            }
        )
    return rows


def write_rollout_explorer(output_dir: Path, rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "rollout_all_predictions.csv", rows)
    write_csv(output_dir / "rollout_all_summary.csv", summaries)
    data_payload = {
        "latDomain": [ROLLOUT_LAT_MIN, ROLLOUT_LAT_MAX],
        "rows": rows,
        "summaries": summaries,
    }
    (output_dir / "rollout_data.js").write_text(
        "window.ROLLOUT_DATA = "
        + json.dumps(data_payload, default=json_safe, allow_nan=False)
        + ";\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(ROLLOUT_EXPLORER_HTML, encoding="utf-8")
    (output_dir / "rollout_styles.css").write_text(ROLLOUT_EXPLORER_CSS, encoding="utf-8")
    (output_dir / "rollout_app.js").write_text(ROLLOUT_EXPLORER_JS, encoding="utf-8")


def run_all_test_rollouts(
    *,
    df: pd.DataFrame,
    output_dir: Path,
    setup_names: list[str],
    device: torch.device,
) -> None:
    featured = add_southbound_compact_features(df)
    app_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    all_output_dir = output_dir / "rollout_all"
    all_output_dir.mkdir(parents=True, exist_ok=True)

    for setup_name in setup_names:
        fly_threshold_km = 30.0 if setup_name == "setup_30km" else 10.0
        setup_dir = output_dir / setup_name
        window_data = build_windows(df, k=ROLLOUT_CONTEXT_K, fly_threshold_km=fly_threshold_km)
        _, test_idx = make_chronological_split(window_data)
        test_path_ids = sorted(set(str(path_id) for path_id in window_data.target_path_ids[test_idx]))

        direct_row = best_summary_row(setup_dir, "direct")
        lstm_row = best_summary_row(setup_dir, "triline", kind="triline_lstm")
        transformer_row = best_summary_row(setup_dir, "triline", kind="triline_transformer")
        direct_model, direct_checkpoint = load_checkpoint_model(
            model_checkpoint_path(setup_dir, direct_row), "direct", device
        )
        lstm_model, lstm_checkpoint = load_checkpoint_model(
            model_checkpoint_path(setup_dir, lstm_row), "triline", device
        )
        transformer_model, transformer_checkpoint = load_checkpoint_model(
            model_checkpoint_path(setup_dir, transformer_row), "triline", device
        )
        logging.info("%s rolling out %s test paths", setup_name, len(test_path_ids))

        for path_id in test_path_ids:
            path_df = (
                featured[featured[GROUP_COLUMN].astype(str) == path_id]
                .sort_values("date")
                .reset_index(drop=True)
            )
            if len(path_df) <= ROLLOUT_CONTEXT_K:
                skipped_rows.append(
                    {
                        "setup": setup_name,
                        "path_id": path_id,
                        "reason": "not_enough_rows_for_30_observed_days",
                        "rows": int(len(path_df)),
                    }
                )
                continue

            context_days, observed_displacement_km = observed_context_days_for_path(path_df)
            if context_days >= len(path_df):
                skipped_rows.append(
                    {
                        "setup": setup_name,
                        "path_id": path_id,
                        "reason": "no_rollout_days_after_50km_observed_context",
                        "rows": int(len(path_df)),
                        "observed_days": int(context_days),
                        "observed_displacement_km": float(observed_displacement_km),
                    }
                )
                continue

            rollout_steps = len(path_df) - context_days
            direct_lat, direct_lon = rollout_direct_model(
                direct_model,
                direct_checkpoint,
                path_df,
                rollout_steps,
                device,
                context_days=context_days,
            )
            lstm_lat, lstm_lon, lstm_fly_prob = rollout_triline_model(
                lstm_model,
                lstm_checkpoint,
                path_df,
                rollout_steps,
                device,
                context_days=context_days,
            )
            transformer_lat, transformer_lon, transformer_fly_prob = rollout_triline_model(
                transformer_model,
                transformer_checkpoint,
                path_df,
                rollout_steps,
                device,
                context_days=context_days,
            )
            baselines = rollout_baselines(path_df, rollout_steps, context_days=context_days)
            app_rows.extend(
                build_rollout_rows_for_path(
                    setup_name=setup_name,
                    path_id=path_id,
                    path_df=path_df,
                    context_days=context_days,
                    direct_lat=direct_lat,
                    direct_lon=direct_lon,
                    lstm_lat=lstm_lat,
                    lstm_lon=lstm_lon,
                    lstm_fly_prob=lstm_fly_prob,
                    transformer_lat=transformer_lat,
                    transformer_lon=transformer_lon,
                    transformer_fly_prob=transformer_fly_prob,
                    baselines=baselines,
                    direct_model_name=str(direct_row["model"]),
                    lstm_model_name=str(lstm_row["model"]),
                    transformer_model_name=str(transformer_row["model"]),
                )
            )
            direct_mean, direct_final = rollout_error_summary(path_df, direct_lat, direct_lon, context_days)
            lstm_mean, lstm_final = rollout_error_summary(path_df, lstm_lat, lstm_lon, context_days)
            transformer_mean, transformer_final = rollout_error_summary(
                path_df,
                transformer_lat,
                transformer_lon,
                context_days,
            )
            persistence_mean, persistence_final = rollout_error_summary(
                path_df, baselines["persistence"][0], baselines["persistence"][1], context_days
            )
            velocity_mean, velocity_final = rollout_error_summary(
                path_df, baselines["const_velocity"][0], baselines["const_velocity"][1], context_days
            )
            first = path_df.iloc[0]
            summary_rows.append(
                {
                    "setup": setup_name,
                    "path_id": path_id,
                    "source_bird_id": str(first.get(IDENTITY_COLUMN, "")),
                    "path_year": int(first.get("path_year", 0) or 0),
                    "path_copy_index": int(first.get("path_copy_index", 0) or 0),
                    "records": int(len(path_df)),
                    "observed_days": int(context_days),
                    "rollout_steps": int(rollout_steps),
                    "observed_displacement_km": float(observed_displacement_km),
                    "direct_model": str(direct_row["model"]),
                    "direct_mean_rollout_error_km": direct_mean,
                    "direct_final_rollout_error_km": direct_final,
                    "lstm_model": str(lstm_row["model"]),
                    "lstm_mean_rollout_error_km": lstm_mean,
                    "lstm_final_rollout_error_km": lstm_final,
                    "transformer_model": str(transformer_row["model"]),
                    "transformer_mean_rollout_error_km": transformer_mean,
                    "transformer_final_rollout_error_km": transformer_final,
                    "persistence_mean_rollout_error_km": persistence_mean,
                    "persistence_final_rollout_error_km": persistence_final,
                    "const_velocity_mean_rollout_error_km": velocity_mean,
                    "const_velocity_final_rollout_error_km": velocity_final,
                    "lat_min": float(path_df["lat_median"].min()),
                    "lat_max": float(path_df["lat_median"].max()),
                    "lon_min": float(path_df["lon_median"].min()),
                    "lon_max": float(path_df["lon_median"].max()),
                }
            )

        release_torch_objects(direct_model, lstm_model, transformer_model)

    write_rollout_explorer(all_output_dir, app_rows, summary_rows)
    write_csv(all_output_dir / "rollout_all_skipped.csv", skipped_rows)
    write_json(
        all_output_dir / "rollout_all_index.json",
        {
            "rows": int(len(app_rows)),
            "paths": int(len(summary_rows)),
            "skipped_paths": int(len(skipped_rows)),
            "lat_domain": [ROLLOUT_LAT_MIN, ROLLOUT_LAT_MAX],
            "min_observed_days": ROLLOUT_CONTEXT_K,
            "min_observed_displacement_km": ROLLOUT_MIN_DISPLACEMENT_KM,
            "html": str(all_output_dir / "index.html"),
            "predictions_csv": str(all_output_dir / "rollout_all_predictions.csv"),
            "summary_csv": str(all_output_dir / "rollout_all_summary.csv"),
        },
    )
    logging.info("All-path rollout explorer saved to %s", all_output_dir)


ROLLOUT_EXPLORER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Weather Rollout Explorer</title>
    <link rel="stylesheet" href="rollout_styles.css" />
  </head>
  <body>
    <main class="app-shell">
      <aside class="sidebar">
        <div class="title-block">
          <p class="eyebrow">Path Experiment With Weather</p>
          <h1>Rollout Explorer</h1>
        </div>

        <label class="field">
          <span>Setup</span>
          <select id="setupSelect"></select>
        </label>

        <label class="field">
          <span>Test path</span>
          <select id="pathSelect"></select>
        </label>

        <div class="toggle-grid" aria-label="Visible series">
          <label><input id="showObserved" type="checkbox" checked /> Observed</label>
          <label><input id="showTruth" type="checkbox" checked /> Truth</label>
          <label><input id="showDirect" type="checkbox" checked /> Direct</label>
          <label><input id="showLstm" type="checkbox" checked /> LSTM</label>
          <label><input id="showTransformer" type="checkbox" checked /> Transformer</label>
          <label><input id="showPredictedPoints" type="checkbox" /> Predicted points</label>
          <label><input id="showPersistence" type="checkbox" /> Persistence</label>
          <label><input id="showVelocity" type="checkbox" /> Velocity</label>
        </div>

        <div class="stats-grid" aria-label="Rollout summary">
          <section class="stat">
            <span>Observed</span>
            <strong id="observedDays">--</strong>
          </section>
          <section class="stat">
            <span>Rollout</span>
            <strong id="rolloutDays">--</strong>
          </section>
          <section class="stat">
            <span>Direct mean</span>
            <strong id="directMean">--</strong>
          </section>
          <section class="stat">
            <span>LSTM mean</span>
            <strong id="lstmMean">--</strong>
          </section>
          <section class="stat">
            <span>Direct final</span>
            <strong id="directFinal">--</strong>
          </section>
          <section class="stat">
            <span>LSTM final</span>
            <strong id="lstmFinal">--</strong>
          </section>
          <section class="stat">
            <span>Transformer mean</span>
            <strong id="transformerMean">--</strong>
          </section>
          <section class="stat">
            <span>Transformer final</span>
            <strong id="transformerFinal">--</strong>
          </section>
        </div>

        <section class="details-panel">
          <h2 id="selectedPathName">Select a path</h2>
          <dl>
            <div>
              <dt>Source bird</dt>
              <dd id="sourceBird">--</dd>
            </div>
            <div>
              <dt>Date range</dt>
              <dd id="dateRange">--</dd>
            </div>
            <div>
              <dt>Observed displacement</dt>
              <dd id="observedDistance">--</dd>
            </div>
            <div>
              <dt>Models</dt>
              <dd id="modelNames">--</dd>
            </div>
          </dl>
        </section>
      </aside>

      <section class="map-panel">
        <header class="map-toolbar">
          <div>
            <p class="eyebrow">GPS grid: latitude 55 to 30</p>
            <h2 id="mapTitle">Autoregressive rollout</h2>
          </div>
          <div class="toolbar-actions">
            <button id="fitPathButton" type="button" class="icon-button" title="Fit selected path longitude">Fit Path</button>
            <button id="fitAllButton" type="button" class="icon-button" title="Fit all rollout longitudes">Fit All</button>
          </div>
        </header>

        <div class="plot-wrap">
          <svg id="rolloutPlot" role="img" aria-label="Autoregressive rollout on a GPS grid"></svg>
          <div id="tooltip" class="tooltip" hidden></div>
        </div>

        <footer class="legend-row" aria-label="Color legend">
          <span class="legend-swatch observed"></span><span>Observed context</span>
          <span class="legend-swatch truth"></span><span>Ground truth</span>
          <span class="legend-swatch direct"></span><span>Direct</span>
          <span class="legend-swatch lstm"></span><span>LSTM</span>
          <span class="legend-swatch transformer"></span><span>Transformer</span>
        </footer>
      </section>
    </main>
    <script src="rollout_data.js"></script>
    <script src="rollout_app.js"></script>
  </body>
</html>
"""


ROLLOUT_EXPLORER_CSS = """:root {
  color-scheme: light;
  --bg: #f4f6f3;
  --panel: #ffffff;
  --panel-2: #eef2ef;
  --ink: #16211f;
  --muted: #63716d;
  --line: #d7ded9;
  --accent: #0f766e;
  --accent-2: #c2410c;
  --direct: #2563eb;
  --lstm: #dc2626;
  --transformer: #7c3aed;
  --shadow: 0 18px 44px rgba(22, 33, 31, 0.12);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button, input, select { font: inherit; }

.app-shell {
  display: grid;
  grid-template-columns: minmax(300px, 390px) minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 18px;
  border-right: 1px solid var(--line);
  background: var(--panel);
  padding: 24px;
  overflow-y: auto;
}

.title-block { display: grid; gap: 6px; }

.eyebrow {
  margin: 0;
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1, h2 { margin: 0; letter-spacing: 0; }
h1 { max-width: 10ch; font-size: clamp(2rem, 4vw, 3.15rem); line-height: 0.95; }
h2 { font-size: 1.05rem; line-height: 1.25; }

.field {
  display: grid;
  gap: 8px;
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 700;
}

select {
  width: 100%;
  min-height: 42px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  padding: 9px 11px;
}

select:focus, button:focus-visible {
  outline: 3px solid rgba(15, 118, 110, 0.22);
  outline-offset: 2px;
}

.toggle-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 12px;
  color: var(--ink);
  font-size: 0.86rem;
  font-weight: 800;
}

.toggle-grid label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.stat, .details-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-2);
}

.stat {
  display: grid;
  min-height: 82px;
  align-content: space-between;
  padding: 12px;
}

.stat span, dt {
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

.stat strong { font-size: 1.1rem; line-height: 1.12; }

.details-panel {
  display: grid;
  gap: 14px;
  padding: 16px;
}

.details-panel dl { display: grid; gap: 12px; margin: 0; }
.details-panel div { display: grid; gap: 3px; }
dd { margin: 0; color: var(--ink); font-weight: 700; overflow-wrap: anywhere; }

.map-panel {
  display: grid;
  grid-template-rows: auto minmax(420px, 1fr) auto;
  gap: 14px;
  padding: 24px;
  min-width: 0;
}

.map-toolbar, .legend-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.toolbar-actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }

.icon-button {
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  cursor: pointer;
  font-weight: 800;
  padding: 8px 12px;
}

.icon-button:hover { border-color: rgba(15, 118, 110, 0.55); color: var(--accent); }

.plot-wrap {
  position: relative;
  min-height: 420px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfa;
  box-shadow: var(--shadow);
  overflow: hidden;
}

#rolloutPlot { display: block; width: 100%; height: 100%; min-height: 420px; }

.axis-label, .tick-label { fill: var(--muted); font-size: 12px; user-select: none; }
.grid-line { stroke: #dce4df; stroke-width: 1; }
.axis-line { stroke: #96a39e; stroke-width: 1.4; }

.series-line {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.truth-line { stroke: rgba(22, 33, 31, 0.68); stroke-width: 2.5; }
.observed-line { stroke: var(--accent); stroke-width: 5; }
.direct-line { stroke: var(--direct); stroke-width: 3; stroke-dasharray: 8 6; }
.lstm-line { stroke: var(--lstm); stroke-width: 3; stroke-dasharray: 3 5; }
.transformer-line { stroke: var(--transformer); stroke-width: 3; stroke-dasharray: 9 5 2 5; }
.persistence-line { stroke: #6b7280; stroke-width: 2.2; stroke-dasharray: 2 6; }
.velocity-line { stroke: #9333ea; stroke-width: 2.2; stroke-dasharray: 8 6; }

.path-dot {
  cursor: pointer;
  stroke: #ffffff;
  stroke-width: 1.4;
}

.path-dot:hover, .path-dot.is-active { stroke: var(--ink); stroke-width: 2.2; }
.context-dot { fill: var(--accent); }
.rollout-dot { fill: var(--accent-2); }
.predicted-dot { fill: #fff; stroke-width: 2; }
.direct-dot { stroke: var(--direct); }
.lstm-dot { stroke: var(--lstm); }
.transformer-dot { stroke: var(--transformer); }
.start-marker { fill: var(--accent); }
.end-marker { fill: var(--accent-2); }

.marker-label {
  fill: var(--ink);
  font-size: 12px;
  font-weight: 800;
  paint-order: stroke;
  stroke: #fff;
  stroke-width: 4px;
}

.tooltip {
  position: absolute;
  z-index: 10;
  max-width: min(320px, calc(100% - 20px));
  border: 1px solid rgba(22, 33, 31, 0.13);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 14px 32px rgba(22, 33, 31, 0.16);
  color: var(--ink);
  font-size: 0.84rem;
  line-height: 1.45;
  padding: 10px 12px;
  pointer-events: none;
}

.tooltip strong { display: block; margin-bottom: 4px; }

.legend-row {
  justify-content: flex-end;
  color: var(--muted);
  font-size: 0.83rem;
  font-weight: 800;
}

.legend-swatch {
  display: inline-block;
  width: 28px;
  height: 4px;
  border-radius: 999px;
}

.legend-swatch.observed { background: var(--accent); }
.legend-swatch.truth { background: var(--ink); opacity: 0.7; }
.legend-swatch.direct { background: var(--direct); }
.legend-swatch.lstm { background: var(--lstm); }
.legend-swatch.transformer { background: var(--transformer); }

@media (max-width: 860px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
  h1 { max-width: none; font-size: 2rem; line-height: 1; }
  .map-panel { padding: 16px; grid-template-rows: auto minmax(360px, 64vh) auto; }
  .map-toolbar { align-items: flex-start; flex-direction: column; }
  .toolbar-actions { justify-content: flex-start; }
  #rolloutPlot, .plot-wrap { min-height: 360px; }
}

@media (max-width: 520px) {
  .sidebar { padding: 18px; }
  .stats-grid { grid-template-columns: 1fr; }
}
"""


ROLLOUT_EXPLORER_JS = """const DATA = window.ROLLOUT_DATA || { rows: [], summaries: [], latDomain: [30, 55] };
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  setup: "",
  pathId: "",
  viewMode: "path",
  activeDot: null,
};

const els = {
  setupSelect: document.getElementById("setupSelect"),
  pathSelect: document.getElementById("pathSelect"),
  showObserved: document.getElementById("showObserved"),
  showTruth: document.getElementById("showTruth"),
  showDirect: document.getElementById("showDirect"),
  showLstm: document.getElementById("showLstm"),
  showTransformer: document.getElementById("showTransformer"),
  showPredictedPoints: document.getElementById("showPredictedPoints"),
  showPersistence: document.getElementById("showPersistence"),
  showVelocity: document.getElementById("showVelocity"),
  plot: document.getElementById("rolloutPlot"),
  tooltip: document.getElementById("tooltip"),
  selectedPathName: document.getElementById("selectedPathName"),
  mapTitle: document.getElementById("mapTitle"),
  observedDays: document.getElementById("observedDays"),
  rolloutDays: document.getElementById("rolloutDays"),
  directMean: document.getElementById("directMean"),
  lstmMean: document.getElementById("lstmMean"),
  transformerMean: document.getElementById("transformerMean"),
  directFinal: document.getElementById("directFinal"),
  lstmFinal: document.getElementById("lstmFinal"),
  transformerFinal: document.getElementById("transformerFinal"),
  sourceBird: document.getElementById("sourceBird"),
  dateRange: document.getElementById("dateRange"),
  observedDistance: document.getElementById("observedDistance"),
  modelNames: document.getElementById("modelNames"),
  fitPathButton: document.getElementById("fitPathButton"),
  fitAllButton: document.getElementById("fitAllButton"),
};

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const precise = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

const summariesByKey = new Map(DATA.summaries.map((row) => [`${row.setup}::${row.path_id}`, row]));

function rowsForSelection() {
  return DATA.rows.filter((row) => row.setup === state.setup && row.path_id === state.pathId);
}

function rowsForDomain(pathRows) {
  if (state.viewMode === "all") {
    return DATA.rows.filter((row) => row.setup === state.setup);
  }
  return pathRows;
}

function extent(rows, keys) {
  let min = Infinity;
  let max = -Infinity;
  rows.forEach((row) => {
    keys.forEach((key) => {
      const value = Number(row[key]);
      if (Number.isFinite(value)) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    });
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (min === max) {
    return [min - 0.5, max + 0.5];
  }
  return [min, max];
}

function paddedDomain([min, max], ratio = 0.08) {
  const span = max - min || 1;
  return [min - span * ratio, max + span * ratio];
}

function buildTicks([min, max], count = 6) {
  const ticks = [];
  const step = (max - min) / Math.max(count - 1, 1);
  for (let i = 0; i < count; i += 1) {
    ticks.push(min + step * i);
  }
  return ticks;
}

function formatKm(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${fmt.format(n)} km` : "--";
}

function formatDegree(value, axis) {
  const direction = axis === "lat"
    ? value >= 0 ? "N" : "S"
    : value >= 0 ? "E" : "W";
  return `${Math.abs(value).toFixed(2)} deg ${direction}`;
}

function createSvgElement(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  return el;
}

function pathData(rows, latKey, lonKey, x, y) {
  return rows
    .map((row, index) => `${index === 0 ? "M" : "L"} ${x(Number(row[lonKey])).toFixed(2)} ${y(Number(row[latKey])).toFixed(2)}`)
    .join(" ");
}

function drawLine(svg, rows, latKey, lonKey, className, x, y) {
  const valid = rows.filter((row) => Number.isFinite(Number(row[latKey])) && Number.isFinite(Number(row[lonKey])));
  if (valid.length < 2) return;
  svg.appendChild(createSvgElement("path", {
    class: `series-line ${className}`,
    d: pathData(valid, latKey, lonKey, x, y),
  }));
}

function drawPredictedDots(svg, rows, latKey, lonKey, className, label, errorKey, x, y) {
  const group = createSvgElement("g");
  rows
    .filter((row) => row.phase === "rollout")
    .forEach((row) => {
      const lat = Number(row[latKey]);
      const lon = Number(row[lonKey]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const dot = createSvgElement("circle", {
        class: `path-dot predicted-dot ${className}`,
        cx: x(lon),
        cy: y(lat),
        r: 4.2,
        tabindex: "0",
      });
      dot.addEventListener("mouseenter", (event) => showPredictionTooltip(event, row, label, lat, lon, errorKey));
      dot.addEventListener("mousemove", positionTooltip);
      dot.addEventListener("mouseleave", hideTooltip);
      dot.addEventListener("focus", (event) => showPredictionTooltip(event, row, label, lat, lon, errorKey));
      dot.addEventListener("blur", hideTooltip);
      group.appendChild(dot);
    });
  svg.appendChild(group);
}

function addEndpoint(svg, cx, cy, label, className) {
  svg.appendChild(createSvgElement("circle", { class: className, cx, cy, r: 6.8 }));
  const text = createSvgElement("text", { class: "marker-label", x: cx + 10, y: cy - 10 });
  text.textContent = label;
  svg.appendChild(text);
}

function drawPlot(rows) {
  const svg = els.plot;
  svg.replaceChildren();
  const rect = svg.getBoundingClientRect();
  const width = Math.max(rect.width, 720);
  const height = Math.max(rect.height, 420);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!rows.length) return;

  const margin = { top: 28, right: 34, bottom: 48, left: 70 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const domainRows = rowsForDomain(rows);
  const lonDomain = paddedDomain(extent(domainRows, [
    "true_lon", "direct_lon", "lstm_lon", "transformer_lon", "persistence_lon", "const_velocity_lon",
  ]));
  const latDomain = [Number(DATA.latDomain[0]), Number(DATA.latDomain[1])];
  const x = (lon) => margin.left + ((lon - lonDomain[0]) / (lonDomain[1] - lonDomain[0])) * plotWidth;
  const y = (lat) => margin.top + (1 - ((lat - latDomain[0]) / (latDomain[1] - latDomain[0]))) * plotHeight;

  const grid = createSvgElement("g");
  buildTicks(lonDomain, 7).forEach((tick) => {
    const tx = x(tick);
    grid.appendChild(createSvgElement("line", {
      class: "grid-line", x1: tx, y1: margin.top, x2: tx, y2: margin.top + plotHeight,
    }));
    const label = createSvgElement("text", {
      class: "tick-label", x: tx, y: height - 22, "text-anchor": "middle",
    });
    label.textContent = formatDegree(tick, "lon");
    grid.appendChild(label);
  });
  [55, 50, 45, 40, 35, 30].forEach((tick) => {
    const ty = y(tick);
    grid.appendChild(createSvgElement("line", {
      class: "grid-line", x1: margin.left, y1: ty, x2: margin.left + plotWidth, y2: ty,
    }));
    const label = createSvgElement("text", { class: "tick-label", x: 14, y: ty + 4 });
    label.textContent = formatDegree(tick, "lat");
    grid.appendChild(label);
  });
  grid.appendChild(createSvgElement("line", {
    class: "axis-line", x1: margin.left, y1: margin.top + plotHeight, x2: margin.left + plotWidth, y2: margin.top + plotHeight,
  }));
  grid.appendChild(createSvgElement("line", {
    class: "axis-line", x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotHeight,
  }));
  svg.appendChild(grid);

  if (els.showTruth.checked) drawLine(svg, rows, "true_lat", "true_lon", "truth-line", x, y);
  if (els.showObserved.checked) drawLine(svg, rows.filter((row) => row.phase === "observed"), "true_lat", "true_lon", "observed-line", x, y);
  if (els.showDirect.checked) drawLine(svg, rows, "direct_lat", "direct_lon", "direct-line", x, y);
  if (els.showLstm.checked) drawLine(svg, rows, "lstm_lat", "lstm_lon", "lstm-line", x, y);
  if (els.showTransformer.checked) drawLine(svg, rows, "transformer_lat", "transformer_lon", "transformer-line", x, y);
  if (els.showPersistence.checked) drawLine(svg, rows, "persistence_lat", "persistence_lon", "persistence-line", x, y);
  if (els.showVelocity.checked) drawLine(svg, rows, "const_velocity_lat", "const_velocity_lon", "velocity-line", x, y);
  if (els.showPredictedPoints.checked) {
    if (els.showDirect.checked) drawPredictedDots(svg, rows, "direct_lat", "direct_lon", "direct-dot", "Direct", "direct_error_km", x, y);
    if (els.showLstm.checked) drawPredictedDots(svg, rows, "lstm_lat", "lstm_lon", "lstm-dot", "LSTM", "lstm_error_km", x, y);
    if (els.showTransformer.checked) drawPredictedDots(svg, rows, "transformer_lat", "transformer_lon", "transformer-dot", "Transformer", "transformer_error_km", x, y);
  }

  const dots = createSvgElement("g");
  rows.forEach((row, index) => {
    const dot = createSvgElement("circle", {
      class: `path-dot ${row.phase === "observed" ? "context-dot" : "rollout-dot"}`,
      cx: x(Number(row.true_lon)),
      cy: y(Number(row.true_lat)),
      r: row.phase === "observed" ? 4.4 : 5.4,
      tabindex: "0",
      "data-index": index,
    });
    dot.addEventListener("mouseenter", (event) => showTooltip(event, row, index, rows.length));
    dot.addEventListener("mousemove", positionTooltip);
    dot.addEventListener("mouseleave", hideTooltip);
    dot.addEventListener("focus", (event) => showTooltip(event, row, index, rows.length));
    dot.addEventListener("blur", hideTooltip);
    dots.appendChild(dot);
  });
  svg.appendChild(dots);

  const first = rows[0];
  const lastObserved = [...rows].reverse().find((row) => row.phase === "observed") || first;
  const last = rows[rows.length - 1];
  addEndpoint(svg, x(Number(first.true_lon)), y(Number(first.true_lat)), "Start", "start-marker");
  addEndpoint(svg, x(Number(lastObserved.true_lon)), y(Number(lastObserved.true_lat)), "Observed", "start-marker");
  addEndpoint(svg, x(Number(last.true_lon)), y(Number(last.true_lat)), "End", "end-marker");
}

function showTooltip(event, row, index, total) {
  if (state.activeDot) state.activeDot.classList.remove("is-active");
  state.activeDot = event.currentTarget;
  state.activeDot.classList.add("is-active");
  els.tooltip.innerHTML = `
    <strong>${row.date} (${index + 1} / ${total})</strong>
    ${row.phase} day ${row.step_index}<br>
    True ${Number(row.true_lat).toFixed(5)}, ${Number(row.true_lon).toFixed(5)}<br>
    Step ${formatKm(row.target_step_km)}<br>
    Direct error ${formatKm(row.direct_error_km)}<br>
    LSTM error ${formatKm(row.lstm_error_km)}<br>
    Transformer error ${formatKm(row.transformer_error_km)}
  `;
  els.tooltip.hidden = false;
  positionTooltip(event);
}

function showPredictionTooltip(event, row, label, lat, lon, errorKey) {
  if (state.activeDot) state.activeDot.classList.remove("is-active");
  state.activeDot = event.currentTarget;
  state.activeDot.classList.add("is-active");
  els.tooltip.innerHTML = `
    <strong>${label} prediction: ${row.date}</strong>
    Rollout day ${row.rollout_step}<br>
    Pred ${lat.toFixed(5)}, ${lon.toFixed(5)}<br>
    True ${Number(row.true_lat).toFixed(5)}, ${Number(row.true_lon).toFixed(5)}<br>
    Error ${formatKm(row[errorKey])}
  `;
  els.tooltip.hidden = false;
  positionTooltip(event);
}

function positionTooltip(event) {
  const wrap = els.plot.parentElement.getBoundingClientRect();
  const tooltip = els.tooltip;
  const tipRect = tooltip.getBoundingClientRect();
  const sourceX = event.clientX ?? wrap.left + Number(event.currentTarget.getAttribute("cx"));
  const sourceY = event.clientY ?? wrap.top + Number(event.currentTarget.getAttribute("cy"));
  let left = sourceX - wrap.left + 14;
  let top = sourceY - wrap.top + 14;
  if (left + tipRect.width > wrap.width - 10) left = sourceX - wrap.left - tipRect.width - 14;
  if (top + tipRect.height > wrap.height - 10) top = sourceY - wrap.top - tipRect.height - 14;
  tooltip.style.left = `${Math.max(10, left)}px`;
  tooltip.style.top = `${Math.max(10, top)}px`;
}

function hideTooltip() {
  if (state.activeDot) {
    state.activeDot.classList.remove("is-active");
    state.activeDot = null;
  }
  els.tooltip.hidden = true;
}

function updateSummary(rows) {
  const summary = summariesByKey.get(`${state.setup}::${state.pathId}`) || {};
  const first = rows[0] || {};
  const last = rows[rows.length - 1] || {};
  els.selectedPathName.textContent = state.pathId || "Select a path";
  els.mapTitle.textContent = `${state.setup}: ${state.pathId}`;
  els.observedDays.textContent = `${summary.observed_days ?? "--"} days`;
  els.rolloutDays.textContent = `${summary.rollout_steps ?? "--"} days`;
  els.directMean.textContent = formatKm(summary.direct_mean_rollout_error_km);
  els.lstmMean.textContent = formatKm(summary.lstm_mean_rollout_error_km);
  els.transformerMean.textContent = formatKm(summary.transformer_mean_rollout_error_km);
  els.directFinal.textContent = formatKm(summary.direct_final_rollout_error_km);
  els.lstmFinal.textContent = formatKm(summary.lstm_final_rollout_error_km);
  els.transformerFinal.textContent = formatKm(summary.transformer_final_rollout_error_km);
  els.sourceBird.textContent = summary.source_bird_id || first.source_bird_id || "--";
  els.dateRange.textContent = first.date && last.date ? `${first.date} to ${last.date}` : "--";
  els.observedDistance.textContent = formatKm(summary.observed_displacement_km);
  els.modelNames.textContent = `Direct: ${summary.direct_model || first.direct_model || "--"}; LSTM: ${summary.lstm_model || first.lstm_model || "--"}; Transformer: ${summary.transformer_model || first.transformer_model || "--"}`;
}

function render() {
  const rows = rowsForSelection();
  hideTooltip();
  updateSummary(rows);
  drawPlot(rows);
}

function populateSetupSelect() {
  const setups = [...new Set(DATA.summaries.map((row) => row.setup))].sort();
  els.setupSelect.replaceChildren();
  setups.forEach((setup) => {
    const option = document.createElement("option");
    option.value = setup;
    option.textContent = setup;
    els.setupSelect.appendChild(option);
  });
  state.setup = setups[0] || "";
  els.setupSelect.value = state.setup;
}

function populatePathSelect() {
  const summaries = DATA.summaries
    .filter((row) => row.setup === state.setup)
    .sort((a, b) => Number(b.rollout_steps) - Number(a.rollout_steps) || a.path_id.localeCompare(b.path_id));
  els.pathSelect.replaceChildren();
  summaries.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.path_id;
    option.textContent = `${row.source_bird_id} / ${row.path_year} / ${row.rollout_steps} rollout days`;
    els.pathSelect.appendChild(option);
  });
  state.pathId = summaries[0]?.path_id || "";
  els.pathSelect.value = state.pathId;
}

els.setupSelect.addEventListener("change", () => {
  state.setup = els.setupSelect.value;
  state.viewMode = "path";
  populatePathSelect();
  render();
});

els.pathSelect.addEventListener("change", () => {
  state.pathId = els.pathSelect.value;
  state.viewMode = "path";
  render();
});

[els.showObserved, els.showTruth, els.showDirect, els.showLstm, els.showTransformer, els.showPredictedPoints, els.showPersistence, els.showVelocity]
  .forEach((input) => input.addEventListener("input", render));

els.fitPathButton.addEventListener("click", () => {
  state.viewMode = "path";
  render();
});

els.fitAllButton.addEventListener("click", () => {
  state.viewMode = "all";
  render();
});

window.addEventListener("resize", render);

populateSetupSelect();
populatePathSelect();
render();
"""


def run_setup(
    *,
    setup_name: str,
    fly_threshold_km: float,
    df: pd.DataFrame,
    output_dir: Path,
    model_specs: list[dict[str, object]],
    include_full_baselines: bool,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    setup_dir = output_dir / setup_name
    setup_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Running %s with fly threshold %.1f km", setup_name, fly_threshold_km)

    all_rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    triline_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    cache: dict[int, tuple[WindowData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for k in needed_k_values(model_specs, include_full_baselines):
        window_data = build_windows(df, k=k, fly_threshold_km=fly_threshold_km)
        if len(window_data.features) == 0:
            logging.warning("%s k=%s has no windows; skipping", setup_name, k)
            continue
        train_idx, test_idx = make_chronological_split(window_data)
        feature_mean, feature_std = fit_normalizer(window_data.features, train_idx)
        normalized_features = normalize_features(window_data.features, feature_mean, feature_std)
        cache[k] = (window_data, train_idx, test_idx, feature_mean, feature_std)

    baseline_k_values = FULL_BASELINE_K if include_full_baselines else [7, 30]
    for k in baseline_k_values:
        if k not in cache:
            continue
        window_data, train_idx, test_idx, _, _ = cache[k]
        baseline_dir = setup_dir / f"k_{k}" / "baselines"
        for baseline_name in ["persistence", "const_velocity"]:
            row = run_baseline(
                name=baseline_name,
                window_data=window_data,
                test_idx=test_idx,
                output_dir=baseline_dir,
                fly_threshold_km=fly_threshold_km,
            )
            row.update({"setup": setup_name})
            baseline_rows.append(row)
            all_rows.append(row)

    for spec in model_specs:
        k = int(spec["k"])
        if k not in cache:
            continue
        window_data, train_idx, test_idx, feature_mean, feature_std = cache[k]
        normalized_features = normalize_features(window_data.features, feature_mean, feature_std)
        normalized_for_model = select_feature_mode(
            normalized_features, window_data.feature_columns, spec.get("feature_mode", "full")
        )
        model_dir = setup_dir / f"k_{k}"
        logging.info("%s running %s", setup_name, spec["name"])
        if str(spec["kind"]).startswith("direct"):
            row = train_direct_model(
                model_spec=spec,
                window_data=window_data,
                train_idx=train_idx,
                test_idx=test_idx,
                normalized_features=normalized_for_model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                output_dir=model_dir,
                fly_threshold_km=fly_threshold_km,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
            )
            row.update({"setup": setup_name})
            direct_rows.append(row)
        else:
            row = train_triline_model(
                model_spec=spec,
                window_data=window_data,
                train_idx=train_idx,
                test_idx=test_idx,
                normalized_features=normalized_for_model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                output_dir=model_dir,
                fly_threshold_km=fly_threshold_km,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
            )
            row.update({"setup": setup_name})
            triline_rows.append(row)
        all_rows.append(row)

    write_csv(setup_dir / "comparison_summary.csv", all_rows)
    write_csv(setup_dir / "baseline_summary.csv", baseline_rows)
    write_csv(setup_dir / "direct_summary.csv", direct_rows)
    write_csv(setup_dir / "triline_summary.csv", triline_rows)
    return baseline_rows, direct_rows, triline_rows


COMPARISON_METRICS = [
    "mean_error_km",
    "median_error_km",
    "p90_error_km",
    "p95_error_km",
    "migration50_error_mean_km",
    "stationary_error_mean_km",
    "fly_error_mean_km",
    "gated_mean_error_km",
    "gated_median_error_km",
    "gated_p90_error_km",
    "gated_p95_error_km",
    "gated_migration50_error_mean_km",
    "gated_stationary_error_mean_km",
    "gated_fly_error_mean_km",
]


def create_weather_vs_noweather_comparison(
    output_dir: Path,
    weather_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    no_weather_csv = NOWEATHER_OUTPUT_DIR / "comparison_summary.csv"
    if not no_weather_csv.exists() or not weather_rows:
        return []

    weather_df = pd.DataFrame(weather_rows)
    no_weather_df = pd.read_csv(no_weather_csv)
    join_keys = ["setup", "model_family", "model", "kind", "k"]
    for key in join_keys:
        if key not in weather_df.columns or key not in no_weather_df.columns:
            return []

    merged = weather_df.merge(
        no_weather_df,
        on=join_keys,
        how="inner",
        suffixes=("_weather", "_noweather"),
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        payload: dict[str, object] = {key: row[key] for key in join_keys}
        for metric in COMPARISON_METRICS:
            weather_key = f"{metric}_weather"
            noweather_key = f"{metric}_noweather"
            if weather_key not in row.index or noweather_key not in row.index:
                continue
            weather_value = pd.to_numeric(pd.Series([row[weather_key]]), errors="coerce").iloc[0]
            noweather_value = pd.to_numeric(pd.Series([row[noweather_key]]), errors="coerce").iloc[0]
            if pd.isna(weather_value) or pd.isna(noweather_value):
                continue
            payload[f"{metric}_weather"] = float(weather_value)
            payload[f"{metric}_noweather"] = float(noweather_value)
            payload[f"{metric}_delta_weather_minus_noweather"] = float(weather_value - noweather_value)
            payload[f"{metric}_improved"] = bool(weather_value < noweather_value)
        rows.append(payload)

    write_csv(output_dir / "weather_vs_noweather_comparison.csv", rows)
    return rows


def create_rollout_weather_vs_noweather_comparison(output_dir: Path) -> None:
    weather_csv = output_dir / "rollout_all" / "rollout_all_summary.csv"
    no_weather_csv = NOWEATHER_OUTPUT_DIR / "rollout_all" / "rollout_all_summary.csv"
    if not weather_csv.exists() or not no_weather_csv.exists():
        return

    weather_df = pd.read_csv(weather_csv)
    no_weather_df = pd.read_csv(no_weather_csv)
    join_keys = ["setup", "path_id"]
    for key in join_keys:
        if key not in weather_df.columns or key not in no_weather_df.columns:
            return
    metric_columns = [
        "direct_mean_rollout_error_km",
        "direct_final_rollout_error_km",
        "lstm_mean_rollout_error_km",
        "lstm_final_rollout_error_km",
        "transformer_mean_rollout_error_km",
        "transformer_final_rollout_error_km",
    ]
    merged = weather_df.merge(
        no_weather_df,
        on=join_keys,
        how="inner",
        suffixes=("_weather", "_noweather"),
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        payload: dict[str, object] = {key: row[key] for key in join_keys}
        for metric in metric_columns:
            weather_key = f"{metric}_weather"
            noweather_key = f"{metric}_noweather"
            if weather_key not in row.index or noweather_key not in row.index:
                continue
            weather_value = pd.to_numeric(pd.Series([row[weather_key]]), errors="coerce").iloc[0]
            noweather_value = pd.to_numeric(pd.Series([row[noweather_key]]), errors="coerce").iloc[0]
            if pd.isna(weather_value) or pd.isna(noweather_value):
                continue
            payload[f"{metric}_weather"] = float(weather_value)
            payload[f"{metric}_noweather"] = float(noweather_value)
            payload[f"{metric}_delta_weather_minus_noweather"] = float(weather_value - noweather_value)
            payload[f"{metric}_improved"] = bool(weather_value < noweather_value)
        rows.append(payload)
    write_csv(output_dir / "rollout_weather_vs_noweather_comparison.csv", rows)


def write_analysis(
    output_dir: Path,
    rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]] | None = None,
) -> None:
    if not rows:
        (output_dir / "analysis.md").write_text("# Path Experiment With Weather\n\nNo rows produced.\n", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    lines = [
        "# Path Experiment With Weather",
        "",
        "## Setup",
        "",
        "- Dataset: southbound path segments.",
        "- Features: compact 18-feature inputs with 10 path features and 8 weather features.",
        "- Weather match: exact bird/date join with nearest same-bird date fallback.",
        "- Split: chronological 80/20 within constructed windows.",
        "- Identity: source bird ID embedding for neural models.",
        "",
        "## Best Mean GPS Error By Setup",
        "",
        "| Setup | Model | Family | k | Mean km | Median km | Fly recall | Params |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for setup, setup_df in df.groupby("setup"):
        metric_df = setup_df.dropna(subset=["mean_error_km"]).copy()
        best = metric_df.loc[metric_df["mean_error_km"].astype(float).idxmin()]
        fly_recall = best.get("fly_recall", "")
        fly_recall_text = "" if pd.isna(fly_recall) or fly_recall == "" else f"{float(fly_recall):.4f}"
        lines.append(
            f"| {setup} | {best['model']} | {best['model_family']} | {int(best['k'])} | "
            f"{float(best['mean_error_km']):.4f} | {float(best['median_error_km']):.4f} | "
            f"{fly_recall_text} | {int(best.get('params', 0) or 0)} |"
        )

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        lines.extend(
            [
                "",
                "## Weather vs No-Weather",
                "",
                "| Setup | Model | Family | k | Weather Mean km | No-Weather Mean km | Delta km | Improved |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        display_comparison = comparison_df.dropna(
            subset=["mean_error_km_weather", "mean_error_km_noweather"]
        ).sort_values(["setup", "mean_error_km_weather"])
        for _, row in display_comparison.iterrows():
            delta = float(row["mean_error_km_delta_weather_minus_noweather"])
            improved = "yes" if bool(row["mean_error_km_improved"]) else "no"
            lines.append(
                f"| {row['setup']} | {row['model']} | {row['model_family']} | {int(row['k'])} | "
                f"{float(row['mean_error_km_weather']):.4f} | {float(row['mean_error_km_noweather']):.4f} | "
                f"{delta:.4f} | {improved} |"
            )

    lines.extend(["", "## Full Comparison", ""])
    display = df.sort_values(["setup", "model_family", "k", "model"])
    lines.extend(
        [
            "| Setup | Family | Model | k | Mean km | Median km | P90 km | Fly Mean km | Migration50 Mean km |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in display.iterrows():
        def fmt_cell(value: object) -> str:
            if value is None or value == "" or pd.isna(value):
                return ""
            return f"{float(value):.4f}"

        lines.append(
            f"| {row['setup']} | {row['model_family']} | {row['model']} | {int(row['k'])} | "
            f"{fmt_cell(row.get('mean_error_km'))} | {fmt_cell(row.get('median_error_km'))} | "
            f"{fmt_cell(row.get('p90_error_km'))} | {fmt_cell(row.get('fly_error_mean_km'))} | "
            f"{fmt_cell(row.get('migration50_error_mean_km'))} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Triline rows report ungated GPS reconstruction as primary metrics.",
            "- Gated GPS metrics are stored in each triline `metrics.json` and the CSV summaries with `gated_` prefixes.",
            "- Direct model rows repeat per setup because setup-specific fly thresholds change stratified error slices.",
            "- Autoregressive rollout uses actual future daily weather as exogenous forecast/oracle weather.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather-enabled southbound path trajectory experiment")
    parser.add_argument("--output-dir", default="Path_Experiment_WithWeather")
    parser.add_argument("--input-csv", default=str(SOUTHBOUND_CSV))
    parser.add_argument("--prejoined-weather", action="store_true")
    parser.add_argument("--matched-output-name", default=None)
    parser.add_argument("--matrix", choices=["full", "smoke"], default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--visualize-rollout", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--rollout-all", action="store_true")
    parser.add_argument("--rollout-all-only", action="store_true")
    parser.add_argument("--rollout-length", type=int, default=60)
    parser.add_argument(
        "--rollout-setup",
        choices=["both", "setup_30km", "setup_10km"],
        default="both",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_device(device_arg: str, smoke: bool) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if smoke:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = ROOT / input_csv
    smoke = bool(args.smoke or args.matrix == "smoke")
    max_epochs = 1 if smoke else int(args.max_epochs)
    patience = 1 if smoke else int(args.patience)
    model_specs = smoke_model_specs() if smoke else full_model_specs()
    include_full_baselines = not smoke

    set_seed(SEED)
    device = choose_device(args.device, smoke)
    completed_path = output_dir / "COMPLETED.txt"
    if (
        not args.worker
        and device.type == "cuda"
        and not args.rollout_only
        and not args.rollout_all_only
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        if completed_path.exists():
            completed_path.unlink()
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        result = subprocess.run(command, cwd=ROOT)
        if completed_path.exists():
            os._exit(0)
        raise RuntimeError(f"CUDA worker failed before completion; exit code {result.returncode}")

    setup_logging(output_dir)
    logging.info("Using device: %s", device)
    if device.type == "cuda":
        logging.info("CUDA device: %s", torch.cuda.get_device_name(0))

    raw_df = pd.read_csv(input_csv)
    if args.prejoined_weather:
        df, weather_match_summary = use_prejoined_weather_paths(raw_df, input_csv)
    else:
        weather_df = load_weather_data()
        df, weather_match_summary = add_weather_to_paths(raw_df, weather_df)
    matched_output_name = args.matched_output_name
    if matched_output_name is None:
        matched_output_name = (
            "prejoined_southbound_paths_with_weather.csv"
            if args.prejoined_weather
            else "dataset2_southbound_paths_with_weather.csv"
        )
    matched_csv = output_dir / matched_output_name
    df.to_csv(matched_csv, index=False)
    write_json(output_dir / "weather_match_summary.json", weather_match_summary)
    rollout_setup_names = (
        ["setup_30km", "setup_10km"] if args.rollout_setup == "both" else [str(args.rollout_setup)]
    )
    if args.rollout_only:
        run_rollout_visualizations(
            df=df,
            output_dir=output_dir,
            rollout_length=int(args.rollout_length),
            setup_names=rollout_setup_names,
            device=device,
        )
        (output_dir / "ROLLOUT_COMPLETED.txt").write_text(
            f"Rollout visualizations saved to {output_dir}\n", encoding="utf-8"
        )
        create_rollout_weather_vs_noweather_comparison(output_dir)
        os._exit(0)
    if args.rollout_all_only:
        run_all_test_rollouts(
            df=df,
            output_dir=output_dir,
            setup_names=rollout_setup_names,
            device=device,
        )
        (output_dir / "ROLLOUT_ALL_COMPLETED.txt").write_text(
            f"All-path rollout explorer saved to {output_dir / 'rollout_all'}\n",
            encoding="utf-8",
        )
        create_rollout_weather_vs_noweather_comparison(output_dir)
        os._exit(0)

    data_summary = summarize_data(df, input_csv)
    data_summary["setups"] = {
        "setup_30km": {"source": "flynofly_path/southbound_compact", "fly_threshold_km": 30.0},
        "setup_10km": {"source": "flynofly_path/southbound_compact_threshold10", "fly_threshold_km": 10.0},
    }
    write_json(output_dir / "data_summary.json", data_summary)
    write_json(
        output_dir / "run_config.json",
        {
            "seed": SEED,
            "matrix": "smoke" if smoke else "full",
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": 128,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=4)",
            "early_stopping_patience": patience,
            "triline_loss_weights": {"fly": 1.0, "distance": 1.0, "direction": 0.5},
            "device": str(device),
            "input_csv": str(input_csv),
            "prejoined_weather": bool(args.prejoined_weather),
            "matched_dataset_csv": str(matched_csv),
            "weather_data": True,
            "weather_match_policy": "exact bird/date, nearest same-bird date fallback",
            "weather_rollout_policy": "actual future daily weather as exogenous forecast/oracle input",
            "feature_columns": COMPACT_FEATURE_COLUMNS,
            "weather_feature_columns": WEATHER_FEATURE_COLUMNS,
            "model_specs": model_specs,
        },
    )

    all_baseline_rows: list[dict[str, object]] = []
    all_direct_rows: list[dict[str, object]] = []
    all_triline_rows: list[dict[str, object]] = []
    for setup_name, fly_threshold_km in [("setup_30km", 30.0), ("setup_10km", 10.0)]:
        baseline_rows, direct_rows, triline_rows = run_setup(
            setup_name=setup_name,
            fly_threshold_km=fly_threshold_km,
            df=df,
            output_dir=output_dir,
            model_specs=model_specs,
            include_full_baselines=include_full_baselines,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
        all_baseline_rows.extend(baseline_rows)
        all_direct_rows.extend(direct_rows)
        all_triline_rows.extend(triline_rows)

    all_rows = all_baseline_rows + all_direct_rows + all_triline_rows
    write_csv(output_dir / "baseline_summary.csv", all_baseline_rows)
    write_csv(output_dir / "direct_summary.csv", all_direct_rows)
    write_csv(output_dir / "triline_summary.csv", all_triline_rows)
    write_csv(output_dir / "comparison_summary.csv", all_rows)
    comparison_rows = create_weather_vs_noweather_comparison(output_dir, all_rows)
    write_analysis(output_dir, all_rows, comparison_rows)
    if args.visualize_rollout:
        run_rollout_visualizations(
            df=df,
            output_dir=output_dir,
            rollout_length=int(args.rollout_length),
            setup_names=rollout_setup_names,
            device=device,
        )
        create_rollout_weather_vs_noweather_comparison(output_dir)
    if args.rollout_all:
        run_all_test_rollouts(
            df=df,
            output_dir=output_dir,
            setup_names=rollout_setup_names,
            device=device,
        )
        create_rollout_weather_vs_noweather_comparison(output_dir)
        (output_dir / "ROLLOUT_ALL_COMPLETED.txt").write_text(
            f"All-path rollout explorer saved to {output_dir / 'rollout_all'}\n",
            encoding="utf-8",
        )
    (output_dir / "COMPLETED.txt").write_text(
        f"Done. Results saved to {output_dir}\n", encoding="utf-8"
    )
    os._exit(0)


if __name__ == "__main__":
    main()
