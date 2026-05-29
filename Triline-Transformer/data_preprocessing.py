from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_FEATURE_COLUMNS = [
    "lat_median",
    "lon_median",
    "delta_lat",
    "delta_lon",
    "step_length_km",
    "speed_km_per_day",
    "heading_sin",
    "heading_cos",
    "turning_sin",
    "turning_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "stopover_duration_days",
    "n_points",
]

ROLLING_FEATURE_COLUMNS = [
    "step_mean_3",
    "step_mean_7",
    "step_mean_14",
    "step_max_3",
    "step_max_7",
    "step_max_14",
    "speed_mean_3",
    "speed_mean_7",
    "speed_mean_14",
    "speed_max_3",
    "speed_max_7",
    "speed_max_14",
    "lat_change_3",
    "lat_change_7",
    "lat_change_14",
    "lon_change_3",
    "lon_change_7",
    "lon_change_14",
    "displacement_km_3",
    "displacement_km_7",
    "displacement_km_14",
    "lat_trend_7",
    "lon_trend_7",
    "days_since_fly_30",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ROLLING_FEATURE_COLUMNS
TARGET_MONTHS = {9, 10, 11, 12}


@dataclass(frozen=True)
class WindowData:
    features: np.ndarray
    labels: np.ndarray
    log_distance_targets: np.ndarray
    direction_targets: np.ndarray
    bird_ids: np.ndarray
    target_dates: np.ndarray
    target_step_km: np.ndarray
    bird_to_idx: dict[str, int]
    feature_columns: list[str]


def preprocess_movement_data(
    input_csv: Path,
    output_csv: Path,
    min_bird_records: int = 100,
    min_lat: float = 30.0,
    max_lat: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Filter bird eligibility and latitude, preserving all months as model context."""
    df = pd.read_csv(input_csv)
    original_rows = len(df)
    original_birds = df["individual_local_identifier"].nunique()

    df["date"] = pd.to_datetime(df["date"], utc=True)
    bird_counts = df.groupby("individual_local_identifier").size()
    eligible_birds = bird_counts[bird_counts >= min_bird_records].index

    cleaned = df[
        df["individual_local_identifier"].isin(eligible_birds)
        & df["lat_median"].between(min_lat, max_lat, inclusive="both")
    ].copy()
    cleaned = cleaned.sort_values(["individual_local_identifier", "date"]).reset_index(drop=True)

    target_rows = cleaned[cleaned["date"].dt.month.isin(TARGET_MONTHS)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv, index=False)

    summary: dict[str, int | float | str] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "original_rows": int(original_rows),
        "original_birds": int(original_birds),
        "eligible_birds_min_100_records": int(len(eligible_birds)),
        "context_rows_lat30_50_all_months": int(len(cleaned)),
        "context_birds": int(cleaned["individual_local_identifier"].nunique()),
        "sep_dec_target_candidate_rows": int(len(target_rows)),
        "sep_dec_target_fly_rate_threshold_30km": float((target_rows["step_length_km"] > 30.0).mean())
        if len(target_rows)
        else 0.0,
    }
    return cleaned, summary


def _rolling_prior_or_current(series: pd.Series, window: int, reducer: str) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=1)
    if reducer == "mean":
        return rolling.mean()
    if reducer == "max":
        return rolling.max()
    raise ValueError(f"Unknown reducer: {reducer}")


def _days_since_fly(step_lengths: pd.Series, threshold_km: float = 30.0) -> pd.Series:
    values: list[int] = []
    last_fly_pos: int | None = None
    for pos, step in enumerate(step_lengths.fillna(0.0).to_numpy()):
        if step > threshold_km:
            last_fly_pos = pos
            values.append(0)
        elif last_fly_pos is None:
            values.append(999)
        else:
            values.append(pos - last_fly_pos)
    return pd.Series(values, index=step_lengths.index, dtype=float)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily base and temporal features from observed records only."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["individual_local_identifier", "date"]).reset_index(drop=True)

    out["delta_lat"] = out.groupby("individual_local_identifier")["lat_median"].diff().fillna(0.0)
    out["delta_lon"] = out.groupby("individual_local_identifier")["lon_median"].diff().fillna(0.0)

    heading_rad = np.deg2rad(out["heading_deg"].fillna(0.0).astype(float))
    turning_rad = np.deg2rad(out["turning_angle_deg"].fillna(0.0).astype(float))
    out["heading_sin"] = np.sin(heading_rad)
    out["heading_cos"] = np.cos(heading_rad)
    out["turning_sin"] = np.sin(turning_rad)
    out["turning_cos"] = np.cos(turning_rad)

    day_of_year = out["date"].dt.dayofyear.astype(float)
    month = out["date"].dt.month.astype(float)
    out["doy_sin"] = np.sin(2.0 * math.pi * day_of_year / 366.0)
    out["doy_cos"] = np.cos(2.0 * math.pi * day_of_year / 366.0)
    out["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)

    grouped = out.groupby("individual_local_identifier", group_keys=False)
    for window in (3, 7, 14):
        out[f"step_mean_{window}"] = grouped["step_length_km"].transform(
            lambda s, w=window: _rolling_prior_or_current(s.fillna(0.0), w, "mean")
        )
        out[f"step_max_{window}"] = grouped["step_length_km"].transform(
            lambda s, w=window: _rolling_prior_or_current(s.fillna(0.0), w, "max")
        )
        out[f"speed_mean_{window}"] = grouped["speed_km_per_day"].transform(
            lambda s, w=window: _rolling_prior_or_current(s.fillna(0.0), w, "mean")
        )
        out[f"speed_max_{window}"] = grouped["speed_km_per_day"].transform(
            lambda s, w=window: _rolling_prior_or_current(s.fillna(0.0), w, "max")
        )
        out[f"lat_change_{window}"] = grouped["lat_median"].transform(
            lambda s, w=window: s.diff(w - 1).fillna(0.0)
        )
        out[f"lon_change_{window}"] = grouped["lon_median"].transform(
            lambda s, w=window: s.diff(w - 1).fillna(0.0)
        )
        mean_lat = out["lat_median"].astype(float)
        out[f"displacement_km_{window}"] = np.sqrt(
            (out[f"lat_change_{window}"] * 111.0) ** 2
            + (out[f"lon_change_{window}"] * 111.0 * np.cos(np.deg2rad(mean_lat))) ** 2
        )

    out["lat_trend_7"] = out["lat_change_7"] / 7.0
    out["lon_trend_7"] = out["lon_change_7"] / 7.0
    out["days_since_fly_30"] = grouped["step_length_km"].transform(_days_since_fly)

    for column in FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out


def build_windows(
    df: pd.DataFrame,
    k: int,
    fly_threshold_km: float = 30.0,
    target_months: set[int] | None = None,
) -> WindowData:
    """Build calendar-consecutive windows with Sep-Dec targets and all-month context."""
    target_months = target_months or TARGET_MONTHS
    featured = add_engineered_features(df)
    bird_names = sorted(featured["individual_local_identifier"].unique())
    bird_to_idx = {bird: idx for idx, bird in enumerate(bird_names)}

    features: list[np.ndarray] = []
    labels: list[float] = []
    log_distance_targets: list[float] = []
    direction_targets: list[tuple[float, float]] = []
    bird_ids: list[int] = []
    target_dates: list[np.datetime64] = []
    target_step_km: list[float] = []

    for bird, bird_df in featured.groupby("individual_local_identifier", sort=False):
        bird_df = bird_df.sort_values("date").reset_index(drop=True)
        dates = bird_df["date"].dt.normalize()
        day_diffs = dates.diff().dt.days.fillna(1).to_numpy()

        for start in range(0, len(bird_df) - k):
            end = start + k
            target_row = bird_df.iloc[end]
            if int(target_row["date"].month) not in target_months:
                continue
            if not np.all(day_diffs[start + 1 : end + 1] == 1):
                continue

            step_km = float(target_row["step_length_km"])
            heading_deg = float(target_row["heading_deg"]) if pd.notna(target_row["heading_deg"]) else 0.0
            heading_rad = math.radians(heading_deg)

            input_rows = bird_df.iloc[start:end]
            features.append(input_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            labels.append(float(step_km > fly_threshold_km))
            log_distance_targets.append(math.log1p(max(step_km, 0.0)))
            direction_targets.append((math.sin(heading_rad), math.cos(heading_rad)))
            bird_ids.append(bird_to_idx[bird])
            target_dates.append(np.datetime64(target_row["date"].to_datetime64()))
            target_step_km.append(step_km)

    if not features:
        empty_features = np.empty((0, k, len(FEATURE_COLUMNS)), dtype=np.float32)
        return WindowData(
            empty_features,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=np.float32),
            bird_to_idx,
            FEATURE_COLUMNS.copy(),
        )

    return WindowData(
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        log_distance_targets=np.asarray(log_distance_targets, dtype=np.float32),
        direction_targets=np.asarray(direction_targets, dtype=np.float32),
        bird_ids=np.asarray(bird_ids, dtype=np.int64),
        target_dates=np.asarray(target_dates, dtype="datetime64[ns]"),
        target_step_km=np.asarray(target_step_km, dtype=np.float32),
        bird_to_idx=bird_to_idx,
        feature_columns=FEATURE_COLUMNS.copy(),
    )


def make_split(
    window_data: WindowData,
    split_mode: str,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "chronological":
        order = np.argsort(window_data.target_dates, kind="stable")
        split_idx = int(len(order) * train_fraction)
        if split_idx <= 0 or split_idx >= len(order):
            raise ValueError(f"Need enough windows for an 80/20 split, got {len(order)} windows.")
        return order[:split_idx], order[split_idx:]

    if split_mode == "stratified":
        indices = np.arange(len(window_data.labels))
        train_idx, test_idx = train_test_split(
            indices,
            train_size=train_fraction,
            random_state=seed,
            stratify=window_data.labels,
        )
        return np.asarray(train_idx), np.asarray(test_idx)

    raise ValueError(f"Unknown split mode: {split_mode}")


def fit_normalizer(features: np.ndarray, train_idx: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    train_features = features[np.asarray(list(train_idx))]
    flat = train_features.reshape(-1, train_features.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
