from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import run_path_experiment_no_weather as no_weather
import run_path_experiment_with_weather as with_weather


SETUP_NAME = "fly_threshold_10km"
FLY_THRESHOLD_KM = 10.0
ROLLOUT_CONTEXT_DAYS = 60
ROLLOUT_MIN_DISPLACEMENT_KM = 50.0
DEFAULT_GIF_DAYS = 14
DEFAULT_MAX_GIF_CASES = 7

MODEL_STYLES = {
    "truth": {"label": "True", "color": "#111111", "linestyle": "-", "linewidth": 2.6},
    "direct": {"label": "Direct", "color": "#c47a00", "linestyle": "--", "linewidth": 2.2},
    "lstm": {"label": "LSTM", "color": "#2563eb", "linestyle": "-", "linewidth": 2.2},
    "transformer": {"label": "Transformer", "color": "#7c3aed", "linestyle": "-", "linewidth": 2.2},
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def release_torch_objects(*objects: object) -> None:
    del objects
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_condition(
    *,
    label: str,
    runner: Any,
    df: pd.DataFrame,
    result_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    featured = runner.add_southbound_compact_features(df)
    setup_dir = result_dir / SETUP_NAME
    if not setup_dir.exists():
        raise FileNotFoundError(f"Missing setup directory: {setup_dir}")

    app_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    window_data = runner.build_windows(df, k=runner.ROLLOUT_CONTEXT_K, fly_threshold_km=FLY_THRESHOLD_KM)
    _, test_idx = runner.make_chronological_split(window_data)
    test_path_ids = sorted(set(str(path_id) for path_id in window_data.target_path_ids[test_idx]))

    direct_row = runner.best_summary_row(setup_dir, "direct")
    lstm_row = runner.best_summary_row(setup_dir, "triline", kind="triline_lstm")
    transformer_row = runner.best_summary_row(setup_dir, "triline", kind="triline_transformer")

    direct_model, direct_checkpoint = runner.load_checkpoint_model(
        runner.model_checkpoint_path(setup_dir, direct_row), "direct", device
    )
    lstm_model, lstm_checkpoint = runner.load_checkpoint_model(
        runner.model_checkpoint_path(setup_dir, lstm_row), "triline", device
    )
    transformer_model, transformer_checkpoint = runner.load_checkpoint_model(
        runner.model_checkpoint_path(setup_dir, transformer_row), "triline", device
    )

    for path_id in test_path_ids:
        path_df = (
            featured[featured[runner.GROUP_COLUMN].astype(str).eq(path_id)]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(path_df) <= ROLLOUT_CONTEXT_DAYS:
            skipped_rows.append(
                {
                    "condition": label,
                    "setup": SETUP_NAME,
                    "path_id": path_id,
                    "reason": "not_enough_rows_for_60_observed_days",
                    "rows": int(len(path_df)),
                }
            )
            continue

        context_days, observed_displacement_km = runner.observed_context_days_for_path(
            path_df,
            min_context_days=ROLLOUT_CONTEXT_DAYS,
            min_displacement_km=ROLLOUT_MIN_DISPLACEMENT_KM,
        )
        if context_days >= len(path_df):
            skipped_rows.append(
                {
                    "condition": label,
                    "setup": SETUP_NAME,
                    "path_id": path_id,
                    "reason": "no_rollout_days_after_60_day_50km_observed_context",
                    "rows": int(len(path_df)),
                    "observed_days": int(context_days),
                    "observed_displacement_km": float(observed_displacement_km),
                }
            )
            continue

        rollout_steps = len(path_df) - context_days
        direct_lat, direct_lon = runner.rollout_direct_model(
            direct_model,
            direct_checkpoint,
            path_df,
            rollout_steps,
            device,
            context_days=context_days,
        )
        lstm_lat, lstm_lon, lstm_fly_prob = runner.rollout_triline_model(
            lstm_model,
            lstm_checkpoint,
            path_df,
            rollout_steps,
            device,
            context_days=context_days,
        )
        transformer_lat, transformer_lon, transformer_fly_prob = runner.rollout_triline_model(
            transformer_model,
            transformer_checkpoint,
            path_df,
            rollout_steps,
            device,
            context_days=context_days,
        )
        baselines = runner.rollout_baselines(path_df, rollout_steps, context_days=context_days)

        rows = runner.build_rollout_rows_for_path(
            setup_name=SETUP_NAME,
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
        for row in rows:
            row["condition"] = label
        app_rows.extend(rows)

        direct_mean, direct_final = runner.rollout_error_summary(path_df, direct_lat, direct_lon, context_days)
        lstm_mean, lstm_final = runner.rollout_error_summary(path_df, lstm_lat, lstm_lon, context_days)
        transformer_mean, transformer_final = runner.rollout_error_summary(
            path_df, transformer_lat, transformer_lon, context_days
        )
        persistence_mean, persistence_final = runner.rollout_error_summary(
            path_df, baselines["persistence"][0], baselines["persistence"][1], context_days
        )
        velocity_mean, velocity_final = runner.rollout_error_summary(
            path_df, baselines["const_velocity"][0], baselines["const_velocity"][1], context_days
        )
        first = path_df.iloc[0]
        summary_rows.append(
            {
                "condition": label,
                "setup": SETUP_NAME,
                "path_id": path_id,
                "source_bird_id": str(first.get(runner.IDENTITY_COLUMN, "")),
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

    write_csv(output_dir / "rollout_all_predictions.csv", app_rows)
    write_csv(output_dir / "rollout_all_summary.csv", summary_rows)
    write_csv(output_dir / "rollout_all_skipped.csv", skipped_rows)
    index = {
        "condition": label,
        "rows": int(len(app_rows)),
        "paths": int(len(summary_rows)),
        "skipped_paths": int(len(skipped_rows)),
        "setup": SETUP_NAME,
        "fly_threshold_km": FLY_THRESHOLD_KM,
        "min_observed_days": ROLLOUT_CONTEXT_DAYS,
        "min_observed_displacement_km": ROLLOUT_MIN_DISPLACEMENT_KM,
        "direct_model": str(direct_row["model"]),
        "lstm_model": str(lstm_row["model"]),
        "transformer_model": str(transformer_row["model"]),
        "predictions_csv": str(output_dir / "rollout_all_predictions.csv"),
        "summary_csv": str(output_dir / "rollout_all_summary.csv"),
        "skipped_csv": str(output_dir / "rollout_all_skipped.csv"),
    }
    write_json(output_dir / "rollout_all_index.json", index)
    return index


def metric_value(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("inf")


def select_first_unique(
    candidates: pd.DataFrame,
    selected_keys: set[str],
    reason: str,
) -> dict[str, object] | None:
    for _, row in candidates.iterrows():
        key = str(row["path_id"])
        if key in selected_keys:
            continue
        selected_keys.add(key)
        payload = row.to_dict()
        payload["selection_reason"] = reason
        return payload
    return None


def select_gif_cases(
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    condition: str,
    max_cases: int,
) -> list[dict[str, object]]:
    metrics = [
        "direct_mean_rollout_error_km",
        "lstm_mean_rollout_error_km",
        "transformer_mean_rollout_error_km",
        "rollout_steps",
        "observed_displacement_km",
    ]
    candidates = numeric(summary, metrics)
    candidates = candidates[
        (candidates["rollout_steps"].astype(float) >= 3)
        & (candidates["observed_displacement_km"].astype(float) >= ROLLOUT_MIN_DISPLACEMENT_KM)
    ].copy()
    if candidates.empty:
        candidates = numeric(summary, metrics)
        candidates = candidates[candidates["rollout_steps"].astype(float) >= 1].copy()
    if candidates.empty:
        return []

    model_error_cols = [
        "direct_mean_rollout_error_km",
        "lstm_mean_rollout_error_km",
        "transformer_mean_rollout_error_km",
    ]
    candidates["best_model_mean_error_km"] = candidates[model_error_cols].min(axis=1)
    candidates["all_model_mean_error_km"] = candidates[model_error_cols].mean(axis=1)
    candidates["lstm_margin_km"] = candidates["lstm_mean_rollout_error_km"] - candidates[
        ["direct_mean_rollout_error_km", "transformer_mean_rollout_error_km"]
    ].min(axis=1)
    candidates["transformer_margin_km"] = candidates["transformer_mean_rollout_error_km"] - candidates[
        ["direct_mean_rollout_error_km", "lstm_mean_rollout_error_km"]
    ].min(axis=1)
    candidates["direct_margin_km"] = candidates["direct_mean_rollout_error_km"] - candidates[
        ["lstm_mean_rollout_error_km", "transformer_mean_rollout_error_km"]
    ].min(axis=1)

    recipes: list[tuple[pd.DataFrame, str]] = [
        (
            candidates.sort_values(["best_model_mean_error_km", "rollout_steps"], ascending=[True, False]),
            "clear_success_lowest_mean_error",
        ),
        (
            candidates.sort_values(["lstm_margin_km", "lstm_mean_rollout_error_km"], ascending=[True, True]),
            "lstm_best_or_nearest_best",
        ),
        (
            candidates.sort_values(
                ["transformer_margin_km", "transformer_mean_rollout_error_km"], ascending=[True, True]
            ),
            "transformer_best_or_nearest_best",
        ),
        (
            candidates.sort_values(["direct_margin_km", "direct_mean_rollout_error_km"], ascending=[True, True]),
            "direct_best_or_nearest_best",
        ),
    ]

    if not contrast.empty:
        condition_col = "with_minus_without_mean_km" if condition == "with_weather" else "without_minus_with_mean_km"
        contrast_rows = contrast.sort_values(condition_col, key=lambda s: s.abs(), ascending=False)
        contrast_payloads: list[dict[str, object]] = []
        for _, row in contrast_rows.iterrows():
            match = candidates[candidates["path_id"].astype(str).eq(str(row["path_id"]))]
            if not match.empty:
                payload = match.iloc[0].to_dict()
                payload["weather_contrast_abs_km"] = abs(float(row[condition_col]))
                contrast_payloads.append(payload)
        if contrast_payloads:
            recipes.append((pd.DataFrame(contrast_payloads), "largest_weather_vs_noweather_contrast"))

    recipes.extend(
        [
            (
                candidates.sort_values(["all_model_mean_error_km", "rollout_steps"], ascending=[False, False]),
                "hard_failure_high_average_error",
            ),
            (
                candidates.sort_values(["rollout_steps", "best_model_mean_error_km"], ascending=[False, True]),
                "long_rollout_story",
            ),
        ]
    )

    selected: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    for frame, reason in recipes:
        if len(selected) >= max_cases:
            break
        row = select_first_unique(frame, selected_keys, reason)
        if row is not None:
            selected.append(row)
    return selected[:max_cases]


def slugify(value: str, max_len: int = 110) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    text = "".join(safe).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text[:max_len] or "rollout_case"


def finite_bounds(values: list[float], pad_fraction: float = 0.08) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    low = float(np.min(arr))
    high = float(np.max(arr))
    if math.isclose(low, high):
        low -= 0.25
        high += 0.25
    pad = max((high - low) * pad_fraction, 0.05)
    return low - pad, high + pad


def case_rows(predictions: pd.DataFrame, path_id: str) -> pd.DataFrame:
    rows = predictions[predictions["path_id"].astype(str).eq(path_id)].copy()
    rows["step_index"] = pd.to_numeric(rows["step_index"], errors="coerce")
    return rows.sort_values("step_index").reset_index(drop=True)


def frame_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy())


def model_points(rows: pd.DataFrame, prefix: str, context_days: int, day_count: int) -> tuple[np.ndarray, np.ndarray]:
    idx = list(range(context_days - 1, context_days + day_count))
    return (
        rows.loc[idx, f"{prefix}_lon"].astype(float).to_numpy(),
        rows.loc[idx, f"{prefix}_lat"].astype(float).to_numpy(),
    )


def render_frame(
    *,
    rows: pd.DataFrame,
    summary: pd.Series,
    condition: str,
    selection_reason: str,
    day_count: int,
    total_days: int,
    bounds: tuple[float, float, float, float],
) -> Image.Image:
    context_days = int(summary["observed_days"])
    rollout_idx = context_days + day_count - 1
    visible_context_start = max(0, context_days - 30)
    obs = rows.iloc[visible_context_start:context_days]
    true_future = rows.iloc[context_days - 1 : context_days + day_count]
    current = rows.iloc[rollout_idx]

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=110)
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, color="#d9ded8", linewidth=0.7, alpha=0.75)
    ax.plot(obs["true_lon"], obs["true_lat"], color="#96a09a", linewidth=2.0, label="Observed context")
    ax.scatter(
        [float(rows.iloc[context_days - 1]["true_lon"])],
        [float(rows.iloc[context_days - 1]["true_lat"])],
        color="#2f6f4e",
        s=58,
        zorder=6,
        label="Forecast anchor",
    )
    ax.plot(
        true_future["true_lon"],
        true_future["true_lat"],
        color=MODEL_STYLES["truth"]["color"],
        linewidth=MODEL_STYLES["truth"]["linewidth"],
        label="True future",
    )
    for prefix in ["direct", "lstm", "transformer"]:
        lon, lat = model_points(rows, prefix, context_days, day_count)
        style = MODEL_STYLES[prefix]
        ax.plot(
            lon,
            lat,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        ax.scatter([lon[-1]], [lat[-1]], color=style["color"], s=46, edgecolors="white", linewidth=0.8, zorder=7)

    ax.scatter(
        [float(current["true_lon"])],
        [float(current["true_lat"])],
        color="#111111",
        s=52,
        marker="x",
        linewidth=2.0,
        zorder=8,
    )
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{condition}: {summary['source_bird_id']} / {summary['path_year']} / rollout day {day_count} of {total_days}",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        (
            f"{selection_reason}\n"
            f"Direct {metric_value(current, 'direct_error_km'):.1f} km | "
            f"LSTM {metric_value(current, 'lstm_error_km'):.1f} km | "
            f"Transformer {metric_value(current, 'transformer_error_km'):.1f} km"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#111827",
        bbox={"facecolor": "#ffffff", "edgecolor": "#d1d5db", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
    )
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()
    image = frame_to_image(fig)
    plt.close(fig)
    return image


def write_gifs(
    *,
    condition: str,
    output_dir: Path,
    cases: list[dict[str, object]],
    gif_days: int,
) -> list[dict[str, object]]:
    if not cases:
        return []
    predictions = pd.read_csv(output_dir / "rollout_all_predictions.csv")
    summary = pd.read_csv(output_dir / "rollout_all_summary.csv")
    gif_dir = output_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for case_number, case in enumerate(cases, start=1):
        path_id = str(case["path_id"])
        summary_match = summary[summary["path_id"].astype(str).eq(path_id)]
        if summary_match.empty:
            continue
        summary_row = summary_match.iloc[0]
        rows = case_rows(predictions, path_id)
        if rows.empty:
            continue
        total_days = min(int(summary_row["rollout_steps"]), int(gif_days))
        if total_days <= 0:
            continue
        max_idx = int(summary_row["observed_days"]) + total_days
        visible = rows.iloc[:max_idx]
        lon_values = []
        lat_values = []
        for prefix in ["true", "direct", "lstm", "transformer"]:
            lon_values.extend(visible[f"{prefix}_lon"].astype(float).tolist())
            lat_values.extend(visible[f"{prefix}_lat"].astype(float).tolist())
        bounds = (*finite_bounds(lon_values), *finite_bounds(lat_values))
        frames = [
            render_frame(
                rows=rows,
                summary=summary_row,
                condition=condition,
                selection_reason=str(case.get("selection_reason", "")),
                day_count=day,
                total_days=total_days,
                bounds=bounds,
            )
            for day in range(1, total_days + 1)
        ]
        filename = (
            f"{case_number:02d}_{slugify(str(case.get('selection_reason', 'case')), 42)}_"
            f"{slugify(path_id, 52)}.gif"
        )
        gif_path = gif_dir / filename
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=550, loop=0)
        manifest.append(
            {
                "condition": condition,
                "path_id": path_id,
                "selection_reason": str(case.get("selection_reason", "")),
                "gif": str(gif_path),
                "frames": int(len(frames)),
            }
        )

    write_csv(gif_dir / "gif_manifest.csv", manifest)
    return manifest


def aggregate_condition(label: str, summary: pd.DataFrame, skipped_count: int) -> dict[str, object]:
    metrics = [
        "direct_mean_rollout_error_km",
        "direct_final_rollout_error_km",
        "lstm_mean_rollout_error_km",
        "lstm_final_rollout_error_km",
        "transformer_mean_rollout_error_km",
        "transformer_final_rollout_error_km",
        "persistence_mean_rollout_error_km",
        "const_velocity_mean_rollout_error_km",
    ]
    numeric_summary = numeric(summary, metrics)
    row: dict[str, object] = {
        "condition": label,
        "paths": int(len(numeric_summary)),
        "skipped_paths": int(skipped_count),
    }
    for metric in metrics:
        row[metric] = float(numeric_summary[metric].mean()) if metric in numeric_summary else None
    return row


def create_comparison(output_root: Path) -> pd.DataFrame:
    with_summary = pd.read_csv(output_root / "with_weather" / "rollout_all_summary.csv")
    without_summary = pd.read_csv(output_root / "without_weather" / "rollout_all_summary.csv")
    rows: list[dict[str, object]] = []
    for model in ["direct", "lstm", "transformer"]:
        metric = f"{model}_mean_rollout_error_km"
        final_metric = f"{model}_final_rollout_error_km"
        merged = with_summary[["path_id", metric, final_metric]].merge(
            without_summary[["path_id", metric, final_metric]],
            on="path_id",
            suffixes=("_with_weather", "_without_weather"),
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "path_id": row["path_id"],
                    "model": model,
                    "with_mean_rollout_error_km": row[f"{metric}_with_weather"],
                    "without_mean_rollout_error_km": row[f"{metric}_without_weather"],
                    "with_minus_without_mean_km": row[f"{metric}_with_weather"]
                    - row[f"{metric}_without_weather"],
                    "without_minus_with_mean_km": row[f"{metric}_without_weather"]
                    - row[f"{metric}_with_weather"],
                    "with_final_rollout_error_km": row[f"{final_metric}_with_weather"],
                    "without_final_rollout_error_km": row[f"{final_metric}_without_weather"],
                    "with_minus_without_final_km": row[f"{final_metric}_with_weather"]
                    - row[f"{final_metric}_without_weather"],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "rollout_error_comparison.csv", index=False)
    return comparison


def aggregate_comparison(output_root: Path) -> pd.DataFrame:
    with_summary = pd.read_csv(output_root / "with_weather" / "rollout_all_summary.csv")
    without_summary = pd.read_csv(output_root / "without_weather" / "rollout_all_summary.csv")
    skipped_with = pd.read_csv(output_root / "with_weather" / "rollout_all_skipped.csv")
    skipped_without = pd.read_csv(output_root / "without_weather" / "rollout_all_skipped.csv")
    rows = [
        aggregate_condition("with_weather", with_summary, len(skipped_with)),
        aggregate_condition("without_weather", without_summary, len(skipped_without)),
    ]
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(output_root / "rollout_aggregate_summary.csv", index=False)
    return aggregate


def format_km(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(number):
        return "n/a"
    return f"{float(number):.2f}"


def build_analysis_section(output_root: Path, aggregate: pd.DataFrame) -> str:
    by_condition = {str(row["condition"]): row for _, row in aggregate.iterrows()}
    with_row = by_condition.get("with_weather", {})
    without_row = by_condition.get("without_weather", {})
    model_rows = []
    for label, prefix in [("Direct", "direct"), ("LSTM", "lstm"), ("Transformer", "transformer")]:
        with_mean = with_row.get(f"{prefix}_mean_rollout_error_km")
        without_mean = without_row.get(f"{prefix}_mean_rollout_error_km")
        with_final = with_row.get(f"{prefix}_final_rollout_error_km")
        without_final = without_row.get(f"{prefix}_final_rollout_error_km")
        delta = pd.to_numeric(pd.Series([with_mean]), errors="coerce").iloc[0] - pd.to_numeric(
            pd.Series([without_mean]), errors="coerce"
        ).iloc[0]
        model_rows.append(
            f"| {label} | {format_km(with_mean)} | {format_km(without_mean)} | {format_km(delta)} | "
            f"{format_km(with_final)} | {format_km(without_final)} |"
        )

    deltas = []
    for prefix in ["direct", "lstm", "transformer"]:
        with_mean = pd.to_numeric(pd.Series([with_row.get(f"{prefix}_mean_rollout_error_km")]), errors="coerce").iloc[0]
        without_mean = pd.to_numeric(
            pd.Series([without_row.get(f"{prefix}_mean_rollout_error_km")]), errors="coerce"
        ).iloc[0]
        if np.isfinite(with_mean) and np.isfinite(without_mean):
            deltas.append(with_mean - without_mean)
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    if np.isfinite(mean_delta):
        if mean_delta < -0.25:
            interpretation = "Weather improves average rollout error across the selected model families."
        elif mean_delta > 0.25:
            interpretation = "Weather slightly worsens average rollout error across the selected model families."
        else:
            interpretation = "Weather has little aggregate effect on rollout stability in this evaluation."
    else:
        interpretation = "Weather impact could not be summarized because one condition had no valid rollout paths."

    return (
        "\n---\n\n"
        "## Rollout Evaluation\n\n"
        f"Autoregressive rollouts use a minimum {ROLLOUT_CONTEXT_DAYS}-day observed context and extend that context "
        f"until observed displacement reaches {ROLLOUT_MIN_DISPLACEMENT_KM:.0f} km when possible. Each model then "
        "rolls forward for the rest of the path using its trained trailing context window.\n\n"
        "| Model | With weather mean km | Without weather mean km | With - without mean km | "
        "With weather final km | Without weather final km |\n"
        "|-------|---------------------:|------------------------:|-----------------------:|"
        "----------------------:|-------------------------:|\n"
        + "\n".join(model_rows)
        + "\n\n"
        f"Valid rollout paths: with weather {int(with_row.get('paths', 0))}, without weather "
        f"{int(without_row.get('paths', 0))}. Skipped paths: with weather "
        f"{int(with_row.get('skipped_paths', 0))}, without weather {int(without_row.get('skipped_paths', 0))}.\n\n"
        f"{interpretation}\n\n"
        "Artifacts:\n"
        f"- Per-step and per-path errors: `{output_root / 'with_weather'}`, `{output_root / 'without_weather'}`\n"
        f"- Weather comparison CSV: `{output_root / 'rollout_error_comparison.csv'}`\n"
        f"- Representative GIFs: `{output_root / 'with_weather' / 'gifs'}`, `{output_root / 'without_weather' / 'gifs'}`\n"
    )


def update_analysis_md(analysis_path: Path, section: str) -> None:
    text = analysis_path.read_text(encoding="utf-8")
    marker = "\n---\n\n## Rollout Evaluation\n"
    idx = text.find(marker)
    if idx >= 0:
        text = text[:idx].rstrip() + section
    else:
        text = text.rstrip() + section
    analysis_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run current with/without weather rollout GIF evaluation.")
    parser.add_argument("--input-csv", default=str(ROOT / "data" / "combined_southbound_paths_with_weather_matched.csv"))
    parser.add_argument("--output-root", default=str(ROOT / "rollout"))
    parser.add_argument("--with-weather-root", default=str(ROOT / "with_weather"))
    parser.add_argument("--without-weather-root", default=str(ROOT / "without_weather"))
    parser.add_argument("--analysis-md", default=str(ROOT / "analysis.md"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-gif-cases", type=int, default=DEFAULT_MAX_GIF_CASES)
    parser.add_argument("--gif-days", type=int, default=DEFAULT_GIF_DAYS)
    args = parser.parse_args()

    device = choose_device(args.device)
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    df = pd.read_csv(input_csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")

    output_root = Path(args.output_root)
    with_index = evaluate_condition(
        label="with_weather",
        runner=with_weather,
        df=df,
        result_dir=Path(args.with_weather_root),
        output_dir=output_root / "with_weather",
        device=device,
    )
    without_index = evaluate_condition(
        label="without_weather",
        runner=no_weather,
        df=df,
        result_dir=Path(args.without_weather_root),
        output_dir=output_root / "without_weather",
        device=device,
    )

    comparison = create_comparison(output_root)
    aggregate = aggregate_comparison(output_root)

    contrast_by_path = (
        comparison.groupby("path_id", as_index=False)
        .agg(
            with_minus_without_mean_km=("with_minus_without_mean_km", "mean"),
            without_minus_with_mean_km=("without_minus_with_mean_km", "mean"),
        )
    )
    with_cases = select_gif_cases(
        pd.read_csv(output_root / "with_weather" / "rollout_all_summary.csv"),
        contrast_by_path,
        "with_weather",
        int(args.max_gif_cases),
    )
    without_cases = select_gif_cases(
        pd.read_csv(output_root / "without_weather" / "rollout_all_summary.csv"),
        contrast_by_path,
        "without_weather",
        int(args.max_gif_cases),
    )
    with_gifs = write_gifs(
        condition="with_weather",
        output_dir=output_root / "with_weather",
        cases=with_cases,
        gif_days=int(args.gif_days),
    )
    without_gifs = write_gifs(
        condition="without_weather",
        output_dir=output_root / "without_weather",
        cases=without_cases,
        gif_days=int(args.gif_days),
    )

    index = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "device": str(device),
        "setup": SETUP_NAME,
        "min_observed_days": ROLLOUT_CONTEXT_DAYS,
        "min_observed_displacement_km": ROLLOUT_MIN_DISPLACEMENT_KM,
        "with_weather": with_index,
        "without_weather": without_index,
        "with_weather_gifs": with_gifs,
        "without_weather_gifs": without_gifs,
        "comparison_csv": str(output_root / "rollout_error_comparison.csv"),
        "aggregate_csv": str(output_root / "rollout_aggregate_summary.csv"),
    }
    write_json(output_root / "rollout_index.json", index)
    update_analysis_md(Path(args.analysis_md), build_analysis_section(output_root, aggregate))
    print(json.dumps(index, indent=2, default=str))


if __name__ == "__main__":
    main()
