from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import run_path_experiment_no_weather as no_weather
import run_path_experiment_with_weather as with_weather
from run_current_rollout_gifs import (
    DEFAULT_GIF_DAYS,
    DEFAULT_MAX_GIF_CASES,
    FLY_THRESHOLD_KM,
    ROLLOUT_CONTEXT_DAYS,
    ROLLOUT_MIN_DISPLACEMENT_KM,
    SETUP_NAME,
    aggregate_comparison,
    choose_device,
    create_comparison,
    select_gif_cases,
    write_csv,
    write_gifs,
    write_json,
)


PREDICTION_MODE = "true_context"


def release_torch_objects(*objects: object) -> None:
    del objects
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def true_context_direct_model(
    *,
    runner: Any,
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    path_df: pd.DataFrame,
    context_days: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    model_spec = dict(checkpoint["model_spec"])
    k = int(model_spec["k"])
    pred_lat = path_df["lat_median"].astype(float).iloc[:context_days].tolist()
    pred_lon = path_df["lon_median"].astype(float).iloc[:context_days].tolist()
    bird_to_idx = checkpoint["bird_to_idx"]
    bird_name = str(path_df.iloc[0][runner.IDENTITY_COLUMN])
    bird_id = int(bird_to_idx.get(bird_name, 0))
    target_mean = np.asarray(checkpoint["target_delta_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_delta_std"], dtype=np.float32)

    with torch.no_grad():
        for target_index in range(context_days, len(path_df)):
            context_rows = runner.dataframe_context_rows(path_df, target_index)
            features = runner.normalize_model_window(runner.context_feature_window(context_rows, k), checkpoint)
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            bird_tensor = torch.as_tensor([bird_id], dtype=torch.long, device=device)
            pred_norm = model(feature_tensor, bird_tensor).detach().cpu().numpy()[0]
            delta = pred_norm * target_std + target_mean
            previous = path_df.iloc[target_index - 1]
            pred_lat.append(float(previous["lat_median"]) + float(delta[0]))
            pred_lon.append(float(previous["lon_median"]) + float(delta[1]))
    return pred_lat, pred_lon


def true_context_triline_model(
    *,
    runner: Any,
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    path_df: pd.DataFrame,
    context_days: int,
    device: torch.device,
) -> tuple[list[float], list[float], list[float]]:
    model_spec = dict(checkpoint["model_spec"])
    k = int(model_spec["k"])
    pred_lat = path_df["lat_median"].astype(float).iloc[:context_days].tolist()
    pred_lon = path_df["lon_median"].astype(float).iloc[:context_days].tolist()
    fly_probs = [float("nan")] * context_days
    bird_to_idx = checkpoint["bird_to_idx"]
    bird_name = str(path_df.iloc[0][runner.IDENTITY_COLUMN])
    bird_id = int(bird_to_idx.get(bird_name, 0))

    with torch.no_grad():
        for target_index in range(context_days, len(path_df)):
            context_rows = runner.dataframe_context_rows(path_df, target_index)
            features = runner.normalize_model_window(runner.context_feature_window(context_rows, k), checkpoint)
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            bird_tensor = torch.as_tensor([bird_id], dtype=torch.long, device=device)
            outputs = model(feature_tensor, bird_tensor)
            fly_prob = float(torch.sigmoid(outputs["fly_logit"]).detach().cpu().numpy()[0])
            distance = float(torch.expm1(outputs["log_distance"]).clamp_min(0.0).detach().cpu().numpy()[0])
            direction = outputs["direction"].detach().cpu().numpy()
            previous = path_df.iloc[target_index - 1]
            next_lat_arr, next_lon_arr = runner.reconstruct_from_distance_heading(
                np.array([float(previous["lat_median"])]),
                np.array([float(previous["lon_median"])]),
                np.array([distance]),
                direction,
            )
            pred_lat.append(float(next_lat_arr[0]))
            pred_lon.append(float(next_lon_arr[0]))
            fly_probs.append(fly_prob)
    return pred_lat, pred_lon, fly_probs


def true_context_baselines(
    path_df: pd.DataFrame,
    context_days: int,
) -> dict[str, tuple[list[float], list[float]]]:
    true_lat = path_df["lat_median"].astype(float).tolist()
    true_lon = path_df["lon_median"].astype(float).tolist()
    persistence_lat = true_lat[:context_days]
    persistence_lon = true_lon[:context_days]
    velocity_lat = true_lat[:context_days]
    velocity_lon = true_lon[:context_days]

    for target_index in range(context_days, len(path_df)):
        previous = path_df.iloc[target_index - 1]
        before_previous = path_df.iloc[target_index - 2]
        persistence_lat.append(float(previous["lat_median"]))
        persistence_lon.append(float(previous["lon_median"]))
        velocity_lat.append(float(previous["lat_median"]) + float(previous["lat_median"] - before_previous["lat_median"]))
        velocity_lon.append(float(previous["lon_median"]) + float(previous["lon_median"] - before_previous["lon_median"]))
    return {
        "persistence": (persistence_lat, persistence_lon),
        "const_velocity": (velocity_lat, velocity_lon),
    }


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
                    "prediction_mode": PREDICTION_MODE,
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
                    "prediction_mode": PREDICTION_MODE,
                    "setup": SETUP_NAME,
                    "path_id": path_id,
                    "reason": "no_prediction_days_after_60_day_50km_observed_context",
                    "rows": int(len(path_df)),
                    "observed_days": int(context_days),
                    "observed_displacement_km": float(observed_displacement_km),
                }
            )
            continue

        direct_lat, direct_lon = true_context_direct_model(
            runner=runner,
            model=direct_model,
            checkpoint=direct_checkpoint,
            path_df=path_df,
            context_days=context_days,
            device=device,
        )
        lstm_lat, lstm_lon, lstm_fly_prob = true_context_triline_model(
            runner=runner,
            model=lstm_model,
            checkpoint=lstm_checkpoint,
            path_df=path_df,
            context_days=context_days,
            device=device,
        )
        transformer_lat, transformer_lon, transformer_fly_prob = true_context_triline_model(
            runner=runner,
            model=transformer_model,
            checkpoint=transformer_checkpoint,
            path_df=path_df,
            context_days=context_days,
            device=device,
        )
        baselines = true_context_baselines(path_df, context_days)
        rollout_steps = len(path_df) - context_days

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
            row["prediction_mode"] = PREDICTION_MODE
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
                "prediction_mode": PREDICTION_MODE,
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
        "prediction_mode": PREDICTION_MODE,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run true-context current with/without weather rollout GIF evaluation.")
    parser.add_argument("--input-csv", default=str(ROOT / "data" / "combined_southbound_paths_with_weather_matched.csv"))
    parser.add_argument("--output-root", default=str(ROOT / "roll_out_trueContext"))
    parser.add_argument("--with-weather-root", default=str(ROOT / "with_weather"))
    parser.add_argument("--without-weather-root", default=str(ROOT / "without_weather"))
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
        condition="with_weather true-context",
        output_dir=output_root / "with_weather",
        cases=with_cases,
        gif_days=int(args.gif_days),
    )
    without_gifs = write_gifs(
        condition="without_weather true-context",
        output_dir=output_root / "without_weather",
        cases=without_cases,
        gif_days=int(args.gif_days),
    )

    index = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "device": str(device),
        "setup": SETUP_NAME,
        "prediction_mode": PREDICTION_MODE,
        "min_observed_days": ROLLOUT_CONTEXT_DAYS,
        "min_observed_displacement_km": ROLLOUT_MIN_DISPLACEMENT_KM,
        "with_weather": with_index,
        "without_weather": without_index,
        "with_weather_gifs": with_gifs,
        "without_weather_gifs": without_gifs,
        "comparison_csv": str(output_root / "rollout_error_comparison.csv"),
        "aggregate_csv": str(output_root / "rollout_aggregate_summary.csv"),
        "note": "Future-day predictions use true preceding context and are not fed back autoregressively.",
    }
    write_json(output_root / "rollout_index.json", index)
    print(json.dumps(index, indent=2, default=str))


if __name__ == "__main__":
    main()
