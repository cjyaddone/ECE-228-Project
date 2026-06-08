from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "Final_Path_Experiment"
TRILINE_ROOT = ROOT / "ECE-228-Project" / "model_files"
CLEANED_WEATHER_SOURCE = (
    ROOT / "ECE-228-Project" / "data" / "combined_southbound_paths_with_weather_matched.csv"
)
CLEANED_WEATHER_COPY = FINAL_ROOT / "data" / "combined_southbound_paths_with_weather_matched.csv"
SETUP_NAME = "fly_threshold_10km"
FLY_THRESHOLD_KM = 10.0
K_VALUES = [7, 14, 30]
SEED = 42


def load_module(module_name: str, path: Path) -> Any:
    if str(TRILINE_ROOT) not in sys.path:
        sys.path.insert(0, str(TRILINE_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def model_specs(smoke: bool) -> list[dict[str, object]]:
    k_values = [7] if smoke else K_VALUES
    specs: list[dict[str, object]] = []
    for k in k_values:
        specs.extend(
            [
                {
                    "name": f"direct_mlp_sequence_k{k}",
                    "kind": "direct_mlp_sequence",
                    "k": k,
                },
                {
                    "name": f"direct_transformer_4l_k{k}",
                    "kind": "direct_transformer",
                    "k": k,
                    "n_layers": 4,
                },
                {
                    "name": f"triline_lstm_4l_k{k}",
                    "kind": "triline_lstm",
                    "k": k,
                    "n_layers": 4,
                },
                {
                    "name": f"triline_transformer_v2_k{k}",
                    "kind": "triline_transformer",
                    "k": k,
                    "n_layers": 4,
                },
            ]
        )
    return specs


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


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
        for row in rows:
            writer.writerow(row)


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


def copy_inputs_and_sources() -> None:
    if not CLEANED_WEATHER_SOURCE.exists():
        raise FileNotFoundError(f"Missing cleaned weather data: {CLEANED_WEATHER_SOURCE}")
    CLEANED_WEATHER_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_WEATHER_SOURCE, CLEANED_WEATHER_COPY)

    model_files = FINAL_ROOT / "model_files"
    model_files.mkdir(parents=True, exist_ok=True)
    for source in [
        TRILINE_ROOT / "model.py",
        TRILINE_ROOT / "run_path_experiment_with_weather.py",
        TRILINE_ROOT / "run_path_experiment_no_weather.py",
        Path(__file__).resolve(),
    ]:
        shutil.copy2(source, model_files / source.name)


def apply_dataset2_precipitation_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure precipitation columns are numeric and add log1p transform.

    The combined CSV from ``data/combined_southbound_paths_with_weather_matched.csv``
    already contains correctly-scaled precipitation values so no Dataset-2-specific
    correction is needed at this point.
    """
    out = df.copy()
    out["precipitation_mm"] = pd.to_numeric(out["precipitation_mm"], errors="coerce").fillna(0.0)
    out["precipitation_mm"] = out["precipitation_mm"].clip(lower=0.0)
    out["precipitation_log1p_mm"] = np.log1p(out["precipitation_mm"])
    return out


def run_one_test(
    *,
    label: str,
    runner: Any,
    df: pd.DataFrame,
    output_dir: Path,
    specs: list[dict[str, object]],
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> list[dict[str, object]]:
    setup_dir = output_dir / SETUP_NAME
    setup_dir.mkdir(parents=True, exist_ok=True)
    direct_rows: list[dict[str, object]] = []
    triline_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    cache: dict[int, tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    logging.info("%s: building windows for threshold %.1f km", label, FLY_THRESHOLD_KM)
    for k in sorted({int(spec["k"]) for spec in specs}):
        window_data = runner.build_windows(df, k=k, fly_threshold_km=FLY_THRESHOLD_KM)
        if len(window_data.features) == 0:
            logging.warning("%s: k=%s has no windows; skipping", label, k)
            continue
        train_idx, test_idx = runner.make_chronological_split(window_data)
        feature_mean, feature_std = runner.fit_normalizer(window_data.features, train_idx)
        cache[k] = (window_data, train_idx, test_idx, feature_mean, feature_std)

    for spec in specs:
        k = int(spec["k"])
        if k not in cache:
            continue
        window_data, train_idx, test_idx, feature_mean, feature_std = cache[k]
        normalized_features = runner.normalize_features(window_data.features, feature_mean, feature_std)
        normalized_for_model = runner.select_feature_mode(
            normalized_features,
            window_data.feature_columns,
            spec.get("feature_mode", "full"),
        )
        model_parent = setup_dir / f"k_{k}"
        start = time.time()
        logging.info("%s: training %s", label, spec["name"])
        if str(spec["kind"]).startswith("direct"):
            row = runner.train_direct_model(
                model_spec=spec,
                window_data=window_data,
                train_idx=train_idx,
                test_idx=test_idx,
                normalized_features=normalized_for_model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                output_dir=model_parent,
                fly_threshold_km=FLY_THRESHOLD_KM,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
            )
            direct_rows.append(row)
        else:
            row = runner.train_triline_model(
                model_spec=spec,
                window_data=window_data,
                train_idx=train_idx,
                test_idx=test_idx,
                normalized_features=normalized_for_model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                output_dir=model_parent,
                fly_threshold_km=FLY_THRESHOLD_KM,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
            )
            triline_rows.append(row)
        row.update({"setup": SETUP_NAME, "test": label, "runtime_seconds_final_runner": time.time() - start})
        all_rows.append(row)

    write_csv(output_dir / "direct_summary.csv", direct_rows)
    write_csv(output_dir / "triline_summary.csv", triline_rows)
    write_csv(output_dir / "comparison_summary.csv", all_rows)
    write_csv(setup_dir / "direct_summary.csv", direct_rows)
    write_csv(setup_dir / "triline_summary.csv", triline_rows)
    write_csv(setup_dir / "comparison_summary.csv", all_rows)
    return all_rows


def summarize_data(df: pd.DataFrame, feature_columns: list[str], weather_data: bool) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "paths": int(df["path_id"].nunique()),
        "source_birds": int(df["source_individual_local_identifier"].nunique()),
        "date_min": str(pd.to_datetime(df["date"], utc=True, format="mixed").min()),
        "date_max": str(pd.to_datetime(df["date"], utc=True, format="mixed").max()),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "weather_data": weather_data,
        "setup": {"name": SETUP_NAME, "fly_threshold_km": FLY_THRESHOLD_KM},
    }


def write_run_config(
    output_dir: Path,
    *,
    label: str,
    specs: list[dict[str, object]],
    max_epochs: int,
    patience: int,
    device: torch.device,
    weather_data: bool,
) -> None:
    write_json(
        output_dir / "run_config.json",
        {
            "seed": SEED,
            "test": label,
            "k_values": K_VALUES,
            "smoke_k_values": [7],
            "fly_threshold_km": FLY_THRESHOLD_KM,
            "setup_name": SETUP_NAME,
            "max_epochs": max_epochs,
            "patience": patience,
            "device": str(device),
            "weather_data": weather_data,
            "cleaned_weather_csv": str(CLEANED_WEATHER_COPY),
            "model_specs": specs,
        },
    )


def write_analysis(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    lines = [f"# {title}", "", f"- Setup: `{SETUP_NAME}`.", f"- Fly threshold: {FLY_THRESHOLD_KM:.1f} km.", ""]
    if not rows:
        lines.append("No rows produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    lines.extend(
        [
            "## Results",
            "",
            "| Model | Family | k | Mean km | Median km | P90 km | Fly recall |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in df.sort_values(["k", "model"]).iterrows():
        fly_recall = row.get("fly_recall", "")
        fly_recall_text = "" if fly_recall == "" or pd.isna(fly_recall) else f"{float(fly_recall):.4f}"
        lines.append(
            f"| {row['model']} | {row['model_family']} | {int(row['k'])} | "
            f"{float(row['mean_error_km']):.4f} | {float(row['median_error_km']):.4f} | "
            f"{float(row['p90_error_km']):.4f} | {fly_recall_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_weather_comparison(
    output_path: Path,
    weather_rows: list[dict[str, object]],
    noweather_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not weather_rows or not noweather_rows:
        write_csv(output_path, [])
        return []
    weather_df = pd.DataFrame(weather_rows)
    noweather_df = pd.DataFrame(noweather_rows)
    join_keys = ["setup", "model_family", "model", "kind", "k"]
    merged = weather_df.merge(
        noweather_df,
        on=join_keys,
        how="inner",
        suffixes=("_weather", "_noweather"),
    )
    metrics = [
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
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        payload: dict[str, object] = {key: row[key] for key in join_keys}
        for metric in metrics:
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
    write_csv(output_path, rows)
    return rows


def validate_outputs(output_dir: Path, expected_count: int) -> dict[str, object]:
    comparison = pd.read_csv(output_dir / "comparison_summary.csv")
    setup_values = sorted(comparison["setup_fly_threshold_km"].dropna().unique().tolist())
    k_values = sorted(comparison["k"].dropna().astype(int).unique().tolist())
    model_dirs = [
        output_dir / SETUP_NAME / f"k_{int(row['k'])}" / str(row["model"])
        for _, row in comparison.iterrows()
    ]
    missing_files: list[str] = []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            missing_files.append(str(model_dir))
            continue
        for filename in ["best_model.pt", "metrics.json", "predictions.csv", "training_log.csv"]:
            if not (model_dir / filename).exists():
                missing_files.append(str(model_dir / filename))
    return {
        "output_dir": str(output_dir),
        "rows": int(len(comparison)),
        "model_dir_count": len(model_dirs),
        "expected_model_dir_count": expected_count,
        "setup_fly_threshold_km_values": setup_values,
        "k_values": k_values,
        "missing_required_files": missing_files,
        "passed": (
            len(comparison) == expected_count
            and len(model_dirs) == expected_count
            and setup_values == [FLY_THRESHOLD_KM]
            and k_values == ([7] if expected_count == 4 else K_VALUES)
            and not missing_files
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final simplified path experiment matrix.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = args.mode == "smoke"
    max_epochs = 1 if smoke else int(args.max_epochs)
    patience = 1 if smoke else int(args.patience)
    specs = model_specs(smoke)
    expected_count = len(specs)

    setup_logging(FINAL_ROOT / f"run_{args.mode}.log")
    device = choose_device(args.device, smoke)
    logging.info("Using device: %s", device)

    copy_inputs_and_sources()
    with_weather = load_module(
        "final_with_weather_runner",
        TRILINE_ROOT / "run_path_experiment_with_weather.py",
    )
    no_weather = load_module(
        "final_no_weather_runner",
        TRILINE_ROOT / "run_path_experiment_no_weather.py",
    )
    with_weather.set_seed(SEED)
    no_weather.set_seed(SEED)

    df = pd.read_csv(CLEANED_WEATHER_COPY)
    df = apply_dataset2_precipitation_fix(df)
    df.to_csv(CLEANED_WEATHER_COPY, index=False)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    with_weather_dir = FINAL_ROOT / "with_weather"
    no_weather_dir = FINAL_ROOT / "without_weather"

    write_run_config(
        with_weather_dir,
        label="with_weather",
        specs=specs,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        weather_data=True,
    )
    write_run_config(
        no_weather_dir,
        label="without_weather",
        specs=specs,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        weather_data=False,
    )
    write_json(
        with_weather_dir / "data_summary.json",
        summarize_data(df, with_weather.COMPACT_FEATURE_COLUMNS, weather_data=True),
    )
    write_json(
        no_weather_dir / "data_summary.json",
        summarize_data(df, no_weather.COMPACT_FEATURE_COLUMNS, weather_data=False),
    )

    weather_rows = run_one_test(
        label="with_weather",
        runner=with_weather,
        df=df,
        output_dir=with_weather_dir,
        specs=specs,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    noweather_rows = run_one_test(
        label="without_weather",
        runner=no_weather,
        df=df,
        output_dir=no_weather_dir,
        specs=specs,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )

    write_analysis(with_weather_dir / "analysis.md", "Final Path Experiment With Weather", weather_rows)
    write_analysis(no_weather_dir / "analysis.md", "Final Path Experiment Without Weather", noweather_rows)
    comparison_rows = write_weather_comparison(
        FINAL_ROOT / "weather_vs_noweather_comparison.csv",
        weather_rows,
        noweather_rows,
    )
    write_json(
        FINAL_ROOT / f"validation_{args.mode}.json",
        {
            "mode": args.mode,
            "with_weather": validate_outputs(with_weather_dir, expected_count),
            "without_weather": validate_outputs(no_weather_dir, expected_count),
            "weather_vs_noweather_rows": len(comparison_rows),
        },
    )
    (FINAL_ROOT / f"COMPLETED_{args.mode.upper()}.txt").write_text(
        f"Done. {args.mode} results saved to {FINAL_ROOT}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
