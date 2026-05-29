from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from data_preprocessing import (
    build_windows,
    fit_normalizer,
    make_split,
    normalize_features,
    preprocess_movement_data,
)
from model import TrilineTransformer
from run_original_flynofly import (
    preprocess_original,
    train_one_k as train_original_one_k,
    write_analysis as write_original_analysis,
    write_csv as write_original_csv,
)
from train_test_fly_nofly import (
    CLEANED_CSV,
    FLY_THRESHOLD_KM,
    INPUT_CSV,
    K_VALUES,
    ROOT,
    SEED,
    SPLIT_MODES,
    run_baselines,
    set_seed,
    train_network_one_split,
    write_analysis as write_improved_analysis,
    write_csv as write_improved_csv,
)


COMPACT_FEATURE_COLUMNS = [
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

VARIANTS = [
    ("compact_with_bird_id", True),
    ("compact_without_bird_id", False),
]


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def write_metadata(output_dir: Path, setup: str, variant: str, use_bird_id: bool) -> None:
    payload = {
        "setup": setup,
        "variant": variant,
        "use_bird_id": use_bird_id,
        "n_features": len(COMPACT_FEATURE_COLUMNS),
        "feature_columns": COMPACT_FEATURE_COLUMNS,
        "fly_threshold_km": FLY_THRESHOLD_KM,
        "k_values": K_VALUES,
    }
    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def smoke_check(cleaned_df: pd.DataFrame) -> None:
    window_data = build_windows(
        cleaned_df,
        k=K_VALUES[0],
        fly_threshold_km=FLY_THRESHOLD_KM,
        feature_columns=COMPACT_FEATURE_COLUMNS,
    )
    expected_shape = (len(window_data.labels), K_VALUES[0], len(COMPACT_FEATURE_COLUMNS))
    if window_data.features.shape != expected_shape:
        raise RuntimeError(f"Unexpected compact feature shape: {window_data.features.shape}")

    features = torch.as_tensor(window_data.features[:4], dtype=torch.float32)
    bird_ids = torch.as_tensor(window_data.bird_ids[:4], dtype=torch.long)
    for use_bird_id in (True, False):
        model = TrilineTransformer(
            n_features=features.shape[-1],
            n_birds=len(window_data.bird_to_idx),
            max_k=K_VALUES[0],
            use_bird_id=use_bird_id,
        )
        outputs = model(features, bird_ids)
        if outputs["fly_logit"].shape != (features.shape[0],):
            raise RuntimeError(f"Bad fly_logit shape with use_bird_id={use_bird_id}")


def run_original_variants(base_output_dir: Path) -> list[dict[str, object]]:
    filtered_df, data_summary = preprocess_original(
        INPUT_CSV,
        base_output_dir / "filtered_sep_dec.csv",
    )
    rows_for_top_level: list[dict[str, object]] = []

    for variant, use_bird_id in VARIANTS:
        variant_dir = base_output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(variant_dir, "Original", variant, use_bird_id)
        with (variant_dir / "data_summary.json").open("w", encoding="utf-8") as f:
            json.dump(data_summary, f, indent=2)

        summary_rows: list[dict[str, object]] = []
        for k in K_VALUES:
            print(f"Original {variant}: building windows k={k}")
            window_data = build_windows(
                filtered_df,
                k=k,
                fly_threshold_km=FLY_THRESHOLD_KM,
                feature_columns=COMPACT_FEATURE_COLUMNS,
            )
            train_idx, test_idx = make_split(
                window_data,
                split_mode="chronological",
                train_fraction=0.8,
                seed=SEED + k,
            )
            mean, std = fit_normalizer(window_data.features, train_idx)
            normalized_features = normalize_features(window_data.features, mean, std)
            summary_rows.append(
                train_original_one_k(
                    window_data,
                    k,
                    train_idx,
                    test_idx,
                    normalized_features,
                    mean,
                    std,
                    variant_dir,
                    data_summary,
                    use_bird_id=use_bird_id,
                    model_name=variant,
                )
            )

        write_original_csv(variant_dir / "summary.csv", summary_rows)
        write_original_analysis(variant_dir, summary_rows, data_summary)
        best = max(summary_rows, key=lambda row: float(row["f1"]))
        rows_for_top_level.append({"setup": "Original", "variant": variant, **best})

    return rows_for_top_level


def run_improved_variants(base_output_dir: Path) -> list[dict[str, object]]:
    cleaned_df, data_summary = preprocess_movement_data(INPUT_CSV, CLEANED_CSV)
    rows_for_top_level: list[dict[str, object]] = []

    for variant, use_bird_id in VARIANTS:
        variant_dir = base_output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(variant_dir, "Improved", variant, use_bird_id)
        with (variant_dir / "data_summary.json").open("w", encoding="utf-8") as f:
            json.dump(data_summary, f, indent=2)

        neural_rows: list[dict[str, object]] = []
        baseline_rows: list[dict[str, object]] = []
        for k in K_VALUES:
            print(f"Improved {variant}: building windows k={k}")
            window_data = build_windows(
                cleaned_df,
                k=k,
                fly_threshold_km=FLY_THRESHOLD_KM,
                feature_columns=COMPACT_FEATURE_COLUMNS,
            )
            raw_features = window_data.features

            for split_mode in SPLIT_MODES:
                print(f"Improved {variant}: running {split_mode} k={k}")
                train_idx, test_idx = make_split(
                    window_data,
                    split_mode=split_mode,
                    train_fraction=0.8,
                    seed=SEED + k,
                )
                mean, std = fit_normalizer(raw_features, train_idx)
                normalized_features = normalize_features(raw_features, mean, std)
                neural_rows.append(
                    train_network_one_split(
                        window_data,
                        k,
                        split_mode,
                        train_idx,
                        test_idx,
                        normalized_features,
                        mean,
                        std,
                        variant_dir,
                        data_summary,
                        use_bird_id=use_bird_id,
                        model_name=variant,
                    )
                )
                baseline_rows.extend(
                    run_baselines(
                        window_data,
                        k,
                        split_mode,
                        train_idx,
                        test_idx,
                        raw_features,
                        normalized_features,
                        variant_dir,
                    )
                )

        write_improved_csv(variant_dir / "summary.csv", neural_rows)
        write_improved_csv(variant_dir / "baseline_summary.csv", baseline_rows)
        write_improved_csv(variant_dir / "comparison_summary.csv", neural_rows + baseline_rows)
        write_improved_analysis(variant_dir, neural_rows, baseline_rows, data_summary)
        best = max(neural_rows, key=lambda row: float(row["f1"]))
        rows_for_top_level.append({"setup": "Improved", "variant": variant, **best})

    return rows_for_top_level


def best_row(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "f1" not in df:
        return None
    row = df.loc[df["f1"].astype(float).idxmax()].to_dict()
    return {key: json_safe(value) for key, value in row.items()}


def format_row(row: dict[str, object] | None, label: str) -> str:
    if row is None:
        return f"| {label} | -- | -- | -- | -- | -- | -- | -- | -- |"
    split = row.get("split_mode", "chronological")
    model = row.get("model", label)
    k = int(float(row["k"]))
    return (
        f"| {label} | {split} | {model} | {k} | {float(row['f1']):.4f} | "
        f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | "
        f"{int(float(row['fp']))} | {float(row['test_fly_rate']):.4f} |"
    )


def write_top_level_analysis(output_dir: Path, compact_rows: list[dict[str, object]]) -> None:
    full_original = best_row(ROOT / "flynofly_path" / "FlyNofly" / "summary.csv")
    full_improved = best_row(ROOT / "flynofly_path" / "FlyNofly_Improved" / "summary.csv")

    compact_lookup = {
        (str(row["setup"]), str(row["variant"])): row
        for row in compact_rows
    }

    lines = [
        "# Compact Fly/No-Fly Feature Analysis",
        "",
        "## Feature Set",
        "",
        "The compact experiments used 10 input features per day:",
        "",
        "```text",
        "\n".join(COMPACT_FEATURE_COLUMNS),
        "```",
        "",
        "Each compact setup was tested with and without the bird identity embedding.",
        "",
        "## Best Neural Results",
        "",
        "| Experiment | Split | Model | k | F1 | Precision | Recall | FP | Test fly rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        format_row(full_original, "Original full features"),
        format_row(compact_lookup.get(("Original", "compact_with_bird_id")), "Original compact with bird ID"),
        format_row(compact_lookup.get(("Original", "compact_without_bird_id")), "Original compact without bird ID"),
        format_row(full_improved, "Improved full features"),
        format_row(compact_lookup.get(("Improved", "compact_with_bird_id")), "Improved compact with bird ID"),
        format_row(compact_lookup.get(("Improved", "compact_without_bird_id")), "Improved compact without bird ID"),
        "",
        "## Notes",
        "",
        "- Original uses Sep-Dec-only filtering before windowing and fixed probability threshold 0.5.",
        "- Improved uses all-month context, Sep-Dec target days, and tuned probability thresholds.",
        "- Prediction CSVs still include bird IDs for inspection; the without-bird-ID variants do not feed identity into the model.",
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact feature Fly/No-Fly experiments")
    parser.add_argument("--output-dir", default="flynofly_path/compact_features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df, _ = preprocess_movement_data(INPUT_CSV, CLEANED_CSV)
    smoke_check(cleaned_df)
    print("Compact smoke check passed")

    rows: list[dict[str, object]] = []
    rows.extend(run_original_variants(output_dir / "Original"))
    rows.extend(run_improved_variants(output_dir / "Improved"))
    write_top_level_analysis(output_dir, rows)
    print(f"Done. Compact results saved to {output_dir}")


if __name__ == "__main__":
    main()
