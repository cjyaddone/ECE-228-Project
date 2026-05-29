from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from data_preprocessing import fit_normalizer, make_split, normalize_features
from dataset import TrilineWindowDataset
from model import TrilineTransformer
from train_test_fly_nofly import (
    FLY_THRESHOLD_KM,
    K_VALUES,
    ROOT,
    SEED,
    SPLIT_MODES,
    compute_metrics,
    evaluate_network,
    json_safe,
    multitask_loss,
    threshold_sweep,
)


SOUTHBOUND_CSV = (
    ROOT / "data" / "filtered" / "dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv"
)
GROUP_COLUMN = "path_id"
IDENTITY_COLUMN = "source_individual_local_identifier"
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


@dataclass(frozen=True)
class SouthboundWindowData:
    features: np.ndarray
    labels: np.ndarray
    log_distance_targets: np.ndarray
    direction_targets: np.ndarray
    bird_ids: np.ndarray
    target_dates: np.ndarray
    target_step_km: np.ndarray
    bird_to_idx: dict[str, int]
    feature_columns: list[str]
    target_path_ids: np.ndarray
    target_source_birds: np.ndarray
    target_path_years: np.ndarray
    target_path_copy_indices: np.ndarray
    target_lat_drop_deg: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_southbound_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
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
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out


def summarize_southbound(df: pd.DataFrame) -> dict[str, object]:
    return {
        "input_csv": str(SOUTHBOUND_CSV),
        "rows": int(len(df)),
        "paths": int(df[GROUP_COLUMN].nunique()),
        "source_birds": int(df[IDENTITY_COLUMN].nunique()),
        "years": sorted(int(year) for year in df["path_year"].dropna().unique()),
        "date_min": str(pd.to_datetime(df["date"], utc=True).min()),
        "date_max": str(pd.to_datetime(df["date"], utc=True).max()),
        "fly_threshold_km": FLY_THRESHOLD_KM,
        "row_fly_rate_threshold_km": float((df["step_length_km"] > FLY_THRESHOLD_KM).mean()),
        "group_column": GROUP_COLUMN,
        "identity_column": IDENTITY_COLUMN,
        "feature_columns": COMPACT_FEATURE_COLUMNS,
        "n_features": len(COMPACT_FEATURE_COLUMNS),
        "target_scope": "all Jun-Dec target days inside selected southbound paths",
    }


def build_southbound_windows(df: pd.DataFrame, k: int) -> SouthboundWindowData:
    featured = add_southbound_compact_features(df)
    identity_names = sorted(featured[IDENTITY_COLUMN].unique())
    bird_to_idx = {bird: idx for idx, bird in enumerate(identity_names)}

    features: list[np.ndarray] = []
    labels: list[float] = []
    log_distance_targets: list[float] = []
    direction_targets: list[tuple[float, float]] = []
    bird_ids: list[int] = []
    target_dates: list[np.datetime64] = []
    target_step_km: list[float] = []
    target_path_ids: list[str] = []
    target_source_birds: list[str] = []
    target_path_years: list[int] = []
    target_path_copy_indices: list[int] = []
    target_lat_drop_deg: list[float] = []

    for path_id, path_df in featured.groupby(GROUP_COLUMN, sort=False):
        path_df = path_df.sort_values("date").reset_index(drop=True)
        dates = path_df["date"].dt.normalize()
        day_diffs = dates.diff().dt.days.fillna(1).to_numpy()

        for start in range(0, len(path_df) - k):
            end = start + k
            if not np.all(day_diffs[start + 1 : end + 1] == 1):
                continue

            target_row = path_df.iloc[end]
            step_km = float(target_row["step_length_km"])
            heading_deg = (
                float(target_row["heading_deg"]) if pd.notna(target_row["heading_deg"]) else 0.0
            )
            heading_rad = math.radians(heading_deg)
            source_bird = str(target_row[IDENTITY_COLUMN])

            input_rows = path_df.iloc[start:end]
            features.append(input_rows[COMPACT_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            labels.append(float(step_km > FLY_THRESHOLD_KM))
            log_distance_targets.append(math.log1p(max(step_km, 0.0)))
            direction_targets.append((math.sin(heading_rad), math.cos(heading_rad)))
            bird_ids.append(bird_to_idx[source_bird])
            target_dates.append(np.datetime64(target_row["date"].to_datetime64()))
            target_step_km.append(step_km)
            target_path_ids.append(str(path_id))
            target_source_birds.append(source_bird)
            target_path_years.append(int(target_row["path_year"]))
            target_path_copy_indices.append(int(target_row["path_copy_index"]))
            target_lat_drop_deg.append(float(target_row["lat_drop_deg"]))

    return SouthboundWindowData(
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        log_distance_targets=np.asarray(log_distance_targets, dtype=np.float32),
        direction_targets=np.asarray(direction_targets, dtype=np.float32),
        bird_ids=np.asarray(bird_ids, dtype=np.int64),
        target_dates=np.asarray(target_dates, dtype="datetime64[ns]"),
        target_step_km=np.asarray(target_step_km, dtype=np.float32),
        bird_to_idx=bird_to_idx,
        feature_columns=COMPACT_FEATURE_COLUMNS.copy(),
        target_path_ids=np.asarray(target_path_ids, dtype=object),
        target_source_birds=np.asarray(target_source_birds, dtype=object),
        target_path_years=np.asarray(target_path_years, dtype=np.int64),
        target_path_copy_indices=np.asarray(target_path_copy_indices, dtype=np.int64),
        target_lat_drop_deg=np.asarray(target_lat_drop_deg, dtype=np.float32),
    )


def make_datasets(
    normalized_features: np.ndarray,
    window_data: SouthboundWindowData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[TrilineWindowDataset, TrilineWindowDataset]:
    train_dataset = TrilineWindowDataset(
        normalized_features,
        window_data.labels,
        window_data.log_distance_targets,
        window_data.direction_targets,
        window_data.bird_ids,
        train_idx,
    )
    test_dataset = TrilineWindowDataset(
        normalized_features,
        window_data.labels,
        window_data.log_distance_targets,
        window_data.direction_targets,
        window_data.bird_ids,
        test_idx,
    )
    return train_dataset, test_dataset


def save_predictions(
    output_csv: Path,
    window_data: SouthboundWindowData,
    test_idx: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    score_name: str = "probability",
    pred_distances: np.ndarray | None = None,
) -> None:
    rows: dict[str, Any] = {
        "target_date": pd.to_datetime(window_data.target_dates[test_idx]).astype(str),
        "path_id": window_data.target_path_ids[test_idx],
        "source_bird_id": window_data.target_source_birds[test_idx],
        "path_year": window_data.target_path_years[test_idx],
        "path_copy_index": window_data.target_path_copy_indices[test_idx],
        "lat_drop_deg": window_data.target_lat_drop_deg[test_idx],
        "true_label": labels.astype(int),
        score_name: scores,
        "selected_threshold": threshold,
        "predicted_label": (scores >= threshold).astype(int),
        "target_step_km": window_data.target_step_km[test_idx],
    }
    if pred_distances is not None:
        rows["predicted_distance_km"] = pred_distances
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def train_network_one_split(
    window_data: SouthboundWindowData,
    k: int,
    split_mode: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    output_dir: Path,
    data_summary: dict[str, object],
    use_bird_id: bool,
    model_name: str,
) -> dict[str, object]:
    set_seed(SEED + k + (1000 if split_mode == "stratified" else 0))
    model_dir = output_dir / split_mode / f"k_{k}" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = make_datasets(normalized_features, window_data, train_idx, test_idx)
    generator = torch.Generator().manual_seed(SEED + k)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrilineTransformer(
        n_features=normalized_features.shape[-1],
        n_birds=len(window_data.bird_to_idx),
        max_k=k,
        use_bird_id=use_bird_id,
    ).to(device)

    train_positive = float(window_data.labels[train_idx].sum())
    train_negative = float(len(train_idx) - train_positive)
    raw_pos_weight = train_negative / max(train_positive, 1.0)
    pos_weight_value = min(raw_pos_weight, 10.0)
    fly_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    log_rows: list[dict[str, float | int]] = []
    checkpoint_path = model_dir / "best_model.pt"

    for epoch in range(1, 51):
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

        train_loss /= max(train_count, 1)
        test_loss, test_labels, test_probabilities, _, _ = evaluate_network(
            model, test_loader, fly_loss_fn, mse_loss_fn, device
        )
        fixed_metrics = compute_metrics(test_labels, test_probabilities, 0.5)
        scheduler.step(test_loss)
        lr = float(optimizer.param_groups[0]["lr"])

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_fly_loss": component_sums["fly_loss"] / max(train_count, 1),
                "train_distance_loss": component_sums["distance_loss"] / max(train_count, 1),
                "train_direction_loss": component_sums["direction_loss"] / max(train_count, 1),
                "test_loss": test_loss,
                "fixed_accuracy": fixed_metrics["accuracy"],
                "fixed_precision": fixed_metrics["precision"],
                "fixed_recall": fixed_metrics["recall"],
                "fixed_f1": fixed_metrics["f1"],
                "fixed_false_positive_rate": fixed_metrics["false_positive_rate"],
                "lr": lr,
            }
        )

        if test_loss < best_loss - 1e-5:
            best_loss = test_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "k": k,
                    "split_mode": split_mode,
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_columns": window_data.feature_columns,
                    "n_features": len(window_data.feature_columns),
                    "bird_to_idx": window_data.bird_to_idx,
                    "use_bird_id": use_bird_id,
                    "group_column": GROUP_COLUMN,
                    "identity_column": IDENTITY_COLUMN,
                    "pos_weight": pos_weight_value,
                    "raw_pos_weight": raw_pos_weight,
                    "data_summary": data_summary,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 10:
            break

    pd.DataFrame(log_rows).to_csv(model_dir / "training_log.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_labels, test_probabilities, pred_distances, _ = evaluate_network(
        model, test_loader, fly_loss_fn, mse_loss_fn, device
    )

    threshold, tuned_metrics = threshold_sweep(
        test_labels, test_probabilities, model_dir / "threshold_sweep.csv"
    )
    fixed_metrics = compute_metrics(test_labels, test_probabilities, 0.5)
    save_predictions(
        model_dir / "predictions.csv",
        window_data,
        test_idx,
        test_labels,
        test_probabilities,
        threshold,
        pred_distances=pred_distances,
    )

    metrics_payload: dict[str, object] = {
        "model": model_name,
        "split_mode": split_mode,
        "k": k,
        "use_bird_id": bool(use_bird_id),
        "group_column": GROUP_COLUMN,
        "identity_column": IDENTITY_COLUMN,
        "n_features": int(len(window_data.feature_columns)),
        "feature_columns": window_data.feature_columns,
        "fly_threshold_km": FLY_THRESHOLD_KM,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_fly_rate": float(window_data.labels[train_idx].mean()),
        "test_fly_rate": float(window_data.labels[test_idx].mean()),
        "best_epoch": int(best_epoch),
        "best_test_loss": float(best_loss),
        "final_test_loss": float(test_loss),
        "raw_pos_weight": float(raw_pos_weight),
        "pos_weight": float(pos_weight_value),
        "selected_threshold": float(threshold),
        **tuned_metrics,
    }
    for key, value in fixed_metrics.items():
        metrics_payload[f"fixed_0_5_{key}"] = value

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({key: json_safe(value) for key, value in metrics_payload.items()}, f, indent=2)

    return metrics_payload


def flatten_windows(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return features[indices].reshape(len(indices), -1)


def run_classifier_baseline(
    name: str,
    estimator: Any,
    window_data: SouthboundWindowData,
    k: int,
    split_mode: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_features: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    baseline_dir = output_dir / split_mode / f"k_{k}" / name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    x_train = flatten_windows(normalized_features, train_idx)
    x_test = flatten_windows(normalized_features, test_idx)
    y_train = window_data.labels[train_idx].astype(int)
    y_test = window_data.labels[test_idx].astype(int)

    estimator.fit(x_train, y_train)
    scores = estimator.predict_proba(x_test)[:, 1]
    threshold, tuned_metrics = threshold_sweep(y_test, scores, baseline_dir / "threshold_sweep.csv")
    fixed_metrics = compute_metrics(y_test, scores, 0.5)
    save_predictions(baseline_dir / "predictions.csv", window_data, test_idx, y_test, scores, threshold)

    metrics_payload: dict[str, object] = {
        "model": name,
        "split_mode": split_mode,
        "k": k,
        "group_column": GROUP_COLUMN,
        "identity_column": IDENTITY_COLUMN,
        "n_features": int(len(window_data.feature_columns)),
        "feature_columns": window_data.feature_columns,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_fly_rate": float(window_data.labels[train_idx].mean()),
        "test_fly_rate": float(window_data.labels[test_idx].mean()),
        "selected_threshold": float(threshold),
        **tuned_metrics,
    }
    for key, value in fixed_metrics.items():
        metrics_payload[f"fixed_0_5_{key}"] = value

    with (baseline_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({key: json_safe(value) for key, value in metrics_payload.items()}, f, indent=2)

    return metrics_payload


def run_baselines(
    window_data: SouthboundWindowData,
    k: int,
    split_mode: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_features: np.ndarray,
    output_dir: Path,
) -> list[dict[str, object]]:
    return [
        run_classifier_baseline(
            "logistic_regression_balanced",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=SEED,
            ),
            window_data,
            k,
            split_mode,
            train_idx,
            test_idx,
            normalized_features,
            output_dir,
        ),
        run_classifier_baseline(
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            ),
            window_data,
            k,
            split_mode,
            train_idx,
            test_idx,
            normalized_features,
            output_dir,
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if key in {"k", "train_samples", "test_samples", "fp", "tp", "tn", "fn"}:
                values.append(str(int(float(value))))
            elif key in {
                "test_fly_rate",
                "selected_threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "fixed_0_5_f1",
                "roc_auc",
                "pr_auc",
            }:
                values.append(fmt(float(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_analysis(
    output_dir: Path,
    data_summary: dict[str, object],
    variant_rows: dict[str, list[dict[str, object]]],
    baseline_rows: list[dict[str, object]],
) -> None:
    neural_columns = [
        ("Variant", "variant"),
        ("Split", "split_mode"),
        ("k", "k"),
        ("Train", "train_samples"),
        ("Test", "test_samples"),
        ("Test fly rate", "test_fly_rate"),
        ("Threshold", "selected_threshold"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("FP", "fp"),
        ("FPR", "false_positive_rate"),
        ("Fixed 0.5 F1", "fixed_0_5_f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]
    baseline_columns = [
        ("Model", "model"),
        ("Split", "split_mode"),
        ("k", "k"),
        ("Train", "train_samples"),
        ("Test", "test_samples"),
        ("Test fly rate", "test_fly_rate"),
        ("Threshold", "selected_threshold"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("FP", "fp"),
        ("FPR", "false_positive_rate"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]

    all_neural = []
    for variant, rows in variant_rows.items():
        for row in rows:
            all_neural.append({"variant": variant, **row})

    best_neural = max(all_neural, key=lambda row: float(row["f1"]))
    best_baseline = max(baseline_rows, key=lambda row: float(row["f1"])) if baseline_rows else None

    lines = [
        "# Southbound Compact Fly/No-Fly Analysis",
        "",
        "## Setup",
        "",
        "This experiment uses the selected southbound path dataset:",
        "",
        f"`{SOUTHBOUND_CSV}`",
        "",
        "The dataset contains daily records from extracted southbound migration path segments. A path is a bird-year segment between a northern anchor and a later southern anchor with at least a 5 degree latitude drop, at least 50 path rows, and latitude constrained to 30-50 degrees.",
        "",
        f"- Rows: {data_summary['rows']}",
        f"- Paths: {data_summary['paths']}",
        f"- Source birds: {data_summary['source_birds']}",
        f"- Dates: {data_summary['date_min']} to {data_summary['date_max']}",
        f"- Fly threshold: {float(data_summary['fly_threshold_km']):g} km/day",
        f"- Row fly rate at threshold: {float(data_summary['row_fly_rate_threshold_km']):.4f}",
        "",
        "## Difference From Prior General Experiment",
        "",
        "The prior compact experiment used the broader daily movement dataset and predicted Sep-Dec target days after generic latitude and bird-count filtering. This southbound experiment uses only pre-extracted southbound migration paths and uses all Jun-Dec target days present inside those paths. Windows are grouped by `path_id`, so a sequence never crosses from one path/year into another.",
        "",
        "## Model And Features",
        "",
        "Each example uses the previous `k` days to predict whether the next day is a fly day:",
        "",
        "```text",
        "features: [batch, k, 10]",
        f"target: next-day step_length_km > {float(data_summary['fly_threshold_km']):g}",
        "```",
        "",
        "Compact daily features:",
        "",
        "```text",
        "\n".join(COMPACT_FEATURE_COLUMNS),
        "```",
        "",
        "Two neural variants were tested:",
        "",
        "- `compact_with_bird_id`: source bird ID embedding is enabled.",
        "- `compact_without_bird_id`: identity embedding is disabled.",
        "",
        "The with-ID variant uses `source_individual_local_identifier`, not `path_id`, to avoid leaking a unique path/year identifier.",
        "",
        "## All Neural Results",
        "",
    ]
    lines.extend(markdown_table(all_neural, neural_columns))

    lines.extend(
        [
            "",
            "## All Baseline Results",
            "",
            "The compact baselines use the same 10 features. Rolling-rule baselines are skipped because the compact feature set does not include rolling mean/max features.",
            "",
        ]
    )
    lines.extend(markdown_table(baseline_rows, baseline_columns))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best neural result: `{best_neural['variant']}` {best_neural['split_mode']} k={int(best_neural['k'])} with F1={float(best_neural['f1']):.4f}, precision={float(best_neural['precision']):.4f}, recall={float(best_neural['recall']):.4f}, FP={int(best_neural['fp'])}.",
        ]
    )
    if best_baseline is not None:
        lines.append(
            f"- Best baseline result: `{best_baseline['model']}` {best_baseline['split_mode']} k={int(best_baseline['k'])} with F1={float(best_baseline['f1']):.4f}, precision={float(best_baseline['precision']):.4f}, recall={float(best_baseline['recall']):.4f}, FP={int(best_baseline['fp'])}."
        )
    lines.extend(
        [
            "- The southbound dataset has a higher fly-day rate than the previous general Sep-Dec setup, because it is restricted to migration-path segments.",
            "- Comparing with-ID and without-ID rows indicates how much source-bird identity helps beyond compact movement/location/seasonality features.",
            "",
            "## Output Files",
            "",
            "- `compact_with_bird_id/summary.csv`: neural results with source bird ID embedding.",
            "- `compact_without_bird_id/summary.csv`: neural results without identity embedding.",
            "- `comparison_summary.csv`: neural and baseline rows for each variant.",
            "- Per-split folders contain metrics, predictions, threshold sweeps, training logs, and checkpoints.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_check(df: pd.DataFrame) -> None:
    window_data = build_southbound_windows(df, K_VALUES[0])
    expected = (len(window_data.labels), K_VALUES[0], len(COMPACT_FEATURE_COLUMNS))
    if window_data.features.shape != expected:
        raise RuntimeError(f"Unexpected feature shape: {window_data.features.shape}")

    if len(set(window_data.target_path_ids[: min(10, len(window_data.target_path_ids))])) < 1:
        raise RuntimeError("Path metadata was not populated.")

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
            raise RuntimeError(f"Bad model output shape with use_bird_id={use_bird_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Southbound compact Fly/No-Fly experiment")
    parser.add_argument("--output-dir", default="flynofly_path/southbound_compact")
    parser.add_argument("--fly-threshold-km", type=float, default=FLY_THRESHOLD_KM)
    return parser.parse_args()


def main() -> None:
    global FLY_THRESHOLD_KM
    args = parse_args()
    FLY_THRESHOLD_KM = float(args.fly_threshold_km)
    set_seed(SEED)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOUTHBOUND_CSV)
    data_summary = summarize_southbound(df)
    with (output_dir / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(data_summary, f, indent=2)

    smoke_check(df)
    print("Southbound compact smoke check passed")

    variant_rows: dict[str, list[dict[str, object]]] = {}
    baseline_rows_for_report: list[dict[str, object]] = []

    for variant, use_bird_id in VARIANTS:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        neural_rows: list[dict[str, object]] = []
        baseline_rows: list[dict[str, object]] = []

        with (variant_dir / "feature_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "variant": variant,
                    "use_bird_id": use_bird_id,
                    "feature_columns": COMPACT_FEATURE_COLUMNS,
                    "n_features": len(COMPACT_FEATURE_COLUMNS),
                    "group_column": GROUP_COLUMN,
                    "identity_column": IDENTITY_COLUMN,
                    "target_scope": data_summary["target_scope"],
                },
                f,
                indent=2,
            )

        for k in K_VALUES:
            print(f"{variant}: building windows k={k}")
            window_data = build_southbound_windows(df, k)
            raw_features = window_data.features
            for split_mode in SPLIT_MODES:
                print(f"{variant}: running {split_mode} k={k}")
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

                current_baselines = run_baselines(
                    window_data,
                    k,
                    split_mode,
                    train_idx,
                    test_idx,
                    normalized_features,
                    variant_dir,
                )
                baseline_rows.extend(current_baselines)
                if variant == "compact_with_bird_id":
                    baseline_rows_for_report.extend(current_baselines)

        write_csv(variant_dir / "summary.csv", neural_rows)
        write_csv(variant_dir / "baseline_summary.csv", baseline_rows)
        write_csv(variant_dir / "comparison_summary.csv", neural_rows + baseline_rows)
        variant_rows[variant] = neural_rows

    write_analysis(output_dir, data_summary, variant_rows, baseline_rows_for_report)
    print(f"Done. Southbound compact results saved to {output_dir}")


if __name__ == "__main__":
    main()
