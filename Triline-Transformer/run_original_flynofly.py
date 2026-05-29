from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preprocessing import (
    build_windows,
    fit_normalizer,
    make_split,
    normalize_features,
)
from dataset import TrilineWindowDataset
from model import TrilineTransformer
from train_test_fly_nofly import (
    FLY_THRESHOLD_KM,
    INPUT_CSV,
    K_VALUES,
    SEED,
    compute_metrics,
    evaluate_network,
    json_safe,
    multitask_loss,
    save_predictions,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def preprocess_original(input_csv: Path, output_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(input_csv)
    original_rows = len(df)
    original_birds = df["individual_local_identifier"].nunique()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    bird_counts = df.groupby("individual_local_identifier").size()
    eligible_birds = bird_counts[bird_counts >= 100].index
    filtered = df[
        df["individual_local_identifier"].isin(eligible_birds)
        & df["lat_median"].between(30.0, 50.0, inclusive="both")
        & df["date"].dt.month.isin({9, 10, 11, 12})
    ].copy()
    filtered = filtered.sort_values(["individual_local_identifier", "date"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "original_rows": int(original_rows),
        "original_birds": int(original_birds),
        "eligible_birds_min_100_records": int(len(eligible_birds)),
        "filtered_rows_sep_dec_lat30_50": int(len(filtered)),
        "filtered_birds": int(filtered["individual_local_identifier"].nunique()),
        "target_fly_rate_threshold_30km": float((filtered["step_length_km"] > FLY_THRESHOLD_KM).mean())
        if len(filtered)
        else 0.0,
    }
    return filtered, summary


def make_datasets(
    normalized_features: np.ndarray,
    window_data: Any,
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


def train_one_k(
    window_data: Any,
    k: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    normalized_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    output_dir: Path,
    data_summary: dict[str, object],
) -> dict[str, object]:
    set_seed(SEED + k)
    model_dir = output_dir / f"k_{k}"
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
    ).to(device)

    train_positive = float(window_data.labels[train_idx].sum())
    train_negative = float(len(train_idx) - train_positive)
    pos_weight_value = train_negative / max(train_positive, 1.0)
    fly_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    log_rows: list[dict[str, float | int | None]] = []
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
        metrics = compute_metrics(test_labels, test_probabilities, 0.5)
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
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
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
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_columns": window_data.feature_columns,
                    "bird_to_idx": window_data.bird_to_idx,
                    "pos_weight": pos_weight_value,
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
    test_loss, test_labels, test_probabilities, _, _ = evaluate_network(
        model, test_loader, fly_loss_fn, mse_loss_fn, device
    )
    metrics = compute_metrics(test_labels, test_probabilities, 0.5)
    save_predictions(
        model_dir / "predictions.csv",
        window_data,
        test_idx,
        test_labels,
        test_probabilities,
        0.5,
    )

    metrics_payload: dict[str, object] = {
        "k": k,
        "fly_threshold_km": FLY_THRESHOLD_KM,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_fly_rate": float(window_data.labels[train_idx].mean()),
        "test_fly_rate": float(window_data.labels[test_idx].mean()),
        "best_epoch": int(best_epoch),
        "best_test_loss": float(best_loss),
        "final_test_loss": float(test_loss),
        "pos_weight": float(pos_weight_value),
        **metrics,
    }

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({key: json_safe(value) for key, value in metrics_payload.items()}, f, indent=2)

    return metrics_payload


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(output_dir: Path, rows: list[dict[str, object]], data_summary: dict[str, object]) -> None:
    best_f1 = max(rows, key=lambda row: float(row["f1"]))
    best_recall = max(rows, key=lambda row: float(row["recall"]))
    lines = [
        "# TestFlyNofly Analysis",
        "",
        "## Data",
        "",
        f"- Source rows/birds: {data_summary['original_rows']} rows, {data_summary['original_birds']} birds",
        f"- Filtered rows/birds: {data_summary['filtered_rows_sep_dec_lat30_50']} rows, {data_summary['filtered_birds']} birds",
        "- Filters: birds with at least 100 original daily records, latitude 30-50, dates Sep 1-Dec 31",
        f"- Fly threshold: {FLY_THRESHOLD_KM:g} km next-day step length",
        "",
        "## Results",
        "",
        "| k | Train | Test | Test fly rate | Accuracy | Precision | Recall | F1 | FP | FPR | ROC-AUC | PR-AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: int(item["k"])):
        lines.append(
            f"| {int(row['k'])} | {int(row['train_samples'])} | {int(row['test_samples'])} | "
            f"{row['test_fly_rate']:.4f} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {int(row['fp'])} | "
            f"{row['false_positive_rate']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Best F1 was k={int(best_f1['k'])} with F1={best_f1['f1']:.4f}.",
            f"- Best recall was k={int(best_recall['k'])} with recall={best_recall['recall']:.4f}.",
            "- Validation is used as the requested test split; no separate held-out test split was created.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original TestFlyNofly Triline experiment")
    parser.add_argument("--output-dir", default="flynofly_path/original")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_df, data_summary = preprocess_original(INPUT_CSV, output_dir / "filtered_sep_dec.csv")
    with (output_dir / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(data_summary, f, indent=2)

    rows: list[dict[str, object]] = []
    for k in K_VALUES:
        print(f"Building original windows k={k}")
        window_data = build_windows(filtered_df, k=k, fly_threshold_km=FLY_THRESHOLD_KM)
        train_idx, test_idx = make_split(
            window_data, split_mode="chronological", train_fraction=0.8, seed=SEED + k
        )
        mean, std = fit_normalizer(window_data.features, train_idx)
        normalized_features = normalize_features(window_data.features, mean, std)
        rows.append(
            train_one_k(
                window_data,
                k,
                train_idx,
                test_idx,
                normalized_features,
                mean,
                std,
                output_dir,
                data_summary,
            )
        )

    write_csv(output_dir / "summary.csv", rows)
    write_analysis(output_dir, rows, data_summary)
    print(f"Done. Original results saved to {output_dir}")


if __name__ == "__main__":
    main()
