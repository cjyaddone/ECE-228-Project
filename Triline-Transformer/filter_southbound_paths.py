from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "dataset2_daily_movement.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "filtered" / "dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "data" / "filtered" / "dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5_summary.csv"
)
DEFAULT_AUDIT = (
    ROOT / "data" / "filtered" / "dataset2_southbound_paths_jun_dec_lat30_50_min50_drop5_audit.json"
)


def make_path_id(bird_id: str, year: int, copy_index: int) -> str:
    safe = (
        bird_id.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"{safe.strip('_')}__{year}__southbound_{copy_index:02d}"


def extract_one_southbound_path(
    group: pd.DataFrame,
    min_path_rows: int,
    min_lat_drop: float,
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    """Extract the north-anchor to later south-anchor segment for one bird-year."""
    group = group.sort_values("date").reset_index(drop=True)
    lats = group["lat_median"].to_numpy()

    north_idx = int(lats.argmax())
    if north_idx >= len(group) - 1:
        return None, None

    later_lats = lats[north_idx:]
    south_idx = north_idx + int(later_lats.argmin())
    lat_drop = float(lats[north_idx] - lats[south_idx])
    path_rows = int(south_idx - north_idx + 1)

    if south_idx <= north_idx or lat_drop < min_lat_drop or path_rows < min_path_rows:
        return None, None

    path = group.iloc[north_idx : south_idx + 1].copy()
    summary = {
        "path_rows": path_rows,
        "north_anchor_date": group.loc[north_idx, "date"],
        "south_anchor_date": group.loc[south_idx, "date"],
        "north_anchor_lat": float(group.loc[north_idx, "lat_median"]),
        "south_anchor_lat": float(group.loc[south_idx, "lat_median"]),
        "north_anchor_lon": float(group.loc[north_idx, "lon_median"]),
        "south_anchor_lon": float(group.loc[south_idx, "lon_median"]),
        "lat_drop_deg": lat_drop,
    }
    return path, summary


def filter_southbound_paths(
    input_csv: Path,
    output_csv: Path,
    summary_csv: Path,
    audit_json: Path,
    min_lat: float = 30.0,
    max_lat: float = 50.0,
    start_month: int = 6,
    end_month: int = 12,
    min_corridor_rows: int = 50,
    min_path_rows: int = 50,
    min_lat_drop: float = 5.0,
) -> dict[str, object]:
    df = pd.read_csv(input_csv)
    original_rows = int(len(df))
    original_birds = int(df["individual_local_identifier"].nunique())

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["year"] = df["date"].dt.year

    corridor = df[
        df["date"].dt.month.between(start_month, end_month, inclusive="both")
        & df["lat_median"].between(min_lat, max_lat, inclusive="both")
    ].copy()
    corridor = corridor.sort_values(["individual_local_identifier", "date"]).reset_index(drop=True)

    path_frames: list[pd.DataFrame] = []
    path_summaries: list[dict[str, object]] = []
    rejected_counts = {
        "too_few_corridor_rows": 0,
        "no_later_south_anchor": 0,
        "insufficient_drop_or_path_rows": 0,
    }

    grouped = corridor.groupby(["individual_local_identifier", "year"], sort=True)
    candidate_groups = 0
    for (bird_id, year), group in grouped:
        corridor_rows = int(len(group))
        if corridor_rows < min_corridor_rows:
            rejected_counts["too_few_corridor_rows"] += 1
            continue

        candidate_groups += 1
        path, summary = extract_one_southbound_path(group, min_path_rows, min_lat_drop)
        if path is None or summary is None:
            lats = group["lat_median"].to_numpy()
            north_idx = int(lats.argmax())
            if north_idx >= len(group) - 1:
                rejected_counts["no_later_south_anchor"] += 1
            else:
                rejected_counts["insufficient_drop_or_path_rows"] += 1
            continue

        copy_index = 1
        path_id = make_path_id(str(bird_id), int(year), copy_index)
        path = path.copy()
        path.insert(0, "path_id", path_id)
        path.insert(1, "source_individual_local_identifier", bird_id)
        path.insert(2, "path_year", int(year))
        path.insert(3, "path_copy_index", copy_index)
        path.insert(4, "corridor_rows", corridor_rows)
        path.insert(5, "path_rows", int(summary["path_rows"]))
        path.insert(6, "lat_drop_deg", float(summary["lat_drop_deg"]))

        path_frames.append(path)
        path_summaries.append(
            {
                "path_id": path_id,
                "source_individual_local_identifier": bird_id,
                "path_year": int(year),
                "path_copy_index": copy_index,
                "corridor_rows": corridor_rows,
                **summary,
            }
        )

    if path_frames:
        output = pd.concat(path_frames, ignore_index=True)
    else:
        output = pd.DataFrame()
    summary_df = pd.DataFrame(path_summaries)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    audit = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "summary_csv": str(summary_csv),
        "original_rows": original_rows,
        "original_birds": original_birds,
        "corridor_rows": int(len(corridor)),
        "corridor_birds": int(corridor["individual_local_identifier"].nunique()),
        "corridor_bird_years": int(grouped.ngroups),
        "candidate_bird_years_min_corridor_rows": candidate_groups,
        "output_rows": int(len(output)),
        "output_paths": int(len(summary_df)),
        "output_source_birds": int(summary_df["source_individual_local_identifier"].nunique())
        if len(summary_df)
        else 0,
        "start_month": start_month,
        "end_month": end_month,
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_corridor_rows": min_corridor_rows,
        "min_path_rows": min_path_rows,
        "min_lat_drop": min_lat_drop,
        "rejected_counts": rejected_counts,
    }
    with audit_json.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset2 daily movement into June-Dec southbound path copies."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--min-lat", type=float, default=30.0)
    parser.add_argument("--max-lat", type=float, default=50.0)
    parser.add_argument("--start-month", type=int, default=6)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--min-corridor-rows", type=int, default=50)
    parser.add_argument("--min-path-rows", type=int, default=50)
    parser.add_argument("--min-lat-drop", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = filter_southbound_paths(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
        audit_json=args.audit_json,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        start_month=args.start_month,
        end_month=args.end_month,
        min_corridor_rows=args.min_corridor_rows,
        min_path_rows=args.min_path_rows,
        min_lat_drop=args.min_lat_drop,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
