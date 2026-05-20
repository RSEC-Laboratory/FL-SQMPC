#!/usr/bin/env python3
"""Create Bot-IoT processed tensor variants by dropping raw feature columns."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "Bot_IoT_processed"


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one column is required")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-columns", type=parse_csv_strings, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory for this derived dataset.",
    )
    return parser.parse_args()


def load_meta(source: Path) -> dict:
    meta_path = source / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def feature_drop_indices(meta: dict, drop_columns: list[str]) -> tuple[list[int], dict[str, list[int]]]:
    continuous = list(meta.get("continuous_features", []))
    categorical = list(meta.get("categorical_features", []))
    one_hot_ranges = meta.get("one_hot_ranges", {})
    indices_by_column: dict[str, list[int]] = {}

    for column in drop_columns:
        if column in continuous:
            indices_by_column[column] = [continuous.index(column)]
        elif column in categorical:
            if column not in one_hot_ranges:
                raise ValueError(f"Categorical column {column!r} has no one-hot range in meta.json.")
            start, end = one_hot_ranges[column]
            indices_by_column[column] = list(range(int(start), int(end)))
        else:
            raise ValueError(f"Column {column!r} is not listed as continuous or categorical in meta.json.")

    indices = sorted({idx for column_indices in indices_by_column.values() for idx in column_indices})
    return indices, indices_by_column


def updated_meta(meta: dict, source: Path, args: argparse.Namespace, keep_indices: list[int]) -> dict:
    drop_columns = list(args.drop_columns)
    drop_set = set(drop_columns)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    old_one_hot_ranges = meta.get("one_hot_ranges", {})

    new_continuous = [column for column in meta.get("continuous_features", []) if column not in drop_set]
    new_categorical = [column for column in meta.get("categorical_features", []) if column not in drop_set]
    new_one_hot_ranges: dict[str, list[int]] = {}
    for column in new_categorical:
        start, end = old_one_hot_ranges[column]
        mapped = [old_to_new[idx] for idx in range(int(start), int(end)) if idx in old_to_new]
        if not mapped:
            raise ValueError(f"No retained one-hot columns for categorical feature {column!r}.")
        expected = list(range(min(mapped), max(mapped) + 1))
        if mapped != expected:
            raise ValueError(f"Retained one-hot range for {column!r} is not contiguous.")
        new_one_hot_ranges[column] = [min(mapped), max(mapped) + 1]

    new_meta = dict(meta)
    new_meta["source_dataset_id"] = meta.get("dataset_id")
    new_meta["source_dataset_path"] = str(source)
    new_meta["dataset_id"] = args.dataset_id
    new_meta["display_name"] = args.display_name or f"{meta.get('display_name', 'Bot-IoT')} ({args.dataset_id})"
    new_meta["continuous_features"] = new_continuous
    new_meta["categorical_features"] = new_categorical
    new_meta["num_continuous_features"] = len(new_continuous)
    new_meta["num_categorical_features"] = len(new_categorical)
    new_meta["one_hot_ranges"] = new_one_hot_ranges
    new_meta["processed_feature_count"] = len(keep_indices)
    if "original_feature_count" in new_meta:
        new_meta["original_feature_count"] = int(new_meta["original_feature_count"]) - len(drop_columns)
    new_meta["feature_variant"] = {
        "source_processed_feature_count": meta.get("processed_feature_count"),
        "dropped_columns": drop_columns,
        "retained_processed_feature_count": len(keep_indices),
    }
    return new_meta


def save_filtered_split(source: Path, output: Path, split_name: str, keep_indices: list[int]) -> None:
    tensor = torch.load(source / split_name, map_location="cpu")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor in {split_name}, got shape {tuple(tensor.shape)}.")
    filtered = tensor[:, keep_indices].contiguous()
    torch.save(filtered, output / split_name)


def main() -> None:
    args = parse_args()
    source = args.source
    output = args.output

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    meta = load_meta(source)
    processed_feature_count = int(meta["processed_feature_count"])
    drop_indices, indices_by_column = feature_drop_indices(meta, args.drop_columns)
    keep_indices = [idx for idx in range(processed_feature_count) if idx not in set(drop_indices)]

    save_filtered_split(source, output, "X_train.pt", keep_indices)
    save_filtered_split(source, output, "X_test.pt", keep_indices)
    shutil.copy2(source / "y_train.pt", output / "y_train.pt")
    shutil.copy2(source / "y_test.pt", output / "y_test.pt")

    new_meta = updated_meta(meta, source, args, keep_indices)
    new_meta["feature_variant"]["dropped_processed_indices_by_column"] = indices_by_column
    (output / "meta.json").write_text(json.dumps(new_meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"wrote {output} with {len(keep_indices)} features "
        f"(dropped {len(drop_indices)} processed columns from {args.drop_columns})"
    )


if __name__ == "__main__":
    main()
