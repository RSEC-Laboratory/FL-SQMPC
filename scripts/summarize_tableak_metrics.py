#!/usr/bin/env python3
"""Aggregate batch-1 Tableak reconstruction metrics into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_recon_table import classify_method, load_tolerance_value


DATASET_LABEL = {
    "ciciot2023_tensor": "CICIoT2023",
    "ton_iot_tensor": "ToN-IoT",
    "bot_iot_minmax": "Bot-IoT",
}

METHOD_LABEL = {
    "vanilla_fl": "Vanilla FL",
    "dp_eps8": "DP-SGD eps=8",
    "dp_eps50": "DP-SGD eps=50",
    "cidiot": "CIDIoT",
    "sqmpc_qb2": "SQMPC B=2",
    "sqmpc_qb12": "SQMPC B=12",
}

METHOD_ORDER = ["vanilla_fl", "dp_eps8", "dp_eps50", "cidiot", "sqmpc_qb2", "sqmpc_qb12"]
DATASET_ORDER = ["ciciot2023_tensor", "ton_iot_tensor", "bot_iot_minmax"]
PARTITION_ORDER = ["iid", "non_iid"]
ROUND_ORDER = [1, 10]

METRIC_KEYS = [
    "reconstruction_accuracy",
    "fixed_0_05_tolerance_accuracy",
    "continuous_10pct_range_accuracy",
    "continuous_mae",
    "continuous_rmse",
    "categorical_exact_match",
    "label_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--range-tolerance", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str, str, int], list[dict[str, float]]] = defaultdict(list)
    for summary_path in sorted(args.results_dir.glob("*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = summary.get("metadata", {})
        if int(metadata.get("batch_size", 0)) != 1:
            continue
        method = classify_method(metadata)
        dataset = metadata.get("dataset")
        partition = metadata.get("partition")
        round_id = int(metadata.get("round", 0))
        if method not in METHOD_LABEL or dataset not in DATASET_LABEL:
            continue
        if partition not in PARTITION_ORDER or round_id not in ROUND_ORDER:
            continue
        metrics = dict(summary.get("metrics", {}))
        recon_path = _resolve_reconstruction_path(summary, args.results_dir)
        artifact_path = args.artifact_dir / recon_path.name.replace("_reconstruction.pt", ".pt")
        metrics["continuous_10pct_range_accuracy"] = load_tolerance_value(
            recon_path=recon_path,
            artifact_path=artifact_path,
            tolerance=args.range_tolerance,
        )
        grouped[(dataset, method, partition, round_id)].append(metrics)

    rows = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            for partition in PARTITION_ORDER:
                for round_id in ROUND_ORDER:
                    values = grouped.get((dataset, method, partition, round_id), [])
                    if not values:
                        continue
                    row: dict[str, Any] = {
                        "dataset": DATASET_LABEL[dataset],
                        "dataset_id": dataset,
                        "method": METHOD_LABEL[method],
                        "method_key": method,
                        "partition": partition,
                        "round": round_id,
                        "n": len(values),
                    }
                    for metric_key in METRIC_KEYS:
                        metric_values = [
                            float(item[metric_key])
                            for item in values
                            if item.get(metric_key) is not None
                        ]
                        row[f"{metric_key}_mean"] = mean(metric_values) if metric_values else ""
                        row[f"{metric_key}_std"] = stdev(metric_values) if len(metric_values) > 1 else 0.0
                    rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows)
    print(args.output_csv)
    print(args.output_md)


def _resolve_reconstruction_path(summary: dict[str, Any], results_dir: Path) -> Path:
    path = Path(summary["reconstruction_path"])
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    candidate = results_dir / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "method",
        "partition",
        "round",
        "n",
        "reconstruction_accuracy",
        "fixed_0_05_tolerance_accuracy",
        "continuous_10pct_range_accuracy",
        "continuous_mae",
        "continuous_rmse",
        "categorical_exact_match",
        "label_accuracy",
    ]
    lines = [
        "# Batch-1 Tableak Reconstruction Metrics",
        "",
        "Values are mean ± sample standard deviation over seeds.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        rendered = [
            str(row["dataset"]),
            str(row["method"]),
            str(row["partition"]),
            str(row["round"]),
            str(row["n"]),
        ]
        for metric_key in columns[5:]:
            rendered.append(_mean_std(row, metric_key))
        lines.append("| " + " | ".join(rendered) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std(row: dict[str, Any], metric_key: str) -> str:
    mean_value = row.get(f"{metric_key}_mean")
    std_value = row.get(f"{metric_key}_std")
    if mean_value == "":
        return ""
    return f"{float(mean_value):.4f} ± {float(std_value):.4f}"


if __name__ == "__main__":
    main()
