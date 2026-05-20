#!/usr/bin/env python3
"""Summarize Tableak reconstructions with range-normalized accuracy.

The attack is run once; this script computes post-hoc reconstruction metrics
from each saved reconstruction.pt + artifact.pt pair. A continuous feature is
counted as correct when

    |x_reconstructed - x_true| <= tolerance * (train_max_i - train_min_i).

Categorical features, when present, are scored by exact one-hot group match.
Zero-range continuous features are excluded from range-normalized accuracy and
reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]

DATASET_LABELS = {
    "ciciot2023_tensor": "CICIoT2023",
    "ton_iot_tensor": "ToN-IoT",
    "bot_iot_no_pkseqid_saddr_daddr": "Bot-IoT",
    "bot_iot_minmax": "Bot-IoT",
}

METHOD_ORDER = [
    "vanilla_fl",
    "sqmpc",
    "cidiot_no_dp",
    "cidiot_dp",
]

METHOD_LABELS = {
    "vanilla_fl": "Vanilla FL",
    "sqmpc": "SQMPC",
    "cidiot_no_dp": "CIDIoT no-DP",
    "cidiot_dp": "CIDIoT DP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--range-tolerance", type=float, default=0.30)
    return parser.parse_args()


def classify_method(metadata: dict[str, Any]) -> str | None:
    method = str(metadata.get("method", ""))
    surface = str(metadata.get("obfuscation_surface", ""))
    if method.startswith("sqmpc") and bool(metadata.get("no_quantize", False)):
        return "vanilla_fl"
    if method.startswith("sqmpc"):
        return "sqmpc"
    if method.startswith("cidiot") and surface == "dp_noisy_update":
        return "cidiot_dp"
    if method.startswith("cidiot"):
        return "cidiot_no_dp"
    return None


def resolve_path(path_text: str, fallback_dir: Path) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([ROOT / path, fallback_dir / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1] if candidates else path


def load_range(dataset_path: Path, num_continuous: int) -> torch.Tensor:
    x_train = torch.load(dataset_path / "X_train.pt", map_location="cpu", weights_only=False).float()
    continuous = x_train[:, :num_continuous]
    return continuous.max(dim=0).values - continuous.min(dim=0).values


def align_rows(reconstructed: torch.Tensor, true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cost = torch.cdist(reconstructed.float(), true.float(), p=2).numpy()
    rec_rows, true_rows = linear_sum_assignment(cost)
    return reconstructed[rec_rows].float(), true[true_rows].float()


def compute_range_metrics(
    summary: dict[str, Any],
    artifact_path: Path,
    reconstruction_path: Path,
    tolerance: float,
) -> dict[str, float]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    reconstruction = torch.load(reconstruction_path, map_location="cpu", weights_only=False)
    reconstructed, true = align_rows(
        reconstruction["reconstructed_features"].detach().cpu(),
        artifact["true_batch_features"].detach().cpu(),
    )
    metadata = artifact["metadata"]
    dataset_meta = metadata.get("dataset_metadata", {})
    num_continuous = int(dataset_meta.get("num_continuous_features", 0))
    one_hot_ranges = dataset_meta.get("one_hot_ranges", {})
    categorical_features = list(dataset_meta.get("categorical_features", []))

    metrics: dict[str, float] = {
        "range_tolerance": float(tolerance),
        "num_continuous_features": float(num_continuous),
        "num_range_scored_continuous_features": 0.0,
        "num_zero_range_continuous_features": 0.0,
        "continuous_acc_range": 0.0,
        "categorical_exact_match": 0.0,
        "reconstruction_acc_range": 0.0,
    }

    continuous_mask = torch.empty((true.shape[0], 0), dtype=torch.bool)
    if num_continuous:
        diff = (reconstructed[:, :num_continuous] - true[:, :num_continuous]).abs()
        feature_range = load_range(ROOT / metadata["dataset_path"], num_continuous)
        valid_range = feature_range > 1e-12
        metrics["num_range_scored_continuous_features"] = float(valid_range.sum().item())
        metrics["num_zero_range_continuous_features"] = float((~valid_range).sum().item())
        metrics["continuous_mae"] = float(diff.mean().item())
        metrics["continuous_rmse"] = float(torch.sqrt((diff**2).mean()).item())
        if valid_range.any():
            normalized_diff = diff[:, valid_range] / feature_range[valid_range].view(1, -1)
            continuous_mask = normalized_diff <= tolerance
            metrics["continuous_acc_range"] = float(continuous_mask.float().mean().item())
        else:
            metrics["continuous_acc_range"] = 0.0
    else:
        metrics["continuous_mae"] = 0.0
        metrics["continuous_rmse"] = 0.0

    categorical_masks = []
    for feature_name in categorical_features:
        start, stop = one_hot_ranges[feature_name]
        rec_choice = torch.argmax(reconstructed[:, start:stop], dim=1)
        true_choice = torch.argmax(true[:, start:stop], dim=1)
        categorical_masks.append(rec_choice == true_choice)
    if categorical_masks:
        categorical_mask = torch.stack(categorical_masks, dim=1)
        metrics["categorical_exact_match"] = float(categorical_mask.float().mean().item())
        categorical_flat = categorical_mask.reshape(-1)
    else:
        categorical_flat = torch.empty((0,), dtype=torch.bool)

    combined = torch.cat([continuous_mask.reshape(-1), categorical_flat])
    if combined.numel():
        metrics["reconstruction_acc_range"] = float(combined.float().mean().item())

    summary_metrics = summary.get("metrics", {})
    metrics["paper_reconstruction_accuracy"] = float(
        summary_metrics.get("reconstruction_accuracy", 0.0)
    )
    metrics["label_accuracy"] = float(summary_metrics.get("label_accuracy", 0.0))
    return metrics


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for summary_path in sorted(args.results_dir.glob("*_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping incomplete summary {summary_path}: {exc}", file=sys.stderr)
            continue
        metadata = summary.get("metadata", {})
        method_key = classify_method(metadata)
        if method_key is None:
            continue
        dataset_id = str(metadata.get("dataset", ""))
        partition = str(metadata.get("partition", ""))
        round_id = int(metadata.get("round", 0))
        artifact_path = resolve_path(summary["artifact_path"], args.artifact_dir)
        reconstruction_path = resolve_path(summary["reconstruction_path"], args.results_dir)
        try:
            metrics = compute_range_metrics(
                summary=summary,
                artifact_path=artifact_path,
                reconstruction_path=reconstruction_path,
                tolerance=args.range_tolerance,
            )
        except (OSError, RuntimeError, KeyError, EOFError, ValueError) as exc:
            print(f"warning: skipping incomplete reconstruction {summary_path}: {exc}", file=sys.stderr)
            continue
        row = {
            "dataset": DATASET_LABELS.get(dataset_id, dataset_id),
            "dataset_id": dataset_id,
            "method": METHOD_LABELS.get(method_key, method_key),
            "method_key": method_key,
            "partition": partition,
            "round": round_id,
            "seed": int(metadata.get("seed", -1)),
            **metrics,
        }
        rows.append(row)
        grouped[(dataset_id, method_key, partition, round_id)].append(row)

    aggregate_rows = []
    for key in sorted(grouped, key=_group_sort_key):
        values = grouped[key]
        first = values[0]
        aggregate: dict[str, Any] = {
            "dataset": first["dataset"],
            "dataset_id": first["dataset_id"],
            "method": first["method"],
            "method_key": first["method_key"],
            "partition": first["partition"],
            "round": first["round"],
            "n": len(values),
        }
        metric_keys = [
            "reconstruction_acc_range",
            "continuous_acc_range",
            "categorical_exact_match",
            "continuous_mae",
            "continuous_rmse",
            "paper_reconstruction_accuracy",
            "label_accuracy",
            "num_range_scored_continuous_features",
            "num_zero_range_continuous_features",
        ]
        for metric_key in metric_keys:
            metric_values = [float(item[metric_key]) for item in values]
            aggregate[f"{metric_key}_mean"] = mean(metric_values)
            aggregate[f"{metric_key}_std"] = stdev(metric_values) if len(metric_values) > 1 else 0.0
        aggregate_rows.append(aggregate)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, aggregate_rows)
    write_markdown(args.output_md, aggregate_rows, args.range_tolerance)
    print(args.output_csv)
    print(args.output_md)


def _group_sort_key(key: tuple[str, str, str, int]) -> tuple[int, int, str, int]:
    dataset, method, partition, round_id = key
    dataset_order = ["ciciot2023_tensor", "ton_iot_tensor", "bot_iot_no_pkseqid_saddr_daddr", "bot_iot_minmax"]
    return (
        dataset_order.index(dataset) if dataset in dataset_order else len(dataset_order),
        METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER),
        partition,
        round_id,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], tolerance: float) -> None:
    tol_label = f"Acc@{int(round(tolerance * 100))}% range"
    headers = [
        "Dataset",
        "Method",
        "Part.",
        "Round",
        "n",
        tol_label,
        "Cont. " + tol_label,
        "Cat. exact",
        "MAE",
        "RMSE",
        "Paper acc.",
        "Zero-range cont.",
    ]
    lines = [
        f"# Tableak Reconstruction Metrics ({tol_label})",
        "",
        "Values are mean +/- sample standard deviation over seeds. Zero-range continuous features are excluded from range-normalized accuracy.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    str(row["partition"]),
                    str(row["round"]),
                    str(row["n"]),
                    _mean_std_pct(row, "reconstruction_acc_range"),
                    _mean_std_pct(row, "continuous_acc_range"),
                    _mean_std_pct(row, "categorical_exact_match"),
                    _mean_std(row, "continuous_mae"),
                    _mean_std(row, "continuous_rmse"),
                    _mean_std_pct(row, "paper_reconstruction_accuracy"),
                    _mean_std(row, "num_zero_range_continuous_features"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std(row: dict[str, Any], key: str) -> str:
    return f"{float(row[f'{key}_mean']):.4f} +/- {float(row[f'{key}_std']):.4f}"


def _mean_std_pct(row: dict[str, Any], key: str) -> str:
    return f"{100.0 * float(row[f'{key}_mean']):.2f}% +/- {100.0 * float(row[f'{key}_std']):.2f}%"


if __name__ == "__main__":
    main()
