#!/usr/bin/env python3
"""Generate accuracy and loss plots from experiment result JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "output" / "results"
DEFAULT_PLOTS_DIR = ROOT / "output" / "plots"
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fl_sqmpc_accuracy import dataset_id_for_path

METHOD_LABELS = {
    "sqmpc": "FL-SQMPC",
    "cidiot_implemented": "CIDIoT",
    "vanilla": "Vanilla FL",
}

METHOD_COLORS = {
    "sqmpc": "#1f77b4",
    "cidiot_implemented": "#d62728",
    "vanilla": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--task", default="multiclass")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def method_key(result: dict[str, Any]) -> str:
    cfg = result["config"]
    return cfg.get("method", cfg.get("mode", "unknown"))


def result_dataset_id(result: dict[str, Any]) -> str:
    cfg = result["config"]
    return str(cfg.get("dataset_id") or dataset_id_for_path(Path(cfg["dataset"])))


def load_results(results_dir: Path, task: str, dataset_id: str | None, include_smoke: bool) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = result["config"]
        if cfg["task"] != task:
            continue
        if dataset_id is not None and result_dataset_id(result) != dataset_id:
            continue
        if cfg.get("max_rows") is not None and not include_smoke:
            continue
        result["_path"] = path
        results.append(result)
    return results


def series_for(result: dict[str, Any], metric: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in result.get("history", []):
        if metric in row:
            xs.append(int(row["round"]))
            ys.append(float(row[metric]))
    return xs, ys


def loss_series_for(result: dict[str, Any]) -> tuple[str | None, list[int], list[float]]:
    for metric, label in [
        ("eval_loss", "evaluation cross-entropy loss"),
        ("discriminator_class_loss", "discriminator class loss"),
        ("generator_loss", "generator loss"),
    ]:
        xs, ys = series_for(result, metric)
        if xs:
            return label, xs, ys
    return None, [], []


def plot_aggregated_lines(
    *,
    ax: Any,
    series_by_label: dict[tuple[str, str], list[tuple[list[int], list[float]]]],
) -> bool:
    plotted = False
    for (key, label_suffix), series_list in sorted(series_by_label.items()):
        by_round: dict[int, list[float]] = {}
        for xs, ys in series_list:
            for round_id, value in zip(xs, ys):
                by_round.setdefault(round_id, []).append(value)
        if not by_round:
            continue

        xs = sorted(by_round)
        means = np.array([np.mean(by_round[round_id]) for round_id in xs], dtype=float)
        stds = np.array(
            [np.std(by_round[round_id], ddof=1) if len(by_round[round_id]) > 1 else 0.0 for round_id in xs],
            dtype=float,
        )
        label = METHOD_LABELS.get(key, key)
        if label_suffix:
            label = f"{label} ({label_suffix})"
        seed_count = len(series_list)
        if seed_count > 1:
            # label = f"{label}, mean +/- std (n={seed_count})"
            label = f"{label}"

        color = METHOD_COLORS.get(key)
        ax.plot(xs, means, linewidth=2.0, label=label, color=color)
        if seed_count > 1:
            ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)
        plotted = True

    return plotted


def plot_metric(
    *,
    results: list[dict[str, Any]],
    partition: str,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    series_by_label: dict[tuple[str, str], list[tuple[list[int], list[float]]]] = {}
    all_values = []
    for result in results:
        cfg = result["config"]
        if cfg["partition"] != partition:
            continue
        key = method_key(result)
        xs, ys = series_for(result, metric)
        if not xs:
            continue
        all_values.extend(ys)
        series_by_label.setdefault((key, ""), []).append((xs, ys))

    plotted = plot_aggregated_lines(ax=ax, series_by_label=series_by_label)
    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"{ylabel} - {partition.replace('_', '-')}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    if all_values:
        bottom = max(0.0, min(all_values) - 0.05)
        top = min(1.0, max(all_values) + 0.05)
        if top - bottom < 0.1:
            center = (top + bottom) / 2.0
            bottom = max(0.0, center - 0.05)
            top = min(1.0, center + 0.05)
        ax.set_ylim(bottom=0.5, top=top)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_loss(
    *,
    results: list[dict[str, Any]],
    partition: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    series_by_label: dict[tuple[str, str], list[tuple[list[int], list[float]]]] = {}
    for result in results:
        cfg = result["config"]
        if cfg["partition"] != partition:
            continue
        key = method_key(result)
        loss_label, xs, ys = loss_series_for(result)
        if not xs or loss_label is None:
            continue
        series_by_label.setdefault((key, loss_label), []).append((xs, ys))

    plotted = plot_aggregated_lines(ax=ax, series_by_label=series_by_label)
    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"Loss - {partition.replace('_', '-')}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results_dir, args.task, args.dataset_id, args.include_smoke)
    dataset_ids = [args.dataset_id] if args.dataset_id else sorted({result_dataset_id(result) for result in results})

    for dataset_id in dataset_ids:
        dataset_results = [result for result in results if result_dataset_id(result) == dataset_id]
        partitions = sorted({result["config"]["partition"] for result in dataset_results})
        for partition in partitions:
            plot_metric(
                results=dataset_results,
                partition=partition,
                metric="accuracy",
                ylabel="Accuracy",
                output_path=args.plots_dir / f"accuracy_{dataset_id}_{args.task}_{partition}.png",
            )
            plot_loss(
                results=dataset_results,
                partition=partition,
                output_path=args.plots_dir / f"loss_{dataset_id}_{args.task}_{partition}.png",
            )

    print(f"wrote plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
