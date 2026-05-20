#!/usr/bin/env python3
"""Generate SQMPC-only SVG accuracy and loss figures.

The output is one accuracy SVG and one loss SVG per dataset.  Each figure
contains one plot with four overlaid curves: Binary IID, Binary non-IID,
Multiclass IID, and Multiclass non-IID.  Curves are aggregated across seeds
as mean +/- sample standard deviation and smoothed for plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_fl_sqmpc_accuracy import dataset_id_for_path, requires_log_scale

DEFAULT_RESULTS_DIR = ROOT / "output" / "results_20260511_sqmpc_figures"
DEFAULT_FIGURES_DIR = ROOT / "output" / "figures_20260511_sqmpc"
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str


DATASETS = [
    DatasetSpec("ciciot2023", "CICIoT2023"),
    DatasetSpec("ton_iot_extracted", "ToN-IoT"),
    DatasetSpec("bot_iot", "Bot-IoT"),
]

PANEL_ORDER = [
    ("binary", "iid", "Binary IID"),
    ("binary", "non_iid", "Binary non-IID"),
    ("multiclass", "iid", "Multiclass IID"),
    ("multiclass", "non_iid", "Multiclass non-IID"),
]

PANEL_STYLES = {
    ("binary", "iid"): {"color": "#1f77b4", "linestyle": "-"},
    ("binary", "non_iid"): {"color": "#ff7f0e", "linestyle": "-"},
    ("multiclass", "iid"): {"color": "#2ca02c", "linestyle": "--"},
    ("multiclass", "non_iid"): {"color": "#d62728", "linestyle": "--"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--required-seeds", type=int, default=3)
    return parser.parse_args()


def method_key(result: dict[str, Any]) -> str:
    cfg = result["config"]
    return str(cfg.get("method", cfg.get("mode", "unknown")))


def result_dataset_id(result: dict[str, Any]) -> str:
    cfg = result["config"]
    return str(cfg.get("dataset_id") or dataset_id_for_path(Path(cfg["dataset"])))


def load_sqmpc_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = result["config"]
        if method_key(result) != "sqmpc":
            continue
        if cfg.get("max_rows") is not None:
            continue
        if requires_log_scale(result_dataset_id(result)) and not bool(cfg.get("log_scale", False)):
            continue
        result["_path"] = path
        results.append(result)
    return results


def metric_series(result: dict[str, Any], metric: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in result.get("history", []):
        if metric in row:
            xs.append(int(row["round"]))
            ys.append(float(row[metric]))
    return np.asarray(xs, dtype=int), np.asarray(ys, dtype=float)


def aggregate_series(results: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_round: dict[int, list[float]] = {}
    for result in results:
        xs, ys = metric_series(result, metric)
        for round_id, value in zip(xs, ys, strict=True):
            by_round.setdefault(int(round_id), []).append(float(value))

    if not by_round:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float)

    rounds = np.asarray(sorted(by_round), dtype=int)
    means = np.asarray([np.mean(by_round[round_id]) for round_id in rounds], dtype=float)
    stds = np.asarray(
        [
            np.std(by_round[round_id], ddof=1) if len(by_round[round_id]) > 1 else 0.0
            for round_id in rounds
        ],
        dtype=float,
    )
    return rounds, means, stds


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    window = min(int(window), len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def final_metric_stats(results: list[dict[str, Any]], metric: str) -> tuple[float, float, list[int]]:
    values = []
    seeds = []
    for result in results:
        cfg = result["config"]
        seeds.append(int(cfg["seed"]))
        if metric in result.get("final_metrics", {}):
            values.append(float(result["final_metrics"][metric]))
        else:
            _, ys = metric_series(result, metric)
            if len(ys):
                values.append(float(ys[-1]))
    if not values:
        return float("nan"), float("nan"), sorted(set(seeds))
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        sorted(set(seeds)),
    )


def plot_dataset_metric(
    *,
    dataset: DatasetSpec,
    results: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
    required_seeds: int,
) -> list[dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    summary_rows = []
    plotted_any = False
    global_lower: list[float] = []
    global_upper: list[float] = []

    for task, partition, title in PANEL_ORDER:
        panel_results = [
            result
            for result in results
            if result_dataset_id(result) == dataset.dataset_id
            and result["config"]["task"] == task
            and result["config"]["partition"] == partition
        ]
        rounds, means, stds = aggregate_series(panel_results, metric)
        final_mean, final_std, seeds = final_metric_stats(panel_results, metric)
        summary_rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset": dataset.display_name,
                "task": task,
                "partition": partition,
                "metric": metric,
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "final_mean": final_mean,
                "final_std": final_std,
            }
        )

        if len(rounds) == 0:
            continue

        smoothed_mean = smooth(means, smooth_window)
        smoothed_std = smooth(stds, smooth_window)
        lower = smoothed_mean - smoothed_std
        upper = smoothed_mean + smoothed_std
        if metric == "accuracy":
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

        style = PANEL_STYLES[(task, partition)]
        ax.plot(
            rounds,
            smoothed_mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=f"{title} ({final_mean:.4f} +/- {final_std:.4f})",
        )
        ax.fill_between(rounds, lower, upper, color=style["color"], alpha=0.11, linewidth=0)
        global_lower.extend(lower.tolist())
        global_upper.extend(upper.tolist())
        plotted_any = True

    ax.set_title(f"SQMPC {ylabel} - {dataset.display_name}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted_any:
        if metric == "accuracy":
            y_min = max(0.0, float(np.nanmin(global_lower)) - 0.03)
            y_max = min(1.0, float(np.nanmax(global_upper)) + 0.03)
            if y_max - y_min < 0.08:
                center = (y_max + y_min) / 2.0
                y_min = max(0.0, center - 0.04)
                y_max = min(1.0, center + 0.04)
            ax.set_ylim(y_min, y_max)
        else:
            y_min = max(0.0, float(np.nanmin(global_lower)) * 0.95)
            y_max = float(np.nanmax(global_upper)) * 1.08
            if y_max <= y_min:
                y_max = y_min + 1.0
            ax.set_ylim(y_min, y_max)
        ax.legend(frameon=False, fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "Missing results", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return summary_rows


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No SQMPC result rows found.\n", encoding="utf-8")
        return
    table = df.copy()
    for column in ["final_mean", "final_std"]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be positive")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_sqmpc_results(args.results_dir)
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        summary_rows.extend(
            plot_dataset_metric(
                dataset=dataset,
                results=results,
                metric="accuracy",
                ylabel="Accuracy",
                output_path=args.figures_dir / f"accuracy_{dataset.dataset_id}_sqmpc.svg",
                smooth_window=args.smooth_window,
                required_seeds=args.required_seeds,
            )
        )
        summary_rows.extend(
            plot_dataset_metric(
                dataset=dataset,
                results=results,
                metric="eval_loss",
                ylabel="Loss",
                output_path=args.figures_dir / f"loss_{dataset.dataset_id}_sqmpc.svg",
                smooth_window=args.smooth_window,
                required_seeds=args.required_seeds,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.figures_dir / "sqmpc_final_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, args.figures_dir / "sqmpc_final_metrics_summary.md")

    incomplete = summary[summary["seed_count"] < args.required_seeds]
    if not incomplete.empty:
        print("warning: some panels have fewer than the required seeds", file=sys.stderr)
        print(incomplete[["dataset_id", "task", "partition", "metric", "seed_count"]].to_string(index=False), file=sys.stderr)

    print(f"wrote SQMPC figures and summary tables to {args.figures_dir}")


if __name__ == "__main__":
    main()
