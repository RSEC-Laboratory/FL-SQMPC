#!/usr/bin/env python3
"""Generate SQMPC-only SVG figures for IID and Dirichlet client partitions."""

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

DEFAULT_RESULTS_DIR = ROOT / "output" / "results_20260511_sqmpc_dirichlet"
DEFAULT_IID_RESULTS_DIR = ROOT / "output" / "results_20260511_sqmpc_figures"
DEFAULT_FIGURES_DIR = ROOT / "output" / "figures_20260511_sqmpc_dirichlet"
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

TASKS = [
    ("binary", "Binary"),
    ("multiclass", "Multiclass"),
]

STYLE_BY_TASK_SERIES = {
    ("binary", "iid"): {"color": "#1f77b4", "linestyle": "-"},
    ("binary", "high"): {"color": "#111111", "linestyle": "-"},
    ("binary", "low"): {"color": "#ff7f0e", "linestyle": "-"},
    ("multiclass", "iid"): {"color": "#2ca02c", "linestyle": "--"},
    ("multiclass", "high"): {"color": "#6f6f6f", "linestyle": "--"},
    ("multiclass", "low"): {"color": "#d62728", "linestyle": "--"},
}


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one alpha is required")
    if any(alpha <= 0.0 for alpha in values):
        raise argparse.ArgumentTypeError("all alpha values must be positive")
    return values


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one dataset id is required")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--iid-results-dir", type=Path, default=DEFAULT_IID_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--datasets",
        type=parse_csv_strings,
        default=parse_csv_strings(",".join(dataset.dataset_id for dataset in DATASETS)),
        help="Comma-separated dataset ids to plot: ciciot2023,ton_iot_extracted,bot_iot.",
    )
    parser.add_argument("--alphas", type=parse_csv_floats, default=parse_csv_floats("10.0,0.1"))
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
        if cfg.get("partition") != "dirichlet":
            continue
        if cfg.get("max_rows") is not None:
            continue
        if requires_log_scale(result_dataset_id(result)) and not bool(cfg.get("log_scale", False)):
            continue
        result["_path"] = path
        results.append(result)
    return results


def load_iid_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = result["config"]
        if method_key(result) != "sqmpc":
            continue
        if cfg.get("partition") != "iid":
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


def alpha_label(alpha: float, alphas: list[float]) -> str:
    if len(alphas) == 1:
        return "low" if alpha <= 1.0 else "high"
    high = max(alphas)
    low = min(alphas)
    if np.isclose(alpha, high):
        return "high"
    if np.isclose(alpha, low):
        return "low"
    return f"a{alpha:g}"


def plot_dataset_metric(
    *,
    dataset: DatasetSpec,
    results: list[dict[str, Any]],
    iid_results: list[dict[str, Any]],
    alphas: list[float],
    metric: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
) -> list[dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    summary_rows = []
    plotted_any = False
    global_lower: list[float] = []
    global_upper: list[float] = []

    for task, task_label in TASKS:
        selected_iid = [
            result
            for result in iid_results
            if result_dataset_id(result) == dataset.dataset_id
            and result["config"]["task"] == task
            and result["config"]["partition"] == "iid"
        ]
        rounds, means, stds = aggregate_series(selected_iid, metric)
        final_mean, final_std, seeds = final_metric_stats(selected_iid, metric)
        summary_rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset": dataset.display_name,
                "task": task,
                "partition": "iid",
                "dirichlet_alpha": "",
                "dirichlet_concentration": "iid",
                "metric": metric,
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "final_mean": final_mean,
                "final_std": final_std,
            }
        )
        if len(rounds):
            smoothed_mean = smooth(means, smooth_window)
            smoothed_std = smooth(stds, smooth_window)
            lower = smoothed_mean - smoothed_std
            upper = smoothed_mean + smoothed_std
            if metric == "accuracy":
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

            style = STYLE_BY_TASK_SERIES[(task, "iid")]
            # label = f"{task_label} IID ({final_mean:.4f} +/- {final_std:.4f})"
            label = f"{task_label} IID"

            ax.plot(
                rounds,
                smoothed_mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                label=label,
            )
            ax.legend(fontsize=12.0)
            ax.fill_between(rounds, lower, upper, color=style["color"], alpha=0.08, linewidth=0)
            global_lower.extend(lower.tolist())
            global_upper.extend(upper.tolist())
            plotted_any = True

        for alpha in alphas:
            selected = [
                result
                for result in results
                if result_dataset_id(result) == dataset.dataset_id
                and result["config"]["task"] == task
                and np.isclose(float(result["config"].get("dirichlet_alpha")), alpha)
            ]
            rounds, means, stds = aggregate_series(selected, metric)
            final_mean, final_std, seeds = final_metric_stats(selected, metric)
            concentration = alpha_label(alpha, alphas)
            summary_rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset": dataset.display_name,
                    "task": task,
                    "partition": "dirichlet",
                    "dirichlet_alpha": alpha,
                    "dirichlet_concentration": concentration,
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

            style = STYLE_BY_TASK_SERIES.get((task, concentration), {"color": None, "linestyle": "-"})
            # label = f"{task_label} Dirichlet alpha={alpha:g} ({final_mean:.4f} +/- {final_std:.4f})"
            label = f"{task_label} non-IID"
            ax.plot(
                rounds,
                smoothed_mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=label,
            )
            ax.fill_between(rounds, lower, upper, color=style["color"], alpha=0.11, linewidth=0)
            global_lower.extend(lower.tolist())
            global_upper.extend(upper.tolist())
            plotted_any = True

    # ax.set_title(f"SQMPC IID and non-IID {ylabel} - {dataset.display_name}")
    # ax.set_title(f"{dataset.display_name}", fontsize=16.0)
    ax.set_xlabel("Communication round", fontsize=14.0)
    ax.set_ylabel(ylabel, fontsize=14.0)

    ax.tick_params(axis='both', labelsize=12)

    ax.grid(True, alpha=0.3)
    if plotted_any:
        if metric == "accuracy":
            y_min = max(0.0, float(np.nanmin(global_lower)) - 0.03)
            y_max = min(1.0, float(np.nanmax(global_upper)) + 0.03)
            if y_max - y_min < 0.08:
                center = (y_max + y_min) / 2.0
                y_min = max(0.0, center - 0.04)
                y_max = min(1.0, center + 0.04)
            # ax.set_ylim(y_min, y_max)
            ax.set_ylim(y_min, 1.0)
        else:
            y_min = max(0.0, float(np.nanmin(global_lower)) * 0.95)
            y_max = float(np.nanmax(global_upper)) * 1.08
            if y_max <= y_min:
                y_max = y_min + 1.0
            ax.set_ylim(y_min, y_max)
        ax.legend(frameon=False, fontsize=12.0, loc="best")
    else:
        ax.text(0.5, 0.5, "Missing results", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return summary_rows


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No SQMPC IID/Dirichlet result rows found.\n", encoding="utf-8")
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
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be positive")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_sqmpc_results(args.results_dir)
    iid_results = load_iid_results(args.iid_results_dir)
    summary_rows: list[dict[str, Any]] = []
    selected_datasets = {dataset_id for dataset_id in args.datasets}
    known_datasets = {dataset.dataset_id for dataset in DATASETS}
    unknown_datasets = sorted(selected_datasets - known_datasets)
    if unknown_datasets:
        raise ValueError(f"unknown dataset ids: {unknown_datasets}")
    for dataset in DATASETS:
        if dataset.dataset_id not in selected_datasets:
            continue
        summary_rows.extend(
            plot_dataset_metric(
                dataset=dataset,
                results=results,
                iid_results=iid_results,
                alphas=args.alphas,
                metric="accuracy",
                ylabel="Accuracy",
                output_path=args.figures_dir / f"accuracy_{dataset.dataset_id}_sqmpc_dirichlet.svg",
                smooth_window=args.smooth_window,
            )
        )
        summary_rows.extend(
            plot_dataset_metric(
                dataset=dataset,
                results=results,
                iid_results=iid_results,
                alphas=args.alphas,
                metric="eval_loss",
                ylabel="Loss",
                output_path=args.figures_dir / f"loss_{dataset.dataset_id}_sqmpc_dirichlet.svg",
                smooth_window=args.smooth_window,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.figures_dir / "sqmpc_dirichlet_final_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, args.figures_dir / "sqmpc_dirichlet_final_metrics_summary.md")

    incomplete = summary[summary["seed_count"] < args.required_seeds]
    if not incomplete.empty:
        print("warning: some panels have fewer than the required seeds", file=sys.stderr)
        print(
            incomplete[
                ["dataset_id", "task", "dirichlet_alpha", "metric", "seed_count"]
            ].to_string(index=False),
            file=sys.stderr,
        )

    print(f"wrote SQMPC Dirichlet figures and summary tables to {args.figures_dir}")


if __name__ == "__main__":
    main()
