#!/usr/bin/env python3
"""Plot 5-method accuracy comparison: 6 figures (3 datasets x {IID, Dirichlet a=0.3})."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATASET_LABELS = {
    "ciciot2023": "CICIoT2023",
    "ton_iot_extracted": "ToN-IoT",
    "bot_iot": "Bot-IoT",
}

# Plot order, style, and display label for each method.
METHOD_STYLE = {
    "sqmpc_lr1e3":  dict(label="FL-SQMPC",   color="#005AB5", linestyle="-"),
    "cidiot":       dict(label="CIDIoT",                    color="#008000", linestyle="-"),
    "he":           dict(label="HE (CKKS, exact)",          color="#DC3220", linestyle="--"),
    "dpsgd_eps8":   dict(label="DP-SGD (eps=8, delta=1e-5)",  color="#7B3294", linestyle=":"),
    "dpsgd_eps50":  dict(label="DP-SGD (eps=50, delta=1e-5)", color="#F17C05", linestyle=":"),
}
METHOD_ORDER = ["sqmpc_lr1e3", "cidiot", "he", "dpsgd_eps8", "dpsgd_eps50"]


@dataclass
class Run:
    method: str
    dataset_id: str
    partition: str  # "iid" or "dirichlet"
    dirichlet_alpha: float | None
    seed: int
    rounds: list[int]
    accuracy: list[float]
    eval_loss: list[float]


def classify_method(cfg: dict[str, Any]) -> str | None:
    method_field = cfg.get("method", "")
    if isinstance(method_field, str) and method_field.startswith("cidiot"):
        return "cidiot"
    mode = cfg.get("mode")
    lr = cfg.get("learning_rate")
    if mode == "he":
        if lr is not None and math.isclose(lr, 1e-3, rel_tol=1e-3):
            return "he"
        return None
    if mode == "dpsgd":
        eps = cfg.get("dp_epsilon")
        if eps is None:
            return None
        if math.isclose(eps, 8.0, rel_tol=1e-3):
            return "dpsgd_eps8"
        if math.isclose(eps, 50.0, rel_tol=1e-3):
            return "dpsgd_eps50"
        return None
    # Old client-level dp mode results are ignored on purpose.
    if mode == "dp":
        return None
    if mode == "sqmpc" and lr is not None:
        if math.isclose(lr, 2e-4, rel_tol=1e-3):
            return "sqmpc_lr2e4"
        if math.isclose(lr, 1e-3, rel_tol=1e-3):
            return "sqmpc_lr1e3"
    return None


def load_runs(results_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        cfg = data.get("config", {})
        if cfg.get("task") != "multiclass":
            continue
        method = classify_method(cfg)
        if method is None:
            continue
        history = data.get("history", [])
        if not history:
            continue
        rounds = [int(h["round"]) for h in history]
        accs = [float(h["accuracy"]) for h in history]
        losses = [float(h["eval_loss"]) for h in history]
        partition = cfg.get("partition", "iid")
        alpha = cfg.get("dirichlet_alpha")
        runs.append(Run(
            method=method,
            dataset_id=cfg.get("dataset_id", ""),
            partition=partition,
            dirichlet_alpha=alpha,
            seed=int(cfg.get("seed", -1)),
            rounds=rounds,
            accuracy=accs,
            eval_loss=losses,
        ))
    return runs


def aggregate(runs: list[Run], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    by_round: dict[int, list[float]] = defaultdict(list)
    seeds: set[int] = set()
    for run in runs:
        seeds.add(run.seed)
        values = run.accuracy if metric == "accuracy" else run.eval_loss
        for r, v in zip(run.rounds, values):
            by_round[r].append(v)
    if not by_round:
        return np.array([]), np.array([]), np.array([]), []
    rs = np.array(sorted(by_round.keys()), dtype=int)
    means = np.array([np.mean(by_round[r]) for r in rs], dtype=float)
    stds = np.array([np.std(by_round[r], ddof=1) if len(by_round[r]) > 1 else 0.0 for r in rs], dtype=float)
    return rs, means, stds, sorted(seeds)


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


def plot_one_panel(
    runs_by_method: dict[str, list[Run]],
    *,
    dataset_id: str,
    partition_label: str,
    metric: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    plotted = False
    for method in METHOD_ORDER:
        runs = runs_by_method.get(method, [])
        if not runs:
            continue
        rs, means, stds, seeds = aggregate(runs, metric)
        if rs.size == 0:
            continue
        style = METHOD_STYLE[method]
        sm = smooth(means, smooth_window)
        ss = smooth(stds, smooth_window)
        # label = f"{style['label']} (n={len(seeds)})"
        label = f"{style['label']}"
        ax.plot(rs, sm, label=label, color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
        ax.fill_between(rs, sm - ss, sm + ss, color=style["color"], alpha=0.15, linewidth=0)
        plotted = True
    title = f"{DATASET_LABELS.get(dataset_id, dataset_id)} - multiclass - {partition_label}"
    # ax.set_title(title)
    ax.set_xlabel("communication round", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="lower right" if metric == "accuracy" else "upper right", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="ciciot2023,ton_iot_extracted,bot_iot")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--also-loss", action="store_true",
                        help="Additionally emit eval_loss panels (6 extra figures).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    partition_specs = [
        ("iid", "IID"),
        ("dirichlet", f"Dirichlet alpha={args.dirichlet_alpha:g}"),
    ]

    for dataset_id in datasets:
        for partition_key, partition_label in partition_specs:
            runs_by_method: dict[str, list[Run]] = defaultdict(list)
            for run in runs:
                if run.dataset_id != dataset_id:
                    continue
                if run.partition != partition_key:
                    continue
                if partition_key == "dirichlet" and run.dirichlet_alpha is not None:
                    if not math.isclose(run.dirichlet_alpha, args.dirichlet_alpha, rel_tol=1e-3):
                        continue
                runs_by_method[run.method].append(run)

            acc_path = args.figures_dir / f"accuracy_{dataset_id}_multiclass_{partition_key}.svg"
            plot_one_panel(
                runs_by_method,
                dataset_id=dataset_id,
                partition_label=partition_label,
                metric="accuracy",
                ylabel="test accuracy",
                output_path=acc_path,
                smooth_window=args.smooth_window,
            )
            print(f"wrote {acc_path}")

            if args.also_loss:
                loss_path = args.figures_dir / f"loss_{dataset_id}_multiclass_{partition_key}.svg"
                plot_one_panel(
                    runs_by_method,
                    dataset_id=dataset_id,
                    partition_label=partition_label,
                    metric="eval_loss",
                    ylabel="cross-entropy loss",
                    output_path=loss_path,
                    smooth_window=args.smooth_window,
                )
                print(f"wrote {loss_path}")


if __name__ == "__main__":
    main()
