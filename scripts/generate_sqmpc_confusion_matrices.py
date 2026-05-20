#!/usr/bin/env python3
"""Render confusion-matrix figures for SQMPC multiclass runs.

For each (dataset, partition) cell, averages the per-seed confusion matrices
stored in `final_metrics.confusion_matrix` (added by run_fl_sqmpc_accuracy.py
after 2026-05-18). Emits two separate SVGs per cell:
  * <name>_counts.svg     — mean integer counts across seeds
  * <name>_normalized.svg — row-normalized (recall per true class)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one item required")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        type=parse_csv_strings,
        default=[d.dataset_id for d in DATASETS],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Dirichlet alpha (used only to filter result files for the non-IID cell).",
    )
    parser.add_argument(
        "--required-seeds",
        type=int,
        default=3,
        help="Skip the cell if fewer than this many seeds are present.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in glob.glob(str(results_dir / "sqmpc_*multiclass*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        cfg = data.get("config", {})
        cm = data.get("final_metrics", {}).get("confusion_matrix")
        if cm is None:
            continue
        if cfg.get("mode") != "sqmpc" or cfg.get("task") != "multiclass":
            continue
        out.append(data)
    return out


def select_cell(
    results: list[dict[str, Any]],
    dataset_id: str,
    partition_kind: str,
    alpha: float,
) -> list[dict[str, Any]]:
    selected = []
    for d in results:
        cfg = d.get("config", {})
        if cfg.get("dataset_id") != dataset_id:
            continue
        part = cfg.get("partition")
        if partition_kind == "iid":
            if part != "iid":
                continue
        elif partition_kind == "dirichlet":
            if part != "dirichlet":
                continue
            if abs(float(cfg.get("dirichlet_alpha", -1.0)) - alpha) > 1e-9:
                continue
        else:
            continue
        selected.append(d)
    return selected


def stack_confusion_matrices(results: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    cms: list[np.ndarray] = []
    class_names: list[str] | None = None
    for d in results:
        cm = np.asarray(d["final_metrics"]["confusion_matrix"], dtype=np.int64)
        if class_names is None:
            class_names = list(d.get("class_names", []))
        cms.append(cm)
    if not cms:
        raise ValueError("no confusion matrices to stack")
    return np.stack(cms, axis=0), class_names or []


def _plot_single_panel(
    matrix: np.ndarray,
    annotations: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "d",
) -> None:
    n_classes = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal", vmin=vmin, vmax=vmax)
    # ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    mat_max = matrix.max() if matrix.size else 0.0
    thresh = (vmax if vmax is not None else mat_max) / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = annotations[i, j]
            text = format(val, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_confusion_matrices(
    cm_stack: np.ndarray,
    class_names: list[str],
    title_counts: str,
    title_normalized: str,
    output_path_counts: Path,
    output_path_normalized: Path,
) -> None:
    cm_counts_mean = cm_stack.mean(axis=0)
    cm_counts_int = np.rint(cm_counts_mean).astype(int)
    row_sums = cm_counts_mean.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_counts_mean,
        np.where(row_sums == 0, 1.0, row_sums),
        out=np.zeros_like(cm_counts_mean, dtype=float),
        where=row_sums != 0,
    )

    _plot_single_panel(
        matrix=cm_counts_mean,
        annotations=cm_counts_int,
        class_names=class_names,
        title=title_counts,
        output_path=output_path_counts,
        fmt="d",
    )
    _plot_single_panel(
        matrix=cm_norm,
        annotations=cm_norm,
        class_names=class_names,
        title=title_normalized,
        output_path=output_path_normalized,
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
    )


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(f"no SQMPC multiclass results with confusion matrices found in {args.results_dir}")

    cells = [("iid", "IID"), ("dirichlet", f"Non-IID Dirichlet ($\\alpha$={args.alpha:g})")]

    dataset_display = {d.dataset_id: d.display_name for d in DATASETS}

    for dataset_id in args.datasets:
        for partition_kind, partition_label in cells:
            selected = select_cell(results, dataset_id, partition_kind, args.alpha)
            if len(selected) < args.required_seeds:
                print(
                    f"skip {dataset_id} / {partition_kind}: have {len(selected)} seeds, "
                    f"need {args.required_seeds}",
                    flush=True,
                )
                continue
            try:
                cm_stack, class_names = stack_confusion_matrices(selected)
            except ValueError as exc:
                print(f"skip {dataset_id} / {partition_kind}: {exc}", flush=True)
                continue
            display = dataset_display.get(dataset_id, dataset_id)
            n_seeds = cm_stack.shape[0]
            title_counts = (
                f"{display} — {partition_label}\n"
                f"Mean test counts ({n_seeds} seeds)"
            )
            title_normalized = (
                f"{display} — {partition_label}\n"
                f"Row-normalized (recall per true class)"
            )
            partition_tag = "iid" if partition_kind == "iid" else f"dirichlet_a{args.alpha:g}".replace(".", "p")
            base = args.figures_dir / f"confusion_matrix_{dataset_id}_sqmpc_{partition_tag}"
            out_counts = base.with_name(base.name + "_counts.svg")
            out_normalized = base.with_name(base.name + "_normalized.svg")
            plot_confusion_matrices(
                cm_stack,
                class_names,
                title_counts,
                title_normalized,
                out_counts,
                out_normalized,
            )
            print(f"wrote {out_counts}", flush=True)
            print(f"wrote {out_normalized}", flush=True)


if __name__ == "__main__":
    main()
