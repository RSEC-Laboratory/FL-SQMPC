#!/usr/bin/env python3
"""Generate the 5-method FL training-time bar chart.

The HE/SQMPC/Vanilla/DP values are approximated from the manuscript
Fig. 10 screenshot. CIDIoT values are estimated by linearly scaling the
mean 30-round wall-clock runtimes from the implemented CIDIoT runs in
``output/results_20260512_method_comparison_r30``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROUNDS = (50, 100, 150)
DATASET_LABELS = {
    "bot_iot": "BotIoT Multiclass",
    "ciciot2023": "CICIoT2023 Multiclass",
    "ton_iot_extracted": "ToN-IoT Multiclass",
}
REFERENCE_DATASET_IDS = ("bot_iot", "ciciot2023")
DATASET_ROWS = {
    "bot_iot": 51130,
    "ciciot2023": 53249,
    "ton_iot_extracted": 57313,
}
METHOD_ORDER = ("Vanilla FL", "HE", "SMPC", "DP", "CIDIoT", "FL-SQMPC")
ROUND_STYLE = {
    50: {"label": "50 rounds", "color": "#1f77b4"},
    100: {"label": "100 rounds", "color": "#ff0000"},
    150: {"label": "150 rounds", "color": "#2ca02c"},
}


@dataclass(frozen=True)
class TimingRecord:
    dataset_id: str
    method: str
    rounds: int
    seconds: float
    source: str


BASE_TIMINGS = [
    # Existing Fig. 10 values approximated from the provided screenshot.
    TimingRecord("bot_iot", "HE", 50, 5.3, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "HE", 100, 12.6, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "HE", 150, 18.6, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "SMPC", 50, 0.9, "mirrors_sqmpc_timings"),
    TimingRecord("bot_iot", "SMPC", 100, 2.2, "mirrors_sqmpc_timings"),
    TimingRecord("bot_iot", "SMPC", 150, 3.6, "mirrors_sqmpc_timings"),
    TimingRecord("bot_iot", "FL-SQMPC", 50, 0.9, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "FL-SQMPC", 100, 2.2, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "FL-SQMPC", 150, 3.6, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "Vanilla FL", 50, 0.03, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "Vanilla FL", 100, 0.06, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "Vanilla FL", 150, 0.08, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "DP", 50, 0.06, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "DP", 100, 0.26, "fig10_screenshot_approx"),
    TimingRecord("bot_iot", "DP", 150, 0.53, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "HE", 50, 11.6, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "HE", 100, 24.3, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "HE", 150, 38.1, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "SMPC", 50, 0.8, "mirrors_sqmpc_timings"),
    TimingRecord("ciciot2023", "SMPC", 100, 2.0, "mirrors_sqmpc_timings"),
    TimingRecord("ciciot2023", "SMPC", 150, 3.5, "mirrors_sqmpc_timings"),
    TimingRecord("ciciot2023", "FL-SQMPC", 50, 0.8, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "FL-SQMPC", 100, 2.0, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "FL-SQMPC", 150, 3.5, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "Vanilla FL", 50, 0.03, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "Vanilla FL", 100, 0.05, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "Vanilla FL", 150, 0.08, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "DP", 50, 0.05, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "DP", 100, 0.18, "fig10_screenshot_approx"),
    TimingRecord("ciciot2023", "DP", 150, 0.27, "fig10_screenshot_approx"),
]


def load_cidiot_estimates(results_dir: Path) -> list[TimingRecord]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cfg = data.get("config", {})
        if not str(cfg.get("method", "")).startswith("cidiot"):
            continue
        if cfg.get("task") != "multiclass" or cfg.get("partition") != "iid":
            continue
        if int(cfg.get("rounds", 0)) != 30:
            continue
        dataset_id = str(cfg.get("dataset_id", ""))
        if dataset_id not in REFERENCE_DATASET_IDS:
            continue
        elapsed = data.get("final_metrics", {}).get("elapsed_seconds")
        if elapsed is not None:
            by_dataset[dataset_id].append(float(elapsed))

    records: list[TimingRecord] = []
    for dataset_id, seconds_values in sorted(by_dataset.items()):
        per_round = mean(seconds_values) / 30.0
        for rounds in ROUNDS:
            records.append(
                TimingRecord(
                    dataset_id,
                    "CIDIoT",
                    rounds,
                    per_round * rounds,
                    "estimated_from_cidiot_r30_elapsed_seconds",
                )
            )
    return records


def estimate_ton_iot_from_references(records: list[TimingRecord]) -> list[TimingRecord]:
    values = {
        (record.dataset_id, record.method, record.rounds): record.seconds
        for record in records
    }
    reference_size = mean(DATASET_ROWS[dataset_id] for dataset_id in REFERENCE_DATASET_IDS)
    ton_scale = DATASET_ROWS["ton_iot_extracted"] / reference_size

    ton_records: list[TimingRecord] = []
    for method in METHOD_ORDER:
        for rounds in ROUNDS:
            reference_values = [
                values[(dataset_id, method, rounds)]
                for dataset_id in REFERENCE_DATASET_IDS
            ]
            ton_records.append(
                TimingRecord(
                    "ton_iot_extracted",
                    method,
                    rounds,
                    mean(reference_values) * ton_scale,
                    "estimated_from_bot_iot_ciciot2023_size_scaled",
                )
            )
    return ton_records


def write_csv(records: list[TimingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset_id", "method", "rounds", "seconds", "source"])
        for record in records:
            writer.writerow(
                [
                    record.dataset_id,
                    record.method,
                    record.rounds,
                    f"{record.seconds:.6f}",
                    record.source,
                ]
            )


def plot_combined(records: list[TimingRecord], output_path: Path, yscale: str) -> None:
    values = {
        (record.dataset_id, record.method, record.rounds): record.seconds
        for record in records
    }

    dataset_ids = list(DATASET_LABELS)
    fig, axes = plt.subplots(1, len(dataset_ids), figsize=(7.0 * len(dataset_ids), 4.8), sharey=(yscale == "log"))
    axes = np.atleast_1d(axes)
    bar_width = 0.23
    x = np.arange(len(METHOD_ORDER))
    offsets = {50: -bar_width, 100: 0.0, 150: bar_width}

    for panel_idx, (ax, dataset_id) in enumerate(zip(axes, dataset_ids, strict=True)):
        for rounds in ROUNDS:
            heights = [
                values.get((dataset_id, method, rounds), np.nan)
                for method in METHOD_ORDER
            ]
            style = ROUND_STYLE[rounds]
            ax.bar(
                x + offsets[rounds],
                heights,
                width=bar_width,
                color=style["color"],
                label=style["label"],
                linewidth=0,
            )
        # ax.set_title(f"{DATASET_LABELS[dataset_id]} - Total time of FL process training", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, fontsize=12)
        ax.set_ylabel("time(s)")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=12, frameon=True)
        panel_label = f"({chr(ord('a') + panel_idx)})"
        ax.text(0.5, -0.22, panel_label, transform=ax.transAxes, ha="center", va="top", fontsize=16)
        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(0.02, None)
        else:
            ax.set_ylim(0.0, None)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_dataset(records: list[TimingRecord], output_path: Path, dataset_id: str, yscale: str) -> None:
    values = {
        (record.dataset_id, record.method, record.rounds): record.seconds
        for record in records
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bar_width = 0.23
    x = np.arange(len(METHOD_ORDER))
    offsets = {50: -bar_width, 100: 0.0, 150: bar_width}

    for rounds in ROUNDS:
        heights = [
            values.get((dataset_id, method, rounds), np.nan)
            for method in METHOD_ORDER
        ]
        style = ROUND_STYLE[rounds]
        ax.bar(
            x + offsets[rounds],
            heights,
            width=bar_width,
            color=style["color"],
            label=style["label"],
            linewidth=0,
        )

    # ax.set_title(f"{DATASET_LABELS[dataset_id]} - Total time of FL process training", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, fontsize=9)
    ax.set_ylabel("time(s)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(0.02, None)
    else:
        ax.set_ylim(0.0, None)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "output" / "results_20260512_method_comparison_r30",
        help="Directory containing CIDIoT r30 JSON files used for CIDIoT timing estimates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "figures_20260513_fl_time_5methods",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "output" / "tables" / "fl_time_5methods_multiclass.csv",
    )
    parser.add_argument("--yscale", choices=["log", "linear", "both"], default="both")
    parser.add_argument(
        "--layout",
        choices=["split", "combined", "both"],
        default="split",
        help="split writes one file per dataset; combined writes the two-panel figure.",
    )
    parser.add_argument("--formats", default="pdf,svg,png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(BASE_TIMINGS)
    cidiot_records = load_cidiot_estimates(args.results_dir)
    if len(cidiot_records) != len(REFERENCE_DATASET_IDS) * len(ROUNDS):
        raise RuntimeError(
            f"Expected CIDIoT estimates for {len(REFERENCE_DATASET_IDS)} datasets and {len(ROUNDS)} round counts, "
            f"found {len(cidiot_records)} records in {args.results_dir}."
        )
    records.extend(cidiot_records)
    records.extend(estimate_ton_iot_from_references(records))
    write_csv(records, args.csv_path)

    yscales = ("log", "linear") if args.yscale == "both" else (args.yscale,)
    formats = [part.strip().lower() for part in args.formats.split(",") if part.strip()]
    for yscale in yscales:
        for fmt in formats:
            if args.layout in {"combined", "both"}:
                output_path = args.output_dir / f"fl_time_5methods_multiclass_{yscale}.{fmt}"
                plot_combined(records, output_path, yscale)
                print(f"wrote {output_path}")
            if args.layout in {"split", "both"}:
                for dataset_id in DATASET_LABELS:
                    output_path = args.output_dir / f"fl_time_5methods_{dataset_id}_multiclass_{yscale}.{fmt}"
                    plot_dataset(records, output_path, dataset_id, yscale)
                    print(f"wrote {output_path}")
    print(f"wrote {args.csv_path}")


if __name__ == "__main__":
    main()
