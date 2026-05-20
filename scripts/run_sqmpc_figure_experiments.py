#!/usr/bin/env python3
"""Run the SQMPC experiment matrix needed for the SVG figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

DATASETS = [
    ("ciciot2023", ROOT / "ciciot2023_processed" / "CicIoT_extracted02.csv"),
    ("ton_iot_extracted", ROOT / "ToN_IoT_processed" / "ToN_IoT_extracted.csv"),
    ("bot_iot", ROOT / "Bot_IoT_processed"),
]
TASKS = ("binary", "multiclass")
PARTITIONS = ("iid", "non_iid")


def parse_csv_ints(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "results_20260511_sqmpc_figures")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "output" / "figures_20260511_sqmpc")
    parser.add_argument("--comparison-run-id", default="sqmpc-figures-20260511")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def log_scale_args(dataset_id: str) -> list[str]:
    return ["--log-scale"] if dataset_id == "ton_iot_extracted" else []


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        for dataset_id, dataset_path in DATASETS:
            for task in TASKS:
                for partition in PARTITIONS:
                    for seed in args.seeds:
                        run_command(
                            [
                                sys.executable,
                                str(SCRIPTS / "run_fl_sqmpc_accuracy.py"),
                                "--mode",
                                "sqmpc",
                                "--dataset",
                                str(dataset_path),
                                "--task",
                                task,
                                "--partition",
                                partition,
                                "--clients",
                                str(args.clients),
                                "--rounds",
                                str(args.rounds),
                                "--batch-size",
                                str(args.batch_size),
                                "--seed",
                                str(seed),
                                "--device",
                                args.device,
                                "--comparison-run-id",
                                args.comparison_run_id,
                                "--output-dir",
                                str(args.output_dir),
                            ]
                            + log_scale_args(dataset_id)
                        )

    if not args.skip_figures:
        run_command(
            [
                sys.executable,
                str(SCRIPTS / "generate_sqmpc_figures.py"),
                "--results-dir",
                str(args.output_dir),
                "--figures-dir",
                str(args.figures_dir),
                "--smooth-window",
                str(args.smooth_window),
                "--required-seeds",
                str(len(args.seeds)),
            ]
        )


if __name__ == "__main__":
    main()
