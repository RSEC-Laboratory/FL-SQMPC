#!/usr/bin/env python3
"""Run the SQMPC Dirichlet experiment matrix needed for SVG figures."""

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


def parse_csv_ints(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


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
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument(
        "--datasets",
        type=parse_csv_strings,
        default=parse_csv_strings(",".join(dataset_id for dataset_id, _ in DATASETS)),
        help="Comma-separated dataset ids to run: ciciot2023,ton_iot_extracted,bot_iot.",
    )
    parser.add_argument("--alphas", type=parse_csv_floats, default=parse_csv_floats("10.0,0.1"))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "results_20260511_sqmpc_dirichlet")
    parser.add_argument("--iid-results-dir", type=Path, default=ROOT / "output" / "results_20260511_sqmpc_figures")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "output" / "figures_20260511_sqmpc_dirichlet")
    parser.add_argument("--comparison-run-id", default="sqmpc-dirichlet-20260511")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--run-iid", action="store_true", help="Also run matching IID baselines into --output-dir.")
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def log_scale_args(dataset_id: str, requested: bool) -> list[str]:
    return ["--log-scale"] if requested or dataset_id == "ton_iot_extracted" else []


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        selected_datasets = {dataset_id for dataset_id in args.datasets}
        known_datasets = {dataset_id for dataset_id, _ in DATASETS}
        unknown_datasets = sorted(selected_datasets - known_datasets)
        if unknown_datasets:
            raise ValueError(f"unknown dataset ids: {unknown_datasets}")
        for dataset_id, dataset_path in DATASETS:
            if dataset_id not in selected_datasets:
                continue
            for task in TASKS:
                if args.run_iid:
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
                                "iid",
                                "--clients",
                                str(args.clients),
                                "--rounds",
                                str(args.rounds),
                                "--batch-size",
                                str(args.batch_size),
                                "--optimizer",
                                args.optimizer,
                                "--learning-rate",
                                str(args.learning_rate),
                                "--seed",
                                str(seed),
                                "--device",
                                args.device,
                                "--comparison-run-id",
                                args.comparison_run_id,
                                "--output-dir",
                                str(args.output_dir),
                            ]
                            + log_scale_args(dataset_id, args.log_scale)
                        )
                for alpha in args.alphas:
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
                                "dirichlet",
                                "--dirichlet-alpha",
                                str(alpha),
                                "--clients",
                                str(args.clients),
                                "--rounds",
                                str(args.rounds),
                                "--batch-size",
                                str(args.batch_size),
                                "--optimizer",
                                args.optimizer,
                                "--learning-rate",
                                str(args.learning_rate),
                                "--seed",
                                str(seed),
                                "--device",
                                args.device,
                                "--comparison-run-id",
                                args.comparison_run_id,
                                "--output-dir",
                                str(args.output_dir),
                            ]
                            + log_scale_args(dataset_id, args.log_scale)
                        )

    if not args.skip_figures:
        run_command(
            [
                sys.executable,
                str(SCRIPTS / "generate_sqmpc_dirichlet_figures.py"),
                "--results-dir",
                str(args.output_dir),
                "--iid-results-dir",
                str(args.iid_results_dir),
                "--figures-dir",
                str(args.figures_dir),
                "--datasets",
                ",".join(args.datasets),
                "--alphas",
                ",".join(str(alpha) for alpha in args.alphas),
                "--smooth-window",
                str(args.smooth_window),
                "--required-seeds",
                str(len(args.seeds)),
            ]
        )


if __name__ == "__main__":
    main()
