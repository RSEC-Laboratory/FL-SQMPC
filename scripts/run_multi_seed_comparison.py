#!/usr/bin/env python3
"""Run matched FL-SQMPC and CIDIoT comparisons for multiple seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_fl_sqmpc_accuracy import DEFAULT_DATASET, dataset_id_for_path


def parse_csv_ints(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one partition is required")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--partitions", type=parse_csv_strings, default=parse_csv_strings("iid,non_iid"))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--comparison-run-id", default=None)
    parser.add_argument("--skip-fl-sqmpc", action="store_true")
    parser.add_argument("--skip-cidiot", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--log-scale", action="store_true",
                        help="Apply log1p before MinMax scaling on features.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output dir for results (e.g. dated subdir).")
    parser.add_argument("--tables-dir", type=Path, default=None,
                        help="Override output dir for comparison tables.")
    parser.add_argument("--plots-dir", type=Path, default=None,
                        help="Override output dir for training curve plots.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def add_common_args(base: list[str], args: argparse.Namespace, partition: str, seed: int, run_id: str) -> list[str]:
    command = [
        *base,
        "--task",
        args.task,
        "--dataset",
        str(args.dataset),
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
        run_id,
    ]
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    if args.log_scale:
        command.append("--log-scale")
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    return command


def main() -> None:
    args = parse_args()
    run_id = args.comparison_run_id or datetime.now(timezone.utc).strftime("multiseed-%Y%m%dT%H%M%SZ")

    for seed in args.seeds:
        for partition in args.partitions:
            if not args.skip_fl_sqmpc:
                run_command(
                    add_common_args(
                        [sys.executable, str(SCRIPTS / "run_fl_sqmpc_accuracy.py"), "--mode", "sqmpc"],
                        args,
                        partition,
                        seed,
                        run_id,
                    )
                )
            if not args.skip_cidiot:
                run_command(
                    add_common_args(
                        [sys.executable, str(SCRIPTS / "run_cidiot_accuracy.py")],
                        args,
                        partition,
                        seed,
                        run_id,
                    )
                )

    compare_cmd = [sys.executable, str(SCRIPTS / "compare_accuracy.py")]
    if args.output_dir is not None:
        compare_cmd.extend(["--results-dir", str(args.output_dir)])
    if args.tables_dir is not None:
        compare_cmd.extend(["--tables-dir", str(args.tables_dir)])
    run_command(compare_cmd)

    if not args.skip_plots:
        plot_cmd = [
            sys.executable,
            str(SCRIPTS / "plot_training_curves.py"),
            "--task",
            args.task,
            "--dataset-id",
            dataset_id_for_path(args.dataset),
        ]
        if args.output_dir is not None:
            plot_cmd.extend(["--results-dir", str(args.output_dir)])
        if args.plots_dir is not None:
            plot_cmd.extend(["--plots-dir", str(args.plots_dir)])
        run_command(plot_cmd)


if __name__ == "__main__":
    main()
