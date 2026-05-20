#!/usr/bin/env python3
"""Run FL-SQMPC and CIDIoT with matched training settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_fl_sqmpc_accuracy import DEFAULT_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--partitions", default="iid,non_iid")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--comparison-run-id", default=None)
    parser.add_argument("--skip-fl-sqmpc", action="store_true")
    parser.add_argument("--skip-cidiot", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def add_common_args(base: list[str], args: argparse.Namespace, partition: str, run_id: str) -> list[str]:
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
        str(args.seed),
        "--device",
        args.device,
        "--comparison-run-id",
        run_id,
    ]
    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    return command


def main() -> None:
    args = parse_args()
    run_id = args.comparison_run_id or datetime.now(timezone.utc).strftime("matched-training-%Y%m%dT%H%M%SZ")
    partitions = [part.strip() for part in args.partitions.split(",") if part.strip()]

    for partition in partitions:
        if not args.skip_fl_sqmpc:
            run_command(
                add_common_args(
                    [sys.executable, str(SCRIPTS / "run_fl_sqmpc_accuracy.py"), "--mode", "sqmpc"],
                    args,
                    partition,
                    run_id,
                )
            )
        if not args.skip_cidiot:
            run_command(
                add_common_args(
                    [sys.executable, str(SCRIPTS / "run_cidiot_accuracy.py")],
                    args,
                    partition,
                    run_id,
                )
            )

    run_command([sys.executable, str(SCRIPTS / "compare_accuracy.py")])


if __name__ == "__main__":
    main()
