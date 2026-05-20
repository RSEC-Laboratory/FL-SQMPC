#!/usr/bin/env python3
"""SQMPC quantization-bit ablation.

For each uniform bit width B and seed, runs:
  (1) SQMPC FL training (final test accuracy under quantization B), and
  (2) FedAvg-TabLeak attack on a captured client update (reconstruction
      accuracy as the attacker's success measure).

Both share dataset, partition, seed, optimizer, and learning-rate; only the
per-layer quantization bits change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def parse_csv_ints(value: str) -> list[int]:
    items = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError("at least one value required")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "Bot_IoT_processed_no_pkSeqID_saddr_daddr")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--partition", choices=["iid", "non_iid"], default="iid")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bits", type=parse_csv_ints, default=parse_csv_ints("2,4,6,8,10,12,16"))
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--attack-round", type=int, default=5,
                        help="Round at which to capture the SQMPC update for TabLeak attack.")
    parser.add_argument("--attack-client", type=int, default=0)
    parser.add_argument("--attack-batch-size", type=int, default=4)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--ensemble-restarts", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--training-output-dir", type=Path,
                        default=ROOT / "output" / "results_20260512_sqmpc_bits_ablation")
    parser.add_argument("--tableak-artifact-dir", type=Path,
                        default=ROOT / "output" / "results_20260512_sqmpc_bits_ablation" / "tableak_artifacts")
    parser.add_argument("--tableak-results-dir", type=Path,
                        default=ROOT / "output" / "results_20260512_sqmpc_bits_ablation" / "tableak_results")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-attack", action="store_true")
    parser.add_argument("--q-b1-strict", action="store_true",
                        help="Pass --q-b1-strict to training and attack runners.")
    parser.add_argument("--include-float", action="store_true",
                        help="Also run a no-quantization (float) variant per seed.")
    parser.add_argument("--only-float", action="store_true",
                        help="Run only the no-quantization variant (skip the --bits sweep).")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    args.training_output_dir.mkdir(parents=True, exist_ok=True)
    args.tableak_artifact_dir.mkdir(parents=True, exist_ok=True)
    args.tableak_results_dir.mkdir(parents=True, exist_ok=True)

    bit_list = [] if args.only_float else list(args.bits)
    float_list = [True] if (args.include_float or args.only_float) else []

    for bits in bit_list:
        for seed in args.seeds:
            if not args.skip_training:
                cmd = [
                    sys.executable,
                    str(SCRIPTS / "run_fl_sqmpc_accuracy.py"),
                    "--mode", "sqmpc",
                    "--dataset", str(args.dataset_dir),
                    "--task", args.task,
                    "--partition", args.partition,
                    "--clients", str(args.clients),
                    "--rounds", str(args.rounds),
                    "--batch-size", str(args.batch_size),
                    "--learning-rate", str(args.learning_rate),
                    "--optimizer", "adam",
                    "--q-bits-first", str(bits),
                    "--q-bits-mid", str(bits),
                    "--q-bits-last", str(bits),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--output-dir", str(args.training_output_dir),
                ]
                if args.q_b1_strict:
                    cmd.append("--q-b1-strict")
                run_command(cmd)

            if not args.skip_attack:
                cmd = [
                    sys.executable,
                    str(SCRIPTS / "run_tableak_evaluation.py"),
                    "--method", "sqmpc",
                    "--dataset", str(args.dataset_dir),
                    "--task", args.task,
                    "--partition", args.partition,
                    "--clients", str(args.clients),
                    "--round", str(args.attack_round),
                    "--client", str(args.attack_client),
                    "--learning-rate", str(args.learning_rate),
                    "--optimizer", "adam",
                    "--q-bits-first", str(bits),
                    "--q-bits-mid", str(bits),
                    "--q-bits-last", str(bits),
                    "--attack-batch-size", str(args.attack_batch_size),
                    "--attack-steps", str(args.attack_steps),
                    "--ensemble-restarts", str(args.ensemble_restarts),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--artifact-dir", str(args.tableak_artifact_dir),
                    "--output-dir", str(args.tableak_results_dir),
                ]
                if args.q_b1_strict:
                    cmd.append("--q-b1-strict")
                run_command(cmd)

    for _ in float_list:
        for seed in args.seeds:
            if not args.skip_training:
                cmd = [
                    sys.executable,
                    str(SCRIPTS / "run_fl_sqmpc_accuracy.py"),
                    "--mode", "sqmpc",
                    "--dataset", str(args.dataset_dir),
                    "--task", args.task,
                    "--partition", args.partition,
                    "--clients", str(args.clients),
                    "--rounds", str(args.rounds),
                    "--batch-size", str(args.batch_size),
                    "--learning-rate", str(args.learning_rate),
                    "--optimizer", "adam",
                    "--no-quantize",
                    "--seed", str(seed),
                    "--device", args.device,
                    "--output-dir", str(args.training_output_dir),
                ]
                run_command(cmd)

            if not args.skip_attack:
                cmd = [
                    sys.executable,
                    str(SCRIPTS / "run_tableak_evaluation.py"),
                    "--method", "sqmpc",
                    "--dataset", str(args.dataset_dir),
                    "--task", args.task,
                    "--partition", args.partition,
                    "--clients", str(args.clients),
                    "--round", str(args.attack_round),
                    "--client", str(args.attack_client),
                    "--learning-rate", str(args.learning_rate),
                    "--optimizer", "adam",
                    "--no-quantize",
                    "--attack-batch-size", str(args.attack_batch_size),
                    "--attack-steps", str(args.attack_steps),
                    "--ensemble-restarts", str(args.ensemble_restarts),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--artifact-dir", str(args.tableak_artifact_dir),
                    "--output-dir", str(args.tableak_results_dir),
                ]
                run_command(cmd)


if __name__ == "__main__":
    main()
