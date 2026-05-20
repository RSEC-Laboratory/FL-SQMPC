#!/usr/bin/env python3
"""Reconstruction-accuracy table driver.

Sweeps 5 obfuscation methods x 3 datasets x 2 partitions x 2 attack rounds
x N seeds, running TabLeak (gradient inversion) against a captured client
update for each cell. Designed to feed `generate_recon_table.py`.

Methods covered (all share Adam@lr=1e-3, batch=1 attack):
  vanilla_fl       SQMPC runner with --no-quantize (no obfuscation).
  dp_eps8          SQMPC runner with DP-SGD (Opacus, eps=8, delta=1e-5).
  dp_eps50         SQMPC runner with DP-SGD (Opacus, eps=50, delta=1e-5).
  cidiot           CIDIoT runner (TCN + CLS-GAN baseline).
  sqmpc_qb2        SQMPC runner with uniform 2-bit quantization.
  sqmpc_qb12       SQMPC runner with uniform 12-bit quantization.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

DATASET_PATHS = {
    "ciciot2023": ROOT / "ciciot2023_processed_tensor",
    "ton_iot":    ROOT / "ton_iot_processed_tensor",
    "bot_iot":    ROOT / "Bot_IoT_processed_minmax",
}

# Method label -> (runner script, fixed extra args).
METHODS: dict[str, tuple[str, list[str]]] = {
    "vanilla_fl":  ("run_tableak_evaluation.py", ["--method", "sqmpc", "--no-quantize"]),
    "dp_eps8":     ("run_tableak_evaluation.py", [
        "--method", "sqmpc", "--dpsgd-enabled",
        "--dp-epsilon", "8.0", "--dp-delta", "1e-5", "--dp-clip-norm", "0.5",
        "--dpsgd-rounds", "30",
    ]),
    "dp_eps50":    ("run_tableak_evaluation.py", [
        "--method", "sqmpc", "--dpsgd-enabled",
        "--dp-epsilon", "50.0", "--dp-delta", "1e-5", "--dp-clip-norm", "0.5",
        "--dpsgd-rounds", "30",
    ]),
    "cidiot":      ("run_tableak_evaluation.py", ["--method", "cidiot"]),
    "sqmpc_qb2":   ("run_tableak_evaluation.py", [
        "--method", "sqmpc",
        "--q-bits-first", "2", "--q-bits-mid", "2", "--q-bits-last", "2",
    ]),
    "sqmpc_qb12":  ("run_tableak_evaluation.py", [
        "--method", "sqmpc",
        "--q-bits-first", "12", "--q-bits-mid", "12", "--q-bits-last", "12",
    ]),
}


def parse_csv_strings(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=parse_csv_strings,
                        default=parse_csv_strings("ciciot2023,ton_iot,bot_iot"))
    parser.add_argument("--partitions", type=parse_csv_strings,
                        default=parse_csv_strings("iid,non_iid"))
    parser.add_argument("--methods", type=parse_csv_strings,
                        default=parse_csv_strings(",".join(METHODS.keys())))
    parser.add_argument("--rounds", type=parse_csv_ints, default=parse_csv_ints("1,10"))
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--attack-steps", type=int, default=1500)
    parser.add_argument("--ensemble-restarts", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    selected_methods = [m for m in args.methods if m in METHODS]
    for dataset_id in args.datasets:
        if dataset_id not in DATASET_PATHS:
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        dataset_path = DATASET_PATHS[dataset_id]
        for partition in args.partitions:
            for method_label in selected_methods:
                runner, fixed_args = METHODS[method_label]
                for round_id in args.rounds:
                    for seed in args.seeds:
                        cmd = [
                            sys.executable,
                            str(SCRIPTS / runner),
                            "--dataset", str(dataset_path),
                            "--task", "multiclass",
                            "--partition", partition,
                            "--clients", str(args.clients),
                            "--round", str(round_id),
                            "--client", "0",
                            "--learning-rate", str(args.learning_rate),
                            "--optimizer", "adam",
                            "--attack-batch-size", str(args.attack_batch_size),
                            "--attack-steps", str(args.attack_steps),
                            "--ensemble-restarts", str(args.ensemble_restarts),
                            "--seed", str(seed),
                            "--device", args.device,
                            "--artifact-dir", str(args.artifact_dir),
                            "--output-dir", str(args.results_dir),
                        ]
                        cmd.extend(fixed_args)
                        run_command(cmd)


if __name__ == "__main__":
    main()
