#!/usr/bin/env python3
"""Five-method comparison on multiclass FL accuracy.

Runs SQMPC (lr=2e-4 + lr=1e-3), CIDIoT, HE (= vanilla FedAvg), and DP
(= vanilla FedAvg + per-client L2 clip + Gaussian noise, accountant from RDP)
across the three datasets and {IID, Dirichlet alpha=0.3} partitions.

Output schema matches the other runners so the figure generator can join
results uniformly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

DATASET_PATHS = {
    "ciciot2023": ROOT / "ciciot2023_processed" / "CicIoT_extracted02.csv",
    "ton_iot_extracted": ROOT / "ToN_IoT_processed" / "ToN_IoT_extracted.csv",
    "bot_iot": ROOT / "Bot_IoT_processed",
}

# Each method definition: (label, runner_script, fixed_args)
METHODS = [
    (
        "sqmpc_lr2e4",
        "run_fl_sqmpc_accuracy.py",
        ["--mode", "sqmpc", "--optimizer", "adam", "--learning-rate", "0.0002"],
    ),
    (
        "sqmpc_lr1e3",
        "run_fl_sqmpc_accuracy.py",
        ["--mode", "sqmpc", "--optimizer", "adam", "--learning-rate", "0.001"],
    ),
    (
        "he",
        "run_fl_sqmpc_accuracy.py",
        ["--mode", "he", "--optimizer", "adam", "--learning-rate", "0.001"],
    ),
    (
        "dpsgd_eps8",
        "run_fl_sqmpc_accuracy.py",
        [
            "--mode", "dpsgd", "--optimizer", "adam", "--learning-rate", "0.0002",
            "--dp-epsilon", "8.0", "--dp-delta", "1e-5", "--dp-clip-norm", "1.0",
        ],
    ),
    (
        "dpsgd_eps50",
        "run_fl_sqmpc_accuracy.py",
        [
            "--mode", "dpsgd", "--optimizer", "adam", "--learning-rate", "0.0002",
            "--dp-epsilon", "50.0", "--dp-delta", "1e-5", "--dp-clip-norm", "1.0",
        ],
    ),
    (
        "cidiot",
        "run_cidiot_accuracy.py",
        [
            "--learning-rate", "0.001", "--beta1", "0.5", "--beta2", "0.999",
            "--latent-dim", "100", "--discriminator-steps", "5",
            "--supervised-pretrain-steps", "200",
            "--source-loss-weight", "0.0",
            "--fake-class-loss-weight", "1.0",
        ],
    ),
]


def parse_csv_strings(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=parse_csv_strings,
                        default=parse_csv_strings(",".join(DATASET_PATHS.keys())))
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-run-id", default="sqmpc-method-cmp-20260512")
    parser.add_argument("--methods", type=parse_csv_strings,
                        default=parse_csv_strings(",".join(label for label, _, _ in METHODS)),
                        help="Subset of methods to run. Defaults to all.")
    return parser.parse_args()


def log_scale_args(dataset_id: str) -> list[str]:
    return ["--log-scale"] if dataset_id == "ton_iot_extracted" else []


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = {label for label in args.methods}
    methods = [m for m in METHODS if m[0] in selected]

    for dataset_id in args.datasets:
        if dataset_id not in DATASET_PATHS:
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        dataset_path = DATASET_PATHS[dataset_id]
        for partition_spec in ("iid", "dirichlet"):
            for method_label, runner, fixed_args in methods:
                for seed in args.seeds:
                    cmd = [
                        sys.executable,
                        str(SCRIPTS / runner),
                        "--dataset", str(dataset_path),
                        "--task", "multiclass",
                        "--partition", partition_spec,
                        "--clients", str(args.clients),
                        "--rounds", str(args.rounds),
                        "--batch-size", str(args.batch_size),
                        "--seed", str(seed),
                        "--device", args.device,
                        "--comparison-run-id", args.comparison_run_id,
                        "--output-dir", str(args.output_dir),
                    ]
                    cmd.extend(fixed_args)
                    if partition_spec == "dirichlet":
                        cmd.extend(["--dirichlet-alpha", str(args.dirichlet_alpha)])
                    cmd.extend(log_scale_args(dataset_id))
                    print("running:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
