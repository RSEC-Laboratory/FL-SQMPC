#!/usr/bin/env python3
"""Recompute reconstruction metrics for existing FedAvg-TabLeak summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_tableak_evaluation import compute_reconstruction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "output" / "tableak_results",
    )
    parser.add_argument(
        "--pattern",
        default="*bot_iot_no_pkseqid_saddr_daddr*_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = sorted(args.results_dir.glob(args.pattern))
    if not summary_paths:
        raise FileNotFoundError(f"No summaries found under {args.results_dir} with {args.pattern!r}")

    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        artifact = torch.load(ROOT / summary["artifact_path"], map_location="cpu", weights_only=False)
        reconstruction = torch.load(
            ROOT / summary["reconstruction_path"],
            map_location="cpu",
            weights_only=False,
        )
        metadata = summary["metadata"]
        metrics = compute_reconstruction_metrics(
            reconstructed_features=reconstruction["reconstructed_features"],
            reconstructed_labels=reconstruction["reconstructed_labels"],
            true_features=artifact["true_batch_features"],
            true_labels=torch.tensor(artifact["true_batch_labels"], dtype=torch.long),
            metadata=metadata["dataset_metadata"],
            dataset_path=ROOT / metadata["dataset_path"],
        )
        summary["metrics"] = metrics
        summary_path.write_text(
            json.dumps(_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        print(summary_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
