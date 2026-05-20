#!/usr/bin/env python3
"""Add predictive model accuracy metrics to FedAvg-TabLeak summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from run_cidiot_accuracy import TCNDiscriminator, evaluate_ensemble
from run_fl_sqmpc_accuracy import MLP, evaluate, prepare_dataset, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default="*bot_iot_no_pkseqid_saddr_daddr*_summary.json",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    summary_paths = sorted(args.results_dir.glob(args.pattern))
    if not summary_paths:
        raise FileNotFoundError(f"No summaries found under {args.results_dir} with {args.pattern!r}")

    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        artifact = torch.load(_resolve_path(summary["artifact_path"]), map_location="cpu", weights_only=False)
        metadata = summary["metadata"]
        prepared = prepare_dataset(
            Path(metadata["dataset_path"]),
            metadata["task"],
            metadata.get("max_rows"),
            int(metadata["seed"]),
            float(metadata.get("test_size", 0.30)),
            log_scale=bool(metadata.get("log_scale", False)),
        )
        pre_model = _build_model(metadata, prepared, device)
        pre_model.load_state_dict(artifact["global_model_state"], strict=True)
        pre_metrics = _evaluate_model(pre_model, metadata, prepared, device)

        post_state = _apply_delta_to_state(
            artifact["global_model_state"],
            artifact["raw_update_tensors"],
        )
        post_model = _build_model(metadata, prepared, device)
        post_model.load_state_dict(post_state, strict=True)
        post_metrics = _evaluate_model(post_model, metadata, prepared, device)

        summary["model_metrics"] = {
            "accuracy_definition": (
                "pre_attack_model is the model state the attacked client starts from; "
                "post_observed_client_model is that state plus the exact observed "
                "post-obfuscation update used as the Tableak target."
            ),
            "pre_attack_model": pre_metrics,
            "post_observed_client_model": post_metrics,
        }
        summary_path.write_text(
            json.dumps(_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        print(summary_path)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _build_model(metadata: dict[str, Any], prepared: Any, device: torch.device) -> torch.nn.Module:
    if metadata["architecture"] == "fl_sqmpc_mlp":
        model = MLP(
            input_dim=prepared.x_train.shape[1],
            output_dim=len(prepared.class_names),
            hidden_sizes=[int(size) for size in metadata["hidden_sizes"]],
        )
    elif metadata["architecture"] == "cidiot_tcn_discriminator":
        model = TCNDiscriminator(
            feature_dim=prepared.x_train.shape[1],
            num_classes=len(prepared.class_names),
            channels=int(metadata["tcn_channels"]),
            hidden_dim=int(metadata["hidden_dim"]),
        )
    else:
        raise ValueError(f"Unsupported architecture: {metadata['architecture']}")
    return model.to(device)


def _evaluate_model(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    prepared: Any,
    device: torch.device,
) -> dict[str, float]:
    if metadata["architecture"] == "fl_sqmpc_mlp":
        return evaluate(model, prepared.x_test, prepared.y_test, device)
    if metadata["architecture"] == "cidiot_tcn_discriminator":
        return evaluate_ensemble([model], prepared.x_test, prepared.y_test, device)
    raise ValueError(f"Unsupported architecture: {metadata['architecture']}")


def _apply_delta_to_state(
    pre_state: OrderedDict[str, torch.Tensor],
    delta: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    final_state = OrderedDict((name, tensor.clone()) for name, tensor in pre_state.items())
    for name, update in delta.items():
        final_state[name] = final_state[name] + update
    return final_state


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
