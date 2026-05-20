from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import torch

from .artifacts import build_attack_context, recover_labels_from_bias_gradient
from .common import (
    ReconstructionResult,
    compute_parameter_gradients,
    l2_gradient_matching_loss,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IDLGConfig:
    steps: int = 3000
    learning_rate: float = 1e-2
    device: str = "cpu"
    continuous_bounds: tuple[float, float] = (-5.0, 5.0)


def _progress_interval(total_steps: int) -> int:
    return max(1, total_steps // 4)


def run_idlg(
    captured_update_path: str | Path,
    processed_root: str | Path | None = None,
    config: IDLGConfig | None = None,
) -> ReconstructionResult:
    config = config or IDLGConfig()
    context = build_attack_context(
        captured_update_path=captured_update_path,
        processed_root=processed_root,
        device=config.device,
    )
    recovered_labels = recover_labels_from_bias_gradient(context).to(context.device)
    dummy_features = torch.randn(
        context.captured_update.batch_size,
        context.input_dim,
        device=context.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([dummy_features], lr=config.learning_rate)
    objective_history: list[float] = []
    progress_interval = _progress_interval(config.steps)
    log.info(
        "iDLG: start artifact=%s steps=%d lr=%.4g batch_size=%d",
        Path(captured_update_path).name,
        config.steps,
        config.learning_rate,
        context.captured_update.batch_size,
    )

    for step in range(config.steps):
        optimizer.zero_grad()
        candidate_gradients = compute_parameter_gradients(context, dummy_features, recovered_labels)
        objective = l2_gradient_matching_loss(candidate_gradients, context.observed_gradients)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_features.clamp_(*config.continuous_bounds)
        objective_value = float(objective.item())
        objective_history.append(objective_value)
        if (
            step == 0
            or step + 1 == config.steps
            or (step + 1) % progress_interval == 0
        ):
            log.info(
                "iDLG: progress artifact=%s step=%d/%d objective=%.6f",
                Path(captured_update_path).name,
                step + 1,
                config.steps,
                objective_value,
            )

    log.info(
        "iDLG: done artifact=%s final_objective=%.6f",
        Path(captured_update_path).name,
        float(objective_history[-1]),
    )
    return ReconstructionResult(
        attack_name="idlg",
        reconstructed_features=dummy_features.detach().cpu(),
        reconstructed_labels=recovered_labels.detach().cpu(),
        final_objective=float(objective_history[-1]),
        objective_history=objective_history,
        metadata={
            **context.captured_update.metadata,
            "attack_name": "idlg",
            "attack_steps": config.steps,
            "attack_learning_rate": config.learning_rate,
            "label_recovery": "single-sample exact, batched bias-mass heuristic",
        },
    )
