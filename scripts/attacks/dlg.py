from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import torch

from .artifacts import build_attack_context
from .common import (
    ReconstructionResult,
    compute_parameter_gradients,
    l2_gradient_matching_loss,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DLGConfig:
    steps: int = 3000
    learning_rate: float = 1e-2
    device: str = "cpu"
    continuous_bounds: tuple[float, float] = (-5.0, 5.0)


def _progress_interval(total_steps: int) -> int:
    return max(1, total_steps // 4)


def run_dlg(
    captured_update_path: str | Path,
    processed_root: str | Path | None = None,
    config: DLGConfig | None = None,
) -> ReconstructionResult:
    config = config or DLGConfig()
    context = build_attack_context(
        captured_update_path=captured_update_path,
        processed_root=processed_root,
        device=config.device,
    )

    dummy_features = torch.randn(
        context.captured_update.batch_size,
        context.input_dim,
        device=context.device,
        requires_grad=True,
    )
    dummy_label_logits = torch.randn(
        context.captured_update.batch_size,
        context.num_classes,
        device=context.device,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([dummy_features, dummy_label_logits], lr=config.learning_rate)
    objective_history: list[float] = []
    progress_interval = _progress_interval(config.steps)
    log.info(
        "DLG: start artifact=%s steps=%d lr=%.4g batch_size=%d",
        Path(captured_update_path).name,
        config.steps,
        config.learning_rate,
        context.captured_update.batch_size,
    )

    for step in range(config.steps):
        optimizer.zero_grad()
        soft_labels = torch.softmax(dummy_label_logits, dim=1)
        candidate_gradients = compute_parameter_gradients(context, dummy_features, soft_labels)
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
                "DLG: progress artifact=%s step=%d/%d objective=%.6f",
                Path(captured_update_path).name,
                step + 1,
                config.steps,
                objective_value,
            )

    reconstructed_features = dummy_features.detach().cpu()
    reconstructed_labels = torch.argmax(dummy_label_logits.detach(), dim=1).cpu()
    log.info(
        "DLG: done artifact=%s final_objective=%.6f",
        Path(captured_update_path).name,
        float(objective_history[-1]),
    )
    return ReconstructionResult(
        attack_name="dlg",
        reconstructed_features=reconstructed_features,
        reconstructed_labels=reconstructed_labels,
        final_objective=float(objective_history[-1]),
        objective_history=objective_history,
        metadata={
            **context.captured_update.metadata,
            "attack_name": "dlg",
            "attack_steps": config.steps,
            "attack_learning_rate": config.learning_rate,
        },
    )
