from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .artifacts import build_attack_context, recover_labels_from_bias_gradient
from .common import (
    ReconstructionResult,
    compute_parameter_gradients,
    cosine_gradient_matching_loss,
    project_tabular_batch,
    relax_tabular_batch,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TabLeakConfig:
    steps: int = 1500
    learning_rate: float = 1e-2
    ensemble_restarts: int = 32
    temperature_start: float = 1.0
    temperature_end: float = 0.01
    device: str = "cpu"
    continuous_bounds: tuple[float, float] = (-5.0, 5.0)


def _progress_interval(total_units: int) -> int:
    return max(1, total_units // 8)


def run_tableak(
    captured_update_path: str | Path,
    processed_root: str | Path | None = None,
    config: TabLeakConfig | None = None,
) -> ReconstructionResult:
    config = config or TabLeakConfig()
    context = build_attack_context(
        captured_update_path=captured_update_path,
        processed_root=processed_root,
        device=config.device,
    )
    recovered_labels = recover_labels_from_bias_gradient(context).to(context.device)
    metadata = context.dataset.metadata

    restart_batches: list[torch.Tensor] = []
    restart_losses: list[float] = []
    restart_histories: list[list[float]] = []
    total_units = max(1, config.ensemble_restarts * config.steps)
    progress_interval = _progress_interval(total_units)
    completed_units = 0
    log.info(
        "TabLeak: start artifact=%s restarts=%d steps=%d lr=%.4g batch_size=%d",
        Path(captured_update_path).name,
        config.ensemble_restarts,
        config.steps,
        config.learning_rate,
        context.captured_update.batch_size,
    )

    for restart_index in range(config.ensemble_restarts):
        log.info(
            "TabLeak: restart %d/%d start artifact=%s",
            restart_index + 1,
            config.ensemble_restarts,
            Path(captured_update_path).name,
        )
        latent = torch.randn(
            context.captured_update.batch_size,
            context.input_dim,
            device=context.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([latent], lr=config.learning_rate)
        objective_history: list[float] = []

        for step in range(config.steps):
            optimizer.zero_grad()
            temperature = _annealed_temperature(
                step=step,
                total_steps=config.steps,
                start=config.temperature_start,
                end=config.temperature_end,
            )
            relaxed_batch = relax_tabular_batch(
                latent=latent,
                metadata=metadata,
                temperature=temperature,
                continuous_bounds=config.continuous_bounds,
            )
            candidate_gradients = compute_parameter_gradients(context, relaxed_batch, recovered_labels)
            objective = cosine_gradient_matching_loss(candidate_gradients, context.observed_gradients)
            objective.backward()
            optimizer.step()
            objective_value = float(objective.item())
            objective_history.append(objective_value)
            completed_units += 1
            if (
                completed_units == 1
                or completed_units == total_units
                or completed_units % progress_interval == 0
            ):
                log.info(
                    "TabLeak: progress artifact=%s restart=%d/%d step=%d/%d overall=%d/%d objective=%.6f",
                    Path(captured_update_path).name,
                    restart_index + 1,
                    config.ensemble_restarts,
                    step + 1,
                    config.steps,
                    completed_units,
                    total_units,
                    objective_value,
                )

        hard_batch = project_tabular_batch(
            relax_tabular_batch(
                latent=latent.detach(),
                metadata=metadata,
                temperature=config.temperature_end,
                continuous_bounds=config.continuous_bounds,
            ),
            metadata=metadata,
            continuous_bounds=config.continuous_bounds,
        )
        restart_batches.append(hard_batch.detach().cpu())
        restart_losses.append(float(objective_history[-1]))
        restart_histories.append(objective_history)
        log.info(
            "TabLeak: restart %d/%d done artifact=%s final_objective=%.6f",
            restart_index + 1,
            config.ensemble_restarts,
            Path(captured_update_path).name,
            float(objective_history[-1]),
        )

    best_index = int(np.argmin(np.asarray(restart_losses)))
    best_batch = restart_batches[best_index]
    aligned_batches = [best_batch]
    for restart_index, batch in enumerate(restart_batches):
        if restart_index == best_index:
            continue
        aligned_batches.append(_align_batch_to_reference(batch, best_batch))

    pooled_batch, categorical_entropy, continuous_std = _pool_restarts(
        aligned_batches,
        metadata=metadata,
        continuous_bounds=config.continuous_bounds,
    )
    log.info(
        "TabLeak: done artifact=%s best_restart=%d/%d final_objective=%.6f",
        Path(captured_update_path).name,
        best_index + 1,
        config.ensemble_restarts,
        float(restart_losses[best_index]),
    )
    return ReconstructionResult(
        attack_name="tableak",
        reconstructed_features=pooled_batch.cpu(),
        reconstructed_labels=recovered_labels.detach().cpu(),
        final_objective=float(restart_losses[best_index]),
        objective_history=restart_histories[best_index],
        metadata={
            **context.captured_update.metadata,
            "attack_name": "tableak",
            "attack_steps": config.steps,
            "attack_learning_rate": config.learning_rate,
            "ensemble_restarts": config.ensemble_restarts,
            "temperature_start": config.temperature_start,
            "temperature_end": config.temperature_end,
            "label_recovery": "single-sample exact, batched bias-mass heuristic",
        },
        diagnostics={
            "restart_losses": restart_losses,
            "best_restart_index": best_index,
            "categorical_entropy": categorical_entropy,
            "continuous_std": continuous_std,
        },
    )


def _annealed_temperature(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    fraction = step / float(total_steps - 1)
    return float(start * ((end / start) ** fraction))


def _align_batch_to_reference(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if candidate.shape[0] <= 1:
        return candidate
    cost = torch.cdist(candidate, reference, p=2).cpu().numpy()
    row_indices, column_indices = linear_sum_assignment(cost)
    aligned = torch.empty_like(candidate)
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        aligned[column_index] = candidate[row_index]
    return aligned


def _pool_restarts(
    batches: list[torch.Tensor],
    metadata: dict[str, object],
    continuous_bounds: tuple[float, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    stacked = torch.stack(batches, dim=0)
    pooled = torch.zeros_like(stacked[0])
    num_continuous = int(metadata.get("num_continuous_features", 0))

    if num_continuous:
        pooled[:, :num_continuous] = stacked[:, :, :num_continuous].mean(dim=0).clamp(
            *continuous_bounds
        )
        continuous_std = stacked[:, :, :num_continuous].std(dim=0, unbiased=False)
    else:
        continuous_std = torch.empty((stacked.shape[1], 0), dtype=stacked.dtype)

    categorical_entropy: dict[str, torch.Tensor] = {}
    one_hot_ranges = metadata.get("one_hot_ranges", {})
    for feature_name in metadata.get("categorical_features", []):
        start, stop = one_hot_ranges[feature_name]
        votes = torch.argmax(stacked[:, :, start:stop], dim=2)
        winners: list[int] = []
        entropies: list[float] = []
        group_size = stop - start
        normalizer = np.log(max(group_size, 2))
        for row_index in range(votes.shape[1]):
            counts = torch.bincount(votes[:, row_index], minlength=group_size).to(torch.float32)
            probabilities = counts / counts.sum()
            entropy = -torch.sum(
                probabilities[probabilities > 0] * torch.log(probabilities[probabilities > 0])
            )
            entropies.append(float(entropy.item() / normalizer if normalizer > 0 else 0.0))
            winners.append(int(torch.argmax(counts).item()))

        pooled[:, start:stop] = 0.0
        pooled[
            torch.arange(pooled.shape[0]),
            torch.tensor(winners, dtype=torch.long) + start,
        ] = 1.0
        categorical_entropy[feature_name] = torch.tensor(entropies, dtype=torch.float32)

    return pooled, categorical_entropy, continuous_std
