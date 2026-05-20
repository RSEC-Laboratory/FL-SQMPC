from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .artifacts import AttackContext, instantiate_attacked_model


@dataclass
class ReconstructionResult:
    attack_name: str
    reconstructed_features: torch.Tensor
    reconstructed_labels: torch.Tensor
    final_objective: float
    objective_history: list[float]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def save_reconstruction_result(
    result: ReconstructionResult,
    destination: str | Path,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "attack_name": result.attack_name,
            "reconstructed_features": result.reconstructed_features.detach().cpu(),
            "reconstructed_labels": result.reconstructed_labels.detach().cpu(),
            "final_objective": float(result.final_objective),
            "objective_history": [float(value) for value in result.objective_history],
            "metadata": dict(result.metadata),
            "diagnostics": _serialize_mapping(result.diagnostics),
        },
        destination_path,
    )
    return destination_path


def compute_parameter_gradients(
    context: AttackContext,
    candidate_features: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    model = instantiate_attacked_model(context)
    batchnorm_modules, original_batchnorm_states = _set_batchnorm_eval_for_singleton_batch(
        model,
        batch_size=int(candidate_features.shape[0]),
    )
    try:
        logits = model(candidate_features)
    finally:
        _restore_batchnorm_training_states(batchnorm_modules, original_batchnorm_states)
    loss = _compute_training_loss(logits, candidate_labels)
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        create_graph=True,
    )
    return tuple(gradient for gradient in gradients)


def l2_gradient_matching_loss(
    candidate_gradients: tuple[torch.Tensor, ...],
    observed_gradients: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return sum(
        (candidate - observed).pow(2).sum()
        for candidate, observed in zip(candidate_gradients, observed_gradients, strict=True)
    )


def cosine_gradient_matching_loss(
    candidate_gradients: tuple[torch.Tensor, ...],
    observed_gradients: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    dot_product = sum(
        (candidate * observed).sum()
        for candidate, observed in zip(candidate_gradients, observed_gradients, strict=True)
    )
    candidate_norm = torch.sqrt(
        sum(candidate.pow(2).sum() for candidate in candidate_gradients) + 1e-12
    )
    observed_norm = torch.sqrt(
        sum(observed.pow(2).sum() for observed in observed_gradients) + 1e-12
    )
    return 1.0 - dot_product / (candidate_norm * observed_norm + 1e-12)


def project_tabular_batch(
    batch: torch.Tensor,
    metadata: dict[str, Any],
    continuous_bounds: tuple[float, float],
) -> torch.Tensor:
    projected = batch.detach().clone()
    num_continuous = int(metadata.get("num_continuous_features", 0))
    if num_continuous:
        projected[:, :num_continuous] = projected[:, :num_continuous].clamp(*continuous_bounds)

    one_hot_ranges = metadata.get("one_hot_ranges", {})
    for feature_name in metadata.get("categorical_features", []):
        start, stop = one_hot_ranges[feature_name]
        group = projected[:, start:stop]
        winners = torch.argmax(group, dim=1)
        group.zero_()
        group.scatter_(1, winners.unsqueeze(1), 1.0)
    return projected


def relax_tabular_batch(
    latent: torch.Tensor,
    metadata: dict[str, Any],
    temperature: float,
    continuous_bounds: tuple[float, float],
) -> torch.Tensor:
    num_continuous = int(metadata.get("num_continuous_features", 0))
    parts: list[torch.Tensor] = []
    if num_continuous:
        parts.append(latent[:, :num_continuous].clamp(*continuous_bounds))

    one_hot_ranges = metadata.get("one_hot_ranges", {})
    for feature_name in metadata.get("categorical_features", []):
        start, stop = one_hot_ranges[feature_name]
        parts.append(
            torch.softmax(
                latent[:, start:stop] / max(temperature, 1e-6),
                dim=1,
            )
        )

    if not parts:
        return latent.clone()
    return torch.cat(parts, dim=1)


def _compute_training_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.dtype.is_floating_point:
        return -(labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    return F.cross_entropy(logits, labels)


def _serialize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = value.detach().cpu()
        elif isinstance(value, dict):
            serialized[key] = _serialize_mapping(value)
        elif isinstance(value, list):
            serialized[key] = [
                item.detach().cpu() if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def _set_batchnorm_eval_for_singleton_batch(
    model: torch.nn.Module,
    batch_size: int,
) -> tuple[list[torch.nn.modules.batchnorm._BatchNorm], list[bool]]:
    if batch_size > 1:
        return [], []
    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    original_states = [module.training for module in batchnorm_modules]
    for module in batchnorm_modules:
        module.eval()
    return batchnorm_modules, original_states


def _restore_batchnorm_training_states(
    batchnorm_modules: list[torch.nn.modules.batchnorm._BatchNorm],
    original_states: list[bool],
) -> None:
    for module, state in zip(batchnorm_modules, original_states, strict=True):
        module.train(state)
