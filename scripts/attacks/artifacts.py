from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ProcessedDatasetBundle:
    metadata: dict[str, Any]
    y_train: torch.Tensor
    input_dim: int
    num_classes: int


@dataclass(frozen=True)
class CapturedUpdate:
    metadata: dict[str, Any]
    global_model_state: dict[str, torch.Tensor]
    raw_update_tensors: dict[str, torch.Tensor]
    true_batch_labels: list[int]
    batch_indices: list[int]
    true_batch_features: torch.Tensor | None = None
    extra_tensors: dict[str, torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        if self.true_batch_labels:
            return len(self.true_batch_labels)
        return int(self.metadata["batch_size"])


@dataclass(frozen=True)
class AttackContext:
    capture_path: Path
    captured_update: CapturedUpdate
    dataset: ProcessedDatasetBundle
    architecture: str
    input_dim: int
    num_classes: int
    parameter_names: tuple[str, ...]
    observed_gradients: tuple[torch.Tensor, ...]
    device: torch.device


def load_captured_update(path: str | Path) -> CapturedUpdate:
    artifact = torch.load(Path(path), weights_only=False)
    required_keys = {
        "metadata",
        "global_model_state",
        "raw_update_tensors",
        "true_batch_labels",
        "batch_indices",
    }
    missing = required_keys.difference(artifact)
    if missing:
        raise KeyError(f"Captured update artifact is missing keys: {sorted(missing)}")
    return CapturedUpdate(
        metadata=dict(artifact["metadata"]),
        global_model_state={
            name: tensor.detach().cpu().clone()
            for name, tensor in artifact["global_model_state"].items()
        },
        raw_update_tensors={
            name: tensor.detach().cpu().clone()
            for name, tensor in artifact["raw_update_tensors"].items()
        },
        true_batch_labels=list(artifact["true_batch_labels"]),
        batch_indices=list(artifact["batch_indices"]),
        true_batch_features=(
            artifact["true_batch_features"].detach().cpu().clone()
            if "true_batch_features" in artifact
            else None
        ),
        extra_tensors={
            name: tensor.detach().cpu().clone()
            for name, tensor in artifact.get("extra_tensors", {}).items()
        },
    )


def load_processed_dataset(
    dataset: str | Path,
    processed_root: str | Path | None = None,
) -> ProcessedDatasetBundle:
    dataset_path = Path(dataset)
    if not dataset_path.exists() and processed_root is not None:
        candidate = Path(processed_root) / str(dataset)
        if candidate.exists():
            dataset_path = candidate
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset path does not exist: {dataset}")
    if not dataset_path.is_dir():
        raise ValueError(
            "FedAvg-TabLeak currently expects a processed tensor dataset directory."
        )

    meta_path = dataset_path / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    x_train = torch.load(dataset_path / "X_train.pt", map_location="cpu")
    y_train = torch.load(dataset_path / "y_train.pt", map_location="cpu").long()
    label_map = metadata.get("label_map", {})
    num_classes = len(label_map) if label_map else int(torch.max(y_train).item() + 1)
    return ProcessedDatasetBundle(
        metadata=metadata,
        y_train=y_train,
        input_dim=int(x_train.shape[1]),
        num_classes=int(num_classes),
    )


def build_attack_context(
    captured_update_path: str | Path,
    processed_root: str | Path | None = None,
    device: str = "cpu",
) -> AttackContext:
    capture_path = Path(captured_update_path)
    captured_update = load_captured_update(capture_path)
    metadata = captured_update.metadata
    dataset_id = str(metadata.get("dataset_path", metadata["dataset"]))
    architecture = str(metadata["architecture"])
    dataset = load_processed_dataset(dataset_id, processed_root=processed_root)

    attack_model = _create_attack_model(
        architecture=architecture,
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        metadata=metadata,
    )
    parameter_names = tuple(name for name, _ in attack_model.named_parameters())
    observed_gradients = _extract_observed_gradients(captured_update, parameter_names)
    return AttackContext(
        capture_path=capture_path,
        captured_update=captured_update,
        dataset=dataset,
        architecture=architecture,
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        parameter_names=parameter_names,
        observed_gradients=tuple(gradient.to(device) for gradient in observed_gradients),
        device=torch.device(device),
    )


def instantiate_attacked_model(context: AttackContext) -> torch.nn.Module:
    model = _create_attack_model(
        architecture=context.architecture,
        input_dim=context.input_dim,
        num_classes=context.num_classes,
        metadata=context.captured_update.metadata,
    )
    model.load_state_dict(context.captured_update.global_model_state, strict=True)
    model.train()
    if bool(context.captured_update.metadata.get("freeze_batchnorm", False)):
        _freeze_batchnorm_modules(model)
    return model.to(context.device)


def _freeze_batchnorm_modules(model: torch.nn.Module) -> None:
    import torch.nn as _nn
    for module in model.modules():
        if isinstance(module, _nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.track_running_stats = True


def _create_attack_model(
    architecture: str,
    input_dim: int,
    num_classes: int,
    metadata: dict[str, Any],
) -> torch.nn.Module:
    normalized = architecture.lower()
    if normalized in {"mlp", "fl_sqmpc_mlp", "sqmpc_mlp"}:
        hidden_sizes = metadata.get("hidden_sizes", [128, 64])
        if isinstance(hidden_sizes, str):
            hidden_sizes = [
                int(part.strip())
                for part in hidden_sizes.split(",")
                if part.strip()
            ]
        return _AttackMLP(
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_sizes=[int(size) for size in hidden_sizes],
        )
    if normalized in {"cidiot_tcn_discriminator", "tcn_discriminator"}:
        from run_cidiot_accuracy import TCNDiscriminator

        return TCNDiscriminator(
            feature_dim=input_dim,
            num_classes=num_classes,
            channels=int(metadata.get("tcn_channels", 32)),
            hidden_dim=int(metadata.get("hidden_dim", 64)),
        )
    raise ValueError(f"Unsupported attack model architecture: {architecture!r}")


class _AttackMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def recover_labels_from_bias_gradient(context: AttackContext) -> torch.Tensor:
    """Recover the batch label multiset from the captured gradient.

    B == 1 (exact):
        Uses the iDLG theorem (Zhao et al., 2020): the true label is the unique
        class with a negative final-bias gradient entry.  This is provably exact
        for cross-entropy loss with a softmax output.

    B > 1 (HEURISTIC — not the cited batch method):
        Uses a hybrid signal from the final-layer weight-gradient row minima
        (GradInversion-style presence detection) and the final-bias gradient mass
        to estimate how many times each class appears in the batch.

        This is NOT the batch label-recovery method cited by the TabLeak paper or
        any other primary reference.  It is an engineering approximation that works
        reasonably for small B on tabular IDS data but has no theoretical guarantee.
        Results for B > 1 should be treated as heuristic estimates and flagged as
        such in any paper discussion of label recovery accuracy.

        A principled batched implementation (e.g., the method from Yin et al., 2021
        or the iDLG batch extension) should replace this before any publication-grade
        B > 1 label-recovery analysis.
    """
    weight_name = _find_final_parameter_name(context.parameter_names, suffix="weight")
    bias_name = _find_final_parameter_name(context.parameter_names, suffix="bias")
    weight_index = context.parameter_names.index(weight_name)
    bias_index = context.parameter_names.index(bias_name)
    weight_gradient = context.observed_gradients[weight_index].detach().cpu()
    bias_gradient = context.observed_gradients[bias_index].detach().cpu()
    batch_size = context.captured_update.batch_size

    if batch_size == 1:
        # Exact iDLG label recovery (Zhao et al., 2020, Proposition 1).
        return torch.tensor([int(torch.argmin(bias_gradient).item())], dtype=torch.long)

    # HEURISTIC: everything below is a heuristic approximation for B > 1.
    # See docstring above for caveats.

    # Step 1: detect which classes are likely present using weight-gradient row minima.
    # A class whose row-minimum is negative has a non-trivial contribution to the
    # observed gradient, suggesting at least one sample from that class is in the batch.
    row_minima = torch.amin(weight_gradient, dim=1)
    presence_ranking = torch.argsort(row_minima, descending=False).tolist()
    initial_counts = torch.zeros_like(bias_gradient, dtype=torch.long)
    for class_id in presence_ranking:
        if row_minima[class_id] >= 0 or int(initial_counts.sum().item()) >= batch_size:
            break
        initial_counts[class_id] += 1

    if int(initial_counts.sum().item()) == 0:
        initial_counts[int(torch.argmin(bias_gradient).item())] = 1

    # Step 2: allocate remaining slots proportionally using a hybrid mass that
    # combines bias-gradient magnitude and weight-gradient row-minima.
    hybrid_mass = _build_hybrid_label_mass(
        bias_gradient=bias_gradient,
        row_minima=row_minima,
    )
    target_counts = _allocate_label_counts(
        mass=hybrid_mass,
        batch_size=batch_size,
        initial_counts=initial_counts,
    )

    labels: list[int] = []
    for class_id, count in enumerate(target_counts.tolist()):
        labels.extend([class_id] * count)
    labels = labels[:batch_size]
    if len(labels) < batch_size:
        fallback = int(torch.argmin(bias_gradient).item())
        labels.extend([fallback] * (batch_size - len(labels)))

    labels.sort()
    return torch.tensor(labels, dtype=torch.long)


def _extract_observed_gradients(
    captured_update: CapturedUpdate,
    parameter_names: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    metadata = captured_update.metadata
    update_kind = str(metadata.get("update_kind", ""))
    local_steps = int(metadata.get("local_steps", 1))
    learning_rate = float(metadata.get("learning_rate", 0.0))

    if update_kind != "gradient" or local_steps != 1:
        raise ValueError(
            "This phase-3 attack implementation currently supports only single-step "
            "captured gradients (`local_steps == 1`)."
        )
    if learning_rate <= 0:
        raise ValueError(f"Invalid learning rate in capture metadata: {learning_rate}")

    gradients: list[torch.Tensor] = []
    for name in parameter_names:
        if name not in captured_update.raw_update_tensors:
            raise KeyError(f"Captured update is missing parameter tensor: {name}")
        gradients.append((-captured_update.raw_update_tensors[name] / learning_rate).detach().cpu())
    return tuple(gradients)


def _build_hybrid_label_mass(
    bias_gradient: torch.Tensor,
    row_minima: torch.Tensor,
) -> torch.Tensor:
    bias_mass = torch.clamp(-bias_gradient, min=0.0)
    row_mass = torch.clamp(-row_minima, min=0.0)
    hybrid = bias_mass + row_mass
    if float(hybrid.sum().item()) > 0.0:
        return hybrid
    if float(bias_mass.sum().item()) > 0.0:
        return bias_mass
    return torch.softmax(-bias_gradient, dim=0)


def _allocate_label_counts(
    mass: torch.Tensor,
    batch_size: int,
    initial_counts: torch.Tensor,
) -> torch.Tensor:
    counts = initial_counts.clone()
    remaining = batch_size - int(counts.sum().item())
    if remaining <= 0:
        return counts

    total_mass = float(mass.sum().item())
    if total_mass <= 0.0:
        mass = torch.ones_like(mass)
        total_mass = float(mass.sum().item())

    expected_total = batch_size * mass / total_mass
    extra_expected = torch.clamp(expected_total - counts.to(expected_total.dtype), min=0.0)
    extra_counts = torch.floor(extra_expected).to(torch.long)

    capped_extra = torch.minimum(extra_counts, torch.full_like(extra_counts, remaining))
    counts += capped_extra
    remaining = batch_size - int(counts.sum().item())
    if remaining <= 0:
        return counts

    residual = torch.clamp(expected_total - counts.to(expected_total.dtype), min=0.0)
    ranking = torch.argsort(residual, descending=True).tolist()
    if not ranking:
        ranking = torch.argsort(mass, descending=True).tolist()
    for class_id in ranking:
        if remaining <= 0:
            break
        counts[class_id] += 1
        remaining -= 1

    return counts


def _find_final_parameter_name(parameter_names: tuple[str, ...], suffix: str) -> str:
    candidates = [name for name in parameter_names if name.endswith(suffix)]
    if not candidates:
        raise KeyError(f"Could not find a final parameter with suffix {suffix!r}")
    return candidates[-1]
