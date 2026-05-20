from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from torch.func import functional_call

from .artifacts import (
    ProcessedDatasetBundle,
    load_captured_update,
    load_processed_dataset,
    _create_attack_model,
    _find_final_parameter_name,
    _build_hybrid_label_mass,
    _allocate_label_counts,
)
from .common import (
    ReconstructionResult,
    cosine_gradient_matching_loss,
    project_tabular_batch,
    relax_tabular_batch,
)
from .tableak import _align_batch_to_reference, _pool_restarts

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FedAvgTabLeakConfig:
    steps: int = 1000
    learning_rate: float = 1e-2
    ensemble_restarts: int = 8
    known_local_epochs: int | None = None
    known_local_batch_size: int | None = None
    temperature_start: float = 1.0
    temperature_end: float = 0.01
    epoch_matching_prior_weight: float = 0.1
    sign_trick: bool = False
    use_oracle_labels: bool = True
    device: str = "cpu"
    continuous_bounds: tuple[float, float] = (-5.0, 5.0)
    # When True, the simulated inner loop reuses the same oracle-label ordering
    # for every local epoch (Tableak Config 52 behaviour). This is required to
    # reproduce Tableak's decreasing-ARS-with-T trend for the survey.
    # Default False keeps the stronger survey attack that exploits per-epoch
    # batch scheduling recorded by the capture.
    tableak_compat: bool = False


@dataclass(frozen=True)
class FedAvgAttackContext:
    capture_path: Path
    captured_update: object
    dataset: ProcessedDatasetBundle
    model: torch.nn.Module
    original_params: OrderedDict[str, torch.Tensor]
    buffers: OrderedDict[str, torch.Tensor]
    observed_update: tuple[torch.Tensor, ...]
    local_epochs: int          # T (epochs or steps depending on step_mode)
    local_batch_size: int      # B
    latent_size: int           # rows per latent = batch_size in both modes
    step_mode: bool            # True: T steps cycling one latent; False: T epoch latents
    local_dataset_size: int    # full local dataset N
    labels: torch.Tensor       # cycling labels (T*B) in step_mode, sequential (N) in epoch mode
    epoch_label_sequences: tuple[torch.Tensor, ...] | None
    extra_tensors: dict[str, torch.Tensor]
    input_dim: int
    device: torch.device
    tableak_compat: bool       # if True, reuse `labels` across every epoch (Tableak parity)


def run_fedavg_tableak(
    captured_update_path: str | Path,
    processed_root: str | Path | None = None,
    config: FedAvgTabLeakConfig | None = None,
) -> ReconstructionResult:
    config = config or FedAvgTabLeakConfig()
    context = build_fedavg_attack_context(
        captured_update_path=captured_update_path,
        processed_root=processed_root,
        known_local_epochs=config.known_local_epochs,
        known_local_batch_size=config.known_local_batch_size,
        use_oracle_labels=config.use_oracle_labels,
        device=config.device,
        tableak_compat=config.tableak_compat,
    )
    metadata = context.dataset.metadata

    # In step_mode: ONE latent (the local dataset) cycled T times.
    # In epoch mode: T separate epoch latents (reference paper design).
    n_latents = 1 if context.step_mode else context.local_epochs

    restart_batches: list[torch.Tensor] = []
    restart_losses: list[float] = []
    restart_histories: list[list[float]] = []
    total_units = max(1, config.ensemble_restarts * config.steps)
    progress_interval = max(1, total_units // 8)
    completed_units = 0

    log.info(
        "FedAvg TabLeak: artifact=%s T=%d B=%d latent_size=%d n_latents=%d "
        "restarts=%d steps=%d step_mode=%s",
        context.capture_path.name,
        context.local_epochs,
        context.local_batch_size,
        context.latent_size,
        n_latents,
        config.ensemble_restarts,
        config.steps,
        context.step_mode,
    )

    for restart_index in range(config.ensemble_restarts):
        latents = [
            torch.randn(
                context.latent_size,
                context.input_dim,
                device=context.device,
                requires_grad=True,
            )
            for _ in range(n_latents)
        ]
        optimizer = torch.optim.Adam(latents, lr=config.learning_rate)
        objective_history: list[float] = []

        for step in range(config.steps):
            optimizer.zero_grad()
            temperature = _annealed_temperature(
                step=step,
                total_steps=config.steps,
                start=config.temperature_start,
                end=config.temperature_end,
            )
            relaxed_batches = [
                relax_tabular_batch(
                    latent=latent,
                    metadata=metadata,
                    temperature=temperature,
                    continuous_bounds=config.continuous_bounds,
                )
                for latent in latents
            ]
            candidate_update = _simulate_local_training_for_attack(
                context=context,
                candidate_batches=relaxed_batches,
            )
            objective = cosine_gradient_matching_loss(candidate_update, context.observed_update)

            # Epoch-matching prior only in epoch mode with multiple latents
            if (
                not context.step_mode
                and context.local_epochs > 1
                and config.epoch_matching_prior_weight > 0
            ):
                objective = objective + (
                    config.epoch_matching_prior_weight
                    * _epoch_matching_prior_mean_square_error(relaxed_batches)
                )

            objective.backward()
            if config.sign_trick:
                for latent in latents:
                    if latent.grad is not None:
                        latent.grad.sign_()
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
                    "FedAvg TabLeak: artifact=%s restart=%d/%d step=%d/%d "
                    "overall=%d/%d objective=%.6f",
                    context.capture_path.name,
                    restart_index + 1,
                    config.ensemble_restarts,
                    step + 1,
                    config.steps,
                    completed_units,
                    total_units,
                    objective_value,
                )

        hard_batches = [
            project_tabular_batch(
                relax_tabular_batch(
                    latent=latent.detach(),
                    metadata=metadata,
                    temperature=config.temperature_end,
                    continuous_bounds=config.continuous_bounds,
                ),
                metadata=metadata,
                continuous_bounds=config.continuous_bounds,
            )
            for latent in latents
        ]

        if context.step_mode:
            # Single latent — no epoch pooling needed
            pooled_batch = hard_batches[0]
        else:
            pooled_batch = _pool_epoch_batches(
                batches=hard_batches,
                metadata=metadata,
                continuous_bounds=config.continuous_bounds,
            )

        # In epoch mode use T copies of the pooled batch so the simulation
        # covers the same number of gradient steps as the observed update.
        n_final_batches = 1 if context.step_mode else context.local_epochs
        final_candidate_update = _simulate_local_training_for_attack(
            context=context,
            candidate_batches=[pooled_batch.to(context.device)] * n_final_batches,
        )
        final_objective = cosine_gradient_matching_loss(
            final_candidate_update,
            context.observed_update,
        )

        restart_batches.append(pooled_batch.detach().cpu())
        restart_losses.append(float(final_objective.item()))
        restart_histories.append(objective_history)

    best_index = int(torch.tensor(restart_losses).argmin().item())
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
    return ReconstructionResult(
        attack_name="fedavg_tableak",
        reconstructed_features=pooled_batch.cpu(),
        reconstructed_labels=context.labels[: context.captured_update.batch_size].detach().cpu(),
        final_objective=float(restart_losses[best_index]),
        objective_history=restart_histories[best_index],
        metadata={
            **context.captured_update.metadata,
            "attack_name": "fedavg_tableak",
            "attack_steps": config.steps,
            "attack_learning_rate": config.learning_rate,
            "ensemble_restarts": config.ensemble_restarts,
            "temperature_start": config.temperature_start,
            "temperature_end": config.temperature_end,
            "epoch_matching_prior_weight": config.epoch_matching_prior_weight,
            "label_mode": "oracle" if config.use_oracle_labels else "recovered",
        },
        diagnostics={
            "restart_losses": restart_losses,
            "best_restart_index": best_index,
            "categorical_entropy": categorical_entropy,
            "continuous_std": continuous_std,
            "local_epochs": context.local_epochs,
            "step_mode": context.step_mode,
        },
    )


def build_fedavg_attack_context(
    captured_update_path: str | Path,
    processed_root: str | Path | None = None,
    known_local_epochs: int | None = None,
    known_local_batch_size: int | None = None,
    use_oracle_labels: bool = True,
    device: str = "cpu",
    tableak_compat: bool = False,
) -> FedAvgAttackContext:
    capture_path = Path(captured_update_path)
    captured_update = load_captured_update(capture_path)
    metadata = captured_update.metadata
    if str(metadata.get("capture_kind", "")) != "fedavg_multistep":
        raise ValueError(
            "run_fedavg_tableak expects a fedavg_multistep artifact; got "
            f"{metadata.get('capture_kind', '<missing>')!r}"
        )
    if not bool(metadata.get("freeze_batchnorm", False)):
        raise ValueError(
            "run_fedavg_tableak currently requires freeze_batchnorm=True in the capture metadata"
        )

    resolved_local_epochs = (
        int(known_local_epochs)
        if known_local_epochs is not None
        else int(metadata["local_epochs"])
    )
    resolved_local_batch_size = (
        int(known_local_batch_size)
        if known_local_batch_size is not None
        else int(metadata["local_batch_size"])
    )

    step_mode = bool(metadata.get("steps_mode", False))
    # latent_size = number of unique training samples (comparison target)
    latent_size = captured_update.batch_size
    local_dataset_size = int(metadata.get("num_examples", captured_update.batch_size))

    dataset = load_processed_dataset(
        str(metadata.get("dataset_path", metadata["dataset"])),
        processed_root=processed_root,
    )
    model = _create_attack_model(
        architecture=str(metadata["architecture"]),
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        metadata=metadata,
    )
    cleaned_global_state = _strip_opacus_prefix(captured_update.global_model_state)
    cleaned_update_tensors = _strip_opacus_prefix(captured_update.raw_update_tensors)
    model.load_state_dict(cleaned_global_state, strict=True)
    model.eval()

    original_params = OrderedDict(
        (name, parameter.detach().to(device).clone().requires_grad_(True))
        for name, parameter in model.named_parameters()
    )
    buffers = OrderedDict(
        (name, buffer.detach().to(device))
        for name, buffer in model.named_buffers()
    )
    observed_update = tuple(
        cleaned_update_tensors[name].detach().to(device)
        for name in original_params
    )

    epoch_label_sequences: tuple[torch.Tensor, ...] | None = None

    # In step_mode use the full cycling label sequence (T*B entries) so the simulation
    # can look up the correct label for each cycling step.
    step_cycling = metadata.get("step_cycling_labels")
    if use_oracle_labels:
        if step_mode and step_cycling is not None:
            labels = torch.tensor(step_cycling, dtype=torch.long, device=device)
        else:
            labels = torch.tensor(captured_update.true_batch_labels, dtype=torch.long, device=device)
            epoch_label_sequences = _build_epoch_label_sequences(
                local_batch_schedule=metadata.get("local_batch_schedule"),
                train_targets=dataset.y_train,
                fallback_labels=labels,
                local_epochs=resolved_local_epochs,
                device=torch.device(device),
            )
    else:
        # In tableak_compat mode we reuse a single flat label ordering for all epochs,
        # so we don't need the per-epoch schedule and can proceed with recovered labels
        # even when shuffle=True.  Without tableak_compat, per-epoch label order is
        # unknowable from the model delta alone, so we reject that combination.
        if not step_mode and bool(metadata.get("shuffle", False)) and not tableak_compat:
            raise ValueError(
                "run_fedavg_tableak with use_oracle_labels=False and shuffle=True "
                "requires tableak_compat=True (the per-epoch label order is not "
                "recoverable from the final model delta alone)."
            )
        labels = _recover_labels_from_model_delta(
            captured_update=captured_update,
            local_dataset_size=local_dataset_size,
            num_classes=dataset.num_classes,
        ).to(device)
        if not step_mode:
            epoch_label_sequences = tuple(labels.clone() for _ in range(resolved_local_epochs))

    return FedAvgAttackContext(
        capture_path=capture_path,
        captured_update=captured_update,
        dataset=dataset,
        model=model.to(device),
        original_params=original_params,
        buffers=buffers,
        observed_update=observed_update,
        local_epochs=resolved_local_epochs,
        local_batch_size=resolved_local_batch_size,
        latent_size=latent_size,
        step_mode=step_mode,
        local_dataset_size=local_dataset_size,
        labels=labels,
        epoch_label_sequences=epoch_label_sequences,
        extra_tensors={
            name: tensor.to(device)
            for name, tensor in (captured_update.extra_tensors or {}).items()
        },
        input_dim=dataset.input_dim,
        device=torch.device(device),
        tableak_compat=tableak_compat,
    )


def _simulate_local_training_for_attack(
    context: FedAvgAttackContext,
    candidate_batches: list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    current_params = OrderedDict(
        (name, parameter)
        for name, parameter in context.original_params.items()
    )
    optimizer_state = _init_optimizer_state(context)
    optimizer_step = 0

    if context.step_mode:
        # One latent (candidate_batches[0]) representing the full covered local dataset.
        # Cycle through it for T steps in mini-batches of B — exactly mirroring the capture.
        dataset_latent = candidate_batches[0]
        B = context.local_batch_size
        n_covered = dataset_latent.shape[0]
        num_positions = n_covered // B
        if num_positions <= 0:
            raise ValueError(
                "FedAvg-TabLeak step mode requires local_batch_size <= latent_size; "
                f"got B={B}, latent_size={n_covered}."
            )

        for step in range(context.local_epochs):
            pos = step % num_positions
            candidate_features = dataset_latent[pos * B : (pos + 1) * B]
            label_start = step * B
            candidate_labels = context.labels[label_start : label_start + B]

            loss = _compute_local_loss(
                context=context,
                current_params=current_params,
                candidate_features=candidate_features,
                candidate_labels=candidate_labels,
                step_index=step,
            )
            gradients = torch.autograd.grad(
                loss,
                tuple(current_params.values()),
                create_graph=True,
            )
            optimizer_step += 1
            current_params = _apply_optimizer_step(
                context=context,
                current_params=current_params,
                gradients=gradients,
                optimizer_state=optimizer_state,
                optimizer_step=optimizer_step,
            )
    else:
        # T epoch latents: each latent is the full local dataset processed in mini-batches.
        # In Tableak-compat mode we deliberately ignore the recorded per-epoch batch
        # schedule and reuse `context.labels` for every epoch so the simulation
        # mirrors Tableak Config 52 (no oracle knowledge of per-epoch shuffling).
        if context.tableak_compat:
            epoch_label_pool = context.labels
        else:
            if context.epoch_label_sequences is None:
                raise ValueError(
                    "Epoch-mode FedAvg attack context is missing epoch label sequences."
                )
            epoch_label_pool = None  # type: ignore[assignment]
        for epoch_index, batch in enumerate(candidate_batches):
            if context.tableak_compat:
                epoch_labels = epoch_label_pool
            else:
                epoch_labels = context.epoch_label_sequences[epoch_index]
            for start in range(0, batch.shape[0], context.local_batch_size):
                candidate_features = batch[start : start + context.local_batch_size]
                candidate_labels = epoch_labels[start : start + context.local_batch_size]
                step_index = optimizer_step

                loss = _compute_local_loss(
                    context=context,
                    current_params=current_params,
                    candidate_features=candidate_features,
                    candidate_labels=candidate_labels,
                    step_index=step_index,
                )
                gradients = torch.autograd.grad(
                    loss,
                    tuple(current_params.values()),
                    create_graph=True,
                )
                optimizer_step += 1
                current_params = _apply_optimizer_step(
                    context=context,
                    current_params=current_params,
                    gradients=gradients,
                    optimizer_state=optimizer_state,
                    optimizer_step=optimizer_step,
                )

    return tuple(
        current_params[name] - original_param
        for name, original_param in context.original_params.items()
    )


def _compute_local_loss(
    context: FedAvgAttackContext,
    current_params: OrderedDict[str, torch.Tensor],
    candidate_features: torch.Tensor,
    candidate_labels: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    loss_kind = str(context.captured_update.metadata.get("loss_kind", "cross_entropy"))
    output = functional_call(
        context.model,
        (current_params, context.buffers),
        (candidate_features,),
    )
    if loss_kind == "cross_entropy":
        logits = output[1] if isinstance(output, tuple) else output
        return torch.nn.functional.cross_entropy(logits, candidate_labels)

    if loss_kind == "cidiot_discriminator":
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("CIDIoT discriminator loss expects model output `(source, class_logits)`.")
        real_source, real_class = output
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        ce = torch.nn.functional.cross_entropy
        source_loss_weight = float(context.captured_update.metadata.get("source_loss_weight", 0.1))
        real_class_loss_weight = float(context.captured_update.metadata.get("real_class_loss_weight", 1.0))
        fake_class_loss_weight = float(context.captured_update.metadata.get("fake_class_loss_weight", 0.0))

        real_source_loss = bce(real_source, torch.ones_like(real_source))
        source_loss = real_source_loss
        class_loss = real_class_loss_weight * ce(real_class, candidate_labels)

        fake_features = context.extra_tensors.get("fake_features")
        fake_labels = context.extra_tensors.get("fake_labels")
        if fake_features is not None:
            fake_batch = _select_step_batch(fake_features, step_index, candidate_features.shape[0])
            fake_output = functional_call(
                context.model,
                (current_params, context.buffers),
                (fake_batch,),
            )
            if not isinstance(fake_output, tuple) or len(fake_output) != 2:
                raise TypeError("CIDIoT discriminator fake loss expects `(source, class_logits)`.")
            fake_source, fake_class = fake_output
            source_loss = source_loss + bce(fake_source, torch.zeros_like(fake_source))
            if fake_class_loss_weight > 0.0:
                if fake_labels is None:
                    raise KeyError("fake_class_loss_weight > 0 requires `fake_labels` in extra_tensors.")
                fake_y = _select_step_batch(fake_labels, step_index, candidate_features.shape[0]).long()
                class_loss = class_loss + fake_class_loss_weight * ce(fake_class, fake_y)

        return source_loss_weight * source_loss + class_loss

    raise ValueError(f"Unsupported FedAvg-TabLeak loss_kind: {loss_kind!r}")


def _select_step_batch(tensor: torch.Tensor, step_index: int, batch_size: int) -> torch.Tensor:
    if tensor.dim() == 0:
        raise ValueError("Step tensor must have at least one dimension.")
    if tensor.dim() >= 3:
        index = min(step_index, tensor.shape[0] - 1)
        return tensor[index]
    start = step_index * batch_size
    stop = start + batch_size
    if stop <= tensor.shape[0]:
        return tensor[start:stop]
    return tensor[:batch_size]


def _init_optimizer_state(
    context: FedAvgAttackContext,
) -> dict[str, OrderedDict[str, torch.Tensor]]:
    optimizer_name = str(context.captured_update.metadata.get("optimizer", "sgd")).lower()
    if optimizer_name != "adam":
        return {}
    return {
        "m": OrderedDict(
            (name, torch.zeros_like(param))
            for name, param in context.original_params.items()
        ),
        "v": OrderedDict(
            (name, torch.zeros_like(param))
            for name, param in context.original_params.items()
        ),
    }


def _apply_optimizer_step(
    context: FedAvgAttackContext,
    current_params: OrderedDict[str, torch.Tensor],
    gradients: tuple[torch.Tensor, ...],
    optimizer_state: dict[str, OrderedDict[str, torch.Tensor]],
    optimizer_step: int,
) -> OrderedDict[str, torch.Tensor]:
    metadata = context.captured_update.metadata
    learning_rate = float(metadata["learning_rate"])
    optimizer_name = str(metadata.get("optimizer", "sgd")).lower()

    if optimizer_name == "sgd":
        return OrderedDict(
            (name, param - learning_rate * grad)
            for (name, param), grad in zip(current_params.items(), gradients, strict=True)
        )

    if optimizer_name == "adam":
        beta1 = float(metadata.get("beta1", 0.9))
        beta2 = float(metadata.get("beta2", 0.999))
        eps = float(metadata.get("adam_eps", 1e-8))
        updated = OrderedDict()
        for (name, param), grad in zip(current_params.items(), gradients, strict=True):
            m_prev = optimizer_state["m"][name]
            v_prev = optimizer_state["v"][name]
            m_next = beta1 * m_prev + (1.0 - beta1) * grad
            v_next = beta2 * v_prev + (1.0 - beta2) * grad.pow(2)
            optimizer_state["m"][name] = m_next
            optimizer_state["v"][name] = v_next
            m_hat = m_next / (1.0 - beta1 ** optimizer_step)
            v_hat = v_next / (1.0 - beta2 ** optimizer_step)
            # Clamp the second moment inside sqrt for stable higher-order
            # differentiation during inversion. PyTorch Adam uses sqrt(v)+eps,
            # but sqrt'(0) can make the attack optimizer produce NaNs.
            denom = torch.sqrt(torch.clamp(v_hat, min=eps * eps)) + eps
            updated[name] = param - learning_rate * m_hat / denom
        return updated

    raise ValueError(f"Unsupported optimizer for FedAvg-TabLeak simulation: {optimizer_name!r}")


def _epoch_matching_prior_mean_square_error(candidate_batches: list[torch.Tensor]) -> torch.Tensor:
    if len(candidate_batches) <= 1:
        return torch.zeros((), device=candidate_batches[0].device)
    n_epochs = len(candidate_batches)
    n_features = candidate_batches[0].shape[1]
    epoch_means = torch.stack([batch.mean(dim=0) for batch in candidate_batches], dim=0)
    prior = torch.zeros((), device=epoch_means.device)
    for epoch_index in range(n_epochs):
        prior = prior + (
            (epoch_means - epoch_means[epoch_index]).pow(2).sum()
            / (n_epochs ** 2 * max(n_features, 1))
        )
    return prior


def _pool_epoch_batches(
    batches: list[torch.Tensor],
    metadata: dict[str, object],
    continuous_bounds: tuple[float, float],
) -> torch.Tensor:
    if len(batches) == 1:
        return batches[0]
    reference = batches[0]
    aligned = [reference]
    for batch in batches[1:]:
        aligned.append(_align_batch_to_reference(batch, reference))
    pooled, _, _ = _pool_restarts(
        aligned,
        metadata=metadata,
        continuous_bounds=continuous_bounds,
    )
    return pooled


def _annealed_temperature(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    fraction = step / float(total_steps - 1)
    return float(start * ((end / start) ** fraction))


def _build_epoch_label_sequences(
    local_batch_schedule: object,
    train_targets: torch.Tensor,
    fallback_labels: torch.Tensor,
    local_epochs: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(local_batch_schedule, list) or not local_batch_schedule:
        return tuple(fallback_labels.clone() for _ in range(local_epochs))

    epoch_label_sequences: list[torch.Tensor] = []
    for epoch_batches in local_batch_schedule:
        if not isinstance(epoch_batches, list):
            raise TypeError("local_batch_schedule entries must be lists of batch indices")
        flattened_indices = [int(index) for batch in epoch_batches for index in batch]
        labels = train_targets[flattened_indices].detach().to(device=device, dtype=torch.long)
        epoch_label_sequences.append(labels)

    if not epoch_label_sequences:
        return tuple(fallback_labels.clone() for _ in range(local_epochs))
    if len(epoch_label_sequences) < local_epochs:
        epoch_label_sequences.extend(
            epoch_label_sequences[-1].clone()
            for _ in range(local_epochs - len(epoch_label_sequences))
        )
    return tuple(epoch_label_sequences[:local_epochs])


def _recover_labels_from_model_delta(
    captured_update: object,
    local_dataset_size: int,
    num_classes: int,
) -> torch.Tensor:
    """Recover batch labels from a FedAvg model delta using the bias-gradient heuristic.

    The model delta is delta = theta_new - theta_old = -lr * sum_t(grad_t).
    Negating it gives a proxy for the cumulative gradient, to which we apply
    the same bias-mass heuristic used in the FedSGD TabLeak attack.
    For T=1 this is exact (up to lr scaling); for T>1 it is a heuristic.
    """
    parameter_names = tuple(captured_update.raw_update_tensors.keys())
    bias_name = _find_final_parameter_name(parameter_names, suffix="bias")
    weight_name = _find_final_parameter_name(parameter_names, suffix="weight")

    # Negate the delta to get a proxy for the cumulative gradient direction.
    bias_gradient = -captured_update.raw_update_tensors[bias_name].detach().cpu()
    weight_gradient = -captured_update.raw_update_tensors[weight_name].detach().cpu()

    if local_dataset_size == 1:
        return torch.tensor([int(torch.argmin(bias_gradient).item())], dtype=torch.long)

    row_minima = torch.amin(weight_gradient, dim=1)
    presence_ranking = torch.argsort(row_minima, descending=False).tolist()
    initial_counts = torch.zeros(num_classes, dtype=torch.long)
    for class_id in presence_ranking:
        if row_minima[class_id] >= 0 or int(initial_counts.sum().item()) >= local_dataset_size:
            break
        initial_counts[class_id] += 1

    if int(initial_counts.sum().item()) == 0:
        initial_counts[int(torch.argmin(bias_gradient).item())] = 1

    hybrid_mass = _build_hybrid_label_mass(
        bias_gradient=bias_gradient,
        row_minima=row_minima,
    )
    target_counts = _allocate_label_counts(
        mass=hybrid_mass,
        batch_size=local_dataset_size,
        initial_counts=initial_counts,
    )

    labels: list[int] = []
    for class_id, count in enumerate(target_counts.tolist()):
        labels.extend([class_id] * count)
    labels = labels[:local_dataset_size]
    if len(labels) < local_dataset_size:
        fallback = int(torch.argmin(bias_gradient).item())
        labels.extend([fallback] * (local_dataset_size - len(labels)))

    labels.sort()
    return torch.tensor(labels, dtype=torch.long)


def _strip_opacus_prefix(
    tensors: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, tensor in tensors.items():
        clean_key = key.replace("_module.", "", 1) if key.startswith("_module.") else key
        cleaned[clean_key] = tensor
    return cleaned
