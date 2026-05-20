#!/usr/bin/env python3
"""Capture one post-obfuscation client update and run FedAvg-TabLeak."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import sqmpc_core
from attacks.common import save_reconstruction_result
from attacks.fedavg_tableak import FedAvgTabLeakConfig, run_fedavg_tableak
from run_cidiot_accuracy import ConditionalTCNGenerator, TCNDiscriminator, set_requires_grad
from run_fl_sqmpc_accuracy import (
    DEFAULT_DATASET,
    ExperimentConfig,
    MLP,
    aggregate_updates,
    apply_global_update,
    make_loaders,
    parse_hidden_sizes,
    partition_iid,
    partition_non_iid,
    prepare_dataset,
    select_device,
    set_seed,
    train_local,
)


@dataclass(frozen=True)
class CaptureResult:
    artifact_path: Path
    artifact: dict[str, Any]
    true_features: torch.Tensor
    true_labels: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sqmpc", "cidiot"], required=True)
    parser.add_argument("--obfuscation-surface", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--partition", choices=["iid", "non_iid"], default="iid")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=None)
    parser.add_argument("--attack-batch-size", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--log-scale", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--q-bits-first", type=int, default=12)
    parser.add_argument("--q-bits-mid", type=int, default=8)
    parser.add_argument("--q-bits-last", type=int, default=12)
    parser.add_argument("--q-b1-strict", action="store_true",
                        help="When B=1, force every element to +/-scale (strict 2-level).")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip SQMPC quantization on the captured update (float-precision attack).")
    parser.add_argument("--dpsgd-enabled", action="store_true",
                        help="Replace the captured step with a DP-SGD step (Opacus per-sample clip + Gaussian noise). "
                             "Skips SQMPC quantization automatically.")
    parser.add_argument("--dpsgd-rounds", type=int, default=30,
                        help="Total rounds T used by the RDP accountant to derive sigma from --dp-epsilon.")

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--label-dim", type=int, default=16)
    parser.add_argument("--tcn-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cidiot-beta1", type=float, default=0.5)
    parser.add_argument("--cidiot-beta2", type=float, default=0.999)
    parser.add_argument("--source-loss-weight", type=float, default=0.1)
    parser.add_argument("--real-class-loss-weight", type=float, default=1.0)
    parser.add_argument("--fake-class-loss-weight", type=float, default=0.0)
    parser.add_argument("--supervised-pretrain-steps", type=int, default=0)
    parser.add_argument("--cidiot-capture-mode", choices=["discriminator_delta", "generator_sample_gradient"],
                        default="discriminator_delta",
                        help="Which CIDIoT artifact to capture for the attack. "
                             "'discriminator_delta' (default, legacy) captures the local D_k weight delta "
                             "which CIDIoT never transmits over the wire (strictly stronger attacker). "
                             "'generator_sample_gradient' captures the per-sample gradient "
                             "∂L_D(G(z))/∂G(z) that CIDIoT actually uploads (Eqs. 23-24); "
                             "x_real does not appear directly in this signal.")
    parser.add_argument("--cidiot-checkpoint", type=Path, default=None,
                        help="Path to a .checkpoint.pt produced by run_cidiot_accuracy.py --save-checkpoint. "
                             "Required for --cidiot-capture-mode generator_sample_gradient (uses the "
                             "trained D_k and G); optional for discriminator_delta (replaces the "
                             "freshly-instantiated models otherwise).")
    parser.add_argument("--cidiot-dp-uplink", action="store_true",
                        help="When capturing generator_sample_gradient, apply Gaussian mechanism "
                             "(Eq. 24) with --dp-clip-norm and --dp-noise-multiplier "
                             "(or --dp-epsilon to derive sigma).")
    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-rdp-max-order", type=int, default=512)

    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--attack-lr", type=float, default=1e-2)
    parser.add_argument("--ensemble-restarts", type=int, default=4)
    parser.add_argument("--continuous-min", type=float, default=0.0)
    parser.add_argument("--continuous-max", type=float, default=1.0)
    parser.add_argument("--skip-attack", action="store_true")

    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "output" / "tableak_artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "tableak_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    local_batch_size = args.local_batch_size or args.attack_batch_size
    _validate_batch_shape(args.attack_batch_size, local_batch_size)

    prepared = prepare_dataset(
        args.dataset,
        args.task,
        args.max_rows,
        args.seed,
        args.test_size,
        log_scale=args.log_scale,
    )
    partitions = (
        partition_iid(prepared.y_train, args.clients, args.seed)
        if args.partition == "iid"
        else partition_non_iid(prepared.y_train, args.clients, args.num_shards, args.seed)
    )
    if args.client < 0 or args.client >= len(partitions):
        raise ValueError(f"--client must be in [0, {len(partitions) - 1}]")

    attack_indices = _select_attack_indices(
        partitions[args.client],
        args.attack_batch_size,
        args.seed,
        args.client,
        args.round,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "sqmpc":
        capture = capture_sqmpc_artifact(
            args=args,
            prepared=prepared,
            partitions=partitions,
            attack_indices=attack_indices,
            local_batch_size=local_batch_size,
            device=device,
        )
    else:
        if args.cidiot_capture_mode == "generator_sample_gradient":
            capture = capture_cidiot_sample_gradient(
                args=args,
                prepared=prepared,
                partitions=partitions,
                attack_indices=attack_indices,
                local_batch_size=local_batch_size,
                device=device,
            )
        else:
            capture = capture_cidiot_artifact(
                args=args,
                prepared=prepared,
                partitions=partitions,
                attack_indices=attack_indices,
                local_batch_size=local_batch_size,
                device=device,
            )

    if args.skip_attack:
        print(capture.artifact_path)
        return

    # Sample-gradient mode uses a separate input-gradient inversion attack
    # because the wire signal is `s = ∂L_D(G(z))/∂G(z)`, not a parameter delta.
    if args.method == "cidiot" and args.cidiot_capture_mode == "generator_sample_gradient":
        start_time = time.perf_counter()
        result = run_cidiot_sample_gradient_attack(
            capture=capture,
            attack_steps=args.attack_steps,
            attack_lr=args.attack_lr,
            ensemble_restarts=args.ensemble_restarts,
            device=device,
        )
        elapsed = time.perf_counter() - start_time
        reconstruction_path = args.output_dir / f"{capture.artifact_path.stem}_reconstruction.pt"
        save_reconstruction_result(result, reconstruction_path)
        metrics = compute_reconstruction_metrics(
            reconstructed_features=result.reconstructed_features,
            reconstructed_labels=result.reconstructed_labels,
            true_features=capture.true_features,
            true_labels=capture.true_labels,
            metadata=capture.artifact["metadata"]["dataset_metadata"],
            dataset_path=Path(capture.artifact["metadata"]["dataset_path"]),
        )
        summary = {
            "artifact_path": str(capture.artifact_path),
            "reconstruction_path": str(reconstruction_path),
            "elapsed_seconds": float(elapsed),
            "final_objective": float(result.final_objective),
            "metrics": metrics,
            "metadata": result.metadata,
            "diagnostics": _jsonable_diagnostics(result.diagnostics),
        }
        summary_path = args.output_dir / f"{capture.artifact_path.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary_path)
        return

    start_time = time.perf_counter()
    result = run_fedavg_tableak(
        captured_update_path=capture.artifact_path,
        processed_root=None,
        config=FedAvgTabLeakConfig(
            steps=args.attack_steps,
            learning_rate=args.attack_lr,
            ensemble_restarts=args.ensemble_restarts,
            known_local_epochs=int(capture.artifact["metadata"]["local_epochs"]),
            known_local_batch_size=local_batch_size,
            temperature_start=1.0,
            temperature_end=0.01,
            epoch_matching_prior_weight=0.0,
            use_oracle_labels=True,
            device=str(device),
            continuous_bounds=(args.continuous_min, args.continuous_max),
        ),
    )
    elapsed = time.perf_counter() - start_time

    reconstruction_path = args.output_dir / f"{capture.artifact_path.stem}_reconstruction.pt"
    save_reconstruction_result(result, reconstruction_path)
    metrics = compute_reconstruction_metrics(
        reconstructed_features=result.reconstructed_features,
        reconstructed_labels=result.reconstructed_labels,
        true_features=capture.true_features,
        true_labels=capture.true_labels,
        metadata=capture.artifact["metadata"]["dataset_metadata"],
        dataset_path=Path(capture.artifact["metadata"]["dataset_path"]),
    )
    summary = {
        "artifact_path": str(capture.artifact_path),
        "reconstruction_path": str(reconstruction_path),
        "elapsed_seconds": float(elapsed),
        "final_objective": float(result.final_objective),
        "metrics": metrics,
        "metadata": result.metadata,
        "diagnostics": _jsonable_diagnostics(result.diagnostics),
    }
    summary_path = args.output_dir / f"{capture.artifact_path.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


def capture_sqmpc_artifact(
    args: argparse.Namespace,
    prepared: Any,
    partitions: list[np.ndarray],
    attack_indices: np.ndarray,
    local_batch_size: int,
    device: torch.device,
) -> CaptureResult:
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    cfg = ExperimentConfig(
        dataset_id=prepared.dataset_id,
        dataset=str(args.dataset),
        task=args.task,
        partition=args.partition,
        mode="sqmpc",
        clients=args.clients,
        rounds=max(args.round - 1, 0),
        local_epochs=args.local_epochs,
        batch_size=local_batch_size,
        learning_rate=args.learning_rate,
        server_lr=1.0,
        optimizer=args.optimizer,
        hidden_sizes=hidden_sizes,
        test_size=prepared.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        num_shards=args.num_shards,
        dirichlet_alpha=None,
        log_scale=bool(getattr(args, "log_scale", False)),
        q_bits_first=args.q_bits_first,
        q_bits_mid=args.q_bits_mid,
        q_bits_last=args.q_bits_last,
        q_b1_strict=bool(args.q_b1_strict),
        no_quantize=bool(args.no_quantize),
        dp_epsilon=None,
        dp_delta=None,
        dp_clip_norm=None,
        dp_noise_multiplier=None,
        dp_adaptive_clipping=False,
        device=str(device),
        comparison_run_id=None,
    )
    model = MLP(
        input_dim=prepared.x_train.shape[1],
        output_dim=len(prepared.class_names),
        hidden_sizes=hidden_sizes,
    ).to(device)

    if args.round > 1:
        loaders = make_loaders(prepared.x_train, prepared.y_train, partitions, local_batch_size)
        client_sizes = [int(len(part)) for part in partitions]
        for _ in range(1, args.round):
            client_updates = [train_local(model, loader, cfg, device) for loader in loaders]
            aggregate = aggregate_updates(client_updates, client_sizes, cfg)
            apply_global_update(model, aggregate, 1.0, device)

    true_features = torch.from_numpy(prepared.x_train[attack_indices]).float()
    true_labels = torch.from_numpy(prepared.y_train[attack_indices]).long()
    pre_state = _state_dict_cpu(model)
    dpsgd_sigma = None
    if args.dpsgd_enabled:
        from dp_he import sigma_from_epsilon_rdp
        dpsgd_sigma = (
            float(args.dp_noise_multiplier)
            if args.dp_noise_multiplier is not None
            else sigma_from_epsilon_rdp(
                epsilon=args.dp_epsilon,
                steps=args.dpsgd_rounds,
                delta=args.dp_delta,
            )
        )
        print(
            f"DP-SGD capture: epsilon={args.dp_epsilon}, delta={args.dp_delta}, "
            f"T={args.dpsgd_rounds}, clip={args.dp_clip_norm}, sigma={dpsgd_sigma:.4f}",
            flush=True,
        )
        raw_delta = _train_model_delta_dpsgd(
            model=model,
            features=true_features,
            labels=true_labels,
            local_epochs=args.local_epochs,
            local_batch_size=local_batch_size,
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            dp_clip_norm=args.dp_clip_norm,
            dp_noise_multiplier=dpsgd_sigma,
            device=device,
        )
    else:
        raw_delta = _train_model_delta(
            model=model,
            features=true_features,
            labels=true_labels,
            local_epochs=args.local_epochs,
            local_batch_size=local_batch_size,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            device=device,
        )
    if args.dpsgd_enabled:
        observed_delta = OrderedDict(
            (name, tensor.detach().cpu())
            for name, tensor in raw_delta.items()
        )
    else:
        sqmpc_core.SQMPC_Q_BITS_FIRST = args.q_bits_first
        sqmpc_core.SQMPC_Q_BITS_MID = args.q_bits_mid
        sqmpc_core.SQMPC_Q_BITS_LAST = args.q_bits_last
        sqmpc_core.SQMPC_Q_B1_STRICT = bool(args.q_b1_strict)
        sqmpc_core.SQMPC_Q_SCOPE = "none" if args.no_quantize else "all"
        obfuscated_delta_list = sqmpc_core.mixed_quantize_mlp([raw_delta[name] for name in raw_delta])
        observed_delta = OrderedDict(
            (name, tensor.detach().cpu())
            for name, tensor in zip(raw_delta, obfuscated_delta_list, strict=True)
        )
    final_obfuscated_state = _apply_delta_to_state(pre_state, observed_delta)
    diagnostics = _validate_observed_delta(
        pre_state=pre_state,
        final_state=final_obfuscated_state,
        observed_delta=observed_delta,
        raw_delta=raw_delta,
    )

    metadata = _base_capture_metadata(
        args=args,
        prepared=prepared,
        architecture="fl_sqmpc_mlp",
        loss_kind="cross_entropy",
        local_batch_size=local_batch_size,
        attack_indices=attack_indices,
        true_labels=true_labels,
        hidden_sizes=hidden_sizes,
        optimizer=args.optimizer,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        obfuscation_surface=(
            args.obfuscation_surface
            or ("dpsgd_client" if args.dpsgd_enabled else "colluding_quantized_client")
        ),
        extra={
            "q_bits_first": args.q_bits_first,
            "q_bits_mid": args.q_bits_mid,
            "q_bits_last": args.q_bits_last,
            "q_b1_strict": bool(args.q_b1_strict),
            "no_quantize": bool(args.no_quantize),
            "dpsgd_enabled": bool(args.dpsgd_enabled),
            "dp_epsilon": (float(args.dp_epsilon) if args.dpsgd_enabled else None),
            "dp_delta": (float(args.dp_delta) if args.dpsgd_enabled else None),
            "dp_clip_norm": (float(args.dp_clip_norm) if args.dpsgd_enabled else None),
            "dp_noise_multiplier": (float(dpsgd_sigma) if args.dpsgd_enabled and dpsgd_sigma is not None else None),
            "dpsgd_rounds": (int(args.dpsgd_rounds) if args.dpsgd_enabled else None),
            **diagnostics,
        },
    )
    artifact = _build_artifact(
        metadata=metadata,
        global_model_state=pre_state,
        raw_update_tensors=observed_delta,
        true_features=true_features,
        true_labels=true_labels,
        batch_indices=attack_indices,
    )
    artifact_path = _save_artifact(args, metadata, artifact)
    return CaptureResult(artifact_path, artifact, true_features, true_labels)


def capture_cidiot_sample_gradient(
    args: argparse.Namespace,
    prepared: Any,
    partitions: list[np.ndarray],
    attack_indices: np.ndarray,
    local_batch_size: int,
    device: torch.device,
) -> CaptureResult:
    """Capture the wire signal CIDIoT actually uploads (paper Eq. 23-24).

    The edge computes `s_k = ∂L_D_k(G(ẑ_k))/∂G(ẑ_k)` on its trained D_k at the
    server-known fake samples, optionally adds Gaussian noise (Eq. 24), then
    uploads. `x_real` does not appear directly in this gradient — it influences
    the signal only through D_k's parameters (which never leave the edge).

    We require `--cidiot-checkpoint` so the captured signal reflects a properly
    trained D_k from a real CIDIoT training run, not a randomly-initialised one.
    """
    if args.cidiot_checkpoint is None:
        raise ValueError(
            "--cidiot-capture-mode generator_sample_gradient requires "
            "--cidiot-checkpoint <path-to-.checkpoint.pt produced by run_cidiot_accuracy.py>"
        )
    ckpt = torch.load(args.cidiot_checkpoint, map_location=device, weights_only=False)
    feature_dim = int(ckpt["feature_dim"])
    if feature_dim != prepared.x_train.shape[1]:
        raise ValueError(
            f"Checkpoint feature_dim ({feature_dim}) != dataset feature_dim "
            f"({prepared.x_train.shape[1]}). Use the same preprocessing as the checkpoint's training run."
        )
    num_classes = int(ckpt["num_classes"])
    ckpt_cfg = ckpt["config"]
    generator = ConditionalTCNGenerator(
        feature_dim=feature_dim,
        num_classes=num_classes,
        latent_dim=int(ckpt_cfg["latent_dim"]),
        label_dim=int(ckpt_cfg["label_dim"]),
        channels=int(ckpt_cfg["tcn_channels"]),
        bounded_output=not bool(ckpt_cfg.get("lsgan_enabled", False)),
    ).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()
    set_requires_grad(generator, False)

    if args.client < 0 or args.client >= len(ckpt["discriminator_state_dicts"]):
        raise ValueError(
            f"Checkpoint has {len(ckpt['discriminator_state_dicts'])} D_k's; --client must be in range"
        )
    discriminator = TCNDiscriminator(
        feature_dim=feature_dim,
        num_classes=num_classes,
        channels=int(ckpt_cfg["tcn_channels"]),
        hidden_dim=int(ckpt_cfg["hidden_dim"]),
    ).to(device)
    discriminator.load_state_dict(ckpt["discriminator_state_dicts"][args.client])
    discriminator.eval()
    set_requires_grad(discriminator, False)

    rng = np.random.default_rng(args.seed + 7919 * (args.round + 1))
    true_features = torch.from_numpy(prepared.x_train[attack_indices]).float().to(device)
    true_labels = torch.from_numpy(prepared.y_train[attack_indices]).long().to(device)

    # Server-known: ẑ_k = (z_k, y_k). Use the captured-batch labels as the
    # pseudo-labels for the fake samples (paper assumes server picks fake labels;
    # using the real labels here is the worst-case-for-CIDIoT setup).
    z = torch.randn(len(true_labels), int(ckpt_cfg["latent_dim"]), device=device)
    fake_y = true_labels.clone()
    fake_x = generator(z, fake_y).detach()
    fake_x.requires_grad_(True)

    # Generator loss as seen by edge: paper Eq. 13-14: L_G = adv + λ_G · CE
    # (we use the same WGAN-GP convention as the training run for adv).
    src_logits, cls_logits = discriminator(fake_x)
    if bool(ckpt_cfg.get("lsgan_enabled", False)) and bool(ckpt_cfg.get("wgan_gp", False)):
        adv = src_logits.mean()
    elif bool(ckpt_cfg.get("lsgan_enabled", False)):
        # Paper Eq. 8 / Eq. 13: G minimises E[D(G(z))]
        adv = src_logits.mean()
    else:
        # Vanilla BCE: G wants D(fake) = 1
        adv = nn.BCEWithLogitsLoss()(src_logits, torch.ones_like(src_logits))
    cls = nn.CrossEntropyLoss()(cls_logits, fake_y)
    g_loss = adv + float(ckpt_cfg.get("generator_class_loss_weight", 1.0)) * cls

    raw_sample_gradient = torch.autograd.grad(g_loss, fake_x, retain_graph=False)[0].detach()

    surface = args.obfuscation_surface or "none"
    dp_requested = bool(args.cidiot_dp_uplink) or surface in {"dp", "dp_noisy_update", "cidiot_dp", "gaussian_dp"}
    dp_diagnostics: dict[str, Any] = {"dp_enabled": False}
    if dp_requested:
        # Paper Eq. 24: per-sample clip + Gaussian noise on the sample gradient.
        clip_norm = float(args.dp_clip_norm)
        if args.dp_noise_multiplier is None:
            noise_multiplier, achieved_epsilon, best_order = _derive_noise_multiplier_from_rdp(
                epsilon=float(args.dp_epsilon),
                delta=float(args.dp_delta),
                batch_size=local_batch_size,
                max_order=int(args.dp_rdp_max_order),
            )
            noise_source = "rdp_budget"
        else:
            noise_multiplier = float(args.dp_noise_multiplier)
            achieved_epsilon, best_order = _epsilon_from_rdp(
                sigma=noise_multiplier,
                delta=float(args.dp_delta),
                batch_size=local_batch_size,
                max_order=int(args.dp_rdp_max_order),
            )
            noise_source = "explicit"
        flat = raw_sample_gradient.reshape(raw_sample_gradient.shape[0], -1)
        per_sample_norm = flat.norm(p=2, dim=1).clamp_min(1e-12)
        coef = torch.clamp(clip_norm / per_sample_norm, max=1.0)
        clipped = raw_sample_gradient * coef.view(-1, 1)
        noise_std = noise_multiplier * clip_norm
        torch.manual_seed(args.seed + 17389 * (args.client + 1) + 2267 * args.round)
        noise = torch.randn_like(clipped) * noise_std
        observed_sample_gradient = clipped + noise
        dp_diagnostics = {
            "dp_enabled": True,
            "dp_mechanism": "per_sample_clip_plus_gaussian_paper_eq24",
            "dp_clip_norm": clip_norm,
            "dp_noise_multiplier": float(noise_multiplier),
            "dp_noise_std": float(noise_std),
            "dp_noise_source": noise_source,
            "dp_epsilon": float(args.dp_epsilon),
            "dp_delta": float(args.dp_delta),
            "dp_rdp_achieved_epsilon": float(achieved_epsilon),
            "dp_rdp_best_order": int(best_order),
            "dp_raw_sample_gradient_l2_mean": float(per_sample_norm.mean().cpu()),
            "dp_clip_coefficient_mean": float(coef.mean().cpu()),
        }
        surface = "cidiot_dp_uplink"
    else:
        observed_sample_gradient = raw_sample_gradient

    metadata = _base_capture_metadata(
        args=args,
        prepared=prepared,
        architecture="cidiot_tcn_discriminator",
        loss_kind="cidiot_sample_gradient",
        local_batch_size=local_batch_size,
        attack_indices=attack_indices,
        true_labels=true_labels.cpu(),
        hidden_sizes=None,
        optimizer="adam",
        beta1=args.cidiot_beta1,
        beta2=args.cidiot_beta2,
        adam_eps=args.adam_eps,
        obfuscation_surface=surface,
        extra={
            "cidiot_capture_mode": "generator_sample_gradient",
            "cidiot_checkpoint": str(args.cidiot_checkpoint),
            "cidiot_checkpoint_final_accuracy": (
                float(ckpt.get("final_accuracy")) if ckpt.get("final_accuracy") is not None else None
            ),
            "tcn_channels": int(ckpt_cfg["tcn_channels"]),
            "hidden_dim": int(ckpt_cfg["hidden_dim"]),
            "latent_dim": int(ckpt_cfg["latent_dim"]),
            "label_dim": int(ckpt_cfg["label_dim"]),
            "lsgan_enabled": bool(ckpt_cfg.get("lsgan_enabled", False)),
            "wgan_gp": bool(ckpt_cfg.get("wgan_gp", False)),
            "generator_class_loss_weight": float(ckpt_cfg.get("generator_class_loss_weight", 1.0)),
            **dp_diagnostics,
        },
    )
    artifact = _build_artifact(
        metadata=metadata,
        global_model_state=_state_dict_cpu(discriminator),
        raw_update_tensors=OrderedDict(),  # No parameter delta in this attack
        true_features=true_features.detach().cpu(),
        true_labels=true_labels.cpu(),
        batch_indices=attack_indices,
        extra_tensors={
            "sample_gradient": observed_sample_gradient.detach().cpu(),
            "raw_sample_gradient": raw_sample_gradient.detach().cpu(),
            "fake_x": fake_x.detach().cpu(),
            "fake_y": fake_y.detach().cpu(),
            "z": z.detach().cpu(),
        },
    )
    artifact_path = _save_artifact(args, metadata, artifact)
    return CaptureResult(artifact_path, artifact, true_features.detach().cpu(), true_labels.cpu())


def capture_cidiot_artifact(
    args: argparse.Namespace,
    prepared: Any,
    partitions: list[np.ndarray],
    attack_indices: np.ndarray,
    local_batch_size: int,
    device: torch.device,
) -> CaptureResult:
    num_classes = len(prepared.class_names)
    rng = np.random.default_rng(args.seed)
    generator = ConditionalTCNGenerator(
        feature_dim=prepared.x_train.shape[1],
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        label_dim=args.label_dim,
        channels=args.tcn_channels,
    ).to(device)
    discriminator = TCNDiscriminator(
        feature_dim=prepared.x_train.shape[1],
        num_classes=num_classes,
        channels=args.tcn_channels,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if args.supervised_pretrain_steps > 0:
        optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.learning_rate,
            betas=(args.cidiot_beta1, args.cidiot_beta2),
            eps=args.adam_eps,
        )
        _pretrain_cidiot_discriminator(
            discriminator=discriminator,
            optimizer=optimizer,
            x_train=prepared.x_train,
            y_train=prepared.y_train,
            client_indices=partitions[args.client],
            steps=args.supervised_pretrain_steps,
            batch_size=local_batch_size,
            source_loss_weight=args.source_loss_weight,
            rng=rng,
            device=device,
        )

    true_features = torch.from_numpy(prepared.x_train[attack_indices]).float()
    true_labels = torch.from_numpy(prepared.y_train[attack_indices]).long()
    pre_state = _state_dict_cpu(discriminator)
    fake_features, fake_labels = _make_cidiot_fake_batches(
        generator=generator,
        total_rows=len(true_labels),
        batch_size=local_batch_size,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        device=device,
    )
    raw_delta = _train_cidiot_discriminator_delta(
        discriminator=discriminator,
        true_features=true_features,
        true_labels=true_labels,
        fake_features=fake_features,
        fake_labels=fake_labels,
        local_batch_size=local_batch_size,
        learning_rate=args.learning_rate,
        beta1=args.cidiot_beta1,
        beta2=args.cidiot_beta2,
        adam_eps=args.adam_eps,
        source_loss_weight=args.source_loss_weight,
        real_class_loss_weight=args.real_class_loss_weight,
        fake_class_loss_weight=args.fake_class_loss_weight,
        device=device,
    )
    surface = args.obfuscation_surface or "none"
    dp_requested = surface in {"dp", "dp_noisy_update", "cidiot_dp", "gaussian_dp"}
    dp_diagnostics: dict[str, Any] = {"dp_enabled": False}
    if dp_requested:
        observed_delta, dp_diagnostics = _apply_cidiot_dp_upload_noise(
            raw_delta=raw_delta,
            clip_norm=args.dp_clip_norm,
            noise_multiplier=args.dp_noise_multiplier,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            batch_size=local_batch_size,
            max_order=args.dp_rdp_max_order,
            seed=args.seed + 17389 * (args.client + 1) + 2267 * args.round,
        )
        surface = "dp_noisy_update"
    else:
        observed_delta = raw_delta
    final_state = _apply_delta_to_state(pre_state, observed_delta)
    diagnostics = _validate_observed_delta(
        pre_state=pre_state,
        final_state=final_state,
        observed_delta=observed_delta,
        raw_delta=raw_delta,
    )
    metadata = _base_capture_metadata(
        args=args,
        prepared=prepared,
        architecture="cidiot_tcn_discriminator",
        loss_kind="cidiot_discriminator",
        local_batch_size=local_batch_size,
        attack_indices=attack_indices,
        true_labels=true_labels,
        hidden_sizes=None,
        optimizer="adam",
        beta1=args.cidiot_beta1,
        beta2=args.cidiot_beta2,
        adam_eps=args.adam_eps,
        obfuscation_surface=surface,
        extra={
            "tcn_channels": args.tcn_channels,
            "hidden_dim": args.hidden_dim,
            "source_loss_weight": args.source_loss_weight,
            "real_class_loss_weight": args.real_class_loss_weight,
            "fake_class_loss_weight": args.fake_class_loss_weight,
            "supervised_pretrain_steps": args.supervised_pretrain_steps,
            **dp_diagnostics,
            **diagnostics,
        },
    )
    artifact = _build_artifact(
        metadata=metadata,
        global_model_state=pre_state,
        raw_update_tensors=observed_delta,
        true_features=true_features,
        true_labels=true_labels,
        batch_indices=attack_indices,
        extra_tensors={
            "fake_features": fake_features.detach().cpu(),
            "fake_labels": fake_labels.detach().cpu(),
        },
    )
    artifact_path = _save_artifact(args, metadata, artifact)
    return CaptureResult(artifact_path, artifact, true_features, true_labels)


def _train_model_delta(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    local_epochs: int,
    local_batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    local_model = copy.deepcopy(model).to(device)
    pre_params = OrderedDict(
        (name, param.detach().cpu().clone())
        for name, param in model.named_parameters()
    )
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=adam_eps,
        )
    loader = _attack_loader(features, labels, local_batch_size)
    criterion = nn.CrossEntropyLoss()
    local_model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(local_model(xb), yb)
            loss.backward()
            optimizer.step()
    return OrderedDict(
        (name, param.detach().cpu() - pre_params[name])
        for name, param in local_model.named_parameters()
    )


def _train_model_delta_dpsgd(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    local_epochs: int,
    local_batch_size: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    dp_clip_norm: float,
    dp_noise_multiplier: float,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """Take ``local_epochs`` Adam DP-SGD steps (Opacus per-sample clip + Gaussian noise)
    on the supplied (features, labels) batch, returning the resulting delta."""
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    local_model = copy.deepcopy(model).to(device)
    pre_params = OrderedDict(
        (name, param.detach().cpu().clone())
        for name, param in model.named_parameters()
    )
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=adam_eps,
    )
    dataset = TensorDataset(features.to(device), labels.to(device))
    loader = DataLoader(dataset, batch_size=local_batch_size, shuffle=False, drop_last=False)
    privacy_engine = PrivacyEngine(accountant="rdp")
    local_model, optimizer, loader = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=dp_noise_multiplier,
        max_grad_norm=dp_clip_norm,
    )
    criterion = nn.CrossEntropyLoss()
    local_model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(local_model(xb), yb)
            loss.backward()
            optimizer.step()
    # Opacus wraps the module; pull the underlying GradSampleModule's wrapped model.
    wrapped = local_model._module if hasattr(local_model, "_module") else local_model
    return OrderedDict(
        (name, param.detach().cpu() - pre_params[name])
        for name, param in wrapped.named_parameters()
    )


def _train_cidiot_discriminator_delta(
    discriminator: TCNDiscriminator,
    true_features: torch.Tensor,
    true_labels: torch.Tensor,
    fake_features: torch.Tensor,
    fake_labels: torch.Tensor,
    local_batch_size: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    source_loss_weight: float,
    real_class_loss_weight: float,
    fake_class_loss_weight: float,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    local_model = copy.deepcopy(discriminator).to(device)
    pre_params = OrderedDict(
        (name, param.detach().cpu().clone())
        for name, param in discriminator.named_parameters()
    )
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=adam_eps,
    )
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    local_model.train()
    for start in range(0, len(true_labels), local_batch_size):
        real_x = true_features[start : start + local_batch_size].to(device)
        real_y = true_labels[start : start + local_batch_size].to(device)
        fake_x = fake_features[start : start + local_batch_size].to(device)
        fake_y = fake_labels[start : start + local_batch_size].to(device)
        real_source, real_class = local_model(real_x)
        fake_source, fake_class = local_model(fake_x)
        source_loss = bce(real_source, torch.ones_like(real_source)) + bce(
            fake_source,
            torch.zeros_like(fake_source),
        )
        class_loss = real_class_loss_weight * ce(real_class, real_y)
        if fake_class_loss_weight > 0.0:
            class_loss = class_loss + fake_class_loss_weight * ce(fake_class, fake_y)
        loss = source_loss_weight * source_loss + class_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return OrderedDict(
        (name, param.detach().cpu() - pre_params[name])
        for name, param in local_model.named_parameters()
    )


def _pretrain_cidiot_discriminator(
    discriminator: TCNDiscriminator,
    optimizer: torch.optim.Optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: np.ndarray,
    steps: int,
    batch_size: int,
    source_loss_weight: float,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    discriminator.train()
    for _ in range(steps):
        chosen = rng.choice(client_indices, size=batch_size, replace=len(client_indices) < batch_size)
        xb = torch.from_numpy(x_train[chosen]).float().to(device)
        yb = torch.from_numpy(y_train[chosen]).long().to(device)
        real_source, real_class = discriminator(xb)
        loss = source_loss_weight * bce(real_source, torch.ones_like(real_source)) + ce(real_class, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _make_cidiot_fake_batches(
    generator: ConditionalTCNGenerator,
    total_rows: int,
    batch_size: int,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator.eval()
    fake_features = []
    fake_labels = []
    with torch.no_grad():
        for start in range(0, total_rows, batch_size):
            rows = min(batch_size, total_rows - start)
            labels = torch.randint(0, num_classes, (rows,), device=device)
            noise = torch.randn(rows, latent_dim, device=device)
            fake_features.append(generator(noise, labels).detach().cpu())
            fake_labels.append(labels.detach().cpu())
    return torch.cat(fake_features, dim=0), torch.cat(fake_labels, dim=0)


def run_cidiot_sample_gradient_attack(
    capture: CaptureResult,
    attack_steps: int,
    attack_lr: float,
    ensemble_restarts: int,
    device: torch.device,
):
    """Naive gradient-inversion attack on CIDIoT's wire signal (sample gradient).

    The captured signal is `s_obs = ∂L_D(G(z))/∂G(z)` — a per-sample gradient
    evaluated at server-known fake samples G(z), not at x_real. We naively
    treat s_obs as if it were the per-input gradient `∂L_D(x_real, y)/∂x_real`
    and run input-gradient inversion: find x_candidate s.t.
    `∂L_D(x_candidate, y)/∂x_candidate ≈ s_obs` (cosine matching).

    Expected result: reconstruction at random-noise level, because x_real never
    appears in the loss landscape of s_obs. This is the structural defense of
    CIDIoT's threat model (paper §VI-A): without D_k weights leaving the edge,
    no gradient-inversion attack on the wire signal has x_real as a solution.
    """
    from attacks.common import ReconstructionResult

    artifact = capture.artifact
    metadata = artifact["metadata"]
    if metadata.get("loss_kind") != "cidiot_sample_gradient":
        raise ValueError("attack expects loss_kind=cidiot_sample_gradient")

    s_obs = artifact["extra_tensors"]["sample_gradient"].to(device)
    ckpt_feature_dim = int(s_obs.shape[1])
    # Derive num_classes from the D_k's class_head shape in the global_model_state,
    # which is the authoritative source (the captured-batch labels may be partial).
    num_classes = int(artifact["global_model_state"]["class_head.weight"].shape[0])
    discriminator = TCNDiscriminator(
        feature_dim=ckpt_feature_dim,
        num_classes=num_classes,
        channels=int(metadata["tcn_channels"]),
        hidden_dim=int(metadata["hidden_dim"]),
    ).to(device)
    discriminator.load_state_dict(artifact["global_model_state"])
    discriminator.eval()
    set_requires_grad(discriminator, False)

    true_labels = capture.true_labels.to(device).long()
    batch_size, feature_dim = s_obs.shape

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    source_loss_weight = float(metadata.get("source_loss_weight", 1.0))
    real_class_loss_weight = float(metadata.get("real_class_loss_weight", 1.0))

    def candidate_input_gradient(x_cand: torch.Tensor, y_cand: torch.Tensor) -> torch.Tensor:
        x = x_cand.detach().clone().requires_grad_(True)
        src_logits, cls_logits = discriminator(x)
        # Mirror the D-side loss that the wire signal corresponds to.
        # Source: vanilla BCE-on-real (since attacker uses D-classification
        # interpretation); class: CE.
        source_loss = bce(src_logits, torch.ones_like(src_logits))
        class_loss = ce(cls_logits, y_cand)
        loss = source_loss_weight * source_loss + real_class_loss_weight * class_loss
        grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        return grad

    def match_loss(g_cand: torch.Tensor, g_obs: torch.Tensor) -> torch.Tensor:
        # Cosine-distance match: 1 - cos(per-sample-flattened).
        a = g_cand.reshape(g_cand.shape[0], -1)
        b = g_obs.reshape(g_obs.shape[0], -1)
        return (1.0 - torch.nn.functional.cosine_similarity(a, b, dim=1)).mean()

    best_x: torch.Tensor | None = None
    best_objective = float("inf")
    best_history: list[float] = []
    for restart in range(ensemble_restarts):
        torch.manual_seed(int(metadata.get("seed", 0)) + 31 * restart + 7)
        x_candidate = torch.randn(batch_size, feature_dim, device=device) * 0.5
        x_candidate.requires_grad_(True)
        optimizer = torch.optim.Adam([x_candidate], lr=attack_lr)
        history: list[float] = []
        for step in range(attack_steps):
            optimizer.zero_grad()
            g_cand = candidate_input_gradient(x_candidate, true_labels)
            loss = match_loss(g_cand, s_obs)
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))
        final_obj = history[-1]
        if final_obj < best_objective:
            best_objective = final_obj
            best_x = x_candidate.detach().clone()
            best_history = history

    return ReconstructionResult(
        attack_name="cidiot_sample_gradient_naive_input_gradient_inversion",
        reconstructed_features=best_x.detach().cpu(),
        reconstructed_labels=true_labels.detach().cpu(),
        final_objective=float(best_objective),
        objective_history=best_history,
        metadata={
            "attack_kind": "cidiot_sample_gradient_naive",
            "attack_steps": int(attack_steps),
            "attack_lr": float(attack_lr),
            "ensemble_restarts": int(ensemble_restarts),
            "note": (
                "Naive input-gradient inversion on CIDIoT's wire signal. The "
                "observed signal is ∂L_D(G(z))/∂G(z), evaluated at G(z) not x_real, "
                "so x_real is structurally absent from the optimization landscape. "
                "Recovery is expected at random-noise level."
            ),
        },
        diagnostics={
            "best_restart_objective": float(best_objective),
            "all_restart_final_objectives": [],
        },
    )


def _attack_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(features.float(), labels.long()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def _state_dict_cpu(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )


def _apply_delta_to_state(
    pre_state: OrderedDict[str, torch.Tensor],
    delta: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    final_state = OrderedDict((name, tensor.clone()) for name, tensor in pre_state.items())
    for name, update in delta.items():
        final_state[name] = final_state[name] + update
    return final_state


def _validate_observed_delta(
    pre_state: OrderedDict[str, torch.Tensor],
    final_state: OrderedDict[str, torch.Tensor],
    observed_delta: OrderedDict[str, torch.Tensor],
    raw_delta: OrderedDict[str, torch.Tensor],
) -> dict[str, float]:
    invariant_max = 0.0
    raw_to_obfuscated_max = 0.0
    for name, observed in observed_delta.items():
        expected = final_state[name] - pre_state[name]
        invariant_max = max(invariant_max, float((observed - expected).abs().max().item()))
        raw_to_obfuscated_max = max(
            raw_to_obfuscated_max,
            float((observed - raw_delta[name]).abs().max().item()),
        )
    if invariant_max > 1e-5:
        raise AssertionError(f"Post-obfuscation delta invariant failed: {invariant_max:.6g}")
    return {
        "post_obfuscation_invariant_max_abs_diff": invariant_max,
        "raw_to_obfuscated_max_abs_diff": raw_to_obfuscated_max,
    }


def _apply_cidiot_dp_upload_noise(
    raw_delta: OrderedDict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float | None,
    epsilon: float,
    delta: float,
    batch_size: int,
    max_order: int,
    seed: int,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Approximate the CIDIoT paper's upload DP surface on the captured update.

    The paper clips sample gradients and adds Gaussian noise before upload. This
    FedAvg-TabLeak artifact observes a final client update, so the closest local
    attack-surface equivalent is clipping/noising the final observed delta.
    """
    if clip_norm <= 0.0:
        raise ValueError("--dp-clip-norm must be positive")
    if noise_multiplier is None:
        noise_multiplier, achieved_epsilon, best_order = _derive_noise_multiplier_from_rdp(
            epsilon=epsilon,
            delta=delta,
            batch_size=batch_size,
            max_order=max_order,
        )
        noise_source = "rdp_budget"
    else:
        if noise_multiplier < 0.0:
            raise ValueError("--dp-noise-multiplier must be non-negative")
        achieved_epsilon, best_order = _epsilon_from_rdp(
            sigma=noise_multiplier,
            delta=delta,
            batch_size=batch_size,
            max_order=max_order,
        )
        noise_source = "explicit"

    raw_norm = _delta_l2_norm(raw_delta)
    clip_coefficient = min(1.0, clip_norm / (raw_norm + 1e-12))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    observed_delta: OrderedDict[str, torch.Tensor] = OrderedDict()
    noise_sq_sum = 0.0
    noise_max_abs = 0.0
    noise_std = float(noise_multiplier * clip_norm)
    for name, tensor in raw_delta.items():
        clipped = tensor.detach().cpu() * clip_coefficient
        noise = torch.randn(
            clipped.shape,
            generator=generator,
            dtype=clipped.dtype,
            device=clipped.device,
        ) * noise_std
        observed_delta[name] = clipped + noise
        noise_sq_sum += float((noise.float() ** 2).sum().item())
        noise_max_abs = max(noise_max_abs, float(noise.abs().max().item()))

    return observed_delta, {
        "dp_enabled": True,
        "dp_mechanism": "clip_client_delta_add_gaussian_noise_approx_cidiot_upload_dp",
        "dp_reference": "CIDIoT Gaussian mechanism M_sigma,C(x)=clip(x,C)+N(0,sigma^2 C^2 I)",
        "dp_surface_note": (
            "The CIDIoT paper applies this mechanism to uploaded sample gradients; "
            "this FedAvg-TabLeak experiment applies the same clip/noise form to "
            "the final observed client delta because that is the attack artifact."
        ),
        "dp_epsilon": float(epsilon),
        "dp_delta": float(delta),
        "dp_clip_norm": float(clip_norm),
        "dp_noise_multiplier": float(noise_multiplier),
        "dp_noise_std": noise_std,
        "dp_noise_source": noise_source,
        "dp_rdp_batch_size": int(batch_size),
        "dp_rdp_max_order": int(max_order),
        "dp_rdp_achieved_epsilon": float(achieved_epsilon),
        "dp_rdp_best_order": int(best_order),
        "dp_raw_delta_l2_norm": float(raw_norm),
        "dp_clip_coefficient": float(clip_coefficient),
        "dp_clipped_delta_l2_norm": float(raw_norm * clip_coefficient),
        "dp_noise_l2_norm": float(math.sqrt(noise_sq_sum)),
        "dp_noise_max_abs": float(noise_max_abs),
    }


def _delta_l2_norm(delta: OrderedDict[str, torch.Tensor]) -> float:
    squared_sum = 0.0
    for tensor in delta.values():
        squared_sum += float((tensor.detach().cpu().float() ** 2).sum().item())
    return math.sqrt(squared_sum)


def _derive_noise_multiplier_from_rdp(
    epsilon: float,
    delta: float,
    batch_size: int,
    max_order: int,
) -> tuple[float, float, int]:
    if epsilon <= 0.0:
        raise ValueError("--dp-epsilon must be positive")
    _validate_rdp_inputs(delta=delta, batch_size=batch_size, max_order=max_order)
    high = 1.0
    high_epsilon, _ = _epsilon_from_rdp(
        sigma=high,
        delta=delta,
        batch_size=batch_size,
        max_order=max_order,
    )
    while high_epsilon > epsilon:
        high *= 2.0
        high_epsilon, _ = _epsilon_from_rdp(
            sigma=high,
            delta=delta,
            batch_size=batch_size,
            max_order=max_order,
        )
        if high > 1e6:
            raise ValueError("Could not derive a finite DP noise multiplier")

    low = 1e-12
    for _ in range(80):
        mid = (low + high) / 2.0
        mid_epsilon, _ = _epsilon_from_rdp(
            sigma=mid,
            delta=delta,
            batch_size=batch_size,
            max_order=max_order,
        )
        if mid_epsilon > epsilon:
            low = mid
        else:
            high = mid
    achieved, best_order = _epsilon_from_rdp(
        sigma=high,
        delta=delta,
        batch_size=batch_size,
        max_order=max_order,
    )
    return high, achieved, best_order


def _epsilon_from_rdp(
    sigma: float,
    delta: float,
    batch_size: int,
    max_order: int,
) -> tuple[float, int]:
    if sigma <= 0.0:
        return math.inf, 2
    _validate_rdp_inputs(delta=delta, batch_size=batch_size, max_order=max_order)
    log_delta = math.log(1.0 / delta)
    best_epsilon = math.inf
    best_order = 2
    for order in range(2, max_order + 1):
        rdp = (2.0 * batch_size * order) / (sigma**2)
        epsilon = rdp + log_delta / (order - 1)
        if epsilon < best_epsilon:
            best_epsilon = epsilon
            best_order = order
    return best_epsilon, best_order


def _validate_rdp_inputs(delta: float, batch_size: int, max_order: int) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError("--dp-delta must be in (0, 1)")
    if batch_size <= 0:
        raise ValueError("RDP batch size must be positive")
    if max_order < 2:
        raise ValueError("--dp-rdp-max-order must be at least 2")


def _base_capture_metadata(
    args: argparse.Namespace,
    prepared: Any,
    architecture: str,
    loss_kind: str,
    local_batch_size: int,
    attack_indices: np.ndarray,
    true_labels: torch.Tensor,
    hidden_sizes: list[int] | None,
    optimizer: str,
    beta1: float,
    beta2: float,
    adam_eps: float,
    obfuscation_surface: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    steps_per_epoch = args.attack_batch_size // local_batch_size
    total_steps = args.local_epochs * steps_per_epoch
    label_sequence = []
    for _ in range(args.local_epochs):
        label_sequence.extend(true_labels.tolist())
    metadata: dict[str, Any] = {
        "capture_kind": "fedavg_multistep",
        "method": args.method,
        "obfuscation_surface": obfuscation_surface,
        "dataset": prepared.dataset_id,
        "dataset_path": str(args.dataset),
        "dataset_metadata": _dataset_metadata(args.dataset),
        "architecture": architecture,
        "loss_kind": loss_kind,
        "task": args.task,
        "partition": args.partition,
        "seed": args.seed,
        "round": args.round,
        "client": args.client,
        "clients": args.clients,
        "batch_size": args.attack_batch_size,
        "num_examples": args.attack_batch_size,
        "local_epochs": total_steps,
        "local_training_epochs": args.local_epochs,
        "local_batch_size": local_batch_size,
        "steps_mode": True,
        "step_cycling_labels": label_sequence,
        "learning_rate": args.learning_rate,
        "optimizer": optimizer,
        "beta1": beta1,
        "beta2": beta2,
        "adam_eps": adam_eps,
        "freeze_batchnorm": True,
        "batch_indices": [int(index) for index in attack_indices.tolist()],
        "max_rows": args.max_rows,
        "num_shards": args.num_shards,
        "test_size": prepared.test_size,
    }
    if hidden_sizes is not None:
        metadata["hidden_sizes"] = hidden_sizes
    metadata.update(extra)
    return metadata


def _build_artifact(
    metadata: dict[str, Any],
    global_model_state: OrderedDict[str, torch.Tensor],
    raw_update_tensors: OrderedDict[str, torch.Tensor],
    true_features: torch.Tensor,
    true_labels: torch.Tensor,
    batch_indices: np.ndarray,
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "global_model_state": global_model_state,
        "raw_update_tensors": raw_update_tensors,
        "true_batch_features": true_features.detach().cpu(),
        "true_batch_labels": [int(label) for label in true_labels.tolist()],
        "batch_indices": [int(index) for index in batch_indices.tolist()],
        "extra_tensors": extra_tensors or {},
    }


def _save_artifact(args: argparse.Namespace, metadata: dict[str, Any], artifact: dict[str, Any]) -> Path:
    suffix = f"maxrows{args.max_rows}" if args.max_rows else "full"
    bits_tag = ""
    if args.method == "sqmpc":
        if getattr(args, "dpsgd_enabled", False):
            eps_text = f"{float(args.dp_epsilon):g}".replace(".", "p").replace("-", "m")
            bits_tag = f"_dpsgd_eps{eps_text}"
        elif args.no_quantize:
            bits_tag = "_qFloat"
        else:
            bf, bm, bl = args.q_bits_first, args.q_bits_mid, args.q_bits_last
            bits_tag = f"_qB{bf}" if bf == bm == bl else f"_qBf{bf}m{bm}l{bl}"
            if bf == 1 and args.q_b1_strict:
                bits_tag = f"{bits_tag}s"
    stem = (
        f"{args.method}_{metadata['obfuscation_surface']}_{metadata['dataset']}_{args.task}_"
        f"{args.partition}_client{args.client}_round{args.round}_b{args.attack_batch_size}"
        f"{bits_tag}_{suffix}_seed{args.seed}"
    )
    path = args.artifact_dir / f"{stem}.pt"
    torch.save(artifact, path)
    return path


def _dataset_metadata(dataset_path: Path) -> dict[str, Any]:
    meta_path = dataset_path / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "num_continuous_features": 0,
        "categorical_features": [],
        "one_hot_ranges": {},
    }


def _select_attack_indices(
    client_indices: np.ndarray,
    attack_batch_size: int,
    seed: int,
    client: int,
    round_id: int,
) -> np.ndarray:
    if attack_batch_size > len(client_indices):
        raise ValueError(
            f"attack batch size {attack_batch_size} exceeds client partition size {len(client_indices)}"
        )
    rng = np.random.default_rng(seed + 1009 * (client + 1) + 9176 * round_id)
    selected = rng.choice(client_indices, size=attack_batch_size, replace=False)
    return np.array(sorted(selected), dtype=np.int64)


def _validate_batch_shape(attack_batch_size: int, local_batch_size: int) -> None:
    if local_batch_size <= 0:
        raise ValueError("--local-batch-size must be positive")
    if attack_batch_size <= 0:
        raise ValueError("--attack-batch-size must be positive")
    if local_batch_size > attack_batch_size:
        raise ValueError("--local-batch-size must be <= --attack-batch-size")
    if attack_batch_size % local_batch_size != 0:
        raise ValueError("--attack-batch-size must be divisible by --local-batch-size")


def compute_reconstruction_metrics(
    reconstructed_features: torch.Tensor,
    reconstructed_labels: torch.Tensor,
    true_features: torch.Tensor,
    true_labels: torch.Tensor,
    metadata: dict[str, Any],
    dataset_path: Path | None = None,
) -> dict[str, float]:
    reconstructed = reconstructed_features.detach().cpu().float()
    true = true_features.detach().cpu().float()
    if reconstructed.shape != true.shape:
        raise ValueError(f"shape mismatch: reconstructed={tuple(reconstructed.shape)} true={tuple(true.shape)}")
    nonfinite_count = int((~torch.isfinite(reconstructed)).sum().item())
    if nonfinite_count:
        reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=1.0, neginf=0.0)
    cost = torch.cdist(reconstructed, true, p=2).numpy()
    rec_rows, true_rows = linear_sum_assignment(cost)
    reconstructed = reconstructed[rec_rows]
    true = true[true_rows]
    aligned_labels = reconstructed_labels.detach().cpu().long()[rec_rows]
    true_aligned_labels = true_labels.detach().cpu().long()[true_rows]

    num_continuous = int(metadata.get("num_continuous_features", 0))
    metrics: dict[str, float] = {
        "reconstructed_nonfinite_count": float(nonfinite_count),
        "label_accuracy": float((aligned_labels == true_aligned_labels).float().mean().item()),
        "row_l2_mean": float(torch.sqrt(((reconstructed - true) ** 2).sum(dim=1)).mean().item()),
    }
    if num_continuous > 0:
        continuous_diff = reconstructed[:, :num_continuous] - true[:, :num_continuous]
        metrics["continuous_mae"] = float(continuous_diff.abs().mean().item())
        metrics["continuous_rmse"] = float(torch.sqrt((continuous_diff**2).mean()).item())
        metrics["continuous_tolerance_accuracy"] = float((continuous_diff.abs() <= 0.05).float().mean().item())
        sigma_tolerances = _continuous_sigma_tolerances(
            dataset_path=dataset_path,
            num_continuous=num_continuous,
            device=continuous_diff.device,
        )
        reconstruction_continuous_mask = continuous_diff.abs() <= sigma_tolerances.view(1, -1)
        metrics["reconstruction_continuous_accuracy"] = float(
            reconstruction_continuous_mask.float().mean().item()
        )
        metrics["reconstruction_tolerance_factor"] = 0.319
        metrics["reconstruction_mean_continuous_tolerance"] = float(
            sigma_tolerances.mean().item()
        )
    else:
        metrics["continuous_mae"] = 0.0
        metrics["continuous_rmse"] = 0.0
        metrics["continuous_tolerance_accuracy"] = 0.0
        reconstruction_continuous_mask = torch.empty((true.shape[0], 0), dtype=torch.bool)
        metrics["reconstruction_continuous_accuracy"] = 0.0
        metrics["reconstruction_tolerance_factor"] = 0.319
        metrics["reconstruction_mean_continuous_tolerance"] = 0.0

    exact_matches = []
    one_hot_ranges = metadata.get("one_hot_ranges", {})
    for feature_name in metadata.get("categorical_features", []):
        start, stop = one_hot_ranges[feature_name]
        rec_choice = torch.argmax(reconstructed[:, start:stop], dim=1)
        true_choice = torch.argmax(true[:, start:stop], dim=1)
        exact_matches.append((rec_choice == true_choice).float())
    if exact_matches:
        categorical = torch.stack(exact_matches, dim=1)
        metrics["categorical_exact_match"] = float(categorical.mean().item())
        metrics["tableak_tolerance_accuracy"] = float(
            torch.cat(
                [
                    (reconstructed[:, :num_continuous] - true[:, :num_continuous]).abs().reshape(-1) <= 0.05,
                    categorical.bool().reshape(-1),
                ]
            )
            .float()
            .mean()
            .item()
        )
        metrics["fixed_0_05_tolerance_accuracy"] = metrics["tableak_tolerance_accuracy"]
        metrics["reconstruction_accuracy"] = float(
            torch.cat(
                [
                    reconstruction_continuous_mask.reshape(-1),
                    categorical.bool().reshape(-1),
                ]
            )
            .float()
            .mean()
            .item()
        )
    else:
        metrics["categorical_exact_match"] = 0.0
        metrics["tableak_tolerance_accuracy"] = metrics["continuous_tolerance_accuracy"]
        metrics["fixed_0_05_tolerance_accuracy"] = metrics["continuous_tolerance_accuracy"]
        metrics["reconstruction_accuracy"] = metrics["reconstruction_continuous_accuracy"]
    return metrics


def _continuous_sigma_tolerances(
    dataset_path: Path | None,
    num_continuous: int,
    device: torch.device,
) -> torch.Tensor:
    if dataset_path is None:
        return torch.full((num_continuous,), 0.05, dtype=torch.float32, device=device)
    x_train_path = dataset_path / "X_train.pt"
    if not x_train_path.exists():
        return torch.full((num_continuous,), 0.05, dtype=torch.float32, device=device)
    x_train = torch.load(x_train_path, map_location="cpu").float()
    tolerances = 0.319 * x_train[:, :num_continuous].std(dim=0, unbiased=False)
    return tolerances.to(device=device, dtype=torch.float32)


def _jsonable_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            out[key] = _jsonable_diagnostics(value)
        elif isinstance(value, list):
            out[key] = [
                item.detach().cpu().tolist() if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
