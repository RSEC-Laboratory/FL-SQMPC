#!/usr/bin/env python3
"""Run a CIDIoT-style collaborative generative IDS baseline.

This implements the accuracy-relevant parts of the CIDIoT paper: a
cloud-side conditional generator, edge-side TCN discriminators/classifiers,
local discriminator training on real and generated samples, and generator
updates against the edge ensemble. Privacy transport mechanisms such as DP
noise and threshold secret sharing are intentionally excluded by default
because they do not define the detector architecture and can be evaluated
separately if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from experiment_audit import build_training_audit
from run_fl_sqmpc_accuracy import DEFAULT_DATASET, DEFAULT_OUTPUT_DIR
from run_fl_sqmpc_accuracy import partition_dirichlet, partition_iid, partition_non_iid, prepare_dataset
from run_fl_sqmpc_accuracy import select_device, set_seed


@dataclass
class CIDIoTConfig:
    method: str
    dataset_id: str
    dataset: str
    task: str
    partition: str
    dirichlet_alpha: float | None
    clients: int
    rounds: int
    batch_size: int
    discriminator_steps: int
    discriminator_epochs: int
    generator_steps: int
    latent_dim: int
    label_dim: int
    tcn_channels: int
    hidden_dim: int
    learning_rate: float
    beta1: float
    beta2: float
    supervised_pretrain_steps: int
    generator_pretrain_steps: int
    source_loss_weight: float
    real_class_loss_weight: float
    fake_class_loss_weight: float
    generator_class_loss_weight: float
    lsgan_enabled: bool
    wgan_gp: bool
    margin_lambda: float
    gp_lambda: float
    dp_enabled: bool
    dp_epsilon: float
    dp_delta: float
    dp_clip_norm: float
    dp_noise_multiplier: float | None
    dp_effective_noise_multiplier: float | None
    dp_rdp_max_order: int
    dp_rdp_achieved_epsilon: float | None
    dp_rdp_best_order: int | None
    test_size: float
    seed: int
    max_rows: int | None
    num_shards: int
    device: str
    comparison_run_id: str | None


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(1, out_channels),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.shortcut(x))


class ConditionalTCNGenerator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        latent_dim: int,
        label_dim: int,
        channels: int,
        bounded_output: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.bounded_output = bounded_output
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_dim, channels * feature_dim),
            nn.LeakyReLU(0.2),
        )
        self.tcn = nn.Sequential(
            TCNBlock(channels, channels, kernel_size=3, dilation=1),
            TCNBlock(channels, channels, kernel_size=3, dilation=2),
        )
        self.out = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_embedding(labels)
        x = torch.cat([noise, label_vec], dim=1)
        x = self.fc(x).view(noise.size(0), -1, self.feature_dim)
        x = self.tcn(x)
        out = self.out(x).squeeze(1)
        return torch.sigmoid(out) if self.bounded_output else out


class TCNDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, channels: int, hidden_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            TCNBlock(1, channels, kernel_size=3, dilation=1),
            TCNBlock(channels, channels, kernel_size=3, dilation=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.source_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x.unsqueeze(1))
        h = self.proj(h)
        return self.source_head(h).squeeze(1), self.class_head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--partition", choices=["iid", "non_iid", "dirichlet"], default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--discriminator-steps", type=int, default=10)
    parser.add_argument("--discriminator-epochs", type=int, default=0,
                        help="If > 0, each round trains D for this many epochs over the full local "
                             "partition (paper N_d semantics); --discriminator-steps is ignored. "
                             "Default 0 keeps the existing mini-batch-step behaviour.")
    parser.add_argument("--generator-steps", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--label-dim", type=int, default=16)
    parser.add_argument("--tcn-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--supervised-pretrain-steps", type=int, default=200)
    parser.add_argument("--generator-pretrain-steps", type=int, default=0,
                        help="Conditional-MSE warmup of G against real samples before adversarial training. 0 disables.")
    parser.add_argument("--source-loss-weight", type=float, default=0.1)
    parser.add_argument("--real-class-loss-weight", type=float, default=1.0)
    parser.add_argument("--fake-class-loss-weight", type=float, default=0.0)
    parser.add_argument("--generator-class-loss-weight", type=float, default=1.0)
    parser.add_argument("--lsgan-enabled", action="store_true",
                        help="Use LS-GAN margin loss + gradient penalty (paper Eqs. 7,9,13). "
                             "When off, falls back to vanilla BCE-GAN.")
    parser.add_argument("--wgan-gp", action="store_true",
                        help="When set together with --lsgan-enabled, replace the LS-GAN "
                             "Eq.10 source loss with the cleaner WGAN-GP form "
                             "E[D(x)] - E[D(G(z))] + γ·GP. Useful when the LS-GAN "
                             "form diverges (epoch-mode D training).")
    parser.add_argument("--margin-lambda", type=float, default=1.0,
                        help="LS-GAN margin weight λ in Eq. 7.")
    parser.add_argument("--gp-lambda", type=float, default=10.0,
                        help="Gradient-penalty weight γ in Eq. 9.")
    parser.add_argument("--dp-enabled", action="store_true")
    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-rdp-max-order", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="After training, save generator + per-client discriminator state_dicts "
                             "to <output-dir>/<result_basename>.checkpoint.pt for downstream attack reuse.")
    parser.add_argument("--comparison-run-id", default=None)
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Apply log1p to features before MinMax scaling (helps heavy-tailed columns).",
    )
    return parser.parse_args()


def sample_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    client_idx: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    replace = len(client_idx) < batch_size
    chosen = rng.choice(client_idx, size=batch_size, replace=replace)
    xb = torch.from_numpy(x_train[chosen]).float().to(device)
    yb = torch.from_numpy(y_train[chosen]).long().to(device)
    return xb, yb


def sample_noise_labels(
    batch_size: int,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(batch_size, latent_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return noise, labels


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


def _iterate_epochs(
    client_idx: np.ndarray,
    batch_size: int,
    epochs: int,
    rng: np.random.Generator,
):
    """Yield shuffled mini-batch index arrays, `epochs` full passes over `client_idx`."""
    n = len(client_idx)
    for _ in range(epochs):
        order = client_idx[rng.permutation(n)]
        for start in range(0, n, batch_size):
            chunk = order[start : start + batch_size]
            if len(chunk) < 2:
                continue
            yield chunk


def train_discriminators(
    discriminators: list[TCNDiscriminator],
    generator: ConditionalTCNGenerator,
    optimizers: list[torch.optim.Optimizer],
    x_train: np.ndarray,
    y_train: np.ndarray,
    partitions: list[np.ndarray],
    cfg: CIDIoTConfig,
    num_classes: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    source_losses = []
    class_losses = []
    gp_values = []
    hinge_values = []

    generator.eval()
    use_epoch_mode = cfg.discriminator_epochs > 0
    for discriminator, optimizer, client_idx in zip(discriminators, optimizers, partitions, strict=True):
        discriminator.train()
        if use_epoch_mode:
            # Paper N_d semantics: D trains for `discriminator_epochs` epochs over the full
            # local partition (each epoch = shuffled pass in mini-batches of cfg.batch_size).
            batch_iter = _iterate_epochs(client_idx, cfg.batch_size, cfg.discriminator_epochs, rng)
        else:
            batch_iter = (
                rng.choice(client_idx, size=cfg.batch_size, replace=len(client_idx) < cfg.batch_size)
                for _ in range(cfg.discriminator_steps)
            )
        for chosen in batch_iter:
            real_x = torch.from_numpy(x_train[chosen]).float().to(device)
            real_y = torch.from_numpy(y_train[chosen]).long().to(device)
            noise, fake_y = sample_noise_labels(real_x.size(0), cfg.latent_dim, num_classes, device)
            with torch.no_grad():
                fake_x = generator(noise, fake_y)

            real_source, real_class = discriminator(real_x)
            fake_source, fake_class = discriminator(fake_x)

            class_loss = cfg.real_class_loss_weight * ce(real_class, real_y)
            if cfg.fake_class_loss_weight > 0.0:
                class_loss = class_loss + cfg.fake_class_loss_weight * ce(fake_class, fake_y)

            if cfg.lsgan_enabled:
                # Margin is per-sample L1 distance normalised by feature_dim.
                margin = (real_x - fake_x).abs().mean(dim=1)
                hinge = torch.relu(margin + real_source - fake_source).mean()
                gp = _gradient_penalty(discriminator, real_x, fake_x, device)
                if cfg.wgan_gp:
                    # WGAN-GP form: L_D = E[D(x)] - E[D(G(z))] + γ·GP.
                    # Cleaner & well-studied. Used when paper Eq. 10 form diverges.
                    adv_part = real_source.mean() - fake_source.mean() + cfg.margin_lambda * hinge
                else:
                    # Paper Eq. 10:
                    #   L_adv(D) = E[D(x)] + λ·hinge - (D(G(z)))_+ + γ·GP
                    fake_positive = torch.relu(fake_source).mean()
                    adv_part = real_source.mean() + cfg.margin_lambda * hinge - fake_positive
                source_loss = cfg.source_loss_weight * adv_part + cfg.gp_lambda * gp
                hinge_values.append(float(hinge.detach().cpu()))
                gp_values.append(float(gp.detach().cpu()))
                loss = source_loss + class_loss
            else:
                real_source_loss = bce(real_source, torch.ones_like(real_source))
                fake_source_loss = bce(fake_source, torch.zeros_like(fake_source))
                source_loss = real_source_loss + fake_source_loss
                loss = cfg.source_loss_weight * source_loss + class_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            source_losses.append(float(source_loss.detach().cpu()))
            class_losses.append(float(class_loss.detach().cpu()))

    metrics = {
        "discriminator_source_loss": float(np.mean(source_losses)),
        "discriminator_class_loss": float(np.mean(class_losses)),
    }
    if cfg.lsgan_enabled:
        metrics["discriminator_hinge"] = float(np.mean(hinge_values)) if hinge_values else 0.0
        metrics["discriminator_gp"] = float(np.mean(gp_values)) if gp_values else 0.0
    return metrics


def _gradient_penalty(
    discriminator: TCNDiscriminator,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute ε·E_{x̂}[(||∇_{x̂} D(x̂)||₂ − 1)²] from paper Eq. 9 / WGAN-GP."""
    batch_size = real_x.size(0)
    zeta = torch.rand(batch_size, 1, device=device)
    x_hat = zeta * real_x + (1.0 - zeta) * fake_x
    x_hat.requires_grad_(True)
    src, _ = discriminator(x_hat)
    grad = torch.autograd.grad(
        outputs=src.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norms = grad.norm(2, dim=1)
    return ((grad_norms - 1.0) ** 2).mean()


def pretrain_generator(
    generator: ConditionalTCNGenerator,
    optimizer: torch.optim.Optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: CIDIoTConfig,
    num_classes: int,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    """Warm up G via conditional MSE against real samples of the same class.

    Not in the paper, but a practical fix to ensure G produces non-trivial samples
    by the time adversarial co-training starts — otherwise D trivially separates
    noise from real and the GAN game collapses (the failure mode observed in
    the 2026-05-17 loss-weight ablation, see PROGRESS.md)."""
    if cfg.generator_pretrain_steps <= 0:
        return
    full_idx = np.arange(len(y_train), dtype=np.int64)
    generator.train()
    for _ in range(cfg.generator_pretrain_steps):
        real_x, real_y = sample_batch(x_train, y_train, full_idx, cfg.batch_size, rng, device)
        noise = torch.randn(real_x.size(0), cfg.latent_dim, device=device)
        fake_x = generator(noise, real_y)
        loss = ((fake_x - real_x) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def pretrain_discriminators(
    discriminators: list[TCNDiscriminator],
    optimizers: list[torch.optim.Optimizer],
    x_train: np.ndarray,
    y_train: np.ndarray,
    partitions: list[np.ndarray],
    cfg: CIDIoTConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    if cfg.supervised_pretrain_steps <= 0:
        return

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    for discriminator, optimizer, client_idx in zip(discriminators, optimizers, partitions, strict=True):
        discriminator.train()
        for _ in range(cfg.supervised_pretrain_steps):
            real_x, real_y = sample_batch(x_train, y_train, client_idx, cfg.batch_size, rng, device)
            real_source, real_class = discriminator(real_x)
            source_loss = bce(real_source, torch.ones_like(real_source))
            class_loss = ce(real_class, real_y)
            loss = cfg.source_loss_weight * source_loss + cfg.real_class_loss_weight * class_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def train_generator(
    discriminators: list[TCNDiscriminator],
    generator: ConditionalTCNGenerator,
    optimizer: torch.optim.Optimizer,
    cfg: CIDIoTConfig,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    losses = []
    grad_norms = []
    clip_coefficients = []
    noise_norms = []

    generator.train()
    for discriminator in discriminators:
        discriminator.eval()
        set_requires_grad(discriminator, False)

    def _per_discriminator_loss(disc: TCNDiscriminator, fake_x: torch.Tensor, fake_y: torch.Tensor) -> torch.Tensor:
        fake_source, fake_class = disc(fake_x)
        if cfg.lsgan_enabled:
            # Paper Eq. 8 / Eq. 13: G minimises E[D(G(z))] (push critic up by minimising its score on fakes).
            adv = fake_source.mean()
        else:
            adv = bce(fake_source, torch.ones_like(fake_source))
        cls = ce(fake_class, fake_y)
        return adv + cfg.generator_class_loss_weight * cls

    for _ in range(cfg.generator_steps):
        noise, fake_y = sample_noise_labels(cfg.batch_size, cfg.latent_dim, num_classes, device)
        fake_x = generator(noise, fake_y)
        optimizer.zero_grad(set_to_none=True)
        if cfg.dp_enabled:
            sanitized_gradient = torch.zeros_like(fake_x)
            raw_loss = torch.zeros((), device=device)
            for discriminator in discriminators:
                loss = _per_discriminator_loss(discriminator, fake_x, fake_y)
                raw_loss = raw_loss + loss.detach()
                sample_gradient = torch.autograd.grad(
                    loss,
                    fake_x,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                sanitized, diagnostics = sanitize_sample_gradient(sample_gradient, cfg)
                sanitized_gradient = sanitized_gradient + sanitized
                grad_norms.append(diagnostics["raw_sample_gradient_l2_mean"])
                clip_coefficients.append(diagnostics["clip_coefficient_mean"])
                noise_norms.append(diagnostics["noise_l2_norm"])
            sanitized_gradient = sanitized_gradient / float(len(discriminators))
            fake_x.backward(sanitized_gradient)
            loss_value = raw_loss / float(len(discriminators))
        else:
            loss = torch.zeros((), device=device)
            for discriminator in discriminators:
                loss = loss + _per_discriminator_loss(discriminator, fake_x, fake_y)
            loss_value = loss / float(len(discriminators))
            loss_value.backward()
        optimizer.step()
        losses.append(float(loss_value.detach().cpu()))

    for discriminator in discriminators:
        set_requires_grad(discriminator, True)

    metrics = {"generator_loss": float(np.mean(losses))}
    if cfg.dp_enabled:
        metrics.update(
            {
                "dp_noise_multiplier": float(cfg.dp_effective_noise_multiplier or 0.0),
                "dp_sample_gradient_l2_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "dp_clip_coefficient_mean": float(np.mean(clip_coefficients)) if clip_coefficients else 0.0,
                "dp_noise_l2_mean": float(np.mean(noise_norms)) if noise_norms else 0.0,
            }
        )
    return metrics


def sanitize_sample_gradient(
    sample_gradient: torch.Tensor,
    cfg: CIDIoTConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cfg.dp_effective_noise_multiplier is None:
        raise ValueError("DP is enabled but no effective noise multiplier was configured")
    if cfg.dp_clip_norm <= 0.0:
        raise ValueError("DP clip norm must be positive")
    flat = sample_gradient.detach().reshape(sample_gradient.shape[0], -1)
    norms = flat.norm(p=2, dim=1).clamp_min(1e-12)
    coefficients = torch.clamp(cfg.dp_clip_norm / norms, max=1.0)
    clipped = sample_gradient.detach() * coefficients.view(-1, *([1] * (sample_gradient.dim() - 1)))
    noise = torch.randn_like(clipped) * float(cfg.dp_effective_noise_multiplier * cfg.dp_clip_norm)
    sanitized = clipped + noise
    return sanitized, {
        "raw_sample_gradient_l2_mean": float(norms.mean().cpu()),
        "clip_coefficient_mean": float(coefficients.mean().cpu()),
        "noise_l2_norm": float(noise.float().norm(p=2).cpu()),
    }


def evaluate_ensemble(
    discriminators: list[TCNDiscriminator],
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    for discriminator in discriminators:
        discriminator.eval()

    preds: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, len(y_test), 4096):
            xb = torch.from_numpy(x_test[start : start + 4096]).float().to(device)
            yb = torch.from_numpy(y_test[start : start + 4096]).long().to(device)
            probs = []
            for discriminator in discriminators:
                _, class_logits = discriminator(xb)
                probs.append(torch.softmax(class_logits, dim=1))
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            selected = avg_probs.gather(1, yb.view(-1, 1)).clamp_min(1e-12)
            total_loss += float((-torch.log(selected)).sum().cpu())
            total_count += int(len(yb))
            preds.append(avg_probs.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(preds)
    return {
        "eval_loss": float(total_loss / max(total_count, 1)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }


def run_experiment(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)

    prepared = prepare_dataset(
        args.dataset,
        args.task,
        args.max_rows,
        args.seed,
        args.test_size,
        log_scale=getattr(args, "log_scale", False),
    )
    x_train = prepared.x_train
    x_test = prepared.x_test
    y_train = prepared.y_train
    y_test = prepared.y_test
    class_names = prepared.class_names
    if args.partition == "iid":
        partitions = partition_iid(y_train, args.clients, args.seed)
    elif args.partition == "non_iid":
        partitions = partition_non_iid(y_train, args.clients, args.num_shards, args.seed)
    else:
        alpha = 0.1 if args.dirichlet_alpha is None else args.dirichlet_alpha
        partitions = partition_dirichlet(y_train, args.clients, alpha, args.seed)

    feature_dim = x_train.shape[1]
    num_classes = len(class_names)
    dp_effective_noise_multiplier = args.dp_noise_multiplier
    dp_achieved_epsilon = None
    dp_best_order = None
    if args.dp_enabled:
        if dp_effective_noise_multiplier is None:
            dp_effective_noise_multiplier, dp_achieved_epsilon, dp_best_order = derive_noise_multiplier_from_rdp(
                epsilon=args.dp_epsilon,
                delta=args.dp_delta,
                batch_size=args.batch_size,
                max_order=args.dp_rdp_max_order,
            )
        else:
            dp_achieved_epsilon, dp_best_order = epsilon_from_rdp(
                sigma=dp_effective_noise_multiplier,
                delta=args.dp_delta,
                batch_size=args.batch_size,
                max_order=args.dp_rdp_max_order,
            )
    cfg = CIDIoTConfig(
        method="cidiot_implemented_dp" if args.dp_enabled else "cidiot_implemented",
        dataset_id=prepared.dataset_id,
        dataset=str(args.dataset),
        task=args.task,
        partition=args.partition,
        dirichlet_alpha=(args.dirichlet_alpha if args.partition == "dirichlet" else None),
        clients=args.clients,
        rounds=args.rounds,
        batch_size=args.batch_size,
        discriminator_steps=args.discriminator_steps,
        discriminator_epochs=args.discriminator_epochs,
        generator_steps=args.generator_steps,
        latent_dim=args.latent_dim,
        label_dim=args.label_dim,
        tcn_channels=args.tcn_channels,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        supervised_pretrain_steps=args.supervised_pretrain_steps,
        generator_pretrain_steps=args.generator_pretrain_steps,
        source_loss_weight=args.source_loss_weight,
        real_class_loss_weight=args.real_class_loss_weight,
        fake_class_loss_weight=args.fake_class_loss_weight,
        generator_class_loss_weight=args.generator_class_loss_weight,
        lsgan_enabled=bool(args.lsgan_enabled),
        wgan_gp=bool(args.wgan_gp),
        margin_lambda=args.margin_lambda,
        gp_lambda=args.gp_lambda,
        dp_enabled=bool(args.dp_enabled),
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_effective_noise_multiplier=dp_effective_noise_multiplier,
        dp_rdp_max_order=args.dp_rdp_max_order,
        dp_rdp_achieved_epsilon=dp_achieved_epsilon,
        dp_rdp_best_order=dp_best_order,
        test_size=prepared.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        num_shards=args.num_shards,
        device=str(device),
        comparison_run_id=args.comparison_run_id,
    )

    generator = ConditionalTCNGenerator(
        feature_dim=feature_dim,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        label_dim=args.label_dim,
        channels=args.tcn_channels,
        bounded_output=not bool(args.lsgan_enabled),
    ).to(device)
    discriminators = [
        TCNDiscriminator(feature_dim, num_classes, args.tcn_channels, args.hidden_dim).to(device)
        for _ in range(args.clients)
    ]

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    discriminator_optimizers = [
        torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        for discriminator in discriminators
    ]

    start_time = time.perf_counter()
    history = []
    pretrain_generator(generator, generator_optimizer, x_train, y_train, cfg, num_classes, rng, device)
    pretrain_discriminators(discriminators, discriminator_optimizers, x_train, y_train, partitions, cfg, rng, device)
    for round_id in range(1, args.rounds + 1):
        disc_metrics = train_discriminators(
            discriminators,
            generator,
            discriminator_optimizers,
            x_train,
            y_train,
            partitions,
            cfg,
            num_classes,
            rng,
            device,
        )
        gen_metrics = train_generator(discriminators, generator, generator_optimizer, cfg, num_classes, device)
        eval_metrics = evaluate_ensemble(discriminators, x_test, y_test, device)
        metrics = {"round": round_id, **eval_metrics, **disc_metrics, **gen_metrics}
        history.append(metrics)
        print(
            f"round={round_id:03d} accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} generator_loss={metrics['generator_loss']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    final_metrics = dict(history[-1])
    final_metrics["elapsed_seconds"] = float(elapsed)

    if bool(getattr(args, "save_checkpoint", False)):
        ckpt_path = args.output_dir / (Path(result_name({"config": asdict(cfg)})).stem + ".checkpoint.pt")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dicts": [d.state_dict() for d in discriminators],
                "config": asdict(cfg),
                "partitions": [part.tolist() for part in partitions],
                "feature_dim": int(feature_dim),
                "num_classes": int(num_classes),
                "class_names": list(class_names),
            },
            ckpt_path,
        )
        print(f"wrote checkpoint {ckpt_path}", flush=True)

    return {
        "config": asdict(cfg),
        "training_audit": build_training_audit(
            config=asdict(cfg),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            partitions=partitions,
        ),
        "class_names": class_names,
        "dataset_rows": prepared.dataset_rows,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "client_sizes": [int(len(part)) for part in partitions],
        "final_metrics": final_metrics,
        "history": history,
    }


def result_name(result: dict) -> str:
    cfg = result["config"]
    suffix = f"maxrows{cfg['max_rows']}" if cfg["max_rows"] else "full"
    method = "cidiot_dp" if cfg.get("dp_enabled") else "cidiot"
    partition = cfg["partition"]
    if partition == "dirichlet" and cfg.get("dirichlet_alpha") is not None:
        alpha_text = f"{float(cfg['dirichlet_alpha']):g}".replace(".", "p").replace("-", "m")
        partition = f"dirichlet_a{alpha_text}"
    return (
        f"{method}_{cfg.get('dataset_id', 'dataset')}_{cfg['task']}_{partition}_c{cfg['clients']}_"
        f"r{cfg['rounds']}_{suffix}_seed{cfg['seed']}.json"
    )


def derive_noise_multiplier_from_rdp(
    epsilon: float,
    delta: float,
    batch_size: int,
    max_order: int,
) -> tuple[float, float, int]:
    if epsilon <= 0.0:
        raise ValueError("--dp-epsilon must be positive")
    validate_rdp_inputs(delta=delta, batch_size=batch_size, max_order=max_order)
    high = 1.0
    high_epsilon, _ = epsilon_from_rdp(
        sigma=high,
        delta=delta,
        batch_size=batch_size,
        max_order=max_order,
    )
    while high_epsilon > epsilon:
        high *= 2.0
        high_epsilon, _ = epsilon_from_rdp(
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
        mid_epsilon, _ = epsilon_from_rdp(
            sigma=mid,
            delta=delta,
            batch_size=batch_size,
            max_order=max_order,
        )
        if mid_epsilon > epsilon:
            low = mid
        else:
            high = mid
    achieved, best_order = epsilon_from_rdp(
        sigma=high,
        delta=delta,
        batch_size=batch_size,
        max_order=max_order,
    )
    return high, achieved, best_order


def epsilon_from_rdp(
    sigma: float,
    delta: float,
    batch_size: int,
    max_order: int,
) -> tuple[float, int]:
    if sigma <= 0.0:
        return math.inf, 2
    validate_rdp_inputs(delta=delta, batch_size=batch_size, max_order=max_order)
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


def validate_rdp_inputs(delta: float, batch_size: int, max_order: int) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError("--dp-delta must be in (0, 1)")
    if batch_size <= 0:
        raise ValueError("DP batch size must be positive")
    if max_order < 2:
        raise ValueError("--dp-rdp-max-order must be at least 2")


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_name(result)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
