#!/usr/bin/env python3
"""Run CICIoT2023 FL-SQMPC accuracy experiments.

This simulates the accuracy-relevant part of FL-SQMPC: FedAvg with
mixed-precision quantization applied to client updates. The SMPC share
transport is intentionally skipped because correct secret sharing and
reconstruction returns the same aggregate and should not change accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sqmpc_core
from dp_he import dp_clip_and_noise_flat, sigma_from_epsilon_rdp
from experiment_audit import build_training_audit

try:
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False

DEFAULT_DATASET = ROOT / "ciciot2023_processed" / "CicIoT_extracted02.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "results"
TENSOR_SPLIT_FILES = ("X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt")

MULTICLASS_LABELS = {
    "Benign": 0,
    "DDoS": 1,
    "DoS": 2,
    "Recon": 3,
    "Web": 4,
    "BruteForce": 5,
    "Spoofing": 6,
    "Mirai": 7,
}


@dataclass
class ExperimentConfig:
    dataset_id: str
    dataset: str
    task: str
    partition: str
    mode: str
    clients: int
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    server_lr: float
    optimizer: str
    hidden_sizes: list[int]
    test_size: float
    seed: int
    max_rows: int | None
    num_shards: int
    dirichlet_alpha: float | None
    log_scale: bool
    q_bits_first: int
    q_bits_mid: int
    q_bits_last: int
    q_b1_strict: bool
    no_quantize: bool
    dp_epsilon: float | None
    dp_delta: float | None
    dp_clip_norm: float | None
    dp_noise_multiplier: float | None
    dp_adaptive_clipping: bool
    device: str
    comparison_run_id: str | None


@dataclass
class PreparedDataset:
    dataset_id: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    dataset_rows: int
    test_size: float


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Iterable[int]):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    parser.add_argument("--partition", choices=["iid", "non_iid", "dirichlet"], default="iid")
    parser.add_argument(
        "--mode",
        choices=["vanilla", "sqmpc", "he", "dp", "dpsgd"],
        default="sqmpc",
        help=(
            "vanilla=FedAvg, sqmpc=mixed-precision quantization, "
            "he=accuracy-equivalent to vanilla (CKKS is exact), "
            "dp=client-level DP (clip whole update + Gaussian noise), "
            "dpsgd=sample-level DP-SGD per client via Opacus"
        ),
    )
    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=0.5)
    parser.add_argument("--dp-noise-multiplier", type=float, default=None,
                        help="If None, derive from (epsilon, delta, rounds) via RDP accountant.")
    parser.add_argument("--dp-adaptive-clipping", action="store_true",
                        help="Use AdaptiveClipper (percentile-based) instead of fixed clip norm.")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=None,
        help="Dirichlet concentration parameter used when --partition dirichlet.",
    )
    parser.add_argument("--q-bits-first", type=int, default=12)
    parser.add_argument("--q-bits-mid", type=int, default=8)
    parser.add_argument("--q-bits-last", type=int, default=12)
    parser.add_argument(
        "--q-b1-strict",
        action="store_true",
        help="When B=1, force every element to +/-scale (no zero preservation).",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip the SQMPC quantization step entirely (float-precision FedAvg).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--comparison-run-id", default=None)
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Apply log1p to features before MinMax scaling (helps heavy-tailed columns).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--hidden-sizes must contain at least one integer")
    return sizes


def is_tensor_split_dataset(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in TENSOR_SPLIT_FILES)


def dataset_id_for_path(path: Path) -> str:
    if path.is_dir() and (path / "meta.json").exists():
        try:
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            dataset_id = meta.get("dataset_id")
            if dataset_id:
                return str(dataset_id)
        except (OSError, json.JSONDecodeError):
            pass

    path_text = str(path).lower()
    if "ciciot" in path_text:
        return "ciciot2023"
    if "ton" in path_text and "iot" in path_text:
        return "ton_iot_extracted"
    if "bot" in path_text and "iot" in path_text:
        return "bot_iot"

    name = path.stem if path.is_file() else path.name
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    return slug or "dataset"


def requires_log_scale(dataset_id: str) -> bool:
    return dataset_id == "ton_iot_extracted"


def stratified_subset_indices(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size >= len(y):
        return np.arange(len(y), dtype=np.int64)
    if size < len(np.unique(y)):
        raise ValueError("--max-rows is too small for stratified sampling.")
    idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=size,
        stratify=y,
        random_state=seed,
    )
    return np.array(sorted(idx), dtype=np.int64)


def apply_max_rows_to_existing_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_rows is None:
        return x_train, x_test, y_train, y_test

    total_rows = len(y_train) + len(y_test)
    if max_rows >= total_rows:
        return x_train, x_test, y_train, y_test

    num_classes = max(len(np.unique(y_train)), len(np.unique(y_test)))
    if max_rows < num_classes * 2:
        raise ValueError("--max-rows is too small for stratified train/test sampling.")

    test_ratio = len(y_test) / float(total_rows)
    target_test = max(num_classes, int(round(max_rows * test_ratio)))
    target_train = max_rows - target_test
    if target_train < num_classes:
        target_train = num_classes
        target_test = max_rows - target_train
    if target_test < num_classes:
        raise ValueError("--max-rows leaves too few test rows for stratified sampling.")

    train_idx = stratified_subset_indices(y_train, target_train, seed)
    test_idx = stratified_subset_indices(y_test, target_test, seed + 1)
    return x_train[train_idx], x_test[test_idx], y_train[train_idx], y_test[test_idx]


def load_tensor_split_dataset(path: Path, task: str, max_rows: int | None, seed: int) -> PreparedDataset:
    if not is_tensor_split_dataset(path):
        raise ValueError(f"Expected processed tensor split files under {path}.")

    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    x_train = torch.load(path / "X_train.pt", map_location="cpu").detach().cpu().numpy().astype(np.float32, copy=False)
    y_train = torch.load(path / "y_train.pt", map_location="cpu").detach().cpu().numpy().astype(np.int64, copy=False)
    x_test = torch.load(path / "X_test.pt", map_location="cpu").detach().cpu().numpy().astype(np.float32, copy=False)
    y_test = torch.load(path / "y_test.pt", map_location="cpu").detach().cpu().numpy().astype(np.int64, copy=False)

    train_valid = np.isfinite(x_train).all(axis=1)
    test_valid = np.isfinite(x_test).all(axis=1)
    x_train = x_train[train_valid]
    y_train = y_train[train_valid]
    x_test = x_test[test_valid]
    y_test = y_test[test_valid]

    label_map = meta.get("label_map", {})
    if task == "binary":
        normal_label = label_map.get("Normal", label_map.get("Benign"))
        if normal_label is None:
            raise ValueError("Binary task for tensor datasets requires a Normal or Benign label in meta.json.")
        y_train = (y_train != int(normal_label)).astype(np.int64)
        y_test = (y_test != int(normal_label)).astype(np.int64)
        class_names = ["Normal", "Attack"]
    else:
        if label_map:
            inverse = sorted((int(label), str(name)) for name, label in label_map.items())
            label_to_index = {label: index for index, (label, _) in enumerate(inverse)}
            known_labels = set(label_to_index)
            observed = set(int(label) for label in np.unique(np.concatenate([y_train, y_test])))
            unknown = sorted(observed - known_labels)
            if unknown:
                raise ValueError(f"Unknown labels in tensor dataset: {unknown}")
            y_train = np.array([label_to_index[int(label)] for label in y_train], dtype=np.int64)
            y_test = np.array([label_to_index[int(label)] for label in y_test], dtype=np.int64)
            class_names = [name for _, name in inverse]
        else:
            labels = sorted(int(label) for label in np.unique(np.concatenate([y_train, y_test])))
            label_to_index = {label: index for index, label in enumerate(labels)}
            y_train = np.array([label_to_index[int(label)] for label in y_train], dtype=np.int64)
            y_test = np.array([label_to_index[int(label)] for label in y_test], dtype=np.int64)
            class_names = [str(label) for label in labels]

    x_train, x_test, y_train, y_test = apply_max_rows_to_existing_split(
        x_train,
        x_test,
        y_train,
        y_test,
        max_rows,
        seed,
    )
    dataset_rows = int(len(y_train) + len(y_test))
    actual_test_size = float(len(y_test) / max(dataset_rows, 1))
    return PreparedDataset(
        dataset_id=dataset_id_for_path(path),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        dataset_rows=dataset_rows,
        test_size=actual_test_size,
    )


def load_dataset(path: Path, task: str, max_rows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if path.is_dir():
        raise ValueError("Directory datasets contain an existing split; use prepare_dataset instead.")

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if "category" not in df.columns or "Label" not in df.columns:
        raise ValueError("Expected CICIoT columns 'Label' and 'category'.")

    categories = df["category"].astype(str)
    if task == "binary":
        benign_aliases = {"Benign", "benign", "normal", "Normal"}
        y = (~categories.isin(benign_aliases)).astype(np.int64).to_numpy()
        class_names = ["Benign", "Attack"]
    else:
        observed = set(categories.unique())
        if observed.issubset(set(MULTICLASS_LABELS)):
            label_map = MULTICLASS_LABELS
        else:
            label_map = {name: idx for idx, name in enumerate(sorted(observed))}
        y = categories.map(label_map).astype(np.int64).to_numpy()
        class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    feature_df = df.drop(columns=["Label", "category"])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    valid = ~feature_df.isna().any(axis=1).to_numpy()
    x = feature_df.loc[valid].astype(np.float32).to_numpy()
    y = y[valid]

    if max_rows is not None and max_rows < len(y):
        if max_rows < len(np.unique(y)) * 2:
            raise ValueError("--max-rows is too small for stratified sampling.")
        idx, _ = train_test_split(
            np.arange(len(y)),
            train_size=max_rows,
            stratify=y,
            random_state=seed,
        )
        x = x[idx]
        y = y[idx]

    return x, y, class_names


def prepare_dataset(
    path: Path,
    task: str,
    max_rows: int | None,
    seed: int,
    test_size: float,
    log_scale: bool = False,
) -> PreparedDataset:
    if is_tensor_split_dataset(path):
        return load_tensor_split_dataset(path, task, max_rows, seed)

    x, y, class_names = load_dataset(path, task, max_rows, seed)
    x_train, x_test, y_train, y_test = split_and_scale(x, y, test_size, seed, log_scale=log_scale)
    return PreparedDataset(
        dataset_id=dataset_id_for_path(path),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        dataset_rows=int(len(y)),
        test_size=float(test_size),
    )


def split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    log_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    if log_scale:
        x_train = np.log1p(np.clip(x_train, 0.0, None))
        x_test = np.log1p(np.clip(x_test, 0.0, None))
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    x_train = np.clip(x_train, 0.0, 1.0)
    x_test = np.clip(x_test, 0.0, 1.0)
    return x_train, x_test, y_train, y_test


def partition_iid(y: np.ndarray, clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_parts: list[list[int]] = [[] for _ in range(clients)]
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        for client_id, split in enumerate(np.array_split(label_idx, clients)):
            client_parts[client_id].extend(split.tolist())
    return [np.array(sorted(part), dtype=np.int64) for part in client_parts]


def partition_non_iid(y: np.ndarray, clients: int, num_shards: int, seed: int) -> list[np.ndarray]:
    if num_shards < clients:
        raise ValueError("--num-shards must be >= --clients")
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(y, kind="stable")
    shards = [shard for shard in np.array_split(sorted_idx, num_shards) if len(shard) > 0]
    rng.shuffle(shards)
    client_parts: list[list[int]] = [[] for _ in range(clients)]
    for shard_id, shard in enumerate(shards):
        client_parts[shard_id % clients].extend(shard.tolist())
    return [np.array(sorted(part), dtype=np.int64) for part in client_parts]


def partition_dirichlet(
    y: np.ndarray,
    clients: int,
    alpha: float,
    seed: int,
    max_attempts: int = 100,
) -> list[np.ndarray]:
    if clients <= 0:
        raise ValueError("--clients must be positive")
    if alpha <= 0.0:
        raise ValueError("--dirichlet-alpha must be positive")

    labels = np.unique(y)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        client_parts: list[list[int]] = [[] for _ in range(clients)]
        for label in labels:
            label_idx = np.where(y == label)[0]
            rng.shuffle(label_idx)
            proportions = rng.dirichlet(np.full(clients, alpha, dtype=float))
            cut_points = (np.cumsum(proportions)[:-1] * len(label_idx)).astype(int)
            for client_id, split in enumerate(np.split(label_idx, cut_points)):
                client_parts[client_id].extend(split.tolist())
        if all(client_parts):
            return [np.array(sorted(part), dtype=np.int64) for part in client_parts]

    raise RuntimeError(
        "Dirichlet partition produced an empty client after repeated attempts; "
        "increase --dirichlet-alpha or reduce --clients."
    )


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    partitions: list[np.ndarray],
    batch_size: int,
) -> list[DataLoader]:
    loaders = []
    for idx in partitions:
        dataset = TensorDataset(
            torch.from_numpy(x_train[idx]).float(),
            torch.from_numpy(y_train[idx]).long(),
        )
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders


def make_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_local(
    global_model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    return_state: bool = False,
):
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = make_optimizer(local_model, cfg.optimizer, cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if cfg.mode == "dpsgd":
        if not _HAS_OPACUS:
            raise RuntimeError("--mode dpsgd requires opacus to be installed")
        privacy_engine = PrivacyEngine(accountant="rdp")
        local_model, optimizer, loader = privacy_engine.make_private(
            module=local_model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=cfg.dp_noise_multiplier,
            max_grad_norm=cfg.dp_clip_norm,
        )

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(local_model(xb), yb)
            loss.backward()
            optimizer.step()

    if cfg.mode == "dpsgd":
        local_params = [p for p in local_model._module.parameters()]
    else:
        local_params = list(local_model.parameters())

    updates = []
    with torch.no_grad():
        for local_param, global_param in zip(local_params, global_model.parameters(), strict=True):
            updates.append((local_param.detach() - global_param.detach().to(device)).cpu())

    if return_state:
        if cfg.mode == "dpsgd":
            state = {k: v.detach().cpu().clone() for k, v in local_model._module.state_dict().items()}
        else:
            state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
        return updates, state
    return updates


def compute_client_eval_metrics(
    template_model: nn.Module,
    client_states: list[dict[str, torch.Tensor]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate K post-local-training client models on the test set.

    Returns metrics that match the CIDIoT paper's reporting convention:
    - mean_client_*: mean of per-client metrics (matches their "AvgAcc"
      column, which is "average performance of all edge nodes").
    - ensemble_*: macro metrics on argmax of mean softmax across clients
      (matches the structure of CIDIoT's "Accuracy" column, where their
      per-edge TCN classifiers are ensembled at test time).
    """
    per_client_acc: list[float] = []
    per_client_precision: list[float] = []
    per_client_recall: list[float] = []
    per_client_f1: list[float] = []
    probs_sum: np.ndarray | None = None
    for state in client_states:
        cm = copy.deepcopy(template_model).to(device)
        cm.load_state_dict(state)
        cm.eval()
        chunk_probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(y_test), 4096):
                xb = torch.from_numpy(x_test[start : start + 4096]).float().to(device)
                logits = cm(xb)
                chunk_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(chunk_probs, axis=0)
        preds = probs.argmax(axis=1)
        per_client_acc.append(float(accuracy_score(y_test, preds)))
        per_client_precision.append(float(precision_score(y_test, preds, average="macro", zero_division=0)))
        per_client_recall.append(float(recall_score(y_test, preds, average="macro", zero_division=0)))
        per_client_f1.append(float(f1_score(y_test, preds, average="macro", zero_division=0)))
        probs_sum = probs if probs_sum is None else probs_sum + probs
    assert probs_sum is not None
    ensemble_preds = probs_sum.argmax(axis=1)
    return {
        "mean_client_accuracy":  float(np.mean(per_client_acc)),
        "mean_client_precision": float(np.mean(per_client_precision)),
        "mean_client_recall":    float(np.mean(per_client_recall)),
        "mean_client_f1":        float(np.mean(per_client_f1)),
        "ensemble_accuracy":  float(accuracy_score(y_test, ensemble_preds)),
        "ensemble_precision": float(precision_score(y_test, ensemble_preds, average="macro", zero_division=0)),
        "ensemble_recall":    float(recall_score(y_test, ensemble_preds, average="macro", zero_division=0)),
        "ensemble_f1":        float(f1_score(y_test, ensemble_preds, average="macro", zero_division=0)),
        "per_client_accuracy": per_client_acc,
    }


def apply_quantization(updates: list[torch.Tensor], cfg: ExperimentConfig) -> list[torch.Tensor]:
    if cfg.no_quantize:
        return [u.cpu() for u in updates]
    sqmpc_core.SQMPC_Q_BITS_FIRST = cfg.q_bits_first
    sqmpc_core.SQMPC_Q_BITS_MID = cfg.q_bits_mid
    sqmpc_core.SQMPC_Q_BITS_LAST = cfg.q_bits_last
    sqmpc_core.SQMPC_Q_B1_STRICT = bool(cfg.q_b1_strict)
    sqmpc_core.SQMPC_Q_SCOPE = "all"
    return [tensor.cpu() for tensor in sqmpc_core.mixed_quantize_mlp([u.cpu() for u in updates])]


def aggregate_updates(
    client_updates: list[list[torch.Tensor]],
    client_sizes: list[int],
    cfg: ExperimentConfig,
) -> list[torch.Tensor]:
    total = float(sum(client_sizes))
    aggregate = [torch.zeros_like(layer_update) for layer_update in client_updates[0]]
    for updates, size in zip(client_updates, client_sizes, strict=True):
        if cfg.mode == "sqmpc":
            effective_updates = apply_quantization(updates, cfg)
        elif cfg.mode == "dp":
            effective_updates, _ = dp_clip_and_noise_flat(
                updates,
                C=cfg.dp_clip_norm,
                sigma=cfg.dp_noise_multiplier,
                adaptive=cfg.dp_adaptive_clipping,
            )
        else:
            effective_updates = updates
        weight = float(size) / total
        for layer_id, update in enumerate(effective_updates):
            aggregate[layer_id] += update * weight
    return aggregate


def apply_global_update(model: nn.Module, aggregate: list[torch.Tensor], server_lr: float, device: torch.device) -> None:
    with torch.no_grad():
        for param, update in zip(model.parameters(), aggregate, strict=True):
            param.add_(update.to(device), alpha=server_lr)


def predict_labels(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).float().to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for start in range(0, len(y), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).float().to(device)
            yb = torch.from_numpy(y[start : start + 4096]).long().to(device)
            logits = model(xb)
            total_loss += float(criterion(logits, yb).cpu())
            total_count += int(len(yb))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return {
        "eval_loss": float(total_loss / max(total_count, 1)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
    }


def run_experiment(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = select_device(args.device)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    dataset_id = dataset_id_for_path(args.dataset)
    requested_log_scale = bool(getattr(args, "log_scale", False))
    effective_log_scale = requested_log_scale or requires_log_scale(dataset_id)
    if effective_log_scale and not requested_log_scale:
        print("forcing --log-scale for ToN-IoT dataset", flush=True)
    prepared = prepare_dataset(
        args.dataset,
        args.task,
        args.max_rows,
        args.seed,
        args.test_size,
        log_scale=effective_log_scale,
    )

    cfg = ExperimentConfig(
        dataset_id=prepared.dataset_id,
        dataset=str(args.dataset),
        task=args.task,
        partition=args.partition,
        mode=args.mode,
        clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        server_lr=args.server_lr,
        optimizer=args.optimizer,
        hidden_sizes=hidden_sizes,
        test_size=prepared.test_size,
        seed=args.seed,
        max_rows=args.max_rows,
        num_shards=args.num_shards,
        dirichlet_alpha=args.dirichlet_alpha,
        log_scale=effective_log_scale,
        q_bits_first=args.q_bits_first,
        q_bits_mid=args.q_bits_mid,
        q_bits_last=args.q_bits_last,
        q_b1_strict=bool(args.q_b1_strict),
        no_quantize=bool(args.no_quantize),
        dp_epsilon=(args.dp_epsilon if args.mode in {"dp", "dpsgd"} else None),
        dp_delta=(args.dp_delta if args.mode in {"dp", "dpsgd"} else None),
        dp_clip_norm=(args.dp_clip_norm if args.mode in {"dp", "dpsgd"} else None),
        dp_noise_multiplier=None,
        dp_adaptive_clipping=bool(args.dp_adaptive_clipping),
        device=str(device),
        comparison_run_id=args.comparison_run_id,
    )
    if args.mode == "dp":
        if args.dp_noise_multiplier is not None:
            cfg.dp_noise_multiplier = float(args.dp_noise_multiplier)
        else:
            cfg.dp_noise_multiplier = sigma_from_epsilon_rdp(
                epsilon=args.dp_epsilon,
                steps=args.rounds,
                delta=args.dp_delta,
            )
        print(
            f"DP-FedAvg: epsilon={args.dp_epsilon}, delta={args.dp_delta}, "
            f"rounds={args.rounds}, clip={args.dp_clip_norm}, sigma={cfg.dp_noise_multiplier:.4f}",
            flush=True,
        )
    elif args.mode == "dpsgd":
        if not _HAS_OPACUS:
            raise RuntimeError("--mode dpsgd requires opacus")
        cfg.dp_epsilon = args.dp_epsilon
        cfg.dp_delta = args.dp_delta
        cfg.dp_clip_norm = args.dp_clip_norm

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
        cfg.dirichlet_alpha = alpha
        partitions = partition_dirichlet(y_train, args.clients, alpha, args.seed)

    client_sizes = [int(len(part)) for part in partitions]
    if any(size == 0 for size in client_sizes):
        raise RuntimeError(f"Empty client partition detected: {client_sizes}")

    loaders = make_loaders(x_train, y_train, partitions, args.batch_size)
    model = MLP(input_dim=x_train.shape[1], output_dim=len(class_names), hidden_sizes=hidden_sizes).to(device)

    if args.mode == "dpsgd":
        avg_client_size = float(np.mean(client_sizes))
        sample_rate = float(args.batch_size) / max(avg_client_size, 1.0)
        total_epochs = int(args.local_epochs * args.rounds)
        cfg.dp_noise_multiplier = float(get_noise_multiplier(
            target_epsilon=args.dp_epsilon,
            target_delta=args.dp_delta,
            sample_rate=sample_rate,
            epochs=total_epochs,
            accountant="rdp",
        ))
        print(
            f"DP-SGD (Opacus): epsilon={args.dp_epsilon}, delta={args.dp_delta}, "
            f"sample_rate={sample_rate:.5f}, total_epochs={total_epochs}, "
            f"clip={args.dp_clip_norm}, sigma={cfg.dp_noise_multiplier:.4f}",
            flush=True,
        )

    start_time = time.perf_counter()
    history = []
    final_client_states: list[dict[str, torch.Tensor]] | None = None
    for round_id in range(1, args.rounds + 1):
        is_final = round_id == args.rounds
        if is_final:
            results = [
                train_local(model, loader, cfg, device, return_state=True)
                for loader in loaders
            ]
            client_updates = [r[0] for r in results]
            final_client_states = [r[1] for r in results]
        else:
            client_updates = [train_local(model, loader, cfg, device) for loader in loaders]
        aggregate = aggregate_updates(client_updates, client_sizes, cfg)
        apply_global_update(model, aggregate, args.server_lr, device)
        metrics = evaluate(model, x_test, y_test, device)
        metrics["round"] = round_id
        history.append(metrics)
        print(
            f"round={round_id:03d} accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    final_metrics = dict(history[-1])
    final_metrics["elapsed_seconds"] = float(elapsed)

    y_pred_final = predict_labels(model, x_test, device)
    cm_labels = list(range(len(class_names)))
    final_metrics["confusion_matrix"] = confusion_matrix(
        y_test, y_pred_final, labels=cm_labels
    ).astype(int).tolist()

    if final_client_states is not None:
        client_eval = compute_client_eval_metrics(
            template_model=model,
            client_states=final_client_states,
            x_test=x_test,
            y_test=y_test,
            device=device,
        )
        for k, v in client_eval.items():
            final_metrics[k] = v
        print(
            f"final round client eval: "
            f"mean_client_accuracy={client_eval['mean_client_accuracy']:.4f} "
            f"ensemble_accuracy={client_eval['ensemble_accuracy']:.4f} "
            f"mean_client_f1={client_eval['mean_client_f1']:.4f} "
            f"ensemble_f1={client_eval['ensemble_f1']:.4f}",
            flush=True,
        )

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
        "client_sizes": client_sizes,
        "final_metrics": final_metrics,
        "history": history,
    }


def result_name(result: dict) -> str:
    cfg = result["config"]
    suffix_parts = []
    if cfg.get("log_scale"):
        suffix_parts.append("logscale")
    suffix_parts.append(f"maxrows{cfg['max_rows']}" if cfg["max_rows"] else "full")
    suffix = "_".join(suffix_parts)
    partition = cfg["partition"]
    if partition == "dirichlet":
        alpha_text = f"{float(cfg['dirichlet_alpha']):g}".replace(".", "p").replace("-", "m")
        partition = f"dirichlet_a{alpha_text}"
    mode_prefix = cfg["mode"]
    if mode_prefix in {"sqmpc", "vanilla", "he", "dp", "dpsgd"}:
        lr_text = f"{float(cfg['learning_rate']):g}".replace(".", "p").replace("-", "m")
        mode_prefix = f"{mode_prefix}_lr{lr_text}"
    if cfg["mode"] in {"dp", "dpsgd"} and cfg.get("dp_epsilon") is not None:
        eps_text = f"{float(cfg['dp_epsilon']):g}".replace(".", "p").replace("-", "m")
        mode_prefix = f"{mode_prefix}_eps{eps_text}"
    if cfg["mode"] == "sqmpc":
        if cfg.get("no_quantize"):
            mode_prefix = f"{mode_prefix}_qFloat"
        else:
            bf = cfg.get("q_bits_first"); bm = cfg.get("q_bits_mid"); bl = cfg.get("q_bits_last")
            if bf is not None:
                bits_tag = f"qB{bf}" if bf == bm == bl else f"qBf{bf}m{bm}l{bl}"
                if bf == 1 and cfg.get("q_b1_strict"):
                    bits_tag = f"{bits_tag}s"
                mode_prefix = f"{mode_prefix}_{bits_tag}"
    hidden = cfg.get("hidden_sizes") or []
    hidden_tag = "h" + "x".join(str(h) for h in hidden) if hidden else ""
    return (
        f"{mode_prefix}_{cfg.get('dataset_id', 'dataset')}_{cfg['task']}_{partition}_"
        f"c{cfg['clients']}_r{cfg['rounds']}_{hidden_tag}_{suffix}_seed{cfg['seed']}.json"
    )


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_name(result)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
