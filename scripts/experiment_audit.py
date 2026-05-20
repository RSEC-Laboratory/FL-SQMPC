"""Shared training-setup audit helpers for experiment result files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


COMMON_TRAINING_KEYS = [
    "dataset",
    "task",
    "partition",
    "clients",
    "rounds",
    "batch_size",
    "test_size",
    "seed",
    "max_rows",
    "num_shards",
    "dirichlet_alpha",
    "log_scale",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def common_training_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in COMMON_TRAINING_KEYS}


def partition_fingerprints(partitions: list[np.ndarray], y_train: np.ndarray) -> list[dict[str, Any]]:
    fingerprints = []
    for client_id, indices in enumerate(partitions):
        labels, counts = np.unique(y_train[indices], return_counts=True)
        fingerprints.append(
            {
                "client_id": client_id,
                "size": int(len(indices)),
                "indices_sha256": array_sha256(indices.astype(np.int64)),
                "label_counts": {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)},
            }
        )
    return fingerprints


def build_training_audit(
    *,
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    partitions: list[np.ndarray],
) -> dict[str, Any]:
    dataset = Path(config["dataset"])
    audit = {
        "common_settings": common_training_settings(config),
        "dataset_sha256": file_sha256(dataset),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "split_fingerprints": {
            "x_train_sha256": array_sha256(x_train),
            "y_train_sha256": array_sha256(y_train.astype(np.int64)),
            "x_test_sha256": array_sha256(x_test),
            "y_test_sha256": array_sha256(y_test.astype(np.int64)),
        },
        "client_partitions": partition_fingerprints(partitions, y_train),
    }
    audit["training_setup_signature"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return audit
