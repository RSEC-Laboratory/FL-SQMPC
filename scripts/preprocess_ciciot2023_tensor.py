#!/usr/bin/env python3
"""Preprocess CICIoT2023 CSV into a tensor-split directory compatible with
TabLeak and the FL pipeline's ``is_tensor_split_dataset`` loader.

Outputs (under ``--output-dir``):
  X_train.pt, X_test.pt, y_train.pt, y_test.pt -- float32 features, int64 labels
  scaler.pkl                                   -- fitted sklearn MinMaxScaler
  meta.json                                    -- TabLeak-compatible metadata

All CICIoT2023 features are numeric (flag counts, byte stats, etc.), so we
treat every feature as continuous and emit an empty categorical schema.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "ciciot2023_processed" / "CicIoT_extracted02.csv"


# Coarse-category label map matching MULTICLASS_LABELS in run_fl_sqmpc_accuracy.py.
LABEL_MAP = {
    "Benign":     0,
    "DDoS":       1,
    "DoS":        2,
    "Recon":      3,
    "Web":        4,
    "BruteForce": 5,
    "Spoofing":   6,
    "Mirai":      7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "ciciot2023_processed_tensor")
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-scale", action="store_true",
                        help="Apply log1p to features before MinMax scaling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)
    if "category" not in df.columns or "Label" not in df.columns:
        raise SystemExit("expected 'Label' and 'category' columns in CSV")

    categories = df["category"].astype(str)
    if args.task == "binary":
        benign_aliases = {"Benign", "benign", "normal", "Normal"}
        y_int = (~categories.isin(benign_aliases)).astype(np.int64).to_numpy()
        label_map = {"Benign": 0, "Attack": 1}
        class_names = ["Benign", "Attack"]
    else:
        observed = sorted(set(categories.unique()))
        unknown = [c for c in observed if c not in LABEL_MAP]
        if unknown:
            raise SystemExit(f"unknown category labels: {unknown}")
        label_map = {name: LABEL_MAP[name] for name in observed}
        y_int = categories.map(label_map).astype(np.int64).to_numpy()
        class_names = sorted(label_map, key=lambda n: label_map[n])

    feature_df = df.drop(columns=["Label", "category"])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    valid_mask = ~feature_df.isna().any(axis=1).to_numpy()
    feature_df = feature_df.loc[valid_mask].reset_index(drop=True)
    y_int = y_int[valid_mask]

    feature_names = list(feature_df.columns)
    x = feature_df.astype(np.float32).to_numpy()

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y_int, test_size=args.test_size, stratify=y_int, random_state=args.seed,
    )
    if args.log_scale:
        x_train_raw = np.log1p(np.clip(x_train_raw, 0.0, None))
        x_test_raw = np.log1p(np.clip(x_test_raw, 0.0, None))
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    torch.save(torch.from_numpy(x_train), args.output_dir / "X_train.pt")
    torch.save(torch.from_numpy(x_test), args.output_dir / "X_test.pt")
    torch.save(torch.from_numpy(y_train.astype(np.int64)), args.output_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_test.astype(np.int64)), args.output_dir / "y_test.pt")

    with open(args.output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    class_frequencies = {name: int((y_int == idx).sum()) for name, idx in label_map.items()}

    meta = {
        "auxiliary_target_columns": ["Label"],
        "categorical_features": [],
        "class_frequencies": class_frequencies,
        "continuous_features": feature_names,
        "dataset_id": "ciciot2023_tensor" if args.task == "multiclass" else "ciciot2023_tensor_binary",
        "display_name": "CICIoT2023 (tensor split)",
        "dropped_columns": [],
        "label_map": label_map,
        "log_scale": bool(args.log_scale),
        "num_categorical_features": 0,
        "num_continuous_features": len(feature_names),
        "one_hot_ranges": {},
        "original_feature_count": len(feature_names),
        "processed_feature_count": len(feature_names),
        "target_column": "category" if args.task == "multiclass" else "Label",
        "task": args.task,
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "seed": int(args.seed),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {args.output_dir} (train={len(y_train)}, test={len(y_test)}, classes={class_names})")


if __name__ == "__main__":
    main()
