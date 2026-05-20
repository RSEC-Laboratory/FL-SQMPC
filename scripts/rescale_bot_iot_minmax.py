#!/usr/bin/env python3
"""Produce a [0,1] MinMax-scaled copy of the existing Bot-IoT tensor split.

The current Bot_IoT_processed_no_pkSeqID_saddr_daddr/ directory uses arbitrary
feature scaling (some continuous columns span 100+ units, others include
negative values). TabLeak's gradient-inversion attack clamps its candidate
reconstructions to [continuous_min, continuous_max] = [0, 1], so the attack
literally cannot reach Bot-IoT's true feature values. This script rescales
every column by the train-set per-column (min, max) so reconstructions live
in the same [0, 1] range as the data.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "Bot_IoT_processed_no_pkSeqID_saddr_daddr")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "Bot_IoT_processed_minmax")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train = torch.load(args.input_dir / "X_train.pt", map_location="cpu").float()
    x_test = torch.load(args.input_dir / "X_test.pt", map_location="cpu").float()
    y_train = torch.load(args.input_dir / "y_train.pt", map_location="cpu")
    y_test = torch.load(args.input_dir / "y_test.pt", map_location="cpu")

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    x_train_scaled = scaler.fit_transform(x_train.numpy()).astype(np.float32)
    x_test_scaled = scaler.transform(x_test.numpy()).astype(np.float32)
    # Clip in case test has values outside train min/max (would extrapolate past [0,1]).
    x_test_scaled = np.clip(x_test_scaled, 0.0, 1.0)

    torch.save(torch.from_numpy(x_train_scaled), args.output_dir / "X_train.pt")
    torch.save(torch.from_numpy(x_test_scaled), args.output_dir / "X_test.pt")
    torch.save(y_train, args.output_dir / "y_train.pt")
    torch.save(y_test, args.output_dir / "y_test.pt")

    with open(args.output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    # Copy + amend meta.
    meta = json.loads((args.input_dir / "meta.json").read_text())
    meta = dict(meta)
    meta["dataset_id"] = "bot_iot_minmax"
    meta["display_name"] = "Bot-IoT (MinMax rescaled to [0,1])"
    meta["log_scale"] = False
    meta["preprocessing_note"] = (
        "Re-scaled from Bot_IoT_processed_no_pkSeqID_saddr_daddr using "
        "sklearn MinMaxScaler fit on X_train; X_test was clipped to [0,1] "
        "to avoid extrapolation past the train range."
    )
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {args.output_dir}")
    print(f"  X_train min/max: {x_train_scaled.min():.4f} / {x_train_scaled.max():.4f}")
    print(f"  X_test  min/max: {x_test_scaled.min():.4f} / {x_test_scaled.max():.4f}")


if __name__ == "__main__":
    main()
