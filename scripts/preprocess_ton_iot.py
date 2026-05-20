#!/usr/bin/env python3
"""Preprocess raw ToN_IoT CSV into the shared feature/label format.

Default input/output paths (relative to repo root):
  --raw-path  ToN_IoT_raw/ToN_IoT_extracted 1.csv
  --out-dir   ToN_IoT_processed/

Column conventions match the existing runners:
  - Label: binary int (0=normal, 1=attack)
  - category: multiclass string (e.g. normal, password, ddos ...)
  - All other columns: numeric features

Processing steps:
  1. Drop identifier and high-cardinality string columns:
     src_ip, dst_ip, dns_query
  2. Label-encode all remaining categorical (str-typed) columns.
  3. Rename 'label' -> 'Label', 'type' -> 'category'.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def _find_project_root() -> Path:
    """Walk up from the script to find the dir that contains ToN_IoT_raw."""
    candidate = Path(__file__).resolve().parents[1]
    for _ in range(6):
        if (candidate / "ToN_IoT_raw").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parents[1]


ROOT = _find_project_root()

DROP_COLS = {"src_ip", "dst_ip", "dns_query"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=ROOT / "ToN_IoT_raw" / "ToN_IoT_extracted 1.csv",
        help="Path to the raw ToN_IoT CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "ToN_IoT_processed",
        help="Output directory for the processed CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path: Path = args.raw_path
    out_path: Path = args.out_dir / "ToN_IoT_extracted.csv"

    if not raw_path.exists():
        sys.exit(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    label = df["label"].copy()
    category = df["type"].copy()

    to_drop = (DROP_COLS | {"label", "type"}) & set(df.columns)
    df = df.drop(columns=list(to_drop))

    str_cols = [c for c in df.columns if df[c].dtype.kind == "O"]
    print(f"Label-encoding {len(str_cols)} categorical columns: {str_cols}")

    for c in str_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    df["Label"] = label.values
    df["category"] = category.values

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {out_path}")
    print(f"\nLabel distribution:\n{df['Label'].value_counts().to_string()}")
    print(f"\nCategory distribution:\n{df['category'].value_counts().to_string()}")
    print(f"\nFeature count: {len(df.columns) - 2}")


if __name__ == "__main__":
    main()
