#!/usr/bin/env python3
"""Add or refresh training-setup audits in result JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_accuracy import audit_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "output" / "results")
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in sorted(args.results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        if config.get("max_rows") is not None and not args.include_smoke:
            continue
        result["training_audit"] = audit_from_config(config)
        result.pop("environment", None)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"refreshed {path}")


if __name__ == "__main__":
    main()
