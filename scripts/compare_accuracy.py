#!/usr/bin/env python3
"""Build implemented FL-SQMPC vs implemented CIDIoT accuracy tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_audit import build_training_audit
from run_fl_sqmpc_accuracy import dataset_id_for_path, partition_iid, partition_non_iid, prepare_dataset

DEFAULT_RESULTS_DIR = ROOT / "output" / "results"
DEFAULT_TABLES_DIR = ROOT / "output" / "tables"

CID_IOT_REFERENCE = {
    ("ciciot2023", "multiclass", "iid"): 0.7720,
    ("ciciot2023", "multiclass", "non_iid"): 0.7196,
}

METHOD_LABELS = {
    "cidiot_implemented": "CIDIoT implemented",
    "sqmpc": "FL-SQMPC implemented without SMPC transport",
    "vanilla": "Vanilla FL implemented",
}


def audit_from_config(config: dict) -> dict:
    prepared = prepare_dataset(
        Path(config["dataset"]),
        config["task"],
        config["max_rows"],
        config["seed"],
        config["test_size"],
    )
    if config["partition"] == "iid":
        partitions = partition_iid(prepared.y_train, config["clients"], config["seed"])
    else:
        partitions = partition_non_iid(prepared.y_train, config["clients"], config["num_shards"], config["seed"])
    return build_training_audit(
        config=config,
        x_train=prepared.x_train,
        y_train=prepared.y_train,
        x_test=prepared.x_test,
        y_test=prepared.y_test,
        partitions=partitions,
    )


def config_dataset_id(config: dict) -> str:
    return str(config.get("dataset_id") or dataset_id_for_path(Path(config["dataset"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--include-smoke", action="store_true", help="Include max-rows smoke-test results.")
    return parser.parse_args()


def load_rows(results_dir: Path, include_smoke: bool) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg = result["config"]
        if cfg["max_rows"] is not None and not include_smoke:
            continue
        metrics = result["final_metrics"]
        training_audit = result.get("training_audit") or audit_from_config(cfg)
        dataset_id = config_dataset_id(cfg)
        reference = CID_IOT_REFERENCE.get((dataset_id, cfg["task"], cfg["partition"]))
        method_key = cfg.get("method", cfg.get("mode", "unknown"))
        rows.append(
            {
                "result_file": path.name,
                "dataset_id": dataset_id,
                "dataset": cfg["dataset"],
                "method": METHOD_LABELS.get(method_key, method_key),
                "method_key": method_key,
                "task": cfg["task"],
                "partition": cfg["partition"],
                "clients": cfg["clients"],
                "rounds": cfg["rounds"],
                "seed": cfg["seed"],
                "max_rows": cfg["max_rows"] or "full",
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "cidiot_paper_accuracy_reference": reference,
                "dataset_sha256": training_audit["dataset_sha256"],
                "train_rows": training_audit["train_rows"],
                "test_rows": training_audit["test_rows"],
                "training_setup_signature": training_audit["training_setup_signature"],
                "comparison_run_id": cfg.get("comparison_run_id"),
            }
        )
    return rows


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)

    group_cols = ["dataset_id", "dataset_sha256", "task", "partition", "clients", "rounds", "seed", "max_rows"]
    for (dataset_id, dataset_sha256, task, partition, clients, rounds, seed, max_rows), group in df.groupby(group_cols):
        sqmpc = group[group["method_key"] == "sqmpc"]
        cidiot = group[group["method_key"] == "cidiot_implemented"]
        if sqmpc.empty or cidiot.empty:
            continue
        sqmpc_row = sqmpc.sort_values(["rounds", "result_file"]).iloc[-1]
        cidiot_row = cidiot.sort_values(["rounds", "result_file"]).iloc[-1]
        reference = CID_IOT_REFERENCE.get((dataset_id, task, partition))
        rows.append(
            {
                "dataset_id": dataset_id,
                "task": task,
                "partition": partition,
                "clients": clients,
                "rounds": rounds,
                "seed": seed,
                "max_rows": max_rows,
                "fl_sqmpc_accuracy": sqmpc_row["accuracy"],
                "cidiot_implemented_accuracy": cidiot_row["accuracy"],
                "fl_sqmpc_minus_cidiot_implemented": sqmpc_row["accuracy"] - cidiot_row["accuracy"],
                "cidiot_paper_accuracy_reference": reference,
                "same_training_setup": sqmpc_row["training_setup_signature"] == cidiot_row["training_setup_signature"],
                "dataset_sha256": dataset_sha256,
            }
        )
    return pd.DataFrame(rows)


def build_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pairwise.empty:
        return pd.DataFrame(rows)

    group_cols = ["dataset_id", "dataset_sha256", "task", "partition", "clients", "rounds", "max_rows"]
    metric_cols = [
        "fl_sqmpc_accuracy",
        "cidiot_implemented_accuracy",
        "fl_sqmpc_minus_cidiot_implemented",
    ]
    for keys, group in pairwise.groupby(group_cols):
        dataset_id, dataset_sha256, task, partition, clients, rounds, max_rows = keys
        row = {
            "dataset_id": dataset_id,
            "task": task,
            "partition": partition,
            "clients": clients,
            "rounds": rounds,
            "max_rows": max_rows,
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].unique())),
            "all_same_training_setup": bool(group["same_training_setup"].all()),
            "dataset_sha256": dataset_sha256,
            "cidiot_paper_accuracy_reference": group["cidiot_paper_accuracy_reference"].iloc[0],
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        output_path.write_text("No result JSON files found.\n", encoding="utf-8")
        return

    table = df.copy()
    numeric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "balanced_accuracy",
        "cidiot_paper_accuracy_reference",
        "fl_sqmpc_accuracy",
        "cidiot_implemented_accuracy",
        "fl_sqmpc_minus_cidiot_implemented",
        "fl_sqmpc_accuracy_mean",
        "fl_sqmpc_accuracy_std",
        "cidiot_implemented_accuracy_mean",
        "cidiot_implemented_accuracy_std",
        "fl_sqmpc_minus_cidiot_implemented_mean",
        "fl_sqmpc_minus_cidiot_implemented_std",
    ]
    for col in numeric_cols:
        if col in table.columns:
            table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")

    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir, args.include_smoke)
    df = pd.DataFrame(rows)
    public_df = df
    csv_path = args.tables_dir / "accuracy_comparison.csv"
    md_path = args.tables_dir / "accuracy_comparison.md"
    pairwise_csv_path = args.tables_dir / "accuracy_pairwise.csv"
    pairwise_md_path = args.tables_dir / "accuracy_pairwise.md"
    summary_csv_path = args.tables_dir / "accuracy_summary.csv"
    summary_md_path = args.tables_dir / "accuracy_summary.md"
    public_df.to_csv(csv_path, index=False)
    write_markdown(public_df, md_path)
    pairwise = build_pairwise(df)
    pairwise.to_csv(pairwise_csv_path, index=False)
    write_markdown(pairwise, pairwise_md_path)
    summary = build_summary(pairwise)
    summary.to_csv(summary_csv_path, index=False)
    write_markdown(summary, summary_md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {pairwise_csv_path}")
    print(f"wrote {pairwise_md_path}")
    print(f"wrote {summary_csv_path}")
    print(f"wrote {summary_md_path}")


if __name__ == "__main__":
    main()
