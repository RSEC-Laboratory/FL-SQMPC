"""Aggregate SQMPC final metrics across seeds for all three datasets in
IID and Dirichlet non-IID settings, lr=1e-3, 100 rounds.

Sources (all SQMPC, lr=1e-3, K=5, batch=100, Adam, MLP [128,64], r=100):
- CICIoT2023 / Bot-IoT: results_20260511_sqmpc_dirichlet_a0p1_r100/
                       and    results_20260511_sqmpc_dirichlet_a0p3_r100/
- ToN-IoT:              same dirs, log_scale=True variants ("_logscale_full_*")

Writes: output/tables_20260514_sqmpc_metrics/sqmpc_full_metrics.{csv,md}
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR_A01 = ROOT / "output" / "results_20260511_sqmpc_dirichlet_a0p1_r100"
DIR_A03 = ROOT / "output" / "results_20260511_sqmpc_dirichlet_a0p3_r100"
OUT = ROOT / "output" / "tables_20260514_sqmpc_metrics"

# (dataset_id, dataset_label, filename_suffix)
# ToN-IoT uses the log_scale=True variant ("_logscale" suffix in filenames)
DATASETS = [
    ("ciciot2023", "CICIoT2023", ""),
    ("bot_iot", "Bot-IoT", ""),
    ("ton_iot_extracted", "ToN-IoT", "_logscale"),
]
TASKS = ["binary", "multiclass"]

# (label, source_dir, partition_token)
PARTITIONS = [
    ("IID",                 DIR_A01, "iid"),                 # IID files identical in both dirs
    ("Dirichlet α=0.3",     DIR_A03, "dirichlet_a0p3"),
    ("Dirichlet α=0.1",     DIR_A01, "dirichlet_a0p1"),
]

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("f1_macro", "F1 (macro)"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("eval_loss", "Eval loss"),
]


def load_runs(dataset_id: str, task: str, src_dir: Path,
              partition_token: str, suffix: str):
    pattern = (
        f"sqmpc_{dataset_id}_{task}_{partition_token}"
        f"_c5_r100{suffix}_full_seed*.json"
    )
    runs = []
    for path in sorted(src_dir.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        cfg = d.get("config", {})
        if abs(cfg.get("learning_rate", 0.0) - 1e-3) > 1e-12:
            continue
        if cfg.get("mode") != "sqmpc":
            continue
        runs.append((cfg.get("seed"), d["final_metrics"]))
    return runs


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_id, dataset_label, suffix in DATASETS:
        for task in TASKS:
            for part_label, src_dir, part_token in PARTITIONS:
                runs = load_runs(dataset_id, task, src_dir, part_token, suffix)
                seeds = ",".join(str(s) for s, _ in runs)
                row = {
                    "dataset": dataset_label,
                    "task": task,
                    "partition": part_label,
                    "lr": "1e-3",
                    "rounds": 100,
                    "clients": 5,
                    "seeds": seeds,
                    "n_seeds": len(runs),
                }
                for key, label in METRICS:
                    vals = [m.get(key) for _, m in runs if m.get(key) is not None]
                    mu, sd = mean_std(vals)
                    row[f"{label}_mean"] = mu
                    row[f"{label}_std"] = sd
                rows.append(row)

    csv_path = OUT / "sqmpc_full_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUT / "sqmpc_full_metrics.md"
    metric_labels = [label for _, label in METRICS]
    headers = ["Dataset", "Task", "Partition", "Seeds", *metric_labels]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        def fmt(label: str) -> str:
            mu = r[f"{label}_mean"]
            sd = r[f"{label}_std"]
            if mu != mu:  # NaN
                return "n/a"
            return f"{mu:.4f} ± {sd:.4f}"
        lines.append(
            "| "
            + " | ".join([
                r["dataset"], r["task"], r["partition"], r["seeds"] or "-",
                *(fmt(label) for label in metric_labels),
            ])
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
