#!/usr/bin/env python3
"""Build a LaTeX reconstruction-accuracy table from the recon-table sweep.

For each (method, dataset, partition, capture round), aggregates over seeds and
reports "fraction of continuous features reconstructed within +/- tolerance of
the true min-max-scaled value", computed live from each saved reconstruction.pt
+ artifact.pt pair (so any tolerance can be requested without re-running the
attacks).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


METHOD_LABEL = {
    "vanilla_fl":  "Vanilla FL",
    "dp_eps8":     r"DP-SGD ($\varepsilon{=}8$)",
    "dp_eps50":    r"DP-SGD ($\varepsilon{=}50$)",
    "cidiot":      "CIDIoT",
    "sqmpc_qb2":   "SQMPC ($B{=}2$)",
    "sqmpc_qb12":  "SQMPC ($B{=}12$)",
}
METHOD_ORDER = ["vanilla_fl", "dp_eps8", "dp_eps50", "cidiot", "sqmpc_qb2", "sqmpc_qb12"]

DATASET_LABEL = {
    "ciciot2023_tensor": "CICIoT2023",
    "ton_iot_tensor":    "ToN-IoT",
    "bot_iot_minmax":    "Bot-IoT",
}
DATASET_ORDER = ["ciciot2023_tensor", "ton_iot_tensor", "bot_iot_minmax"]

PARTITION_LABEL = {"iid": "IID", "non_iid": "non-IID"}
PARTITION_ORDER = ["iid", "non_iid"]


def classify_method(metadata: dict) -> str | None:
    """Identify which row this artifact corresponds to."""
    method = metadata.get("method", "") or ""
    if method.startswith("cidiot"):
        return "cidiot"
    if method.startswith("sqmpc"):
        if metadata.get("dpsgd_enabled"):
            eps = metadata.get("dp_epsilon")
            if eps is None:
                return None
            if abs(float(eps) - 8.0) < 1e-3:
                return "dp_eps8"
            if abs(float(eps) - 50.0) < 1e-3:
                return "dp_eps50"
            return None
        if metadata.get("no_quantize"):
            return "vanilla_fl"
        bf = metadata.get("q_bits_first")
        bm = metadata.get("q_bits_mid")
        bl = metadata.get("q_bits_last")
        if bf == bm == bl:
            if bf == 2:
                return "sqmpc_qb2"
            if bf == 12:
                return "sqmpc_qb12"
    return None


_RANGE_CACHE: dict[str, torch.Tensor] = {}


def _per_feature_range(dataset_path: str) -> torch.Tensor | None:
    """Return (max - min) per column on the training set. Cached by path."""
    if dataset_path in _RANGE_CACHE:
        return _RANGE_CACHE[dataset_path]
    path = Path(dataset_path)
    if not path.is_absolute():
        path = ROOT / path
    x_path = path / "X_train.pt"
    if not x_path.exists():
        _RANGE_CACHE[dataset_path] = None
        return None
    x = torch.load(x_path, map_location="cpu", weights_only=False).float()
    rng = (x.max(dim=0).values - x.min(dim=0).values).clamp_min(1e-12)
    _RANGE_CACHE[dataset_path] = rng
    return rng


def load_tolerance_value(
    recon_path: Path,
    artifact_path: Path,
    tolerance: float,
) -> float | None:
    """Return fraction of continuous features within +/- tolerance of the
    per-feature training-set range (so tolerance=0.10 means "within 10% of
    that feature's natural range", regardless of underlying scaling)."""
    try:
        recon = torch.load(recon_path, map_location="cpu", weights_only=False)
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    rf = recon.get("reconstructed_features")
    tf = artifact.get("true_batch_features")
    if rf is None or tf is None or rf.numel() == 0 or tf.numel() == 0:
        return None
    n = min(rf.shape[0], tf.shape[0])
    if n == 0:
        return None
    rf = rf[:n].float()
    tf = tf[:n].float()
    metadata = artifact.get("metadata", {})
    dataset_meta = metadata.get("dataset_metadata", {})
    n_cont = int(dataset_meta.get("num_continuous_features", rf.shape[1]))
    if n_cont <= 0:
        return None
    cont_recon = rf[:, :n_cont]
    cont_true = tf[:, :n_cont]
    abs_diff = (cont_recon - cont_true).abs()
    dataset_path = metadata.get("dataset_path") or metadata.get("dataset", "")
    feature_range = _per_feature_range(str(dataset_path))
    if feature_range is None:
        # Fall back to absolute tolerance.
        return float((abs_diff <= tolerance).float().mean().item())
    feat_range_cont = feature_range[:n_cont].view(1, -1)
    norm_diff = abs_diff / feat_range_cont
    return float((norm_diff <= tolerance).float().mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with TabLeak summary JSONs.")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Directory with TabLeak capture artifact .pt files.")
    parser.add_argument("--metric", default="reconstruction_accuracy",
                        choices=[
                            "reconstruction_accuracy",
                            "reconstruction_continuous_accuracy",
                            "categorical_exact_match",
                            "continuous_tolerance_accuracy",
                            "tableak_tolerance_accuracy",
                            "custom_tolerance",
                        ],
                        help="Which metric to put in each cell. "
                             "'reconstruction_accuracy' is the TabLeak paper's combined "
                             "continuous (sigma-tolerance) + categorical (exact match) score. "
                             "'custom_tolerance' uses --tolerance with per-feature range normalization.")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Only used when --metric custom_tolerance: per-feature tolerance "
                             "as fraction of training-set range. Default 0.10 (10%).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .tex path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # (method, dataset, partition, round) -> list of values across seeds
    cells: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)

    for summary_path in sorted(args.results_dir.glob("*_summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        metadata = summary.get("metadata", {})
        method_key = classify_method(metadata)
        if method_key is None:
            continue
        dataset_id = metadata.get("dataset")
        partition = metadata.get("partition")
        round_id = int(metadata.get("round", 0))
        if dataset_id not in DATASET_LABEL or partition not in PARTITION_LABEL:
            continue
        if args.metric == "custom_tolerance":
            recon_rel = summary.get("reconstruction_path")
            if not recon_rel:
                continue
            recon_path = Path(recon_rel)
            if not recon_path.is_absolute():
                recon_path = ROOT / recon_path
            if not recon_path.exists():
                recon_path = args.results_dir / recon_path.name
            if not recon_path.exists():
                continue
            artifact_name = recon_path.name.replace("_reconstruction.pt", ".pt")
            artifact_path = args.artifact_dir / artifact_name
            if not artifact_path.exists():
                continue
            value = load_tolerance_value(recon_path, artifact_path, args.tolerance)
        else:
            value = summary.get("metrics", {}).get(args.metric)
            if value is not None:
                value = float(value)
        if value is None:
            continue
        cells[(method_key, dataset_id, partition, round_id)].append(value)

    print(f"Loaded {sum(len(v) for v in cells.values())} per-seed measurements")
    if args.metric == "custom_tolerance":
        print(f"Metric: per-feature normalized tolerance, ±{args.tolerance*100:.1f}% of training-set range")
    else:
        print(f"Metric: summary['metrics']['{args.metric}']")
    print()
    print(f"{'method':<22s} {'dataset':<22s} {'part':<8s} {'r':>3s}  {'mean ± std (n)':>20s}")
    for key, vals in sorted(cells.items()):
        m, d, p, r = key
        arr = np.array(vals, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"  {m:<20s} {d:<22s} {p:<8s} {r:>3d}  {mean:.3f} ± {std:.3f} (n={len(arr)})")

    # Emit LaTeX.
    rows = METHOD_ORDER
    # Group columns: dataset > partition > round
    rounds = [1, 10]
    lines: list[str] = []
    lines.append(r"% Reconstruction-accuracy table generated by generate_recon_table.py")
    lines.append(r"% Tolerance = " + f"{args.tolerance*100:.1f}% of [0,1] range; 3 seeds, mean +/- std")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # 6 dataset x partition combos, each with R1+R10 = 12 numeric columns
    col_spec = "l" + "cc" * (len(DATASET_ORDER) * len(PARTITION_ORDER))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    # Header row 1: datasets
    header_ds = [r"Method"]
    for d in DATASET_ORDER:
        header_ds.append(
            r"\multicolumn{4}{c}{" + DATASET_LABEL[d] + "}"
        )
    lines.append(" & ".join(header_ds) + r" \\")
    # Header row 2: partitions
    header_part = [""]
    for d in DATASET_ORDER:
        for p in PARTITION_ORDER:
            header_part.append(r"\multicolumn{2}{c}{" + PARTITION_LABEL[p] + "}")
    lines.append(" & ".join(header_part) + r" \\")
    # Header row 3: rounds
    header_r = [""]
    for d in DATASET_ORDER:
        for p in PARTITION_ORDER:
            for r in rounds:
                header_r.append(f"R{r}")
    lines.append(" & ".join(header_r) + r" \\")
    lines.append(r"\midrule")

    for m in rows:
        if not any(((m, d, p, r) in cells) for d in DATASET_ORDER for p in PARTITION_ORDER for r in rounds):
            continue
        row = [METHOD_LABEL[m]]
        for d in DATASET_ORDER:
            for p in PARTITION_ORDER:
                for r in rounds:
                    vals = cells.get((m, d, p, r), [])
                    if not vals:
                        row.append("--")
                    else:
                        arr = np.array(vals, dtype=float)
                        mean = arr.mean()
                        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                        row.append(f"{mean:.2f} {{\\scriptsize $\\pm${std:.2f}}}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if args.metric == "custom_tolerance":
        tol_pct = f"{args.tolerance*100:.0f}\\%"
        metric_desc = (
            r"fraction of continuous features whose reconstruction lies within $\pm$"
            + tol_pct + r" of that feature's training-set range"
        )
    elif args.metric == "reconstruction_accuracy":
        metric_desc = (
            r"TabLeak reconstruction accuracy: continuous features within "
            r"$0.319\sigma$ tolerance (per-feature) plus categorical features "
            r"matching exactly, averaged over all features"
        )
    elif args.metric == "reconstruction_continuous_accuracy":
        metric_desc = (
            r"fraction of continuous features within $0.319\sigma$ tolerance"
        )
    elif args.metric == "categorical_exact_match":
        metric_desc = r"fraction of categorical features matched exactly"
    else:
        metric_desc = f"summary metric \\texttt{{{args.metric}}}"
    lines.append(
        r"\caption{Reconstruction accuracy (" + metric_desc + r") "
        r"across three IoT IDS datasets, IID and non-IID ($\alpha{=}0.3$) partitions, "
        r"and two FedAvg capture rounds. "
        r"Attack: TabLeak with batch size 1, 1500 optimization steps, 32 ensemble "
        r"restarts, oracle labels; 5 clients, Adam $\eta{=}10^{-3}$; "
        r"each cell is mean $\pm$ std over 3 seeds.}"
    )
    lines.append(r"\label{tab:recon}")
    lines.append(r"\end{table*}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
