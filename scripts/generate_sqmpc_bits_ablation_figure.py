#!/usr/bin/env python3
"""Plot SQMPC quantization-bit ablation: model accuracy vs reconstruction accuracy.

Reads SQMPC training JSONs (one per (bits, seed)) and TabLeak attack summaries
(one per (bits, seed)) from a shared output directory, then plots a single
figure with two y-axes:

  left  y-axis -> final test accuracy of the federated model (utility)
  right y-axis -> TabLeak reconstruction accuracy on the captured update (attacker)

Both as a function of uniform quantization bit-width B.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _parse_tolerances(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing SQMPC training JSONs.")
    parser.add_argument("--tableak-results-dir", type=Path, required=True,
                        help="Directory containing TabLeak attack summary JSONs.")
    parser.add_argument("--tableak-artifact-dir", type=Path, default=None,
                        help="Directory containing TabLeak capture artifacts (.pt). "
                             "Defaults to <tableak-results-dir>/../tableak_artifacts.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output SVG path.")
    parser.add_argument("--reconstruction-metric", default="reconstruction_accuracy",
                        help="Key inside summary['metrics'] to plot as the exact-match curve.")
    parser.add_argument("--continuous-tolerance", type=_parse_tolerances, default=None,
                        help="Comma-separated absolute tolerances (e.g. 0.05,0.10,0.20). "
                             "If set, overlay one right-axis curve per tolerance using "
                             "fraction-within-tolerance of the continuous features, computed "
                             "from the saved reconstruction.pt + artifact.pt pairs.")
    return parser.parse_args()


def _variant_key(
    bits_first: int,
    bits_mid: int,
    bits_last: int,
    strict: bool,
    no_quantize: bool = False,
) -> str:
    """Stable string key for the figure: uniform bits, with optional 'strict' tag."""
    if no_quantize:
        return "float"
    if not (bits_first == bits_mid == bits_last):
        return f"f{bits_first}m{bits_mid}l{bits_last}"
    if bits_first == 1 and strict:
        return "1s"
    return str(int(bits_first))


def load_training_results(results_dir: Path) -> dict[tuple[str, int], float]:
    """Map (variant_key, seed) -> final-round test accuracy."""
    by_key: dict[tuple[str, int], float] = {}
    for path in sorted(results_dir.glob("sqmpc_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        cfg = data.get("config", {})
        if cfg.get("mode") != "sqmpc":
            continue
        no_quantize = bool(cfg.get("no_quantize", False))
        bits_first = cfg.get("q_bits_first")
        bits_mid = cfg.get("q_bits_mid")
        bits_last = cfg.get("q_bits_last")
        if not no_quantize:
            if not (bits_first == bits_mid == bits_last):
                continue  # only uniform bit settings
        strict = bool(cfg.get("q_b1_strict", False))
        key = _variant_key(
            int(bits_first) if bits_first is not None else 0,
            int(bits_mid) if bits_mid is not None else 0,
            int(bits_last) if bits_last is not None else 0,
            strict,
            no_quantize,
        )
        seed = int(cfg.get("seed", -1))
        history = data.get("history", [])
        if not history:
            continue
        by_key[(key, seed)] = float(history[-1]["accuracy"])
    return by_key


def load_attack_summaries(tableak_dir: Path, metric: str) -> dict[tuple[str, int], float]:
    """Map (variant_key, seed) -> attacker reconstruction metric, parsed from summaries."""
    by_key: dict[tuple[str, int], float] = {}
    for path in sorted(tableak_dir.glob("sqmpc_*summary.json")):
        try:
            summary = json.loads(path.read_text())
        except Exception:
            continue
        metadata = summary.get("metadata", {})
        method_meta = metadata.get("method", "")
        if not str(method_meta).startswith("sqmpc"):
            continue
        bits_first = metadata.get("q_bits_first")
        bits_mid = metadata.get("q_bits_mid")
        bits_last = metadata.get("q_bits_last")
        no_quantize = bool(metadata.get("no_quantize", False))
        if not no_quantize:
            if not (bits_first is not None and bits_first == bits_mid == bits_last):
                continue
        seed = metadata.get("seed")
        if seed is None:
            continue
        strict = bool(metadata.get("q_b1_strict", False))
        key = _variant_key(
            int(bits_first) if bits_first is not None else 0,
            int(bits_mid) if bits_mid is not None else 0,
            int(bits_last) if bits_last is not None else 0,
            strict,
            no_quantize,
        )
        seed = int(seed)
        metric_val = summary.get("metrics", {}).get(metric)
        if metric_val is None:
            continue
        by_key[(key, seed)] = float(metric_val)
    return by_key


def load_tolerance_metrics(
    tableak_results_dir: Path,
    tableak_artifact_dir: Path,
    tolerances: list[float],
) -> dict[float, dict[tuple[str, int], float]]:
    """Recompute "fraction of continuous features within +/- tolerance" from
    saved reconstruction tensors. Returns one (variant_key, seed) -> value
    mapping per tolerance."""
    import torch
    out: dict[float, dict[tuple[str, int], float]] = {tol: {} for tol in tolerances}
    for recon_path in sorted(tableak_results_dir.glob("sqmpc_*_reconstruction.pt")):
        stem = recon_path.name[: -len("_reconstruction.pt")]
        artifact_path = tableak_artifact_dir / f"{stem}.pt"
        if not artifact_path.exists():
            continue
        try:
            recon = torch.load(recon_path, map_location="cpu", weights_only=False)
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        rf = recon.get("reconstructed_features")
        tf = artifact.get("true_batch_features")
        if rf is None or tf is None or rf.numel() == 0 or tf.numel() == 0:
            continue
        n = min(rf.shape[0], tf.shape[0])
        if n == 0:
            continue
        rf = rf[:n].float()
        tf = tf[:n].float()
        metadata = artifact.get("metadata", {})
        dataset_meta = metadata.get("dataset_metadata", {})
        n_cont = int(dataset_meta.get("num_continuous_features", rf.shape[1]))
        if n_cont <= 0:
            continue
        # Standard schema: continuous columns first, categoricals one-hot after.
        cont_recon = rf[:, :n_cont]
        cont_true = tf[:, :n_cont]
        diff = (cont_recon - cont_true).abs()
        bf = metadata.get("q_bits_first")
        bm = metadata.get("q_bits_mid")
        bl = metadata.get("q_bits_last")
        strict = bool(metadata.get("q_b1_strict", False))
        no_quantize = bool(metadata.get("no_quantize", False))
        seed_val = metadata.get("seed")
        if seed_val is None:
            continue
        key = _variant_key(
            int(bf) if bf is not None else 0,
            int(bm) if bm is not None else 0,
            int(bl) if bl is not None else 0,
            strict,
            no_quantize,
        )
        seed = int(seed_val)
        for tol in tolerances:
            out[tol][(key, seed)] = float((diff <= tol).float().mean().item())
    return out


def aggregate(by_key: dict[tuple[str, int], float]) -> dict[str, tuple[float, float, int]]:
    """Aggregate by variant key across seeds -> (mean, std_ddof1, n)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for (variant, _seed), val in by_key.items():
        grouped[variant].append(val)
    out: dict[str, tuple[float, float, int]] = {}
    for variant, vals in grouped.items():
        arr = np.array(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[variant] = (float(arr.mean()), std, len(arr))
    return out


def _variant_sort_key(key: str) -> tuple[int, int]:
    """Sort variants: 1s, 1, 2, 4, ..., 16, float (no quantization last)."""
    if key == "1s":
        return (1, 0)
    if key == "1":
        return (1, 1)
    if key == "float":
        return (10**6, 0)
    try:
        return (int(key), 0)
    except ValueError:
        return (10**9, 0)


def _variant_label(key: str) -> str:
    if key == "1":
        return "1 (3-lvl)"
    if key == "1s":
        return "1 (2-lvl)"
    if key == "float":
        return "float"
    return key


def main() -> None:
    args = parse_args()
    train = aggregate(load_training_results(args.results_dir))
    attack = aggregate(load_attack_summaries(args.tableak_results_dir, args.reconstruction_metric))

    variant_keys = sorted(set(train.keys()) | set(attack.keys()), key=_variant_sort_key)
    if not variant_keys:
        raise SystemExit("no data found")

    train_mean = np.array([train.get(k, (np.nan, np.nan, 0))[0] for k in variant_keys])
    train_std = np.array([train.get(k, (np.nan, np.nan, 0))[1] for k in variant_keys])
    attack_mean = np.array([attack.get(k, (np.nan, np.nan, 0))[0] for k in variant_keys])
    attack_std = np.array([attack.get(k, (np.nan, np.nan, 0))[1] for k in variant_keys])

    labels = [_variant_label(k) for k in variant_keys]
    print("bits          model_acc(mean±std)    recon_acc(mean±std)")
    for i, lbl in enumerate(labels):
        print(f"  {lbl:11s}  {train_mean[i]:.4f} ± {train_std[i]:.4f}    "
              f"{attack_mean[i]:.4f} ± {attack_std[i]:.4f}")

    x_positions = np.arange(len(variant_keys), dtype=float)
    fig, ax_left = plt.subplots(figsize=(8.5, 5.4))
    color_left = "#1f77b4"
    color_right = "#d62728"

    line_left = ax_left.errorbar(
        x_positions, train_mean, yerr=train_std, color=color_left,
        marker="o", linewidth=2, capsize=3, label="Model accuracy (utility)",
    )
    ax_left.set_xlabel(f"Quantization bit-size ($B$)", fontsize=14)
    ax_left.set_ylabel("Model test accuracy", color=color_left, fontsize=14)
    ax_left.tick_params(axis="y", labelcolor=color_left)
    # ax_left.set_ylim(0.0, 1.02)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xticks(x_positions)
    ax_left.set_ylim(0.6,0.9)
    ax_left.set_xticklabels(labels)

    ax_right = ax_left.twinx()
    line_right = ax_right.errorbar(
        x_positions, attack_mean, yerr=attack_std, color=color_right,
        marker="s", linewidth=2, capsize=3, linestyle="--",
        label="Recon accuracy (exact match)",
    )
    ax_right.set_ylabel("Reconstruction accuracy", color=color_right, fontsize=14)
    ax_right.tick_params(axis="y", labelcolor=color_right)
    # ax_right.set_ylim(0.0, 1.02)
    ax_right.set_ylim(0.0, 0.3)

    handles = [line_left, line_right]

    # Optional: overlay tolerance-based curves recomputed from saved reconstruction tensors.
    if args.continuous_tolerance:
        artifact_dir = args.tableak_artifact_dir
        if artifact_dir is None:
            artifact_dir = args.tableak_results_dir.parent / "tableak_artifacts"
        tol_metrics = load_tolerance_metrics(
            args.tableak_results_dir, artifact_dir, args.continuous_tolerance,
        )
        # Color palette for tolerances: light -> dark orange/red.
        palette = ["#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]
        markers = ["^", "D", "v", "P"]
        print()
        print("tolerance curves (fraction of continuous values within +/- tol):")
        for i, tol in enumerate(args.continuous_tolerance):
            tol_by_var = aggregate(tol_metrics.get(tol, {}))
            means = np.array([tol_by_var.get(k, (np.nan, np.nan, 0))[0] for k in variant_keys])
            stds = np.array([tol_by_var.get(k, (np.nan, np.nan, 0))[1] for k in variant_keys])
            color = palette[i % len(palette)]
            marker = markers[i % len(markers)]
            # label = f"Recon @ tol={tol:g} ({tol*100:.0f}%)"
            label = f"Recon @ tol={tol*100:.0f}%"
            line = ax_right.errorbar(
                # x_positions, means, yerr=stds, color=color,
                x_positions, means, color=color,
                marker=marker, linewidth=2, capsize=3, linestyle=":",
                label=label,
            )
            handles.append(line)
            print(f"  tol={tol:g}:")
            for lbl, m, s in zip(labels, means, stds):
                print(f"    {lbl:11s}  {m:.4f} ± {s:.4f}")

    leg_labels = [h.get_label() for h in handles]
    ax_left.legend(handles, leg_labels, loc="upper left", fontsize=12, framealpha=0.9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
