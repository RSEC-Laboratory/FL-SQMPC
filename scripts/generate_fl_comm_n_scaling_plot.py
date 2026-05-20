#!/usr/bin/env python3
"""Plot per-round communication cost as a function of N_clients.

Companion to ``generate_fl_comm_5method_plot.py``. The other script fixes
N_clients=5 and uses methods on the x-axis. This one fixes the methods and
**sweeps N_clients** to show how each protocol's bytes-per-round scale.

Layout: one row per dataset, four columns per row -
client upload / client download / server upload / server download.
One line per method on log-log axes; markers highlight N=7, 255, 1023.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reuse the formulas from the per-round-bars script so the two figures stay
# numerically consistent.
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from generate_fl_comm_5method_plot import (  # noqa: E402
    BATCH_SIZE_DEFAULT,
    DATASET_DIMS,
    DATASET_LABELS,
    FLOAT_BYTES,
    HE_CT_BYTES,
    METHOD_ORDER,
    SQMPC_Q_BITS,
    he_ciphertext_count,
    mlp_param_count,
    sqmpc_bits_per_share,
)


HIGHLIGHT_N = (7, 255, 1023)
N_AGG_DEFAULT = 2

METHOD_COLORS = {
    "Vanilla FL": "#1f77b4",
    "HE": "#d62728",
    "SMPC": "#9467bd",
    "DP": "#8c564b",
    "CIDIoT": "#2ca02c",
    "FL-SQMPC": "#ff7f0e",
}


def per_client_bytes(method: str, dataset_id: str, n_clients: int,
                     n_agg: int, batch_size: int, sqmpc_q_bits: int) -> tuple[float, float]:
    """Return (upload_bytes, download_bytes) per client per round."""
    dims = DATASET_DIMS[dataset_id]
    d, C = dims["d"], dims["C"]
    P = mlp_param_count(d, C)

    if method in {"Vanilla FL", "DP"}:
        return FLOAT_BYTES * P, FLOAT_BYTES * P
    if method == "HE":
        n_ct = he_ciphertext_count(P)
        return n_ct * HE_CT_BYTES, n_ct * HE_CT_BYTES
    if method == "SMPC":
        return n_agg * FLOAT_BYTES * P, FLOAT_BYTES * P
    if method == "CIDIoT":
        flow = batch_size * d * FLOAT_BYTES
        return flow, flow
    if method == "FL-SQMPC":
        bps = sqmpc_bits_per_share(sqmpc_q_bits, n_clients)
        return n_agg * bps * P / 8.0, FLOAT_BYTES * P
    raise ValueError(f"unknown method: {method}")


def per_server_bytes(method: str, dataset_id: str, n_clients: int,
                     n_agg: int, batch_size: int, sqmpc_q_bits: int) -> tuple[float, float]:
    up, dn = per_client_bytes(method, dataset_id, n_clients, n_agg, batch_size, sqmpc_q_bits)
    return n_clients * up, n_clients * dn


def _bytes_axis_formatter(value: float, _pos: int) -> str:
    if value <= 0:
        return "0"
    units = [("B", 1.0), ("KB", 1024.0), ("MB", 1024.0 ** 2), ("GB", 1024.0 ** 3)]
    for label, scale in reversed(units):
        if value >= scale:
            ratio = value / scale
            return f"{ratio:g} {label}" if ratio < 10 else f"{ratio:.0f} {label}"
    return f"{value:g} B"


# Round binary-unit byte tick locations: 1 KB, 10 KB, 100 KB, 1 MB, 10 MB, ... up to 100 GB.
_BYTE_TICKS: list[float] = []
for tier in range(4):  # 0=KB, 1=MB, 2=GB, 3=TB
    unit = 1024 ** (tier + 1)
    for mult in (1, 10, 100):
        _BYTE_TICKS.append(float(unit * mult))
_BYTE_TICKS = sorted(set(_BYTE_TICKS))


def _plot_panel(ax, n_grid: np.ndarray, dataset_id: str, side: str, channel: str,
                n_agg: int, batch_size: int, sqmpc_q_bits: int) -> None:
    title = f"{DATASET_LABELS[dataset_id]} - {side} {channel}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("N_clients")
    ax.set_ylabel("bytes per round")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_bytes_axis_formatter))
    # X-axis: anchor at the highlight Ns plus log decades.
    ax.set_xticks(list(HIGHLIGHT_N))
    ax.set_xticklabels([str(n) for n in HIGHLIGHT_N])

    chan_idx = 0 if channel == "upload" else 1
    fn = per_client_bytes if side == "client" else per_server_bytes

    for method in METHOD_ORDER:
        ys = np.array(
            [fn(method, dataset_id, int(n), n_agg, batch_size, sqmpc_q_bits)[chan_idx]
             for n in n_grid],
            dtype=float,
        )
        ax.plot(
            n_grid, ys,
            color=METHOD_COLORS[method],
            linewidth=1.6,
            label=method,
        )
        # Highlight markers at the requested anchor N values.
        marker_x = np.array(HIGHLIGHT_N, dtype=float)
        marker_y = np.array(
            [fn(method, dataset_id, int(n), n_agg, batch_size, sqmpc_q_bits)[chan_idx]
             for n in marker_x],
            dtype=float,
        )
        ax.scatter(marker_x, marker_y, color=METHOD_COLORS[method],
                   s=22, zorder=5, edgecolor="white", linewidths=0.6)

    # Vertical guides at the highlight Ns.
    for n in HIGHLIGHT_N:
        ax.axvline(n, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)

    # Y-axis ticks: snap to round binary-unit byte values within the data range.
    ymin, ymax = ax.get_ylim()
    visible = [t for t in _BYTE_TICKS if ymin * 0.5 <= t <= ymax * 2.0]
    if visible:
        ax.set_yticks(visible)


SIDE_CHANNEL = (
    ("client", "upload"),
    ("client", "download"),
    ("server", "upload"),
    ("server", "download"),
)


def _build_grid(n_min: int, n_max: int) -> np.ndarray:
    n_grid = np.unique(np.concatenate([
        np.logspace(np.log10(n_min), np.log10(n_max), 64),
        np.array(HIGHLIGHT_N, dtype=float),
    ]))
    return n_grid[(n_grid >= n_min) & (n_grid <= n_max)]


def _add_legend_and_caption(fig, sample_ax, batch_size: int, n_agg: int,
                            sqmpc_q_bits: int) -> None:
    handles, labels = sample_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(METHOD_ORDER),
        fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f"Per-round communication cost vs N_clients  "
        f"(b={batch_size}, n_agg={n_agg}, FL-SQMPC q={sqmpc_q_bits})",
        fontsize=14, y=0.995,
    )
    fig.text(
        0.5, 0.05,
        f"Dotted vertical lines mark N = {', '.join(map(str, HIGHLIGHT_N))}.  "
        f"FL-SQMPC share width grows as ceil(log2(N * 2^q)) bits/share, so its per-client "
        f"upload is the only client-side curve that depends on N.",
        ha="center", va="bottom", fontsize=10, style="italic",
    )


def plot_scaling_combined(output_path: Path, n_min: int, n_max: int,
                          n_agg: int, batch_size: int, sqmpc_q_bits: int) -> None:
    n_grid = _build_grid(n_min, n_max)
    dataset_ids = list(DATASET_LABELS)
    fig, axes = plt.subplots(
        len(dataset_ids), 4,
        figsize=(22.0, 5.0 * len(dataset_ids)),
        sharex=True,
        dpi=140,
    )
    axes = np.atleast_2d(axes)
    for row_idx, dataset_id in enumerate(dataset_ids):
        for col_idx, (side, channel) in enumerate(SIDE_CHANNEL):
            _plot_panel(
                axes[row_idx, col_idx], n_grid, dataset_id, side, channel,
                n_agg, batch_size, sqmpc_q_bits,
            )
    _add_legend_and_caption(fig, axes[0, 0], batch_size, n_agg, sqmpc_q_bits)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_scaling_per_dataset(output_path: Path, dataset_id: str,
                             n_min: int, n_max: int,
                             n_agg: int, batch_size: int,
                             sqmpc_q_bits: int) -> None:
    n_grid = _build_grid(n_min, n_max)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True, dpi=140)
    flat = axes.flatten()
    for ax, (side, channel) in zip(flat, SIDE_CHANNEL, strict=True):
        _plot_panel(ax, n_grid, dataset_id, side, channel,
                    n_agg, batch_size, sqmpc_q_bits)
    _add_legend_and_caption(fig, flat[0], batch_size, n_agg, sqmpc_q_bits)
    fig.suptitle(
        f"{DATASET_LABELS[dataset_id]} - per-round communication vs N_clients  "
        f"(b={batch_size}, n_agg={n_agg}, FL-SQMPC q={sqmpc_q_bits})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "figures_20260515_fl_comm_n_scaling",
    )
    parser.add_argument("--n-min", type=int, default=4)
    parser.add_argument("--n-max", type=int, default=2048)
    parser.add_argument("--n-aggregators", type=int, default=N_AGG_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--sqmpc-q-bits", type=int, default=SQMPC_Q_BITS)
    parser.add_argument("--formats", default="pdf,svg,png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = [p.strip().lower() for p in args.formats.split(",") if p.strip()]
    for fmt in formats:
        out = args.output_dir / f"fl_comm_n_scaling_multiclass.{fmt}"
        plot_scaling_combined(out, args.n_min, args.n_max, args.n_aggregators,
                              args.batch_size, args.sqmpc_q_bits)
        print(f"wrote {out}")
        for dataset_id in DATASET_LABELS:
            out = args.output_dir / f"fl_comm_n_scaling_{dataset_id}.{fmt}"
            plot_scaling_per_dataset(
                out, dataset_id, args.n_min, args.n_max,
                args.n_aggregators, args.batch_size, args.sqmpc_q_bits,
            )
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
