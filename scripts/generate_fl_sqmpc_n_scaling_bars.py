#!/usr/bin/env python3
"""Generate FL-SQMPC-only communication scaling bar charts.

Each dataset gets one figure. For each requested client count, the figure shows
two adjacent stacked bars: client and server. Each stacked bar is split into
upload and download bytes per federated round.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / "tmp" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from generate_fl_comm_5method_plot import (  # noqa: E402
    CHANNEL_STYLE,
    DATASET_DIMS,
    DATASET_LABELS,
    FLOAT_BYTES,
    N_AGG_DEFAULT,
    SQMPC_Q_BITS,
    mlp_param_count,
    sqmpc_bits_per_share,
)


N_CLIENT_VALUES_DEFAULT = (2, 8, 32, 128, 512, 2048)


def _format_bytes(value: float) -> str:
    if value <= 0:
        return "0 B"
    units = [("B", 1.0), ("KB", 1024.0), ("MB", 1024.0 ** 2), ("GB", 1024.0 ** 3)]
    label, scale = units[0]
    for candidate_label, candidate_scale in units:
        if value >= candidate_scale:
            label, scale = candidate_label, candidate_scale
    return f"{value / scale:.1f} {label}"


def _bytes_axis_formatter(value: float, _pos: int) -> str:
    if value <= 0:
        return "0"
    units = [("B", 1.0), ("KB", 1024.0), ("MB", 1024.0 ** 2), ("GB", 1024.0 ** 3)]
    for label, scale in reversed(units):
        if value >= scale:
            ratio = value / scale
            return f"{ratio:g} {label}" if ratio < 10 else f"{ratio:.0f} {label}"
    return f"{value:g} B"


_BYTE_TICKS: list[float] = []
for tier in range(4):  # 0=KB, 1=MB, 2=GB, 3=TB
    unit = 1024 ** (tier + 1)
    for mult in (1, 10, 100):
        _BYTE_TICKS.append(float(unit * mult))
_BYTE_TICKS = sorted(set(_BYTE_TICKS))


def _set_log_byte_ticks(ax, ymax: float) -> None:
    visible = [tick for tick in _BYTE_TICKS if 1e3 <= tick <= ymax * 1.1]
    if visible:
        ax.set_yticks(visible)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_bytes_axis_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())


def sqmpc_bytes(
    dataset_id: str,
    n_clients: int,
    n_agg: int,
    sqmpc_q_bits: int,
) -> dict[str, tuple[float, float]]:
    """Return bytes per round as {side: (upload, download)} for FL-SQMPC."""
    dims = DATASET_DIMS[dataset_id]
    p_count = mlp_param_count(dims["d"], dims["C"])
    bits_per_share = sqmpc_bits_per_share(sqmpc_q_bits, n_clients)
    client_upload = n_agg * bits_per_share * p_count / 8.0
    client_download = FLOAT_BYTES * p_count
    return {
        "client": (client_upload, client_download),
        "server": (n_clients * client_upload, n_clients * client_download),
    }


def plot_dataset(
    output_path: Path,
    dataset_id: str,
    n_values: tuple[int, ...],
    n_agg: int,
    sqmpc_q_bits: int,
) -> None:
    x = np.arange(len(n_values), dtype=float)
    width = 0.34
    side_specs = {
        "client": {"offset": -width / 2, "hatch": "", "label": "client"},
        "server": {"offset": width / 2, "hatch": "///", "label": "server"},
    }

    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=140)
    max_total = 0.0
    for side, spec in side_specs.items():
        uploads = []
        downloads = []
        for n_clients in n_values:
            upload, download = sqmpc_bytes(dataset_id, n_clients, n_agg, sqmpc_q_bits)[side]
            uploads.append(upload)
            downloads.append(download)
        uploads_arr = np.array(uploads, dtype=float)
        downloads_arr = np.array(downloads, dtype=float)
        totals = uploads_arr + downloads_arr
        max_total = max(max_total, float(np.max(totals)))
        positions = x + spec["offset"]

        ax.bar(
            positions,
            uploads_arr,
            width=width,
            color=CHANNEL_STYLE["upload"]["color"],
            edgecolor="#334155",
            linewidth=0.5,
            hatch=spec["hatch"],
        )
        ax.bar(
            positions,
            downloads_arr,
            bottom=uploads_arr,
            width=width,
            color=CHANNEL_STYLE["download"]["color"],
            edgecolor="#334155",
            linewidth=0.5,
            hatch=spec["hatch"],
        )
        for xi, total in zip(positions, totals):
            ax.text(
                xi,
                total * 1.10,
                _format_bytes(float(total)),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_title(
        f"{DATASET_LABELS[dataset_id]} - FL-SQMPC communication vs N_clients "
        f"(n_agg={n_agg}, q={sqmpc_q_bits})",
        fontsize=12,
    )
    ax.set_xlabel("N_clients")
    ax.set_ylabel("bytes per round")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yscale("log")
    ymax = max_total * 45
    ax.set_ylim(1e3, ymax)
    _set_log_byte_ticks(ax, ymax)
    ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=CHANNEL_STYLE["upload"]["color"], label="upload"),
        Patch(facecolor=CHANNEL_STYLE["download"]["color"], label="download"),
        Patch(facecolor="white", edgecolor="#334155", label="client"),
        Patch(facecolor="white", edgecolor="#334155", hatch="///", label="server"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=9, frameon=True)
    fig.text(
        0.5,
        0.02,
        "For each N_clients group, the left stacked bar is one client and the right "
        "stacked bar is aggregate server traffic for the round.",
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(
    output_path: Path,
    n_values: tuple[int, ...],
    n_agg: int,
    sqmpc_q_bits: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dataset_id",
                "method",
                "side",
                "n_clients",
                "n_aggregators",
                "sqmpc_q_bits",
                "bits_per_share",
                "upload_bytes",
                "download_bytes",
                "total_bytes",
            ]
        )
        for dataset_id in DATASET_LABELS:
            for n_clients in n_values:
                bits_per_share = sqmpc_bits_per_share(sqmpc_q_bits, n_clients)
                values = sqmpc_bytes(dataset_id, n_clients, n_agg, sqmpc_q_bits)
                for side, (upload, download) in values.items():
                    writer.writerow(
                        [
                            dataset_id,
                            "FL-SQMPC",
                            side,
                            n_clients,
                            n_agg,
                            sqmpc_q_bits,
                            bits_per_share,
                            f"{upload:.6f}",
                            f"{download:.6f}",
                            f"{upload + download:.6f}",
                        ]
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "figures_20260516_fl_sqmpc_n_scaling",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "output" / "tables" / "fl_sqmpc_n_scaling.csv",
    )
    parser.add_argument("--n-values", default="2,8,32,128,512,2048")
    parser.add_argument("--n-aggregators", type=int, default=N_AGG_DEFAULT)
    parser.add_argument("--sqmpc-q-bits", type=int, default=SQMPC_Q_BITS)
    parser.add_argument("--formats", default="pdf,svg,png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = tuple(int(part.strip()) for part in args.n_values.split(",") if part.strip())
    formats = [part.strip().lower() for part in args.formats.split(",") if part.strip()]
    write_csv(args.csv_path, n_values, args.n_aggregators, args.sqmpc_q_bits)
    for dataset_id in DATASET_LABELS:
        for fmt in formats:
            output_path = args.output_dir / f"fl_sqmpc_n_scaling_{dataset_id}.{fmt}"
            plot_dataset(
                output_path,
                dataset_id,
                n_values,
                args.n_aggregators,
                args.sqmpc_q_bits,
            )
            print(f"wrote {output_path}")
    print(f"wrote {args.csv_path}")


if __name__ == "__main__":
    main()
