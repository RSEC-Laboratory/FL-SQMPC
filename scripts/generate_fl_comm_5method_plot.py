#!/usr/bin/env python3
"""Generate per-round communication-cost bar charts for the six methods.

Companion to ``generate_fl_time_5method_plot.py``. Per-round bytes are derived
analytically from each method's protocol (model size, share count, ciphertext
expansion, data-flow size), separately for a single client and for the server
(aggregated over all participating clients). Stacked bars show upload on the
bottom and download on top. The default dataset figures place client and server
bars next to each other for every method.

Sources:
  * Vanilla FL, DP, SMPC, FL-SQMPC: derived analytically from this codebase's
    MLP architecture (solution.py / run_fl_sqmpc_accuracy.py) and SMPC
    constants in sqmpc_core.py.
  * HE (CKKS): Microsoft SEAL default 128-bit-security parameters
    (N_poly=8192, log Q approx 218 bit, ct approx 440 KB, slots=4096).
  * CIDIoT: per-round comm cost is the data-flow term from Yao et al.,
    "Privacy-Preserving Collaborative Intrusion Detection in Edge of IoT"
    (IEEE IoT-J, vol. 11 no. 9, May 2024), Section VI.B and Table VI.
    Paper reports 180 KB / 169 KB per round on CIC_IoT2023 / ToN_IoT and
    derives the cost as O(2 K b |x|), i.e., batch data flows + sample
    gradients (not generator weights). We recompute that formula on our
    processed feature dims so the comparison is apples-to-apples with the
    other methods, and record the generator-weight alternative in the CSV
    for completeness.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


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


DATASET_LABELS = {
    "bot_iot": "BotIoT Multiclass",
    "ciciot2023": "CICIoT2023 Multiclass",
    "ton_iot_extracted": "ToN-IoT Multiclass",
}

# Processed feature dims and class counts, taken from the meta.json files of
# the corresponding processed tensor splits used by the accuracy runs.
DATASET_DIMS = {
    "bot_iot": {"d": 65, "C": 4},
    "ciciot2023": {"d": 39, "C": 8},
    "ton_iot_extracted": {"d": 39, "C": 10},
}

METHOD_ORDER = ("Vanilla FL", "HE", "SMPC", "DP", "CIDIoT", "FL-SQMPC")

# Federation + protocol constants. Mirror the defaults used by the accuracy
# runs and the CIDIoT paper's experimental setup.
N_CLIENTS_DEFAULT = 5
N_AGG_DEFAULT = 2
BATCH_SIZE_DEFAULT = 100  # matches run_fl_sqmpc_accuracy.py and run_cidiot_accuracy.py
FLOAT_BYTES = 4  # float32

# FL-SQMPC: client updates are quantised to ``SQMPC_Q_BITS`` bits/param and then
# additively secret-shared across ``n_agg`` aggregators. For correct sum across
# ``N_clients`` clients, the share ring must satisfy M >= N_clients * 2^q, so
# each share carries ceil(log2(N_clients * 2^q)) bits on the wire.
#
# Default q=2 preserves the previous communication-figure setting. When these
# plots are paired with saturated-regime accuracy, rerun with --sqmpc-q-bits 4
# or higher as documented in PROGRESS.md.
SQMPC_Q_BITS = 2

# HE (CKKS, N_poly=8192, log Q approx 218 bit): one ciphertext is approximately
# 440 KB and packs N_poly/2 = 4096 slots. Standard 128-bit-security parameters
# (Microsoft SEAL default).
HE_SLOTS = 4096
HE_CT_BYTES = 440 * 1024

# CIDIoT conditional TCN generator hyperparameters (defaults in
# scripts/run_cidiot_accuracy.py). Used only for the "generator-weights"
# alternative recorded in the CSV.
CIDIOT_LATENT_DIM = 100
CIDIOT_LABEL_DIM = 16
CIDIOT_TCN_CHANNELS = 32
CIDIOT_TCN_KERNEL = 3

# CIDIoT paper Table VI (Yao et al., IEEE IoT-J 2024). Whole-network per-round
# communication in bytes. Paper does not cover Bot-IoT.
CIDIOT_PAPER_TABLE_VI_BYTES = {
    "ciciot2023": 180 * 1024,
    "ton_iot_extracted": 169 * 1024,
}

CHANNEL_STYLE = {
    "upload": {"label": "upload", "color": "#1f77b4"},
    "download": {"label": "download", "color": "#ff7f0e"},
}


@dataclass(frozen=True)
class CommRecord:
    dataset_id: str
    method: str
    side: str  # "client" or "server"
    upload_bytes: float
    download_bytes: float
    formula: str
    source: str
    notes: str = ""


@dataclass(frozen=True)
class MethodNote:
    """Static per-method metadata for the figure caption / methodology file."""

    name: str
    client_formula: str
    server_formula: str
    source: str
    notes: str = ""


def mlp_param_count(d: int, C: int) -> int:
    """Total params for the [d -> 128 -> 64 -> C] MLP with biases."""
    return d * 128 + 128 + 128 * 64 + 64 + 64 * C + C


def cidiot_generator_param_count(d: int, C: int) -> int:
    """Total params for the ConditionalTCNGenerator with default hyperparams.

    Layout (see scripts/run_cidiot_accuracy.py):
      * Embedding(C, label_dim)
      * Linear(latent+label, channels*d)
      * 2 x TCNBlock(channels, channels, k=3): each block has 2 Conv1d(c, c, k=3),
        2 GroupNorm(1, c), and (since in==out) an Identity shortcut.
      * Conv1d(channels, 1, k=1)
    """
    label = C * CIDIOT_LABEL_DIM
    fc = (CIDIOT_LATENT_DIM + CIDIOT_LABEL_DIM) * (CIDIOT_TCN_CHANNELS * d) + (
        CIDIOT_TCN_CHANNELS * d
    )
    conv_per_block = (
        CIDIOT_TCN_CHANNELS * CIDIOT_TCN_CHANNELS * CIDIOT_TCN_KERNEL
        + CIDIOT_TCN_CHANNELS
    )
    gn_per_block = 2 * CIDIOT_TCN_CHANNELS
    block_params = 2 * conv_per_block + 2 * gn_per_block
    tcn = 2 * block_params
    out = CIDIOT_TCN_CHANNELS * 1 * 1 + 1
    return label + fc + tcn + out


def he_ciphertext_count(num_params: int) -> int:
    return math.ceil(num_params / HE_SLOTS)


def sqmpc_bits_per_share(q_bits: int, n_clients: int) -> int:
    """Wire width of one additive secret share at q-bit quantisation.

    The aggregating ring Z_M must satisfy M >= n_clients * 2^q so that the
    sum-of-shares recovers the correct integer without modular wrap.
    """
    return max(1, math.ceil(math.log2(n_clients * (2 ** q_bits))))


def build_records(
    n_clients: int, n_agg: int, batch_size: int, sqmpc_q_bits: int
) -> tuple[list[CommRecord], list[CommRecord]]:
    """Return (primary_records, alt_records).

    ``primary_records`` are the bars that appear in the figure. ``alt_records``
    are written to the CSV but not plotted - currently the CIDIoT
    generator-weight variant.
    """

    primary: list[CommRecord] = []
    alternates: list[CommRecord] = []

    bits_per_share = sqmpc_bits_per_share(sqmpc_q_bits, n_clients)
    sqmpc_bytes_per_param = n_agg * bits_per_share / 8.0

    for dataset_id, dims in DATASET_DIMS.items():
        d = dims["d"]
        C = dims["C"]
        P = mlp_param_count(d, C)
        G = cidiot_generator_param_count(d, C)
        n_ct = he_ciphertext_count(P)
        # CIDIoT data-flow size per sample (paper's |x|).
        flow_bytes = d * FLOAT_BYTES
        # Per-edge per-round = b * |x| each direction.
        cidiot_per_edge = batch_size * flow_bytes

        client_costs: dict[str, tuple[float, float, str, str, str]] = {
            "Vanilla FL": (
                FLOAT_BYTES * P,
                FLOAT_BYTES * P,
                f"4 * P  (P = {P:,} params)",
                "analytical (this work)",
                "FedAvg with float32 model weights.",
            ),
            "DP": (
                FLOAT_BYTES * P,
                FLOAT_BYTES * P,
                f"4 * P  (P = {P:,} params)",
                "analytical (this work)",
                "Same wire format as Vanilla FL; DP noise is added before transmission.",
            ),
            "SMPC": (
                n_agg * FLOAT_BYTES * P,
                FLOAT_BYTES * P,
                f"n_agg * 4 * P  (n_agg = {n_agg})",
                "analytical (this work); sqmpc_core.py constants",
                "Additive secret-sharing: each client sends a full-width share to each aggregator.",
            ),
            "FL-SQMPC": (
                n_agg * bits_per_share * P / 8.0,
                FLOAT_BYTES * P,
                (
                    f"n_agg * ceil(log2(N_clients * 2^q)) / 8 * P  "
                    f"(q={sqmpc_q_bits} bits/param, share={bits_per_share} bits)"
                ),
                "analytical (this work)",
                (
                    f"Quantise weights to {sqmpc_q_bits} bits/layer (all 3 layers), then "
                    f"additively secret-share across {n_agg} aggregators. Share width "
                    f"is ceil(log2({n_clients} * 2^{sqmpc_q_bits})) = {bits_per_share} bits "
                    f"so the ring covers correct aggregation across {n_clients} clients."
                ),
            ),
            "HE": (
                n_ct * HE_CT_BYTES,
                n_ct * HE_CT_BYTES,
                f"ceil(P / {HE_SLOTS}) * {HE_CT_BYTES // 1024} KB  (n_ct = {n_ct})",
                "analytical (this work); SEAL CKKS N=8192, log Q=218",
                "CKKS ciphertext approx 440 KB packs 4096 slots at 128-bit security.",
            ),
            "CIDIoT": (
                cidiot_per_edge,
                cidiot_per_edge,
                f"b * |x|  (b={batch_size}, |x| = 4*d = {flow_bytes} B)",
                "Yao et al. IoT-J 2024 Sec. VI.B, recomputed on our feature dims",
                f"Paper formula 2*K*b*|x|; paper reports 180/169 KB on CIC/ToN. "
                f"Generator-weight alternative (|G|={G:,} params): {4*G/1024:.1f} KB up + same down.",
            ),
        }

        for method in METHOD_ORDER:
            up, dn, formula, source, notes = client_costs[method]
            primary.append(
                CommRecord(
                    dataset_id,
                    method,
                    "client",
                    float(up),
                    float(dn),
                    formula,
                    source,
                    notes,
                )
            )
            # Server-side bytes per round = aggregated over all clients.
            # In <- sum of client uploads. Out -> one copy broadcast per client.
            primary.append(
                CommRecord(
                    dataset_id,
                    method,
                    "server",
                    float(n_clients * up),
                    float(n_clients * dn),
                    formula + f"  (x N_clients = {n_clients})",
                    source,
                    notes,
                )
            )

        # CIDIoT generator-weight alternative (recorded in CSV only).
        alternates.append(
            CommRecord(
                dataset_id,
                "CIDIoT (gen weights)",
                "client",
                float(FLOAT_BYTES * G),
                float(FLOAT_BYTES * G),
                f"4 * |G|  (|G| = {G:,} params)",
                "analytical (this work); if generator weights were federated",
                "Not used by CIDIoT-as-published. Listed here only to bound the alternative protocol.",
            )
        )
        alternates.append(
            CommRecord(
                dataset_id,
                "CIDIoT (gen weights)",
                "server",
                float(n_clients * FLOAT_BYTES * G),
                float(n_clients * FLOAT_BYTES * G),
                f"4 * |G| * N_clients",
                "analytical (this work)",
                "",
            )
        )

        # CIDIoT paper-reported value (whole-network per round) recorded as a
        # cross-check. Bot-IoT is not covered by the paper.
        if dataset_id in CIDIOT_PAPER_TABLE_VI_BYTES:
            paper_total = CIDIOT_PAPER_TABLE_VI_BYTES[dataset_id]
            alternates.append(
                CommRecord(
                    dataset_id,
                    "CIDIoT (paper Table VI)",
                    "server",
                    float(paper_total) / 2.0,
                    float(paper_total) / 2.0,
                    "as reported (whole-network per round)",
                    "Yao et al. IoT-J 2024, Table VI",
                    "Paper does not split upload/download; we report half each.",
                )
            )

    return primary, alternates


def method_notes(sqmpc_q_bits: int, n_clients: int, n_agg: int) -> list[MethodNote]:
    bits_per_share = sqmpc_bits_per_share(sqmpc_q_bits, n_clients)
    return [
        MethodNote(
            "Vanilla FL",
            "4 * P",
            "N_clients * 4 * P (both directions)",
            "analytical",
            "FedAvg with float32 model weights.",
        ),
        MethodNote(
            "HE",
            "ceil(P / slots) * ct_size",
            "N_clients * ceil(P / slots) * ct_size",
            "analytical; CKKS N_poly=8192, ct approx 440 KB, slots=4096",
            "Microsoft SEAL default 128-bit security parameters.",
        ),
        MethodNote(
            "SMPC",
            "n_agg * 4 * P (up), 4 * P (dn)",
            "N_clients * n_agg * 4 * P (in), N_clients * 4 * P (out)",
            f"analytical; n_agg={n_agg}",
            "Full-width additive secret shares to each aggregator.",
        ),
        MethodNote(
            "DP",
            "4 * P",
            "same as Vanilla FL",
            "analytical",
            "DP noise added pre-transmission; wire format unchanged.",
        ),
        MethodNote(
            "CIDIoT",
            "b * |x|",
            "N_clients * b * |x|",
            "Yao et al. IoT-J 2024, Sec. VI.B & Table VI (recomputed on our feature dims)",
            "Paper-reported whole-network 180/169 KB on CIC_IoT2023/ToN_IoT; "
            "transmits batch data flows + sample gradients, not generator weights.",
        ),
        MethodNote(
            "FL-SQMPC",
            f"n_agg * ceil(log2(N * 2^q)) / 8 * P (up), 4 * P (dn)  [q={sqmpc_q_bits}, share={bits_per_share} bits]",
            f"N_clients * n_agg * ceil(log2(N * 2^q)) / 8 * P (in), N_clients * 4 * P (out)",
            f"analytical; q={sqmpc_q_bits} bits/param across all 3 layers, additive secret-sharing across {n_agg} aggregators",
            (
                "Quantised weights are additively secret-shared. The share ring must "
                "satisfy M >= N_clients * 2^q so the sum of shares recovers the correct "
                "aggregate without modular wrap; this sets the wire width per share."
            ),
        ),
    ]


def write_csv(
    primary: Iterable[CommRecord],
    alternates: Iterable[CommRecord],
    path: Path,
    n_clients: int,
    n_agg: int,
    batch_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dataset_id",
                "method",
                "side",
                "upload_bytes",
                "download_bytes",
                "total_bytes",
                "formula",
                "source",
                "notes",
                "n_clients",
                "n_aggregators",
                "batch_size",
                "is_primary_bar",
            ]
        )
        for record in primary:
            writer.writerow(
                [
                    record.dataset_id,
                    record.method,
                    record.side,
                    f"{record.upload_bytes:.6f}",
                    f"{record.download_bytes:.6f}",
                    f"{record.upload_bytes + record.download_bytes:.6f}",
                    record.formula,
                    record.source,
                    record.notes,
                    n_clients,
                    n_agg,
                    batch_size,
                    1,
                ]
            )
        for record in alternates:
            writer.writerow(
                [
                    record.dataset_id,
                    record.method,
                    record.side,
                    f"{record.upload_bytes:.6f}",
                    f"{record.download_bytes:.6f}",
                    f"{record.upload_bytes + record.download_bytes:.6f}",
                    record.formula,
                    record.source,
                    record.notes,
                    n_clients,
                    n_agg,
                    batch_size,
                    0,
                ]
            )


def _format_bytes(value: float) -> str:
    if value <= 0:
        return "0 B"
    units = [("B", 1.0), ("KB", 1024.0), ("MB", 1024.0 ** 2), ("GB", 1024.0 ** 3)]
    chosen_label, chosen_scale = units[0]
    for label, scale in units:
        if value >= scale:
            chosen_label, chosen_scale = label, scale
    return f"{value / chosen_scale:.1f} {chosen_label}"


def _bytes_axis_formatter(value: float, _pos: int) -> str:
    if value <= 0:
        return "0"
    units = [("B", 1.0), ("KB", 1024.0), ("MB", 1024.0 ** 2), ("GB", 1024.0 ** 3)]
    for label, scale in reversed(units):
        if value >= scale:
            return f"{value / scale:g} {label}"
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


def _plot_panel(
    ax,
    records: list[CommRecord],
    dataset_id: str,
    side: str,
    yscale: str,
    panel_label: str | None,
) -> None:
    by_method = {
        record.method: record
        for record in records
        if record.dataset_id == dataset_id and record.side == side
    }
    x = np.arange(len(METHOD_ORDER))
    uploads = np.array([by_method[m].upload_bytes for m in METHOD_ORDER], dtype=float)
    downloads = np.array([by_method[m].download_bytes for m in METHOD_ORDER], dtype=float)
    totals = uploads + downloads

    ax.bar(
        x,
        uploads,
        color=CHANNEL_STYLE["upload"]["color"],
        label=CHANNEL_STYLE["upload"]["label"],
        linewidth=0,
        width=0.6,
    )
    ax.bar(
        x,
        downloads,
        bottom=uploads,
        color=CHANNEL_STYLE["download"]["color"],
        label=CHANNEL_STYLE["download"]["label"],
        linewidth=0,
        width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, fontsize=12)
    ax.set_ylabel("bytes per round")
    ax.set_title(f"{DATASET_LABELS[dataset_id]} - {side}", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_bytes_axis_formatter))

    if yscale == "log":
        ax.set_yscale("log")
        ymax = max(totals) * 25
        ax.set_ylim(1e3, ymax)
        _set_log_byte_ticks(ax, ymax)
    else:
        ax.set_ylim(0.0, max(totals) * 1.35)

    # Per-bar value annotations above each stacked bar.
    for xi, total in zip(x, totals):
        if total <= 0:
            continue
        ax.text(
            xi,
            total * (1.05 if yscale == "log" else 1.02),
            _format_bytes(total),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Legend in a corner without bars. For log we have lots of headroom because
    # the y-axis extends well above max(totals); for linear we put it upper-right
    # where Vanilla FL and DP leave space.
    ax.legend(loc="upper right", fontsize=12, frameon=True)

    if panel_label:
        ax.text(
            0.5,
            -0.30,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
        )


def _plot_grouped_side_bars(
    ax,
    records: list[CommRecord],
    dataset_id: str,
    yscale: str,
    n_clients: int,
    n_agg: int,
    sqmpc_q_bits: int,
) -> None:
    by_key = {
        (record.method, record.side): record
        for record in records
        if record.dataset_id == dataset_id
    }
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.34
    side_specs = {
        "client": {"offset": -width / 2, "hatch": "", "label": "client"},
        "server": {"offset": width / 2, "hatch": "///", "label": "server"},
    }
    max_total = 0.0

    for side, spec in side_specs.items():
        positions = x + spec["offset"]
        uploads = np.array(
            [by_key[(method, side)].upload_bytes for method in METHOD_ORDER],
            dtype=float,
        )
        downloads = np.array(
            [by_key[(method, side)].download_bytes for method in METHOD_ORDER],
            dtype=float,
        )
        totals = uploads + downloads
        max_total = max(max_total, float(np.max(totals)))
        ax.bar(
            positions,
            uploads,
            width=width,
            color=CHANNEL_STYLE["upload"]["color"],
            edgecolor="#334155",
            linewidth=0.5,
            hatch=spec["hatch"],
        )
        ax.bar(
            positions,
            downloads,
            bottom=uploads,
            width=width,
            color=CHANNEL_STYLE["download"]["color"],
            edgecolor="#334155",
            linewidth=0.5,
            hatch=spec["hatch"],
        )
        for xi, total in zip(positions, totals):
            if total <= 0:
                continue
            ax.text(
                xi,
                total * (1.10 if yscale == "log" else 1.02),
                _format_bytes(float(total)),
                ha="center",
                va="bottom",
                fontsize=14,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, fontsize=16)
    ax.set_ylabel("bytes per round", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_bytes_axis_formatter))
    if yscale == "log":
        ax.set_yscale("log")
        ymax = max_total * 45
        ax.set_ylim(1e3, ymax)
        _set_log_byte_ticks(ax, ymax)
    else:
        ax.set_ylim(0.0, max_total * 1.45)

    handles = [
        Patch(facecolor=CHANNEL_STYLE["upload"]["color"], label="upload"),
        Patch(facecolor=CHANNEL_STYLE["download"]["color"], label="download"),
        Patch(facecolor="white", edgecolor="#334155", label="client"),
        Patch(facecolor="white", edgecolor="#334155", hatch="///", label="server"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=14, frameon=True)


def _figure_footnote(
    fig,
    alternates: list[CommRecord],
    dataset_ids: list[str],
    n_clients: int,
    batch_size: int,
) -> None:
    """Bottom-of-figure caption explaining the CIDIoT bar and listing the
    paper-reported / alternative numbers for the datasets shown."""

    parts: list[str] = []
    parts.append(
        "CIDIoT bar uses the data-flow formula 2*K*b*|x| from Yao et al. 2024 "
        f"(IEEE IoT-J, Table VI) recomputed on our processed feature dims (b={batch_size}, K={n_clients})."
    )
    paper_parts: list[str] = []
    alt_parts: list[str] = []
    for dataset_id in dataset_ids:
        paper = _cidiot_paper_total(alternates, dataset_id)
        if paper is not None:
            paper_parts.append(
                f"{DATASET_LABELS[dataset_id]}: {_format_bytes(paper)}"
            )
        alt = _cidiot_alt_total(alternates, dataset_id, "server")
        if alt is not None:
            alt_parts.append(
                f"{DATASET_LABELS[dataset_id]}: {_format_bytes(alt)} (server)"
            )
    if paper_parts:
        parts.append("Paper-reported whole-network/round - " + "; ".join(paper_parts) + ".")
    if alt_parts:
        parts.append(
            "Generator-weight alternative (not CIDIoT-as-published) - "
            + "; ".join(alt_parts)
            + "."
        )
    fig.text(
        0.5,
        0.02,
        "  ".join(parts),
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        wrap=True,
    )


def _cidiot_alt_total(
    alternates: list[CommRecord], dataset_id: str, side: str
) -> float | None:
    for record in alternates:
        if (
            record.method == "CIDIoT (gen weights)"
            and record.dataset_id == dataset_id
            and record.side == side
        ):
            return record.upload_bytes + record.download_bytes
    return None


def _cidiot_paper_total(
    alternates: list[CommRecord], dataset_id: str
) -> float | None:
    for record in alternates:
        if (
            record.method == "CIDIoT (paper Table VI)"
            and record.dataset_id == dataset_id
        ):
            return record.upload_bytes + record.download_bytes
    return None


def plot_dataset(
    primary: list[CommRecord],
    alternates: list[CommRecord],
    dataset_id: str,
    output_path: Path,
    yscale: str,
    n_clients: int,
    n_agg: int,
    batch_size: int,
    sqmpc_q_bits: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=140)
    _plot_grouped_side_bars(
        ax, primary, dataset_id, yscale, n_clients, n_agg, sqmpc_q_bits
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    # _figure_footnote(fig, alternates, [dataset_id], n_clients, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_combined(
    primary: list[CommRecord],
    alternates: list[CommRecord],
    output_path: Path,
    yscale: str,
    n_clients: int,
    batch_size: int,
) -> None:
    dataset_ids = list(DATASET_LABELS)
    fig, axes = plt.subplots(
        len(dataset_ids), 2, figsize=(13.5, 5.4 * len(dataset_ids))
    )
    axes = np.atleast_2d(axes)
    for row_idx, dataset_id in enumerate(dataset_ids):
        for col_idx, side in enumerate(("client", "server")):
            panel_label = f"({chr(ord('a') + 2 * row_idx + col_idx)}) {side}"
            _plot_panel(
                axes[row_idx, col_idx],
                primary,
                dataset_id,
                side,
                yscale,
                panel_label,
            )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    # _figure_footnote(fig, alternates, dataset_ids, n_clients, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_methodology(
    output_path: Path,
    n_clients: int,
    n_agg: int,
    batch_size: int,
    sqmpc_q_bits: int,
    primary: list[CommRecord],
    alternates: list[CommRecord],
) -> None:
    """Write a markdown report describing how each bar was computed."""

    lines: list[str] = []
    lines.append("# FL-SQMPC per-round communication cost - methodology\n")
    lines.append(
        "Companion to ``fl_comm_6methods_*`` figures. Every bar in those figures "
        "is an analytical estimate of bytes exchanged in **one** federated round. "
        "Rounds-to-completion (50/100/150) are deliberately omitted so the per-round "
        "structure of each protocol is exposed.\n"
    )
    lines.append("## Federation parameters\n")
    lines.append(f"- N_clients = **{n_clients}** participating edges per round")
    lines.append(f"- n_agg = **{n_agg}** SMPC aggregators (``sqmpc_core.SMPC_NUM_AGGREGATORS``)")
    lines.append(f"- batch size b = **{batch_size}** (matches ``run_fl_sqmpc_accuracy.py`` / ``run_cidiot_accuracy.py`` defaults)")
    lines.append(f"- float32 wire format = {FLOAT_BYTES} bytes per scalar")
    lines.append(
        f"- FL-SQMPC quantisation = **{sqmpc_q_bits} bits/param across all 3 layers**, "
        f"share width = ceil(log2({n_clients} * 2^{sqmpc_q_bits})) = "
        f"**{sqmpc_bits_per_share(sqmpc_q_bits, n_clients)} bits/share**\n"
    )

    lines.append("## Model and dataset dimensions\n")
    lines.append("All non-CIDIoT methods aggregate the same MLP ``[d -> 128 -> 64 -> C]`` with biases.\n")
    lines.append("| Dataset | d (features) | C (classes) | P (params) | |G| (CIDIoT generator) |")
    lines.append("|---|---|---|---|---|")
    for dataset_id, dims in DATASET_DIMS.items():
        d = dims["d"]
        C = dims["C"]
        P = mlp_param_count(d, C)
        G = cidiot_generator_param_count(d, C)
        lines.append(
            f"| {DATASET_LABELS[dataset_id]} | {d} | {C} | {P:,} | {G:,} |"
        )
    lines.append("")

    lines.append("## Per-method formulas\n")
    for note in method_notes(sqmpc_q_bits, n_clients, n_agg):
        lines.append(f"### {note.name}\n")
        lines.append(f"- **Client per round:** ``{note.client_formula}``")
        lines.append(f"- **Server per round:** ``{note.server_formula}``")
        lines.append(f"- **Source:** {note.source}")
        if note.notes:
            lines.append(f"- **Notes:** {note.notes}")
        lines.append("")

    lines.append("## CIDIoT in detail\n")
    lines.append(
        "Yao et al. (IEEE IoT-J vol. 11 no. 9, May 2024) state in Section VI.B that "
        "the per-round communication cost of CIDIoT is ``O(2 K b |x| + K l_ss)``, "
        "where ``K`` is the number of edges, ``b`` the batch size, ``|x|`` the size "
        "of one data flow (one input sample at float32 width), and ``l_ss`` the "
        "secret-share overhead (negligible vs. ``|x|``). Their Table VI reports "
        "whole-network per-round comm:\n"
    )
    lines.append("- CIC_IoT2023: **180 KB / round**")
    lines.append("- ToN_IoT: **169 KB / round**")
    lines.append(
        f"\nWith their setup (K=5, b=100, |x|=47x4 B for CIC_IoT2023) the formula gives "
        f"2 x 5 x 100 x 188 B = 188 KB, matching the reported 180 KB within rounding. "
        f"We recompute this formula on **our** processed feature dims (d=65/39/39 for "
        f"Bot-IoT/CIC/ToN) so CIDIoT is compared like-for-like with the other methods. "
        f"Bot-IoT is not covered by the paper; we extrapolate via 2 K b |x|.\n"
    )
    lines.append(
        "**Generator-weight alternative.** If CIDIoT were implemented by federating "
        "the conditional TCN generator weights instead, the per-client cost would be "
        "``4 |G|`` each way. We record that variant in the CSV (``method = "
        "'CIDIoT (gen weights)'``) for completeness, but it is **not** what the "
        "paper does and is **not** the primary CIDIoT bar.\n"
    )

    lines.append("## FL-SQMPC in detail\n")
    bps = sqmpc_bits_per_share(sqmpc_q_bits, n_clients)
    lines.append(
        f"Each client quantises its weight update to **q = {sqmpc_q_bits} bits/param** "
        f"on all three Linear layers (signed symmetric quantisation, see "
        f"``sqmpc_core._quantize_dequantize_symmetric``). The quantised integer "
        f"value is then **additively secret-shared** across n_agg = {n_agg} "
        f"aggregators: shares ``r_1, ..., r_{{n_agg}}`` are drawn from a ring "
        f"``Z_M`` with ``sum_i r_i = quantised_value (mod M)``.\n"
    )
    lines.append(
        f"For the aggregator to recover the correct sum across N_clients = "
        f"{n_clients} clients without modular wrap, the ring must satisfy "
        f"``M >= N_clients * 2^q`` = ``{n_clients} * 2^{sqmpc_q_bits}`` = "
        f"``{n_clients * (2 ** sqmpc_q_bits)}``. Hence each share occupies "
        f"``ceil(log2({n_clients} * 2^{sqmpc_q_bits})) = {bps} bits`` on the wire, "
        f"and the **per-client upload** is ``n_agg * {bps} * P / 8`` bytes per "
        f"round - i.e. ``{n_agg * bps / 8:.3f} * P`` bytes/round.\n"
    )
    lines.append(
        "The download (server -> client broadcast of the aggregated model) is "
        "left at full float32 precision, ``4 * P`` bytes. Compressing the "
        "broadcast would require an additional protocol step (e.g. quantised "
        "broadcast) and is out of scope for this figure.\n"
    )
    lines.append(
        f"**Why the share width depends on N_clients.** Naively one might say "
        f"\"q={sqmpc_q_bits} bits times n_agg={n_agg} shares = {sqmpc_q_bits * n_agg} "
        f"bits/param.\" That under-counts the secret-sharing overhead: each "
        f"individual share must be uniformly distributed over a ring large "
        f"enough that the **aggregate** of all N_clients shares does not wrap. "
        f"Increasing N_clients increases the wire width per share "
        f"logarithmically.\n"
    )
    lines.append(
        f"**Accuracy at q={sqmpc_q_bits}** is characterised in "
        f"``output/figures_20260512_sqmpc_bits_ablation/`` - the precision drop "
        f"vs full-float aggregation is small (the figure's narrative argument).\n"
    )

    lines.append("## Per-dataset bar values\n")
    by_key = {(r.dataset_id, r.method, r.side): r for r in primary}
    for dataset_id in DATASET_DIMS:
        lines.append(f"### {DATASET_LABELS[dataset_id]}\n")
        for side in ("client", "server"):
            lines.append(f"**{side} per round:**\n")
            lines.append("| Method | upload | download | total |")
            lines.append("|---|---|---|---|")
            for method in METHOD_ORDER:
                r = by_key[(dataset_id, method, side)]
                lines.append(
                    f"| {method} | {_format_bytes(r.upload_bytes)} | "
                    f"{_format_bytes(r.download_bytes)} | "
                    f"{_format_bytes(r.upload_bytes + r.download_bytes)} |"
                )
            lines.append("")
        # Alternates
        alt_lines = []
        for r in alternates:
            if r.dataset_id != dataset_id:
                continue
            alt_lines.append(
                f"- {r.method} ({r.side}): {_format_bytes(r.upload_bytes + r.download_bytes)} - {r.source}"
            )
        if alt_lines:
            lines.append("**Recorded alternatives (not plotted):**\n")
            lines.extend(alt_lines)
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "figures_20260516_fl_comm_6methods",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "output" / "tables" / "fl_comm_6methods_multiclass.csv",
    )
    parser.add_argument(
        "--methodology-path",
        type=Path,
        default=None,
        help="Where to write the per-method methodology markdown. "
        "Defaults to <output-dir>/fl_comm_6methods_methodology.md.",
    )
    parser.add_argument("--yscale", choices=["log", "linear", "both"], default="log")
    parser.add_argument(
        "--layout",
        choices=["split", "combined", "both"],
        default="split",
        help="split = one file per dataset; combined = single multi-panel figure.",
    )
    parser.add_argument("--formats", default="pdf,svg,png")
    parser.add_argument("--n-clients", type=int, default=N_CLIENTS_DEFAULT)
    parser.add_argument("--n-aggregators", type=int, default=N_AGG_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument(
        "--sqmpc-q-bits",
        type=int,
        default=SQMPC_Q_BITS,
        help="Per-parameter quantisation width for FL-SQMPC (applied to all 3 layers).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary, alternates = build_records(
        args.n_clients, args.n_aggregators, args.batch_size, args.sqmpc_q_bits
    )
    write_csv(
        primary, alternates, args.csv_path, args.n_clients, args.n_aggregators, args.batch_size
    )

    methodology_path = args.methodology_path or (
        args.output_dir / "fl_comm_6methods_methodology.md"
    )
    write_methodology(
        methodology_path,
        args.n_clients,
        args.n_aggregators,
        args.batch_size,
        args.sqmpc_q_bits,
        primary,
        alternates,
    )

    yscales = ("log", "linear") if args.yscale == "both" else (args.yscale,)
    formats = [part.strip().lower() for part in args.formats.split(",") if part.strip()]
    for yscale in yscales:
        for fmt in formats:
            if args.layout in {"combined", "both"}:
                output_path = (
                    args.output_dir / f"fl_comm_6methods_multiclass_{yscale}.{fmt}"
                )
                plot_combined(
                    primary, alternates, output_path, yscale, args.n_clients, args.batch_size
                )
                print(f"wrote {output_path}")
            if args.layout in {"split", "both"}:
                for dataset_id in DATASET_LABELS:
                    output_path = (
                        args.output_dir
                        / f"fl_comm_6methods_{dataset_id}_multiclass_{yscale}.{fmt}"
                    )
                    plot_dataset(
                        primary,
                        alternates,
                        dataset_id,
                        output_path,
                        yscale,
                        args.n_clients,
                        args.n_aggregators,
                        args.batch_size,
                        args.sqmpc_q_bits,
                    )
                    print(f"wrote {output_path}")
    print(f"wrote {args.csv_path}")
    print(f"wrote {methodology_path}")


if __name__ == "__main__":
    main()
