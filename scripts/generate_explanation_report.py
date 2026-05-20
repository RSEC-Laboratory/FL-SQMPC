#!/usr/bin/env python3
"""Generate a standalone HTML report explaining the FL-SQMPC vs CIDIoT comparison."""

from __future__ import annotations

import csv
import html
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "reports"
REPORT_PATH = OUTPUT_DIR / "fl_sqmpc_cidiot_comparison_report.html"
RESULTS_DIR = ROOT / "output" / "results"
TABLES_DIR = ROOT / "output" / "tables"
DATASET_PATH = ROOT / "ciciot2023_processed" / "CicIoT_extracted02.csv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], *, compact: bool = False) -> str:
    cls = "table compact" if compact else "table"
    thead = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            text = fmt(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class=\"{cls}\"><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dataset_summary() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_csv(DATASET_PATH).replace([float("inf"), -float("inf")], pd.NA).dropna(axis=0)
    class_rows = [
        {"class": label, "count": int(count)}
        for label, count in df["category"].value_counts().items()
    ]
    features = [column for column in df.columns if column not in {"Label", "category"}]
    return class_rows, {"rows": int(len(df)), "feature_count": len(features), "features": features}


def result_paths(method_prefix: str, partition: str) -> list[Path]:
    return sorted(RESULTS_DIR.glob(f"{method_prefix}_multiclass_{partition}_c5_r30_full_seed*.json"))


def config_from_result(method_prefix: str, partition: str) -> dict[str, Any]:
    return read_json(result_paths(method_prefix, partition)[0])["config"]


def final_metric_rows() -> list[dict[str, Any]]:
    rows = csv_rows(TABLES_DIR / "accuracy_comparison.csv")
    return [
        {
            "method": row["method_key"],
            "partition": row["partition"],
            "seed": row["seed"],
            "accuracy": float(row["accuracy"]),
            "precision_macro": float(row["precision_macro"]),
            "recall_macro": float(row["recall_macro"]),
            "f1_macro": float(row["f1_macro"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
        }
        for row in rows
        if row["method_key"] in {"sqmpc", "cidiot_implemented"}
    ]


def summary_rows() -> list[dict[str, Any]]:
    return [
        {
            "partition": row["partition"],
            "seed_count": int(row["seed_count"]),
            "seeds": row["seeds"],
            "same_setup": row["all_same_training_setup"],
            "fl_mean": float(row["fl_sqmpc_accuracy_mean"]),
            "fl_std": float(row["fl_sqmpc_accuracy_std"]),
            "cid_mean": float(row["cidiot_implemented_accuracy_mean"]),
            "cid_std": float(row["cidiot_implemented_accuracy_std"]),
            "diff_mean": float(row["fl_sqmpc_minus_cidiot_implemented_mean"]),
            "diff_std": float(row["fl_sqmpc_minus_cidiot_implemented_std"]),
            "paper_cid": float(row["cidiot_paper_accuracy_reference"]),
        }
        for row in csv_rows(TABLES_DIR / "accuracy_summary.csv")
    ]


def pairwise_rows() -> list[dict[str, Any]]:
    return [
        {
            "partition": row["partition"],
            "seed": row["seed"],
            "fl": float(row["fl_sqmpc_accuracy"]),
            "cid": float(row["cidiot_implemented_accuracy"]),
            "diff": float(row["fl_sqmpc_minus_cidiot_implemented"]),
            "same_setup": row["same_training_setup"],
        }
        for row in csv_rows(TABLES_DIR / "accuracy_pairwise.csv")
    ]


def section(title: str, body: str, id_: str | None = None) -> str:
    attr = f" id=\"{id_}\"" if id_ else ""
    return f"<section{attr}><h2>{html.escape(title)}</h2>{body}</section>"


def metric_card(label: str, value: str, note: str = "") -> str:
    return (
        "<div class=\"metric-card\">"
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<em>{html.escape(note)}</em>"
        "</div>"
    )


def build_report() -> str:
    class_rows, dataset_info = dataset_summary()
    sqmpc_cfg = config_from_result("sqmpc", "iid")
    cidiot_cfg = config_from_result("cidiot", "iid")
    sample_result = read_json(result_paths("sqmpc", "iid")[0])
    audit = sample_result["training_audit"]
    summary = summary_rows()
    pairwise = pairwise_rows()
    metric_rows = final_metric_rows()

    key_metrics = "".join(
        [
            metric_card("Dataset rows", f"{dataset_info['rows']:,}", "after cleaning"),
            metric_card("Features", str(dataset_info["feature_count"]), "numeric CICIoT2023 columns"),
            metric_card("Train/test", f"{audit['train_rows']:,} / {audit['test_rows']:,}", "70/30 stratified split"),
            metric_card("Seeds", "42, 43, 44", "three independent matched runs"),
            metric_card("Clients", str(sqmpc_cfg["clients"]), "same client partitions per seed"),
            metric_card("Rounds", str(sqmpc_cfg["rounds"]), "synchronous training"),
        ]
    )

    summary_table = html_table(
        summary,
        [
            ("partition", "Partition"),
            ("seed_count", "Seeds"),
            ("fl_mean", "FL-SQMPC mean"),
            ("fl_std", "FL-SQMPC std"),
            ("cid_mean", "CIDIoT mean"),
            ("cid_std", "CIDIoT std"),
            ("diff_mean", "Difference mean"),
            ("diff_std", "Difference std"),
            ("paper_cid", "CIDIoT paper ref."),
            ("same_setup", "Same setup"),
        ],
    )

    pairwise_table = html_table(
        pairwise,
        [
            ("partition", "Partition"),
            ("seed", "Seed"),
            ("fl", "FL-SQMPC"),
            ("cid", "CIDIoT"),
            ("diff", "Difference"),
            ("same_setup", "Same setup"),
        ],
        compact=True,
    )

    metrics_table = html_table(
        metric_rows,
        [
            ("method", "Method"),
            ("partition", "Partition"),
            ("seed", "Seed"),
            ("accuracy", "Accuracy"),
            ("precision_macro", "Precision"),
            ("recall_macro", "Recall"),
            ("f1_macro", "F1"),
            ("balanced_accuracy", "Balanced acc."),
        ],
        compact=True,
    )

    class_table = html_table(class_rows, [("class", "Class"), ("count", "Rows")], compact=True)

    sqmpc_hparams = [
        {"name": "Model", "value": f"MLP hidden sizes {sqmpc_cfg['hidden_sizes']}"},
        {"name": "Optimizer", "value": sqmpc_cfg["optimizer"]},
        {"name": "Learning rate", "value": sqmpc_cfg["learning_rate"]},
        {"name": "Server learning rate", "value": sqmpc_cfg["server_lr"]},
        {"name": "Local epochs", "value": sqmpc_cfg["local_epochs"]},
        {"name": "Batch size", "value": sqmpc_cfg["batch_size"]},
        {"name": "Quantization bits", "value": f"{sqmpc_cfg['q_bits_first']}-{sqmpc_cfg['q_bits_mid']}-{sqmpc_cfg['q_bits_last']}"},
        {"name": "Quantization percentile", "value": "90% absolute-update scale from sqmpc_core.py"},
        {"name": "Aggregation in accuracy runner", "value": "Weighted FedAvg after quantized-dequantized client updates"},
        {"name": "SMPC transport in accuracy runner", "value": "Skipped by design; paper protocol uses additive secret sharing"},
    ]

    cid_hparams = [
        {"name": "Architecture", "value": "Conditional TCN generator plus one TCN discriminator/classifier per client"},
        {"name": "Optimizer", "value": f"Adam beta1={cidiot_cfg['beta1']}, beta2={cidiot_cfg['beta2']}"},
        {"name": "Learning rate", "value": cidiot_cfg["learning_rate"]},
        {"name": "Discriminator steps / round", "value": cidiot_cfg["discriminator_steps"]},
        {"name": "Generator steps / round", "value": cidiot_cfg["generator_steps"]},
        {"name": "Supervised pretrain steps", "value": cidiot_cfg["supervised_pretrain_steps"]},
        {"name": "Latent dimension", "value": cidiot_cfg["latent_dim"]},
        {"name": "Label embedding dimension", "value": cidiot_cfg["label_dim"]},
        {"name": "TCN channels", "value": cidiot_cfg["tcn_channels"]},
        {"name": "Hidden dimension", "value": cidiot_cfg["hidden_dim"]},
        {"name": "Loss weights", "value": "source=0.1, real class=1.0, fake class=0.0, generator class=1.0"},
        {"name": "Privacy transport in accuracy runner", "value": "DP and threshold secret sharing excluded by default"},
    ]

    setup_rows = [
        {"name": "Dataset", "value": str(DATASET_PATH.relative_to(ROOT))},
        {"name": "Task", "value": "Multiclass CICIoT2023 attack-family classification"},
        {"name": "Classes", "value": "Benign, DDoS, DoS, Recon, Web, BruteForce, Spoofing, Mirai"},
        {"name": "Split", "value": "70% train / 30% test, stratified by label"},
        {"name": "Normalization", "value": "Min-max scaling fit on train only, then applied to test"},
        {"name": "IID partition", "value": "Per-class shuffled split across 5 clients"},
        {"name": "Non-IID partition", "value": "Sort by label, split into 100 shards, assign shards to clients"},
        {"name": "Seeds", "value": "42, 43, 44"},
        {"name": "Audit", "value": "All pairwise rows have same_training_setup=True"},
        {"name": "Dataset SHA-256", "value": audit["dataset_sha256"]},
    ]

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>FL-SQMPC vs CIDIoT Accuracy Comparison</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #18202f;
      --muted: #5e6a7d;
      --line: #d9e0ea;
      --blue: #1f77b4;
      --red: #d62728;
      --green: #1b7f5a;
      --code: #0f172a;
      --soft-blue: #e8f2fb;
      --soft-red: #fbecec;
      --soft-green: #eaf7f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      line-height: 1.55;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 32px 40px 26px;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 32px 56px; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; line-height: 1.1; }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    h3 {{ margin: 24px 0 10px; font-size: 18px; }}
    p {{ margin: 0 0 14px; }}
    a {{ color: #0b65a3; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 18px 0;
      padding: 24px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .subtitle {{ color: var(--muted); max-width: 1000px; font-size: 16px; }}
    .toc {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }}
    .toc a {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 11px;
      text-decoration: none;
      background: #f9fbfe;
      color: var(--ink);
      font-size: 13px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 16px 0 6px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .metric-card span, .metric-card em {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-style: normal;
    }}
    .metric-card strong {{
      display: block;
      font-size: 22px;
      margin: 4px 0;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: var(--soft-blue);
      padding: 14px 16px;
      border-radius: 6px;
      margin: 14px 0;
    }}
    .callout.red {{ border-left-color: var(--red); background: var(--soft-red); }}
    .callout.green {{ border-left-color: var(--green); background: var(--soft-green); }}
    .eq {{
      font-family: \"SFMono-Regular\", Consolas, \"Liberation Mono\", monospace;
      background: #f4f6f9;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      overflow-x: auto;
      margin: 10px 0 16px;
      color: var(--code);
      white-space: pre-wrap;
    }}
    pre {{
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 8px;
      background: #111827;
      color: #eef2ff;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.45;
    }}
    code {{
      font-family: \"SFMono-Regular\", Consolas, \"Liberation Mono\", monospace;
      background: #eef2f7;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 18px;
      font-size: 14px;
    }}
    .table.compact {{ font-size: 13px; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f3f6fa;
      font-weight: 650;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }}
    figure img {{ width: 100%; display: block; border-radius: 4px; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    ul, ol {{ margin-top: 8px; }}
    li {{ margin-bottom: 6px; }}
    .small {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
<header>
  <h1>FL-SQMPC vs CIDIoT on CICIoT2023</h1>
  <p class=\"subtitle\">Detailed explanation of what is implemented, the mathematics behind each technique, the matched training protocol, hyperparameters, and the three-seed accuracy/loss results. Generated {date.today().isoformat()}.</p>
  <nav class=\"toc\">
    <a href=\"#scope\">Scope</a>
    <a href=\"#setup\">Shared Setup</a>
    <a href=\"#flsqmpc\">FL-SQMPC</a>
    <a href=\"#cidiot\">CIDIoT</a>
    <a href=\"#fairness\">Fairness Controls</a>
    <a href=\"#results\">Results</a>
    <a href=\"#plots\">Plots</a>
    <a href=\"#reproduce\">Reproduce</a>
  </nav>
</header>
<main>
{section("1. Scope of This Comparison", f'''
  <div class="metrics-grid">{key_metrics}</div>
  <div class="callout green">
    <strong>Bottom line.</strong> Under the matched three-seed experiment, FL-SQMPC achieved higher final accuracy than the implemented CIDIoT baseline on both IID and non-IID CICIoT2023 multiclass settings. The comparison isolates detection accuracy: privacy transport mechanisms are documented, but not all of them are executed when they should not affect classifier accuracy.
  </div>
  <p>This report compares two techniques:</p>
  <ol>
    <li><strong>FL-SQMPC:</strong> a federated MLP trained with FedAvg and mixed-precision SQMPC quantization. The paper-level protocol also encodes updates into a prime field and protects them through additive secret sharing.</li>
    <li><strong>CIDIoT:</strong> a collaborative generative IDS inspired by the IEEE IoT Journal paper. The implemented accuracy baseline uses a cloud-side conditional generator and edge-side TCN discriminators/classifiers trained on real and generated samples.</li>
  </ol>
  <p>The comparison is implementation-vs-implementation. CIDIoT article numbers are retained only as reference values in the result tables.</p>
''', "scope")}

{section("2. Shared Dataset and Training Setup", f'''
  <p>Both methods use the same local CICIoT2023 extract and the same preprocessing and partitioning functions.</p>
  {html_table(setup_rows, [("name", "Item"), ("value", "Value")])}
  <h3>Class Distribution After Cleaning</h3>
  {class_table}
  <h3>Common Evaluation Metrics</h3>
  <div class="eq">Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
Balanced accuracy = mean per-class recall</div>
  <p>The main reported value is final-round test accuracy. Precision, recall, macro F1, balanced accuracy, and evaluation cross-entropy loss are also recorded.</p>
''', "setup")}

{section("3. Technique 1: FL-SQMPC", f'''
  <h3>3.1 Paper-Level Idea</h3>
  <p>FL-SQMPC starts from standard synchronous federated learning. Each client trains locally, forms a model update, quantizes it layer by layer, encodes the result into an integer field, secret-shares the encoded update, and lets the server reconstruct only the cohort aggregate.</p>
  <div class="eq">Client k local objective:
F_k(w) = (1 / n_k) * sum_{{(x,y) in D_k}} loss(f(x; w), y)

Global objective:
F(w) = sum_k (n_k / N) * F_k(w)

Local model update:
Delta w_k^(t) = w_k^(t+1) - w^(t)

Weighted FedAvg:
w^(t+1) = w^(t) + eta_s * sum_k [n_k / sum_j n_j] * Delta w_k^(t)</div>

  <h3>3.2 SQMPC Quantization</h3>
  <p>The implemented runner calls <code>sqmpc_core.mixed_quantize_mlp</code>. For a layer update <code>u</code>, the core quantizer uses a percentile scale from the absolute update values, clamps normalized values to <code>[-1,1]</code>, performs stochastic rounding, then dequantizes back to floating point.</p>
  <div class="eq">scale_l = max(quantile_90(|u_l|), eps)
qmax_b = 2^(b-1) - 1
v = clamp(u_l / scale_l, -1, 1)
q = stochastic_round(v * qmax_b)
u_hat_l = (q / qmax_b) * scale_l</div>

  <h3>3.3 Paper-Level SMPC Transport</h3>
  <p>The manuscript-level protocol protects the encoded quantized update with additive secret sharing over a prime field.</p>
  <div class="eq">Encode:
E_k^(t) = round(sigma * Q_k^(t)) mod M

Weighted integer update:
Z_k^(t) = n_k * E_k^(t) mod M

Secret sharing:
Z_k^(t) = sum_s R_(k,s)^(t) mod M

Aggregator s:
A_s^(t) = sum_k R_(k,s)^(t) mod M

Server reconstruction:
A_w^(t) = sum_s A_s^(t) mod M

Decode:
Delta w_hat^(t) = Recenter(A_w^(t)) / (sigma * N_t)</div>

  <div class="callout">
    <strong>Important implementation distinction.</strong> The accuracy runner uses the same SQMPC quantizer from <code>sqmpc_core.py</code>, then directly performs weighted FedAvg. It intentionally skips the SMPC split/reconstruction transport because correct secret sharing reconstructs the same aggregate and should not change predictive accuracy. This means the accuracy result tests the optimization effect of quantization, not SMPC communication overhead or leakage resistance.
  </div>

  <h3>3.4 FL-SQMPC Hyperparameters Used Here</h3>
  {html_table(sqmpc_hparams, [("name", "Hyperparameter"), ("value", "Value")])}

  <h3>3.5 Algorithm Executed in <code>run_fl_sqmpc_accuracy.py</code></h3>
  <pre>for each communication round t:
    server broadcasts current MLP weights w_t
    for each client k:
        copy global MLP
        train locally for E epochs on client partition D_k
        compute update Delta w_k = w_k_local - w_t
        if mode == "sqmpc":
            quantize/dequantize Delta w_k layer-wise with SQMPC
    aggregate = sum_k (n_k / N) * Delta w_k
    update global model: w_(t+1) = w_t + eta_s * aggregate
    evaluate on the shared test set</pre>
''', "flsqmpc")}

{section("4. Technique 2: CIDIoT", f'''
  <h3>4.1 Paper-Level Idea</h3>
  <p>CIDIoT is a collaborative generative intrusion detection framework. The paper places a generator in the cloud controller and a discriminator/classifier at each edge node. The cloud generator produces class-conditioned synthetic flows; edge nodes train local discriminators using their private real traffic and generated samples. Edge nodes provide generator-training feedback to the cloud. The paper adds differential privacy to uploaded gradient information and dynamic threshold secret sharing to protect generated data sent down to edge nodes.</p>

  <h3>4.2 GAN and Classifier Mathematics</h3>
  <p>The paper starts from a GAN objective and extends it with a conditional, class-aware discriminator. Our implementation captures the accuracy-relevant version with a source head and a class head.</p>
  <div class="eq">Classical GAN objective:
min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]

CIDIoT-style conditional generator input:
z_hat = concat(z, embedding(y))
x_fake = G(z_hat)

Discriminator outputs:
D_source(x) -> real/fake logit
D_class(x)  -> class logits</div>

  <h3>4.3 Implemented CIDIoT Losses</h3>
  <div class="eq">Discriminator source loss:
L_source = BCE(D_source(x_real), 1) + BCE(D_source(x_fake), 0)

Discriminator class loss:
L_class = CE(D_class(x_real), y_real)

Implemented discriminator objective:
L_D = source_weight * L_source + real_class_weight * L_class

Generator loss against edge ensemble:
L_G = (1 / K) * sum_k [BCE(D_k_source(G(z,y)), 1) + generator_class_weight * CE(D_k_class(G(z,y)), y)]</div>

  <h3>4.4 TCN-Based Architecture</h3>
  <p>The implemented generator and discriminators use temporal-convolution-style 1D blocks over the feature vector. Each block uses dilated convolution, group normalization, leaky ReLU, and a residual shortcut. This follows the CIDIoT paper's use of TCN-style representation learning, while the exact code implementation is intentionally compact for the local experiment.</p>

  <h3>4.5 CIDIoT Hyperparameters Used Here</h3>
  {html_table(cid_hparams, [("name", "Hyperparameter"), ("value", "Value")])}

  <div class="callout red">
    <strong>Paper vs. local implementation.</strong> The CIDIoT paper reports Adam learning rate 0.0002, discriminator iterations 5, K=5 edge nodes, T=30 rounds, and batch size 100. In the local implemented baseline, learning rate 0.001, discriminator steps 10, and supervised pretraining 200 were used because the initial adversarial-only variant collapsed to majority-class behavior. This report compares the implemented CIDIoT baseline that trains reliably under the shared local setup.
  </div>

  <h3>4.6 Algorithm Executed in <code>run_cidiot_accuracy.py</code></h3>
  <pre>initialize one conditional TCN generator G in the cloud
initialize one TCN discriminator/classifier D_k per client
supervised-pretrain each D_k on local real data

for each communication round t:
    for each edge client k:
        repeat discriminator_steps:
            sample real batch (x_real, y_real) from D_k
            sample noise z and labels y_fake
            generate x_fake = G(z, y_fake)
            train D_k to classify real/fake and classify real labels
    repeat generator_steps:
        sample noise z and labels y
        generate x_fake = G(z, y)
        train G against the average feedback from all edge discriminators
    evaluate by averaging the class probabilities from all D_k</pre>
''', "cidiot")}

{section("5. Fairness Controls", f'''
  <p>The comparison uses the same experimental/training environment in the sense requested for this project: same data, split procedure, client partition procedure, number of clients, number of rounds, batch size, task, seeds, and evaluation metrics.</p>
  <ul>
    <li><strong>Same train/test split per seed:</strong> both methods use the same stratified 70/30 split for a given seed.</li>
    <li><strong>Same client partitions per seed:</strong> both methods use the same IID or non-IID client index partitions for a given seed and partition mode.</li>
    <li><strong>Same number of rounds:</strong> 30 communication rounds.</li>
    <li><strong>Same client count:</strong> 5 clients / edge nodes.</li>
    <li><strong>Same batch size:</strong> 100.</li>
    <li><strong>Same seeds:</strong> 42, 43, and 44.</li>
    <li><strong>Audit check:</strong> every pairwise result row reports <code>same_training_setup=True</code>.</li>
  </ul>
  <p>The model architectures are not forced to be identical because the comparison is between two different techniques. FL-SQMPC uses an MLP federated classifier. CIDIoT uses the implemented generator plus edge TCN discriminators/classifiers.</p>
''', "fairness")}

{section("6. Results", f'''
  <h3>6.1 Three-Seed Accuracy Summary</h3>
  {summary_table}
  <h3>6.2 Per-Seed Pairwise Accuracy</h3>
  {pairwise_table}
  <h3>6.3 Full Per-Seed Metrics</h3>
  {metrics_table}
  <div class="callout green">
    <strong>Interpretation.</strong> FL-SQMPC is higher on mean final accuracy for both partitions. In IID, FL-SQMPC reaches 0.7775 +/- 0.0075, while implemented CIDIoT reaches 0.6849 +/- 0.0495. In non-IID, FL-SQMPC reaches 0.7695 +/- 0.0160, while implemented CIDIoT reaches 0.7031 +/- 0.0058. Standard deviations are sample standard deviations over seeds 42, 43, and 44.
  </div>
''', "results")}

{section("7. Accuracy and Loss Plots", '''
  <p>The plots aggregate the three seeds into mean curves with standard-deviation bands. They are generated from the JSON histories in <code>output/results/</code>.</p>
  <div class="plot-grid">
    <figure>
      <img src="../plots/accuracy_multiclass_iid.png" alt="IID accuracy plot">
      <figcaption>IID accuracy over communication rounds.</figcaption>
    </figure>
    <figure>
      <img src="../plots/accuracy_multiclass_non_iid.png" alt="Non-IID accuracy plot">
      <figcaption>Non-IID accuracy over communication rounds.</figcaption>
    </figure>
    <figure>
      <img src="../plots/loss_multiclass_iid.png" alt="IID loss plot">
      <figcaption>IID loss over communication rounds.</figcaption>
    </figure>
    <figure>
      <img src="../plots/loss_multiclass_non_iid.png" alt="Non-IID loss plot">
      <figcaption>Non-IID loss over communication rounds.</figcaption>
    </figure>
  </div>
''', "plots")}

{section("8. Reproduction Commands", '''
  <p>Run the full matched three-seed experiment:</p>
  <pre>.venv/bin/python scripts/run_multi_seed_comparison.py --seeds 42,43,44 --task multiclass --partitions iid,non_iid --clients 5 --rounds 30 --batch-size 100 --device cpu</pre>
  <p>Regenerate tables only:</p>
  <pre>.venv/bin/python scripts/compare_accuracy.py</pre>
  <p>Regenerate plots only:</p>
  <pre>.venv/bin/python scripts/plot_training_curves.py --task multiclass</pre>
  <p>Regenerate this HTML report:</p>
  <pre>.venv/bin/python scripts/generate_explanation_report.py</pre>
''', "reproduce")}

{section("9. Traceability", '''
  <ul>
    <li>FL-SQMPC runner: <a href="../../scripts/run_fl_sqmpc_accuracy.py">scripts/run_fl_sqmpc_accuracy.py</a></li>
    <li>SQMPC core quantization/SMPC utilities: <a href="../../sqmpc_core.py">sqmpc_core.py</a></li>
    <li>CIDIoT runner: <a href="../../scripts/run_cidiot_accuracy.py">scripts/run_cidiot_accuracy.py</a></li>
    <li>Pairwise results: <a href="../tables/accuracy_pairwise.md">output/tables/accuracy_pairwise.md</a></li>
    <li>Mean/std results: <a href="../tables/accuracy_summary.md">output/tables/accuracy_summary.md</a></li>
    <li>FL-SQMPC manuscript source: <a href="../../main%20(4).tex">main (4).tex</a></li>
    <li>CIDIoT reference paper: <a href="../../Privacy-Preserving_Collaborative_Intrusion_Detection_in_Edge_of_Internet_of_Things_A_Robust_and_Efficient_Deep_Generative_Learning_Approach.pdf">Privacy-Preserving Collaborative Intrusion Detection...pdf</a></li>
  </ul>
  <p class="small">Note: local browser behavior for links to source files can vary. The files are all in the same project workspace.</p>
''', "traceability")}
</main>
</body>
</html>
"""
    return html_doc


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
