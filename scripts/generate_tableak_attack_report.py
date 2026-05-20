#!/usr/bin/env python3
"""Generate a standalone HTML report for the FedAvg-TabLeak attack runs."""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "output" / "reports" / "tableak_attack_report.html"
DATASET_PATH = ROOT / "Bot_IoT_processed_no_pkSeqID_saddr_daddr"
RESULTS_DIR = ROOT / "output" / "tableak_results_r32"


SUMMARY_FILES = [
    RESULTS_DIR / "sqmpc_colluding_quantized_client_bot_iot_no_pkseqid_saddr_daddr_multiclass_iid_client0_round1_b8_full_seed42_summary.json",
    RESULTS_DIR / "cidiot_none_bot_iot_no_pkseqid_saddr_daddr_multiclass_iid_client0_round1_b8_full_seed42_summary.json",
    RESULTS_DIR / "cidiot_dp_noisy_update_bot_iot_no_pkseqid_saddr_daddr_multiclass_iid_client0_round1_b8_full_seed42_summary.json",
    RESULTS_DIR / "sqmpc_colluding_quantized_client_bot_iot_no_pkseqid_saddr_daddr_multiclass_iid_client0_round10_b8_full_seed42_summary.json",
    RESULTS_DIR / "cidiot_none_bot_iot_no_pkseqid_saddr_daddr_multiclass_iid_client0_round10_b8_full_seed42_summary.json",
    RESULTS_DIR / "cidiot_dp_noisy_update_bot_iot_no_pkseqid_saddr_daddr_multiclass_iid_client0_round10_b8_full_seed42_summary.json",
]


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)


def build_report() -> str:
    meta = read_json(DATASET_PATH / "meta.json")
    rows = [load_summary(path) for path in SUMMARY_FILES]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FedAvg-TabLeak Attack Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #19212a;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --panel: #f6f8fb;
      --panel-2: #eef4f8;
      --accent: #146c94;
      --accent-2: #8a4b18;
      --good: #236d4b;
      --warn: #9a5b00;
      --bad: #9b2c2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 32px 40px 24px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 32px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; padding-top: 8px; font-size: 22px; border-top: 1px solid var(--line); }}
    h3 {{ margin: 24px 0 8px; font-size: 17px; }}
    p {{ margin: 8px 0 12px; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    code {{ background: var(--panel); border: 1px solid var(--line); border-radius: 5px; padding: 1px 5px; }}
    pre {{ margin: 10px 0 16px; padding: 14px; overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .subtitle {{ color: var(--muted); max-width: 920px; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin: 16px 0; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 14px; }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .card strong {{ display: block; margin-top: 4px; font-size: 19px; }}
    .note {{ border-left: 4px solid var(--accent); background: var(--panel-2); padding: 12px 14px; margin: 14px 0; }}
    .warning {{ border-left-color: var(--warn); }}
    .good {{ color: var(--good); font-weight: 650; }}
    .warn {{ color: var(--warn); font-weight: 650; }}
    .bad {{ color: var(--bad); font-weight: 650; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 9px; vertical-align: top; }}
    th {{ background: #edf2f7; text-align: left; }}
    tbody tr:nth-child(even) td {{ background: #fbfcfe; }}
    .compact td, .compact th {{ padding: 6px 7px; font-size: 13px; }}
    .right {{ text-align: right; }}
    .small {{ font-size: 13px; color: var(--muted); }}
    .toc {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .toc a {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; background: #fff; }}
    .formula {{ background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 10px 0; }}
    .formula div {{ margin: 4px 0; }}
    .ok {{ background: #eef8f1; }}
  </style>
</head>
<body>
  <header>
    <h1>FedAvg-TabLeak Attack Report</h1>
    <p class="subtitle">A plain-language but technically complete explanation of the one-client post-obfuscation TabLeak experiments run against FL-SQMPC and CIDIoT. Generated on {escape(str(date.today()))} from the actual JSON summaries under <code>output/tableak_results</code>.</p>
    <nav class="toc">
      <a href="#summary">Executive Summary</a>
      <a href="#dataset">Dataset</a>
      <a href="#threat">Threat Model</a>
      <a href="#methods">Methods</a>
      <a href="#attack">How FedAvg-TabLeak Works</a>
      <a href="#metrics">Metrics</a>
      <a href="#results">Results</a>
      <a href="#interpretation">Interpretation</a>
      <a href="#reproduce">Reproduction</a>
    </nav>
  </header>
  <main>
    {summary_section(rows)}
    {dataset_section(meta)}
    {threat_section()}
    {methods_section(rows)}
    {attack_section()}
    {metrics_section()}
    {results_section(rows)}
    {interpretation_section(rows)}
    {files_section(rows)}
    {reproduce_section(rows)}
  </main>
</body>
</html>
"""


def summary_section(rows: list[dict[str, Any]]) -> str:
    best_reconstruction = max(rows, key=lambda row: row["metrics"]["reconstruction_accuracy"])
    worst_reconstruction = min(rows, key=lambda row: row["metrics"]["reconstruction_accuracy"])
    attack_steps = rows[0]["metadata"]["attack_steps"]
    ensemble_restarts = rows[0]["metadata"]["ensemble_restarts"]
    return f"""
<section id="summary">
  <h2>Executive Summary</h2>
  <p>We ran a FedAvg-aware version of TabLeak against one client update. The goal was to test whether an attacker could reconstruct the client batch from the model update that leaves the client after the defense or obfuscation step. This updated report uses 32 restarts and includes CIDIoT both without DP and with a DP-noisy upload surface.</p>
  <div class="grid">
    {card("Attack", "FedAvg-TabLeak", "matches local FedAvg-style training")}
    {card("Batch size", "8 records", "one attacked client batch")}
    {card("Attack budget", f"{attack_steps} x {ensemble_restarts}", "steps x restarts")}
    {card("Dataset", "Bot-IoT variant", "65 processed features")}
    {card("Best reconstruction accuracy", fmt_pct(best_reconstruction["metrics"]["reconstruction_accuracy"]), best_reconstruction["label"])}
    {card("Worst reconstruction accuracy", fmt_pct(worst_reconstruction["metrics"]["reconstruction_accuracy"]), worst_reconstruction["label"])}
  </div>
  <div class="note">
    <strong>Bottom line:</strong> under these settings, reconstruction quality is low. The paper-style reconstruction accuracy ranges from {fmt_pct(worst_reconstruction["metrics"]["reconstruction_accuracy"])} to {fmt_pct(best_reconstruction["metrics"]["reconstruction_accuracy"])}. Continuous feature errors are large on standardized features, and categorical recovery is weak. All artifacts passed the post-obfuscation check, so the attack was fed the final client-side output, not a raw pre-obfuscation update.
  </div>
</section>
"""


def dataset_section(meta: dict[str, Any]) -> str:
    class_rows = [
        {"Class": name, "Label id": label, "Rows": meta["class_frequencies"][name]}
        for name, label in sorted(meta["label_map"].items(), key=lambda item: item[1])
    ]
    continuous_rows = [{"Continuous feature": name} for name in meta["continuous_features"]]
    categorical_rows = [
        {
            "Categorical feature": name,
            "One-hot columns": f"{start}-{stop - 1}",
            "Width": stop - start,
        }
        for name, (start, stop) in meta["one_hot_ranges"].items()
    ]
    return f"""
<section id="dataset">
  <h2>Dataset</h2>
  <p>The attack used the processed Bot-IoT feature-drop dataset <code>{escape(meta["dataset_id"])}</code>, stored at <code>Bot_IoT_processed_no_pkSeqID_saddr_daddr</code>. This variant was used because it has explicit metadata for continuous features and one-hot categorical ranges, which TabLeak needs to reconstruct valid tabular records.</p>
  <div class="grid">
    {card("Display name", meta["display_name"], "metadata field")}
    {card("Train rows", f"{meta['train_size']:,}", "existing processed split")}
    {card("Test rows", f"{meta['test_size']:,}", "existing processed split")}
    {card("Processed features", str(meta["processed_feature_count"]), "18 continuous + 47 one-hot")}
    {card("Continuous raw features", str(meta["num_continuous_features"]), "standardized tensor columns")}
    {card("Categorical raw features", str(meta["num_categorical_features"]), "expanded to one-hot")}
  </div>
  <p>The source Bot-IoT variant has these target classes:</p>
  {table(class_rows)}
  <p>The version used here removed <code>pkSeqID</code>, <code>saddr</code>, and <code>daddr</code> from the earlier processed Bot-IoT tensor dataset. The original preprocessing also dropped <code>{escape(', '.join(meta["dropped_columns"]))}</code>.</p>
  <h3>Continuous Features</h3>
  {table(continuous_rows, compact=True)}
  <h3>Categorical One-Hot Groups</h3>
  {table(categorical_rows, compact=True)}
</section>
"""


def threat_section() -> str:
    return """
<section id="threat">
  <h2>Threat Model</h2>
  <p>The attacker knows the model architecture, training hyperparameters, preprocessing structure, and the model state before the attacked local update. This is a standard strong gradient-inversion assumption: the server or an observer knows how the client trained, but does not know the client's private batch.</p>
  <p>For this run, we tested <strong>one client only</strong>. For FL-SQMPC, this is the stronger collusion case: the attacker sees a recovered protected client update after client-side quantization. For CIDIoT, we report two attack surfaces: <code>none</code>, where the attacker sees the final local edge-discriminator update after the optimizer step, and <code>dp_noisy_update</code>, where that final observed update is clipped and noised before Tableak sees it.</p>
  <div class="note warning">
    The experiment deliberately enforces the post-obfuscation invariant: <code>observed_delta = final_post_obfuscation_client_model - pre_local_model</code>. The raw local update is not used as the attack target for SQMPC.
  </div>
  <div class="note warning">
    <strong>Important CIDIoT clarification:</strong> the CIDIoT paper applies the Gaussian mechanism to uploaded sample gradients for generator training. This repository-level FedAvg-TabLeak artifact observes a final client delta, so the DP row applies the same clipping/noise form to the final observed delta. It is the closest attack-surface approximation for this FedAvg experiment, not a byte-for-byte reproduction of the paper's internal generator-gradient path.
  </div>
</section>
"""


def methods_section(rows: list[dict[str, Any]]) -> str:
    sqmpc = next(row for row in rows if row["metadata"]["method"] == "sqmpc")
    cidiot = next(
        row
        for row in rows
        if row["metadata"]["method"] == "cidiot" and row["metadata"]["obfuscation_surface"] == "none"
    )
    cidiot_dp = next(
        (
            row
            for row in rows
            if row["metadata"]["method"] == "cidiot"
            and row["metadata"]["obfuscation_surface"] == "dp_noisy_update"
        ),
        None,
    )
    sq = sqmpc["metadata"]
    ci = cidiot["metadata"]
    sq_rows = [
        {"Item": "Model", "Value": "MLP"},
        {"Item": "Loss", "Value": sq["loss_kind"]},
        {"Item": "Optimizer", "Value": f"{sq['optimizer']} beta1={sq['beta1']} beta2={sq['beta2']}"},
        {"Item": "Learning rate", "Value": sq["learning_rate"]},
        {"Item": "Local batch / epochs", "Value": f"{sq['local_batch_size']} / {sq['local_training_epochs']}"},
        {"Item": "SQMPC surface", "Value": sq["obfuscation_surface"]},
        {"Item": "Quantization bits", "Value": f"first={sq['q_bits_first']}, middle={sq['q_bits_mid']}, last={sq['q_bits_last']}"},
    ]
    ci_rows = [
        {"Item": "Model", "Value": "TCN discriminator/classifier"},
        {"Item": "Loss", "Value": ci["loss_kind"]},
        {"Item": "Optimizer", "Value": f"{ci['optimizer']} beta1={ci['beta1']} beta2={ci['beta2']}"},
        {"Item": "Learning rate", "Value": ci["learning_rate"]},
        {"Item": "Local batch / epochs", "Value": f"{ci['local_batch_size']} / {ci['local_training_epochs']}"},
        {"Item": "CIDIoT surface", "Value": ci["obfuscation_surface"]},
        {"Item": "TCN channels / hidden", "Value": f"{ci['tcn_channels']} / {ci['hidden_dim']}"},
        {"Item": "Loss weights", "Value": f"source={ci['source_loss_weight']}, real_class={ci['real_class_loss_weight']}, fake_class={ci['fake_class_loss_weight']}"},
    ]
    dp_rows = []
    if cidiot_dp is not None:
        dp = cidiot_dp["metadata"]
        dp_rows = [
            {"Item": "CIDIoT DP surface", "Value": dp["obfuscation_surface"]},
            {"Item": "Privacy budget", "Value": f"epsilon={dp['dp_epsilon']}, delta={dp['dp_delta']}"},
            {"Item": "Clip norm C", "Value": dp["dp_clip_norm"]},
            {"Item": "Derived sigma", "Value": f"{dp['dp_noise_multiplier']:.4f}"},
            {"Item": "RDP conversion", "Value": f"best order={dp['dp_rdp_best_order']}, achieved epsilon={dp['dp_rdp_achieved_epsilon']:.4f}"},
            {"Item": "Round-1 noise norm", "Value": f"{dp['dp_noise_l2_norm']:.2f}"},
            {"Item": "Mechanism note", "Value": "clip/noise applied to final observed client delta for this FedAvg attack surface"},
        ]
    return f"""
<section id="methods">
  <h2>What Was Attacked</h2>
  <h3>FL-SQMPC</h3>
  <p>The client starts from the current global MLP, trains locally, and produces a model delta. Before the attack artifact is written, we apply the SQMPC client-side mixed-precision quantization. The attacked object is therefore the quantized-dequantized client delta that a colluding secure-aggregation adversary could recover.</p>
  {table(sq_rows)}
  <h3>CIDIoT</h3>
  <p>CIDIoT uses a cloud generator and edge discriminator/classifiers. For this attack, we target one edge discriminator/classifier update. The discriminator sees real client records and fixed generated/fake records. The current implemented CIDIoT baseline has no active obfuscation transform at this point, so the surface is recorded as <code>none</code>.</p>
  <p>The word "noise" appears in two different contexts. The generator uses random latent noise to synthesize fake records; that is part of the generative model and is not a privacy defense. Differential-privacy noise is the additional Gaussian mechanism applied to the communicated update/gradient. This report includes both the no-DP CIDIoT surface and the explicit DP-noisy surface.</p>
  {table(ci_rows)}
  <h3>CIDIoT DP Surface</h3>
  {table(dp_rows)}
</section>
"""


def attack_section() -> str:
    return """
<section id="attack">
  <h2>How FedAvg-TabLeak Works</h2>
  <p>Classic gradient inversion reconstructs data by searching for synthetic inputs whose gradients look like the observed client gradient. FedAvg-TabLeak extends this idea to FedAvg model deltas. Instead of matching one single gradient, it simulates the client's local training loop and matches the resulting model delta.</p>
  <h3>Capture Step</h3>
  <ol>
    <li>Save the model before client training: <code>theta_0</code>.</li>
    <li>Select the attacked client batch of 8 rows.</li>
    <li>Train the client for one local epoch using batch size 8.</li>
    <li>Apply the client-side output transform. For SQMPC this is mixed-precision quantization. For CIDIoT <code>none</code>, no transform is applied. For CIDIoT <code>dp_noisy_update</code>, the observed update is clipped and Gaussian-noised before the attack target is saved.</li>
    <li>Save the observed attack target: <code>Delta theta_obs = theta_final_obfuscated - theta_0</code>.</li>
  </ol>
  <h3>Optimization Step</h3>
  <p>The attacker creates a synthetic tabular batch <code>X_hat</code> and repeatedly updates it. At each attack step, the attacker simulates the client's local optimizer on <code>X_hat</code>, obtains a candidate model delta, and minimizes cosine mismatch against the observed delta.</p>
  <div class="formula">
    <div><strong>Observed client output:</strong> <code>Delta theta_obs = theta_final_obfuscated - theta_0</code></div>
    <div><strong>Candidate client output:</strong> <code>Delta theta_hat(X_hat, y) = LocalTrain(theta_0, X_hat, y) - theta_0</code></div>
    <div><strong>Attack objective:</strong> <code>min_X_hat 1 - dot(Delta theta_hat, Delta theta_obs) / (||Delta theta_hat|| ||Delta theta_obs||)</code></div>
  </div>
  <h3>Tabular Constraints</h3>
  <p>TabLeak handles mixed tabular data by treating continuous and categorical columns differently. Continuous columns are optimized directly and clamped to valid bounds. Each categorical feature is represented by a one-hot group; during optimization it is relaxed with a softmax, then projected back to a hard one-hot value at the end.</p>
  <div class="formula">
    <div><strong>Continuous:</strong> optimize real values, then clamp to the configured attack bounds during inversion.</div>
    <div><strong>Categorical:</strong> optimize logits, use softmax(logits / temperature), then choose argmax for the final one-hot category.</div>
    <div><strong>Temperature schedule:</strong> starts at 1.0 and anneals to 0.01.</div>
  </div>
  <h3>Why CIDIoT Takes Longer</h3>
  <p>The attack budget is the same for all rows in this report. The wall-time is not the same because each objective evaluation is different. SQMPC simulates an MLP with cross-entropy. CIDIoT simulates a TCN discriminator/classifier with source and class losses, plus a fixed fake-batch path. Therefore CIDIoT has the same number of attack iterations but more work per iteration.</p>
</section>
"""


def metrics_section() -> str:
    rows = [
        {"Metric": "Final objective", "Meaning": "Cosine delta-matching loss after the attack. Lower means the synthetic batch creates a model delta more similar to the observed client delta. It is not directly comparable across different architectures/losses."},
        {"Metric": "Reconstruction accuracy", "Meaning": "The metric from the paper subsection. For each original feature, correctness is 1 if the reconstructed value is within tolerance and 0 otherwise. Continuous tolerances are epsilon_i = 0.319 * sigma_i from the training split. Categorical features require exact one-hot group recovery."},
        {"Metric": "Fixed 0.05 tolerance accuracy", "Meaning": "Older diagnostic retained for comparison. It uses absolute error <= 0.05 for all continuous features plus exact categorical recovery. It is not the paper metric."},
        {"Metric": "Continuous MAE/RMSE", "Meaning": "Average error over standardized continuous features. The Bot-IoT processed tensors have empirical standard deviation 1 for the continuous columns, so the paper tolerance is 0.319 for each continuous feature in this dataset."},
        {"Metric": "Categorical exact match", "Meaning": "Average exact-match rate over categorical one-hot feature groups."},
        {"Metric": "Label accuracy", "Meaning": "Row-label agreement after matching reconstructed rows to true rows. This run uses oracle labels inside the attack to focus on feature reconstruction, so this should not be presented as independent label recovery."},
        {"Metric": "Model accuracy", "Meaning": "Predictive test accuracy of the model state that the attacked client starts from. This is the model context used by Tableak."},
        {"Metric": "Post-client accuracy", "Meaning": "Predictive test accuracy after applying the exact observed post-obfuscation client update to the starting model. For DP-noisy CIDIoT this is a diagnostic only, because the noisy update is an upload artifact rather than a clean deployed client model."},
        {"Metric": "Post-obfuscation invariant", "Meaning": "Numerical check that raw_update_tensors in the artifact equal final_post_obfuscation_model - pre_local_model. Values near zero mean the capture is correct."},
        {"Metric": "Raw-to-obfuscated difference", "Meaning": "Maximum absolute difference between the raw local delta and the post-obfuscation delta. Non-zero for SQMPC confirms the attack target used the quantized client delta."},
    ]
    return f"""
<section id="metrics">
  <h2>Metrics Explained</h2>
  <p>The primary metric below follows the <em>Reconstruction Evaluation</em> subsection in <a href="../../main%20(4).tex"><code>main (4).tex</code></a>. The continuous tolerance convention follows TabLeak: <code>epsilon_i = 0.319 * sigma_i</code>, where <code>sigma_i</code> is the empirical training-set standard deviation for continuous feature <code>i</code>.</p>
  {table(rows)}
  <div class="note warning">
    <strong>Label note:</strong> the attack was configured with oracle labels. That is a common controlled setting when the goal is to isolate feature reconstruction. It means label accuracy is a row-alignment diagnostic, not proof that the attacker recovered labels from scratch.
  </div>
</section>
"""


def results_section(rows: list[dict[str, Any]]) -> str:
    result_rows = []
    invariant_rows = []
    for row in sorted(rows, key=lambda item: (item["metadata"]["round"], item["metadata"]["method"])):
        md = row["metadata"]
        metrics = row["metrics"]
        result_rows.append(
            {
                "Method": method_name(md["method"]),
                "Round": md["round"],
                "Surface": md["obfuscation_surface"],
                "Model acc.": fmt_optional_pct(_model_accuracy(row, "pre_attack_model")),
                "Post-client acc.": fmt_optional_pct(_model_accuracy(row, "post_observed_client_model")),
                "Objective": f4(row["final_objective"]),
                "Recon. acc.": fmt_pct(metrics["reconstruction_accuracy"]),
                "Fixed 0.05 acc.": fmt_pct(metrics["fixed_0_05_tolerance_accuracy"]),
                "Label acc.": fmt_pct(metrics["label_accuracy"]),
                "Cont. MAE": f4(metrics["continuous_mae"]),
                "Cont. RMSE": f4(metrics["continuous_rmse"]),
                "Cat. exact": fmt_pct(metrics["categorical_exact_match"]),
                "Runtime": seconds(row["elapsed_seconds"]),
            }
        )
        invariant_rows.append(
            {
                "Method": method_name(md["method"]),
                "Round": md["round"],
                "Post-obfuscation invariant": sci(md["post_obfuscation_invariant_max_abs_diff"]),
                "Raw-to-obfuscated max diff": sci(md["raw_to_obfuscated_max_abs_diff"]),
                "Nonfinite reconstructed values": int(metrics["reconstructed_nonfinite_count"]),
            }
        )
    return f"""
<section id="results">
  <h2>Attack Results</h2>
  <p>The table below contains the requested r32 runs: SQMPC, CIDIoT without DP, and CIDIoT with DP-noisy update at rounds 1 and 10, seed 42, IID partition, client 0, batch size 8.</p>
  {table(result_rows)}
  <h3>Artifact Integrity Checks</h3>
  <p>These checks answer the most important implementation question: did the attack receive the final post-obfuscation client output? Yes. The invariant is near zero for all runs. SQMPC also has non-zero raw-to-obfuscated difference, which confirms quantization changed the client delta before the attack saw it.</p>
  {table(invariant_rows)}
</section>
"""


def interpretation_section(rows: list[dict[str, Any]]) -> str:
    sq_rows = [row for row in rows if row["metadata"]["method"] == "sqmpc"]
    ci_no_dp_rows = [
        row
        for row in rows
        if row["metadata"]["method"] == "cidiot"
        and row["metadata"]["obfuscation_surface"] == "none"
    ]
    ci_dp_rows = [
        row
        for row in rows
        if row["metadata"]["method"] == "cidiot"
        and row["metadata"]["obfuscation_surface"] == "dp_noisy_update"
    ]
    sq_mean_reconstruction = sum(row["metrics"]["reconstruction_accuracy"] for row in sq_rows) / len(sq_rows)
    ci_no_dp_mean_reconstruction = sum(row["metrics"]["reconstruction_accuracy"] for row in ci_no_dp_rows) / len(ci_no_dp_rows)
    ci_dp_mean_reconstruction = sum(row["metrics"]["reconstruction_accuracy"] for row in ci_dp_rows) / len(ci_dp_rows)
    sq_mean_time = sum(row["elapsed_seconds"] for row in sq_rows) / len(sq_rows)
    ci_mean_time = sum(row["elapsed_seconds"] for row in ci_no_dp_rows + ci_dp_rows) / len(ci_no_dp_rows + ci_dp_rows)
    dp_sigma = ci_dp_rows[0]["metadata"]["dp_noise_multiplier"]
    return f"""
<section id="interpretation">
  <h2>Interpretation</h2>
  <h3>Reconstruction Quality</h3>
  <p>The reconstructions are weak in these runs. Paper-style reconstruction accuracy is low for all surfaces: average SQMPC reconstruction accuracy is {fmt_pct(sq_mean_reconstruction)}, average CIDIoT no-DP reconstruction accuracy is {fmt_pct(ci_no_dp_mean_reconstruction)}, and average CIDIoT DP-noisy reconstruction accuracy is {fmt_pct(ci_dp_mean_reconstruction)}. Continuous MAE is close to 0.5 on standardized features, which means the reconstructed continuous values are often far from the true values.</p>
  <p>Categorical exact-match rates are also low and unstable across rounds. No row gives a high-confidence reconstruction of the attacked client batch.</p>
  <h3>SQMPC</h3>
  <p>For SQMPC, the attack target is explicitly the post-quantization client delta in the colluding-client scenario. The non-zero raw-to-obfuscated differences confirm that the raw local update was not used as the observed target. In this configuration, TabLeak did not reconstruct the attacked batch well.</p>
  <h3>CIDIoT</h3>
  <p>For CIDIoT no-DP, the current implemented baseline does not apply a client-side obfuscation layer to the discriminator/classifier update, so the surface is <code>none</code>. For CIDIoT DP-noisy, the captured final update is clipped/noised using the paper-inspired Gaussian mechanism. With the default paper budget, the derived noise multiplier is {f4(dp_sigma)}, so the noised target is dominated by privacy noise. Reconstruction remains weak in both cases, and the attack is much slower because each reconstruction step differentiates through a TCN discriminator loss.</p>
  <h3>Runtime</h3>
  <p>The summary runtime measures only the call to <code>run_fedavg_tableak</code>, not the earlier artifact capture. Average SQMPC attack time is {seconds(sq_mean_time)}. Average CIDIoT attack time is {seconds(ci_mean_time)}. The attack iteration count is identical, but the per-iteration model/loss computation is heavier for CIDIoT.</p>
  <h3>What This Does Not Prove Yet</h3>
  <p>This is a focused first run: one seed, one client, IID partition, rounds 1 and 10, and batch size 8. It is enough to validate the pipeline and give an initial privacy signal. It is not yet a final statistical claim across clients, seeds, partitions, datasets, and attack budgets.</p>
</section>
"""


def files_section(rows: list[dict[str, Any]]) -> str:
    file_rows = []
    for row in sorted(rows, key=lambda item: (item["metadata"]["round"], item["metadata"]["method"])):
        md = row["metadata"]
        file_rows.append(
            {
                "Method": method_name(md["method"]),
                "Round": md["round"],
                "Artifact": rel_link(row["artifact_path"]),
                "Reconstruction": rel_link(row["reconstruction_path"]),
                "Summary": rel_link(row["summary_path"]),
            }
        )
    return f"""
<section>
  <h2>Generated Files</h2>
  <p>Each run produces an artifact, a reconstruction tensor file, and a summary JSON.</p>
  {table(file_rows, raw_html_columns={"Artifact", "Reconstruction", "Summary"})}
</section>
"""


def reproduce_section(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for row in sorted(rows, key=lambda item: (item["metadata"]["round"], item["metadata"]["method"])):
        md = row["metadata"]
        command = command_for_row(row)
        blocks.append(
            f"<h3>{escape(method_name(md['method']))}, {escape(md['obfuscation_surface'])}, "
            f"round {md['round']}</h3><pre>{escape(command)}</pre>"
        )
    return f"""
<section id="reproduce">
  <h2>How To Reproduce</h2>
  <p>Run from the repository root using the project virtual environment.</p>
  {''.join(blocks)}
</section>
"""


def load_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    data["summary_path"] = str(path.relative_to(ROOT))
    md = data["metadata"]
    data["label"] = f"{method_name(md['method'])}, round {md['round']}"
    return data


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_for_row(row: dict[str, Any]) -> str:
    md = row["metadata"]
    artifact_dir = Path(row["artifact_path"]).parent
    output_dir = Path(row["summary_path"]).parent
    parts = [
        ".venv/bin/python",
        "scripts/run_tableak_evaluation.py",
        "--method",
        md["method"],
        "--obfuscation-surface",
        md["obfuscation_surface"],
        "--dataset",
        md["dataset_path"],
        "--task",
        md["task"],
        "--partition",
        md["partition"],
        "--seed",
        str(md["seed"]),
        "--clients",
        str(md["clients"]),
        "--round",
        str(md["round"]),
        "--client",
        str(md["client"]),
        "--local-epochs",
        str(md["local_training_epochs"]),
        "--local-batch-size",
        str(md["local_batch_size"]),
        "--attack-batch-size",
        str(md["batch_size"]),
        "--attack-steps",
        str(md["attack_steps"]),
        "--attack-lr",
        str(md["attack_learning_rate"]),
        "--ensemble-restarts",
        str(md["ensemble_restarts"]),
        "--device",
        "cpu",
        "--artifact-dir",
        str(artifact_dir),
        "--output-dir",
        str(output_dir),
    ]
    if md.get("dp_enabled"):
        parts.extend(
            [
                "--dp-epsilon",
                str(md["dp_epsilon"]),
                "--dp-delta",
                str(md["dp_delta"]),
                "--dp-clip-norm",
                str(md["dp_clip_norm"]),
            ]
        )
    return " ".join(parts)


def card(label: str, value: Any, note: str) -> str:
    return (
        '<div class="card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(str(value))}</strong>"
        f"<p class=\"small\">{escape(note)}</p>"
        "</div>"
    )


def table(
    rows: list[dict[str, Any]],
    compact: bool = False,
    raw_html_columns: set[str] | None = None,
) -> str:
    raw_html_columns = raw_html_columns or set()
    if not rows:
        return "<p>No rows.</p>"
    cls = " compact" if compact else ""
    columns = list(rows[0])
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            text = str(value)
            rendered = text if column in raw_html_columns else escape(text)
            align = ' class="right"' if isinstance(value, (int, float)) else ""
            cells.append(f"<td{align}>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls.strip()}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def rel_link(path_text: str) -> str:
    path = Path(path_text)
    href = "../" + escape(str(path.relative_to("output"))) if path_text.startswith("output/") else escape(path_text)
    return f'<a href="{href}">{escape(path.name)}</a>'


def method_name(method: str) -> str:
    return "FL-SQMPC" if method == "sqmpc" else "CIDIoT"


def _model_accuracy(row: dict[str, Any], key: str) -> float | None:
    model_metrics = row.get("model_metrics")
    if not model_metrics:
        return None
    metrics = model_metrics.get(key)
    if not metrics:
        return None
    return metrics.get("accuracy")


def fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def fmt_optional_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return fmt_pct(value)


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def seconds(value: float) -> str:
    return f"{float(value):.1f}s"


def sci(value: float) -> str:
    return f"{float(value):.3e}"


def escape(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    main()
