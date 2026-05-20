#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN=".venv/bin/python"
RUNNER="scripts/run_tableak_evaluation.py"
RANGE_METRICS_RUNNER="scripts/summarize_tableak_range_metrics.py"

ARTIFACT_DIR="${ARTIFACT_DIR:-output/tableak_artifacts_ciciot_ton_b1_parallel}"
RESULTS_DIR="${RESULTS_DIR:-output/tableak_results_ciciot_ton_b1_parallel}"
LOG_DIR="${LOG_DIR:-output/tableak_logs_ciciot_ton_b1_parallel}"
METRICS_DIR="${METRICS_DIR:-output/tableak_metrics_ciciot_ton_b1_parallel}"
RANGE_TOLERANCE="${RANGE_TOLERANCE:-0.30}"
RUN_RANGE_METRICS="${RUN_RANGE_METRICS:-1}"
REFRESH_RANGE_METRICS_AFTER_JOB="${REFRESH_RANGE_METRICS_AFTER_JOB:-1}"
SEEDS="${SEEDS:-42}"
DATASETS="${DATASETS:-ciciot2023 ton_iot bot_iot}"
BOT_IOT_DATASET="${BOT_IOT_DATASET:-Bot_IoT_processed_minmax}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: expected venv python at $PYTHON_BIN" >&2
  exit 1
fi

RANGE_TOLERANCE_PCT="$(awk "BEGIN { printf \"%.0f\", $RANGE_TOLERANCE * 100 }")"
RANGE_METRICS_BASENAME="tableak_acc${RANGE_TOLERANCE_PCT}_range_summary"

mkdir -p "$ARTIFACT_DIR" "$RESULTS_DIR" "$LOG_DIR" "$METRICS_DIR"

declare -a COMMAND_LABELS
declare -a COMMANDS
declare -a PIDS
declare -a LABELS

add_command() {
  local label="$1"
  shift
  COMMAND_LABELS+=("$label")
  COMMANDS+=("$(printf '%q ' "$@")")
}

run_one() {
  local label="$1"
  shift
  local log_file="$LOG_DIR/${label}.log"

  echo "START $label"
  echo "  log: $log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  (
    echo "Command:"
    printf '%q ' "$@"
    printf '\n\n'
    "$@"
  ) >"$log_file" 2>&1 &

  PIDS+=("$!")
  LABELS+=("$label")
}

refresh_range_metrics() {
  if [[ "$RUN_RANGE_METRICS" != "1" ]]; then
    return 0
  fi

  "$PYTHON_BIN" "$RANGE_METRICS_RUNNER" \
    --results-dir "$RESULTS_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --range-tolerance "$RANGE_TOLERANCE" \
    --output-csv "$METRICS_DIR/${RANGE_METRICS_BASENAME}.csv" \
    --output-md "$METRICS_DIR/${RANGE_METRICS_BASENAME}.md" \
    >"$METRICS_DIR/${RANGE_METRICS_BASENAME}.log" 2>&1
}

wait_for_slot() {
  if [[ "$MAX_PARALLEL" -le 0 ]]; then
    return 0
  fi

  while [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
    wait_for_oldest
  done
}

wait_for_oldest() {
  local pid="${PIDS[0]}"
  local label="${LABELS[0]}"
  if wait "$pid"; then
    echo "OK   $label"
    if [[ "$REFRESH_RANGE_METRICS_AFTER_JOB" == "1" ]]; then
      if refresh_range_metrics; then
        echo "METRICS refreshed: $METRICS_DIR/${RANGE_METRICS_BASENAME}.md"
      else
        echo "WARN range metric refresh failed; see $METRICS_DIR/${RANGE_METRICS_BASENAME}.log" >&2
      fi
    fi
  else
    rc="$?"
    echo "FAIL $label (exit $rc); see $LOG_DIR/${label}.log" >&2
    status=1
  fi

  PIDS=("${PIDS[@]:1}")
  LABELS=("${LABELS[@]:1}")
}

add_method_for_dataset_round() {
  local dataset_label="$1"
  local dataset_path="$2"
  local round="$3"
  local method_key="$4"
  local seed="$5"

  local common_args=(
    --dataset "$dataset_path"
    --task multiclass
    --partition iid
    --seed "$seed"
    --clients 5
    --round "$round"
    --client 0
    --local-epochs 1
    --local-batch-size 1
    --attack-batch-size 1
    --attack-steps 1500
    --attack-lr 0.001
    --ensemble-restarts 32
    --device cpu
    --artifact-dir "$ARTIFACT_DIR"
    --output-dir "$RESULTS_DIR"
  )

  case "$method_key" in
    vanilla_fl)
      add_command "${dataset_label}_vanilla_fl_r${round}_seed${seed}" \
        "$PYTHON_BIN" "$RUNNER" \
        --method sqmpc \
        --obfuscation-surface vanilla_fl \
        --no-quantize \
        "${common_args[@]}"
      ;;
    sqmpc)
      add_command "${dataset_label}_sqmpc_r${round}_seed${seed}" \
        "$PYTHON_BIN" "$RUNNER" \
        --method sqmpc \
        --obfuscation-surface colluding_quantized_client \
        "${common_args[@]}"
      ;;
    cidiot_no_dp)
      add_command "${dataset_label}_cidiot_no_dp_r${round}_seed${seed}" \
        "$PYTHON_BIN" "$RUNNER" \
        --method cidiot \
        --obfuscation-surface none \
        "${common_args[@]}"
      ;;
    cidiot_dp_noisy)
      add_command "${dataset_label}_cidiot_dp_noisy_r${round}_seed${seed}" \
        "$PYTHON_BIN" "$RUNNER" \
        --method cidiot \
        --obfuscation-surface dp_noisy_update \
        --dp-epsilon 8 \
        --dp-delta 1e-5 \
        --dp-clip-norm 1.0 \
        "${common_args[@]}"
      ;;
    *)
      echo "error: unknown method key: $method_key" >&2
      exit 1
      ;;
  esac
}

add_dataset_method_for_round_seed() {
  local dataset_key="$1"
  local round="$2"
  local method_key="$3"
  local seed="$4"

  case "$dataset_key" in
    ciciot2023)
      add_method_for_dataset_round "ciciot2023" "ciciot2023_processed_tensor" "$round" "$method_key" "$seed"
      ;;
    ton_iot)
      add_method_for_dataset_round "ton_iot" "ton_iot_processed_tensor" "$round" "$method_key" "$seed"
      ;;
    bot_iot)
      add_method_for_dataset_round "bot_iot" "$BOT_IOT_DATASET" "$round" "$method_key" "$seed"
      ;;
    bot_iot_no_pkseqid)
      add_method_for_dataset_round "bot_iot_no_pkseqid" "Bot_IoT_processed_no_pkSeqID_saddr_daddr" "$round" "$method_key" "$seed"
      ;;
    *)
      echo "error: unknown dataset key: $dataset_key" >&2
      echo "supported dataset keys: ciciot2023 ton_iot bot_iot bot_iot_no_pkseqid" >&2
      exit 1
      ;;
  esac
}

echo "Writing artifacts to: $ARTIFACT_DIR"
echo "Writing results to:   $RESULTS_DIR"
echo "Writing logs to:      $LOG_DIR"
echo "Writing metrics to:   $METRICS_DIR"
echo "Range metric:         Acc@${RANGE_TOLERANCE_PCT}% of feature range"
echo "Seeds:                $SEEDS"
echo "Datasets:             $DATASETS"
if [[ "$DATASETS" == *"bot_iot"* ]]; then
  echo "Bot-IoT dataset path: $BOT_IOT_DATASET"
fi
if [[ "$REFRESH_RANGE_METRICS_AFTER_JOB" == "1" ]]; then
  echo "Partial metrics:      refresh after each completed job"
else
  echo "Partial metrics:      final summary only"
fi
if [[ "$MAX_PARALLEL" -gt 0 ]]; then
  echo "Max parallel jobs:    $MAX_PARALLEL"
else
  echo "Max parallel jobs:    unlimited"
fi
echo

for seed in $SEEDS; do
  for round in 1 10; do
    for method_key in vanilla_fl sqmpc cidiot_no_dp cidiot_dp_noisy; do
      for dataset_key in $DATASETS; do
        add_dataset_method_for_round_seed "$dataset_key" "$round" "$method_key" "$seed"
      done
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  for i in "${!COMMANDS[@]}"; do
    label="${COMMAND_LABELS[$i]}"
    command="${COMMANDS[$i]}"
    log_file="$LOG_DIR/${label}.log"
    echo "START $label"
    echo "  log: $log_file"
    echo "  command: $command"
  done
  echo
  if [[ "$RUN_RANGE_METRICS" == "1" ]]; then
    echo "Post-run range metrics command:"
    printf '  %q %q --results-dir %q --artifact-dir %q --range-tolerance %q --output-csv %q --output-md %q\n' \
      "$PYTHON_BIN" "$RANGE_METRICS_RUNNER" \
      "$RESULTS_DIR" "$ARTIFACT_DIR" "$RANGE_TOLERANCE" \
      "$METRICS_DIR/${RANGE_METRICS_BASENAME}.csv" \
      "$METRICS_DIR/${RANGE_METRICS_BASENAME}.md"
    echo
  fi
  echo "Dry run only; no jobs launched."
  exit 0
fi

status=0
for i in "${!COMMANDS[@]}"; do
  wait_for_slot
  # shellcheck disable=SC2086
  run_one "${COMMAND_LABELS[$i]}" ${COMMANDS[$i]}
done

echo

while [[ "${#PIDS[@]}" -gt 0 ]]; do
  wait_for_oldest
done

echo
if [[ "$status" -eq 0 ]]; then
  echo "All Tableak jobs completed successfully."
  echo "Summary JSON files are in: $RESULTS_DIR"
  if [[ "$RUN_RANGE_METRICS" == "1" ]]; then
    echo
    echo "Computing Acc@${RANGE_TOLERANCE_PCT}% feature-range reconstruction metrics..."
    if refresh_range_metrics; then
      echo "Range-normalized metrics are in: $METRICS_DIR"
    else
      echo "Range metric summarization failed." >&2
      status=1
    fi
  fi
else
  echo "One or more Tableak jobs failed. Check logs in: $LOG_DIR" >&2
fi

exit "$status"
