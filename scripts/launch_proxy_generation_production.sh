#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

PROVIDERS="${PROVIDERS:-configs/teachers_production.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/proxy_generation}"
MAX_CALLS="${MAX_CALLS:-500}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2000000}"
MAX_COST_USD="${MAX_COST_USD:-500}"
MAX_WALLCLOCK_MINUTES_SERIAL="${MAX_WALLCLOCK_MINUTES_SERIAL:-500}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.2}"
REWARDBENCH_PREF_INPUT="${REWARDBENCH_PREF_INPUT:-data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl}"

mkdir -p \
  "$OUTPUT_ROOT/teacher_outputs" \
  "$OUTPUT_ROOT/api_budget" \
  "$OUTPUT_ROOT/preflight" \
  "$OUTPUT_ROOT/logs"

run_source() {
  local source_name="$1"
  local input_path="$2"
  local target_per_batch="$3"

  local output_path="$OUTPUT_ROOT/teacher_outputs/${source_name}_teachers.jsonl"
  local budget_path="$OUTPUT_ROOT/api_budget/${source_name}_budget.json"
  local preflight_path="$OUTPUT_ROOT/preflight/${source_name}_preflight.json"
  local preflight_md="$OUTPUT_ROOT/preflight/${source_name}_preflight.md"
  local log_path="$OUTPUT_ROOT/logs/${source_name}.log"

  echo "=== ${source_name}: input=${input_path} target_per_batch=${target_per_batch} ==="

  python3 scripts/preflight_real_run.py \
    --input "$input_path" \
    --providers "$PROVIDERS" \
    --required-env GPT_AK_1 \
    --required-env GPT_AK_2 \
    --required-env GPT_AK_3 \
    --required-provider gpt-5.4 \
    --required-provider gpt-5 \
    --required-provider gpt-4o \
    --output "$preflight_path" \
    --output-md "$preflight_md" \
    --strict

  python3 scripts/run_batched_generation.py \
    --input "$input_path" \
    --providers "$PROVIDERS" \
    --output "$output_path" \
    --budget-report "$budget_path" \
    --preflight-report "$preflight_path" \
    --data-source "$source_name" \
    --target-per-batch "$target_per_batch" \
    --sleep "$SLEEP_SECONDS" \
    --max-calls "$MAX_CALLS" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-cost-usd "$MAX_COST_USD" \
    --max-wallclock-minutes-serial "$MAX_WALLCLOCK_MINUTES_SERIAL" \
    2>&1 | tee -a "$log_path"
}

run_source "rewardbench_pref" "$REWARDBENCH_PREF_INPUT" 60
run_source "ifbench" "data/processed/ifbench_queries.jsonl" 120
run_source "writingbench" "data/processed/writingbench_queries.jsonl" 30
run_source "healthbench" "data/processed/healthbench_queries.jsonl" 90
run_source "beir_nq" "data/processed/beir_nq_queries.jsonl" 180

echo "Proxy teacher generation complete. Outputs: $OUTPUT_ROOT/teacher_outputs"
