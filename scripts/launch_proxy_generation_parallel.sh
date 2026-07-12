#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

PROVIDERS="${PROVIDERS:-configs/teachers_production.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/proxy_generation_parallel}"
MAX_CALLS="${MAX_CALLS:-500}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2000000}"
MAX_COST_USD="${MAX_COST_USD:-500}"
MAX_WALLCLOCK_MINUTES_SERIAL="${MAX_WALLCLOCK_MINUTES_SERIAL:-500}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
REWARDBENCH_PREF_INPUT="${REWARDBENCH_PREF_INPUT:-data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl}"

mkdir -p \
  "$OUTPUT_ROOT/providers" \
  "$OUTPUT_ROOT/teacher_outputs" \
  "$OUTPUT_ROOT/api_budget" \
  "$OUTPUT_ROOT/preflight" \
  "$OUTPUT_ROOT/logs"

python3 - "$PROVIDERS" "$OUTPUT_ROOT/providers" <<'PY'
import json
import sys
from pathlib import Path

providers = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
for line_no, line in enumerate(providers.open("r", encoding="utf-8"), start=1):
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    name = row.get("name")
    if not name:
        raise SystemExit(f"Provider missing name at line {line_no}: {providers}")
    path = out_dir / f"{name}.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)
PY

python3 scripts/seed_parallel_proxy_outputs.py \
  --source rewardbench_pref \
  --source ifbench \
  --source writingbench \
  --source healthbench \
  --source beir_nq \
  --parallel-root "$OUTPUT_ROOT/teacher_outputs"

run_job() {
  local source_name="$1"
  local input_path="$2"
  local target_per_batch="$3"
  local teacher="$4"

  local provider_path="$OUTPUT_ROOT/providers/${teacher}.jsonl"
  local output_path="$OUTPUT_ROOT/teacher_outputs/${source_name}/${teacher}.jsonl"
  local budget_path="$OUTPUT_ROOT/api_budget/${source_name}_${teacher}_budget.json"
  local preflight_path="$OUTPUT_ROOT/preflight/${source_name}_${teacher}_preflight.json"
  local preflight_md="$OUTPUT_ROOT/preflight/${source_name}_${teacher}_preflight.md"
  local log_path="$OUTPUT_ROOT/logs/${source_name}_${teacher}.log"

  (
    echo "=== ${source_name}/${teacher}: input=${input_path} target_per_batch=${target_per_batch} ==="
    python3 scripts/preflight_real_run.py \
      --input "$input_path" \
      --providers "$provider_path" \
      --required-provider "$teacher" \
      --output "$preflight_path" \
      --output-md "$preflight_md" \
      --strict

    python3 scripts/run_batched_generation.py \
      --input "$input_path" \
      --providers "$provider_path" \
      --output "$output_path" \
      --budget-report "$budget_path" \
      --preflight-report "$preflight_path" \
      --data-source "$source_name" \
      --target-per-batch "$target_per_batch" \
      --sleep "$SLEEP_SECONDS" \
      --on-error write_error \
      --max-calls "$MAX_CALLS" \
      --max-total-tokens "$MAX_TOTAL_TOKENS" \
      --max-cost-usd "$MAX_COST_USD" \
      --max-wallclock-minutes-serial "$MAX_WALLCLOCK_MINUTES_SERIAL"
  ) > "$log_path" 2>&1 &
  echo "$!" >> "$OUTPUT_ROOT/logs/pids.txt"
  echo "Started ${source_name}/${teacher} pid=$!"
}

: > "$OUTPUT_ROOT/logs/pids.txt"

for teacher in gpt-5.4 gpt-5 gpt-4o; do
  run_job "rewardbench_pref" "$REWARDBENCH_PREF_INPUT" 80 "$teacher"
  run_job "ifbench" "data/processed/ifbench_queries.jsonl" 120 "$teacher"
  run_job "writingbench" "data/processed/writingbench_queries.jsonl" 60 "$teacher"
  run_job "healthbench" "data/processed/healthbench_queries.jsonl" 120 "$teacher"
  run_job "beir_nq" "data/processed/beir_nq_queries.jsonl" 180 "$teacher"
done

echo "Parallel proxy generation launched. PID list: $OUTPUT_ROOT/logs/pids.txt"
echo "Wait with: while read -r pid; do wait \"\$pid\"; done < \"$OUTPUT_ROOT/logs/pids.txt\""
