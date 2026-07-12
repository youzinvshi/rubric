#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/proxy_generation_parallel/gpt5_missing_parallel}"
PROVIDER="${PROVIDER:-configs/teachers_gpt5_selective.jsonl}"
SHARDS="${SHARDS:-4}"
TARGET_PER_BATCH="${TARGET_PER_BATCH:-80}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
MAX_CALLS="${MAX_CALLS:-500}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2000000}"
MAX_COST_USD="${MAX_COST_USD:-500}"
MAX_WALLCLOCK_MINUTES_SERIAL="${MAX_WALLCLOCK_MINUTES_SERIAL:-500}"

mkdir -p \
  "$OUTPUT_ROOT/queues" \
  "$OUTPUT_ROOT/shards" \
  "$OUTPUT_ROOT/teacher_outputs" \
  "$OUTPUT_ROOT/api_budget" \
  "$OUTPUT_ROOT/preflight" \
  "$OUTPUT_ROOT/logs"

build_missing_queue() {
  local source_name="$1"
  local pool_path="$2"
  local existing_path="$3"
  local queue_path="$OUTPUT_ROOT/queues/${source_name}_gpt5_missing.jsonl"

  python3 - "$source_name" "$pool_path" "$existing_path" "$queue_path" <<'PY'
import json
import sys
from pathlib import Path

source_name = sys.argv[1]
pool_path = Path(sys.argv[2])
existing_path = Path(sys.argv[3])
queue_path = Path(sys.argv[4])

done = set()
if existing_path.exists():
    with existing_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            teacher = row.get("teacher") or row.get("method")
            query = row.get("query")
            if teacher == "gpt-5" and query and not row.get("generation_failed"):
                done.add(str(query))

seen = set()
missing = []
with pool_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        query = row.get("query") or row.get("prompt") or row.get("instruction") or row.get("question")
        if not query:
            continue
        query = str(query)
        if query in seen:
            continue
        seen.add(query)
        if query in done:
            continue
        out = dict(row)
        out["query"] = query
        out["data_source"] = source_name
        out["missing_teacher"] = "gpt-5"
        missing.append(out)

queue_path.parent.mkdir(parents=True, exist_ok=True)
with queue_path.open("w", encoding="utf-8") as f:
    for row in missing:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"{source_name}: pool_unique={len(seen)} gpt5_done={len(done)} missing={len(missing)} queue={queue_path}")
PY
}

split_queue() {
  local source_name="$1"
  local queue_path="$OUTPUT_ROOT/queues/${source_name}_gpt5_missing.jsonl"
  local shard_dir="$OUTPUT_ROOT/shards/${source_name}"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"

  python3 - "$queue_path" "$shard_dir" "$SHARDS" <<'PY'
import json
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
n_shards = int(sys.argv[3])
rows = []
with queue_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

counts = [0] * n_shards
files = [
    (shard_dir / f"shard_{idx:02d}.jsonl").open("w", encoding="utf-8")
    for idx in range(n_shards)
]
try:
    for idx, row in enumerate(rows):
        shard_idx = idx % n_shards
        row["shard_id"] = shard_idx
        files[shard_idx].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[shard_idx] += 1
finally:
    for f in files:
        f.close()

for idx, count in enumerate(counts):
    print(f"{shard_dir / f'shard_{idx:02d}.jsonl'} {count}")
PY
}

run_shard() {
  local source_name="$1"
  local shard_path="$2"
  local shard_name
  shard_name="$(basename "$shard_path" .jsonl)"
  local output_path="$OUTPUT_ROOT/teacher_outputs/${source_name}/${shard_name}_gpt-5.jsonl"
  local budget_path="$OUTPUT_ROOT/api_budget/${source_name}_${shard_name}_gpt-5_budget.json"
  local preflight_path="$OUTPUT_ROOT/preflight/${source_name}_${shard_name}_gpt-5_preflight.json"
  local preflight_md="$OUTPUT_ROOT/preflight/${source_name}_${shard_name}_gpt-5_preflight.md"
  local log_path="$OUTPUT_ROOT/logs/${source_name}_${shard_name}_gpt-5.log"

  mkdir -p "$(dirname "$output_path")"
  (
    echo "=== ${source_name}/${shard_name}/gpt-5 input=${shard_path} ==="
    python3 scripts/preflight_real_run.py \
      --input "$shard_path" \
      --providers "$PROVIDER" \
      --required-provider gpt-5 \
      --output "$preflight_path" \
      --output-md "$preflight_md" \
      --strict

    python3 scripts/run_batched_generation.py \
      --input "$shard_path" \
      --providers "$PROVIDER" \
      --output "$output_path" \
      --budget-report "$budget_path" \
      --preflight-report "$preflight_path" \
      --data-source "$source_name" \
      --target-per-batch "$TARGET_PER_BATCH" \
      --sleep "$SLEEP_SECONDS" \
      --on-error write_error \
      --max-calls "$MAX_CALLS" \
      --max-total-tokens "$MAX_TOTAL_TOKENS" \
      --max-cost-usd "$MAX_COST_USD" \
      --max-wallclock-minutes-serial "$MAX_WALLCLOCK_MINUTES_SERIAL"
  ) > "$log_path" 2>&1 &
  echo "$!" >> "$OUTPUT_ROOT/logs/pids.txt"
  echo "Started ${source_name}/${shard_name}/gpt-5 pid=$! log=$log_path"
}

: > "$OUTPUT_ROOT/logs/pids.txt"

build_missing_queue "healthbench" "data/processed/healthbench_queries.jsonl" "outputs/proxy_generation_parallel/teacher_outputs/healthbench/gpt-5.jsonl"
build_missing_queue "beir_nq" "data/processed/beir_nq_queries.jsonl" "outputs/proxy_generation_parallel/teacher_outputs/beir_nq/gpt-5.jsonl"

split_queue "healthbench"
split_queue "beir_nq"

for shard_path in "$OUTPUT_ROOT"/shards/healthbench/*.jsonl; do
  run_shard "healthbench" "$shard_path"
done
for shard_path in "$OUTPUT_ROOT"/shards/beir_nq/*.jsonl; do
  run_shard "beir_nq" "$shard_path"
done

echo "Launched missing GPT-5 generation. PID list: $OUTPUT_ROOT/logs/pids.txt"
