#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 INPUT_JSONL DATA_SOURCE OUTPUT_DIR [SAMPLE_SIZE] [SEED]" >&2
  exit 2
fi

INPUT_JSONL="$1"
DATA_SOURCE="$2"
OUTPUT_DIR="$3"
SAMPLE_SIZE="${4:-200}"
SEED="${5:-13}"

PROVIDER="${VERIFIER_PROVIDER:-configs/verifier.local.jsonl}"
MAX_CALLS="${MAX_CALLS:-2000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4000000}"
MAX_COST_USD="${MAX_COST_USD:-80}"
MAX_RUBRICS="${MAX_RUBRICS:-12}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

mkdir -p "$OUTPUT_DIR"

SAMPLE_PATH="$OUTPUT_DIR/sample_${DATA_SOURCE}.jsonl"
BUDGET_PATH="$OUTPUT_DIR/${DATA_SOURCE}_verifier_budget.json"
VERIFIED_PATH="$OUTPUT_DIR/${DATA_SOURCE}_verified.jsonl"
STATS_PATH="$OUTPUT_DIR/${DATA_SOURCE}_stats.jsonl"
SUMMARY_PATH="$OUTPUT_DIR/${DATA_SOURCE}_summary.json"
VALIDATE_DIR="$OUTPUT_DIR/validate_${DATA_SOURCE}"

python3 - "$INPUT_JSONL" "$SAMPLE_PATH" "$SAMPLE_SIZE" "$SEED" <<'PY'
import json
import random
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
sample_path = Path(sys.argv[2])
sample_size = int(sys.argv[3])
seed = int(sys.argv[4])

rows = []
with input_path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at line {line_no}: {input_path}") from exc

rng = random.Random(seed)
if sample_size > 0 and sample_size < len(rows):
    rows = rng.sample(rows, sample_size)

with sample_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Sampled {len(rows)} records to {sample_path}")
PY

python3 scripts/estimate_api_budget.py \
  --input "$SAMPLE_PATH" \
  --providers "$PROVIDER" \
  --unit-field rubrics \
  --output "$BUDGET_PATH" \
  --max-calls "$MAX_CALLS" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-cost-usd "$MAX_COST_USD" \
  --strict

python3 scripts/filter_rubrics_with_verifier.py \
  --input "$SAMPLE_PATH" \
  --output "$VERIFIED_PATH" \
  --stats-output "$STATS_PATH" \
  --mode api \
  --provider "$PROVIDER" \
  --require-budget-report "$BUDGET_PATH" \
  --data-source "$DATA_SOURCE" \
  --max-rubrics "$MAX_RUBRICS" \
  --reject-generic-terms \
  --drop-empty

python3 scripts/validate_rubric_outputs.py \
  --input "$VERIFIED_PATH" \
  --output-dir "$VALIDATE_DIR" \
  --min-rubrics 1 \
  --max-rubrics "$MAX_RUBRICS" \
  --require-valid-flags \
  --allow-semantic-redundancy \
  --strict

python3 - "$STATS_PATH" "$VERIFIED_PATH" "$SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
verified_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
n_input = 0
n_valid = 0
n_records = 0
reasons = {}
with stats_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        n_records += 1
        n_input += int(row.get("n_input", 0))
        n_valid += int(row.get("n_valid", 0))
        for reason, count in (row.get("invalid_reason_counts") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)

n_nonempty = sum(1 for _ in verified_path.open("r", encoding="utf-8")) if verified_path.exists() else 0
summary = {
    "records_sampled": n_records,
    "records_nonempty_after_filter": n_nonempty,
    "input_rubrics": n_input,
    "valid_rubrics": n_valid,
    "validity": n_valid / max(n_input, 1),
    "api_audit_pass_rate": n_valid / max(n_input, 1),
    "invalid_reason_counts": reasons,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
