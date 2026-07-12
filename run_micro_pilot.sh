#!/usr/bin/env bash
set -e

# Load env vars
set -a
source .env
set +a

mkdir -p outputs/micro_pilot

DATASETS=(
  "rewardbench_pref:data/processed/splits/rewardbench_pref_sft_proxy_train.jsonl"
  "ifbench:data/processed/ifbench_queries.jsonl"
  "writingbench:data/processed/writingbench_queries.jsonl"
  "healthbench:data/processed/healthbench_queries.jsonl"
  "beir_nq:data/processed/beir_nq_queries.jsonl"
)

for entry in "${DATASETS[@]}"; do
  DS="${entry%%:*}"
  PATH_FILE="${entry##*:}"
  
  echo "--- Processing $DS ---"
  head -n 5 "$PATH_FILE" > "outputs/micro_pilot/${DS}_sample.jsonl"
  
  python3 scripts/estimate_api_budget.py \
    --input "outputs/micro_pilot/${DS}_sample.jsonl" \
    --providers configs/teachers_micro_pilot.jsonl \
    --output "outputs/micro_pilot/${DS}_budget.json" \
    --max-calls 100 \
    --max-total-tokens 500000 \
    --max-cost-usd 10.0 \
    --strict

  python3 scripts/generate_teacher_rubrics.py \
    --input "outputs/micro_pilot/${DS}_sample.jsonl" \
    --providers configs/teachers_micro_pilot.jsonl \
    --output "outputs/micro_pilot/${DS}_teachers.jsonl" \
    --data-source "$DS" \
    --sleep 1.0 \
    --require-budget-report "outputs/micro_pilot/${DS}_budget.json"
done

echo "Micro-pilot complete! Check outputs/micro_pilot/"