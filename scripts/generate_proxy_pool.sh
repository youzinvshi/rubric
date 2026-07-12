#!/usr/bin/env bash
set -a
source .env
set +a

# 1. Estimate budget for the full proxy candidates
python3 scripts/estimate_api_budget.py \
  --input data/processed/healthbench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --output outputs/healthbench_budget.json \
  --max-calls 15000 \
  --max-total-tokens 10000000 \
  --max-cost-usd 200.0 \
  --strict

# You would do the same for the others here...

echo "Use scripts/run_batched_generation.py to execute under safety gates."
