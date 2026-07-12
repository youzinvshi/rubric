#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_smoke.example.json

echo "Smoke pipeline complete."
