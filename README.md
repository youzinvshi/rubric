# BlindSpot-RL

BlindSpot-RL studies an upstream failure mode in LLM-as-a-Judge systems:
before a judge assigns a final score or preference, its evaluation-criteria
basis may already omit dimensions that human annotators consider essential.
The project frames these omissions as **evaluation blind spots**, not as a
generic criteria-text polishing problem.

The core artifact is Blind-Spot Coverage (BSC), a computable semantic reward
that measures coverage of human-gold evaluation dimensions while penalizing
redundant and invalid or hallucinated criteria. The current paper title is
`Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation`.

Current claim discipline:

- Safe to report: the 100-example hard-gold diagnostic showing mean coverage
  `0.3692` and mean blind-spot rate `0.6308` against human-gold evaluation
  dimensions.
- Evidence-gated: SFT-only, SFT+GRPO, downstream utility, dimension-level recovery,
  semantic-space visualization, and ablation claims are permitted only after the
  real evidence matrix and submission readiness gates pass.
- BSC-only results are metric-support, not downstream judge-utility evidence;
  RewardBench, RewardBench-2, and JudgeBench rows require the API/model scorer
  and paper-eligibility gates before utility claims are safe.
- Data isolation is part of the scientific claim: RubricBench `test_main`
  stays hard-gold holdout; RewardBench proxy-train is filtered to a 1,785-example
  clean split against RubricBench `test_main` and RewardBench/JudgeBench/
  RewardBench-2 downstream holdouts.
- Proxy-gold criteria are used for training scale and must not be described as
  equivalent to human-gold evaluation dimensions.

## What Is Implemented

- `src/blindspot_rl/reward_bsc.py`
  - `coverage_reward`: Cov, the share of human gold dimensions covered by generated criteria.
  - `redundancy_penalty`: Red, semantic duplicate penalty inside generated criteria.
  - `validity_reward`: R_valid, verifier-based validity ratio.
  - `compute_reward`: `1.0 * Rcov + 0.5 * Rvalid - 0.5 * Rred`.
  - `verl_reward_fn`: small adapter for verl-style reward hooks.
- `src/blindspot_rl/policy_reward.py`
  - Policy-level verl reward hook that scores generated answers against cached generated criteria.
- `scripts/bsc_diagnose.py`
  - Batch diagnostic script for the minimal paper claim: single-model Cov/Blind on gold rubric data.
- `scripts/sweep_bsc_thresholds.py`
  - Runs BSC across threshold grids for robustness/sensitivity tables.
- `scripts/prepare_bsc_eval.py`
  - Joins gold criteria and model-generated criteria by query for BSC diagnostics.
- `scripts/prepare_downstream_eval.py`
  - Joins preference records and generated criteria by query for downstream accuracy evaluation.
- `scripts/convert_to_verl_parquet.py`
  - Converts `{query, gold_rubrics}` records to `{prompt, gold_rubrics, data_source}` parquet/jsonl.
- `scripts/convert_policy_rlvr_data.py`
  - Converts HealthBench-Hard/ArenaHard query pools to policy-RLVR train/validation parquet/jsonl.
- `scripts/download_public_data.py`
  - Downloads HF benchmark splits such as RewardBench/JudgeBench to JSONL.
- `scripts/build_data_pipeline.py`
  - Expands a dataset source manifest into download/normalize stages and a readiness manifest.
- `scripts/normalize_dataset.py`
  - Normalizes raw JSONL/JSON/parquet files into gold, query-pool, or preference schemas.
- `scripts/build_sft_data.py`
  - Builds LLaMA-Factory style SFT data and proxy gold from multi-teacher criteria elicitation.
- `scripts/generate_teacher_rubrics.py`
  - Calls OpenAI-compatible APIs to elicit multi-teacher criteria.
- `scripts/generate_model_rubrics.py`
  - Elicits method-specific evaluation criteria for `base/gpt4o/claude/sft_only/sft_rl` BSC and downstream tables.
- `scripts/filter_rubrics_with_verifier.py`
  - Filters criteria candidates with a Meta-Verifier for SFT targets and `R_valid`.
- `pipelines/run_smoke_pipeline.sh`
  - Runs the local toy end-to-end pipeline without API keys or model downloads.
- `configs/llamafactory_sft.example.yaml`
  - LLaMA-Factory SFT config template.
- `configs/verl_grpo_bsc.example.yaml`
  - verl GRPO config template using the BSC custom reward.
- `configs/verl_policy_grpo_healthbench_hard.example.yaml` and `configs/verl_policy_grpo_arenahard.example.yaml`
  - downstream policy-RLVR config templates using `src/blindspot_rl/policy_reward.py`.
- `scripts/audit_experiment.py`
  - Checks that expected experiment artifacts and summary metrics are present.
- `scripts/build_evidence_matrix.py`
  - Maps paper claims to required artifacts/metrics and marks them safe, missing, or contradicted.
- `scripts/sync_paper_artifacts.py`
  - Syncs exported tables, evidence docs, dashboard, Result Card, gap report, and reviewer-facing docs into the `paper/` manuscript workspace.
- `scripts/check_paper_asset_index.py`
  - Verifies that synced paper assets exist and match the SHA256 fingerprints in `paper/asset_index.md`.
- `scripts/run_experiment_pipeline.py`
  - Runs configurable stage-based experiment pipelines with `--dry-run` and `--only`.
- `scripts/build_experiment_matrix.py`
  - Expands a method list into multi-method BSC/downstream pipeline and audit manifest configs.
- `scripts/make_training_commands.py`
  - Generates executable SFT/GRPO training scripts and a training manifest.
- `scripts/make_downstream_rlvr_commands.py` and `scripts/evaluate_policy_outputs.py`
  - Generate HealthBench-Hard/ArenaHard policy-RLVR command packets, check policy-RLVR input readiness, and summarize policy prediction artifacts.
- `scripts/register_llamafactory_dataset.py`
  - Registers BlindSpot-RL SFT JSONL files in LLaMA-Factory `dataset_info.json`.
- `configs/pipeline_smoke.example.json`
  - JSON pipeline config for the local toy end-to-end run.
- `configs/methods_matrix_smoke.example.json`
  - Example method matrix config for paper-style multi-method tables.
- `configs/training_commands.example.json`
  - Example command-generation config for LLaMA-Factory SFT and verl GRPO.

## Install

For a quick local smoke test:

```bash
pip install -e .
```

For real BGE embeddings and parquet support:

```bash
pip install -r requirements.txt
```

## Run The Toy Diagnostic

This uses the deterministic `token-overlap` embedder, so it does not download
`BAAI/bge-large-en-v1.5`.

```bash
python3 scripts/bsc_diagnose.py \
  --input examples/toy_bsc.jsonl \
  --embedding-model token-overlap \
  --coverage-tau 0.5 \
  --redundancy-tau 0.9 \
  --output-dir outputs/toy_bsc
```

Expected outputs:

- `outputs/toy_bsc/summary.json`
- `outputs/toy_bsc/per_item.csv`

For the paper experiment, replace `--embedding-model token-overlap` with the
default `BAAI/bge-large-en-v1.5` and feed RubricBench-style model outputs.

## Prepare BSC Inputs From Gold And Model Outputs

Real BSC experiments usually have two files: human gold criteria and model
generated criteria. Join them by query before running diagnostics:

```bash
python3 scripts/prepare_bsc_eval.py \
  --gold data/processed/splits/rubricbench_gold_test_main.jsonl \
  --predictions data/processed/base_model_rubrics.jsonl \
  --output data/processed/base_bsc_eval.jsonl \
  --report outputs/base_bsc_join_report.json \
  --model base \
  --data-source rubricbench
```

Then run:

```bash
python3 scripts/bsc_diagnose.py \
  --input data/processed/base_bsc_eval.jsonl \
  --output-dir outputs/base_bsc
```

This is the path for the first paper claim: single-model `Cov` and `Blind`.

To generate the whole minimal-claim pipeline and its evidence gates:

```bash
cp configs/minimal_claim_real.template.json configs/minimal_claim_real.local.json
python3 scripts/build_minimal_claim_pipeline.py \
  --config configs/minimal_claim_real.local.json \
  --pipeline-output configs/pipeline_minimal_claim.generated.json \
  --manifest-output configs/manifest_minimal_claim.generated.json \
  --evidence-output configs/evidence_minimal_claim.generated.json \
  --result-card-output configs/result_card_minimal_claim.generated.json

python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_minimal_claim.generated.json \
  --dry-run
```

Run this before SFT/GRPO: it produces the single-model blind-spot number,
threshold sweep, audit report, and paper artifacts for the motivation section.
The generated minimal pipeline starts by initializing/auditing
`configs/data_sources_real.local.json`, then builds a strict source report scoped
to `rubricbench` so unrelated extension datasets do not block the first BSC
experiment. The initialization report is scoped to `rubricbench` too, and is
included as a Raw Audit Gate for the minimal readiness report and result card.
Before sampling or API generation, it also validates the RubricBench hard-gold
and query-pool files with source/provenance gates and writes
`outputs/data_validation/rubricbench_gold.json`.
Those gates require both `paper_url=https://arxiv.org/abs/2603.01562` and
`source_url=https://huggingface.co/datasets/DonJoey/rubricbench/resolve/main/data/train-00000-of-00001.parquet`,
so a same-paper proxy or stale mirror cannot satisfy the minimal motivation
claim.
The real template also appends a `result_card` stage, so the final output
includes a paper-facing claim decision card under
`outputs/minimal_claim/base/paper_artifacts/`, then syncs it into the paper
asset index when `paper_sync.enabled=true`.
The real template also includes optional startup stages for a reproducible
pilot sample, offline preflight, API budget estimate, and a machine-readable
paid API handoff report:

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_minimal_claim.generated.json \
  --only init_data_source_local_config \
  --only data_source_report \
  --only validate_rubricbench_gold \
  --only validate_rubricbench_queries \
  --only bsc_gold_sanity \
  --only sample_queries \
  --only preflight \
  --only api_budget \
  --only minimal_api_handoff
```

Inspect the generated reports under `outputs/minimal_claim/base/sampling/`,
`outputs/minimal_claim/base/preflight/`, and
`outputs/minimal_claim/base/api_budget/`, then inspect
`outputs/minimal_claim/base/handoff/api_handoff.json` before launching paid
model generation. Keep `preflight.strict=false` while drafting local configs,
then turn it on when you want missing API keys or inputs to stop the preflight
stage itself; paid generation and verifier stages still require the preflight
report to be `ok=true`. The handoff report is the safest resume point: it
re-checks preflight blockers, API budget blockers, BSC gold-sanity status,
stage ordering, and SHA256 consistency for the current input/provider/resume
files. If it says `status=blocked`, follow its `rerun_offline_gates` command
after fixing local configs or API keys; that command refreshes the source
report, hard-gold validation, query validation, BSC gold sanity, sample,
preflight, budget, and handoff without running paid generation.
The paid stages also verify that the preflight report covered the current
input/provider files, so do not reuse an older preflight report after changing
`configs/generators_minimal.local.jsonl`, `configs/verifier.local.jsonl`, or
the sample/query input. The preflight report records SHA256 for those files,
and paid stages reject the report if the current file content no longer
matches. They also re-check the provider `api_key_env` names from the preflight
report against the current shell environment, so rerun preflight after changing
API key exports. In the minimal template, preflight also checks loopback
OpenAI-compatible endpoints such as `http://localhost:8000/v1` with a TCP
connection test, so a stopped local generation server blocks before the paid
range starts. External providers such as OpenAI are not probed by this health
check.
API budget reports also bind the current input/provider SHA256, so rerun both
`preflight` and `api_budget` after changing generation inputs or provider
configs. If a `--resume-output` file appears or changes after budget
estimation, rerun `api_budget` before launching the paid stage.

After the local provider files and API keys are ready, rerun the offline gates,
then run the handoff ready check. It fails closed unless
`outputs/minimal_claim/base/handoff/api_handoff.json` has
`status=ready_for_paid_run`, `ok=true`, and no blockers:

```bash
python3 scripts/check_minimal_api_handoff_ready.py \
  --handoff outputs/minimal_claim/base/handoff/api_handoff.json
```

Only after that check passes, launch the paid part and downstream diagnostics
as one ordered range:

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_minimal_claim.generated.json \
  --from-stage generate_model_rubrics \
  --to-stage result_card \
  --require-ready-handoff outputs/minimal_claim/base/handoff/api_handoff.json
```

This range includes model criteria elicitation, raw criteria validation,
meta-verifier budget estimation, verifier annotation, verified-output
validation, `prepare_bsc`, BSC diagnosis, threshold sweep, bootstrap CI,
summary table, evidence matrix, audit, readiness, and the paper-facing result
card. The runner also supports `--from-stage/--to-stage` with `--dry-run` for a
copyable execution plan before any paid call is made.
When `paper_sync.enabled=true`, the minimal pipeline also runs
`paper_asset_index_check` after syncing paper assets and again after the final
result-card sync. The check writes
`outputs/minimal_claim/base/paper_artifacts/paper_asset_index_check.json` and
`.md`; it verifies that every file listed in `paper/asset_index.md` exists and
matches its SHA256 fingerprint, and it fails under `--strict` if the asset
index itself declares any blockers. The check report is intentionally not added
to `paper/asset_index.md`, which avoids a self-referential hash loop.
For the minimal pipeline, create the local API configs from examples:

```bash
cp configs/generators_minimal.example.jsonl configs/generators_minimal.local.jsonl
cp configs/verifier_minimal.example.jsonl configs/verifier.local.jsonl
export LOCAL_OPENAI_API_KEY=...
export OPENAI_API_KEY=...
```

Keep the generator provider name as `base`; the minimal BSC join filters
`model=base`, and preflight requires this name in
`configs/generators_minimal.local.jsonl`.
Keep the verifier provider name as `meta-verifier`; preflight requires that
name in `configs/verifier.local.jsonl`. Preflight checks the provider file's
own `api_key_env` fields in addition to the minimal pipeline's required
`LOCAL_OPENAI_API_KEY` and `OPENAI_API_KEY`.
After export, the real template can also sync generated artifacts into `paper/`
and write `outputs/minimal_claim/base/readiness/readiness_report.md`, giving a
paper-side check that the motivation claim is backed by the current audit and
evidence matrix.

For a threshold sensitivity appendix:

```bash
python3 scripts/sweep_bsc_thresholds.py \
  --input data/processed/base_bsc_eval.jsonl \
  --coverage-tau 0.70 0.75 0.80 \
  --redundancy-tau 0.80 0.85 0.90 \
  --output-dir outputs/base_bsc_sweep
```

## Input Format For BSC Diagnostics

JSONL is the simplest format:

```json
{"query": "...", "gold_rubrics": ["..."], "response": ["..."], "data_source": "rubricbench"}
```

Accepted aliases:

- prompt: `query`, `prompt`, `instruction`
- gold evaluation dimensions: `gold_rubrics`, `gold`, `rubrics_gold`
- model output: `response`, `model_rubrics`, `generated_rubrics`, `prediction`, `output`

## Convert Proxy Gold Data For verl

```bash
python3 scripts/convert_to_verl_parquet.py \
  --input data/processed/proxy_gold.jsonl \
  --output data/processed/proxy_gold_verl.parquet \
  --data-source multi_teacher_proxy \
  --min-records 1000 \
  --report-output outputs/sft_data/proxy_gold_verl_report.json
```

The converter fails closed if the input path, `--data-source`, or record-level
`data_source` / `split` metadata indicates `test_main`; GRPO data must come
from proxy-gold criteria, not the RubricBench hard-gold holdout.
The report records input/output SHA256 hashes and the converted record count,
so stale GRPO parquet files can be detected before training.

The produced records follow the planned RLVR reward-side format:

```json
{
  "prompt": "为以下query生成评估rubric(原子化/可判yes-no/去冗余):\n{query}",
  "gold_rubrics": ["..."],
  "data_source": "rubricbench",
  "ground_truth": {"gold_rubrics": ["..."]},
  "extra_info": {
    "gold_rubrics": ["..."],
    "query": "{query}",
    "data_source": "rubricbench"
  }
}
```

Keep RubricBench hard-gold reserved for the primary BSC evaluation unless you
are running a deliberately labeled leakage/upper-bound diagnostic. The default
GRPO templates point to `data/processed/proxy_gold_verl.parquet`.

## Download Public Benchmarks

Use presets for common downstream validation sets:

```bash
python3 scripts/download_public_data.py \
  --preset rewardbench \
  --limit 100

python3 scripts/download_public_data.py \
  --preset rewardbench2 \
  --limit 100
```

Or download any HF dataset split explicitly:

```bash
python3 scripts/download_public_data.py \
  --hf-dataset ScalerLab/JudgeBench \
  --split test \
  --output data/raw/judgebench_test.jsonl
```

If an official release is a direct JSON/JSONL URL, use the same downloader:

```bash
python3 scripts/download_public_data.py \
  --url https://example.org/official_dataset.jsonl \
  --output data/raw/official_dataset.jsonl
```

For RubricBench/ResearchRubrics, use the official HF/GitHub release path once
available and save it as JSONL with at least `query` and `gold_rubrics` fields.
In `configs/data_sources_real.template.json`, keep these sources as `manual`
until the official release URL is known; then switch the source to `type: "url"`
with `url` and `output` fields to make acquisition reproducible.

For a reproducible real-data setup plan, generate a data pipeline from the
template:

```bash
python3 scripts/init_data_source_local_config.py \
  --template configs/data_sources_real.template.json \
  --output configs/data_sources_real.local.json \
  --report-json outputs/data_sources/local_config_init.json \
  --report-md outputs/data_sources/local_config_init.md

python3 scripts/build_data_pipeline.py \
  --config configs/data_sources_real.local.json \
  --pipeline-output configs/pipeline_data_real.generated.json \
  --manifest-output configs/manifest_data_real.generated.json
```

Before running downloads or normalization, build the data-source report:

```bash
python3 scripts/build_data_source_report.py \
  --config configs/data_sources_real.local.json \
  --output-json outputs/data_sources/source_report.json \
  --output-md outputs/data_sources/source_report.md
```

For the minimal RubricBench motivation experiment, the generated minimal-claim
pipeline uses scoped data-source gates so unrelated extension datasets do not
block the first BSC run:

```bash
python3 scripts/init_data_source_local_config.py \
  --template configs/data_sources_real.template.json \
  --output configs/data_sources_real.local.json \
  --report-json outputs/data_sources/local_config_init.json \
  --report-md outputs/data_sources/local_config_init.md \
  --required-dataset rubricbench

python3 scripts/build_data_source_report.py \
  --config configs/data_sources_real.local.json \
  --output-json outputs/data_sources/source_report.json \
  --output-md outputs/data_sources/source_report.md \
  --required-dataset rubricbench \
  --strict
```

This report separates manual hard-gold blockers from downloadable HF datasets.
If `configs/data_sources_real.local.json` is missing, the report is still
written with `overall_status=blocked` and next actions for creating the local
config. If the local JSON is malformed, the report is likewise written as
blocked with the syntax location. Fill official URLs and field mappings there;
do not edit generated pipeline JSON directly.
For hard-gold sources, the local config must bind the non-placeholder official
release URL and the raw file SHA256. Use `--fill-present-sha256
--update-existing` only after placing the official raw file at the configured
`raw_path`. The generated `local_config_init.json` is a Raw Audit Gate in the
real-run readiness and result-card configs; if it is missing or blocked, hard-gold
claims remain deferred.
If RubricBench is missing, keep the minimal BSC motivation claim deferred rather
than substituting toy or proxy evidence.

After normalization, run the hard-gold evidence hygiene gate before any BSC
claim:

```bash
python3 scripts/validate_gold_data.py \
  --input data/processed/rubricbench_gold.jsonl \
  --min-records 100 \
  --min-rubrics-per-query 1 \
  --forbidden-data-source toy \
  --forbidden-data-source proxy \
  --output-json outputs/data_validation/rubricbench_gold.json \
  --output-md outputs/data_validation/rubricbench_gold.md \
  --strict
```

Treat this as a Raw Audit Gate: if it fails, do not report RubricBench BSC as a
paper result yet.

The holdout contamination audit recursively inspects visible query/prompt fields
and nested metadata such as `extra_info.query`. This matters for
`proxy_gold_verl.parquet`, where the visible prompt is formatted for training
but the original query is stored in reward metadata and must still be checked
against RubricBench `test_main` and downstream holdouts.

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_data_real.generated.json \
  --dry-run
```

## Assemble The Full Real Run

After regenerating the data and matrix configs, assemble the full real-run
pipeline so preflight, budget estimation, model generation, validation, BSC,
training command generation, downstream, evidence, readiness, dashboard, and
result card stay in one audited execution graph:

```bash
python3 scripts/build_real_run_pipeline.py \
  --config configs/real_run_assembly.template.json \
  --pipeline-output configs/pipeline_real_run.generated.json \
  --manifest-output configs/manifest_real_run.generated.json

python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_real_run.generated.json \
  --dry-run
```

The generated manifest includes the data manifest, main matrix manifest,
domain-scoped model rubric files (`rubricbench`, `rewardbench`, `healthbench`,
and `writingbench`), rubric validation reports,
`outputs/training_commands/{run_sft.sh,run_grpo.sh,training_manifest.json}`,
Evidence Matrix, submission readiness, rebuttal pack, submission gap report,
dashboard, and Result Card outputs. The
pipeline generates auditable training scripts from
`configs/training_commands.example.json`; copy it to a local config only if you
need machine-specific command overrides. It does not automatically launch
GPU-heavy SFT or GRPO training. Run those scripts explicitly on the training
machine after preflight passes.

The real-run pipeline includes a strict manual gate before model generation.
After SFT/GRPO finish and the checkpoints are served through
`configs/generators.local.jsonl`, write
`outputs/training_commands/training_done.json` with the checkpoint paths,
serving URLs, operator, date, and the bound training inputs/reward hook. The gate
validates required JSON keys and exact values for the config/data/reward fields,
so use `outputs/training_commands/training_done.template.json` as the starting
point. The completed file should contain at least:

```json
{
  "sft_checkpoint": "outputs/checkpoints/evaluation_criteria_policy_sft",
  "rl_checkpoint": "outputs/checkpoints/evaluation_criteria_policy_rl",
  "served_methods": ["base", "sft_only", "sft_rl"],
  "served_generators": ["base", "sft_only", "sft_rl"],
  "sft_config": "configs/llamafactory_sft.local.yaml",
  "grpo_config": "configs/verl_grpo_bsc.local.yaml",
  "sft_data": "data/processed/blindspot_sft.jsonl",
  "rl_data": "data/processed/proxy_gold_verl.parquet",
  "rl_data_report": "outputs/sft_data/proxy_gold_verl_report.json",
  "reward_function": "src/blindspot_rl/verl_reward.py:compute_score",
  "operator": "name",
  "date": "YYYY-MM-DD"
}
```

`served_methods` and the compatibility field `served_generators` must include
all three names: `base`, `sft_only`, and `sft_rl`. Without that file, those
keys, the exact config/data/reward values, the required method/provider names,
and the expected checkpoint directories,
`check_manual_gate.py` stops the pipeline before any
`sft_only`/`sft_rl` rubrics can enter the main matrix.
The same gate validates `outputs/sft_data/proxy_gold_verl_report.json` against
the current `data/processed/proxy_gold.jsonl` and
`data/processed/proxy_gold_verl.parquet` SHA256 digests, so stale GRPO data or
a stale conversion report blocks trained-method claims.
It also requires `outputs/verifier/teacher_rubrics_filtered_report.json`, which
binds the raw teacher criteria file, filtered teacher criteria file,
Meta-Verifier provider, SFT-data preflight report, and meta-verifier budget
report by SHA256. The real C0 gate compares that report's `output_sha256`
against `proxy_gold_build_report.json:input_sha256`, so proxy-gold construction
cannot silently consume stale or unaudited verifier-filtered criteria.


RubricBench and ResearchRubrics are marked as manual sources in the template:
place their official releases at the configured `data/raw/*_raw.jsonl` paths,
then run the generated pipeline. RewardBench and JudgeBench use HF download
stages. The generated pipeline also writes schema profiles under
`outputs/data_profiles/`; inspect those files if normalization returns zero
rows, then set `query_key`, `gold_key`, `chosen_key`, or `rejected_key` in the
data source config. Nested paths such as `messages.0.content` are supported.

## Normalize Raw Datasets

After downloading raw data, normalize it into one of the project schemas.

Gold data for BSC/RLVR:

```bash
python3 scripts/normalize_dataset.py \
  --input data/raw/rubricbench_raw.jsonl \
  --output data/processed/rubricbench_gold.jsonl \
  --target gold \
  --data-source rubricbench \
  --dedupe-query
```

Query pool for teacher generation:

```bash
python3 scripts/normalize_dataset.py \
  --input data/raw/rubricbench_raw.jsonl \
  --output data/processed/rubricbench_queries.jsonl \
  --target query_pool \
  --data-source rubricbench \
  --dedupe-query
```

Preference data for downstream validation:

```bash
python3 scripts/normalize_dataset.py \
  --input data/raw/rewardbench_filtered.jsonl \
  --output data/processed/rewardbench_pref.jsonl \
  --target preference \
  --data-source rewardbench
```

If a dataset uses unusual field names, pass `--query-key`, `--gold-key`,
`--chosen-key`, or `--rejected-key` explicitly.

## Build SFT Data From Multi-Teacher Criteria

Input teacher JSONL can mix JSON criteria and markdown bullets:

```json
{"query": "...", "teacher": "gpt-4o", "rubrics": ["..."]}
{"query": "...", "teacher": "claude", "response": "- criterion ..."}
```

Run the builder:

```bash
python3 scripts/build_sft_data.py \
  --input examples/toy_teacher_rubrics.jsonl \
  --sft-output data/processed/toy_sft.jsonl \
  --proxy-gold-output data/processed/toy_proxy_gold.jsonl \
  --stats-output outputs/toy_sft_stats.jsonl \
  --embedding-model token-overlap \
  --dedupe-tau 0.9
```

For real SFT data, switch `--embedding-model` to the default BGE model and feed
GPT-4o/Claude/DeepSeek/Qwen teacher outputs. The SFT output uses:

```json
{"instruction": "...", "input": "{query}", "output": "[rubric json list]"}
```

The proxy-gold output can be passed into `convert_to_verl_parquet.py` for RLVR.

Register the SFT file for LLaMA-Factory:

```bash
python3 scripts/register_llamafactory_dataset.py \
  --dataset-info data/processed/dataset_info.json \
  --name blindspot_sft \
  --file-name blindspot_sft.jsonl \
  --overwrite
```

On a separate training machine, point `--dataset-info` to the LLaMA-Factory
repository's `data/dataset_info.json` instead.

## SFT And GRPO Templates

For SFT, copy and edit:

```bash
cp configs/llamafactory_sft.example.yaml configs/llamafactory_sft.local.yaml
```

The expected SFT file is a LLaMA-Factory JSONL like:

```json
{"instruction": "...", "input": "{query}", "output": "[rubric json list]"}
```

For GRPO/RLVR, copy and edit:

```bash
cp configs/verl_grpo_bsc.example.yaml configs/verl_grpo_bsc.local.yaml
```

The reward hook is [src/blindspot_rl/verl_reward.py](src/blindspot_rl/verl_reward.py).
It requires non-empty `gold_rubrics` from the reward-side data contract and can
resolve that gold from top-level records, `ground_truth`, or `extra_info`;
missing or unparseable gold receives the strong bad-format reward instead of a
positive validity-only score. The validity term uses the fail-closed rule
verifier by default (`BSC_VERIFIER=rule`) and can consume response-aligned
`valid_flags` when a caller supplies already verified generations.
Useful ablation environment variables:

```bash
export BSC_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
export BSC_VERIFIER=rule
export BSC_W_COV=1.0
export BSC_W_VALID=0.5
export BSC_W_RED=0.5
```

Set `BSC_W_RED=0.0` for the no-redundancy ablation,
`BSC_W_VALID=0.0 BSC_VERIFIER=none` for the no-validity ablation, and
`BSC_VERIFIER=none` with the default validity/redundancy weights for the
no-verifier ablation. The coverage-only ablation sets both
`BSC_W_VALID=0.0` and `BSC_W_RED=0.0` with `BSC_VERIFIER=none`.

Generate runnable training scripts:

```bash
python3 scripts/make_training_commands.py \
  --config configs/training_commands.example.json \
  --output-dir outputs/training_commands
```

This writes `run_sft.sh`, `run_grpo.sh`, `run_grpo_no_red.sh`,
`run_grpo_no_valid.sh`, `run_grpo_no_verifier.sh`, `run_grpo_cov_only.sh`,
`training_done.template.json`, `training_manifest.json`, and
`outputs/reward_component_training_ablation/training_done.template.json`.
Run SFT first, then update the GRPO model path if your checkpoint directory
differs. For machine-specific paths, copy the example config to an untracked
local file and pass that path with `--config`.

After the trained reward-component ablations finish, fill
`outputs/reward_component_training_ablation/training_done.json` from the
template. The C7 evidence gate uses that file plus the hard-gold BSC summaries
under
`outputs/reward_component_training_ablation/{no_red,no_valid,no_verifier,cov_only}/`
to distinguish real trained ablations from offline reward re-scoring.

For each trained reward-component variant, generate evaluation criteria for
RubricBench `test_main` from the served checkpoint and evaluate them with the
same hard-gold BSC protocol:

```bash
python3 scripts/generate_model_rubrics.py \
  --input data/processed/splits/rubricbench_gold_test_main.jsonl \
  --providers configs/generators_reward_ablation.local.jsonl \
  --output data/processed/reward_component_training_ablation/no_red/model_rubrics.jsonl \
  --data-source rubricbench \
  --resume

python3 scripts/prepare_bsc_eval.py \
  --gold data/processed/splits/rubricbench_gold_test_main.jsonl \
  --predictions data/processed/reward_component_training_ablation/no_red/model_rubrics.jsonl \
  --output data/processed/reward_component_training_ablation/no_red/bsc_eval.jsonl \
  --report outputs/reward_component_training_ablation/no_red/bsc_join_report.json \
  --model no_red \
  --data-source rubricbench \
  --min-joined 100

python3 scripts/bsc_diagnose.py \
  --input data/processed/reward_component_training_ablation/no_red/bsc_eval.jsonl \
  --embedding-model BAAI/bge-large-en-v1.5 \
  --coverage-tau 0.75 \
  --redundancy-tau 0.85 \
  --output-dir outputs/reward_component_training_ablation/no_red/bsc
```

Repeat the same pattern for `no_valid`, `no_verifier`, and `cov_only`, changing
the method name and output directories. The provider file should expose model
names matching the variant names so `prepare_bsc_eval.py --model {variant}`
joins the intended records.

## Elicit Teacher Criteria With APIs

Copy the provider template and set the matching API key environment variables:

```bash
cp configs/providers.example.jsonl configs/providers.local.jsonl
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...
```

Then generate teacher candidates:

```bash
python3 scripts/estimate_api_budget.py \
  --input data/raw/query_pool.jsonl \
  --providers configs/providers.local.jsonl \
  --output outputs/api_budget/teacher_rubrics_budget.json \
  --output-md outputs/api_budget/teacher_rubrics_budget.md \
  --strict

python3 scripts/generate_teacher_rubrics.py \
  --input data/raw/query_pool.jsonl \
  --providers configs/providers.local.jsonl \
  --output data/processed/teacher_rubrics.jsonl \
  --require-budget-report outputs/api_budget/teacher_rubrics_budget.json \
  --resume \
  --sleep 0.5
```

The output schema is compatible with `build_sft_data.py`.

## Generate Method Criteria For Main Tables

Teacher criteria are for SFT/proxy-gold construction. For the paper comparison
table, generate one criteria file per method or one mixed file with a `model`
filterable method name. The real-run templates use domain-scoped files so BSC,
downstream, and generalization evidence cannot accidentally read the wrong
query pool. The real-run preflight also requires `configs/generators.local.jsonl`
to contain all main-matrix provider/method names: `base`, `gpt4o`, `claude`,
`sft_only`, and `sft_rl`.

```bash
cp configs/generators.example.jsonl configs/generators.local.jsonl
export LOCAL_OPENAI_API_KEY=dummy
python3 scripts/estimate_api_budget.py \
  --input data/processed/rubricbench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --output outputs/api_budget/rubricbench_model_rubrics_budget.json \
  --output-md outputs/api_budget/rubricbench_model_rubrics_budget.md \
  --strict

python3 scripts/generate_model_rubrics.py \
  --input data/processed/rubricbench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --output data/processed/rubricbench_model_rubrics.jsonl \
  --data-source rubricbench_generation \
  --require-budget-report outputs/api_budget/rubricbench_model_rubrics_budget.json \
  --resume \
  --sleep 0.5

python3 scripts/generate_model_rubrics.py \
  --input data/processed/rewardbench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --output data/processed/rewardbench_model_rubrics.jsonl \
  --data-source rewardbench_generation \
  --require-budget-report outputs/api_budget/rewardbench_model_rubrics_budget.json \
  --resume \
  --sleep 0.5

python3 scripts/generate_model_rubrics.py \
  --input data/processed/judgebench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --output data/processed/judgebench_model_rubrics.jsonl \
  --data-source judgebench_generation \
  --require-budget-report outputs/api_budget/judgebench_model_rubrics_budget.json \
  --resume \
  --sleep 0.5

python3 scripts/generate_model_rubrics.py \
  --input data/processed/rewardbench2_queries.clean.jsonl \
  --providers configs/generators.local.jsonl \
  --output data/processed/rewardbench2_model_rubrics.jsonl \
  --data-source rewardbench2_generation \
  --require-budget-report outputs/api_budget/rewardbench2_model_rubrics_budget.json \
  --resume \
  --sleep 0.5

python3 scripts/generate_model_rubrics.py \
  --input data/processed/rmbench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --output data/processed/rmbench_model_rubrics.jsonl \
  --data-source rmbench_generation \
  --require-budget-report outputs/api_budget/rmbench_model_rubrics_budget.json \
  --resume \
  --sleep 0.5
```

Run the same `estimate_api_budget.py` command for each model-rubric domain,
changing `--input`, `--output`, and `--output-md` to match the corresponding
`--require-budget-report` path before issuing API calls.

Each row contains `query`, `method`, `model`, `model_name_or_path`,
`response`, and parsed `rubrics`. The `model` field is intentionally the method
label, so downstream commands can select a method with `--model sft_rl`.

## Filter With Meta-Verifier

For real Hall/R_valid numbers, use API mode:

```bash
cp configs/verifier.example.jsonl configs/verifier.local.jsonl
python3 scripts/estimate_api_budget.py \
  --input data/processed/teacher_rubrics_raw.jsonl \
  --providers configs/verifier.local.jsonl \
  --unit-field rubrics \
  --output outputs/api_budget/meta_verifier_budget.json \
  --output-md outputs/api_budget/meta_verifier_budget.md \
  --strict

python3 scripts/filter_rubrics_with_verifier.py \
  --input data/processed/teacher_rubrics_raw.jsonl \
  --output data/processed/teacher_rubrics_filtered.jsonl \
  --stats-output outputs/verifier_stats.jsonl \
  --report-output outputs/verifier/teacher_rubrics_filtered_report.json \
  --mode api \
  --provider configs/verifier.local.jsonl \
  --require-budget-report outputs/api_budget/meta_verifier_budget.json \
  --require-preflight-report outputs/preflight/sft_data_preflight.json \
  --sleep 0.2
```

The report is part of the C0 data-isolation evidence chain: it records the raw
teacher input SHA256, filtered teacher output SHA256, verifier provider SHA256,
meta-verifier budget SHA256, SFT preflight SHA256, and filtering counts. The
next proxy-gold build report must consume the same filtered-teacher SHA256.

For local smoke tests, use deterministic rule mode:

```bash
python3 scripts/filter_rubrics_with_verifier.py \
  --input examples/toy_teacher_rubrics.jsonl \
  --output data/processed/toy_teacher_filtered.jsonl \
  --stats-output outputs/toy_verifier_stats.jsonl \
  --mode rule
```

## One-Command Smoke Pipeline

Run the full local pipeline:

```bash
bash pipelines/run_smoke_pipeline.sh
```

It verifies the engineering chain:
teacher criteria → verifier filtering → SFT/proxy gold → verl data → BSC →
downstream accuracy → main table → evidence matrix → paper artifacts.

The shell entrypoint is backed by a JSON stage config:

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_smoke.example.json \
  --dry-run
```

For real experiments, copy `configs/pipeline_smoke.example.json`, replace toy
paths with RubricBench/RewardBench paths and API providers, then run the full
pipeline or a subset:

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_real.local.json \
  --only prepare_bsc \
  --only bsc
```

## Multi-Method Experiment Matrix

For the paper main table, define methods once and generate a pipeline plus
matching audit manifest:

```bash
python3 scripts/build_experiment_matrix.py \
  --methods configs/methods_matrix_smoke.example.json \
  --pipeline-output configs/pipeline_matrix_smoke.generated.json \
  --manifest-output configs/manifest_matrix_smoke.generated.json

python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_matrix_smoke.generated.json \
  --dry-run
```

For the real table, generate `data/processed/rubricbench_model_rubrics.jsonl`
for BSC and `data/processed/rewardbench_model_rubrics.jsonl` for downstream
utility, then copy `configs/methods_matrix_real.template.json`. JudgeBench is
kept as a separate difficult-case downstream extension via
`configs/methods_matrix_judgebench.template.json` and
`data/processed/judgebench_model_rubrics.jsonl`. Both templates use
`model_filter` to split `base`, `gpt4o`, `claude`, `sft_only`, and `sft_rl`
within each domain-scoped file.

RewardBench-2 is evaluated through the dedicated multi-candidate downstream
pipeline in `configs/methods_matrix_rewardbench2.template.json`. RM-Bench is
still tracked as generation readiness until its official style-bias schema is
mapped to a dedicated evaluator.

When `common.bootstrap_ci.enabled` is true in the methods config, the generated
matrix pipeline also runs bootstrap CIs for each method's BSC and downstream
per-item CSVs. Those CI JSON reports are passed into `summarize_experiments.py`,
so `main_table.md` and exported LaTeX tables display point estimates with
intervals automatically.

When `common.bsc_sweep.enabled` is true, the matrix pipeline also adds
`sweep_bsc_thresholds.py` stages for the selected methods. The real template
uses this for the `base` and `sft_rl` endpoints so the robustness claim has
audited `threshold_sweep.{csv,json,md}` artifacts.
BSC summaries also report `mean_n_gen`, `mean_n_gold`, and `gen_to_gold_ratio`;
C3 uses these fields to keep a coverage change metric-only when it is mainly a
generated criteria length expansion.

When `common.reward_ablation.enabled` is true, the matrix pipeline runs
`run_bsc_ablation.py` before summarization/export and audits the `full`,
`no_red`, `no_valid`, `no_verifier`, and `cov_only` variant summaries. The real
template runs this on the `sft_rl` endpoint. This is an offline reward
re-scoring diagnostic: it changes the reward weights on the same model outputs,
so it must not be reported as a trained reward-component ablation by itself.
Paper-facing no-redundancy, no-validity, no-verifier, and coverage-only
ablation claims additionally require separately trained variants under
`outputs/reward_component_training_ablation/`, each evaluated on RubricBench
`test_main` with the same BGE/threshold/verifier-backed protocol as the main
hard-gold table.

When `common.teacher_union_ablation.enabled` is true, the matrix pipeline runs
`run_teacher_union_ablation.py` before summarization/export and passes the
resulting CSV into paper artifact export. This covers the single-teacher versus
multi-teacher union ablation required by the method section. In the real run,
this ablation is seed-scoped: it uses RubricBench `train_seed` human-gold
dimensions, not the `test_main` holdout.

When the dimension-transition audit is enabled, the matrix pipeline compares
`base -> sft_only` and `base -> sft_rl` after all method-level BSC diagnostics
and before summarization/export. The real template writes
`transition_summary.json`, `transition_per_item.csv`,
`transition_by_category.csv`, and `transition_gold_items.jsonl` under
`outputs/matrix_real/dimension_transition/`, then passes the summaries into
paper artifact export for `dimension_transition_table.tex`. This is the audit
table used to test when a training-based dimension-level recovery statement is
warranted, rather than only observing aggregate BSC movement. Any
dimension-recovery claim should remain deferred until the hard-gold evidence
gate passes for the relevant trained methods. The C12 gate also binds the
transition-summary input SHA256 values to the corresponding BSC join-report
`output_sha256` fields, requires exact query alignment with no duplicate
records, and checks that the gold-dimension audit rows match the reported
total-gold count before the table can support a dimension-level recovery
statement.

When `common.semantic_space.enabled` is true, the matrix pipeline runs
`build_semantic_space_visualization.py` over the selected method-level
`bsc_eval.jsonl` files. The real template compares `base`, `sft_only`, and
`sft_rl`, writes `semantic_space.svg`, `semantic_space.pdf`,
`semantic_space_points.csv`, and `semantic_space_summary.json` under
`outputs/matrix_real/semantic_space/`, and exports those assets into the paper
artifact bundle. The script supports UMAP/t-SNE projections and records the
requested and actual projection in the JSON summary; if an optional projection
dependency is unavailable, the actual projection records an audited deterministic
PCA fallback. The SVG/PDF pair is the audit figure for the semantic-spread claim;
the CSV/JSON files make every plotted evaluation dimension inspectable,
including point-level nearest-gold category, similarity, and source text for
each generated criterion. C13 validates the exact point CSV schema and requires
generated rows, including SFT+GRPO rows, to carry non-empty nearest-gold id,
category, similarity, and text fields. The JSON summary also records
nearest-gold cluster distributions and normalized cluster entropy by method, so
the visualization claim remains illustrative unless SFT+GRPO avoids local
collapse relative to SFT-only as well as matching the nearest-gold coverage
checks.

The real run also has a separate cross-domain generalization matrix:

```bash
python3 scripts/build_experiment_matrix.py \
  --methods configs/methods_matrix_generalization.template.json \
  --pipeline-output configs/pipeline_matrix_generalization.generated.json \
  --manifest-output configs/manifest_matrix_generalization.generated.json
```

This matrix evaluates `base` and `sft_rl` on HealthBench and WritingBench under
method names such as `healthbench_base` and `writingbench_sft_rl`. The template
expects `data/processed/{healthbench,writingbench}_proxy_gold.jsonl`; label those
results as proxy-domain generalization unless you replace the files with audited
hard-gold domain evaluation dimensions. The generated generalization matrix is
BSC-only by default; it does not require HealthBench/WritingBench
chosen-vs-rejected preference files.

The full real-run assembly now generates teacher and method evaluation criteria
for RubricBench, HealthBench, and WritingBench. Teacher outputs are filtered by
the Meta-Verifier, then `build_sft_data.py` creates:

- `data/processed/blindspot_sft.jsonl` for SFT.
- `data/processed/proxy_gold.jsonl` and `data/processed/proxy_gold_verl.parquet`
  for proxy RL/query training data.
- `data/processed/healthbench_proxy_gold.jsonl` and
  `data/processed/writingbench_proxy_gold.jsonl` for the cross-domain BSC
  generalization matrix.

## Audit Experiment Artifacts

Use the manifest to catch missing outputs before writing tables:

```bash
python3 scripts/audit_experiment.py \
  --manifest configs/experiment_manifest.example.json \
  --output outputs/audit_report.json
```

For real experiments, copy the manifest and replace toy paths with
`base/gpt4o/claude/sft/rl` result paths.

## Build A Claim-Evidence Matrix

Before writing claims into the paper, map each claim to concrete artifacts and
metric gates:

```bash
python3 scripts/build_evidence_matrix.py \
  --config configs/evidence_matrix_smoke.example.json \
  --output-dir outputs/evidence
```

This writes `evidence_matrix.json`, `evidence_matrix.csv`, and
`evidence_matrix.md`. Treat `safe_to_claim` rows as draftable, and treat
`missing_evidence` or `contradicted` rows as experiment TODOs rather than paper
claims.
The real downstream gates also use `table_values` checks on `main_table.csv`
rows, so C4/C9/C10 require the final exported method rows to carry
`downstream_status=pass` and `downstream_paper_claim_eligible=true`, not only
paper-eligible source `summary.json` files.

For real paper claims, start from the stricter template:

```bash
cp configs/evidence_matrix_real.template.json configs/evidence_matrix_real.local.json
python3 scripts/build_evidence_matrix.py \
  --config configs/evidence_matrix_real.local.json \
  --output-dir outputs/evidence_real
```

The evidence matrix supports both scalar metric gates and cross-file
comparisons such as `sft_rl.mean_coverage - base.mean_coverage >= 0.05`.

## Downstream Preference Validation

After a method produces evaluation criteria for RewardBench/JudgeBench prompts,
evaluate whether criterion-based scoring ranks `chosen` above `rejected`:

```bash
python3 scripts/prepare_downstream_eval.py \
  --preferences data/processed/rewardbench_pref.jsonl \
  --rubrics data/processed/base_model_rubrics.jsonl \
  --output data/processed/base_rewardbench_eval.jsonl \
  --report outputs/base_rewardbench_join_report.json \
  --model base \
  --data-source rewardbench

python3 scripts/evaluate_downstream.py \
  --input data/processed/base_rewardbench_eval.jsonl \
  --output-dir outputs/base_downstream
```

The smoke-test scorer is `KeywordRubricScorer`, which is deterministic and
dependency-free. For paper numbers, replace it with an API/LLM scorer that
judges whether each answer satisfies each rubric item. Downstream summaries set
`paper_claim_eligible=false` for keyword smoke runs; evidence gates require
`paper_claim_eligible=true` from API-scored runs with a bound provider and
budget report. `scripts/summarize_experiments.py` only exposes downstream
accuracy/tie/margin metrics when that flag is true, and
`scripts/export_paper_artifacts.py` blocks downstream table rows whose
`downstream_status` is not `pass` or whose
`downstream_paper_claim_eligible` field is not `true`.

API scorer example:

```bash
cp configs/judge_scorer.example.jsonl configs/judge_scorer.local.jsonl
python3 scripts/estimate_api_budget.py \
  --input data/processed/base_rewardbench_eval.jsonl \
  --providers configs/judge_scorer.local.jsonl \
  --unit-field rubrics \
  --calls-per-record-per-provider 2 \
  --output outputs/rewardbench_downstream_api_budget.json \
  --output-md outputs/rewardbench_downstream_api_budget.md \
  --strict

python3 scripts/evaluate_downstream.py \
  --input data/processed/base_rewardbench_eval.jsonl \
  --output-dir outputs/rewardbench_downstream_api \
  --scorer api \
  --provider configs/judge_scorer.local.jsonl \
  --require-budget-report outputs/rewardbench_downstream_api_budget.json \
  --sleep 0.2
```

Expected outputs:

- `outputs/toy_downstream/summary.json`
- `outputs/toy_downstream/per_item.csv`

Paper-facing downstream summaries record `input_sha256`,
`scorer_provider_sha256`, and the validated `budget_contract`. The Evidence
Matrix compares those summary hashes with `contract.input_sha256` and
`contract.providers_sha256` from the budget report, so a stale budget report
with the right path but the wrong file contents cannot satisfy C4/C9/C10.

## Build The Main Experiment Table

Merge BSC and downstream summaries by method name:

```bash
python3 scripts/summarize_experiments.py \
  --bsc base=outputs/base_bsc/summary.json \
  --downstream base=outputs/base_downstream/summary.json \
  --bsc-ci base=outputs/base_bsc_ci/bootstrap_ci.json \
  --downstream-ci base=outputs/base_downstream_ci/bootstrap_ci.json \
  --output-csv outputs/main_table.csv \
  --output-md outputs/main_table.md
```

The markdown table contains the core paper columns: `Cov`, `Blind`, `Red`,
`Hall`, and downstream `Acc`. CI arguments are optional; when present, point
estimate columns remain numeric in CSV, and additional `*_ci` columns are used
for Markdown/LaTeX display.

## Bootstrap Confidence Intervals

After BSC/downstream per-item files are generated, compute bootstrap confidence
intervals for paper metrics:

```bash
python3 scripts/bootstrap_metric_ci.py \
  --input outputs/base_bsc/per_item.csv \
  --metric coverage \
  --metric blind \
  --metric redundancy \
  --metric hallucination \
  --n-boot 1000 \
  --seed 13 \
  --confidence 0.95 \
  --output-json outputs/base_bsc_ci/bootstrap_ci.json \
  --output-csv outputs/base_bsc_ci/bootstrap_ci.csv \
  --output-md outputs/base_bsc_ci/bootstrap_ci.md
```

For downstream preference accuracy, run the same script over
`outputs/*_downstream/per_item.csv` with `--metric correct --metric tie
--metric margin`. Result Cards can display these CI reports when their config
includes `confidence_intervals`. Evidence Matrix metric paths also support
selectors such as `metrics[metric=blind].ci_lower`, which lets claim gates use
CI lower bounds instead of only point estimates.

## Export Paper Artifacts

After generating the hard-gold main table, downstream holdout tables, ablations,
dimension-transition audits, semantic-space artifacts, and Evidence Matrix,
export a compact paper bundle:

```bash
python3 scripts/export_paper_artifacts.py \
  --main-table-csv outputs/matrix_real/main_table.csv \
  --main-table-md outputs/matrix_real/main_table.md \
  --downstream-table-csv RewardBench=outputs/matrix_real/main_table.csv \
  --downstream-table-csv JudgeBench=outputs/matrix_judgebench/main_table.csv \
  --downstream-table-csv RewardBench-2=outputs/matrix_rewardbench2/main_table.csv \
  --ablation-csv outputs/bsc_ablation/ablation_summary.csv \
  --teacher-union-csv outputs/teacher_union_ablation/teacher_union_ablation.csv \
  --verifier-filter-csv outputs/verifier_filter_ablation/verifier_filter_ablation.csv \
  --transition-summary-json outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json \
  --transition-summary-json outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json \
  --semantic-space-dir outputs/matrix_real/semantic_space \
  --audit-report outputs/matrix_real/audit_report.json \
  --evidence-json outputs/evidence_real/evidence_matrix.json \
  --evidence-csv outputs/evidence_real/evidence_matrix.csv \
  --evidence-md outputs/evidence_real/evidence_matrix.md \
  --output-dir outputs/paper_artifacts
```

This writes LaTeX tables (`main_table.tex`, `downstream_utility_table.tex`,
`ablation_table.tex`, `teacher_union_ablation_table.tex`,
`verifier_filter_ablation_table.tex`, `dimension_transition_table.tex`) and
copies C13 semantic-space artifacts only when their provenance contract passes.

Sync exported artifacts into the manuscript scaffold:

```bash
python3 scripts/sync_paper_artifacts.py \
  --artifacts-dir outputs/paper_artifacts \
  --paper-dir paper
```

The manuscript entrypoint is `paper/main.tex`. Synced tables are placed in
`paper/tables/`, and evidence/Result Card docs are indexed in
`paper/asset_index.md` with SHA256 fingerprints. Verify the index before using
these files in a submission draft; the checker also fails if the asset index
declares unresolved blockers:

```bash
python3 scripts/check_paper_asset_index.py \
  --asset-index paper/asset_index.md \
  --output outputs/paper_artifacts/paper_asset_index_check.json \
  --output-md outputs/paper_artifacts/paper_asset_index_check.md \
  --strict
```

For the generated minimal-claim pipeline this check is inserted after
`sync_paper`, after `sync_result_card` as
`paper_asset_index_check_post_sync` (non-strict for blocked-report refreshes),
and finally as `paper_asset_index_check_final` with `--strict` as the submission
hard gate. In the real-run pipeline, the Result Card stage itself is strict, so
a blocked downstream/evidence decision stops before final result-card sync. The
minimal API handoff exposes the same blocked refresh path as
`commands.refresh_blocked_reports`:

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_minimal_claim.generated.json \
  --from-stage audit \
  --to-stage paper_asset_index_check_post_sync
```

Before writing claims into a submission draft, generate a readiness report:

```bash
python3 scripts/check_submission_readiness.py \
  --audit-report outputs/matrix_real/audit_report.json \
  --evidence-matrix outputs/evidence_real/evidence_matrix.json \
  --paper-dir paper \
  --output-json outputs/submission_readiness/readiness_report.json \
  --output-md outputs/submission_readiness/readiness_report.md
```

The readiness report combines artifact audit status, claim-evidence status, and
manuscript/table sync status. Treat hard blockers as submission TODOs.

To turn readiness blockers into a phase-ordered execution checklist, build the
submission gap report:

```bash
python3 scripts/build_submission_gap_report.py \
  --readiness-report outputs/submission_readiness/readiness_report.json \
  --evidence-matrix outputs/evidence_real/evidence_matrix.json \
  --rebuttal-manifest outputs/rebuttal_pack/rebuttal_pack_manifest.json \
  --output-dir outputs/submission_readiness/gap_report
```

The report groups missing evidence into data isolation, SFT/GRPO training,
hard-gold BSC evaluation, downstream utility, ablations, semantic-space
visualization, paper-readiness, and reviewer-facing rebuttal-readiness phases.
It also emits a top-level `Claim Ladder Status` table, using the same
motivation / metric-support / method-support / judge-utility support gates as
the paper, so operator handoff can tell which manuscript conclusion layer is
still blocked. Use this report to decide the next real artifact to produce; do
not write a claim until the corresponding phase, claim-ladder layer, and
required evidence row are unblocked.

## Build The 20-Day Sprint Plan

Generate a day-by-day runbook that links commands, artifacts, evidence gates,
and claim-ladder milestones:

```bash
python3 scripts/build_sprint_plan.py \
  --config configs/sprint_plan_20day.template.json \
  --output-dir outputs/sprint_plan
```

This writes `sprint_plan.md`, `sprint_plan.csv`, and `sprint_plan.json`.
Use it as the execution checklist from data readiness through submission
readiness. Each day records the manuscript conclusion layer it can unlock, and
the Markdown includes the same four-level claim ladder and downgrade rules used
by the paper, gap report, dashboard, Result Card, and rebuttal pack.

## Build Rebuttal Pack

Generate reviewer-facing Q&A readiness from the Evidence Matrix:

```bash
python3 scripts/build_rebuttal_pack.py \
  --evidence-matrix outputs/evidence_real/evidence_matrix.json \
  --readiness-report outputs/submission_readiness/readiness_report.json \
  --output-dir outputs/rebuttal_pack
```

The rebuttal pack marks each reviewer concern as `answer_ready`,
`needs_readiness`, `needs_evidence`, `cannot_claim`, or
`missing_claim_mapping`. A `needs_readiness` entry means the matched Evidence
Matrix rows are safe, but the submission readiness report is still blocked, so
the entry remains a draft response. Its Markdown and
manifest also expose `Claim Ladder Status`, so a reviewer answer cannot outrun the strongest
currently supported manuscript conclusion layer.
In the full real-run assembly, `rebuttal_pack_real` runs after submission
readiness and before the dashboard/Result Card, followed by
`submission_gap_report_real` with the rebuttal manifest attached. The final
paper sync indexes
`outputs/dashboard/real_run_dashboard.{json,md}`,
`outputs/submission_readiness/gap_report/submission_gap_report.{json,md}`,
`outputs/rebuttal_pack/rebuttal_pack.{json,md}`, and
`outputs/rebuttal_pack/rebuttal_pack_manifest.json` with SHA256 fingerprints.
The manifest binds the pack to the exact Evidence Matrix and readiness report
used to generate it. Submission readiness also checks that the dashboard and
reviewer-facing pack JSON, Markdown, and manifest are synced into
`paper/asset_index/` before treating the paper package as ready. The standalone
command above is for local refreshes when you update evidence or readiness
reports outside the full pipeline.

## Reproducible Pilot Sampling

Before full API generation, create fixed-seed pilot subsets for the minimal
motivation run and downstream smoke:

```bash
python3 scripts/sample_records.py \
  --input data/processed/rubricbench_gold.jsonl \
  --output data/samples/rubricbench_gold_pilot100.jsonl \
  --n 100 \
  --seed 13 \
  --stratify-key data_source \
  --dedupe-key query \
  --report outputs/sampling/rubricbench_gold_pilot100.json \
  --report-md outputs/sampling/rubricbench_gold_pilot100.md
```

Use `configs/sample_real.template.json` to create RubricBench and RewardBench
pilot splits consistently across runs.

## Real-Run Preflight

Before launching API-heavy generation or training, run an offline preflight:

```bash
python3 scripts/preflight_real_run.py \
  --input data/processed/rubricbench_gold.jsonl \
  --input data/processed/rubricbench_queries.jsonl \
  --input data/processed/rewardbench_pref.jsonl \
  --providers configs/generators.local.jsonl \
  --providers configs/providers.local.jsonl \
  --providers configs/verifier.local.jsonl \
  --providers configs/judge_scorer.local.jsonl \
  --required-provider-in configs/generators.local.jsonl:base,gpt4o,claude,sft_only,sft_rl \
  --required-provider-in configs/providers.local.jsonl:gpt-4o,deepseek,qwen \
  --required-provider-in configs/verifier.local.jsonl:meta-verifier \
  --required-provider-in configs/judge_scorer.local.jsonl:judge-scorer \
  --training-config configs/training_commands.example.json \
  --required-env LOCAL_OPENAI_API_KEY \
  --required-env OPENAI_API_KEY \
  --required-env ANTHROPIC_API_KEY \
  --required-env DEEPSEEK_API_KEY \
  --required-env DASHSCOPE_API_KEY \
  --output outputs/preflight/real_run_preflight.json \
  --output-md outputs/preflight/real_run_preflight.md \
  --strict
```

This checks input presence/counts, provider schema, required provider names per
config file, API-key environment variables, local checkpoint path readiness, and
training config paths without calling any external API. The real-run template
uses strict mode, so any failed preflight check stops the pipeline before API
generation or training commands run.

## API Budget And Rate-Limit Estimate

Estimate calls, tokens, cost, and minimum wall-clock time before launching
multi-teacher or five-method criteria elicitation:

```bash
python3 scripts/estimate_api_budget.py \
  --input data/processed/rubricbench_queries.jsonl \
  --providers configs/generators.local.jsonl \
  --resume-output data/processed/rubricbench_model_rubrics.jsonl \
  --method-key method \
  --output outputs/api_budget/model_rubrics_budget.json \
  --output-md outputs/api_budget/model_rubrics_budget.md
```

Provider JSONL entries may include optional `qpm`, `tpm`,
`input_cost_per_1k`, and `output_cost_per_1k` fields. Existing resume outputs
are subtracted so the estimate reflects only pending query-method pairs. The
budget report contract records SHA256 for the input and provider files; paid
API stages reject stale budget reports if either file changes after estimation.

## Validate Generated Criteria

After API/model generation and before BSC or SFT data construction, validate the
criteria output shape:

```bash
python3 scripts/validate_rubric_outputs.py \
  --input data/processed/rubricbench_model_rubrics.jsonl \
  --output-dir outputs/validation/rubricbench_model_rubrics \
  --min-rubrics 3 \
  --max-rubrics 12
```

The report flags empty or unparsable outputs, too few/too many rubrics, exact
duplicates, token-overlap redundancy, and generic criteria such as
`helpful`/`clear`/`quality`.

## Build A Run Dashboard

Use the dashboard to summarize preflight, budget, confidence intervals, audit,
evidence, and readiness reports without rerunning experiments:

```bash
python3 scripts/build_run_dashboard.py \
  --config configs/dashboard_real.template.json \
  --output-json outputs/dashboard/real_run_dashboard.json \
  --output-md outputs/dashboard/real_run_dashboard.md
```

The dashboard reports an overall status (`pass`, `warn`, `missing`, or
`blocked`), lists blockers and warnings, and turns them into next actions. It is
intended as the day-to-day control panel for deciding whether to run more data,
upgrade a manuscript claim, or keep a claim deferred.
For real runs, include the RubricBench/ResearchRubrics `gold_validation`
reports so the dashboard shows whether hard-gold evidence has cleared the Raw
Audit Gate. Include `confidence_interval` sections for BSC/downstream bootstrap
reports so weak or missing uncertainty estimates stay visible before claim
upgrade. The real dashboard also uses a `rebuttal_manifest` section for
`outputs/rebuttal_pack/rebuttal_pack_manifest.json`; this verifies reviewer
concern counts, the concern-template SHA256, and checks that the manifest's
input/output SHA256 records still match the current Evidence Matrix, readiness
report, and rebuttal pack files. It also uses a `submission_gap_report` section
for `outputs/submission_readiness/gap_report/submission_gap_report.json`, so the
dashboard lists which paper-readiness phases remain blocked. The
`rebuttal_manifest` section summarizes the reviewer-facing claim ladder as
`claim_ladder_safe=x/4`, and the Result Card carries that value into its
Dashboard Diagnostics table.
To run it through the JSON stage runner, use a pipeline wrapper such as
`configs/pipeline_dashboard_smoke.example.json`.

## Build A Result Card

For paper-facing reporting, build a Result Card after the dashboard/evidence
matrix exists:

```bash
python3 scripts/build_result_card.py \
  --config configs/result_card_smoke.example.json \
  --output-json outputs/toy_result_card/result_card.json \
  --output-md outputs/toy_result_card/result_card.md
```

The Result Card is a defensive writing artifact: it does not recompute results,
but summarizes Raw Audit Gates, manifests, BSC metrics, downstream metrics,
evidence matrix status, readiness, and a `safe_to_claim / deferred / blocked`
claim decision. When `require_downstream=true`, a downstream summary with
`paper_claim_eligible=false` is treated as `not_paper_eligible`, its
accuracy/tie/margin cells are withheld, and the claim decision is blocked.
Use the same claim ladder as the paper: the frozen 100-example hard-gold
diagnostic is only a motivation claim; a hard-gold BSC coverage change with stable
redundancy and hallucination is metric-support; the SFT-only vs SFT+GRPO row
and reward-component ablations are required for method-support; and
RewardBench/RewardBench-2/JudgeBench API-scorer rows are required for
judge-utility support. Downgrade the manuscript sentence rather than the
evidence gate: without downstream support, write metric-only BSC evidence;
without C12, write aggregate coverage change rather than dimension-level recovery;
without C14, write proxy-gold supervision evidence rather than RLVR evidence;
without clean C0 provenance, no trained-method row is paper-facing.
For C0, `overlap_query_count == 0` is not enough if the configured
training-side artifacts are missing: such a raw audit gate remains
`artifact_status=blocked` and `overlap_status=not_auditable`.

## Offline BSC Ablations

Run reward-component ablations on one model-output file:

```bash
python3 scripts/run_bsc_ablation.py \
  --input examples/toy_bsc.jsonl \
  --embedding-model token-overlap \
  --coverage-tau 0.5 \
  --redundancy-tau 0.9 \
  --output-dir outputs/toy_bsc_ablation
```

This command re-scores the same generated criteria under alternate reward
weights. It is useful for checking the reward formula, but it is not the
paper-facing training ablation for the no-redundancy, no-validity,
no-verifier, or coverage-only variants.

It writes `full`, `no_red`, `no_valid`, `no_verifier`, and `cov_only` variants, plus a
markdown summary for the ablation table.

Run the single-teacher vs multi-teacher union ablation on teacher outputs:

```bash
python3 scripts/run_teacher_union_ablation.py \
  --teachers data/processed/teacher_rubrics_filtered.jsonl \
  --gold data/processed/splits/rubricbench_gold_train_seed.jsonl \
  --output-dir outputs/teacher_union_ablation
```

This produces `teacher_union_ablation.csv/md/json` and reports the union's
coverage delta relative to the best single teacher.

## verl Hook Sketch

In a custom reward file used by verl, import:

```python
from blindspot_rl.reward_bsc import compute_reward

def reward_func(data_item, response):
    return compute_reward(
        prompt=data_item["prompt"],
        response=response,
        gold_rubrics=data_item["gold_rubrics"],
    )
```

Add an API verifier later by passing an object with `judge(rubric, prompt=...)`.
If no verifier is passed, `R_valid` defaults to `1.0`, which is useful for pure
coverage experiments but should not be used for the final Hall table.
