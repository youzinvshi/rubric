# BlindSpot-RL Submission Gap Report

- Overall ok: `False`
- Readiness ok: `False`
- Hard blockers: `43`
- Warnings: `2`

## Claim Ladder Status

| Level | Status | Required Claims | Blocking Claims |
| --- | --- | --- | --- |
| motivation | `missing_evidence` | C1, C6 | C1: missing_evidence; C6: missing_evidence |
| metric-support | `missing_evidence` | C0, C2, C3 | C0: missing_evidence; C2: missing_evidence; C3: missing_evidence |
| method-support | `missing_evidence` | C5, C7, C14 | C5: missing_evidence; C7: missing_evidence; C14: missing_evidence |
| judge-utility support | `missing_evidence` | C0, C4, C9, C10, C12 | C0: missing_evidence; C4: missing_evidence; C9: missing_evidence; C10: missing_evidence; C12: missing_evidence |

## Operator Handoff

- Status: `ready_for_operator_input`

### Redacted Env Exports

- `export LOCAL_OPENAI_API_KEY=<REDACTED>`
- `export OPENAI_API_KEY=<REDACTED>`
- `export ANTHROPIC_API_KEY=<REDACTED>`
- `export DEEPSEEK_API_KEY=<REDACTED>`
- `export DASHSCOPE_API_KEY=<REDACTED>`
- `export GPT_AK_1=<REDACTED>`
- `export GPT_AK_2=<REDACTED>`
- `export GPT_AK_3=<REDACTED>`

### Local Files And Entries

- `input_files`: `data/processed/rmbench_queries.jsonl`, `data/processed/healthbench_hard_queries.jsonl`, `data/processed/arenahard_queries.jsonl`
- `provider_configs`: `configs/judge_scorer.local.jsonl`
- `provider_file_entries`: `configs/generators.local.jsonl: gpt4o`, `configs/generators.local.jsonl: claude`, `configs/generators.local.jsonl: sft_only`, `configs/generators.local.jsonl: sft_rl`, `configs/providers.local.jsonl: deepseek`, `configs/providers.local.jsonl: qwen`, `configs/judge_scorer.local.jsonl: judge-scorer`
- `required_providers`: `gpt4o`, `claude`, `sft_only`, `sft_rl`, `deepseek`, `qwen`, `judge-scorer`
- `training_artifacts`: none
- `paper_artifacts`: `tables/main_table.tex`, `tables/rl_stage_ablation_table.tex`, `tables/downstream_utility_table.tex`, `tables/ablation_table.tex`, `tables/teacher_union_ablation_table.tex`, `tables/verifier_filter_ablation_table.tex`, `tables/dimension_transition_table.tex`, `figures/semantic_space.pdf`, `figures/semantic_space.svg`

### Provider Template Commands

- `test -f configs/judge_scorer.local.jsonl || cp configs/judge_scorer.example.jsonl configs/judge_scorer.local.jsonl`
- `test -f configs/generators.local.jsonl || cp configs/generators.example.jsonl configs/generators.local.jsonl`
- `test -f configs/providers.local.jsonl || cp configs/providers.example.jsonl configs/providers.local.jsonl`
- `test -f configs/verifier.local.jsonl || cp configs/verifier.example.jsonl configs/verifier.local.jsonl`

### Provider Entries By File

- `configs/generators.local.jsonl`: `gpt4o`, `claude`, `sft_only`, `sft_rl`
- `configs/providers.local.jsonl`: `deepseek`, `qwen`
- `configs/judge_scorer.local.jsonl`: `judge-scorer`

### API Env By Provider

- `base`: `LOCAL_OPENAI_API_KEY`
- `gpt-5.4`: `GPT_AK_1`
- `gpt-5`: `GPT_AK_2`
- `gpt-4o`: `GPT_AK_3`
- `gemini`: `GPT_AK_1`
- `gemini-2.5-pro`: `GPT_AK_1`
- `meta-verifier`: `GPT_AK_3`

### API Env By File

- `configs/generators.local.jsonl`: `base: LOCAL_OPENAI_API_KEY`, `gpt-5.4: GPT_AK_1`, `gpt-5: GPT_AK_2`, `gpt-4o: GPT_AK_3`, `gemini: GPT_AK_1`
- `configs/providers.local.jsonl`: `gemini-2.5-pro: GPT_AK_1`, `gpt-5.4: GPT_AK_1`, `gpt-5: GPT_AK_2`, `gpt-4o: GPT_AK_3`
- `configs/verifier.local.jsonl`: `meta-verifier: GPT_AK_3`

### Training Data Chain

- none

### Training Artifact Producers

- none

### C0 Provenance Action Chain

- 1. `sft_data_preflight_and_teacher_budgets`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage api_budget_teacher_rubrics --to-stage sft_data_preflight`
  - Outputs: `outputs/api_budget/teacher_rubrics_budget.json`, `outputs/api_budget/healthbench_teacher_rubrics_budget.json`, `outputs/api_budget/writingbench_teacher_rubrics_budget.json`, `outputs/preflight/sft_data_preflight.json`, `outputs/preflight/sft_data_preflight.md`
  - Checks: `teacher-generation budgets are within run limits`, `provider configs expose all teacher and verifier entries`, `GPT_AK_1/GPT_AK_2/GPT_AK_3 are present before paid API calls`
- 2. `generate_teacher_rubrics`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage generate_teacher_rubrics_rubricbench --to-stage generate_teacher_rubrics_writingbench`
  - Outputs: `data/processed/teacher_rubrics_raw.jsonl`
  - Checks: `teacher raw criteria are generated only from train/proxy source pools`, `RubricBench test_main remains excluded from teacher generation`, `resume mode preserves completed provider-query pairs`
- 3. `api_budget_meta_verifier`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage api_budget_meta_verifier --to-stage api_budget_meta_verifier`
  - Outputs: `outputs/api_budget/meta_verifier_budget.json`, `outputs/api_budget/meta_verifier_budget.md`
  - Checks: `meta-verifier budget binds data/processed/teacher_rubrics_raw.jsonl`, `rubric-level verifier units are counted before filtering`, `budget report is present before verifier API mode runs`
- 4. `filter_teacher_rubrics`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage filter_teacher_rubrics --to-stage filter_teacher_rubrics_rewardbench2_downstream_overlap`
  - Outputs: `outputs/verifier/teacher_rubrics_filtered_report.json`, `outputs/contamination_audit/teacher_rubrics_hard_gold_holdout_filter.json`, `outputs/contamination_audit/teacher_rubrics_rewardbench_holdout_filter.json`, `outputs/contamination_audit/teacher_rubrics_judgebench_holdout_filter.json`, `outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json`, `data/processed/teacher_rubrics_training_clean.jsonl`
  - Checks: `API verifier provenance is present`, `teacher criteria are holdout-clean before proxy-gold build`, `teacher clean output SHA feeds proxy-gold build input`
- 5. `build_sft_all_domains`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage build_sft_all_domains --to-stage filter_proxy_gold_rewardbench2_downstream_overlap`
  - Outputs: `outputs/sft_data/proxy_gold_build_report.json`, `data/processed/blindspot_sft.unfiltered.jsonl`, `data/processed/proxy_gold.unfiltered.jsonl`, `outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json`, `outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json`, `data/processed/blindspot_sft.jsonl`, `data/processed/proxy_gold.jsonl`
  - Checks: `build report binds data/processed/teacher_rubrics_training_clean.jsonl`, `build report writes .unfiltered SFT/proxy-gold outputs`, `post-SFT filters write final clean SFT/proxy-gold outputs`
- 6. `convert_proxy_gold_to_verl_and_audit_holdouts`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage convert_proxy_gold_to_verl --to-stage audit_rewardbench2_downstream_holdout_contamination`
  - Outputs: `outputs/sft_data/proxy_gold_verl_report.json`, `data/processed/proxy_gold_verl.parquet`, `outputs/contamination_audit/hard_gold_holdout_contamination.json`, `outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json`, `outputs/contamination_audit/judgebench_downstream_holdout_contamination.json`, `outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json`
  - Checks: `VERL report input SHA matches final clean proxy-gold`, `hard-gold and downstream audits are complete`, `overlap_query_count is zero for all final holdouts`
- 7. `refresh_evidence_and_reports`
  - Command: `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage evidence_real --to-stage sync_result_card_real`
  - Outputs: `outputs/evidence_real/evidence_matrix.json`, `outputs/submission_readiness/readiness_report.json`, `outputs/dashboard/real_run_dashboard.json`, `outputs/result_card/result_card.json`, `paper/asset_index/evidence_matrix.json`
  - Checks: `C0 status is recomputed from refreshed provenance`, `readiness remains fail-closed for unrelated missing evidence`, `paper-facing asset index uses refreshed reports`

### Validation Commands

- `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --only real_run_preflight --only sft_data_preflight`
- `python3 scripts/build_submission_gap_report.py --readiness-report outputs/submission_readiness/readiness_report.json --evidence-matrix outputs/evidence_real/evidence_matrix.json --rebuttal-manifest outputs/rebuttal_pack/rebuttal_pack_manifest.json --preflight-report outputs/preflight/real_run_preflight.json --preflight-report outputs/preflight/sft_data_preflight.json --gate-report outputs/contamination_audit/hard_gold_holdout_contamination.json --gate-report outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json --gate-report outputs/contamination_audit/judgebench_downstream_holdout_contamination.json --gate-report outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json --output-dir outputs/submission_readiness/gap_report`

### Safety Notes

- Do not commit local provider files or API key values.
- Keep hard-gold holdouts out of proxy-gold training inputs.
- Rerun preflight after changing provider files, API key env vars, or input data.

## Execution Sequence

### 1. data_isolation

- Step id: `data_isolation`
- Phase: `data_isolation_hard_gold` (`blocked`)
- Summary: phase_status=blocked; evidence_gates=C0; missing_prerequisites=3; ready_to_run=false
- Blocked by prior phases: none
- Evidence gates: `C0`
- Commands:
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage audit_data_readiness --to-stage audit_rewardbench2_downstream_holdout_contamination`
- Unlocks: `outputs/contamination_audit/*`, `query-disjoint hard-gold/proxy/downstream evidence`
- Notes: Run before training or downstream claims; hard-gold/proxy/downstream isolation is the contamination gate.
- Missing prerequisites:
  - `input_files`: `data/processed/rmbench_queries.jsonl`, `data/processed/healthbench_hard_queries.jsonl`, `data/processed/arenahard_queries.jsonl`

### 2. training_and_serving

- Step id: `training_and_serving`
- Phase: `training_and_serving` (`blocked`)
- Summary: phase_status=blocked; evidence_gates=C2,C3,C14; blocked_by_prior_phases=1; missing_prerequisites=8; ready_to_run=false
- Blocked by prior phases: data_isolation_hard_gold
- Evidence gates: `C2`, `C3`, `C14`
- Commands:
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage training_commands --to-stage training_completion_gate`
- Unlocks: `outputs/training_commands/training_done.json`, `SFT-only and SFT+GRPO serving metadata`
- Notes: The manual completion gate must bind checkpoints, serving endpoints, proxy-gold data, and reward configuration.
- Missing prerequisites:
  - `required_env`: `LOCAL_OPENAI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `GPT_AK_1`, `GPT_AK_2`, `GPT_AK_3`

### 3. model_evaluation_criteria_generation

- Step id: `model_evaluation_criteria_generation`
- Phase: `training_and_serving` (`blocked`)
- Summary: phase_status=blocked; evidence_gates=C2,C3,C4,C9,C10; blocked_by_prior_phases=1; missing_prerequisites=30; ready_to_run=false
- Blocked by prior phases: data_isolation_hard_gold
- Evidence gates: `C2`, `C3`, `C4`, `C9`, `C10`
- Commands:
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage generate_model_rubrics_rubricbench --to-stage validate_writingbench_model_rubrics`
- Unlocks: `data/processed/*_model_rubrics.jsonl`, `verifier-filtered model evaluation-criteria reports`
- Notes: Requires provider configs and budgets; verified evaluation-criteria files feed hard-gold and downstream matrices.
- Missing prerequisites:
  - `api_env`: `LOCAL_OPENAI_API_KEY for provider base`, `GPT_AK_1 for provider gpt-5.4`, `GPT_AK_2 for provider gpt-5`, `GPT_AK_3 for provider gpt-4o`, `GPT_AK_1 for provider gemini`, `GPT_AK_1 for provider gemini-2.5-pro`, `GPT_AK_3 for provider meta-verifier`
  - `required_env`: `LOCAL_OPENAI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `GPT_AK_1`, `GPT_AK_2`, `GPT_AK_3`
  - `provider_configs`: `configs/judge_scorer.local.jsonl`
  - `required_providers`: `gpt4o`, `claude`, `sft_only`, `sft_rl`, `deepseek`, `qwen`, `judge-scorer`
  - `provider_file_entries`: `configs/generators.local.jsonl: gpt4o`, `configs/generators.local.jsonl: claude`, `configs/generators.local.jsonl: sft_only`, `configs/generators.local.jsonl: sft_rl`, `configs/providers.local.jsonl: deepseek`, `configs/providers.local.jsonl: qwen`, `configs/judge_scorer.local.jsonl: judge-scorer`

### 4. main_matrix_and_ablations

- Step id: `main_matrix_and_ablations`
- Phase: `main_hard_gold_bsc` (`blocked`)
- Summary: phase_status=blocked; evidence_gates=C1,C2,C3,C5,C6,C7,C12,C13,C14; blocked_by_prior_phases=2; ready_to_run=false
- Blocked by prior phases: data_isolation_hard_gold, training_and_serving
- Evidence gates: `C1`, `C2`, `C3`, `C5`, `C6`, `C7`, `C12`, `C13`, `C14`
- Commands:
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_real.generated.json`
- Unlocks: `outputs/matrix_real/main_table.csv`, `outputs/bsc_ablation/ablation_summary.csv`, `outputs/teacher_union_ablation/teacher_union_ablation.csv`, `outputs/verifier_filter_ablation/verifier_filter_ablation.csv`, `outputs/matrix_real/semantic_space/*`
- Notes: This produces the hard-gold BSC matrix, RL-stage comparison inputs, ablations, dimension-transition audit, and semantic-space assets.
- Missing prerequisites: none

### 5. downstream_matrices

- Step id: `downstream_matrices`
- Phase: `downstream_utility` (`blocked`)
- Summary: phase_status=blocked; evidence_gates=C4,C9,C10; blocked_by_prior_phases=3; missing_prerequisites=15; ready_to_run=false
- Blocked by prior phases: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Evidence gates: `C4`, `C9`, `C10`
- Commands:
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_judgebench.generated.json`
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_rewardbench2.generated.json`
- Unlocks: `outputs/matrix_judgebench/main_table.csv`, `outputs/matrix_rewardbench2/main_table.csv`, `outputs/generalization_matrix/main_table.csv`
- Notes: Run held-out downstream utility matrices with API/model scorer summaries and paper_claim_eligible provenance.
- Missing prerequisites:
  - `provider_configs`: `configs/judge_scorer.local.jsonl`
  - `required_providers`: `gpt4o`, `claude`, `sft_only`, `sft_rl`, `deepseek`, `qwen`, `judge-scorer`
  - `provider_file_entries`: `configs/generators.local.jsonl: gpt4o`, `configs/generators.local.jsonl: claude`, `configs/generators.local.jsonl: sft_only`, `configs/generators.local.jsonl: sft_rl`, `configs/providers.local.jsonl: deepseek`, `configs/providers.local.jsonl: qwen`, `configs/judge_scorer.local.jsonl: judge-scorer`

### 6. paper_export_and_readiness

- Step id: `paper_export_and_readiness`
- Phase: `paper_readiness` (`blocked`)
- Summary: phase_status=blocked; evidence_gates=C0-C14; blocked_by_prior_phases=6; missing_prerequisites=9; ready_to_run=false
- Blocked by prior phases: ablations, data_isolation_hard_gold, downstream_utility, main_hard_gold_bsc, semantic_space, training_and_serving
- Evidence gates: `C0-C14`
- Commands:
  - `python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage evidence_real --to-stage sync_result_card_real`
- Unlocks: `outputs/paper_artifacts/*`, `paper/asset_index/*`, `outputs/submission_readiness/*`
- Notes: Rebuild evidence, paper tables/figures, readiness, rebuttal pack, dashboard, Result Card, and synced asset index.
- Missing prerequisites:
  - `paper_artifacts`: `tables/main_table.tex`, `tables/rl_stage_ablation_table.tex`, `tables/downstream_utility_table.tex`, `tables/ablation_table.tex`, `tables/teacher_union_ablation_table.tex`, `tables/verifier_filter_ablation_table.tex`, `tables/dimension_transition_table.tex`, `figures/semantic_space.pdf`, `figures/semantic_space.svg`

## Data Isolation And Hard-Gold Holdouts

- Status: `blocked`
- Summary: blocked: claim_gaps=1, readiness_blockers=1
- Depends on: none
- Blocked by prior phases: none
- Claim gaps: C0

### Next Actions

- Run real data normalization and hard-gold/downstream split stages.
- Refresh proxy-train filters and zero-overlap contamination audits before training.

### Claim Discipline

- Do not report hard-gold, trained-method, ablation, downstream, or visualization results until C0 is safe_to_claim with query-disjoint holdouts and SHA-bound training/evaluation provenance.

### Readiness Blockers

- required evidence claim C0 is missing_evidence

### Paper Asset Blockers

- none

### Paper Asset Warnings

- none

### Manifest Gaps

- none

### Claim Checks

- `C0` missing_evidence: RubricBench test_main hard-gold and downstream benchmark holdout queries are query-disjoint from all configured SFT/proxy/RL training artifacts.
  - [missing] Teacher evaluation-criteria verifier-filtering provenance report (outputs/verifier/teacher_rubrics_filtered_report.json)
  - [missing] Teacher evaluation-criteria vs RewardBench filter report (outputs/contamination_audit/teacher_rubrics_rewardbench_holdout_filter.json)
  - [missing] Teacher evaluation-criteria vs JudgeBench filter report (outputs/contamination_audit/teacher_rubrics_judgebench_holdout_filter.json)
  - [missing] Teacher evaluation-criteria vs RewardBench-2 filter report (outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json)
  - [missing] Teacher verifier report binds raw teacher input (outputs/verifier/teacher_rubrics_filtered_report.json)
  - ... 23 more

## SFT/GRPO Training And Serving

- Status: `blocked`
- Summary: blocked: claim_gaps=3, readiness_blockers=9, blocked_by_prior_phases=1
- Depends on: data_isolation_hard_gold
- Blocked by prior phases: data_isolation_hard_gold
- Claim gaps: C14, C2, C3

### Next Actions

- Run SFT-only and SFT+GRPO on proxy-gold data only.
- Write training_done.json with checkpoint paths, serving endpoints, rl_data_report, and reward_function.

### Claim Discipline

- Treat SFT/GRPO as unavailable for paper claims until training_done.json binds SFT-only and SFT+GRPO checkpoints, serving metadata, proxy-gold RL data, and reward configuration.

### Readiness Blockers

- required evidence claim C2 is missing_evidence
- required evidence claim C3 is missing_evidence
- required evidence claim C14 is missing_evidence
- raw gate SFT Data Preflight is blocked: 8 blockers, 0 warnings
- raw gate Training Completion Gate is blocked: 46 blockers, 0 warnings
- raw gate Matrix Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate JudgeBench Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate RewardBench-2 Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate Generalization Trained Method Gate is blocked: 44 blockers, 0 warnings

### Paper Asset Blockers

- none

### Paper Asset Warnings

- none

### Manifest Gaps

- none

### Claim Checks

- `C14` missing_evidence: The GRPO/RLVR stage is tested against SFT-only under matched redundancy and hallucination controls.
  - [missing] Main table (outputs/matrix_real/main_table.csv)
  - [missing] SFT/GRPO training completion manifest (outputs/training_commands/training_done.json)
  - [missing] SFT-only BSC summary (outputs/matrix_real/sft_only/bsc/summary.json)
  - [missing] SFT+RL BSC summary (outputs/matrix_real/sft_rl/bsc/summary.json)
  - [missing] SFT-only BSC bootstrap CI (outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json)
  - ... 81 more
- `C2` missing_evidence: BSC reward training is tested for hard-gold evaluation-dimension coverage changes relative to the base policy.
  - [missing] Main table (outputs/matrix_real/main_table.csv)
  - [missing] Base BSC bootstrap CI (outputs/matrix_real/base/bsc_ci/bootstrap_ci.json)
  - [missing] SFT+RL BSC bootstrap CI (outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json)
  - [missing] Base BSC join report (outputs/matrix_real/base/bsc_join_report.json)
  - [missing] SFT+RL BSC join report (outputs/matrix_real/sft_rl/bsc_join_report.json)
  - ... 12 more
- `C3` missing_evidence: The trained evaluation-criteria policy is tested for whether coverage changes remain reportable under redundancy and hallucination controls.
  - [missing] Base BSC bootstrap CI (outputs/matrix_real/base/bsc_ci/bootstrap_ci.json)
  - [missing] SFT+RL BSC bootstrap CI (outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json)
  - [missing] SFT-only BSC join report (outputs/matrix_real/sft_only/bsc_join_report.json)
  - [missing] SFT+RL BSC join report (outputs/matrix_real/sft_rl/bsc_join_report.json)
  - [missing] Base verifier-backed BSC records (outputs/matrix_real/base/bsc/summary.json)
  - ... 16 more

## Main Hard-Gold BSC Evidence

- Status: `blocked`
- Summary: blocked: claim_gaps=6, readiness_blockers=4, paper_asset_blockers=2, blocked_by_prior_phases=2
- Depends on: data_isolation_hard_gold, training_and_serving
- Blocked by prior phases: data_isolation_hard_gold, training_and_serving
- Claim gaps: C1, C12, C14, C2, C3, C6

### Next Actions

- Run the real RubricBench test_main matrix for base, API teachers, SFT-only, and SFT+GRPO.
- Generate BSC summaries, paired bootstrap CIs, threshold sweeps, dimension-transition summaries, and matrix audit report.

### Claim Discipline

- A BSC coverage change is only metric evidence until C3 controls redundancy/hallucination, C6 verifies threshold robustness and human audit, and C12/C14 audit dimension-level recovery over SFT-only; dimension-level recovery remains a permitted conclusion only after those gates pass.

### Readiness Blockers

- audit report is not ok
- required evidence claim C1 is missing_evidence
- required evidence claim C6 is missing_evidence
- required evidence claim C12 is missing_evidence

### Paper Asset Blockers

- `main_bsc_table`: 1 (main hard-gold BSC paper table; gate C1/C2/C3/C14)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/matrix_real/main_table.csv`
  - Paper artifacts: `main_table.tex`
  - Action: Run the real hard-gold BSC matrix and export the main paper table only after paper_claim_eligible summaries pass.
- `dimension_transition`: 1 (dimension-transition audit paper table; gate C12)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/matrix_real/dimension_transition_summary.csv`
  - Paper artifacts: `dimension_transition_table.tex`
  - Action: Run the dimension-transition audit and export the C12 table before writing dimension-level recovery language.

### Paper Asset Warnings

- none

### Manifest Gaps

- none

### Claim Checks

- `C1` missing_evidence: A single-model evaluation-criteria policy leaves measurable blind spots against human-gold evaluation dimensions on RubricBench.
  - [missing] Base BSC summary (outputs/matrix_real/base/bsc/summary.json)
  - [missing] Base BSC per-item diagnostics (outputs/matrix_real/base/bsc/per_item.csv)
  - [missing] Base BSC bootstrap CI (outputs/matrix_real/base/bsc_ci/bootstrap_ci.json)
  - [missing] RubricBench-scale sample size (outputs/matrix_real/base/bsc/summary.json)
  - [missing] Base blind spot is non-trivial (outputs/matrix_real/base/bsc/summary.json)
  - ... 12 more
- `C12` missing_evidence: SFT+RL is evaluated with a dimension-transition audit on human-gold dimensions rather than only aggregate BSC changes.
  - [missing] Base to SFT-only dimension-transition summary (outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json)
  - [missing] Base to SFT+RL dimension-transition summary (outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json)
  - [missing] Base to SFT+RL category transition table (outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_by_category.csv)
  - [missing] Base to SFT+RL gold-dimension audit rows (outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_gold_items.jsonl)
  - [missing] Base BSC join report for dimension-transition provenance (outputs/matrix_real/base/bsc_join_report.json)
  - ... 67 more
- `C14` missing_evidence: The GRPO/RLVR stage is tested against SFT-only under matched redundancy and hallucination controls.
  - [missing] Main table (outputs/matrix_real/main_table.csv)
  - [missing] SFT/GRPO training completion manifest (outputs/training_commands/training_done.json)
  - [missing] SFT-only BSC summary (outputs/matrix_real/sft_only/bsc/summary.json)
  - [missing] SFT+RL BSC summary (outputs/matrix_real/sft_rl/bsc/summary.json)
  - [missing] SFT-only BSC bootstrap CI (outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json)
  - ... 81 more
- `C2` missing_evidence: BSC reward training is tested for hard-gold evaluation-dimension coverage changes relative to the base policy.
  - [missing] Main table (outputs/matrix_real/main_table.csv)
  - [missing] Base BSC bootstrap CI (outputs/matrix_real/base/bsc_ci/bootstrap_ci.json)
  - [missing] SFT+RL BSC bootstrap CI (outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json)
  - [missing] Base BSC join report (outputs/matrix_real/base/bsc_join_report.json)
  - [missing] SFT+RL BSC join report (outputs/matrix_real/sft_rl/bsc_join_report.json)
  - ... 12 more
- `C3` missing_evidence: The trained evaluation-criteria policy is tested for whether coverage changes remain reportable under redundancy and hallucination controls.
  - [missing] Base BSC bootstrap CI (outputs/matrix_real/base/bsc_ci/bootstrap_ci.json)
  - [missing] SFT+RL BSC bootstrap CI (outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json)
  - [missing] SFT-only BSC join report (outputs/matrix_real/sft_only/bsc_join_report.json)
  - [missing] SFT+RL BSC join report (outputs/matrix_real/sft_rl/bsc_join_report.json)
  - [missing] Base verifier-backed BSC records (outputs/matrix_real/base/bsc/summary.json)
  - ... 16 more
- `C6` missing_evidence: BSC findings are auditable across semantic threshold settings.
  - [missing] Base threshold sweep (outputs/matrix_real/base/bsc_sweep/threshold_sweep.csv)
  - [missing] SFT+RL threshold sweep (outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.csv)
  - [missing] Base BSC human audit annotation pack (outputs/matrix_real/base/bsc_human_audit_pack/audit_items.csv)
  - [missing] SFT+RL BSC human audit annotation pack (outputs/matrix_real/sft_rl/bsc_human_audit_pack/audit_items.csv)
  - [missing] Base BSC C6-gated human-audit summary (outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.json)
  - ... 27 more

## Downstream Judge Utility

- Status: `blocked`
- Summary: blocked: claim_gaps=4, readiness_blockers=8, paper_asset_blockers=1, blocked_by_prior_phases=3
- Depends on: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Blocked by prior phases: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Claim gaps: C10, C11, C4, C9

### Next Actions

- Run paper-facing API/model scorer evaluations on RewardBench, JudgeBench, and RewardBench-2 holdouts.
- Ensure every downstream summary is paper_claim_eligible and bound to its input/provider/budget SHA256 contract.

### Claim Discipline

- Do not describe BSC coverage changes as supporting judge utility until C4/C9/C10 pass with API/model scorer outputs, paper_claim_eligible summaries, and input/provider/budget SHA256 contracts.

### Readiness Blockers

- required evidence claim C4 is missing_evidence
- required evidence claim C9 is missing_evidence
- required evidence claim C10 is missing_evidence
- raw gate Meta-Verifier API Budget is missing: report missing
- raw gate RubricBench Model Criteria Verifier API Budget is missing: report missing
- raw gate Downstream Policy RLVR Completion Gate is blocked: 19 blockers, 0 warnings
- required paper tables not synced: tables/main_table.tex, tables/rl_stage_ablation_table.tex, tables/downstream_utility_table.tex, tables/ablation_table.tex, tables/teacher_union_ablation_table.tex, tables/verifier_filter_ablation_table.tex, tables/dimension_transition_table.tex
- artifact is missing or empty: outputs/paper_artifacts/downstream_utility_table.tex

### Paper Asset Blockers

- `downstream_utility`: 1 (RewardBench/JudgeBench/RewardBench-2 utility table; gate C4/C9/C10)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/matrix_real/main_table.csv`, `outputs/matrix_judgebench/main_table.csv`, `outputs/matrix_rewardbench2/main_table.csv`
  - Paper artifacts: `downstream_utility_table.tex`
  - Action: Run RewardBench, JudgeBench, and RewardBench-2 downstream API/model scorer matrices, then export the combined utility table.

### Paper Asset Warnings

- none

### Manifest Gaps

- none

### Claim Checks

- `C10` missing_evidence: On RewardBench-2, BSC-optimized evaluation criteria are tested for multi-candidate utility relative to the base policy.
  - [missing] RewardBench-2 multi-candidate table (outputs/matrix_rewardbench2/main_table.csv)
  - [missing] RewardBench-2 base downstream bootstrap CI (outputs/matrix_rewardbench2/rewardbench2_base/downstream_ci/bootstrap_ci.json)
  - [missing] RewardBench-2 SFT+RL downstream bootstrap CI (outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_ci/bootstrap_ci.json)
  - [missing] RewardBench-2 base downstream join report (outputs/matrix_rewardbench2/rewardbench2_base/downstream_join_report.json)
  - [missing] RewardBench-2 SFT+RL downstream join report (outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_join_report.json)
  - ... 90 more
- `C11` missing_evidence: HealthBench-Hard and ArenaHard policy-level RLVR results are only reportable after evaluation-criteria policy reward training, policy GRPO, and benchmark evaluation artifacts are complete.
  - [missing] Downstream RLVR command manifest (outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json)
  - [missing] HealthBench-Hard policy checkpoint (outputs/policy_rlvr/healthbench_hard_policy)
  - [missing] HealthBench-Hard evaluation report (outputs/policy_rlvr/healthbench_hard_eval.json)
  - [missing] ArenaHard policy checkpoint (outputs/policy_rlvr/arenahard_policy)
  - [missing] ArenaHard evaluation report (outputs/policy_rlvr/arenahard_eval.json)
  - ... 1 more
- `C4` missing_evidence: BSC-optimized evaluation criteria are tested for downstream chosen-vs-rejected utility relative to the base policy.
  - [missing] Base downstream bootstrap CI (outputs/matrix_real/base/downstream_ci/bootstrap_ci.json)
  - [missing] SFT+RL downstream bootstrap CI (outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.json)
  - [missing] Base downstream join report (outputs/matrix_real/base/downstream_join_report.json)
  - [missing] SFT+RL downstream join report (outputs/matrix_real/sft_rl/downstream_join_report.json)
  - [missing] Base downstream API budget report (outputs/matrix_real/base/downstream_api_budget/budget.json)
  - ... 85 more
- `C9` missing_evidence: On JudgeBench, BSC-optimized evaluation criteria are tested for chosen-vs-rejected utility relative to the base policy.
  - [missing] JudgeBench main table (outputs/matrix_judgebench/main_table.csv)
  - [missing] JudgeBench base downstream bootstrap CI (outputs/matrix_judgebench/judgebench_base/downstream_ci/bootstrap_ci.json)
  - [missing] JudgeBench SFT+RL downstream bootstrap CI (outputs/matrix_judgebench/judgebench_sft_rl/downstream_ci/bootstrap_ci.json)
  - [missing] JudgeBench base downstream join report (outputs/matrix_judgebench/judgebench_base/downstream_join_report.json)
  - [missing] JudgeBench SFT+RL downstream join report (outputs/matrix_judgebench/judgebench_sft_rl/downstream_join_report.json)
  - ... 86 more

## Ablations

- Status: `blocked`
- Summary: blocked: claim_gaps=3, readiness_blockers=7, paper_asset_blockers=4, blocked_by_prior_phases=3
- Depends on: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Blocked by prior phases: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Claim gaps: C14, C5, C7

### Next Actions

- Run separately trained reward-component ablations: no redundancy, no validity, and coverage-only; keep offline reward re-scoring as diagnostic only.
- Run verifier-filter ablation for proxy-gold quality control.
- Run single-teacher versus multi-teacher union and SFT-only versus SFT+GRPO comparisons under the same BGE protocol.

### Claim Discipline

- Do not claim the RL stage is necessary or anti-hacking works from offline re-scoring; paper-facing C7 requires separately trained variants and verifier-filter evidence under the same BGE protocol.

### Readiness Blockers

- required evidence claim C5 is missing_evidence
- required evidence claim C7 is missing_evidence
- raw gate RubricBench Model Criteria Verifier Stats is missing: report missing
- artifact is missing or empty: outputs/paper_artifacts/rl_stage_ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/teacher_union_ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/verifier_filter_ablation_table.tex

### Paper Asset Blockers

- `rl_stage_ablation`: 1 (SFT-only vs SFT+GRPO paper table; gate C14)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/matrix_real/main_table.csv`
  - Paper artifacts: `rl_stage_ablation_table.tex`
  - Action: Run matrix_real through base, sft_only, and sft_rl BSC/downstream summaries, then export paper artifacts.
- `reward_ablation`: 1 (reward-component ablation table; gate C7)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/bsc_ablation/ablation_summary.csv`
  - Paper artifacts: `ablation_table.tex`
  - Action: Run separately trained reward-component ablations and export the reward ablation table.
- `teacher_union_ablation`: 1 (teacher-union ablation table; gate C5)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/teacher_union_ablation/teacher_union_ablation.csv`
  - Paper artifacts: `teacher_union_ablation_table.tex`
  - Action: Run single-teacher versus multi-teacher union ablation, then export the teacher-union table.
- `verifier_filter_ablation`: 1 (verifier-filter ablation table; gate C7)
  - Producer: `scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/verifier_filter_ablation/verifier_filter_ablation.csv`
  - Paper artifacts: `verifier_filter_ablation_table.tex`
  - Action: Run verifier-filter ablation over raw and filtered teacher criteria outputs, then export the verifier-filter table.

### Paper Asset Warnings

- none

### Manifest Gaps

- none

### Claim Checks

- `C14` missing_evidence: The GRPO/RLVR stage is tested against SFT-only under matched redundancy and hallucination controls.
  - [missing] Main table (outputs/matrix_real/main_table.csv)
  - [missing] SFT/GRPO training completion manifest (outputs/training_commands/training_done.json)
  - [missing] SFT-only BSC summary (outputs/matrix_real/sft_only/bsc/summary.json)
  - [missing] SFT+RL BSC summary (outputs/matrix_real/sft_rl/bsc/summary.json)
  - [missing] SFT-only BSC bootstrap CI (outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json)
  - ... 81 more
- `C5` missing_evidence: Multi-teacher criteria union is evaluated against the best single teacher for human-gold evaluation-dimension coverage.
  - [missing] Teacher-union ablation (outputs/teacher_union_ablation/teacher_union_ablation.csv)
  - [missing] Teacher-union ablation JSON (outputs/teacher_union_ablation/teacher_union_ablation.json)
  - [missing] Teacher-union per-item audit (outputs/teacher_union_ablation/teacher_union_per_item.csv)
  - [missing] Union coverage delta over best single teacher (outputs/teacher_union_ablation/teacher_union_ablation.json)
  - [missing] Teacher-union ablation has RubricBench-scale matched queries (outputs/teacher_union_ablation/teacher_union_ablation.json)
  - ... 9 more
- `C7` missing_evidence: Ablations report reward-component variants and the no-verifier-filtering proxy-gold variant.
  - [missing] Reward ablation table (outputs/bsc_ablation/ablation_summary.csv)
  - [missing] Full reward variant (outputs/bsc_ablation/variants/full_summary.json)
  - [missing] No redundancy variant (outputs/bsc_ablation/variants/no_red_summary.json)
  - [missing] No validity variant (outputs/bsc_ablation/variants/no_valid_summary.json)
  - [missing] No verifier variant (outputs/bsc_ablation/variants/no_verifier_summary.json)
  - ... 138 more

## Semantic-Space Visualization

- Status: `blocked`
- Summary: blocked: claim_gaps=1, readiness_blockers=7, paper_asset_blockers=1, blocked_by_prior_phases=3
- Depends on: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Blocked by prior phases: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving
- Claim gaps: C13

### Next Actions

- Build UMAP/PCA semantic-space assets from base, SFT-only, and SFT+GRPO hard-gold BSC eval rows.
- Verify semantic_space_points.csv and semantic_space_summary.json carry nearest-gold audit fields.

### Claim Discipline

- Treat the plot as illustrative until C13 verifies point-level provenance, nearest-gold audit fields, category/cluster coverage, dispersion, and non-empty SVG/PDF/CSV/JSON assets.

### Readiness Blockers

- required evidence claim C13 is missing_evidence
- required paper figures not synced: figures/semantic_space.pdf, figures/semantic_space.svg
- required paper reviewer-facing docs not synced: asset_index/semantic_space_points.csv, asset_index/semantic_space_summary.json
- artifact is missing or empty: outputs/paper_artifacts/semantic_space.svg
- artifact is missing or empty: outputs/paper_artifacts/semantic_space.pdf
- artifact is missing or empty: outputs/paper_artifacts/semantic_space_points.csv
- artifact is missing or empty: outputs/paper_artifacts/semantic_space_summary.json

### Paper Asset Blockers

- `semantic_space`: 4 (semantic-space SVG/PDF/CSV/JSON assets; gate C13)
  - Producer: `scripts/build_semantic_space_visualization.py -> scripts/export_paper_artifacts.py`
  - Source artifacts: `outputs/matrix_real/semantic_space/semantic_space.svg`, `outputs/matrix_real/semantic_space/semantic_space.pdf`, `outputs/matrix_real/semantic_space/semantic_space_points.csv`, `outputs/matrix_real/semantic_space/semantic_space_summary.json`
  - Paper artifacts: `semantic_space.svg`, `semantic_space.pdf`, `semantic_space_points.csv`, `semantic_space_summary.json`
  - Action: Run semantic-space visualization after base, sft_only, and sft_rl hard-gold BSC eval rows exist, then export/copy all four assets.

### Paper Asset Warnings

- none

### Manifest Gaps

- none

### Claim Checks

- `C13` missing_evidence: A semantic-space audit is planned to test for reportable nearest-gold evaluation-dimension region-coverage change without local collapse.
  - [missing] Semantic-space SVG (outputs/matrix_real/semantic_space/semantic_space.svg)
  - [missing] Semantic-space PDF (outputs/matrix_real/semantic_space/semantic_space.pdf)
  - [missing] Semantic-space point CSV (outputs/matrix_real/semantic_space/semantic_space_points.csv)
  - [missing] Semantic-space summary (outputs/matrix_real/semantic_space/semantic_space_summary.json)
  - [missing] Base BSC join report for semantic-space provenance (outputs/matrix_real/base/bsc_join_report.json)
  - ... 76 more

## Paper Assets And AAAI Readiness

- Status: `blocked`
- Summary: blocked: readiness_blockers=7, paper_asset_warnings=1, blocked_by_prior_phases=6
- Depends on: ablations, data_isolation_hard_gold, downstream_utility, main_hard_gold_bsc, semantic_space, training_and_serving
- Blocked by prior phases: ablations, data_isolation_hard_gold, downstream_utility, main_hard_gold_bsc, semantic_space, training_and_serving
- Claim gaps: none

### Next Actions

- Export paper artifacts after real matrices and ablations are complete.
- Sync tables/figures into paper/ and rerun LaTeX plus submission readiness.

### Claim Discipline

- Keep the paper in problem--metric--method framing and write empirical claims only when the relevant evidence rows are safe_to_claim and synced paper assets pass readiness.

### Readiness Blockers

- raw gate Data Readiness Audit is blocked: 10 missing files
- raw gate AAAI LaTeX Compile is blocked: pdf_bytes=0, pages=0, max_pages=0, official_style_active=False, submission_mode_declared=True, bibliography_style_active=False, anonymous_author_declared=True, 3 blockers
- paper asset index is blocked: 11 blockers, 2 warnings, 8 blocker categories, 1 warning categories
- artifact is missing or empty: outputs/paper_artifacts/main_table.tex
- artifact is missing or empty: outputs/paper_artifacts/dimension_transition_table.tex
- raw gate Real Run Preflight is blocked: 33 blockers, 8 warnings
- raw gate Filtered Teacher Evaluation-Criteria Validation is missing: report missing

### Paper Asset Blockers

- none

### Paper Asset Warnings

- `api_handoff`: 2 (API handoff reviewer-facing docs)

### Manifest Gaps

- none

### Claim Checks

- none

## Reviewer-Facing Rebuttal Readiness

- Status: `blocked`
- Summary: blocked: manifest_gaps=2, blocked_by_prior_phases=1
- Depends on: paper_readiness
- Blocked by prior phases: paper_readiness
- Claim gaps: none

### Next Actions

- Regenerate the rebuttal pack after Evidence Matrix rows and submission readiness are refreshed.
- Use only answer_ready rebuttal entries whose matched Evidence Matrix rows are safe_to_claim and whose readiness_ok flag is true.

### Claim Discipline

- Treat rebuttal entries as readiness notes, not new claims; do not use reviewer-facing answers until the pack is generated against the current Evidence Matrix/readiness report, readiness_ok is true, and relevant entries are answer_ready.

### Readiness Blockers

- none

### Paper Asset Blockers

- none

### Paper Asset Warnings

- none

### Manifest Gaps

- rebuttal pack was built while submission readiness was false
- rebuttal pack has no answer_ready entries

### Claim Checks

- none
