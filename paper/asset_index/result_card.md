# BlindSpot-RL Real Result Card

- Experiment ID: `real_main_matrix`
- Scope: `RubricBench hard-gold main matrix, required ablations including C5 single-teacher vs multi-teacher union, RewardBench downstream utility evidence, JudgeBench difficult-case downstream extension, RewardBench-2 multi-candidate downstream extension, RM-Bench generation readiness, and HealthBench/WritingBench proxy-domain generalization gates`
- Claim decision: `blocked`

## Raw Audit Gates

| Gate | Type | Status | Summary |
| --- | --- | --- | --- |
| Data Source Local Config | `data_source_local_config` | `pass` | 0 blockers, source_status=pass |
| Real Run Preflight | `preflight` | `blocked` | 33 blockers, 8 warnings |
| SFT Data Preflight | `preflight` | `blocked` | 8 blockers, 0 warnings |
| Data Readiness Audit | `audit` | `blocked` | 10 missing files |
| AAAI LaTeX Compile | `latex_compile` | `blocked` | pdf_bytes=0, pages=0, max_pages=0, official_style_active=False, bibliography_style_active=False, anonymous_author_declared=True |
| RewardBench Proxy-Train Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 |
| RewardBench Proxy-Train vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| RewardBench Proxy-Train vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| RewardBench Proxy-Train vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=5 |
| RewardBench-2 Query Pool vs RubricBench Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=50 |
| RewardBench-2 Query Pool vs ResearchRubrics Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| RewardBench-2 Multicandidate vs RubricBench Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=50 |
| RewardBench-2 Multicandidate vs ResearchRubrics Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Pre-SFT Clean Proxy-Train vs Hard-Gold Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| Pre-SFT Clean Proxy-Train vs RewardBench Holdout Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| Pre-SFT Clean Proxy-Train vs JudgeBench Holdout Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| Pre-SFT Clean Proxy-Train vs RewardBench-2 Holdout Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| Hard-Gold Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| RewardBench Downstream Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| JudgeBench Downstream Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| RewardBench-2 Downstream Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 |
| Teacher Rubrics vs Hard-Gold Holdout Filter | `contamination_audit` | `blocked` | 1 blockers, removed=0 |
| Teacher Rubrics vs RewardBench Holdout Filter | `contamination_audit` | `missing` | report missing |
| Teacher Rubrics vs JudgeBench Holdout Filter | `contamination_audit` | `missing` | report missing |
| Teacher Rubrics vs RewardBench-2 Holdout Filter | `contamination_audit` | `missing` | report missing |
| BlindSpot SFT vs Hard-Gold Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| BlindSpot SFT vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| BlindSpot SFT vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| BlindSpot SFT vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 |
| Proxy-Gold vs Hard-Gold Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Proxy-Gold vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Proxy-Gold vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Proxy-Gold vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 |
| Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 5735 calls, 8624145 tokens, $0.0000 |
| RewardBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 13665 calls, 18440030 tokens, $0.0000 |
| JudgeBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 2640 calls, 4143735 tokens, $0.0000 |
| RewardBench-2 Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 9070 calls, 11983875 tokens, $0.0000 |
| RM-Bench Model Evaluation-Criteria API Budget | `api_budget` | `missing` | report missing |
| RM-Bench Downstream Schema Contract | `schema_contract` | `missing` | report missing |
| Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 2000 calls, 2990128 tokens, $0.0000 |
| HealthBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 25000 calls, 37499415 tokens, $0.0000 |
| WritingBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 5000 calls, 13117195 tokens, $0.0000 |
| HealthBench Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 20000 calls, 30096620 tokens, $0.0000 |
| WritingBench Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 4000 calls, 10513180 tokens, $0.0000 |
| Meta-Verifier API Budget | `api_budget` | `missing` | report missing |
| RubricBench Model Criteria Verifier API Budget | `api_budget` | `missing` | report missing |
| RewardBench Model Verifier API Budget | `api_budget` | `missing` | report missing |
| JudgeBench Model Verifier API Budget | `api_budget` | `missing` | report missing |
| RewardBench-2 Model Verifier API Budget | `api_budget` | `missing` | report missing |
| RM-Bench Model Verifier API Budget | `api_budget` | `missing` | report missing |
| HealthBench Model Verifier API Budget | `api_budget` | `missing` | report missing |
| WritingBench Model Verifier API Budget | `api_budget` | `missing` | report missing |
| RubricBench Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| RewardBench Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| HealthBench Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| JudgeBench Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| RewardBench-2 Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| RM-Bench Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| WritingBench Model Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| Teacher Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| Filtered Teacher Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| Verifier Filtering Stats | `generic` | `missing` | report missing |
| RubricBench Model Criteria Verifier Stats | `generic` | `missing` | report missing |
| Minimal Claim Audit | `audit` | `missing` | report missing |
| Main Matrix Audit | `audit` | `blocked` | 115 missing files |
| Training Completion Gate | `manual_gate` | `blocked` | 0/6 required paths present; 0/4 JSON contracts valid; 0/5 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid |
| Matrix Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid |
| JudgeBench Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid |
| RewardBench-2 Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid |
| Generalization Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid |
| Downstream Policy RLVR Completion Gate | `manual_gate` | `blocked` | 0/6 required paths present; 0/1 JSON contracts valid; 0/0 JSON contains contracts valid; 0/12 JSON equals contracts valid |
| RubricBench Gold Validation | `gold_validation` | `pass` | 1147 records, 0 blockers |
| ResearchRubrics Gold Validation | `gold_validation` | `pass` | 101 records, 0 blockers |
| Data Source Report | `data_source_report` | `pass` | 10 scoped datasets, 0 blockers |

## Pack Builder Manifest

| Manifest | Status | Summary |
| --- | --- | --- |
| Data Pipeline Manifest | `pass` | 67 required files, 0 summaries |
| Training Manifest | `pass` | 0 required files, 0 summaries |
| Downstream RLVR Manifest | `missing` | manifest missing |
| SFT Data Manifest | `pass` | 49 required files, 0 summaries |
| Main Matrix Manifest | `pass` | 117 required files, 10 summaries |
| Generalization Matrix Manifest | `pass` | 40 required files, 4 summaries |
| JudgeBench Matrix Manifest | `pass` | 74 required files, 10 summaries |
| RewardBench-2 Matrix Manifest | `pass` | 74 required files, 10 summaries |
| Rebuttal Pack Manifest | `warn` | 11 rebuttal entries, answer_ready=0, needs_readiness=0, needs_evidence=11, cannot_claim=0, missing_claim_mapping=0, readiness_ok=False, templates=11 |

### Manifest Diagnostics

- Rebuttal Pack Manifest: warning: rebuttal pack was built while submission readiness was false
- Rebuttal Pack Manifest: warning: rebuttal pack has no answer_ready entries

## BSC Metrics

| Method | Status | N | Cov | Blind | Red | Hall | Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `missing` | 0 |  |  |  |  |  |
| gpt4o | `missing` | 0 |  |  |  |  |  |
| claude | `missing` | 0 |  |  |  |  |  |
| sft_only | `missing` | 0 |  |  |  |  |  |
| sft_rl | `missing` | 0 |  |  |  |  |  |

## Downstream Metrics

| Method | Status | N | Accuracy | Tie | Margin | Scorer | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| base | `missing` |  |  |  |  | None | None |
| gpt4o | `missing` |  |  |  |  | None | None |
| claude | `missing` |  |  |  |  | None | None |
| sft_only | `missing` |  |  |  |  | None | None |
| sft_rl | `missing` |  |  |  |  | None | None |
| judgebench_base | `missing` |  |  |  |  | None | None |
| judgebench_gpt4o | `missing` |  |  |  |  | None | None |
| judgebench_claude | `missing` |  |  |  |  | None | None |
| judgebench_sft_only | `missing` |  |  |  |  | None | None |
| judgebench_sft_rl | `missing` |  |  |  |  | None | None |
| rewardbench2_base | `missing` |  |  |  |  | None | None |
| rewardbench2_gpt4o | `missing` |  |  |  |  | None | None |
| rewardbench2_claude | `missing` |  |  |  |  | None | None |
| rewardbench2_sft_only | `missing` |  |  |  |  | None | None |
| rewardbench2_sft_rl | `missing` |  |  |  |  | None | None |

## Dashboard Diagnostics

| Section | Status | Execution | Training Chain | Prerequisites | Hard Blockers | Claim Ladder | Summary |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Submission Gap Report | `blocked` | 6/6 | 0 | 59 | 43 | - | 8 phases, blocked=8, readiness_ok=False, hard_blockers=43, execution_steps=6, training_chain_steps=0, prereq_items=59 |
| Rebuttal Pack Manifest | `warn` | 0/0 | 0 | 0 | 0 | 0/4 safe; blocked: motivation, metric-support, method-support, judge-utility support | 11 entries, answer_ready=0, needs_readiness=0, needs_evidence=11, cannot_claim=0, missing_claim_mapping=0, readiness_ok=False, claim_ladder_safe=0/4 |

## Confidence Intervals

### Base BSC Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### GPT-4o BSC Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### Claude BSC Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### SFT-only BSC Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### SFT+RL BSC Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### Base Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### GPT-4o Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### Claude Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### SFT-only Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### SFT+RL Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### JudgeBench Base Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### JudgeBench SFT+RL Downstream Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### RewardBench-2 Base Multi-Candidate Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |

### RewardBench-2 SFT+RL Multi-Candidate Bootstrap CI

- Status: `missing`
- Rows: `0`
- Confidence: ``

| Metric | N | Mean | CI Lower | CI Upper | Status |
| --- | ---: | ---: | ---: | ---: | --- |


## Claim Ladder

| Level | Status | Required Claims | Evidence Required | Paper Sentence | Downgrade Rule |
| --- | --- | --- | --- | --- | --- |
| motivation | `missing_evidence` | C1, C6 | Frozen 100-example hard-gold diagnostic with C1 and threshold-robustness C6 gates. | If C1/C6 pass, report systematic evaluation blind spots for the base evaluation-criteria policy. | Does not support any trained-method claim by itself. |
| motivation blockers |  |  | C1: missing_evidence; C6: missing_evidence |  |  |
| metric-support | `missing_evidence` | C0, C2, C3 | Hard-gold BSC coverage change with stable redundancy and hallucination under the fixed protocol. | If C0/C2/C3 pass, report a hard-gold BSC coverage change as metric evidence. | Without downstream support, report metric-only BSC evidence. |
| metric-support blockers |  |  | C0: missing_evidence; C2: missing_evidence; C3: missing_evidence |  |  |
| method-support | `missing_evidence` | C5, C7, C14 | SFT-only vs SFT+GRPO and reward-component ablations pass C5/C7/C14. | If C5/C7/C14 pass, attribute the coverage change to the RLVR reward stage. | Without C14, report proxy-gold supervision evidence rather than RLVR evidence. |
| method-support blockers |  |  | C5: missing_evidence; C7: missing_evidence; C14: missing_evidence |  |  |
| judge-utility support | `missing_evidence` | C0, C4, C9, C10, C12 | RewardBench, RewardBench-2, and JudgeBench API/model scorer rows pass C4/C9/C10. | If C0/C4/C9/C10/C12 pass, report held-out downstream judge-utility support. | Without C12, write aggregate coverage change rather than dimension-level recovery; without C0, no trained-method row is paper-facing. |
| judge-utility support blockers |  |  | C0: missing_evidence; C4: missing_evidence; C9: missing_evidence; C10: missing_evidence; C12: missing_evidence |  |  |

## Claim Decision

- Safe claim: Report only claims whose evidence matrix status is safe_to_claim, whose submission-readiness full required evidence set (C0, C1, C2, C3, C4, C5, C6, C7, C9, C10, C12, C13, C14) passes, and whose Raw Audit Gates pass.
- Deferred claim: Keep single-model blind spot magnitude, teacher-union effects, RL-stage coverage-change support, and downstream utility-support claims deferred until real hard-gold data, generation, training, ablation, and readiness artifacts exist.

### Claim Discipline

- Do not write empirical claims unless the relevant evidence matrix rows are safe_to_claim and readiness is ok.
- Do not treat this package as AAAI-ready while submission readiness is false.
- SFT+GRPO coverage, dimension-level recovery, downstream utility, ablation, and semantic-space claims are permitted only after all required C0-C14 evidence rows are safe_to_claim.
- Do not claim clean hard-gold/proxy/downstream isolation until C0 passes with query-disjoint holdouts and SHA-bound provenance.
- Treat BSC changes as metric evidence only; do not describe them as dimension-level recovery until C3, C12, and C14 pass.
- Do not claim downstream judge-utility support until C4/C9/C10 pass with API/model scorer outputs and paper_claim_eligible summaries.
- Do not claim ablation support until C5/C7 pass with separately trained reward-component variants and verifier-filter evidence.
- Treat semantic-space plots as illustrative until C13 verifies SVG/PDF/CSV/JSON assets and point-level provenance.
- Do not submit the paper package until required paper-facing tables, figures, and reviewer-facing docs are synced and indexed.
- With zero safe_to_claim rows, restrict paper-facing empirical content to planned protocol and clearly marked diagnostic evidence.
- Treat result-card metrics as diagnostic until the Evidence Matrix is present, readable, and every required paper-facing row is safe_to_claim.
- Do not claim clean hard-gold/proxy/downstream isolation until C0 is safe_to_claim with query-disjoint holdouts and SHA-bound provenance.
- Treat BSC coverage changes as metric evidence only; do not call them dimension-level recovery until C3, C12, and C14 pass.
- Do not claim ablation support until C5/C7 pass; offline reward re-scoring remains diagnostic, not paper-facing ablation evidence.
- Treat semantic-space plots as illustrative until C13 verifies point-level provenance, nearest-gold audit fields, and SVG/PDF/CSV/JSON assets.

### Blockers

- Raw audit gate `Real Run Preflight` is blocked: 33 blockers, 8 warnings; blockers: missing input file: data/processed/rmbench_queries.jsonl; missing input file: data/processed/healthbench_hard_queries.jsonl; missing input file: data/processed/arenahard_queries.jsonl; missing API env LOCAL_OPENAI_API_KEY for provider base; missing API env GPT_AK_1 for provider gpt-5.4; missing API env GPT_AK_2 for provider gpt-5; missing API env GPT_AK_3 for provider gpt-4o; missing API env GPT_AK_1 for provider gemini; missing API env GPT_AK_1 for provider gemini-2.5-pro; missing API env GPT_AK_1 for provider gpt-5.4; missing API env GPT_AK_2 for provider gpt-5; missing API env GPT_AK_3 for provider gpt-4o; missing API env GPT_AK_3 for provider meta-verifier; missing provider config: configs/judge_scorer.local.jsonl; missing required provider: gpt4o; missing required provider: claude; missing required provider: sft_only; missing required provider: sft_rl; missing required provider: deepseek; missing required provider: qwen; missing required provider: judge-scorer; missing required provider in configs/generators.local.jsonl: gpt4o; missing required provider in configs/generators.local.jsonl: claude; missing required provider in configs/generators.local.jsonl: sft_only; missing required provider in configs/generators.local.jsonl: sft_rl; missing required provider in configs/providers.local.jsonl: deepseek; missing required provider in configs/providers.local.jsonl: qwen; missing required provider in configs/judge_scorer.local.jsonl: judge-scorer; missing required env var: LOCAL_OPENAI_API_KEY; missing required env var: OPENAI_API_KEY; missing required env var: ANTHROPIC_API_KEY; missing required env var: DEEPSEEK_API_KEY; missing required env var: DASHSCOPE_API_KEY
- Raw audit gate `SFT Data Preflight` is blocked: 8 blockers, 0 warnings; blockers: missing API env GPT_AK_1 for provider gemini-2.5-pro; missing API env GPT_AK_1 for provider gpt-5.4; missing API env GPT_AK_2 for provider gpt-5; missing API env GPT_AK_3 for provider gpt-4o; missing API env GPT_AK_3 for provider meta-verifier; missing required env var: GPT_AK_1; missing required env var: GPT_AK_2; missing required env var: GPT_AK_3
- Raw audit gate `Data Readiness Audit` is blocked: 10 missing files; blockers: data/raw/rmbench_raw.jsonl; outputs/data_profiles/rmbench_schema.json; outputs/schema_contracts/rmbench_downstream_schema.json; data/processed/rmbench_queries.jsonl; data/raw/healthbench_hard_raw.jsonl; outputs/data_profiles/healthbench_hard_schema.json; data/processed/healthbench_hard_queries.jsonl; data/raw/arenahard_raw.jsonl; outputs/data_profiles/arenahard_schema.json; data/processed/arenahard_queries.jsonl
- Raw audit gate `AAAI LaTeX Compile` is blocked: pdf_bytes=0, pages=0, max_pages=0, official_style_active=False, bibliography_style_active=False, anonymous_author_declared=True; blockers: pdflatex binary is not available; missing compiled PDF: paper/main.pdf; missing LaTeX log: paper/main.log
- Raw audit gate `Teacher Rubrics vs Hard-Gold Holdout Filter` is blocked: 1 blockers, removed=0; blockers: input missing or empty: data/processed/teacher_rubrics_filtered.jsonl
- Raw audit gate `Teacher Rubrics vs RewardBench Holdout Filter` is missing: report missing
- Raw audit gate `Teacher Rubrics vs JudgeBench Holdout Filter` is missing: report missing
- Raw audit gate `Teacher Rubrics vs RewardBench-2 Holdout Filter` is missing: report missing
- Raw audit gate `RM-Bench Model Evaluation-Criteria API Budget` is missing: report missing
- Raw audit gate `RM-Bench Downstream Schema Contract` is missing: report missing
- Raw audit gate `Meta-Verifier API Budget` is missing: report missing
- Raw audit gate `RubricBench Model Criteria Verifier API Budget` is missing: report missing
- Raw audit gate `RewardBench Model Verifier API Budget` is missing: report missing
- Raw audit gate `JudgeBench Model Verifier API Budget` is missing: report missing
- Raw audit gate `RewardBench-2 Model Verifier API Budget` is missing: report missing
- Raw audit gate `RM-Bench Model Verifier API Budget` is missing: report missing
- Raw audit gate `HealthBench Model Verifier API Budget` is missing: report missing
- Raw audit gate `WritingBench Model Verifier API Budget` is missing: report missing
- Raw audit gate `RubricBench Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `RewardBench Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `HealthBench Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `JudgeBench Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `RewardBench-2 Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `RM-Bench Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `WritingBench Model Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `Teacher Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `Filtered Teacher Evaluation-Criteria Validation` is missing: report missing
- Raw audit gate `Verifier Filtering Stats` is missing: report missing
- Raw audit gate `RubricBench Model Criteria Verifier Stats` is missing: report missing
- Raw audit gate `Minimal Claim Audit` is missing: report missing
- Raw audit gate `Main Matrix Audit` is blocked: 115 missing files; blockers: data/processed/matrix_real/base/bsc_eval.jsonl; outputs/matrix_real/base/bsc_join_report.json; outputs/matrix_real/base/bsc/summary.json; data/processed/matrix_real/base/downstream_eval.jsonl; outputs/matrix_real/base/downstream_join_report.json; outputs/matrix_real/base/downstream/summary.json; outputs/matrix_real/base/downstream_api_budget/budget.json; outputs/matrix_real/base/downstream_api_budget/budget.md; outputs/matrix_real/base/bsc_ci/bootstrap_ci.json; outputs/matrix_real/base/bsc_ci/bootstrap_ci.csv; outputs/matrix_real/base/bsc_ci/bootstrap_ci.md; outputs/matrix_real/base/downstream_ci/bootstrap_ci.json; outputs/matrix_real/base/downstream_ci/bootstrap_ci.csv; outputs/matrix_real/base/downstream_ci/bootstrap_ci.md; outputs/matrix_real/base/bsc_sweep/threshold_sweep.csv; outputs/matrix_real/base/bsc_sweep/threshold_sweep.json; outputs/matrix_real/base/bsc_sweep/threshold_sweep.md; outputs/matrix_real/base/bsc_human_audit_pack/summary.json; outputs/matrix_real/base/bsc_human_audit_pack/audit_items.csv; outputs/matrix_real/base/bsc_human_audit_pack/audit_items.jsonl; outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.json; outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.md; data/processed/matrix_real/gpt4o/bsc_eval.jsonl; outputs/matrix_real/gpt4o/bsc_join_report.json; outputs/matrix_real/gpt4o/bsc/summary.json; data/processed/matrix_real/gpt4o/downstream_eval.jsonl; outputs/matrix_real/gpt4o/downstream_join_report.json; outputs/matrix_real/gpt4o/downstream/summary.json; outputs/matrix_real/gpt4o/downstream_api_budget/budget.json; outputs/matrix_real/gpt4o/downstream_api_budget/budget.md; outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.json; outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.csv; outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.md; outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.json; outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.csv; outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.md; data/processed/matrix_real/claude/bsc_eval.jsonl; outputs/matrix_real/claude/bsc_join_report.json; outputs/matrix_real/claude/bsc/summary.json; data/processed/matrix_real/claude/downstream_eval.jsonl; outputs/matrix_real/claude/downstream_join_report.json; outputs/matrix_real/claude/downstream/summary.json; outputs/matrix_real/claude/downstream_api_budget/budget.json; outputs/matrix_real/claude/downstream_api_budget/budget.md; outputs/matrix_real/claude/bsc_ci/bootstrap_ci.json; outputs/matrix_real/claude/bsc_ci/bootstrap_ci.csv; outputs/matrix_real/claude/bsc_ci/bootstrap_ci.md; outputs/matrix_real/claude/downstream_ci/bootstrap_ci.json; outputs/matrix_real/claude/downstream_ci/bootstrap_ci.csv; outputs/matrix_real/claude/downstream_ci/bootstrap_ci.md; data/processed/matrix_real/sft_only/bsc_eval.jsonl; outputs/matrix_real/sft_only/bsc_join_report.json; outputs/matrix_real/sft_only/bsc/summary.json; data/processed/matrix_real/sft_only/downstream_eval.jsonl; outputs/matrix_real/sft_only/downstream_join_report.json; outputs/matrix_real/sft_only/downstream/summary.json; outputs/matrix_real/sft_only/downstream_api_budget/budget.json; outputs/matrix_real/sft_only/downstream_api_budget/budget.md; outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json; outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.csv; outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.md; outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.json; outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.csv; outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.md; data/processed/matrix_real/sft_rl/bsc_eval.jsonl; outputs/matrix_real/sft_rl/bsc_join_report.json; outputs/matrix_real/sft_rl/bsc/summary.json; data/processed/matrix_real/sft_rl/downstream_eval.jsonl; outputs/matrix_real/sft_rl/downstream_join_report.json; outputs/matrix_real/sft_rl/downstream/summary.json; outputs/matrix_real/sft_rl/downstream_api_budget/budget.json; outputs/matrix_real/sft_rl/downstream_api_budget/budget.md; outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json; outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.csv; outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.md; outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.json; outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.csv; outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.md; outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.csv; outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.json; outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.md; outputs/matrix_real/sft_rl/bsc_human_audit_pack/summary.json; outputs/matrix_real/sft_rl/bsc_human_audit_pack/audit_items.csv; outputs/matrix_real/sft_rl/bsc_human_audit_pack/audit_items.jsonl; outputs/matrix_real/sft_rl/bsc_human_audit_pack/human_label_summary.json; outputs/matrix_real/sft_rl/bsc_human_audit_pack/human_label_summary.md; outputs/matrix_real/main_table.csv; outputs/matrix_real/main_table.md; outputs/bsc_ablation/ablation_summary.csv; outputs/bsc_ablation/ablation_summary.md; outputs/bsc_ablation/variants/full_summary.json; outputs/bsc_ablation/variants/no_red_summary.json; outputs/bsc_ablation/variants/no_valid_summary.json; outputs/bsc_ablation/variants/no_verifier_summary.json; outputs/bsc_ablation/variants/cov_only_summary.json; outputs/teacher_union_ablation/teacher_union_per_item.csv; outputs/teacher_union_ablation/teacher_union_ablation.csv; outputs/teacher_union_ablation/teacher_union_ablation.md; outputs/teacher_union_ablation/teacher_union_ablation.json; outputs/verifier_filter_ablation/verifier_filter_per_item.csv; outputs/verifier_filter_ablation/verifier_filter_ablation.csv; outputs/verifier_filter_ablation/verifier_filter_ablation.md; outputs/verifier_filter_ablation/verifier_filter_ablation.json; outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json; outputs/matrix_real/dimension_transition/base_to_sft_only/transition_per_item.csv; outputs/matrix_real/dimension_transition/base_to_sft_only/transition_by_category.csv; outputs/matrix_real/dimension_transition/base_to_sft_only/transition_gold_items.jsonl; outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json; outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_per_item.csv; outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_by_category.csv; outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_gold_items.jsonl; outputs/matrix_real/semantic_space/semantic_space.svg; outputs/matrix_real/semantic_space/semantic_space.pdf; outputs/matrix_real/semantic_space/semantic_space_points.csv; outputs/matrix_real/semantic_space/semantic_space_summary.json
- Raw audit gate `Training Completion Gate` is blocked: 0/6 required paths present; 0/4 JSON contracts valid; 0/5 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid; blockers: missing required path: outputs/checkpoints/evaluation_criteria_policy_sft; missing required path: outputs/checkpoints/evaluation_criteria_policy_rl; missing required path: outputs/training_commands/training_done.json; missing required path: outputs/verifier/teacher_rubrics_filtered_report.json; missing required path: outputs/sft_data/proxy_gold_build_report.json; missing required path: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Raw audit gate `Matrix Trained Method Gate` is blocked: 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid; blockers: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json; missing required path: outputs/sft_data/proxy_gold_build_report.json; missing required path: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Raw audit gate `JudgeBench Trained Method Gate` is blocked: 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid; blockers: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json; missing required path: outputs/sft_data/proxy_gold_build_report.json; missing required path: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Raw audit gate `RewardBench-2 Trained Method Gate` is blocked: 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid; blockers: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json; missing required path: outputs/sft_data/proxy_gold_build_report.json; missing required path: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Raw audit gate `Generalization Trained Method Gate` is blocked: 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid; 0/10 JSON SHA256 contracts valid; blockers: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json; missing required path: outputs/sft_data/proxy_gold_build_report.json; missing required path: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/training_commands/training_done.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_build_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json; missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Raw audit gate `Downstream Policy RLVR Completion Gate` is blocked: 0/6 required paths present; 0/1 JSON contracts valid; 0/0 JSON contains contracts valid; 0/12 JSON equals contracts valid; blockers: missing required path: outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json; missing required path: outputs/policy_rlvr/healthbench_hard_policy; missing required path: outputs/policy_rlvr/healthbench_hard_eval.json; missing required path: outputs/policy_rlvr/arenahard_policy; missing required path: outputs/policy_rlvr/arenahard_eval.json; missing required path: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json; missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Manifest `Downstream RLVR Manifest` is missing: manifest missing
- At least one BSC summary is missing or blocked
- At least one required downstream summary is missing, blocked, or not paper-eligible
- readiness report is blocked
- dashboard report is blocked

### Warnings

- Manifest `Rebuttal Pack Manifest` is warn
- Evidence matrix status is warn

## Notes

- This template is intentionally strict: missing real outputs keep claims deferred or blocked.
- Do not replace hard-gold RubricBench evidence with toy or proxy evidence.
