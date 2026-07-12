# BlindSpot-RL Real-Run Dashboard

- Overall status: `blocked`
- Objective: Keep BSC claims evidence-gated from minimal motivation through SFT/GRPO and paper readiness.
- Blockers: `416`
- Warnings: `67`

| Section | Type | Status | Summary | Path |
| --- | --- | --- | --- | --- |
| Data Source Report | `data_source_report` | `pass` | 10 datasets, 0 blockers, 0 warnings | `outputs/data_sources/source_report.json` |
| RubricBench Gold Validation | `gold_validation` | `pass` | 1147 records, min=100, 0 blockers | `outputs/data_validation/rubricbench_gold.json` |
| ResearchRubrics Gold Validation | `gold_validation` | `pass` | 101 records, min=100, 0 blockers | `outputs/data_validation/researchrubrics_gold.json` |
| Real Run Preflight | `preflight` | `blocked` | 33 blockers, 8 warnings | `outputs/preflight/real_run_preflight.json` |
| SFT Data Preflight | `preflight` | `blocked` | 8 blockers, 0 warnings | `outputs/preflight/sft_data_preflight.json` |
| RewardBench Proxy-Train Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 | `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json` |
| Hard-Gold Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/hard_gold_holdout_contamination.json` |
| RewardBench Downstream Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json` |
| RewardBench Proxy-Train vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench_holdout_filter.json` |
| RewardBench Proxy-Train vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_judgebench_holdout_filter.json` |
| RewardBench Proxy-Train vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=5 | `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json` |
| RewardBench-2 Query Pool vs RubricBench Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=50 | `outputs/contamination_audit/rewardbench2_queries_rubricbench_train_seed_filter.json` |
| RewardBench-2 Query Pool vs ResearchRubrics Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/rewardbench2_queries_researchrubrics_train_seed_filter.json` |
| RewardBench-2 Multicandidate vs RubricBench Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=50 | `outputs/contamination_audit/rewardbench2_multicandidate_rubricbench_train_seed_filter.json` |
| RewardBench-2 Multicandidate vs ResearchRubrics Train-Seed Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json` |
| Pre-SFT Clean Proxy-Train vs Hard-Gold Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json` |
| Pre-SFT Clean Proxy-Train vs RewardBench Holdout Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/clean_proxy_train_vs_rewardbench_downstream_audit.json` |
| Pre-SFT Clean Proxy-Train vs JudgeBench Holdout Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/clean_proxy_train_vs_judgebench_downstream_audit.json` |
| Pre-SFT Clean Proxy-Train vs RewardBench-2 Holdout Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/clean_proxy_train_vs_rewardbench2_downstream_audit.json` |
| JudgeBench Downstream Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/judgebench_downstream_holdout_contamination.json` |
| RewardBench-2 Downstream Holdout Contamination Audit | `contamination_audit` | `pass` | 0 blockers, artifact_status=complete, overlap_status=clear, overlaps=0 | `outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json` |
| Teacher Rubrics vs Hard-Gold Holdout Filter | `contamination_audit` | `blocked` | 1 blockers, removed=0 | `outputs/contamination_audit/teacher_rubrics_hard_gold_holdout_filter.json` |
| Teacher Rubrics vs RewardBench Holdout Filter | `contamination_audit` | `warn` | report is missing | `outputs/contamination_audit/teacher_rubrics_rewardbench_holdout_filter.json` |
| Teacher Rubrics vs JudgeBench Holdout Filter | `contamination_audit` | `warn` | report is missing | `outputs/contamination_audit/teacher_rubrics_judgebench_holdout_filter.json` |
| Teacher Rubrics vs RewardBench-2 Holdout Filter | `contamination_audit` | `warn` | report is missing | `outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json` |
| BlindSpot SFT vs Hard-Gold Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/blindspot_sft_hard_gold_holdout_filter.json` |
| BlindSpot SFT vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/blindspot_sft_rewardbench_holdout_filter.json` |
| BlindSpot SFT vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/blindspot_sft_judgebench_holdout_filter.json` |
| BlindSpot SFT vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 | `outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json` |
| Proxy-Gold vs Hard-Gold Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/proxy_gold_hard_gold_holdout_filter.json` |
| Proxy-Gold vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/proxy_gold_rewardbench_holdout_filter.json` |
| Proxy-Gold vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 | `outputs/contamination_audit/proxy_gold_judgebench_holdout_filter.json` |
| Proxy-Gold vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 | `outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json` |
| Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 5735 calls, 8624145 tokens, $0.0000 | `outputs/api_budget/model_rubrics_budget.json` |
| RewardBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 13665 calls, 18440030 tokens, $0.0000 | `outputs/api_budget/rewardbench_model_rubrics_budget.json` |
| JudgeBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 2640 calls, 4143735 tokens, $0.0000 | `outputs/api_budget/judgebench_model_rubrics_budget.json` |
| RewardBench-2 Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 9070 calls, 11983875 tokens, $0.0000 | `outputs/api_budget/rewardbench2_model_rubrics_budget.json` |
| RM-Bench Model Evaluation-Criteria API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/rmbench_model_rubrics_budget.json` |
| Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 2000 calls, 2990128 tokens, $0.0000 | `outputs/api_budget/teacher_rubrics_budget.json` |
| HealthBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 25000 calls, 37499415 tokens, $0.0000 | `outputs/api_budget/healthbench_model_rubrics_budget.json` |
| WritingBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 5000 calls, 13117195 tokens, $0.0000 | `outputs/api_budget/writingbench_model_rubrics_budget.json` |
| HealthBench Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 20000 calls, 30096620 tokens, $0.0000 | `outputs/api_budget/healthbench_teacher_rubrics_budget.json` |
| WritingBench Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 4000 calls, 10513180 tokens, $0.0000 | `outputs/api_budget/writingbench_teacher_rubrics_budget.json` |
| Meta-Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/meta_verifier_budget.json` |
| RubricBench Model Criteria Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/rubricbench_model_rubrics_verifier_budget.json` |
| RewardBench Model Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/rewardbench_model_rubrics_verifier_budget.json` |
| JudgeBench Model Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/judgebench_model_rubrics_verifier_budget.json` |
| RewardBench-2 Model Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/rewardbench2_model_rubrics_verifier_budget.json` |
| RM-Bench Model Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/rmbench_model_rubrics_verifier_budget.json` |
| HealthBench Model Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/healthbench_model_rubrics_verifier_budget.json` |
| WritingBench Model Verifier API Budget | `api_budget` | `warn` | report is missing | `outputs/api_budget/writingbench_model_rubrics_verifier_budget.json` |
| RM-Bench Downstream Schema Contract | `schema_contract` | `warn` | report is missing | `outputs/schema_contracts/rmbench_downstream_schema.json` |
| RubricBench Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/rubricbench_model_rubrics/validation_report.json` |
| RewardBench Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/rewardbench_model_rubrics/validation_report.json` |
| HealthBench Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/healthbench_model_rubrics/validation_report.json` |
| JudgeBench Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/judgebench_model_rubrics/validation_report.json` |
| RewardBench-2 Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/rewardbench2_model_rubrics/validation_report.json` |
| RM-Bench Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/rmbench_model_rubrics/validation_report.json` |
| WritingBench Model Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/writingbench_model_rubrics/validation_report.json` |
| Teacher Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/teacher_rubrics/validation_report.json` |
| Filtered Teacher Evaluation-Criteria Validation | `validation` | `warn` | report is missing | `outputs/validation/teacher_rubrics_filtered/validation_report.json` |
| RubricBench Model Criteria Verifier Stats | `generic` | `warn` | report is missing | `outputs/verifier/rubricbench_model_rubrics_stats.jsonl` |
| Minimal Claim BSC Bootstrap CI | `confidence_interval` | `pass` | 5 metrics, n=100, confidence=0.95 | `outputs/minimal_claim/base/bsc_ci/bootstrap_ci.json` |
| Minimal Claim Audit | `audit` | `pass` | 55 present, 0 missing | `outputs/minimal_claim/base/audit_report.json` |
| Minimal Claim Evidence | `evidence` | `pass` | 2/2 safe claims | `outputs/minimal_claim/base/evidence/evidence_matrix.json` |
| Main Matrix Audit | `audit` | `blocked` | 2 present, 115 missing | `outputs/matrix_real/audit_report.json` |
| Main Matrix Evidence | `evidence` | `warn` | 0/15 safe claims | `outputs/evidence_real/evidence_matrix.json` |
| C5 Teacher-Union Ablation JSON | `generic` | `warn` | report is missing | `outputs/teacher_union_ablation/teacher_union_ablation.json` |
| C5 Teacher-Union Ablation Table | `generic` | `warn` | report is missing | `outputs/teacher_union_ablation/teacher_union_ablation.csv` |
| C5 Teacher-Union Per-Item Audit | `generic` | `warn` | report is missing | `outputs/teacher_union_ablation/teacher_union_per_item.csv` |
| Base BSC Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/base/bsc_ci/bootstrap_ci.json` |
| Base Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/base/downstream_ci/bootstrap_ci.json` |
| GPT-4o BSC Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.json` |
| GPT-4o Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.json` |
| Claude BSC Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/claude/bsc_ci/bootstrap_ci.json` |
| Claude Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/claude/downstream_ci/bootstrap_ci.json` |
| SFT-only BSC Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json` |
| SFT-only Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.json` |
| SFT+RL BSC Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json` |
| SFT+RL Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.json` |
| JudgeBench Base Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_judgebench/judgebench_base/downstream_ci/bootstrap_ci.json` |
| JudgeBench SFT+RL Downstream Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_judgebench/judgebench_sft_rl/downstream_ci/bootstrap_ci.json` |
| RewardBench-2 Base Multi-Candidate Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_rewardbench2/rewardbench2_base/downstream_ci/bootstrap_ci.json` |
| RewardBench-2 SFT+RL Multi-Candidate Bootstrap CI | `confidence_interval` | `warn` | report is missing | `outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_ci/bootstrap_ci.json` |
| Matrix Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid | `outputs/matrix_real/trained_method_gate.json` |
| JudgeBench Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid | `outputs/matrix_judgebench/trained_method_gate.json` |
| RewardBench-2 Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid | `outputs/matrix_rewardbench2/trained_method_gate.json` |
| Generalization Trained Method Gate | `manual_gate` | `blocked` | 0/3 required paths present; 0/4 JSON contracts valid; 0/6 JSON contains contracts valid; 0/21 JSON equals contracts valid | `outputs/generalization_matrix/trained_method_gate.json` |
| Downstream Policy RLVR Completion Gate | `manual_gate` | `blocked` | 0/6 required paths present; 0/1 JSON contracts valid; 0/0 JSON contains contracts valid; 0/12 JSON equals contracts valid | `outputs/policy_rlvr/downstream_rlvr_completion_gate.json` |
| Submission Readiness | `readiness` | `blocked` | ok=False | `outputs/submission_readiness/readiness_report.json` |
| Submission Gap Report | `submission_gap_report` | `blocked` | 8 phases, blocked=8, readiness_ok=False, hard_blockers=43, execution_steps=6, training_chain_steps=0, prereq_items=59 | `outputs/submission_readiness/gap_report/submission_gap_report.json` |
| Rebuttal Pack Manifest | `rebuttal_manifest` | `warn` | 11 entries, answer_ready=0, needs_readiness=0, needs_evidence=11, cannot_claim=0, missing_claim_mapping=0, readiness_ok=False, claim_ladder_safe=0/4 | `outputs/rebuttal_pack/rebuttal_pack_manifest.json` |
| AAAI LaTeX Compile | `latex_compile` | `blocked` | pdf_bytes=0, pages=0/0, official_style_active=False, bibliography_style_active=False, anonymous_author_declared=True | `outputs/submission_readiness/latex_compile_report.json` |

## Blockers

- Real Run Preflight: missing input file: data/processed/rmbench_queries.jsonl
- Real Run Preflight: missing input file: data/processed/healthbench_hard_queries.jsonl
- Real Run Preflight: missing input file: data/processed/arenahard_queries.jsonl
- Real Run Preflight: missing API env LOCAL_OPENAI_API_KEY for provider base
- Real Run Preflight: missing API env GPT_AK_1 for provider gpt-5.4
- Real Run Preflight: missing API env GPT_AK_2 for provider gpt-5
- Real Run Preflight: missing API env GPT_AK_3 for provider gpt-4o
- Real Run Preflight: missing API env GPT_AK_1 for provider gemini
- Real Run Preflight: missing API env GPT_AK_1 for provider gemini-2.5-pro
- Real Run Preflight: missing API env GPT_AK_1 for provider gpt-5.4
- Real Run Preflight: missing API env GPT_AK_2 for provider gpt-5
- Real Run Preflight: missing API env GPT_AK_3 for provider gpt-4o
- Real Run Preflight: missing API env GPT_AK_3 for provider meta-verifier
- Real Run Preflight: missing provider config: configs/judge_scorer.local.jsonl
- Real Run Preflight: missing required provider: gpt4o
- Real Run Preflight: missing required provider: claude
- Real Run Preflight: missing required provider: sft_only
- Real Run Preflight: missing required provider: sft_rl
- Real Run Preflight: missing required provider: deepseek
- Real Run Preflight: missing required provider: qwen
- Real Run Preflight: missing required provider: judge-scorer
- Real Run Preflight: missing required provider in configs/generators.local.jsonl: gpt4o
- Real Run Preflight: missing required provider in configs/generators.local.jsonl: claude
- Real Run Preflight: missing required provider in configs/generators.local.jsonl: sft_only
- Real Run Preflight: missing required provider in configs/generators.local.jsonl: sft_rl
- Real Run Preflight: missing required provider in configs/providers.local.jsonl: deepseek
- Real Run Preflight: missing required provider in configs/providers.local.jsonl: qwen
- Real Run Preflight: missing required provider in configs/judge_scorer.local.jsonl: judge-scorer
- Real Run Preflight: missing required env var: LOCAL_OPENAI_API_KEY
- Real Run Preflight: missing required env var: OPENAI_API_KEY
- Real Run Preflight: missing required env var: ANTHROPIC_API_KEY
- Real Run Preflight: missing required env var: DEEPSEEK_API_KEY
- Real Run Preflight: missing required env var: DASHSCOPE_API_KEY
- SFT Data Preflight: missing API env GPT_AK_1 for provider gemini-2.5-pro
- SFT Data Preflight: missing API env GPT_AK_1 for provider gpt-5.4
- SFT Data Preflight: missing API env GPT_AK_2 for provider gpt-5
- SFT Data Preflight: missing API env GPT_AK_3 for provider gpt-4o
- SFT Data Preflight: missing API env GPT_AK_3 for provider meta-verifier
- SFT Data Preflight: missing required env var: GPT_AK_1
- SFT Data Preflight: missing required env var: GPT_AK_2
- SFT Data Preflight: missing required env var: GPT_AK_3
- Teacher Rubrics vs Hard-Gold Holdout Filter: input missing or empty: data/processed/teacher_rubrics_filtered.jsonl
- Main Matrix Audit: missing artifact data/processed/matrix_real/base/bsc_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc/summary.json
- Main Matrix Audit: missing artifact data/processed/matrix_real/base/downstream_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream_api_budget/budget.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream_api_budget/budget.md
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/base/downstream_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_sweep/threshold_sweep.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_sweep/threshold_sweep.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_sweep/threshold_sweep.md
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_human_audit_pack/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_human_audit_pack/audit_items.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_human_audit_pack/audit_items.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.md
- Main Matrix Audit: missing artifact data/processed/matrix_real/gpt4o/bsc_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/bsc_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/bsc/summary.json
- Main Matrix Audit: missing artifact data/processed/matrix_real/gpt4o/downstream_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream_api_budget/budget.json
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream_api_budget/budget.md
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/bsc_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/gpt4o/downstream_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact data/processed/matrix_real/claude/bsc_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/bsc_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/bsc/summary.json
- Main Matrix Audit: missing artifact data/processed/matrix_real/claude/downstream_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream_api_budget/budget.json
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream_api_budget/budget.md
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/bsc_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/bsc_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/bsc_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/claude/downstream_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact data/processed/matrix_real/sft_only/bsc_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/bsc_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/bsc/summary.json
- Main Matrix Audit: missing artifact data/processed/matrix_real/sft_only/downstream_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream_api_budget/budget.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream_api_budget/budget.md
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_only/downstream_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact data/processed/matrix_real/sft_rl/bsc_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc/summary.json
- Main Matrix Audit: missing artifact data/processed/matrix_real/sft_rl/downstream_eval.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream_join_report.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream_api_budget/budget.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream_api_budget/budget.md
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.md
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.md
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_human_audit_pack/summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_human_audit_pack/audit_items.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_human_audit_pack/audit_items.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_human_audit_pack/human_label_summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/sft_rl/bsc_human_audit_pack/human_label_summary.md
- Main Matrix Audit: missing artifact outputs/matrix_real/main_table.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/main_table.md
- Main Matrix Audit: missing artifact outputs/bsc_ablation/ablation_summary.csv
- Main Matrix Audit: missing artifact outputs/bsc_ablation/ablation_summary.md
- Main Matrix Audit: missing artifact outputs/bsc_ablation/variants/full_summary.json
- Main Matrix Audit: missing artifact outputs/bsc_ablation/variants/no_red_summary.json
- Main Matrix Audit: missing artifact outputs/bsc_ablation/variants/no_valid_summary.json
- Main Matrix Audit: missing artifact outputs/bsc_ablation/variants/no_verifier_summary.json
- Main Matrix Audit: missing artifact outputs/bsc_ablation/variants/cov_only_summary.json
- Main Matrix Audit: missing artifact outputs/teacher_union_ablation/teacher_union_per_item.csv
- Main Matrix Audit: missing artifact outputs/teacher_union_ablation/teacher_union_ablation.csv
- Main Matrix Audit: missing artifact outputs/teacher_union_ablation/teacher_union_ablation.md
- Main Matrix Audit: missing artifact outputs/teacher_union_ablation/teacher_union_ablation.json
- Main Matrix Audit: missing artifact outputs/verifier_filter_ablation/verifier_filter_per_item.csv
- Main Matrix Audit: missing artifact outputs/verifier_filter_ablation/verifier_filter_ablation.csv
- Main Matrix Audit: missing artifact outputs/verifier_filter_ablation/verifier_filter_ablation.md
- Main Matrix Audit: missing artifact outputs/verifier_filter_ablation/verifier_filter_ablation.json
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_only/transition_per_item.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_only/transition_by_category.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_only/transition_gold_items.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_per_item.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_by_category.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_gold_items.jsonl
- Main Matrix Audit: missing artifact outputs/matrix_real/semantic_space/semantic_space.svg
- Main Matrix Audit: missing artifact outputs/matrix_real/semantic_space/semantic_space.pdf
- Main Matrix Audit: missing artifact outputs/matrix_real/semantic_space/semantic_space_points.csv
- Main Matrix Audit: missing artifact outputs/matrix_real/semantic_space/semantic_space_summary.json
- Main Matrix Audit: failed summary check base_bsc
- Main Matrix Audit: failed summary check base_downstream
- Main Matrix Audit: failed summary check gpt4o_bsc
- Main Matrix Audit: failed summary check gpt4o_downstream
- Main Matrix Audit: failed summary check claude_bsc
- Main Matrix Audit: failed summary check claude_downstream
- Main Matrix Audit: failed summary check sft_only_bsc
- Main Matrix Audit: failed summary check sft_only_downstream
- Main Matrix Audit: failed summary check sft_rl_bsc
- Main Matrix Audit: failed summary check sft_rl_downstream
- Matrix Trained Method Gate: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Matrix Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- JudgeBench Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- RewardBench-2 Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required path: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required path: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/training_commands/training_done.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/verifier/teacher_rubrics_filtered_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_build_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Generalization Trained Method Gate: missing required JSON file: outputs/sft_data/proxy_gold_verl_report.json
- Downstream Policy RLVR Completion Gate: missing required path: outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json
- Downstream Policy RLVR Completion Gate: missing required path: outputs/policy_rlvr/healthbench_hard_policy
- Downstream Policy RLVR Completion Gate: missing required path: outputs/policy_rlvr/healthbench_hard_eval.json
- Downstream Policy RLVR Completion Gate: missing required path: outputs/policy_rlvr/arenahard_policy
- Downstream Policy RLVR Completion Gate: missing required path: outputs/policy_rlvr/arenahard_eval.json
- Downstream Policy RLVR Completion Gate: missing required path: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Downstream Policy RLVR Completion Gate: missing required JSON file: outputs/policy_rlvr/downstream_rlvr_done.json
- Submission Readiness: audit report is not ok
- Submission Readiness: required evidence claim C0 is missing_evidence
- Submission Readiness: required evidence claim C1 is missing_evidence
- Submission Readiness: required evidence claim C2 is missing_evidence
- Submission Readiness: required evidence claim C3 is missing_evidence
- Submission Readiness: required evidence claim C4 is missing_evidence
- Submission Readiness: required evidence claim C5 is missing_evidence
- Submission Readiness: required evidence claim C6 is missing_evidence
- Submission Readiness: required evidence claim C7 is missing_evidence
- Submission Readiness: required evidence claim C9 is missing_evidence
- Submission Readiness: required evidence claim C10 is missing_evidence
- Submission Readiness: required evidence claim C12 is missing_evidence
- Submission Readiness: required evidence claim C13 is missing_evidence
- Submission Readiness: required evidence claim C14 is missing_evidence
- Submission Readiness: raw gate Data Readiness Audit is blocked: 10 missing files
- Submission Readiness: raw gate Real Run Preflight is blocked: 33 blockers, 8 warnings
- Submission Readiness: raw gate SFT Data Preflight is blocked: 8 blockers, 0 warnings
- Submission Readiness: raw gate Meta-Verifier API Budget is missing: report missing
- Submission Readiness: raw gate RubricBench Model Criteria Verifier API Budget is missing: report missing
- Submission Readiness: raw gate RubricBench Model Criteria Verifier Stats is missing: report missing
- Submission Readiness: raw gate Filtered Teacher Evaluation-Criteria Validation is missing: report missing
- Submission Readiness: raw gate Training Completion Gate is blocked: 46 blockers, 0 warnings
- Submission Readiness: raw gate Matrix Trained Method Gate is blocked: 44 blockers, 0 warnings
- Submission Readiness: raw gate JudgeBench Trained Method Gate is blocked: 44 blockers, 0 warnings
- Submission Readiness: raw gate RewardBench-2 Trained Method Gate is blocked: 44 blockers, 0 warnings
- Submission Readiness: raw gate Generalization Trained Method Gate is blocked: 44 blockers, 0 warnings
- Submission Readiness: raw gate Downstream Policy RLVR Completion Gate is blocked: 19 blockers, 0 warnings
- Submission Readiness: raw gate AAAI LaTeX Compile is blocked: pdf_bytes=0, pages=0, max_pages=0, official_style_active=False, submission_mode_declared=True, bibliography_style_active=False, anonymous_author_declared=True, 3 blockers
- Submission Readiness: required paper tables not synced: tables/main_table.tex, tables/rl_stage_ablation_table.tex, tables/downstream_utility_table.tex, tables/ablation_table.tex, tables/teacher_union_ablation_table.tex, tables/verifier_filter_ablation_table.tex, tables/dimension_transition_table.tex
- Submission Readiness: required paper figures not synced: figures/semantic_space.pdf, figures/semantic_space.svg
- Submission Readiness: required paper reviewer-facing docs not synced: asset_index/semantic_space_points.csv, asset_index/semantic_space_summary.json
- Submission Readiness: paper asset index is blocked: 11 blockers, 2 warnings, 8 blocker categories, 1 warning categories
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/main_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/rl_stage_ablation_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/downstream_utility_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/ablation_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/teacher_union_ablation_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/verifier_filter_ablation_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/dimension_transition_table.tex
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/semantic_space.svg
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/semantic_space.pdf
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/semantic_space_points.csv
- Submission Readiness: artifact is missing or empty: outputs/paper_artifacts/semantic_space_summary.json
- Submission Gap Report: phase `Data Isolation And Hard-Gold Holdouts` is blocked
- Submission Gap Report: phase `SFT/GRPO Training And Serving` is blocked
- Submission Gap Report: phase `Main Hard-Gold BSC Evidence` is blocked
- Submission Gap Report: phase `Downstream Judge Utility` is blocked
- Submission Gap Report: phase `Ablations` is blocked
- Submission Gap Report: phase `Semantic-Space Visualization` is blocked
- Submission Gap Report: phase `Paper Assets And AAAI Readiness` is blocked
- Submission Gap Report: phase `Reviewer-Facing Rebuttal Readiness` is blocked
- AAAI LaTeX Compile: pdflatex binary is not available
- AAAI LaTeX Compile: missing compiled PDF: paper/main.pdf
- AAAI LaTeX Compile: missing LaTeX log: paper/main.log

## Warnings

- RubricBench Gold Validation: record 45: duplicate query appears in gold data
- RubricBench Gold Validation: record 84: duplicate query appears in gold data
- RubricBench Gold Validation: record 170: duplicate query appears in gold data
- RubricBench Gold Validation: record 293: duplicate query appears in gold data
- RubricBench Gold Validation: record 310: duplicate query appears in gold data
- RubricBench Gold Validation: record 351: duplicate query appears in gold data
- RubricBench Gold Validation: record 359: duplicate query appears in gold data
- RubricBench Gold Validation: record 388: duplicate query appears in gold data
- RubricBench Gold Validation: record 542: duplicate query appears in gold data
- RubricBench Gold Validation: record 656: duplicate query appears in gold data
- RubricBench Gold Validation: record 869: duplicate query appears in gold data
- RubricBench Gold Validation: record 1000: duplicate query appears in gold data
- RubricBench Gold Validation: record 1032: duplicate query appears in gold data
- RubricBench Gold Validation: record 1040: duplicate query appears in gold data
- Real Run Preflight: sft config does not exist yet: configs/llamafactory_sft.local.yaml
- Real Run Preflight: sft env is not set: TOKENIZERS_PARALLELISM
- Real Run Preflight: grpo env is not set: BSC_EMBEDDING_MODEL
- Real Run Preflight: grpo env is not set: BSC_COVERAGE_TAU
- Real Run Preflight: grpo env is not set: BSC_REDUNDANCY_TAU
- Real Run Preflight: grpo env is not set: BSC_W_COV
- Real Run Preflight: grpo env is not set: BSC_W_VALID
- Real Run Preflight: grpo env is not set: BSC_W_RED
- Teacher Rubrics vs RewardBench Holdout Filter: optional report is missing
- Teacher Rubrics vs JudgeBench Holdout Filter: optional report is missing
- Teacher Rubrics vs RewardBench-2 Holdout Filter: optional report is missing
- RM-Bench Model Evaluation-Criteria API Budget: optional report is missing
- Meta-Verifier API Budget: optional report is missing
- RubricBench Model Criteria Verifier API Budget: optional report is missing
- RewardBench Model Verifier API Budget: optional report is missing
- JudgeBench Model Verifier API Budget: optional report is missing
- RewardBench-2 Model Verifier API Budget: optional report is missing
- RM-Bench Model Verifier API Budget: optional report is missing
- HealthBench Model Verifier API Budget: optional report is missing
- WritingBench Model Verifier API Budget: optional report is missing
- RM-Bench Downstream Schema Contract: optional report is missing
- RubricBench Model Evaluation-Criteria Validation: optional report is missing
- RewardBench Model Evaluation-Criteria Validation: optional report is missing
- HealthBench Model Evaluation-Criteria Validation: optional report is missing
- JudgeBench Model Evaluation-Criteria Validation: optional report is missing
- RewardBench-2 Model Evaluation-Criteria Validation: optional report is missing
- RM-Bench Model Evaluation-Criteria Validation: optional report is missing
- WritingBench Model Evaluation-Criteria Validation: optional report is missing
- Teacher Evaluation-Criteria Validation: optional report is missing
- Filtered Teacher Evaluation-Criteria Validation: optional report is missing
- RubricBench Model Criteria Verifier Stats: optional report is missing
- Main Matrix Evidence: 15 claim(s) still missing evidence
- C5 Teacher-Union Ablation JSON: optional report is missing
- C5 Teacher-Union Ablation Table: optional report is missing
- C5 Teacher-Union Per-Item Audit: optional report is missing
- Base BSC Bootstrap CI: optional report is missing
- Base Downstream Bootstrap CI: optional report is missing
- GPT-4o BSC Bootstrap CI: optional report is missing
- GPT-4o Downstream Bootstrap CI: optional report is missing
- Claude BSC Bootstrap CI: optional report is missing
- Claude Downstream Bootstrap CI: optional report is missing
- SFT-only BSC Bootstrap CI: optional report is missing
- SFT-only Downstream Bootstrap CI: optional report is missing
- SFT+RL BSC Bootstrap CI: optional report is missing
- SFT+RL Downstream Bootstrap CI: optional report is missing
- JudgeBench Base Downstream Bootstrap CI: optional report is missing
- JudgeBench SFT+RL Downstream Bootstrap CI: optional report is missing
- RewardBench-2 Base Multi-Candidate Bootstrap CI: optional report is missing
- RewardBench-2 SFT+RL Multi-Candidate Bootstrap CI: optional report is missing
- Submission Readiness: no claim is currently safe_to_claim
- Submission Readiness: some claims still have missing evidence
- Rebuttal Pack Manifest: submission readiness was false when the pack was built
- Rebuttal Pack Manifest: no answer_ready reviewer concern entries

## Next Actions

- Resolve blockers in Real Run Preflight before upgrading paper claims.
- Resolve blockers in SFT Data Preflight before upgrading paper claims.
- Resolve blockers in Teacher Rubrics vs Hard-Gold Holdout Filter before upgrading paper claims.
- Review warnings in Teacher Rubrics vs RewardBench Holdout Filter and keep claims defensive.
- Review warnings in Teacher Rubrics vs JudgeBench Holdout Filter and keep claims defensive.
- Review warnings in Teacher Rubrics vs RewardBench-2 Holdout Filter and keep claims defensive.
- Review warnings in RM-Bench Model Evaluation-Criteria API Budget and keep claims defensive.
- Review warnings in Meta-Verifier API Budget and keep claims defensive.
- Review warnings in RubricBench Model Criteria Verifier API Budget and keep claims defensive.
- Review warnings in RewardBench Model Verifier API Budget and keep claims defensive.
- Review warnings in JudgeBench Model Verifier API Budget and keep claims defensive.
- Review warnings in RewardBench-2 Model Verifier API Budget and keep claims defensive.
- Review warnings in RM-Bench Model Verifier API Budget and keep claims defensive.
- Review warnings in HealthBench Model Verifier API Budget and keep claims defensive.
- Review warnings in WritingBench Model Verifier API Budget and keep claims defensive.
- Review warnings in RM-Bench Downstream Schema Contract and keep claims defensive.
- Review warnings in RubricBench Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in RewardBench Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in HealthBench Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in JudgeBench Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in RewardBench-2 Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in RM-Bench Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in WritingBench Model Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in Teacher Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in Filtered Teacher Evaluation-Criteria Validation and keep claims defensive.
- Review warnings in RubricBench Model Criteria Verifier Stats and keep claims defensive.
- Resolve blockers in Main Matrix Audit before upgrading paper claims.
- Review warnings in Main Matrix Evidence and keep claims defensive.
- Review warnings in C5 Teacher-Union Ablation JSON and keep claims defensive.
- Review warnings in C5 Teacher-Union Ablation Table and keep claims defensive.
- Review warnings in C5 Teacher-Union Per-Item Audit and keep claims defensive.
- Review warnings in Base BSC Bootstrap CI and keep claims defensive.
- Review warnings in Base Downstream Bootstrap CI and keep claims defensive.
- Review warnings in GPT-4o BSC Bootstrap CI and keep claims defensive.
- Review warnings in GPT-4o Downstream Bootstrap CI and keep claims defensive.
- Review warnings in Claude BSC Bootstrap CI and keep claims defensive.
- Review warnings in Claude Downstream Bootstrap CI and keep claims defensive.
- Review warnings in SFT-only BSC Bootstrap CI and keep claims defensive.
- Review warnings in SFT-only Downstream Bootstrap CI and keep claims defensive.
- Review warnings in SFT+RL BSC Bootstrap CI and keep claims defensive.
- Review warnings in SFT+RL Downstream Bootstrap CI and keep claims defensive.
- Review warnings in JudgeBench Base Downstream Bootstrap CI and keep claims defensive.
- Review warnings in JudgeBench SFT+RL Downstream Bootstrap CI and keep claims defensive.
- Review warnings in RewardBench-2 Base Multi-Candidate Bootstrap CI and keep claims defensive.
- Review warnings in RewardBench-2 SFT+RL Multi-Candidate Bootstrap CI and keep claims defensive.
- Resolve blockers in Matrix Trained Method Gate before upgrading paper claims.
- Resolve blockers in JudgeBench Trained Method Gate before upgrading paper claims.
- Resolve blockers in RewardBench-2 Trained Method Gate before upgrading paper claims.
- Resolve blockers in Generalization Trained Method Gate before upgrading paper claims.
- Resolve blockers in Downstream Policy RLVR Completion Gate before upgrading paper claims.
- Resolve blockers in Submission Readiness before upgrading paper claims.
- Resolve blockers in Submission Gap Report before upgrading paper claims.
- Review warnings in Rebuttal Pack Manifest and keep claims defensive.
- Resolve blockers in AAAI LaTeX Compile before upgrading paper claims.
- Run minimal-claim pipeline on RubricBench before investing in SFT/GRPO.
- Only upgrade manuscript claims whose Evidence Matrix status is safe_to_claim.
- Keep every claim with a non-safe Evidence Matrix row out of the paper claim surface.
