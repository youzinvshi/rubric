# BlindSpot-RL Submission Readiness

- Status: `blocked`
- Overall ok: `False`
- Audit ok: `False`
- Hard blockers: `43`
- Hard blocker categories: `16`
- Warnings: `2`
- Safe claims: `0` / `15`
- Missing-evidence claims: `15`
- Contradicted claims: `0`

## Hard Blocker Summary

- `asset_index:dimension_transition`: 1 (asset index: dimension-transition audit paper table)
- `asset_index:downstream_utility`: 1 (asset index: RewardBench/JudgeBench/RewardBench-2 utility table)
- `asset_index:main_bsc_table`: 1 (asset index: main hard-gold BSC paper table)
- `asset_index:reward_ablation`: 1 (asset index: reward-component ablation table)
- `asset_index:rl_stage_ablation`: 1 (asset index: SFT-only vs SFT+GRPO paper table)
- `asset_index:semantic_space`: 1 (asset index: semantic-space SVG/PDF/CSV/JSON assets)
- `asset_index:teacher_union_ablation`: 1 (asset index: teacher-union ablation table)
- `asset_index:verifier_filter_ablation`: 1 (asset index: verifier-filter ablation table)
- `audit_report`: 1 (experiment audit report is not ready); examples: audit report is not ok
- `evidence_claims`: 13 (required Evidence Matrix rows are not safe_to_claim); examples: required evidence claim C0 is missing_evidence; required evidence claim C1 is missing_evidence; required evidence claim C2 is missing_evidence
- `other`: 1 (other submission-readiness blockers); examples: required paper reviewer-facing docs not synced: asset_index/semantic_space_points.csv, asset_index/semantic_space_summary.json
- `paper_artifacts`: 11 (paper artifact files are missing or empty); examples: artifact is missing or empty: outputs/paper_artifacts/main_table.tex; artifact is missing or empty: outputs/paper_artifacts/rl_stage_ablation_table.tex; artifact is missing or empty: outputs/paper_artifacts/downstream_utility_table.tex
- `paper_asset_index`: 1 (paper asset index is blocked); examples: paper asset index is blocked: 11 blockers, 2 warnings, 8 blocker categories, 1 warning categories
- `paper_figures`: 1 (paper-facing figures are not synced); examples: required paper figures not synced: figures/semantic_space.pdf, figures/semantic_space.svg
- `paper_tables`: 1 (paper-facing tables are not synced); examples: required paper tables not synced: tables/main_table.tex, tables/rl_stage_ablation_table.tex, tables/downstream_utility_table.tex, tables/ablation_table.tex, tables/teacher_union_ablation_table.tex, tables/verifier_filter_ablation_table.tex, tables/dimension_transition_table.tex
- `raw_gates`: 14 (raw audit gates are blocked); examples: raw gate Data Readiness Audit is blocked: 10 missing files; raw gate Real Run Preflight is blocked: 33 blockers, 8 warnings; raw gate SFT Data Preflight is blocked: 8 blockers, 0 warnings

## Hard Blockers

- audit report is not ok
- required evidence claim C0 is missing_evidence
- required evidence claim C1 is missing_evidence
- required evidence claim C2 is missing_evidence
- required evidence claim C3 is missing_evidence
- required evidence claim C4 is missing_evidence
- required evidence claim C5 is missing_evidence
- required evidence claim C6 is missing_evidence
- required evidence claim C7 is missing_evidence
- required evidence claim C9 is missing_evidence
- required evidence claim C10 is missing_evidence
- required evidence claim C12 is missing_evidence
- required evidence claim C13 is missing_evidence
- required evidence claim C14 is missing_evidence
- raw gate Data Readiness Audit is blocked: 10 missing files
- raw gate Real Run Preflight is blocked: 33 blockers, 8 warnings
- raw gate SFT Data Preflight is blocked: 8 blockers, 0 warnings
- raw gate Meta-Verifier API Budget is missing: report missing
- raw gate RubricBench Model Criteria Verifier API Budget is missing: report missing
- raw gate RubricBench Model Criteria Verifier Stats is missing: report missing
- raw gate Filtered Teacher Evaluation-Criteria Validation is missing: report missing
- raw gate Training Completion Gate is blocked: 46 blockers, 0 warnings
- raw gate Matrix Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate JudgeBench Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate RewardBench-2 Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate Generalization Trained Method Gate is blocked: 44 blockers, 0 warnings
- raw gate Downstream Policy RLVR Completion Gate is blocked: 19 blockers, 0 warnings
- raw gate AAAI LaTeX Compile is blocked: pdf_bytes=0, pages=0, max_pages=0, official_style_active=False, submission_mode_declared=True, bibliography_style_active=False, anonymous_author_declared=True, 3 blockers
- required paper tables not synced: tables/main_table.tex, tables/rl_stage_ablation_table.tex, tables/downstream_utility_table.tex, tables/ablation_table.tex, tables/teacher_union_ablation_table.tex, tables/verifier_filter_ablation_table.tex, tables/dimension_transition_table.tex
- required paper figures not synced: figures/semantic_space.pdf, figures/semantic_space.svg
- required paper reviewer-facing docs not synced: asset_index/semantic_space_points.csv, asset_index/semantic_space_summary.json
- paper asset index is blocked: 11 blockers, 2 warnings, 8 blocker categories, 1 warning categories
- artifact is missing or empty: outputs/paper_artifacts/main_table.tex
- artifact is missing or empty: outputs/paper_artifacts/rl_stage_ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/downstream_utility_table.tex
- artifact is missing or empty: outputs/paper_artifacts/ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/teacher_union_ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/verifier_filter_ablation_table.tex
- artifact is missing or empty: outputs/paper_artifacts/dimension_transition_table.tex
- artifact is missing or empty: outputs/paper_artifacts/semantic_space.svg
- artifact is missing or empty: outputs/paper_artifacts/semantic_space.pdf
- artifact is missing or empty: outputs/paper_artifacts/semantic_space_points.csv
- artifact is missing or empty: outputs/paper_artifacts/semantic_space_summary.json

## Warnings

- no claim is currently safe_to_claim
- some claims still have missing evidence

## Claim Ladder Status

| Level | Status | Required Claims | Blocking Claims |
| --- | --- | --- | --- |
| motivation | `missing_evidence` | C1, C6 | C1: missing_evidence; C6: missing_evidence |
| metric-support | `missing_evidence` | C0, C2, C3 | C0: missing_evidence; C2: missing_evidence; C3: missing_evidence |
| method-support | `missing_evidence` | C5, C7, C14 | C5: missing_evidence; C7: missing_evidence; C14: missing_evidence |
| judge-utility support | `missing_evidence` | C0, C4, C9, C10, C12 | C0: missing_evidence; C4: missing_evidence; C9: missing_evidence; C10: missing_evidence; C12: missing_evidence |

## Claim Discipline

- Do not treat this package as AAAI-ready while submission readiness is false.
- SFT+GRPO coverage, dimension-level recovery, downstream utility, ablation, and semantic-space claims are permitted only after all required C0-C14 evidence rows are safe_to_claim.
- Do not claim clean hard-gold/proxy/downstream isolation until C0 passes with query-disjoint holdouts and SHA-bound provenance.
- Treat BSC changes as metric evidence only; do not describe them as dimension-level recovery until C3, C12, and C14 pass.
- Do not claim downstream judge-utility support until C4/C9/C10 pass with API/model scorer outputs and paper_claim_eligible summaries.
- Do not claim ablation support until C5/C7 pass with separately trained reward-component variants and verifier-filter evidence.
- Treat semantic-space plots as illustrative until C13 verifies SVG/PDF/CSV/JSON assets and point-level provenance.
- Do not submit the paper package until required paper-facing tables, figures, and reviewer-facing docs are synced and indexed.
- With zero safe_to_claim rows, restrict paper-facing empirical content to planned protocol and clearly marked diagnostic evidence.

## Raw Gates

| Gate | Type | Status | Summary |
| --- | --- | --- | --- |
| Data Source Local Config | `data_source_local_config` | `pass` | 0 blockers, source_status=pass |
| Data Source Report | `data_source_report` | `pass` | 10 scoped datasets, 0 blockers |
| Data Readiness Audit | `audit` | `blocked` | 10 missing files |
| Real Run Preflight | `preflight` | `blocked` | 33 blockers, 8 warnings |
| SFT Data Preflight | `preflight` | `blocked` | 8 blockers, 0 warnings |
| RubricBench Gold Validation | `gold_validation` | `pass` | 1147 records, 0 blockers |
| ResearchRubrics Gold Validation | `gold_validation` | `pass` | 101 records, 0 blockers |
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
| BlindSpot SFT vs Hard-Gold Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| BlindSpot SFT vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| BlindSpot SFT vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| BlindSpot SFT vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 |
| Proxy-Gold vs Hard-Gold Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Proxy-Gold vs RewardBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Proxy-Gold vs JudgeBench Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=0 |
| Proxy-Gold vs RewardBench-2 Holdout Filter | `contamination_audit` | `pass` | 0 blockers, removed=1 |
| RubricBench Model Evaluation-Criteria API Budget | `api_budget` | `pass` | 5735 calls, 0 blockers |
| Teacher Evaluation-Criteria API Budget | `api_budget` | `pass` | 2000 calls, 0 blockers |
| Meta-Verifier API Budget | `api_budget` | `missing` | report missing |
| RubricBench Model Criteria Verifier API Budget | `api_budget` | `missing` | report missing |
| RubricBench Model Criteria Verifier Stats | `generic` | `missing` | report missing |
| Filtered Teacher Evaluation-Criteria Validation | `validation` | `missing` | report missing |
| Training Completion Gate | `manual_gate` | `blocked` | 46 blockers, 0 warnings |
| Matrix Trained Method Gate | `manual_gate` | `blocked` | 44 blockers, 0 warnings |
| JudgeBench Trained Method Gate | `manual_gate` | `blocked` | 44 blockers, 0 warnings |
| RewardBench-2 Trained Method Gate | `manual_gate` | `blocked` | 44 blockers, 0 warnings |
| Generalization Trained Method Gate | `manual_gate` | `blocked` | 44 blockers, 0 warnings |
| Downstream Policy RLVR Completion Gate | `manual_gate` | `blocked` | 19 blockers, 0 warnings |
| AAAI LaTeX Compile | `latex_compile` | `blocked` | pdf_bytes=0, pages=0, max_pages=0, official_style_active=False, submission_mode_declared=True, bibliography_style_active=False, anonymous_author_declared=True, 3 blockers |

## Paper Files

| File | Present | Status | Bytes |
| --- | --- | --- | --- |
| `main.tex` | `True` | `pass` | 1231 |
| `sections/abstract.tex` | `True` | `pass` | 1782 |
| `sections/introduction.tex` | `True` | `pass` | 4175 |
| `sections/blindspot_phenomenon.tex` | `True` | `pass` | 5576 |
| `sections/related_work.tex` | `True` | `pass` | 4767 |
| `sections/method.tex` | `True` | `pass` | 10879 |
| `sections/experiments.tex` | `True` | `pass` | 18059 |
| `sections/limitations.tex` | `True` | `pass` | 4180 |
| `sections/conclusion.tex` | `True` | `pass` | 1785 |
| `tables/main_table.tex` | `False` | `missing` | 0 |
| `tables/rl_stage_ablation_table.tex` | `False` | `missing` | 0 |
| `tables/downstream_utility_table.tex` | `False` | `missing` | 0 |
| `tables/ablation_table.tex` | `False` | `missing` | 0 |
| `tables/teacher_union_ablation_table.tex` | `False` | `missing` | 0 |
| `tables/verifier_filter_ablation_table.tex` | `False` | `missing` | 0 |
| `tables/dimension_transition_table.tex` | `False` | `missing` | 0 |
| `figures/semantic_space.pdf` | `False` | `missing` | 0 |
| `figures/semantic_space.svg` | `False` | `missing` | 0 |
| `asset_index/real_run_dashboard.json` | `True` | `pass` | 145211 |
| `asset_index/real_run_dashboard.md` | `True` | `pass` | 67743 |
| `asset_index/evidence_matrix.json` | `True` | `pass` | 144755 |
| `asset_index/evidence_matrix.csv` | `True` | `pass` | 143069 |
| `asset_index/evidence_matrix.md` | `True` | `pass` | 138337 |
| `asset_index/semantic_space_points.csv` | `False` | `missing` | 0 |
| `asset_index/semantic_space_summary.json` | `False` | `missing` | 0 |
| `asset_index/result_card.json` | `True` | `pass` | 476077 |
| `asset_index/result_card.md` | `True` | `pass` | 51580 |
| `asset_index/submission_gap_report.json` | `True` | `pass` | 190673 |
| `asset_index/submission_gap_report.md` | `True` | `pass` | 42555 |
| `asset_index/readiness_report.json` | `True` | `pass` | 63047 |
| `asset_index/readiness_report.md` | `True` | `pass` | 15824 |
| `asset_index/rebuttal_pack.json` | `True` | `pass` | 177581 |
| `asset_index/rebuttal_pack.md` | `True` | `pass` | 175572 |
| `asset_index/rebuttal_pack_manifest.json` | `True` | `pass` | 4296 |
| `asset_index.md` | `True` | `blocked` | 5095 |

## Paper Asset Blocker Summary

- `main_bsc_table`: 1 (main hard-gold BSC paper table)
- `rl_stage_ablation`: 1 (SFT-only vs SFT+GRPO paper table)
- `downstream_utility`: 1 (RewardBench/JudgeBench/RewardBench-2 utility table)
- `reward_ablation`: 1 (reward-component ablation table)
- `teacher_union_ablation`: 1 (teacher-union ablation table)
- `verifier_filter_ablation`: 1 (verifier-filter ablation table)
- `dimension_transition`: 1 (dimension-transition audit paper table)
- `semantic_space`: 4 (semantic-space SVG/PDF/CSV/JSON assets)
