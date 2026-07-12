#!/usr/bin/env python3
"""Build a phase-grouped submission gap report from readiness and evidence outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.build_result_card import build_claim_ladder_status
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from build_result_card import build_claim_ladder_status


PHASES = [
    {
        "id": "data_isolation_hard_gold",
        "name": "Data Isolation And Hard-Gold Holdouts",
        "depends_on": set(),
        "claim_ids": {"C0"},
        "keywords": ["contamination", "holdout", "data source", "gold validation", "test_main", "overlap"],
        "next_actions": [
            "Run real data normalization and hard-gold/downstream split stages.",
            "Refresh proxy-train filters and zero-overlap contamination audits before training.",
        ],
        "claim_discipline": (
            "Do not report hard-gold, trained-method, ablation, downstream, or visualization results until "
            "C0 is safe_to_claim with query-disjoint holdouts and SHA-bound training/evaluation provenance."
        ),
    },
    {
        "id": "training_and_serving",
        "name": "SFT/GRPO Training And Serving",
        "depends_on": {"data_isolation_hard_gold"},
        "claim_ids": {"C2", "C3", "C14"},
        "keywords": ["training completion", "trained method", "training_done", "checkpoint", "serving", "sft", "grpo"],
        "next_actions": [
            "Run SFT-only and SFT+GRPO on proxy-gold data only.",
            "Write training_done.json with checkpoint paths, serving endpoints, rl_data_report, and reward_function.",
        ],
        "claim_discipline": (
            "Treat SFT/GRPO as unavailable for paper claims until training_done.json binds SFT-only and SFT+GRPO "
            "checkpoints, serving metadata, proxy-gold RL data, and reward configuration."
        ),
    },
    {
        "id": "main_hard_gold_bsc",
        "name": "Main Hard-Gold BSC Evidence",
        "depends_on": {"data_isolation_hard_gold", "training_and_serving"},
        "claim_ids": {"C1", "C2", "C3", "C6", "C12", "C14"},
        "keywords": ["matrix_real", "bsc", "bootstrap", "repair", "threshold", "main table", "audit report"],
        "next_actions": [
            "Run the real RubricBench test_main matrix for base, API teachers, SFT-only, and SFT+GRPO.",
            "Generate BSC summaries, paired bootstrap CIs, threshold sweeps, dimension-transition summaries, and matrix audit report.",
        ],
        "claim_discipline": (
            "A BSC coverage change is only metric evidence until C3 controls redundancy/hallucination, C6 verifies "
            "threshold robustness and human audit, and C12/C14 audit dimension-level recovery over SFT-only; "
            "dimension-level recovery remains a permitted conclusion only after those gates pass."
        ),
    },
    {
        "id": "downstream_utility",
        "name": "Downstream Judge Utility",
        "depends_on": {"data_isolation_hard_gold", "training_and_serving", "main_hard_gold_bsc"},
        "claim_ids": {"C4", "C9", "C10", "C11"},
        "keywords": ["downstream", "rewardbench", "judgebench", "rewardbench-2", "rewardbench2", "api budget"],
        "next_actions": [
            "Run paper-facing API/model scorer evaluations on RewardBench, JudgeBench, and RewardBench-2 holdouts.",
            "Ensure every downstream summary is paper_claim_eligible and bound to its input/provider/budget SHA256 contract.",
        ],
        "claim_discipline": (
            "Do not describe BSC coverage changes as supporting judge utility until C4/C9/C10 pass with API/model scorer outputs, "
            "paper_claim_eligible summaries, and input/provider/budget SHA256 contracts."
        ),
    },
    {
        "id": "ablations",
        "name": "Ablations",
        "depends_on": {"data_isolation_hard_gold", "training_and_serving", "main_hard_gold_bsc"},
        "claim_ids": {"C5", "C7", "C14"},
        "keywords": ["ablation", "teacher-union", "teacher union", "no_red", "no_valid", "verifier", "rl_stage"],
        "next_actions": [
            "Run separately trained reward-component ablations: no redundancy, no validity, and coverage-only; keep offline reward re-scoring as diagnostic only.",
            "Run verifier-filter ablation for proxy-gold quality control.",
            "Run single-teacher versus multi-teacher union and SFT-only versus SFT+GRPO comparisons under the same BGE protocol.",
        ],
        "claim_discipline": (
            "Do not claim the RL stage is necessary or anti-hacking works from offline re-scoring; paper-facing C7 "
            "requires separately trained variants and verifier-filter evidence under the same BGE protocol."
        ),
    },
    {
        "id": "semantic_space",
        "name": "Semantic-Space Visualization",
        "depends_on": {"data_isolation_hard_gold", "training_and_serving", "main_hard_gold_bsc"},
        "claim_ids": {"C13"},
        "keywords": ["semantic_space", "semantic-space", "umap", "figure", "nearest-gold"],
        "next_actions": [
            "Build UMAP/PCA semantic-space assets from base, SFT-only, and SFT+GRPO hard-gold BSC eval rows.",
            "Verify semantic_space_points.csv and semantic_space_summary.json carry nearest-gold audit fields.",
        ],
        "claim_discipline": (
            "Treat the plot as illustrative until C13 verifies point-level provenance, nearest-gold audit fields, "
            "category/cluster coverage, dispersion, and non-empty SVG/PDF/CSV/JSON assets."
        ),
    },
    {
        "id": "paper_readiness",
        "name": "Paper Assets And AAAI Readiness",
        "depends_on": {
            "data_isolation_hard_gold",
            "training_and_serving",
            "main_hard_gold_bsc",
            "downstream_utility",
            "ablations",
            "semantic_space",
        },
        "claim_ids": set(),
        "keywords": ["paper", "latex", "table", "figure", "asset index", "aaai", "readiness"],
        "next_actions": [
            "Export paper artifacts after real matrices and ablations are complete.",
            "Sync tables/figures into paper/ and rerun LaTeX plus submission readiness.",
        ],
        "claim_discipline": (
            "Keep the paper in problem--metric--method framing and write empirical claims only when the relevant "
            "evidence rows are safe_to_claim and synced paper assets pass readiness."
        ),
    },
    {
        "id": "reviewer_response_readiness",
        "name": "Reviewer-Facing Rebuttal Readiness",
        "depends_on": {"paper_readiness"},
        "claim_ids": set(),
        "keywords": ["rebuttal", "reviewer-facing", "reviewer concern", "answer_ready"],
        "next_actions": [
            "Regenerate the rebuttal pack after Evidence Matrix rows and submission readiness are refreshed.",
            "Use only answer_ready rebuttal entries whose matched Evidence Matrix rows are safe_to_claim and whose readiness_ok flag is true.",
        ],
        "claim_discipline": (
            "Treat rebuttal entries as readiness notes, not new claims; do not use reviewer-facing answers until "
            "the pack is generated against the current Evidence Matrix/readiness report, readiness_ok is true, and relevant entries are answer_ready."
        ),
    },
]

ASSET_BLOCKER_PHASES = {
    "main_bsc_table": {
        "phase_id": "main_hard_gold_bsc",
        "claim_ids": ["C1", "C2", "C3", "C14"],
        "evidence_gate": "C1/C2/C3/C14",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/matrix_real/main_table.csv"],
        "paper_artifacts": ["main_table.tex"],
        "action": "Run the real hard-gold BSC matrix and export the main paper table only after paper_claim_eligible summaries pass.",
    },
    "rl_stage_ablation": {
        "phase_id": "ablations",
        "claim_ids": ["C14"],
        "evidence_gate": "C14",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/matrix_real/main_table.csv"],
        "paper_artifacts": ["rl_stage_ablation_table.tex"],
        "action": "Run matrix_real through base, sft_only, and sft_rl BSC/downstream summaries, then export paper artifacts.",
    },
    "downstream_utility": {
        "phase_id": "downstream_utility",
        "claim_ids": ["C4", "C9", "C10"],
        "evidence_gate": "C4/C9/C10",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": [
            "outputs/matrix_real/main_table.csv",
            "outputs/matrix_judgebench/main_table.csv",
            "outputs/matrix_rewardbench2/main_table.csv",
        ],
        "paper_artifacts": ["downstream_utility_table.tex"],
        "action": "Run RewardBench, JudgeBench, and RewardBench-2 downstream API/model scorer matrices, then export the combined utility table.",
    },
    "reward_ablation": {
        "phase_id": "ablations",
        "claim_ids": ["C7"],
        "evidence_gate": "C7",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/bsc_ablation/ablation_summary.csv"],
        "paper_artifacts": ["ablation_table.tex"],
        "action": "Run separately trained reward-component ablations and export the reward ablation table.",
    },
    "teacher_union_ablation": {
        "phase_id": "ablations",
        "claim_ids": ["C5"],
        "evidence_gate": "C5",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/teacher_union_ablation/teacher_union_ablation.csv"],
        "paper_artifacts": ["teacher_union_ablation_table.tex"],
        "action": "Run single-teacher versus multi-teacher union ablation, then export the teacher-union table.",
    },
    "verifier_filter_ablation": {
        "phase_id": "ablations",
        "claim_ids": ["C7"],
        "evidence_gate": "C7",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/verifier_filter_ablation/verifier_filter_ablation.csv"],
        "paper_artifacts": ["verifier_filter_ablation_table.tex"],
        "action": "Run verifier-filter ablation over raw and filtered teacher criteria outputs, then export the verifier-filter table.",
    },
    "dimension_transition": {
        "phase_id": "main_hard_gold_bsc",
        "claim_ids": ["C12"],
        "evidence_gate": "C12",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/matrix_real/dimension_transition_summary.csv"],
        "paper_artifacts": ["dimension_transition_table.tex"],
        "action": "Run the dimension-transition audit and export the C12 table before writing dimension-level recovery language.",
    },
    "semantic_space": {
        "phase_id": "semantic_space",
        "claim_ids": ["C13"],
        "evidence_gate": "C13",
        "producer": "scripts/build_semantic_space_visualization.py -> scripts/export_paper_artifacts.py",
        "source_artifacts": [
            "outputs/matrix_real/semantic_space/semantic_space.svg",
            "outputs/matrix_real/semantic_space/semantic_space.pdf",
            "outputs/matrix_real/semantic_space/semantic_space_points.csv",
            "outputs/matrix_real/semantic_space/semantic_space_summary.json",
        ],
        "paper_artifacts": [
            "semantic_space.svg",
            "semantic_space.pdf",
            "semantic_space_points.csv",
            "semantic_space_summary.json",
        ],
        "action": "Run semantic-space visualization after base, sft_only, and sft_rl hard-gold BSC eval rows exist, then export/copy all four assets.",
    },
    "experiment_summary": {
        "phase_id": "paper_readiness",
        "claim_ids": [],
        "evidence_gate": "C0-C14",
        "producer": "scripts/export_paper_artifacts.py",
        "source_artifacts": ["outputs/matrix_real/audit_report.json", "outputs/evidence_real/evidence_matrix.md"],
        "paper_artifacts": ["experiment_summary.md"],
        "action": "Export the paper-facing experiment summary after real evidence gates are refreshed and claim discipline is safe.",
    },
}

EXECUTION_SEQUENCE = [
    {
        "id": "data_isolation",
        "phase_id": "data_isolation_hard_gold",
        "order": 1,
        "commands": [
            (
                "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
                "--from-stage audit_data_readiness --to-stage audit_rewardbench2_downstream_holdout_contamination"
            ),
        ],
        "unlocks": [
            "outputs/contamination_audit/*",
            "query-disjoint hard-gold/proxy/downstream evidence",
        ],
        "evidence_gates": ["C0"],
        "notes": "Run before training or downstream claims; hard-gold/proxy/downstream isolation is the contamination gate.",
    },
    {
        "id": "training_and_serving",
        "phase_id": "training_and_serving",
        "order": 2,
        "commands": [
            (
                "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
                "--from-stage training_commands --to-stage training_completion_gate"
            ),
        ],
        "unlocks": [
            "outputs/training_commands/training_done.json",
            "SFT-only and SFT+GRPO serving metadata",
        ],
        "evidence_gates": ["C2", "C3", "C14"],
        "notes": "The manual completion gate must bind checkpoints, serving endpoints, proxy-gold data, and reward configuration.",
    },
    {
        "id": "model_evaluation_criteria_generation",
        "phase_id": "training_and_serving",
        "order": 3,
        "commands": [
            (
                "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
                "--from-stage generate_model_rubrics_rubricbench --to-stage validate_writingbench_model_rubrics"
            ),
        ],
        "unlocks": [
            "data/processed/*_model_rubrics.jsonl",
            "verifier-filtered model evaluation-criteria reports",
        ],
        "evidence_gates": ["C2", "C3", "C4", "C9", "C10"],
        "notes": "Requires provider configs and budgets; verified evaluation-criteria files feed hard-gold and downstream matrices.",
    },
    {
        "id": "main_matrix_and_ablations",
        "phase_id": "main_hard_gold_bsc",
        "order": 4,
        "commands": [
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_real.generated.json",
        ],
        "unlocks": [
            "outputs/matrix_real/main_table.csv",
            "outputs/bsc_ablation/ablation_summary.csv",
            "outputs/teacher_union_ablation/teacher_union_ablation.csv",
            "outputs/verifier_filter_ablation/verifier_filter_ablation.csv",
            "outputs/matrix_real/semantic_space/*",
        ],
        "evidence_gates": ["C1", "C2", "C3", "C5", "C6", "C7", "C12", "C13", "C14"],
        "notes": "This produces the hard-gold BSC matrix, RL-stage comparison inputs, ablations, dimension-transition audit, and semantic-space assets.",
    },
    {
        "id": "downstream_matrices",
        "phase_id": "downstream_utility",
        "order": 5,
        "commands": [
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_judgebench.generated.json",
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_rewardbench2.generated.json",
        ],
        "unlocks": [
            "outputs/matrix_judgebench/main_table.csv",
            "outputs/matrix_rewardbench2/main_table.csv",
            "outputs/generalization_matrix/main_table.csv",
        ],
        "evidence_gates": ["C4", "C9", "C10"],
        "notes": "Run held-out downstream utility matrices with API/model scorer summaries and paper_claim_eligible provenance.",
    },
    {
        "id": "paper_export_and_readiness",
        "phase_id": "paper_readiness",
        "order": 6,
        "commands": [
            (
                "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
                "--from-stage evidence_real --to-stage sync_result_card_real"
            ),
        ],
        "unlocks": [
            "outputs/paper_artifacts/*",
            "paper/asset_index/*",
            "outputs/submission_readiness/*",
        ],
        "evidence_gates": ["C0-C14"],
        "notes": "Rebuild evidence, paper tables/figures, readiness, rebuttal pack, dashboard, Result Card, and synced asset index.",
    },
]

TRAINING_ARTIFACT_PRODUCERS = {
    "data/processed/blindspot_sft.unfiltered.jsonl": {
        "stage": "build_sft_all_domains",
        "stage_type": "build_sft",
        "producer_script": "scripts/build_sft_data.py",
        "output_report": "outputs/sft_data/proxy_gold_build_report.json",
        "upstream_inputs": [
            "data/processed/teacher_rubrics_training_clean.jsonl",
            "outputs/verifier/teacher_rubrics_filtered_report.json",
            "outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json",
            "outputs/preflight/sft_data_preflight.json",
        ],
        "sha_contract": "outputs/sft_data/proxy_gold_build_report.json:sft_output_sha256=data/processed/blindspot_sft.unfiltered.jsonl",
        "verify_after": "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage build_sft_all_domains --to-stage filter_proxy_gold_rewardbench2_downstream_overlap",
    },
    "data/processed/proxy_gold.unfiltered.jsonl": {
        "stage": "build_sft_all_domains",
        "stage_type": "build_sft",
        "producer_script": "scripts/build_sft_data.py",
        "output_report": "outputs/sft_data/proxy_gold_build_report.json",
        "upstream_inputs": [
            "data/processed/teacher_rubrics_training_clean.jsonl",
            "outputs/verifier/teacher_rubrics_filtered_report.json",
            "outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json",
            "outputs/preflight/sft_data_preflight.json",
        ],
        "sha_contract": "outputs/sft_data/proxy_gold_build_report.json:proxy_gold_output_sha256=data/processed/proxy_gold.unfiltered.jsonl",
        "verify_after": "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage build_sft_all_domains --to-stage filter_proxy_gold_rewardbench2_downstream_overlap",
    },
    "data/processed/blindspot_sft.jsonl": {
        "stage": "filter_blindspot_sft_rewardbench2_downstream_overlap",
        "stage_type": "filter_holdout_contamination",
        "producer_script": "scripts/filter_holdout_contamination.py",
        "output_report": "outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json",
        "upstream_inputs": [
            "data/processed/blindspot_sft.unfiltered.jsonl",
            "outputs/sft_data/proxy_gold_build_report.json",
            "data/processed/rewardbench2_multicandidate.clean.jsonl",
        ],
        "sha_contract": "outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json:output_sha256=data/processed/blindspot_sft.jsonl",
        "verify_after": "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage filter_blindspot_sft_hard_gold_holdout_overlap --to-stage audit_rewardbench2_downstream_holdout_contamination",
    },
    "data/processed/proxy_gold.jsonl": {
        "stage": "filter_proxy_gold_rewardbench2_downstream_overlap",
        "stage_type": "filter_holdout_contamination",
        "producer_script": "scripts/filter_holdout_contamination.py",
        "output_report": "outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json",
        "upstream_inputs": [
            "data/processed/proxy_gold.unfiltered.jsonl",
            "outputs/sft_data/proxy_gold_build_report.json",
            "data/processed/rewardbench2_multicandidate.clean.jsonl",
        ],
        "sha_contract": "outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json:output_sha256=data/processed/proxy_gold.jsonl",
        "verify_after": "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage filter_proxy_gold_hard_gold_holdout_overlap --to-stage audit_rewardbench2_downstream_holdout_contamination",
    },
    "data/processed/proxy_gold_verl.parquet": {
        "stage": "convert_proxy_gold_to_verl",
        "stage_type": "convert_verl",
        "producer_script": "scripts/convert_to_verl_parquet.py",
        "output_report": "outputs/sft_data/proxy_gold_verl_report.json",
        "upstream_inputs": [
            "data/processed/proxy_gold.jsonl",
            "outputs/sft_data/proxy_gold_build_report.json",
        ],
        "sha_contract": "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
        "verify_after": "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --from-stage convert_proxy_gold_to_verl --to-stage audit_rewardbench2_downstream_holdout_contamination",
    },
}

TRAINING_DATA_CHAIN = [
    {
        "order": 1,
        "stage": "sft_data_preflight",
        "stage_type": "preflight",
        "outputs": [
            "outputs/preflight/sft_data_preflight.json",
            "outputs/preflight/sft_data_preflight.md",
        ],
        "checks": [
            "provider configs",
            "teacher-generation API env",
            "meta-verifier API env",
        ],
    },
    {
        "order": 2,
        "stage": "generate_teacher_rubrics_*",
        "stage_type": "generate_teachers",
        "outputs": ["data/processed/teacher_rubrics_raw.jsonl"],
        "checks": [
            "teacher budget reports",
            "non-test-main source pools",
        ],
    },
    {
        "order": 3,
        "stage": "filter_teacher_rubrics",
        "stage_type": "filter_verifier",
        "outputs": [
            "data/processed/teacher_rubrics_filtered.jsonl",
            "outputs/verifier/teacher_rubrics_filtered_report.json",
        ],
        "checks": [
            "API verifier mode",
            "verifier provider SHA",
            "meta-verifier budget SHA",
        ],
    },
    {
        "order": 4,
        "stage": "validate_filtered_teacher_rubrics",
        "stage_type": "validate_rubrics",
        "outputs": [
            "outputs/validation/teacher_rubrics_filtered/validation_report.json",
            "outputs/validation/teacher_rubrics_filtered/per_record.jsonl",
        ],
        "checks": [
            "non-empty criteria lists",
            "redundancy validation",
        ],
    },
    {
        "order": 5,
        "stage": "build_sft_all_domains",
        "stage_type": "build_sft",
        "outputs": [
            "data/processed/blindspot_sft.unfiltered.jsonl",
            "data/processed/proxy_gold.unfiltered.jsonl",
            "outputs/sft_data/proxy_gold_build_report.json",
        ],
        "checks": [
            "multi-teacher proxy source",
            "test_main forbidden source markers",
            "unfiltered SFT/proxy-gold SHA contracts",
        ],
    },
    {
        "order": 6,
        "stage": "filter_*_holdout_overlap",
        "stage_type": "filter_holdout_contamination",
        "outputs": [
            "data/processed/blindspot_sft.jsonl",
            "data/processed/proxy_gold.jsonl",
            "outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json",
            "outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json",
        ],
        "checks": [
            "hard-gold/downstream holdout filters",
            "final clean SFT/proxy-gold SHA contracts",
            "removed query audit trail",
        ],
    },
    {
        "order": 7,
        "stage": "convert_proxy_gold_to_verl",
        "stage_type": "convert_verl",
        "outputs": [
            "data/processed/proxy_gold_verl.parquet",
            "outputs/sft_data/proxy_gold_verl_report.json",
        ],
        "checks": [
            "proxy-gold input SHA",
            "GRPO parquet output SHA",
            "test_main forbidden source markers",
        ],
    },
    {
        "order": 8,
        "stage": "audit_*_holdout_contamination",
        "stage_type": "holdout_contamination_audit",
        "outputs": [
            "outputs/contamination_audit/hard_gold_holdout_contamination.json",
            "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
            "outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
            "outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
        ],
        "checks": [
            "overlap_query_count == 0",
            "training artifact SHA binding",
            "hard-gold/downstream non-overlap evidence",
        ],
    },
]

C0_PROVENANCE_ACTION_CHAIN = [
    {
        "order": 1,
        "stage": "sft_data_preflight_and_teacher_budgets",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage api_budget_teacher_rubrics --to-stage sft_data_preflight"
        ),
        "outputs": [
            "outputs/api_budget/teacher_rubrics_budget.json",
            "outputs/api_budget/healthbench_teacher_rubrics_budget.json",
            "outputs/api_budget/writingbench_teacher_rubrics_budget.json",
            "outputs/preflight/sft_data_preflight.json",
            "outputs/preflight/sft_data_preflight.md",
        ],
        "checks": [
            "teacher-generation budgets are within run limits",
            "provider configs expose all teacher and verifier entries",
            "GPT_AK_1/GPT_AK_2/GPT_AK_3 are present before paid API calls",
        ],
    },
    {
        "order": 2,
        "stage": "generate_teacher_rubrics",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage generate_teacher_rubrics_rubricbench --to-stage generate_teacher_rubrics_writingbench"
        ),
        "outputs": [
            "data/processed/teacher_rubrics_raw.jsonl",
        ],
        "checks": [
            "teacher raw criteria are generated only from train/proxy source pools",
            "RubricBench test_main remains excluded from teacher generation",
            "resume mode preserves completed provider-query pairs",
        ],
    },
    {
        "order": 3,
        "stage": "api_budget_meta_verifier",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage api_budget_meta_verifier --to-stage api_budget_meta_verifier"
        ),
        "outputs": [
            "outputs/api_budget/meta_verifier_budget.json",
            "outputs/api_budget/meta_verifier_budget.md",
        ],
        "checks": [
            "meta-verifier budget binds data/processed/teacher_rubrics_raw.jsonl",
            "rubric-level verifier units are counted before filtering",
            "budget report is present before verifier API mode runs",
        ],
    },
    {
        "order": 4,
        "stage": "filter_teacher_rubrics",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage filter_teacher_rubrics --to-stage filter_teacher_rubrics_rewardbench2_downstream_overlap"
        ),
        "outputs": [
            "outputs/verifier/teacher_rubrics_filtered_report.json",
            "outputs/contamination_audit/teacher_rubrics_hard_gold_holdout_filter.json",
            "outputs/contamination_audit/teacher_rubrics_rewardbench_holdout_filter.json",
            "outputs/contamination_audit/teacher_rubrics_judgebench_holdout_filter.json",
            "outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json",
            "data/processed/teacher_rubrics_training_clean.jsonl",
        ],
        "checks": [
            "API verifier provenance is present",
            "teacher criteria are holdout-clean before proxy-gold build",
            "teacher clean output SHA feeds proxy-gold build input",
        ],
    },
    {
        "order": 5,
        "stage": "build_sft_all_domains",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage build_sft_all_domains --to-stage filter_proxy_gold_rewardbench2_downstream_overlap"
        ),
        "outputs": [
            "outputs/sft_data/proxy_gold_build_report.json",
            "data/processed/blindspot_sft.unfiltered.jsonl",
            "data/processed/proxy_gold.unfiltered.jsonl",
            "outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json",
            "outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json",
            "data/processed/blindspot_sft.jsonl",
            "data/processed/proxy_gold.jsonl",
        ],
        "checks": [
            "build report binds data/processed/teacher_rubrics_training_clean.jsonl",
            "build report writes .unfiltered SFT/proxy-gold outputs",
            "post-SFT filters write final clean SFT/proxy-gold outputs",
        ],
    },
    {
        "order": 6,
        "stage": "convert_proxy_gold_to_verl_and_audit_holdouts",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage convert_proxy_gold_to_verl --to-stage audit_rewardbench2_downstream_holdout_contamination"
        ),
        "outputs": [
            "outputs/sft_data/proxy_gold_verl_report.json",
            "data/processed/proxy_gold_verl.parquet",
            "outputs/contamination_audit/hard_gold_holdout_contamination.json",
            "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
            "outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
            "outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
        ],
        "checks": [
            "VERL report input SHA matches final clean proxy-gold",
            "hard-gold and downstream audits are complete",
            "overlap_query_count is zero for all final holdouts",
        ],
    },
    {
        "order": 7,
        "stage": "refresh_evidence_and_reports",
        "command": (
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json "
            "--from-stage evidence_real --to-stage sync_result_card_real"
        ),
        "outputs": [
            "outputs/evidence_real/evidence_matrix.json",
            "outputs/submission_readiness/readiness_report.json",
            "outputs/dashboard/real_run_dashboard.json",
            "outputs/result_card/result_card.json",
            "paper/asset_index/evidence_matrix.json",
        ],
        "checks": [
            "C0 status is recomputed from refreshed provenance",
            "readiness remains fail-closed for unrelated missing evidence",
            "paper-facing asset index uses refreshed reports",
        ],
    },
]


def main() -> None:
    args = parse_args()
    readiness = load_json(args.readiness_report)
    evidence_rows = load_json(args.evidence_matrix)
    if not isinstance(evidence_rows, list):
        raise SystemExit(f"Evidence matrix must be a JSON list: {args.evidence_matrix}")
    rebuttal_manifest = load_json(args.rebuttal_manifest) if args.rebuttal_manifest else None
    preflight_reports = [load_json(path) for path in args.preflight_report]
    gate_reports = [load_json(path) for path in args.gate_report]
    report = build_gap_report(
        readiness,
        evidence_rows,
        rebuttal_manifest=rebuttal_manifest,
        preflight_reports=preflight_reports,
        gate_reports=gate_reports,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "submission_gap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "submission_gap_report.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"Submission gap report phases={len(report['phases'])} output={args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a phase-grouped BlindSpot-RL submission gap report.")
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--evidence-matrix", required=True, type=Path)
    parser.add_argument("--rebuttal-manifest", type=Path)
    parser.add_argument("--preflight-report", action="append", default=[], type=Path)
    parser.add_argument("--gate-report", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Required JSON file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON file is not valid: {path}: line {exc.lineno} column {exc.colno}") from exc


def build_gap_report(
    readiness: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    rebuttal_manifest: dict[str, Any] | None = None,
    preflight_reports: list[dict[str, Any]] | None = None,
    gate_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hard_blockers = [str(item) for item in readiness.get("hard_blockers", [])]
    warnings = [str(item) for item in readiness.get("warnings", [])]
    rows_by_claim = {str(row.get("claim_id", "")): row for row in evidence_rows if row.get("claim_id")}
    asset_blockers_by_phase = paper_asset_blockers_by_phase(readiness)
    asset_warnings_by_phase = paper_asset_warnings_by_phase(readiness)
    missing_prerequisites = collect_missing_prerequisites(readiness, preflight_reports or [], gate_reports or [])
    phases = []
    assigned_blockers: set[int] = set()

    for phase in PHASES:
        claim_rows = [
            rows_by_claim[claim_id]
            for claim_id in sorted(phase["claim_ids"])
            if claim_id in rows_by_claim and rows_by_claim[claim_id].get("status") != "safe_to_claim"
        ]
        phase_blockers = []
        for idx, blocker in enumerate(hard_blockers):
            if idx in assigned_blockers:
                continue
            if blocker_matches_phase(blocker, phase, claim_rows):
                phase_blockers.append(blocker)
                assigned_blockers.add(idx)
        manifest_gaps = []
        if phase["id"] == "reviewer_response_readiness":
            manifest_gaps = summarize_rebuttal_manifest_gaps(rebuttal_manifest)
        paper_asset_blockers = asset_blockers_by_phase.get(phase["id"], [])
        paper_asset_warnings = asset_warnings_by_phase.get(phase["id"], [])
        phases.append(
            {
                "id": phase["id"],
                "name": phase["name"],
                "status": "blocked" if claim_rows or phase_blockers or manifest_gaps or paper_asset_blockers else "pass",
                "depends_on": sorted(phase.get("depends_on", set())),
                "blocked_by_prior_phases": [],
                "claim_ids": [row.get("claim_id") for row in claim_rows],
                "claim_gaps": [summarize_claim_gap(row) for row in claim_rows],
                "manifest_gaps": manifest_gaps,
                "paper_asset_blockers": paper_asset_blockers,
                "paper_asset_warnings": paper_asset_warnings,
                "readiness_blockers": phase_blockers,
                "next_actions": phase["next_actions"],
                "claim_discipline": phase["claim_discipline"],
            }
        )

    unassigned = [blocker for idx, blocker in enumerate(hard_blockers) if idx not in assigned_blockers]
    paper_phase = next(item for item in phases if item["id"] == "paper_readiness")
    paper_phase["readiness_blockers"].extend(unassigned)
    if paper_phase["readiness_blockers"]:
        paper_phase["status"] = "blocked"

    blocked_phase_ids = {phase["id"] for phase in phases if phase["status"] == "blocked"}
    for phase in phases:
        phase["blocked_by_prior_phases"] = [
            phase_id for phase_id in phase["depends_on"] if phase_id in blocked_phase_ids
        ]
        phase["summary"] = summarize_phase(phase)
    execution_sequence = build_execution_sequence(phases, missing_prerequisites)
    operator_handoff = build_operator_handoff(missing_prerequisites, rows_by_claim)
    claim_ladder = build_claim_ladder_status({"claims": evidence_rows})

    return {
        "ok": bool(readiness.get("ok")) and not any(phase["status"] == "blocked" for phase in phases),
        "readiness_ok": bool(readiness.get("ok")),
        "evidence": readiness.get("evidence", {}),
        "claim_ladder": claim_ladder,
        "hard_blocker_count": len(hard_blockers),
        "warning_count": len(warnings),
        "missing_prerequisites": missing_prerequisites,
        "operator_handoff": operator_handoff,
        "execution_sequence": execution_sequence,
        "phases": phases,
    }


def build_execution_sequence(
    phases: list[dict[str, Any]],
    missing_prerequisites: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    phases_by_id = {phase["id"]: phase for phase in phases}
    missing_prerequisites = missing_prerequisites or {}
    step_prerequisite_categories = {
        "data_isolation": ["input_files", "training_artifacts"],
        "training_and_serving": ["required_env", "training_artifacts"],
        "model_evaluation_criteria_generation": [
            "api_env",
            "required_env",
            "provider_configs",
            "required_providers",
            "provider_file_entries",
        ],
        "main_matrix_and_ablations": ["training_artifacts"],
        "downstream_matrices": ["provider_configs", "required_providers", "provider_file_entries"],
        "paper_export_and_readiness": ["paper_artifacts"],
    }
    out = []
    for item in EXECUTION_SEQUENCE:
        phase = phases_by_id.get(item["phase_id"], {})
        categories = step_prerequisite_categories.get(item["id"], [])
        step_missing = {
            category: missing_prerequisites.get(category, [])
            for category in categories
            if missing_prerequisites.get(category)
        }
        out.append(
            {
                **item,
                "phase_status": phase.get("status", "unknown"),
                "blocked_by_prior_phases": phase.get("blocked_by_prior_phases", []),
                "missing_prerequisites": step_missing,
                "summary": summarize_execution_step(
                    phase_status=phase.get("status", "unknown"),
                    blocked_by_prior_phases=phase.get("blocked_by_prior_phases", []),
                    missing_prerequisites=step_missing,
                    evidence_gates=item.get("evidence_gates", []),
                ),
            }
        )
    return out


def summarize_execution_step(
    phase_status: str,
    blocked_by_prior_phases: list[str],
    missing_prerequisites: dict[str, list[str]],
    evidence_gates: list[str],
) -> str:
    missing_count = sum(len(values) for values in missing_prerequisites.values())
    parts = [
        f"phase_status={phase_status}",
        f"evidence_gates={','.join(evidence_gates) if evidence_gates else 'none'}",
    ]
    if blocked_by_prior_phases:
        parts.append(f"blocked_by_prior_phases={len(blocked_by_prior_phases)}")
    if missing_count:
        parts.append(f"missing_prerequisites={missing_count}")
    if phase_status == "pass" and not blocked_by_prior_phases and not missing_count:
        parts.append("ready_to_run=true")
    else:
        parts.append("ready_to_run=false")
    return "; ".join(parts)


def summarize_phase(phase: dict[str, Any]) -> str:
    counts = {
        "claim_gaps": len(phase.get("claim_gaps", [])),
        "readiness_blockers": len(phase.get("readiness_blockers", [])),
        "paper_asset_blockers": len(phase.get("paper_asset_blockers", [])),
        "paper_asset_warnings": len(phase.get("paper_asset_warnings", [])),
        "manifest_gaps": len(phase.get("manifest_gaps", [])),
        "blocked_by_prior_phases": len(phase.get("blocked_by_prior_phases", [])),
    }
    active = [f"{name}={count}" for name, count in counts.items() if count]
    if not active:
        return "pass: no open claim, readiness, asset, manifest, or dependency gaps"
    return f"{phase.get('status', 'unknown')}: " + ", ".join(active)


def collect_missing_prerequisites(
    readiness: dict[str, Any],
    preflight_reports: list[dict[str, Any]],
    gate_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    categories = {
        "input_files": [],
        "api_env": [],
        "api_env_by_file": {},
        "api_env_by_provider": {},
        "required_env": [],
        "provider_configs": [],
        "required_providers": [],
        "provider_file_entries": [],
        "training_artifacts": [],
        "paper_artifacts": [],
    }
    blockers = [str(item) for item in readiness.get("hard_blockers", [])]
    for report in preflight_reports:
        blockers.extend(str(item) for item in report.get("hard_blockers", []))
        blockers.extend(str(item) for item in report.get("blockers", []))
        collect_missing_provider_env(categories, report)
    for report in gate_reports or []:
        blockers.extend(str(item) for item in report.get("hard_blockers", []))
        blockers.extend(str(item) for item in report.get("blockers", []))

    for blocker in blockers:
        if blocker.startswith("missing input file: "):
            categories["input_files"].append(blocker.removeprefix("missing input file: ").strip())
        elif blocker.startswith("missing API env "):
            categories["api_env"].append(blocker.removeprefix("missing API env ").strip())
        elif blocker.startswith("missing required env var: "):
            categories["required_env"].append(blocker.removeprefix("missing required env var: ").strip())
        elif blocker.startswith("missing provider config: "):
            categories["provider_configs"].append(blocker.removeprefix("missing provider config: ").strip())
        elif blocker.startswith("missing required provider in "):
            categories["provider_file_entries"].append(blocker.removeprefix("missing required provider in ").strip())
        elif blocker.startswith("missing required provider: "):
            categories["required_providers"].append(blocker.removeprefix("missing required provider: ").strip())
        elif "missing or empty file: data/processed/" in blocker:
            categories["training_artifacts"].append(blocker.split("missing or empty file: ", 1)[1].strip())
        elif blocker.startswith("required paper tables not synced: "):
            categories["paper_artifacts"].extend(split_comma_list(blocker.removeprefix("required paper tables not synced: ")))
        elif blocker.startswith("required paper figures not synced: "):
            categories["paper_artifacts"].extend(split_comma_list(blocker.removeprefix("required paper figures not synced: ")))

    out: dict[str, Any] = {}
    for category, items in categories.items():
        if not items:
            continue
        if isinstance(items, list):
            out[category] = dedupe(items)
        elif isinstance(items, dict):
            out[category] = {
                key: dedupe(value) if isinstance(value, list) else value
                for key, value in items.items()
                if value
            }
    return out


def collect_missing_provider_env(categories: dict[str, Any], report: dict[str, Any]) -> None:
    for group in report.get("providers", []):
        path = str(group.get("path", ""))
        if not path:
            continue
        for provider in group.get("providers", []):
            name = str(provider.get("name", "")).strip()
            env = str(provider.get("api_key_env", "")).strip()
            if not name or not env or provider.get("api_key_present") is not False:
                continue
            categories["api_env_by_file"].setdefault(path, []).append(f"{name}: {env}")
            categories["api_env_by_provider"][name] = env


def split_comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_operator_handoff(
    missing_prerequisites: dict[str, Any],
    rows_by_claim: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    env_vars = dedupe(
        [
            *missing_prerequisites.get("required_env", []),
            *[item.split(" ", 1)[0] for item in missing_prerequisites.get("api_env", []) if item.strip()],
        ]
    )
    provider_entries = group_provider_file_entries(missing_prerequisites.get("provider_file_entries", []))
    provider_paths = dedupe(
        [
            *missing_prerequisites.get("provider_configs", []),
            *provider_entries.keys(),
            *missing_prerequisites.get("api_env_by_file", {}).keys(),
        ]
    )
    c0_row = (rows_by_claim or {}).get("C0", {})
    return {
        "status": "ready_for_operator_input" if missing_prerequisites else "no_missing_prerequisites",
        "redacted_env_exports": [f"export {name}=<REDACTED>" for name in env_vars],
        "input_files": missing_prerequisites.get("input_files", []),
        "provider_configs": missing_prerequisites.get("provider_configs", []),
        "provider_file_entries": missing_prerequisites.get("provider_file_entries", []),
        "provider_entries_by_file": provider_entries,
        "api_env_by_file": missing_prerequisites.get("api_env_by_file", {}),
        "api_env_by_provider": missing_prerequisites.get("api_env_by_provider", {}),
        "provider_template_commands": provider_template_commands(provider_paths),
        "required_providers": missing_prerequisites.get("required_providers", []),
        "training_artifacts": missing_prerequisites.get("training_artifacts", []),
        "training_data_chain": TRAINING_DATA_CHAIN if missing_prerequisites.get("training_artifacts") else [],
        "training_artifact_producers": training_artifact_producers(
            missing_prerequisites.get("training_artifacts", [])
        ),
        "c0_provenance_action_chain": c0_provenance_action_chain(c0_row),
        "paper_artifacts": missing_prerequisites.get("paper_artifacts", []),
        "validation_commands": [
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json --only real_run_preflight --only sft_data_preflight",
            "python3 scripts/build_submission_gap_report.py --readiness-report outputs/submission_readiness/readiness_report.json --evidence-matrix outputs/evidence_real/evidence_matrix.json --rebuttal-manifest outputs/rebuttal_pack/rebuttal_pack_manifest.json --preflight-report outputs/preflight/real_run_preflight.json --preflight-report outputs/preflight/sft_data_preflight.json --gate-report outputs/contamination_audit/hard_gold_holdout_contamination.json --gate-report outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json --gate-report outputs/contamination_audit/judgebench_downstream_holdout_contamination.json --gate-report outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json --output-dir outputs/submission_readiness/gap_report",
        ],
        "safety_notes": [
            "Do not commit local provider files or API key values.",
            "Keep hard-gold holdouts out of proxy-gold training inputs.",
            "Rerun preflight after changing provider files, API key env vars, or input data.",
        ],
    }


def c0_provenance_action_chain(c0_row: dict[str, Any]) -> list[dict[str, Any]]:
    if not c0_row:
        return []
    if c0_row.get("status") == "safe_to_claim":
        return []
    evidence_text = str(c0_row.get("evidence", ""))
    if not any(
        marker in evidence_text
        for marker in [
            "teacher_rubrics_filtered_report.json",
            "teacher_rubrics_rewardbench2_holdout_filter.json",
            "proxy_gold_build_report.json",
            "sft_output=data/processed/blindspot_sft.jsonl",
            "proxy_gold_output=data/processed/proxy_gold.jsonl",
        ]
    ):
        return []
    return C0_PROVENANCE_ACTION_CHAIN


def group_provider_file_entries(entries: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        if ":" not in entry:
            continue
        path, provider = entry.split(":", 1)
        grouped.setdefault(path.strip(), []).append(provider.strip())
    return {path: dedupe(providers) for path, providers in grouped.items()}


def provider_template_commands(provider_paths: list[str]) -> list[str]:
    commands = []
    for path in provider_paths:
        source = provider_example_path(path)
        if source:
            commands.append(f"test -f {path} || cp {source} {path}")
    return commands


def provider_example_path(path: str) -> str:
    if path.endswith(".local.jsonl"):
        return path.replace(".local.jsonl", ".example.jsonl")
    if path.endswith(".local.json"):
        return path.replace(".local.json", ".example.json")
    return ""


def training_artifact_producers(artifacts: list[str]) -> dict[str, dict[str, Any]]:
    producers: dict[str, dict[str, Any]] = {}
    queue = list(artifacts)
    seen: set[str] = set()
    while queue:
        artifact = queue.pop(0)
        if artifact in seen:
            continue
        seen.add(artifact)
        producer = TRAINING_ARTIFACT_PRODUCERS.get(artifact)
        if not producer:
            continue
        producers[artifact] = producer
        queue.extend(
            upstream
            for upstream in producer.get("upstream_inputs", [])
            if upstream in TRAINING_ARTIFACT_PRODUCERS and upstream not in seen
        )
    return producers


def paper_asset_blockers_by_phase(readiness: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in readiness.get("paper", {}).get("asset_index", []):
        for summary in item.get("blocker_summary", []):
            category = str(summary.get("category", ""))
            mapping = ASSET_BLOCKER_PHASES.get(category)
            if not mapping:
                continue
            phase_id = mapping["phase_id"]
            out.setdefault(phase_id, []).append(
                {
                    "category": category,
                    "count": int(summary.get("count", 0) or 0),
                    "label": summary.get("label", ""),
                    "claim_ids": mapping["claim_ids"],
                    "evidence_gate": mapping["evidence_gate"],
                    "producer": mapping["producer"],
                    "source_artifacts": mapping["source_artifacts"],
                    "paper_artifacts": mapping["paper_artifacts"],
                    "action": mapping["action"],
                }
            )
    return out


def paper_asset_warnings_by_phase(readiness: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    for item in readiness.get("paper", {}).get("asset_index", []):
        for summary in item.get("warning_summary", []):
            warnings.append(
                {
                    "category": str(summary.get("category", "")),
                    "count": int(summary.get("count", 0) or 0),
                    "label": summary.get("label", ""),
                    "warnings": summary.get("warnings", []),
                }
            )
    return {"paper_readiness": warnings} if warnings else {}


def blocker_matches_phase(blocker: str, phase: dict[str, Any], claim_rows: list[dict[str, Any]]) -> bool:
    text = blocker.lower()
    for claim_id in phase["claim_ids"]:
        if re.search(rf"\bclaim\s+{re.escape(claim_id.lower())}\b", text):
            return True
    for row in claim_rows:
        claim_id = str(row.get("claim_id", "")).lower()
        if claim_id and re.search(rf"\b{re.escape(claim_id)}\b", text):
            return True
    return any(keyword.lower() in text for keyword in phase["keywords"])


def summarize_claim_gap(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": row.get("claim_id", ""),
        "status": row.get("status", ""),
        "claim": row.get("claim", ""),
        "missing_or_failed_checks": extract_nonpassing_checks(str(row.get("evidence", ""))),
        "notes": row.get("notes", ""),
    }


def extract_nonpassing_checks(evidence: str) -> list[str]:
    checks = []
    for item in evidence.split("; "):
        stripped = item.strip()
        if stripped.startswith("[missing]") or stripped.startswith("[fail]"):
            checks.append(stripped)
    return checks


def summarize_rebuttal_manifest_gaps(manifest: dict[str, Any] | None) -> list[str]:
    if manifest is None:
        return []
    gaps = []
    counts = manifest.get("defense_status_counts", {})
    concern_templates = manifest.get("concern_templates", {})
    entry_count = int(manifest.get("entry_count", 0) or 0)
    if manifest.get("schema_version") != 1:
        gaps.append(f"unsupported rebuttal manifest schema_version={manifest.get('schema_version')}")
    if entry_count <= 0:
        gaps.append("rebuttal pack has no reviewer concern entries")
    if int(concern_templates.get("count", 0) or 0) != entry_count:
        gaps.append("rebuttal concern template count does not match entry_count")
    if not str(concern_templates.get("sha256", "")).strip():
        gaps.append("rebuttal concern template sha256 is missing")
    if not manifest.get("readiness_ok"):
        gaps.append("rebuttal pack was built while submission readiness was false")
    if int(counts.get("answer_ready", 0) or 0) == 0:
        gaps.append("rebuttal pack has no answer_ready entries")
    if int(counts.get("needs_readiness", 0) or 0):
        gaps.append("rebuttal pack has entries waiting for submission readiness")
    return gaps


def dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BlindSpot-RL Submission Gap Report",
        "",
        f"- Overall ok: `{report['ok']}`",
        f"- Readiness ok: `{report['readiness_ok']}`",
        f"- Hard blockers: `{report['hard_blocker_count']}`",
        f"- Warnings: `{report['warning_count']}`",
        "",
    ]
    claim_ladder = report.get("claim_ladder", [])
    lines.extend(
        [
            "## Claim Ladder Status",
            "",
            "| Level | Status | Required Claims | Blocking Claims |",
            "| --- | --- | --- | --- |",
        ]
    )
    if claim_ladder:
        for row in claim_ladder:
            lines.append(
                f"| {row.get('level', '')} | `{row.get('status', '')}` | "
                f"{', '.join(str(item) for item in row.get('required_claim_ids', []))} | "
                f"{'; '.join(str(item) for item in row.get('missing_or_non_safe_claims', [])) or 'none'} |"
            )
    else:
        lines.append("| none |  |  |  |")
    lines.append("")
    handoff = report.get("operator_handoff", {})
    lines.extend(
        [
            "## Operator Handoff",
            "",
            f"- Status: `{handoff.get('status', 'unknown')}`",
            "",
            "### Redacted Env Exports",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in handoff.get("redacted_env_exports", []) or ["none"])
    lines.extend(["", "### Local Files And Entries", ""])
    for key in [
        "input_files",
        "provider_configs",
        "provider_file_entries",
        "required_providers",
        "training_artifacts",
        "paper_artifacts",
    ]:
        lines.append(f"- `{key}`: {', '.join(f'`{item}`' for item in handoff.get(key, [])) or 'none'}")
    lines.extend(["", "### Provider Template Commands", ""])
    lines.extend(f"- `{item}`" for item in handoff.get("provider_template_commands", []) or ["none"])
    lines.extend(["", "### Provider Entries By File", ""])
    entries_by_file = handoff.get("provider_entries_by_file", {})
    if entries_by_file:
        for path, providers in entries_by_file.items():
            lines.append(f"- `{path}`: {', '.join(f'`{item}`' for item in providers)}")
    else:
        lines.append("- none")
    lines.extend(["", "### API Env By Provider", ""])
    api_env_by_provider = handoff.get("api_env_by_provider", {})
    if api_env_by_provider:
        for provider, env_name in api_env_by_provider.items():
            lines.append(f"- `{provider}`: `{env_name}`")
    else:
        lines.append("- none")
    lines.extend(["", "### API Env By File", ""])
    api_env_by_file = handoff.get("api_env_by_file", {})
    if api_env_by_file:
        for path, entries in api_env_by_file.items():
            lines.append(f"- `{path}`: {', '.join(f'`{item}`' for item in entries)}")
    else:
        lines.append("- none")
    lines.extend(["", "### Training Data Chain", ""])
    training_chain = handoff.get("training_data_chain", [])
    if training_chain:
        for item in training_chain:
            outputs = ", ".join(f"`{value}`" for value in item.get("outputs", [])) or "none"
            checks = ", ".join(f"`{value}`" for value in item.get("checks", [])) or "none"
            lines.append(
                f"- {item.get('order')}. `{item.get('stage')}` (`{item.get('stage_type')}`): "
                f"outputs {outputs}; checks {checks}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "### Training Artifact Producers", ""])
    training_producers = handoff.get("training_artifact_producers", {})
    if training_producers:
        for artifact, producer in training_producers.items():
            upstream = ", ".join(f"`{item}`" for item in producer.get("upstream_inputs", [])) or "none"
            lines.extend(
                [
                    f"- `{artifact}`",
                    f"  - Stage: `{producer.get('stage', '')}` (`{producer.get('stage_type', '')}`)",
                    f"  - Producer: `{producer.get('producer_script', '')}`",
                    f"  - Report: `{producer.get('output_report', '')}`",
                    f"  - Upstream: {upstream}",
                    f"  - SHA contract: `{producer.get('sha_contract', '')}`",
                    f"  - Verify: `{producer.get('verify_after', '')}`",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "### C0 Provenance Action Chain", ""])
    c0_actions = handoff.get("c0_provenance_action_chain", [])
    if c0_actions:
        for item in c0_actions:
            outputs = ", ".join(f"`{value}`" for value in item.get("outputs", [])) or "none"
            checks = ", ".join(f"`{value}`" for value in item.get("checks", [])) or "none"
            lines.extend(
                [
                    f"- {item.get('order')}. `{item.get('stage')}`",
                    f"  - Command: `{item.get('command', '')}`",
                    f"  - Outputs: {outputs}",
                    f"  - Checks: {checks}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "### Validation Commands", ""])
    lines.extend(f"- `{item}`" for item in handoff.get("validation_commands", []) or ["none"])
    lines.extend(["", "### Safety Notes", ""])
    lines.extend(f"- {item}" for item in handoff.get("safety_notes", []) or ["none"])
    lines.append("")
    lines.extend(["## Execution Sequence", ""])
    for item in report.get("execution_sequence", []):
        step_label = item.get("label", item["id"])
        lines.extend(
            [
                f"### {item['order']}. {step_label}",
                "",
                f"- Step id: `{item['id']}`",
                f"- Phase: `{item['phase_id']}` (`{item['phase_status']}`)",
                f"- Summary: {item.get('summary', 'not available')}",
                (
                    "- Blocked by prior phases: "
                    + (
                        ", ".join(item["blocked_by_prior_phases"])
                        if item["blocked_by_prior_phases"]
                        else "none"
                    )
                ),
                f"- Evidence gates: {', '.join(f'`{gate}`' for gate in item['evidence_gates'])}",
                "- Commands:",
                *[f"  - `{command}`" for command in item["commands"]],
                f"- Unlocks: {', '.join(f'`{path}`' for path in item['unlocks'])}",
                f"- Notes: {item['notes']}",
            ]
        )
        if item.get("missing_prerequisites"):
            lines.append("- Missing prerequisites:")
            for category, values in item["missing_prerequisites"].items():
                lines.append(f"  - `{category}`: {', '.join(f'`{value}`' for value in values)}")
        else:
            lines.append("- Missing prerequisites: none")
        lines.append("")
    for phase in report["phases"]:
        lines.extend(
            [
                f"## {phase['name']}",
                "",
                f"- Status: `{phase['status']}`",
                f"- Summary: {phase.get('summary', 'not available')}",
                f"- Depends on: {', '.join(phase['depends_on']) if phase['depends_on'] else 'none'}",
                (
                    "- Blocked by prior phases: "
                    + (
                        ", ".join(phase["blocked_by_prior_phases"])
                        if phase["blocked_by_prior_phases"]
                        else "none"
                    )
                ),
                f"- Claim gaps: {', '.join(phase['claim_ids']) if phase['claim_ids'] else 'none'}",
                "",
                "### Next Actions",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in phase["next_actions"])
        lines.extend(["", "### Claim Discipline", ""])
        lines.append(f"- {phase['claim_discipline']}")
        lines.extend(["", "### Readiness Blockers", ""])
        lines.extend(f"- {item}" for item in phase["readiness_blockers"][:20] or ["none"])
        if len(phase["readiness_blockers"]) > 20:
            lines.append(f"- ... {len(phase['readiness_blockers']) - 20} more")
        lines.extend(["", "### Paper Asset Blockers", ""])
        if not phase.get("paper_asset_blockers"):
            lines.append("- none")
        for item in phase.get("paper_asset_blockers", []):
            lines.append(
                "- "
                + f"`{item['category']}`: {item['count']} "
                + f"({item['label']}; gate {item['evidence_gate']})"
            )
            lines.append(f"  - Producer: `{item['producer']}`")
            lines.append(f"  - Source artifacts: {', '.join(f'`{path}`' for path in item['source_artifacts'])}")
            lines.append(f"  - Paper artifacts: {', '.join(f'`{path}`' for path in item['paper_artifacts'])}")
            lines.append(f"  - Action: {item['action']}")
        lines.extend(["", "### Paper Asset Warnings", ""])
        if not phase.get("paper_asset_warnings"):
            lines.append("- none")
        for item in phase.get("paper_asset_warnings", []):
            lines.append("- " + f"`{item['category']}`: {item['count']} ({item['label']})")
            for warning in item.get("warnings", [])[:5]:
                lines.append(f"  - {warning}")
            if len(item.get("warnings", [])) > 5:
                lines.append(f"  - ... {len(item['warnings']) - 5} more")
        lines.extend(["", "### Manifest Gaps", ""])
        lines.extend(f"- {item}" for item in phase.get("manifest_gaps", []) or ["none"])
        lines.extend(["", "### Claim Checks", ""])
        if not phase["claim_gaps"]:
            lines.append("- none")
        for gap in phase["claim_gaps"]:
            lines.append(f"- `{gap['claim_id']}` {gap['status']}: {gap['claim']}")
            for check in gap["missing_or_failed_checks"][:5]:
                lines.append(f"  - {check}")
            if len(gap["missing_or_failed_checks"]) > 5:
                lines.append(f"  - ... {len(gap['missing_or_failed_checks']) - 5} more")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
