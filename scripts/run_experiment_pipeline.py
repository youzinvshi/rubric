#!/usr/bin/env python3
"""Run BlindSpot-RL experiment stages from a JSON config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


STAGE_SCRIPTS = {
    "download": "download_public_data.py",
    "init_data_source_config": "init_data_source_local_config.py",
    "data_source_report": "build_data_source_report.py",
    "profile_data": "profile_dataset_schema.py",
    "schema_contract": "check_downstream_schema.py",
    "normalize": "normalize_dataset.py",
    "split_dataset": "split_dataset.py",
    "filter_verifier": "filter_rubrics_with_verifier.py",
    "build_sft": "build_sft_data.py",
    "convert_verl": "convert_to_verl_parquet.py",
    "convert_policy_rlvr": "convert_policy_rlvr_data.py",
    "bsc_gold_sanity": "run_bsc_gold_sanity.py",
    "minimal_bsc_chain_smoke": "run_minimal_bsc_chain_smoke.py",
    "minimal_api_handoff": "build_minimal_api_handoff.py",
    "prepare_bsc": "prepare_bsc_eval.py",
    "prepare_downstream": "prepare_downstream_eval.py",
    "prepare_multicandidate": "prepare_multicandidate_eval.py",
    "bsc": "bsc_diagnose.py",
    "bsc_human_audit_pack": "build_bsc_human_audit_pack.py",
    "bsc_human_audit_summary": "summarize_bsc_human_audit_labels.py",
    "bsc_sweep": "sweep_bsc_thresholds.py",
    "blindspot_map": "blindspot_attribution.py",
    "budget_curve": "evaluate_budget_curve.py",
    "bootstrap_ci": "bootstrap_metric_ci.py",
    "dimension_transition": "evaluate_blindspot_repair.py",
    # Backward-compatible alias for historical local configs.
    "blindspot_repair": "evaluate_blindspot_repair.py",
    "semantic_space": "build_semantic_space_visualization.py",
    "ablation": "run_bsc_ablation.py",
    "teacher_union_ablation": "run_teacher_union_ablation.py",
    "verifier_filter_ablation": "run_verifier_filter_ablation.py",
    "downstream": "evaluate_downstream.py",
    "multicandidate_downstream": "evaluate_multicandidate_downstream.py",
    "summarize": "summarize_experiments.py",
    "audit": "audit_experiment.py",
    "evidence": "build_evidence_matrix.py",
    "export": "export_paper_artifacts.py",
    "sync_paper": "sync_paper_artifacts.py",
    "paper_asset_index_check": "check_paper_asset_index.py",
    "latex_compile_check": "check_latex_compile.py",
    "submission_readiness": "check_submission_readiness.py",
    "submission_gap_report": "build_submission_gap_report.py",
    "dashboard": "build_run_dashboard.py",
    "result_card": "build_result_card.py",
    "sprint_plan": "build_sprint_plan.py",
    "rebuttal_pack": "build_rebuttal_pack.py",
    "preflight": "preflight_real_run.py",
    "api_budget": "estimate_api_budget.py",
    "validate_rubrics": "validate_rubric_outputs.py",
    "validate_gold": "validate_gold_data.py",
    "holdout_contamination_audit": "audit_holdout_contamination.py",
    "filter_holdout_contamination": "filter_holdout_contamination.py",
    "sample_records": "sample_records.py",
    "generate_teachers": "generate_teacher_rubrics.py",
    "generate_model_rubrics": "generate_model_rubrics.py",
    "register_llamafactory": "register_llamafactory_dataset.py",
    "training_commands": "make_training_commands.py",
    "downstream_rlvr_commands": "make_downstream_rlvr_commands.py",
    "manual_gate": "check_manual_gate.py",
}


NARGS_LIST_ARGS = {
    ("bsc_sweep", "coverage_tau"),
    ("bsc_sweep", "redundancy_tau"),
    ("budget_curve", "k"),
}


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    stages = config.get("stages", [])
    if not stages:
        raise SystemExit("Pipeline config has no stages.")

    selected_stages = select_stages(stages, only=args.only, from_stage=args.from_stage, to_stage=args.to_stage)
    if args.require_ready_handoff:
        require_ready_handoff(args.require_ready_handoff)
    for stage in selected_stages:
        cmd = build_command(stage)
        print("+ " + format_command(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BlindSpot-RL experiment pipeline config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", help="Run only matching stage name. Repeatable.")
    parser.add_argument("--from-stage", help="Run stages starting at this stage name, inclusive.")
    parser.add_argument("--to-stage", help="Run stages through this stage name, inclusive.")
    parser.add_argument(
        "--require-ready-handoff",
        type=Path,
        help="Fail unless the minimal API handoff explicitly allows paid execution.",
    )
    return parser.parse_args()


def load_pipeline_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Pipeline config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Pipeline config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Pipeline config must be a JSON object: {path}")
    return data


def select_stages(
    stages: list[dict[str, Any]],
    only: list[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[dict[str, Any]]:
    names = [stage["name"] for stage in stages]
    if from_stage and from_stage not in names:
        raise SystemExit(f"--from-stage was not found in pipeline: {from_stage}")
    if to_stage and to_stage not in names:
        raise SystemExit(f"--to-stage was not found in pipeline: {to_stage}")

    start = names.index(from_stage) if from_stage else 0
    end = names.index(to_stage) if to_stage else len(stages) - 1
    if start > end:
        raise SystemExit(f"--from-stage must not come after --to-stage: {from_stage} > {to_stage}")

    selected = stages[start : end + 1]
    if only:
        requested = set(only)
        missing = [name for name in only if name not in names]
        if missing:
            raise SystemExit("--only stage was not found in pipeline: " + ", ".join(missing))
        selected = [stage for stage in selected if stage["name"] in requested]
    return selected


def require_ready_handoff(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required API handoff is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Required API handoff is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Required API handoff root must be a JSON object: {path}")
    resume = data.get("resume_requirements", {}) if isinstance(data.get("resume_requirements"), dict) else {}
    blockers = data.get("blockers", []) if isinstance(data.get("blockers"), list) else []
    ready = (
        data.get("ok") is True
        and data.get("status") == "ready_for_paid_run"
        and resume.get("ready") is True
        and not blockers
    )
    if ready:
        return
    missing_env = resume.get("missing_env", [])
    next_command = resume.get("next_command")
    details = [
        f"Required API handoff is not ready: {path}",
        f"status={data.get('status', 'unknown')} ok={data.get('ok')}",
        f"missing_env={','.join(str(item) for item in missing_env) if missing_env else 'none'}",
    ]
    if next_command:
        details.append(f"next_command={next_command}")
    raise SystemExit("\n".join(details))


def build_command(stage: dict[str, Any]) -> list[str]:
    stage_type = stage.get("type", stage["name"])
    script = STAGE_SCRIPTS.get(stage_type)
    if not script:
        raise ValueError(f"Unknown stage type: {stage_type}")

    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    args = stage.get("args", {})
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            if (stage_type, key) in NARGS_LIST_ARGS:
                cmd.append(flag)
                cmd.extend(str(item) for item in value)
                continue
            for item in value:
                cmd.extend([flag, str(item)])
            continue
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    return cmd


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


if __name__ == "__main__":
    main()
