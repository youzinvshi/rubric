#!/usr/bin/env python3
"""Build a machine-readable API handoff for the full real-run pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

try:
    from build_minimal_api_handoff import (
        current_sha_blockers,
        env_export_templates,
        file_sha256,
        load_blockers,
        provider_markdown_rows,
        read_json,
        required_env,
        summarize_preflight,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as scripts.*
    from scripts.build_minimal_api_handoff import (
        current_sha_blockers,
        env_export_templates,
        file_sha256,
        load_blockers,
        provider_markdown_rows,
        read_json,
        required_env,
        summarize_preflight,
    )


PAID_API_STAGE_TYPES = {
    "generate_teachers",
    "generate_model_rubrics",
    "filter_verifier",
    "downstream",
    "multicandidate_downstream",
}


def main() -> None:
    args = parse_args()
    report = build_handoff(
        pipeline_path=args.pipeline,
        preflight_path=args.preflight,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(f"Real API handoff status={report['status']} output={args.output_json}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full real-run API handoff report.")
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_handoff(
    pipeline_path: Path,
    preflight_path: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    pipeline = read_json(pipeline_path)
    preflight = read_json(preflight_path)
    stages = pipeline.get("stages", []) if isinstance(pipeline, dict) else []
    stage_names = [stage.get("name") for stage in stages if isinstance(stage, dict)]
    api_budget_stages = [stage for stage in stages if stage.get("type") == "api_budget"]
    paid_api_stages = [stage for stage in stages if stage.get("type") in PAID_API_STAGE_TYPES]
    manual_gate_stages = [stage for stage in stages if stage.get("type") == "manual_gate"]

    blockers: list[str] = []
    blockers.extend(load_blockers("pipeline", pipeline))
    blockers.extend(load_blockers("preflight", preflight))
    blockers.extend(current_sha_blockers("preflight input", preflight.get("inputs", [])))
    blockers.extend(current_sha_blockers("preflight provider", preflight.get("providers", [])))

    api_budgets = summarize_api_budgets(api_budget_stages)
    manual_gates = summarize_manual_gates(manual_gate_stages)
    paid_plan = summarize_paid_api_stages(paid_api_stages, api_budgets, manual_gates, preflight)

    for budget in api_budgets:
        blockers.extend(f"api_budget {budget['name']}: {item}" for item in budget.get("blockers", []))
    for gate in manual_gates:
        if gate.get("status") != "pass":
            blockers.append(
                f"manual_gate {gate['name']}: {len(gate.get('blockers', []))} blockers"
            )
    for stage in paid_plan:
        if stage.get("status") != "ready_after_shared_gates":
            blockers.append(f"paid_api_stage {stage['name']}: {stage['status']}")

    commands = build_commands(pipeline_path, stages, paid_api_stages, output_json)
    report = {
        "schema_version": 1,
        "scope": "full_real_run_api_handoff",
        "note": (
            "This handoff only audits API/preflight/budget readiness. "
            "It is not experimental evidence and does not permit any paper claim."
        ),
        "ok": not blockers,
        "status": "ready_for_paid_run" if not blockers else "blocked",
        "pipeline": str(pipeline_path),
        "pipeline_sha256": file_sha256(pipeline_path) if pipeline_path.exists() else "",
        "preflight": summarize_preflight(preflight),
        "api_budget_count": len(api_budgets),
        "api_budgets": api_budgets,
        "manual_gates": manual_gates,
        "paid_api_stage_count": len(paid_plan),
        "paid_api_stages": paid_plan,
        "required_env": required_env(preflight),
        "commands": commands,
        "blockers": blockers,
    }
    report["resume_requirements"] = build_resume_requirements(report)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(to_markdown(report), encoding="utf-8")
    return report


def summarize_api_budgets(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for stage in stages:
        args = stage.get("args", {}) if isinstance(stage.get("args"), dict) else {}
        output = args.get("output")
        path = Path(str(output)) if output else None
        data = read_json(path) if path is not None else {"_load_error": "missing output path"}
        blockers = load_blockers("api_budget", data)
        blockers.extend(budget_current_sha_blockers(data))
        total = data.get("total", {}) if isinstance(data.get("total"), dict) else {}
        contract = data.get("contract", {}) if isinstance(data.get("contract"), dict) else {}
        summaries.append(
            {
                "name": stage.get("name"),
                "path": str(path) if path is not None else "",
                "present": bool(path and path.exists()),
                "ok": data.get("ok"),
                "status": "pass" if data.get("ok") and not blockers else "blocked",
                "calls": total.get("calls"),
                "total_tokens": total.get("total_tokens", total.get("tokens")),
                "estimated_cost_usd": total.get("estimated_cost_usd", total.get("cost_usd")),
                "contract": contract,
                "blockers": blockers,
            }
        )
    return summaries


def budget_current_sha_blockers(data: dict[str, Any]) -> list[str]:
    if data.get("_load_error"):
        return []
    contract = data.get("contract", {})
    if not isinstance(contract, dict):
        return ["missing contract"]
    blockers = []
    for key in ("input", "providers", "resume_output"):
        path_text = contract.get(key)
        expected_sha = str(contract.get(f"{key}_sha256", "") or "")
        if not path_text or not expected_sha:
            continue
        path = Path(str(path_text))
        if not path.exists():
            blockers.append(f"contract {key} is missing but had sha256: {path}")
            continue
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            blockers.append(f"contract {key} sha256 does not match current file: {path}")
    return blockers


def summarize_manual_gates(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for stage in stages:
        args = stage.get("args", {}) if isinstance(stage.get("args"), dict) else {}
        output = args.get("output")
        path = Path(str(output)) if output else None
        data = read_json(path) if path is not None else {"_load_error": "missing output path"}
        blockers = load_blockers("manual_gate", data)
        summaries.append(
            {
                "name": stage.get("name"),
                "path": str(path) if path is not None else "",
                "present": bool(path and path.exists()),
                "ok": data.get("ok"),
                "status": "pass" if data.get("ok") and not blockers else "blocked",
                "blockers": blockers,
            }
        )
    return summaries


def summarize_paid_api_stages(
    stages: list[dict[str, Any]],
    api_budgets: list[dict[str, Any]],
    manual_gates: list[dict[str, Any]],
    preflight: dict[str, Any],
) -> list[dict[str, Any]]:
    budget_by_path = {budget.get("path"): budget for budget in api_budgets}
    training_gate_blocked = any(
        gate.get("name") in {"training_completion_gate", "matrix_trained_method_gate"}
        and gate.get("status") != "pass"
        for gate in manual_gates
    )
    plan = []
    for stage in stages:
        args = stage.get("args", {}) if isinstance(stage.get("args"), dict) else {}
        required_budget = args.get("require_budget_report")
        budget = budget_by_path.get(str(required_budget)) if required_budget else None
        status = "ready_after_shared_gates"
        if preflight.get("ok") is not True:
            status = "blocked_by_preflight"
        elif required_budget and (budget is None or budget.get("status") != "pass"):
            status = "blocked_by_api_budget"
        elif stage.get("type") == "generate_model_rubrics" and training_gate_blocked:
            status = "blocked_by_training_gate"
        plan.append(
            {
                "name": stage.get("name"),
                "type": stage.get("type"),
                "input": args.get("input"),
                "providers": args.get("providers") or args.get("provider"),
                "output": args.get("output"),
                "require_budget_report": required_budget,
                "require_preflight_report": args.get("require_preflight_report"),
                "status": status,
            }
        )
    return plan


def build_commands(
    pipeline_path: Path,
    stages: list[dict[str, Any]],
    paid_api_stages: list[dict[str, Any]],
    output_json: Path | None,
) -> dict[str, str]:
    base = ["python3", "scripts/run_experiment_pipeline.py", "--config", str(pipeline_path)]
    api_budget_names = [str(stage.get("name")) for stage in stages if stage.get("type") == "api_budget" and stage.get("name")]
    preflight_names = [str(stage.get("name")) for stage in stages if stage.get("type") == "preflight" and stage.get("name")]
    commands = {
        "refresh_preflight": format_command([*base, *only_args(preflight_names)]),
        "refresh_api_budgets": format_command([*base, *only_args(api_budget_names)]),
        "refresh_blocked_reports": format_command([*base, "--from-stage", "audit", "--to-stage", "paper_asset_index_check_final_real"]),
    }
    if paid_api_stages:
        start = str(paid_api_stages[0].get("name"))
        end = "result_card_real" if any(stage.get("name") == "result_card_real" for stage in stages) else str(paid_api_stages[-1].get("name"))
        commands["paid_range_dry_run"] = format_command([*base, "--from-stage", start, "--to-stage", end, "--dry-run"])
        commands["paid_range_run"] = format_command([*base, "--from-stage", start, "--to-stage", end])
    if output_json is not None:
        commands["handoff_ready_check"] = format_command(["python3", "scripts/check_real_api_handoff_ready.py", "--handoff", str(output_json)])
    return commands


def only_args(names: list[str]) -> list[str]:
    args = []
    for name in names:
        args.extend(["--only", name])
    return args


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_resume_requirements(report: dict[str, Any]) -> dict[str, Any]:
    missing_env = sorted(
        {
            str(item.get("name"))
            for item in report.get("required_env", [])
            if item.get("name") and not item.get("present")
        }
        | {
            str(provider.get("api_key_env"))
            for group in report.get("preflight", {}).get("providers", [])
            for provider in group.get("providers", [])
            if provider.get("api_key_env") and not provider.get("api_key_present")
        }
    )
    failed_local = []
    for group in report.get("preflight", {}).get("providers", []):
        for provider in group.get("providers", []):
            health = provider.get("local_health", {})
            if health.get("checked") and health.get("ok") is False:
                failed_local.append(
                    {
                        "name": provider.get("name"),
                        "base_url": provider.get("base_url"),
                        "host": health.get("host"),
                        "port": health.get("port"),
                        "error": health.get("error", ""),
                    }
                )
    missing_budgets = [item["name"] for item in report.get("api_budgets", []) if item.get("status") != "pass"]
    blocked_manual_gates = [item["name"] for item in report.get("manual_gates", []) if item.get("status") != "pass"]
    commands = report.get("commands", {})
    if report.get("ok"):
        next_command = commands.get("paid_range_run")
    elif report.get("preflight", {}).get("ok") is not True:
        next_command = commands.get("refresh_preflight")
    elif missing_budgets:
        next_command = commands.get("refresh_api_budgets")
    else:
        next_command = commands.get("refresh_blocked_reports")
    return {
        "ready": bool(report.get("ok")),
        "preflight_ok": report.get("preflight", {}).get("ok"),
        "missing_env": missing_env,
        "env_export_templates": env_export_templates(missing_env),
        "failed_local_providers": failed_local,
        "missing_or_blocked_api_budgets": missing_budgets,
        "blocked_manual_gates": blocked_manual_gates,
        "partial_paid_run_allowed": False,
        "partial_paid_run_reason": "Full real-run API execution requires shared preflight, all relevant budget gates, and training/manual gates to pass.",
        "next_command": next_command,
        "paid_run_command": commands.get("paid_range_run") if report.get("ok") else None,
        "blocked_report_refresh_command": commands.get("refresh_blocked_reports"),
    }


def to_markdown(report: dict[str, Any]) -> str:
    resume = report.get("resume_requirements", {})
    lines = [
        "# Real Run API Handoff",
        "",
        f"- Status: `{report['status']}`",
        f"- OK: `{str(report['ok']).lower()}`",
        f"- Scope: `{report['scope']}`",
        f"- Note: {report['note']}",
        f"- API budget stages: `{report['api_budget_count']}`",
        f"- Paid/API stages: `{report['paid_api_stage_count']}`",
        "",
        "## Resume Requirements",
        "",
        f"- Ready: `{str(resume.get('ready')).lower()}`",
        f"- Preflight OK: `{resume.get('preflight_ok')}`",
        f"- Missing env: `{', '.join(resume.get('missing_env', [])) or 'none'}`",
        f"- Missing or blocked API budgets: `{', '.join(resume.get('missing_or_blocked_api_budgets', [])) or 'none'}`",
        f"- Blocked manual gates: `{', '.join(resume.get('blocked_manual_gates', [])) or 'none'}`",
        f"- Partial paid run allowed: `{str(resume.get('partial_paid_run_allowed')).lower()}`",
        f"- Partial paid run reason: {resume.get('partial_paid_run_reason')}",
        "",
        "## Provider Preflight",
        "",
        "| Provider File | Name | Base URL | API Key Env | API Key Present | Local Health |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(provider_markdown_rows(report.get("preflight", {}).get("providers", [])) or ["| none | none | none | none | none | none |"])
    lines.extend(["", "## API Budget Gates", "", "| Stage | Status | Calls | Path |", "| --- | --- | ---: | --- |"])
    for budget in report.get("api_budgets", []):
        lines.append(f"| `{budget.get('name')}` | `{budget.get('status')}` | `{budget.get('calls')}` | `{budget.get('path')}` |")
    lines.extend(["", "## Paid/API Stage Plan", "", "| Stage | Type | Status | Budget |", "| --- | --- | --- | --- |"])
    for stage in report.get("paid_api_stages", []):
        lines.append(
            f"| `{stage.get('name')}` | `{stage.get('type')}` | `{stage.get('status')}` | `{stage.get('require_budget_report') or 'none'}` |"
        )
    env_templates = resume.get("env_export_templates", [])
    if env_templates:
        lines.extend(["", "## Env Export Template", "", "```bash", *env_templates, "```"])
    if resume.get("next_command"):
        lines.extend(["", "## Next Command", "", "```bash", resume["next_command"], "```"])
    lines.extend(["", "## Commands", ""])
    for name, command in report.get("commands", {}).items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend(["## Blockers", ""])
    lines.extend([f"- {item}" for item in report.get("blockers", [])] or ["- none"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
