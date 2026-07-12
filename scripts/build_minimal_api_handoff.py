#!/usr/bin/env python3
"""Build a machine-readable handoff for the minimal paid API boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

OFFLINE_HANDOFF_STAGE_TYPES = {
    "init_data_source_config",
    "data_source_report",
    "validate_gold",
    "bsc_gold_sanity",
    "sample_records",
    "preflight",
    "api_budget",
    "minimal_api_handoff",
}


def main() -> None:
    args = parse_args()
    report = build_handoff(
        pipeline_path=args.pipeline,
        preflight_path=args.preflight,
        api_budget_path=args.api_budget,
        bsc_gold_sanity_path=args.bsc_gold_sanity,
        output_json=args.output_json,
        output_md=args.output_md,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
    )
    print(f"Minimal API handoff status={report['status']} output={args.output_json}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal-claim paid API handoff report.")
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--api-budget", required=True, type=Path)
    parser.add_argument("--bsc-gold-sanity", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--start-stage", default="generate_model_rubrics")
    parser.add_argument("--end-stage", default="result_card")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_handoff(
    pipeline_path: Path,
    preflight_path: Path,
    api_budget_path: Path,
    bsc_gold_sanity_path: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
    start_stage: str = "generate_model_rubrics",
    end_stage: str = "result_card",
) -> dict[str, Any]:
    pipeline = read_json(pipeline_path)
    preflight = read_json(preflight_path)
    budget = read_json(api_budget_path)
    sanity = read_json(bsc_gold_sanity_path)
    stages = pipeline.get("stages", []) if isinstance(pipeline, dict) else []
    stage_names = [stage.get("name") for stage in stages]

    blockers = []
    blockers.extend(load_blockers("pipeline", pipeline))
    blockers.extend(load_blockers("preflight", preflight))
    blockers.extend(load_blockers("api_budget", budget))
    blockers.extend(load_blockers("bsc_gold_sanity", sanity))
    blockers.extend(current_sha_blockers("preflight input", preflight.get("inputs", [])))
    blockers.extend(current_sha_blockers("preflight provider", preflight.get("providers", [])))
    if start_stage not in stage_names:
        blockers.append(f"start stage is missing from pipeline: {start_stage}")
    if end_stage not in stage_names:
        blockers.append(f"end stage is missing from pipeline: {end_stage}")
    if start_stage in stage_names and end_stage in stage_names and stage_names.index(start_stage) > stage_names.index(end_stage):
        blockers.append(f"start stage comes after end stage: {start_stage} > {end_stage}")
    generate_args = stage_args(stages, "generate_model_rubrics")
    blockers.extend(budget_contract_blockers(budget, generate_args))

    commands = build_commands(
        pipeline_path,
        start_stage=start_stage,
        end_stage=end_stage,
        offline_stage_names=offline_handoff_stage_names(stages, start_stage),
        handoff_path=output_json,
    )
    report = {
        "ok": not blockers,
        "status": "ready_for_paid_run" if not blockers else "blocked",
        "pipeline": str(pipeline_path),
        "pipeline_sha256": file_sha256(pipeline_path) if pipeline_path.exists() else "",
        "preflight": summarize_preflight(preflight),
        "api_budget": summarize_budget(budget),
        "bsc_gold_sanity": summarize_sanity(sanity),
        "stage_range": {
            "start": start_stage,
            "end": end_stage,
            "start_index": stage_names.index(start_stage) if start_stage in stage_names else None,
            "end_index": stage_names.index(end_stage) if end_stage in stage_names else None,
        },
        "paid_range_stages": summarize_paid_range_stages(stages, start_stage, end_stage),
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_load_error": f"missing file: {path}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": f"invalid JSON at line {exc.lineno} column {exc.colno}: {path}"}
    if not isinstance(data, dict):
        return {"_load_error": f"JSON root must be an object: {path}"}
    return data


def load_blockers(name: str, data: dict[str, Any]) -> list[str]:
    if data.get("_load_error"):
        return [f"{name}: {data['_load_error']}"]
    blockers = []
    if data.get("ok") is False:
        blockers.append(f"{name}: ok=false")
    seen: set[str] = set()
    for key in ("hard_blockers", "blockers"):
        for blocker in data.get(key, []) or []:
            if blocker in seen:
                continue
            seen.add(blocker)
            blockers.append(f"{name}: {blocker}")
    return blockers


def summarize_preflight(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": data.get("ok"),
        "hard_blockers": data.get("hard_blockers", []),
        "inputs": summarize_paths(data.get("inputs", [])),
        "providers": summarize_provider_groups(data.get("providers", [])),
        "required_env": required_env(data),
    }


def summarize_budget(data: dict[str, Any]) -> dict[str, Any]:
    total = data.get("total", {}) if isinstance(data.get("total"), dict) else {}
    contract = data.get("contract", {}) if isinstance(data.get("contract"), dict) else {}
    return {
        "ok": data.get("ok"),
        "blockers": data.get("blockers", []),
        "calls": total.get("calls"),
        "total_tokens": total.get("total_tokens", total.get("tokens")),
        "estimated_cost_usd": total.get("estimated_cost_usd", total.get("cost_usd")),
        "resume_output": contract.get("resume_output", data.get("resume_output")),
        "resume_output_sha256": contract.get("resume_output_sha256", data.get("resume_output_sha256")),
        "contract": contract,
    }


def summarize_sanity(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": data.get("ok"),
        "n_joined": data.get("n_joined", data.get("n")),
        "mean_coverage": data.get("mean_coverage"),
        "mean_blind": data.get("mean_blind"),
        "blockers": data.get("blockers", []),
    }


def summarize_paths(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(
                {
                    "path": item.get("path"),
                    "present": item.get("present"),
                    "sha256": item.get("sha256"),
                    "records": item.get("records"),
                }
            )
    return out


def summarize_provider_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for item in items:
        if not isinstance(item, dict):
            continue
        group = {
            "path": item.get("path"),
            "present": item.get("present"),
            "sha256": item.get("sha256"),
            "records": item.get("records"),
            "providers": [],
        }
        for provider in item.get("providers", []) or []:
            if not isinstance(provider, dict):
                continue
            group["providers"].append(
                {
                    "name": provider.get("name"),
                    "model": provider.get("model"),
                    "base_url": provider.get("base_url"),
                    "api_key_env": provider.get("api_key_env"),
                    "api_key_present": provider.get("api_key_present"),
                    "base_url_valid": provider.get("base_url_valid"),
                    "local_health": provider.get("local_health", {}),
                }
            )
        groups.append(group)
    return groups


def required_env(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    env_items = preflight.get("env", [])
    if isinstance(env_items, list):
        return [
            {"name": item.get("name"), "present": item.get("present"), "length": item.get("length")}
            for item in env_items
            if isinstance(item, dict)
        ]
    return []


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
    failed_local_providers = []
    for group in report.get("preflight", {}).get("providers", []):
        for provider in group.get("providers", []):
            health = provider.get("local_health", {})
            if health.get("checked") and health.get("ok") is False:
                failed_local_providers.append(
                    {
                        "name": provider.get("name"),
                        "base_url": provider.get("base_url"),
                        "host": health.get("host"),
                        "port": health.get("port"),
                        "error": health.get("error", ""),
                    }
                )
    ready = bool(report.get("ok"))
    commands = report.get("commands", {})
    paid_stage_plan = build_paid_stage_plan(report)
    return {
        "ready": ready,
        "preflight_ok": report.get("preflight", {}).get("ok"),
        "api_budget_ok": report.get("api_budget", {}).get("ok"),
        "bsc_gold_sanity_ok": report.get("bsc_gold_sanity", {}).get("ok"),
        "missing_env": missing_env,
        "env_export_templates": env_export_templates(missing_env),
        "failed_local_providers": failed_local_providers,
        "partial_paid_run_allowed": False,
        "partial_paid_run_reason": "Paid stages require the shared preflight report to be ok=true before any API calls.",
        "paid_stage_plan": paid_stage_plan,
        "next_command": commands.get("paid_range_run") if ready else commands.get("rerun_offline_gates"),
        "paid_run_command": commands.get("paid_range_run") if ready else None,
        "offline_rerun_command": commands.get("rerun_offline_gates"),
        "blocked_report_refresh_command": commands.get("refresh_blocked_reports"),
    }


def env_export_templates(env_names: list[str]) -> list[str]:
    return [f"export {name}=<set-{name.lower().replace('_', '-')}>"
            for name in env_names]


def build_paid_stage_plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    provider_groups = report.get("preflight", {}).get("providers", [])
    generator = provider_records_for_path(provider_groups, report.get("api_budget", {}).get("contract", {}).get("providers"))
    verifier = [
        provider
        for group in provider_groups
        if group.get("path") != report.get("api_budget", {}).get("contract", {}).get("providers")
        for provider in group.get("providers", [])
    ]
    plan = [
        stage_requirement("generate_model_rubrics", generator),
    ]
    verifier_budget = paid_range_stage(report, "verifier_api_budget")
    if verifier_budget:
        plan.append(
            {
                "stage": "verifier_api_budget",
                "requires_env": [],
                "missing_env": [],
                "input": verifier_budget.get("args", {}).get("input"),
                "output": verifier_budget.get("args", {}).get("output"),
                "status": "blocked_until_model_rubrics_exist",
            }
        )
    plan.append(stage_requirement("verify_model_rubrics", verifier))
    plan.append(
        {
            "stage": "bsc_diagnostic_chain",
            "requires_env": [],
            "missing_env": [],
            "requires_verified_valid_flags": True,
            "status": "blocked_until_verified_model_rubrics_exist",
        }
    )
    return plan


def summarize_paid_range_stages(stages: list[Any], start_stage: str, end_stage: str) -> list[dict[str, Any]]:
    names = [stage.get("name") for stage in stages if isinstance(stage, dict)]
    if start_stage not in names or end_stage not in names:
        return []
    start = names.index(start_stage)
    end = names.index(end_stage)
    if start > end:
        return []
    out = []
    for stage in stages[start : end + 1]:
        if not isinstance(stage, dict):
            continue
        out.append(
            {
                "name": stage.get("name"),
                "type": stage.get("type", stage.get("name")),
                "args": stage.get("args", {}),
            }
        )
    return out


def paid_range_stage(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in report.get("paid_range_stages", []):
        if isinstance(stage, dict) and stage.get("name") == name:
            return stage
    return None


def provider_records_for_path(groups: Any, provider_path: Any) -> list[dict[str, Any]]:
    if not provider_path:
        return []
    return [
        provider
        for group in groups
        if isinstance(group, dict) and normalize_contract_value(group.get("path")) == normalize_contract_value(provider_path)
        for provider in group.get("providers", [])
        if isinstance(provider, dict)
    ]


def stage_requirement(stage: str, providers: list[dict[str, Any]]) -> dict[str, Any]:
    required = sorted({str(provider.get("api_key_env")) for provider in providers if provider.get("api_key_env")})
    missing = sorted({str(provider.get("api_key_env")) for provider in providers if provider.get("api_key_env") and not provider.get("api_key_present")})
    return {
        "stage": stage,
        "requires_env": required,
        "missing_env": missing,
        "providers": [
            {
                "name": provider.get("name"),
                "base_url": provider.get("base_url"),
                "api_key_env": provider.get("api_key_env"),
                "api_key_present": provider.get("api_key_present"),
                "local_health": provider.get("local_health", {}),
            }
            for provider in providers
        ],
        "status": "ready_after_shared_preflight" if not missing else "missing_env",
    }


def build_commands(
    pipeline_path: Path,
    start_stage: str,
    end_stage: str,
    offline_stage_names: list[str] | None = None,
    handoff_path: Path | None = None,
) -> dict[str, str]:
    base = ["python3", "scripts/run_experiment_pipeline.py", "--config", str(pipeline_path)]
    offline_stage_names = offline_stage_names or [
        "sample_queries",
        "preflight",
        "api_budget",
        "bsc_gold_sanity",
        "minimal_api_handoff",
    ]
    offline_args = []
    for stage_name in offline_stage_names:
        offline_args.extend(["--only", stage_name])
    commands = {
        "rerun_offline_gates": " ".join(
            shlex.quote(part) for part in [*base, *offline_args]
        ),
        "refresh_blocked_reports": " ".join(
            shlex.quote(part)
            for part in [
                *base,
                "--from-stage",
                "audit",
                "--to-stage",
                "paper_asset_index_check_post_sync",
            ]
        ),
    }
    if handoff_path is not None:
        gated_range = [
            *base,
            "--from-stage",
            start_stage,
            "--to-stage",
            end_stage,
            "--require-ready-handoff",
            str(handoff_path),
        ]
        commands["paid_range_dry_run"] = " ".join(
            shlex.quote(part) for part in [*gated_range, "--dry-run"]
        )
        commands["paid_range_run"] = " ".join(
            shlex.quote(part) for part in gated_range
        )
        commands["handoff_ready_check"] = " ".join(
            shlex.quote(part)
            for part in [
                "python3",
                "scripts/check_minimal_api_handoff_ready.py",
                "--handoff",
                str(handoff_path),
            ]
        )
    else:
        commands["paid_range_dry_run"] = " ".join(shlex.quote(part) for part in [*base, "--from-stage", start_stage, "--to-stage", end_stage, "--dry-run"])
        commands["paid_range_run"] = " ".join(shlex.quote(part) for part in [*base, "--from-stage", start_stage, "--to-stage", end_stage])
    return commands


def offline_handoff_stage_names(stages: list[Any], start_stage: str) -> list[str]:
    names = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        if name == start_stage:
            break
        stage_type = stage.get("type", name)
        if name and stage_type in OFFLINE_HANDOFF_STAGE_TYPES:
            names.append(str(name))
    return names


def stage_args(stages: list[Any], name: str) -> dict[str, Any]:
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == name and isinstance(stage.get("args"), dict):
            return stage["args"]
    return {}


def current_sha_blockers(label: str, items: Any) -> list[str]:
    blockers = []
    if not isinstance(items, list):
        return blockers
    for item in items:
        if not isinstance(item, dict):
            continue
        path_text = item.get("path")
        expected = item.get("sha256")
        if not path_text or not expected:
            continue
        path = Path(str(path_text))
        if not path.exists():
            blockers.append(f"{label} disappeared after preflight: {path}")
            continue
        actual = file_sha256(path)
        if actual != expected:
            blockers.append(f"{label} sha256 changed after preflight: {path}")
    return blockers


def budget_contract_blockers(budget: dict[str, Any], generate_args: dict[str, Any]) -> list[str]:
    if budget.get("_load_error"):
        return []
    contract = budget.get("contract", {})
    if not isinstance(contract, dict):
        return ["api_budget: missing contract"]
    blockers = []
    comparisons = [
        ("input", generate_args.get("input")),
        ("providers", generate_args.get("providers")),
        ("resume_output", generate_args.get("output")),
        ("method_key", "method"),
        ("calls_per_record_per_provider", 1),
    ]
    for key, expected in comparisons:
        if expected is None:
            continue
        actual = contract.get(key)
        if normalize_contract_value(actual) != normalize_contract_value(expected):
            blockers.append(f"api_budget contract {key} does not match pipeline generate_model_rubrics: {actual} != {expected}")

    for key in ("input", "providers", "resume_output"):
        path_text = contract.get(key)
        sha_key = f"{key}_sha256"
        expected_sha = str(contract.get(sha_key, "") or "")
        if not path_text:
            continue
        path = Path(str(path_text))
        if not path.exists():
            if expected_sha:
                blockers.append(f"api_budget contract {key} is missing but had sha256: {path}")
            continue
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            blockers.append(f"api_budget contract {key} sha256 does not match current file: {path}")
    return blockers


def normalize_contract_value(value: Any) -> str:
    return "" if value is None else str(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Minimal Claim API Handoff",
        "",
        f"- Status: `{report['status']}`",
        f"- OK: `{str(report['ok']).lower()}`",
        f"- Stage range: `{report['stage_range']['start']}` -> `{report['stage_range']['end']}`",
        "",
        "## Required Environment",
        "",
    ]
    for item in report["required_env"]:
        lines.append(f"- `{item.get('name')}` present=`{item.get('present')}` length=`{item.get('length')}`")
    if not report["required_env"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Provider Preflight",
            "",
            "| Provider File | Name | Base URL | API Key Env | API Key Present | Local Health |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    provider_rows = provider_markdown_rows(report.get("preflight", {}).get("providers", []))
    lines.extend(provider_rows or ["| none | none | none | none | none | none |"])
    resume = report.get("resume_requirements", {})
    lines.extend(
        [
            "",
            "## Resume Requirements",
            "",
            f"- Ready: `{str(resume.get('ready')).lower()}`",
            f"- Preflight OK: `{resume.get('preflight_ok')}`",
            f"- API Budget OK: `{resume.get('api_budget_ok')}`",
            f"- BSC Gold Sanity OK: `{resume.get('bsc_gold_sanity_ok')}`",
            f"- Missing env: `{', '.join(resume.get('missing_env', [])) or 'none'}`",
            f"- Partial paid run allowed: `{str(resume.get('partial_paid_run_allowed')).lower()}`",
            f"- Partial paid run reason: {resume.get('partial_paid_run_reason')}",
            "",
            "### Failed Local Providers",
            "",
        ]
    )
    failed_local = resume.get("failed_local_providers", [])
    if failed_local:
        for item in failed_local:
            lines.append(
                f"- `{item.get('name')}` `{item.get('base_url')}` "
                f"host=`{item.get('host')}` port=`{item.get('port')}` error=`{item.get('error')}`"
            )
    else:
        lines.append("- none")
    env_templates = resume.get("env_export_templates", [])
    if env_templates:
        lines.extend(["", "### Env Export Template", "", "```bash", *env_templates, "```"])
    next_command = resume.get("next_command")
    if next_command:
        lines.extend(["", "### Next Command", "", "```bash", next_command, "```"])
    lines.extend(["", "### Paid Stage Plan", ""])
    for item in resume.get("paid_stage_plan", []):
        lines.append(
            f"- `{item.get('stage')}` status=`{item.get('status')}` "
            f"requires_env=`{', '.join(item.get('requires_env', [])) or 'none'}` "
            f"missing_env=`{', '.join(item.get('missing_env', [])) or 'none'}`"
        )
    lines.extend(["", "## Commands", ""])
    for name, command in report["commands"].items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend(["## Blockers", ""])
    lines.extend([f"- {item}" for item in report["blockers"]] or ["- none"])
    return "\n".join(lines) + "\n"


def provider_markdown_rows(groups: Any) -> list[str]:
    rows = []
    if not isinstance(groups, list):
        return rows
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_path = group.get("path", "")
        providers = group.get("providers", []) or []
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            health = provider.get("local_health", {})
            rows.append(
                "| "
                + " | ".join(
                    [
                        f"`{group_path}`",
                        f"`{provider.get('name', '')}`",
                        f"`{provider.get('base_url', '')}`",
                        f"`{provider.get('api_key_env', '')}`",
                        f"`{provider.get('api_key_present')}`",
                        f"`{format_local_health(health)}`",
                    ]
                )
                + " |"
            )
    return rows


def format_local_health(health: Any) -> str:
    if not isinstance(health, dict) or not health.get("checked"):
        return "not_checked"
    status = "ok" if health.get("ok") else "failed"
    host = health.get("host", "")
    port = health.get("port", "")
    location = f" {host}:{port}" if host or port else ""
    return f"{status}{location}".strip()


if __name__ == "__main__":
    main()
