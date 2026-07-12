#!/usr/bin/env python3
"""Build a compact BlindSpot-RL run dashboard from existing reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STATUS_ORDER = {"pass": 0, "warn": 1, "missing": 2, "blocked": 3}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dashboard = build_dashboard(config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(dashboard), encoding="utf-8")
    print(f"Run dashboard status={dashboard['overall_status']} report={args.output_json}")
    if args.strict and dashboard["overall_status"] in {"blocked", "missing"}:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BlindSpot-RL run dashboard.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Run Dashboard config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Run Dashboard config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Run Dashboard config must be a JSON object: {path}")
    return data


def build_dashboard(config: dict[str, Any]) -> dict[str, Any]:
    sections = [summarize_section(section) for section in config.get("sections", [])]
    overall_status = worst_status([section["status"] for section in sections])
    blockers = collect_items(sections, "blockers")
    warnings = collect_items(sections, "warnings")
    next_actions = build_next_actions(sections, config)
    return {
        "title": config.get("title", "BlindSpot-RL Run Dashboard"),
        "objective": config.get("objective", ""),
        "overall_status": overall_status,
        "sections": sections,
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": next_actions,
    }


def summarize_section(section: dict[str, Any]) -> dict[str, Any]:
    path = Path(section["path"])
    section_type = section.get("type", "generic")
    label = section.get("name", path.name)
    required = bool(section.get("required", True))
    if not path.exists() or path.stat().st_size == 0:
        status = "missing" if required else "warn"
        return {
            "name": label,
            "type": section_type,
            "path": str(path),
            "status": status,
            "summary": "report is missing",
            "metrics": {},
            "blockers": [f"{label}: missing required report {path}"] if required else [],
            "warnings": [] if required else [f"{label}: optional report is missing"],
        }

    data = read_json(path)
    if is_load_error(data):
        return section_result(
            label,
            section_type,
            path,
            "blocked",
            f"report is not readable: {data['_load_error']}",
            {"read_error": data["_load_error"]},
            [f"{label}: report is not readable: {data['_load_error']}"],
            [],
        )
    if section_type == "audit":
        return summarize_audit(label, path, data)
    if section_type == "evidence":
        return summarize_evidence(label, path, data)
    if section_type == "readiness":
        return summarize_readiness(label, path, data)
    if section_type == "preflight":
        return summarize_preflight(label, path, data)
    if section_type == "api_budget":
        return summarize_api_budget(label, path, data)
    if section_type == "contamination_audit":
        return summarize_contamination_audit(label, path, data)
    if section_type == "latex_compile":
        return summarize_latex_compile(label, path, data)
    if section_type == "data_source_report":
        return summarize_data_source_report(label, path, data)
    if section_type == "schema_contract":
        return summarize_schema_contract(label, path, data)
    if section_type == "validation":
        return summarize_validation(label, path, data)
    if section_type == "gold_validation":
        return summarize_gold_validation(label, path, data)
    if section_type == "confidence_interval":
        return summarize_confidence_interval(label, path, data)
    if section_type == "sampling":
        return summarize_sampling(label, path, data)
    if section_type == "manual_gate":
        return summarize_manual_gate(label, path, data)
    if section_type == "rebuttal_manifest":
        return summarize_rebuttal_manifest(label, path, data)
    if section_type == "submission_gap_report":
        return summarize_submission_gap_report(label, path, data)
    return summarize_generic(label, path, data)


def summarize_audit(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    missing = list(data.get("missing_files", []))
    failed_summaries = [item for item in data.get("summary_checks", []) if not item.get("ok")]
    blockers = [f"{label}: missing artifact {item}" for item in missing]
    blockers.extend(f"{label}: failed summary check {item.get('name')}" for item in failed_summaries)
    status = "pass" if data.get("ok") else "blocked"
    return section_result(
        label,
        "audit",
        path,
        status,
        f"{len(data.get('present_files', []))} present, {len(missing)} missing",
        {"present_files": len(data.get("present_files", [])), "missing_files": len(missing)},
        blockers,
        [],
    )


def summarize_evidence(label: str, path: Path, data: Any) -> dict[str, Any]:
    rows = data if isinstance(data, list) else []
    counts = {"safe_to_claim": 0, "missing_evidence": 0, "contradicted": 0, "not_yet_supported": 0}
    for row in rows:
        status = str(row.get("status", "not_yet_supported"))
        counts[status] = counts.get(status, 0) + 1
    counts["total"] = len(rows)
    blockers = []
    warnings = []
    if counts["contradicted"]:
        blockers.append(f"{label}: {counts['contradicted']} contradicted claim(s)")
    if not rows:
        blockers.append(f"{label}: no evidence rows")
    if counts["missing_evidence"]:
        warnings.append(f"{label}: {counts['missing_evidence']} claim(s) still missing evidence")
    if counts["not_yet_supported"]:
        warnings.append(f"{label}: {counts['not_yet_supported']} claim(s) not yet supported")
    status = "blocked" if blockers else "warn" if warnings else "pass"
    summary = f"{counts['safe_to_claim']}/{counts['total']} safe claims"
    return section_result(label, "evidence", path, status, summary, counts, blockers, warnings)


def summarize_readiness(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("hard_blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    status = "pass" if data.get("ok") else "blocked"
    metrics = {
        "safe_to_claim": data.get("evidence", {}).get("safe_to_claim", 0),
        "total_claims": data.get("evidence", {}).get("total", 0),
    }
    return section_result(label, "readiness", path, status, f"ok={bool(data.get('ok'))}", metrics, blockers, warnings)


def summarize_preflight(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("hard_blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    status = "pass" if data.get("ok") else "blocked"
    metrics = {
        "inputs": len(data.get("inputs", [])),
        "provider_files": len(data.get("providers", [])),
        "hard_blockers": len(blockers),
        "warnings": len(warnings),
    }
    return section_result(label, "preflight", path, status, f"{len(blockers)} blockers, {len(warnings)} warnings", metrics, blockers, warnings)


def summarize_api_budget(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    total = data.get("total", {})
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    status = "pass" if data.get("ok", True) and not blockers else "blocked"
    metrics = {
        "queries": data.get("n_queries", 0),
        "providers": data.get("n_providers", 0),
        "calls": total.get("calls", 0),
        "total_tokens": total.get("total_tokens", 0),
        "estimated_cost_usd": total.get("estimated_cost_usd", 0.0),
        "blockers": len(blockers),
    }
    summary = (
        f"{metrics['calls']} calls, {metrics['total_tokens']} tokens, "
        f"${float(metrics['estimated_cost_usd']):.4f}"
    )
    return section_result(label, "api_budget", path, status, summary, metrics, blockers, [])


def summarize_contamination_audit(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    overlap_count = int(data.get("overlap_query_count", 0) or 0)
    removed_records = int(data.get("removed_records", 0) or 0)
    if overlap_count:
        blockers.append(f"{label}: {overlap_count} overlapping holdout query(s) found")
    artifact_status = data.get("artifact_status")
    overlap_status = data.get("overlap_status")
    if artifact_status and artifact_status != "complete":
        blockers.append(f"{label}: artifact_status={artifact_status}")
    if overlap_status and overlap_status != "clear":
        blockers.append(f"{label}: overlap_status={overlap_status}")
    status = "pass" if data.get("ok") and not blockers else "blocked"
    metrics = {
        "blockers": len(blockers),
        "overlap_query_count": overlap_count,
        "removed_records": removed_records,
        "holdout_unique_queries": data.get("holdout_unique_queries", 0),
        "training_unique_queries": data.get("training_unique_queries", 0),
    }
    summary_parts = [f"{len(blockers)} blockers"]
    if artifact_status:
        summary_parts.append(f"artifact_status={artifact_status}")
    if overlap_status:
        summary_parts.append(f"overlap_status={overlap_status}")
    if "overlap_query_count" in data:
        summary_parts.append(f"overlaps={overlap_count}")
    if "removed_records" in data:
        summary_parts.append(f"removed={removed_records}")
    return section_result(
        label,
        "contamination_audit",
        path,
        status,
        ", ".join(summary_parts),
        metrics,
        blockers,
        warnings,
    )


def summarize_latex_compile(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    metrics = {
        "pdf_bytes": data.get("pdf_bytes", 0),
        "page_count": data.get("page_count", 0),
        "max_pages": data.get("max_pages", 0),
        "official_style_active": bool(data.get("official_style_active")),
        "official_style_files_present": bool(data.get("official_style_files_present")),
        "submission_mode_declared": bool(data.get("submission_mode_declared")),
        "bibliography_style_active": bool(data.get("bibliography_style_active")),
        "anonymous_author_declared": bool(data.get("anonymous_author_declared")),
        "blockers": len(blockers),
    }
    status = "pass" if data.get("ok") and not blockers else "blocked"
    summary = (
        f"pdf_bytes={metrics['pdf_bytes']}, pages={metrics['page_count']}/{metrics['max_pages']}, "
        f"official_style_active={metrics['official_style_active']}, "
        f"bibliography_style_active={metrics['bibliography_style_active']}, "
        f"anonymous_author_declared={metrics['anonymous_author_declared']}"
    )
    return section_result(label, "latex_compile", path, status, summary, metrics, blockers, warnings)


def summarize_data_source_report(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    datasets = data.get("datasets", [])
    metrics = {
        "datasets": len(datasets),
        "blockers": len(blockers),
        "warnings": len(warnings),
    }
    status = data.get("overall_status", "pass")
    if status not in STATUS_ORDER:
        status = "blocked" if blockers else "warn" if warnings else "pass"
    summary = f"{metrics['datasets']} datasets, {metrics['blockers']} blockers, {metrics['warnings']} warnings"
    return section_result(label, "data_source_report", path, status, summary, metrics, blockers, warnings)


def summarize_schema_contract(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    metrics = {
        "records": data.get("n_records", 0),
        "targets": len(data.get("targets", [])),
        "compatible_targets": sum(1 for item in data.get("targets", []) if item.get("ok")),
    }
    selected = data.get("selected_target") or ""
    status = "pass" if data.get("ok") and not blockers else "blocked"
    summary = f"selected={selected or 'none'}, compatible={metrics['compatible_targets']}/{metrics['targets']}"
    return section_result(label, "schema_contract", path, status, summary, metrics, blockers, warnings)


def summarize_validation(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    records = data.get("records", data.get("n_records", 0))
    if "failed_records" in data:
        failed_records = data.get("failed_records", 0)
    elif "n_failed" in data:
        failed_records = data.get("n_failed", 0)
    else:
        failed_records = max(int(records or 0) - int(data.get("ok_records", 0) or 0), 0)
    metrics = {
        "records": records,
        "ok_records": data.get("ok_records", 0),
        "failed_records": failed_records,
        "ok_rate": data.get("ok_rate", 0.0),
    }
    blockers = []
    if metrics["failed_records"]:
        blockers.append(f"{label}: {metrics['failed_records']} criteria output record(s) failed validation")
    if records == 0:
        blockers.append(f"{label}: no criteria output records validated")
    status = "pass" if data.get("ok") and not blockers else "blocked"
    return section_result(label, "validation", path, status, f"{metrics['failed_records']} failed records", metrics, blockers, [])


def summarize_gold_validation(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    warnings = [f"{label}: {item}" for item in data.get("warnings", [])]
    metrics = {
        "records": data.get("n_records", 0),
        "min_records": data.get("min_records", 0),
        "min_rubrics_per_query": data.get("min_rubrics_per_query", 0),
        "blockers": len(blockers),
        "warnings": len(warnings),
    }
    status = "pass" if data.get("ok") else "blocked"
    summary = (
        f"{metrics['records']} records, min={metrics['min_records']}, "
        f"{metrics['blockers']} blockers"
    )
    return section_result(label, "gold_validation", path, status, summary, metrics, blockers, warnings)


def summarize_confidence_interval(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    metric_rows = data.get("metrics", [])
    missing = [item for item in metric_rows if item.get("status") != "pass"]
    warnings = [
        f"{label}: metric {item.get('metric', '<unknown>')} status={item.get('status', '<missing>')}"
        for item in missing
    ]
    if not metric_rows:
        warnings.append(f"{label}: no CI metrics reported")
    metrics = {
        "records": data.get("n", 0),
        "metric_count": len(metric_rows),
        "missing_metrics": len(missing),
        "confidence": data.get("confidence", ""),
        "n_boot": data.get("n_boot", 0),
    }
    status = "warn" if warnings else "pass"
    summary = (
        f"{metrics['metric_count']} metrics, n={metrics['records']}, "
        f"confidence={metrics['confidence']}"
    )
    return section_result(label, "confidence_interval", path, status, summary, metrics, [], warnings)


def summarize_sampling(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "input_records": data.get("input_records", 0),
        "sampled_records": data.get("sampled_records", 0),
        "seed": data.get("seed", ""),
    }
    return section_result(
        label,
        "sampling",
        path,
        "pass",
        f"{metrics['sampled_records']} sampled from {metrics['input_records']}",
        metrics,
        [],
        [],
    )


def summarize_manual_gate(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    blockers = [f"{label}: {item}" for item in data.get("blockers", [])]
    json_checks = data.get("json_checks", [])
    json_contains_checks = data.get("json_contains_checks", [])
    json_equals_checks = data.get("json_equals_checks", [])
    metrics = {
        "checks": len(data.get("checks", [])),
        "present": sum(1 for item in data.get("checks", []) if item.get("present")),
        "missing": sum(1 for item in data.get("checks", []) if not item.get("present")),
        "json_checks": len(json_checks),
        "json_valid": sum(1 for item in json_checks if item.get("valid_json") and not item.get("missing_keys")),
        "json_missing_keys": sum(len(item.get("missing_keys", [])) for item in json_checks),
        "json_contains_checks": len(json_contains_checks),
        "json_contains_valid": sum(1 for item in json_contains_checks if item.get("valid_json") and not item.get("missing_values")),
        "json_missing_values": sum(len(item.get("missing_values", [])) for item in json_contains_checks),
        "json_equals_checks": len(json_equals_checks),
        "json_equals_valid": sum(1 for item in json_equals_checks if item.get("valid_json") and item.get("matches")),
        "json_mismatches": sum(1 for item in json_equals_checks if item.get("valid_json") and not item.get("matches")),
    }
    status = "pass" if data.get("ok") and not blockers else "blocked"
    summary_parts = [
        f"{metrics['present']}/{metrics['checks']} required paths present",
        f"{metrics['json_valid']}/{metrics['json_checks']} JSON contracts valid",
        f"{metrics['json_contains_valid']}/{metrics['json_contains_checks']} JSON contains contracts valid",
    ]
    if json_equals_checks:
        summary_parts.append(f"{metrics['json_equals_valid']}/{metrics['json_equals_checks']} JSON equals contracts valid")
    summary = "; ".join(summary_parts)
    return section_result(label, "manual_gate", path, status, summary, metrics, blockers, [])


def count_prerequisite_items(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return sum(count_prerequisite_items(item) for item in value.values())
    return 1 if value else 0


def summarize_rebuttal_manifest(label: str, path: Path, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return section_result(
            label,
            "rebuttal_manifest",
            path,
            "blocked",
            "manifest is not a JSON object",
            {"json_type": type(data).__name__},
            [f"{label}: manifest is not a JSON object"],
            [],
        )

    counts = data.get("defense_status_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    entry_count = int(data.get("entry_count", 0) or 0)
    blockers = []
    warnings = []
    if data.get("schema_version") != 1:
        blockers.append(f"{label}: unsupported schema_version={data.get('schema_version')}")
    if entry_count <= 0:
        blockers.append(f"{label}: no reviewer concern entries")
    if counts and sum(int(value or 0) for value in counts.values()) != entry_count:
        blockers.append(f"{label}: defense_status_counts do not sum to entry_count")
    concern_templates = data.get("concern_templates", {})
    if not isinstance(concern_templates, dict):
        concern_templates = {}
        blockers.append(f"{label}: concern_templates record is missing or invalid")
    concern_count = int(concern_templates.get("count", 0) or 0)
    if concern_count != entry_count:
        blockers.append(f"{label}: concern template count does not match entry_count")
    if not str(concern_templates.get("sha256", "")).strip():
        blockers.append(f"{label}: concern template sha256 is missing")
    if not data.get("readiness_ok"):
        warnings.append(f"{label}: submission readiness was false when the pack was built")
    if int(counts.get("answer_ready", 0) or 0) == 0:
        warnings.append(f"{label}: no answer_ready reviewer concern entries")
    if int(counts.get("needs_readiness", 0) or 0):
        warnings.append(f"{label}: some reviewer concern entries still need submission readiness")

    input_checks = check_manifest_file_records(label, data.get("inputs", {}), "input")
    output_checks = check_manifest_file_records(label, data.get("outputs", {}), "output")
    blockers.extend(input_checks["blockers"])
    blockers.extend(output_checks["blockers"])
    claim_ladder_metrics = summarize_claim_ladder(data.get("claim_ladder", []))

    metrics = {
        "entry_count": entry_count,
        "answer_ready": int(counts.get("answer_ready", 0) or 0),
        "needs_readiness": int(counts.get("needs_readiness", 0) or 0),
        "needs_evidence": int(counts.get("needs_evidence", 0) or 0),
        "cannot_claim": int(counts.get("cannot_claim", 0) or 0),
        "missing_claim_mapping": int(counts.get("missing_claim_mapping", 0) or 0),
        "readiness_ok": bool(data.get("readiness_ok")),
        "concern_templates": concern_count,
        "matched_claim_ids": len(data.get("matched_claim_ids", [])) if isinstance(data.get("matched_claim_ids"), list) else 0,
        "input_records": input_checks["checked"],
        "output_records": output_checks["checked"],
        "sha_mismatches": input_checks["sha_mismatches"] + output_checks["sha_mismatches"],
        **claim_ladder_metrics,
    }
    status = "blocked" if blockers else "warn" if warnings else "pass"
    summary = (
        f"{entry_count} entries, answer_ready={metrics['answer_ready']}, "
        f"needs_readiness={metrics['needs_readiness']}, "
        f"needs_evidence={metrics['needs_evidence']}, "
        f"cannot_claim={metrics['cannot_claim']}, "
        f"missing_claim_mapping={metrics['missing_claim_mapping']}, "
        f"readiness_ok={metrics['readiness_ok']}, "
        f"claim_ladder_safe={metrics['claim_ladder_safe']}/{metrics['claim_ladder_levels']}"
    )
    return section_result(label, "rebuttal_manifest", path, status, summary, metrics, blockers, warnings)


def summarize_claim_ladder(raw_ladder: Any) -> dict[str, Any]:
    ladder = raw_ladder if isinstance(raw_ladder, list) else []
    levels = [row for row in ladder if isinstance(row, dict)]
    safe_levels = [row for row in levels if row.get("status") == "safe_to_claim"]
    missing_levels = [row for row in levels if row.get("status") == "missing_evidence"]
    blocked_levels = [row for row in levels if row.get("status") == "blocked"]
    non_safe_levels = [
        str(row.get("level", "<unknown>"))
        for row in levels
        if row.get("status") != "safe_to_claim"
    ]
    return {
        "claim_ladder_levels": len(levels),
        "claim_ladder_safe": len(safe_levels),
        "claim_ladder_missing": len(missing_levels),
        "claim_ladder_blocked": len(blocked_levels),
        "claim_ladder_non_safe_levels": non_safe_levels,
    }


def summarize_submission_gap_report(label: str, path: Path, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return section_result(
            label,
            "submission_gap_report",
            path,
            "blocked",
            "gap report is not a JSON object",
            {"json_type": type(data).__name__},
            [f"{label}: gap report is not a JSON object"],
            [],
        )

    phases = data.get("phases", [])
    if not isinstance(phases, list):
        phases = []
    blocked_phases = [
        phase for phase in phases
        if isinstance(phase, dict) and phase.get("status") == "blocked"
    ]
    warning_phases = [
        phase for phase in phases
        if isinstance(phase, dict) and phase.get("status") == "warn"
    ]
    execution_sequence = data.get("execution_sequence", [])
    if not isinstance(execution_sequence, list):
        execution_sequence = []
    missing_prerequisites = data.get("missing_prerequisites", {})
    if not isinstance(missing_prerequisites, dict):
        missing_prerequisites = {}
    missing_prerequisite_items = count_prerequisite_items(missing_prerequisites)
    operator_handoff = data.get("operator_handoff", {})
    if not isinstance(operator_handoff, dict):
        operator_handoff = {}
    training_data_chain = operator_handoff.get("training_data_chain", [])
    if not isinstance(training_data_chain, list):
        training_data_chain = []
    blocked_execution_steps = [
        step for step in execution_sequence
        if isinstance(step, dict) and (step.get("phase_status") == "blocked" or step.get("blocked_by_prior_phases"))
    ]
    blockers = [
        f"{label}: phase `{phase.get('name', phase.get('id', '<unknown>'))}` is blocked"
        for phase in blocked_phases
    ]
    warnings = [
        f"{label}: phase `{phase.get('name', phase.get('id', '<unknown>'))}` is warn"
        for phase in warning_phases
    ]
    if not phases:
        blockers.append(f"{label}: no gap-report phases")

    metrics = {
        "phase_count": len(phases),
        "blocked_phases": len(blocked_phases),
        "warning_phases": len(warning_phases),
        "readiness_ok": bool(data.get("readiness_ok")),
        "hard_blockers": int(data.get("hard_blocker_count", 0) or 0),
        "warnings": int(data.get("warning_count", 0) or 0),
        "execution_steps": len(execution_sequence),
        "blocked_execution_steps": len(blocked_execution_steps),
        "training_chain_steps": len(training_data_chain),
        "missing_prerequisite_categories": len(missing_prerequisites),
        "missing_prerequisite_items": missing_prerequisite_items,
    }
    status = "blocked" if blockers or not data.get("ok") else "warn" if warnings else "pass"
    summary = (
        f"{metrics['phase_count']} phases, blocked={metrics['blocked_phases']}, "
        f"readiness_ok={metrics['readiness_ok']}, hard_blockers={metrics['hard_blockers']}, "
        f"execution_steps={metrics['execution_steps']}, "
        f"training_chain_steps={metrics['training_chain_steps']}, "
        f"prereq_items={metrics['missing_prerequisite_items']}"
    )
    return section_result(label, "submission_gap_report", path, status, summary, metrics, blockers, warnings)


def check_manifest_file_records(label: str, records: Any, kind: str) -> dict[str, Any]:
    blockers = []
    checked = 0
    sha_mismatches = 0
    if not isinstance(records, dict):
        return {
            "checked": 0,
            "sha_mismatches": 0,
            "blockers": [f"{label}: manifest {kind} records are not a JSON object"],
        }
    for name, record in records.items():
        if not isinstance(record, dict):
            blockers.append(f"{label}: manifest {kind} `{name}` is not a JSON object")
            continue
        raw_path = str(record.get("path", "")).strip()
        expected_sha = str(record.get("sha256", "")).strip()
        if not raw_path:
            blockers.append(f"{label}: manifest {kind} `{name}` is missing a path")
            continue
        checked += 1
        target = Path(raw_path)
        if not target.exists() or target.stat().st_size == 0:
            blockers.append(f"{label}: manifest {kind} `{name}` path is missing or empty: {raw_path}")
            continue
        if not expected_sha:
            blockers.append(f"{label}: manifest {kind} `{name}` is missing sha256")
            continue
        actual_sha = file_sha256(target)
        if actual_sha != expected_sha:
            sha_mismatches += 1
            blockers.append(f"{label}: manifest {kind} `{name}` sha256 is stale for {raw_path}")
    return {"checked": checked, "sha_mismatches": sha_mismatches, "blockers": blockers}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_generic(label: str, path: Path, data: Any) -> dict[str, Any]:
    metrics = {"json_type": type(data).__name__}
    return section_result(label, "generic", path, "pass", "report present", metrics, [], [])


def section_result(
    label: str,
    section_type: str,
    path: Path,
    status: str,
    summary: str,
    metrics: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "name": label,
        "type": section_type,
        "path": str(path),
        "status": status,
        "summary": summary,
        "metrics": metrics,
        "blockers": blockers,
        "warnings": warnings,
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": f"{path}: not valid JSON at line {exc.lineno} column {exc.colno}"}
    except OSError as exc:
        return {"_load_error": f"{path}: {exc}"}


def is_load_error(data: Any) -> bool:
    return isinstance(data, dict) and bool(data.get("_load_error"))


def worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    return max(statuses, key=lambda item: STATUS_ORDER.get(item, 99))


def collect_items(sections: list[dict[str, Any]], key: str) -> list[str]:
    out: list[str] = []
    for section in sections:
        out.extend(section.get(key, []))
    return out


def build_next_actions(sections: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    actions = []
    for section in sections:
        if section["status"] == "missing":
            actions.append(f"Generate missing report for {section['name']}: {section['path']}")
        elif section["status"] == "blocked":
            actions.append(f"Resolve blockers in {section['name']} before upgrading paper claims.")
        elif section["status"] == "warn":
            actions.append(f"Review warnings in {section['name']} and keep claims defensive.")
    actions.extend(config.get("next_actions", []))
    return dedupe(actions)


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def to_markdown(dashboard: dict[str, Any]) -> str:
    lines = [
        f"# {dashboard['title']}",
        "",
        f"- Overall status: `{dashboard['overall_status']}`",
    ]
    if dashboard.get("objective"):
        lines.append(f"- Objective: {dashboard['objective']}")
    lines.extend(
        [
            f"- Blockers: `{len(dashboard['blockers'])}`",
            f"- Warnings: `{len(dashboard['warnings'])}`",
            "",
            "| Section | Type | Status | Summary | Path |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for section in dashboard["sections"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(section["name"]),
                    code(section["type"]),
                    code(section["status"]),
                    escape_md(section["summary"]),
                    code(section["path"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in dashboard["blockers"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in dashboard["warnings"]] or ["- none"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend([f"- {item}" for item in dashboard["next_actions"]] or ["- none"])
    return "\n".join(lines) + "\n"


def code(value: Any) -> str:
    return f"`{value}`"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
