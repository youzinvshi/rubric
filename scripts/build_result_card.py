#!/usr/bin/env python3
"""Build a paper-facing Result Card from BlindSpot-RL evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFE_STATUSES = {"pass", "safe_to_claim", "ok"}
BLOCKED_STATUSES = {"blocked", "contradicted", "fail", "failed", "missing", "not_paper_eligible"}
DOWNSTREAM_NOT_ELIGIBLE_ERROR = (
    "downstream summary is not paper_claim_eligible; use API scorer with bound "
    "provider and budget report"
)


def default_claim_ladder() -> list[dict[str, Any]]:
    return [
        {
            "level": "motivation",
            "required_claim_ids": ["C1", "C6"],
            "evidence_required": "Frozen 100-example hard-gold diagnostic with C1 and threshold-robustness C6 gates.",
            "paper_sentence": "If C1/C6 pass, report systematic evaluation blind spots for the base evaluation-criteria policy.",
            "downgrade_rule": "Does not support any trained-method claim by itself.",
        },
        {
            "level": "metric-support",
            "required_claim_ids": ["C0", "C2", "C3"],
            "evidence_required": "Hard-gold BSC coverage change with stable redundancy and hallucination under the fixed protocol.",
            "paper_sentence": "If C0/C2/C3 pass, report a hard-gold BSC coverage change as metric evidence.",
            "downgrade_rule": "Without downstream support, report metric-only BSC evidence.",
        },
        {
            "level": "method-support",
            "required_claim_ids": ["C5", "C7", "C14"],
            "evidence_required": "SFT-only vs SFT+GRPO and reward-component ablations pass C5/C7/C14.",
            "paper_sentence": "If C5/C7/C14 pass, attribute the coverage change to the RLVR reward stage.",
            "downgrade_rule": "Without C14, report proxy-gold supervision evidence rather than RLVR evidence.",
        },
        {
            "level": "judge-utility support",
            "required_claim_ids": ["C0", "C4", "C9", "C10", "C12"],
            "evidence_required": "RewardBench, RewardBench-2, and JudgeBench API/model scorer rows pass C4/C9/C10.",
            "paper_sentence": "If C0/C4/C9/C10/C12 pass, report held-out downstream judge-utility support.",
            "downgrade_rule": "Without C12, write aggregate coverage change rather than dimension-level recovery; without C0, no trained-method row is paper-facing.",
        },
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    card = build_result_card(config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(card), encoding="utf-8")
    print(f"Result card status={card['claim_decision']['status']} report={args.output_json}")
    if args.strict and card["claim_decision"]["status"] == "blocked":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BlindSpot-RL Result Card.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Result Card config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Result Card config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Result Card config must be a JSON object: {path}")
    return data


def build_result_card(config: dict[str, Any]) -> dict[str, Any]:
    raw_audit = collect_raw_audit(config)
    manifests = collect_manifests(config.get("manifests", []))
    metrics = collect_metrics(config)
    confidence_intervals = collect_confidence_intervals(config.get("confidence_intervals", []))
    claim_evidence = collect_claim_evidence(config.get("evidence_matrix"))
    readiness = collect_json_section("readiness", config.get("readiness_report"))
    dashboard = collect_json_section("dashboard", config.get("dashboard"))
    dashboard_diagnostics = summarize_dashboard_diagnostics(dashboard)
    claim_decision = decide_claim(config, raw_audit, manifests, metrics, claim_evidence, readiness, dashboard)
    claim_ladder = build_claim_ladder_status(claim_evidence)
    return {
        "title": config.get("title", "BlindSpot-RL Result Card"),
        "experiment_id": config.get("experiment_id", "unknown"),
        "scope": config.get("scope", "unspecified"),
        "status": claim_decision["status"],
        "ok": claim_decision["status"] == "safe_to_claim",
        "raw_audit_gates": raw_audit,
        "pack_builder_manifest": manifests,
        "metrics": metrics,
        "confidence_intervals": confidence_intervals,
        "claim_evidence": claim_evidence,
        "readiness": readiness,
        "dashboard": dashboard,
        "dashboard_diagnostics": dashboard_diagnostics,
        "claim_ladder": claim_ladder,
        "claim_decision": claim_decision,
        "notes": config.get("notes", []),
    }


def collect_raw_audit(config: dict[str, Any]) -> list[dict[str, Any]]:
    gates = []
    seen: set[tuple[str, str]] = set()

    def add_gate(gate: dict[str, Any]) -> None:
        key = (str(gate.get("type", "")), str(gate.get("path", "")))
        if key in seen:
            return
        seen.add(key)
        gates.append(gate)

    for item in config.get("raw_audit_gates", []):
        add_gate(
            summarize_gate(
                item.get("name", "gate"),
                item.get("type", "generic"),
                item.get("path", ""),
                item.get("required_datasets", []),
            )
        )
    for item in config.get("gold_validations", []):
        add_gate(summarize_gate(item.get("name", "gold_validation"), "gold_validation", item.get("path", "")))
    for item in config.get("query_validations", []):
        add_gate(summarize_gate(item.get("name", "query_validation"), "query_validation", item.get("path", "")))
    for item in config.get("data_source_reports", []):
        add_gate(
            summarize_gate(
                item.get("name", "data_source_report"),
                "data_source_report",
                item.get("path", ""),
                item.get("required_datasets", []),
            )
        )
    return gates


def summarize_gate(
    name: str,
    gate_type: str,
    raw_path: str,
    required_datasets: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(raw_path) if raw_path else None
    if not path or not path.exists():
        return {"name": name, "type": gate_type, "path": raw_path, "status": "missing", "summary": "report missing"}
    if gate_type == "generic":
        size = path.stat().st_size
        blockers = [] if size > 0 else ["artifact is empty"]
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if not blockers else "blocked",
            "summary": f"artifact present, bytes={size}",
            "blockers": blockers,
            "warnings": [],
        }
    data = read_json(path)
    if is_load_error(data):
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "blocked",
            "summary": f"report is not readable: {data['_load_error']}",
            "blockers": [f"report is not readable: {data['_load_error']}"],
            "warnings": [],
        }
    if gate_type in {"gold_validation", "query_validation"}:
        status = "pass" if data.get("ok") else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": f"{data.get('n_records', 0)} records, {len(data.get('blockers', []))} blockers",
            "blockers": data.get("blockers", []),
            "warnings": data.get("warnings", []),
        }
    if gate_type == "data_source_report":
        source_summary = summarize_data_source_report(data, required_datasets or [])
        return {
            "name": name,
            "type": gate_type,
            "required_datasets": required_datasets or [],
            "path": str(path),
            **source_summary,
        }
    if gate_type == "data_source_local_config":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        status = "pass" if data.get("ok") and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": f"{len(blockers)} blockers, source_status={data.get('source_overall_status', 'unknown')}",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "api_budget":
        total = data.get("total", {})
        blockers = data.get("blockers", [])
        status = "pass" if data.get("ok", True) and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": (
                f"{total.get('calls', 0)} calls, "
                f"{total.get('total_tokens', 0)} tokens, "
                f"${float(total.get('estimated_cost_usd', 0.0) or 0.0):.4f}"
            ),
            "blockers": blockers,
            "warnings": [],
        }
    if gate_type == "contamination_audit":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        overlap_count = int(data.get("overlap_query_count", 0) or 0)
        removed_records = int(data.get("removed_records", 0) or 0)
        if overlap_count:
            blockers = blockers + [f"{overlap_count} overlapping holdout query(s) found"]
        summary_parts = [f"{len(blockers)} blockers"]
        artifact_status = data.get("artifact_status")
        if artifact_status:
            summary_parts.append(f"artifact_status={artifact_status}")
        overlap_status = data.get("overlap_status")
        if overlap_status:
            summary_parts.append(f"overlap_status={overlap_status}")
        if "overlap_query_count" in data:
            summary_parts.append(f"overlaps={overlap_count}")
        if "removed_records" in data:
            summary_parts.append(f"removed={removed_records}")
        status = "pass" if data.get("ok") and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": ", ".join(summary_parts),
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "latex_compile":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        status = "pass" if data.get("ok") and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": (
                f"pdf_bytes={data.get('pdf_bytes', 0)}, "
                f"pages={data.get('page_count', 0)}, "
                f"max_pages={data.get('max_pages', 0)}, "
                f"official_style_active={bool(data.get('official_style_active'))}, "
                f"bibliography_style_active={bool(data.get('bibliography_style_active'))}, "
                f"anonymous_author_declared={bool(data.get('anonymous_author_declared'))}"
            ),
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "minimal_api_handoff":
        blockers = data.get("blockers", [])
        commands = data.get("commands", {}) if isinstance(data.get("commands"), dict) else {}
        resume = data.get("resume_requirements", {}) if isinstance(data.get("resume_requirements"), dict) else {}
        status = "pass" if data.get("ok") and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": (
                f"status={data.get('status', 'unknown')}, "
                f"{len(blockers)} blockers, commands={len(commands)}, "
                f"missing_env={format_csv(resume.get('missing_env', []))}, "
                f"next={classify_handoff_next_command(resume)}"
            ),
            "blockers": blockers,
            "warnings": [],
        }
    if gate_type == "preflight":
        blockers = data.get("hard_blockers", [])
        warnings = data.get("warnings", [])
        status = "pass" if data.get("ok") and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": f"{len(blockers)} blockers, {len(warnings)} warnings",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "schema_contract":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        status = "pass" if data.get("ok") and not blockers else "blocked"
        selected = data.get("selected_target") or "none"
        compatible = sum(1 for item in data.get("targets", []) if item.get("ok"))
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": f"selected={selected}, compatible_targets={compatible}",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "audit":
        status = "pass" if data.get("ok") else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": f"{len(data.get('missing_files', []))} missing files",
            "blockers": data.get("missing_files", []),
            "warnings": data.get("warnings", []),
        }
    if gate_type == "manual_gate":
        blockers = data.get("blockers", [])
        status = "pass" if data.get("ok") and not blockers else "blocked"
        present = sum(1 for item in data.get("checks", []) if item.get("present"))
        total = len(data.get("checks", []))
        json_checks = data.get("json_checks", [])
        json_valid = sum(1 for item in json_checks if item.get("valid_json") and not item.get("missing_keys"))
        json_contains_checks = data.get("json_contains_checks", [])
        json_contains_valid = sum(1 for item in json_contains_checks if item.get("valid_json") and not item.get("missing_values"))
        json_equals_checks = data.get("json_equals_checks", [])
        json_equals_valid = sum(1 for item in json_equals_checks if item.get("valid_json") and item.get("matches"))
        json_sha256_checks = data.get("json_sha256_checks", [])
        json_sha256_valid = sum(1 for item in json_sha256_checks if item.get("valid_json") and item.get("matches"))
        summary_parts = [
            f"{present}/{total} required paths present",
            f"{json_valid}/{len(json_checks)} JSON contracts valid",
            f"{json_contains_valid}/{len(json_contains_checks)} JSON contains contracts valid",
        ]
        if json_equals_checks:
            summary_parts.append(f"{json_equals_valid}/{len(json_equals_checks)} JSON equals contracts valid")
        if json_sha256_checks:
            summary_parts.append(f"{json_sha256_valid}/{len(json_sha256_checks)} JSON SHA256 contracts valid")
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": "; ".join(summary_parts),
            "blockers": blockers,
            "warnings": [],
        }
    if gate_type == "validation":
        n_records = int(data.get("n_records", data.get("records", 0)) or 0)
        ok_records = int(data.get("ok_records", 0) or 0)
        failed_records = int(data.get("failed_records", data.get("n_failed", max(n_records - ok_records, 0))) or 0)
        blockers = []
        if failed_records:
            blockers.append(f"{failed_records} criteria output record(s) failed validation")
        if n_records == 0:
            blockers.append("no criteria output records validated")
        status = "pass" if data.get("ok") and not blockers else "blocked"
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": status,
            "summary": f"{ok_records}/{n_records} valid records",
            "blockers": blockers,
            "warnings": data.get("warnings", []),
        }
    status = normalize_status(data.get("status") or data.get("overall_status") or ("pass" if data.get("ok") else "warn"))
    return {"name": name, "type": gate_type, "path": str(path), "status": status, "summary": "loaded"}


def collect_manifests(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests = []
    for item in items:
        name = item.get("name", "manifest")
        raw_path = item.get("path", "")
        path = Path(raw_path) if raw_path else None
        if not path or not path.exists():
            manifests.append({"name": name, "path": raw_path, "status": "missing", "summary": "manifest missing"})
            continue
        data = read_json(path)
        if is_load_error(data):
            manifests.append(
                {
                    "name": name,
                    "path": str(path),
                    "status": "blocked",
                    "summary": f"manifest is not readable: {data['_load_error']}",
                    "required_files": 0,
                    "summaries": 0,
                    "blockers": [f"manifest is not readable: {data['_load_error']}"],
                }
            )
            continue
        blockers = data.get("blockers", []) if isinstance(data, dict) else []
        manifest_warnings = summarize_manifest_warnings(data)
        status = (
            "blocked"
            if blockers or (isinstance(data, dict) and data.get("ok") is False)
            else "warn"
            if manifest_warnings
            else "pass"
        )
        manifests.append(
            {
                "name": name,
                "path": str(path),
                "status": status,
                "summary": summarize_manifest(data),
                "required_files": len(data.get("required_files", [])) if isinstance(data, dict) else 0,
                "summaries": len(data.get("summaries", [])) if isinstance(data, dict) else 0,
                "blockers": blockers,
                "warnings": manifest_warnings,
            }
        )
    return manifests


def summarize_manifest(data: Any) -> str:
    if not isinstance(data, dict):
        return "manifest loaded"
    if is_rebuttal_pack_manifest(data):
        counts = data.get("defense_status_counts", {})
        concern_templates = data.get("concern_templates", {})
        return (
            f"{int(data.get('entry_count', 0) or 0)} rebuttal entries, "
            f"answer_ready={int(counts.get('answer_ready', 0) or 0)}, "
            f"needs_readiness={int(counts.get('needs_readiness', 0) or 0)}, "
            f"needs_evidence={int(counts.get('needs_evidence', 0) or 0)}, "
            f"cannot_claim={int(counts.get('cannot_claim', 0) or 0)}, "
            f"missing_claim_mapping={int(counts.get('missing_claim_mapping', 0) or 0)}, "
            f"readiness_ok={bool(data.get('readiness_ok'))}, "
            f"templates={int(concern_templates.get('count', 0) or 0)}"
        )
    if "stages" in data:
        return f"{len(data.get('stages', []))} pipeline stages"
    return f"{len(data.get('required_files', []))} required files, {len(data.get('summaries', []))} summaries"


def summarize_manifest_warnings(data: Any) -> list[str]:
    if not isinstance(data, dict) or not is_rebuttal_pack_manifest(data):
        return []
    warnings = []
    counts = data.get("defense_status_counts", {})
    concern_templates = data.get("concern_templates", {})
    entry_count = int(data.get("entry_count", 0) or 0)
    if data.get("schema_version") != 1:
        warnings.append("unsupported rebuttal manifest schema_version")
    if int(concern_templates.get("count", 0) or 0) != entry_count:
        warnings.append("rebuttal concern template count does not match entry_count")
    if not str(concern_templates.get("sha256", "")).strip():
        warnings.append("rebuttal concern template sha256 is missing")
    if not data.get("readiness_ok"):
        warnings.append("rebuttal pack was built while submission readiness was false")
    if int(counts.get("answer_ready", 0) or 0) == 0:
        warnings.append("rebuttal pack has no answer_ready entries")
    return warnings


def is_rebuttal_pack_manifest(data: dict[str, Any]) -> bool:
    return "defense_status_counts" in data and "concern_templates" in data and "entry_count" in data


def summarize_data_source_report(data: dict[str, Any], required_datasets: list[str]) -> dict[str, Any]:
    datasets = data.get("datasets", [])
    selected = datasets
    missing = []
    if required_datasets:
        by_name = {dataset.get("name"): dataset for dataset in datasets}
        selected = []
        for name in required_datasets:
            dataset = by_name.get(name)
            if dataset is None:
                missing.append(f"data source report is missing required dataset: {name}")
            else:
                selected.append(dataset)
    blockers = list(missing)
    warnings = []
    for dataset in selected:
        blockers.extend(dataset.get("blockers", []))
        warnings.extend(dataset.get("warnings", []))
    if not required_datasets:
        blockers = data.get("blockers", blockers)
        warnings = data.get("warnings", warnings)
        status = data.get("overall_status", "pass")
    else:
        status = "blocked" if blockers else "warn" if warnings else "pass"
    return {
        "status": status,
        "summary": f"{len(selected)} scoped datasets, {len(blockers)} blockers",
        "blockers": blockers,
        "warnings": warnings,
    }


def collect_metrics(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "bsc": [collect_bsc(item) for item in config.get("bsc_summaries", [])],
        "downstream": [collect_downstream(item) for item in config.get("downstream_summaries", [])],
    }


def collect_confidence_intervals(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for item in items:
        name = item.get("name", "confidence_interval")
        raw_path = item.get("path", "")
        path = Path(raw_path) if raw_path else None
        if not path or not path.exists():
            reports.append({"name": name, "path": raw_path, "status": "missing", "metrics": []})
            continue
        data = read_json(path)
        if is_load_error(data):
            reports.append(
                {
                    "name": name,
                    "path": str(path),
                    "status": "blocked",
                    "metrics": [],
                    "blockers": [f"confidence interval report is not readable: {data['_load_error']}"],
                }
            )
            continue
        reports.append(
            {
                "name": name,
                "path": str(path),
                "status": "pass",
                "n": data.get("n", 0),
                "confidence": data.get("confidence"),
                "metrics": data.get("metrics", []),
            }
        )
    return reports


def collect_bsc(item: dict[str, Any]) -> dict[str, Any]:
    row = load_metric(item)
    data = row.pop("data", {})
    row.update(
        {
            "n": data.get("n", 0),
            "coverage": data.get("mean_coverage"),
            "blind": data.get("mean_blind"),
            "redundancy": data.get("mean_redundancy"),
            "hallucination": data.get("mean_hallucination"),
            "reward": data.get("mean_reward"),
        }
    )
    return row


def collect_downstream(item: dict[str, Any]) -> dict[str, Any]:
    row = load_metric(item)
    data = row.pop("data", {})
    eligible = data.get("paper_claim_eligible") is True
    if row["status"] == "pass" and not eligible:
        row["status"] = "not_paper_eligible"
        row["blockers"] = [DOWNSTREAM_NOT_ELIGIBLE_ERROR]
    row.update(
        {
            "n": data.get("n", 0) if eligible else "",
            "accuracy": data.get("accuracy") if eligible else None,
            "tie_rate": data.get("tie_rate") if eligible else None,
            "mean_margin": data.get("mean_margin") if eligible else None,
            "scorer": data.get("scorer"),
            "paper_claim_eligible": data.get("paper_claim_eligible"),
        }
    )
    return row


def load_metric(item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("method") or item.get("name", "unknown")
    raw_path = item.get("path", "")
    path = Path(raw_path) if raw_path else None
    if not path or not path.exists():
        return {"method": name, "path": raw_path, "status": "missing", "data": {}}
    data = read_json(path)
    if is_load_error(data):
        return {
            "method": name,
            "path": str(path),
            "status": "blocked",
            "data": {},
            "blockers": [f"metric summary is not readable: {data['_load_error']}"],
        }
    return {"method": name, "path": str(path), "status": "pass", "data": data}


def collect_claim_evidence(raw_path: str | None) -> dict[str, Any]:
    if not raw_path:
        return {"status": "missing", "safe_to_claim": 0, "contradicted": 0, "missing_evidence": 0, "claims": []}
    path = Path(raw_path)
    if not path.exists():
        return {"status": "missing", "path": raw_path, "safe_to_claim": 0, "contradicted": 0, "missing_evidence": 0, "claims": []}
    claims = read_json(path)
    if is_load_error(claims):
        return {
            "status": "blocked",
            "path": str(path),
            "safe_to_claim": 0,
            "contradicted": 0,
            "missing_evidence": 0,
            "claims": [],
            "blockers": [f"evidence matrix is not readable: {claims['_load_error']}"],
        }
    if not isinstance(claims, list):
        claims = []
    counts = CounterLike()
    for claim in claims:
        counts.add(claim.get("status", "missing_evidence"))
    status = "blocked" if counts["contradicted"] else "warn" if counts["missing_evidence"] else "pass"
    return {
        "status": status,
        "path": str(path),
        "safe_to_claim": counts["safe_to_claim"],
        "contradicted": counts["contradicted"],
        "missing_evidence": counts["missing_evidence"],
        "claims": claims,
    }


def collect_json_section(name: str, raw_path: str | None) -> dict[str, Any]:
    if not raw_path:
        return {"name": name, "status": "omitted", "path": ""}
    path = Path(raw_path)
    if not path.exists():
        return {"name": name, "status": "missing", "path": raw_path}
    data = read_json(path)
    if is_load_error(data):
        return {
            "name": name,
            "status": "blocked",
            "path": str(path),
            "data": {},
            "blockers": [f"{name} report is not readable: {data['_load_error']}"],
        }
    if data.get("ok") is False:
        status = "blocked"
    else:
        status = normalize_status(data.get("overall_status") or data.get("status") or ("pass" if data.get("ok") else "warn"))
    return {"name": name, "status": status, "path": str(path), "data": data}


def summarize_dashboard_diagnostics(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    data = dashboard.get("data", {})
    if not isinstance(data, dict):
        return []
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        return []
    diagnostics = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_type = section.get("type", "")
        metrics = section.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        if section_type == "submission_gap_report":
            diagnostics.append(
                {
                    "name": section.get("name", "Submission Gap Report"),
                    "type": section_type,
                    "status": section.get("status", ""),
                    "summary": section.get("summary", ""),
                    "training_chain_steps": int(metrics.get("training_chain_steps", 0) or 0),
                    "execution_steps": int(metrics.get("execution_steps", 0) or 0),
                    "blocked_execution_steps": int(metrics.get("blocked_execution_steps", 0) or 0),
                    "missing_prerequisite_items": int(metrics.get("missing_prerequisite_items", 0) or 0),
                    "hard_blockers": int(metrics.get("hard_blockers", 0) or 0),
                    "claim_ladder_safe": None,
                    "claim_ladder_levels": None,
                    "claim_ladder_non_safe_levels": [],
                }
            )
        elif section_type == "rebuttal_manifest":
            diagnostics.append(
                {
                    "name": section.get("name", "Rebuttal Pack Manifest"),
                    "type": section_type,
                    "status": section.get("status", ""),
                    "summary": section.get("summary", ""),
                    "training_chain_steps": 0,
                    "execution_steps": 0,
                    "blocked_execution_steps": 0,
                    "missing_prerequisite_items": 0,
                    "hard_blockers": 0,
                    "claim_ladder_safe": int(metrics.get("claim_ladder_safe", 0) or 0),
                    "claim_ladder_levels": int(metrics.get("claim_ladder_levels", 0) or 0),
                    "claim_ladder_non_safe_levels": (
                        metrics.get("claim_ladder_non_safe_levels", [])
                        if isinstance(metrics.get("claim_ladder_non_safe_levels"), list)
                        else []
                    ),
                }
            )
    return diagnostics


def build_claim_ladder_status(claim_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_claim = {
        str(row.get("claim_id")): row
        for row in claim_evidence.get("claims", [])
        if isinstance(row, dict) and row.get("claim_id")
    }
    ladder = []
    for item in default_claim_ladder():
        row = dict(item)
        required = [str(claim_id) for claim_id in row.get("required_claim_ids", [])]
        non_safe = []
        blocked = []
        for claim_id in required:
            claim = rows_by_claim.get(claim_id)
            if claim is None:
                non_safe.append(f"{claim_id}: missing")
                continue
            status = str(claim.get("status", "missing_evidence"))
            if status != "safe_to_claim":
                non_safe.append(f"{claim_id}: {status}")
            if status in {"contradicted", "blocked", "fail", "failed"}:
                blocked.append(f"{claim_id}: {status}")
        if blocked:
            status = "blocked"
        elif non_safe:
            status = "missing_evidence"
        else:
            status = "safe_to_claim"
        row.update(
            {
                "status": status,
                "missing_or_non_safe_claims": non_safe,
            }
        )
        ladder.append(row)
    return ladder


def decide_claim(
    config: dict[str, Any],
    raw_audit: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    metrics: dict[str, list[dict[str, Any]]],
    claim_evidence: dict[str, Any],
    readiness: dict[str, Any],
    dashboard: dict[str, Any],
) -> dict[str, Any]:
    blockers = []
    warnings = []
    for gate in raw_audit:
        if gate["status"] in BLOCKED_STATUSES:
            gate_blockers = gate.get("blockers", [])
            detail = f": {gate.get('summary', '')}"
            if gate_blockers:
                detail = f"{detail}; blockers: {'; '.join(str(item) for item in gate_blockers)}"
            blockers.append(f"Raw audit gate `{gate['name']}` is {gate['status']}{detail}")
        elif gate["status"] != "pass":
            warnings.append(f"Raw audit gate `{gate['name']}` is {gate['status']}")
    for manifest in manifests:
        if manifest["status"] in BLOCKED_STATUSES:
            blockers.append(f"Manifest `{manifest['name']}` is {manifest['status']}: {manifest.get('summary', '')}")
        elif manifest["status"] != "pass":
            warnings.append(f"Manifest `{manifest['name']}` is {manifest['status']}")
    if any(row["status"] in {"missing", "blocked"} for row in metrics["bsc"]):
        blockers.append("At least one BSC summary is missing or blocked")
    if config.get("require_downstream", False) and any(
        row["status"] in BLOCKED_STATUSES for row in metrics["downstream"]
    ):
        blockers.append("At least one required downstream summary is missing, blocked, or not paper-eligible")
    if claim_evidence["status"] == "blocked":
        blockers.append("Evidence matrix has contradicted claims")
    elif claim_evidence["status"] in {"missing", "warn"}:
        warnings.append(f"Evidence matrix status is {claim_evidence['status']}")
    for section in [readiness, dashboard]:
        if section["status"] == "omitted":
            continue
        if section["status"] == "blocked":
            blockers.append(f"{section['name']} report is blocked")
        elif section["status"] in {"missing", "warn"}:
            warnings.append(f"{section['name']} report is {section['status']}")
    status = "blocked" if blockers else "deferred" if warnings else "safe_to_claim"
    claim_discipline = build_claim_discipline(status, config, metrics, claim_evidence, readiness)
    return {
        "status": status,
        "safe_claim": config.get("safe_claim", ""),
        "deferred_claim": config.get("deferred_claim", ""),
        "claim_discipline": claim_discipline,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_claim_discipline(
    status: str,
    config: dict[str, Any],
    metrics: dict[str, list[dict[str, Any]]],
    claim_evidence: dict[str, Any],
    readiness: dict[str, Any],
) -> list[str]:
    if status == "safe_to_claim":
        return [
            "Configured raw audit, evidence, readiness, and metric gates passed; report only the configured safe_claim."
        ]

    rows_by_claim = {
        str(row.get("claim_id")): row
        for row in claim_evidence.get("claims", [])
        if isinstance(row, dict) and row.get("claim_id")
    }
    readiness_blockers = readiness.get("data", {}).get("hard_blockers", []) if isinstance(readiness.get("data"), dict) else []
    readiness_discipline = (
        readiness.get("data", {}).get("claim_discipline", [])
        if isinstance(readiness.get("data"), dict)
        else []
    )
    discipline = [
        "Do not write empirical claims unless the relevant evidence matrix rows are safe_to_claim and readiness is ok."
    ]
    if isinstance(readiness_discipline, list):
        discipline.extend(str(item) for item in readiness_discipline if str(item).strip())

    if claim_evidence.get("status") in {"missing", "blocked", "warn"}:
        discipline.append(
            "Treat result-card metrics as diagnostic until the Evidence Matrix is present, readable, and every required paper-facing row is safe_to_claim."
        )
    if claim_is_not_safe(rows_by_claim, "C0") or any("claim C0" in str(item) for item in readiness_blockers):
        discipline.append(
            "Do not claim clean hard-gold/proxy/downstream isolation until C0 is safe_to_claim with query-disjoint holdouts and SHA-bound provenance."
        )
    if any(claim_is_not_safe(rows_by_claim, claim_id) for claim_id in ["C2", "C3", "C12", "C14"]):
        discipline.append(
            "Treat BSC coverage changes as metric evidence only; do not call them dimension-level recovery until C3, C12, and C14 pass."
        )
    if any(claim_is_not_safe(rows_by_claim, claim_id) for claim_id in ["C4", "C9", "C10"]) or (
        config.get("require_downstream", False)
        and any(row.get("status") in BLOCKED_STATUSES for row in metrics.get("downstream", []))
    ):
        discipline.append(
            "Do not claim downstream judge-utility support until C4/C9/C10 pass with API/model scorer outputs and paper_claim_eligible summaries."
        )
    if any(claim_is_not_safe(rows_by_claim, claim_id) for claim_id in ["C5", "C7"]):
        discipline.append(
            "Do not claim ablation support until C5/C7 pass; offline reward re-scoring remains diagnostic, not paper-facing ablation evidence."
        )
    if claim_is_not_safe(rows_by_claim, "C13"):
        discipline.append(
            "Treat semantic-space plots as illustrative until C13 verifies point-level provenance, nearest-gold audit fields, and SVG/PDF/CSV/JSON assets."
        )

    seen = set()
    unique = []
    for item in discipline:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def claim_is_not_safe(rows_by_claim: dict[str, dict[str, Any]], claim_id: str) -> bool:
    row = rows_by_claim.get(claim_id)
    return row is not None and row.get("status") != "safe_to_claim"


def normalize_status(status: Any) -> str:
    status = str(status).lower()
    if status in SAFE_STATUSES:
        return "pass"
    if status in BLOCKED_STATUSES:
        return "blocked"
    if status in {"warn", "warning", "deferred"}:
        return "warn"
    return status or "missing"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": f"{path}: not valid JSON at line {exc.lineno} column {exc.colno}"}
    except OSError as exc:
        return {"_load_error": f"{path}: {exc}"}


def is_load_error(data: Any) -> bool:
    return isinstance(data, dict) and bool(data.get("_load_error"))


class CounterLike:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def add(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def __getitem__(self, key: str) -> int:
        return self.counts.get(key, 0)


def to_markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# {card['title']}",
        "",
        f"- Experiment ID: `{card['experiment_id']}`",
        f"- Scope: `{card['scope']}`",
        f"- Claim decision: `{card['claim_decision']['status']}`",
        "",
        "## Raw Audit Gates",
        "",
        "| Gate | Type | Status | Summary |",
        "| --- | --- | --- | --- |",
    ]
    for gate in card["raw_audit_gates"]:
        lines.append(f"| {escape_md(gate['name'])} | `{gate['type']}` | `{gate['status']}` | {escape_md(gate.get('summary', ''))} |")
    lines.extend(["", "## Pack Builder Manifest", "", "| Manifest | Status | Summary |", "| --- | --- | --- |"])
    for manifest in card["pack_builder_manifest"]:
        lines.append(f"| {escape_md(manifest['name'])} | `{manifest['status']}` | {escape_md(manifest.get('summary', ''))} |")
    manifest_diagnostics = []
    for manifest in card["pack_builder_manifest"]:
        for item in manifest.get("blockers", []):
            manifest_diagnostics.append(f"{manifest['name']}: blocker: {item}")
        for item in manifest.get("warnings", []):
            manifest_diagnostics.append(f"{manifest['name']}: warning: {item}")
    if manifest_diagnostics:
        lines.extend(["", "### Manifest Diagnostics", ""])
        lines.extend(f"- {escape_md(item)}" for item in manifest_diagnostics)
    lines.extend(["", "## BSC Metrics", "", "| Method | Status | N | Cov | Blind | Red | Hall | Reward |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in card["metrics"]["bsc"]:
        lines.append(
            f"| {escape_md(row['method'])} | `{row['status']}` | {row.get('n', 0)} | {fmt(row.get('coverage'))} | {fmt(row.get('blind'))} | {fmt(row.get('redundancy'))} | {fmt(row.get('hallucination'))} | {fmt(row.get('reward'))} |"
        )
    lines.extend(["", "## Downstream Metrics", "", "| Method | Status | N | Accuracy | Tie | Margin | Scorer | Eligible |", "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in card["metrics"]["downstream"]:
        lines.append(
            f"| {escape_md(row['method'])} | `{row['status']}` | {row.get('n', 0)} | {fmt(row.get('accuracy'))} | {fmt(row.get('tie_rate'))} | {fmt(row.get('mean_margin'))} | {escape_md(row.get('scorer', ''))} | {escape_md(str(row.get('paper_claim_eligible', '')))} |"
        )
    lines.extend(
        [
            "",
            "## Dashboard Diagnostics",
            "",
            "| Section | Status | Execution | Training Chain | Prerequisites | Hard Blockers | Claim Ladder | Summary |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in card.get("dashboard_diagnostics", []):
        claim_ladder = "-"
        if row.get("claim_ladder_levels") is not None:
            claim_ladder = f"{row.get('claim_ladder_safe', 0)}/{row.get('claim_ladder_levels', 0)}"
            non_safe_levels = row.get("claim_ladder_non_safe_levels", [])
            if non_safe_levels:
                claim_ladder = f"{claim_ladder} safe; blocked: {', '.join(str(item) for item in non_safe_levels)}"
        lines.append(
            f"| {escape_md(row.get('name', ''))} | `{row.get('status', '')}` | "
            f"{row.get('blocked_execution_steps', 0)}/{row.get('execution_steps', 0)} | "
            f"{row.get('training_chain_steps', 0)} | "
            f"{row.get('missing_prerequisite_items', 0)} | "
            f"{row.get('hard_blockers', 0)} | {escape_md(claim_ladder)} | "
            f"{escape_md(row.get('summary', ''))} |"
        )
    if not card.get("dashboard_diagnostics"):
        lines.append("| none |  |  |  |  |  |  |  |")
    lines.extend(["", "## Confidence Intervals", ""])
    if card.get("confidence_intervals"):
        for report in card["confidence_intervals"]:
            lines.extend(
                [
                    f"### {escape_md(report['name'])}",
                    "",
                    f"- Status: `{report['status']}`",
                    f"- Rows: `{report.get('n', 0)}`",
                    f"- Confidence: `{report.get('confidence', '')}`",
                    "",
                    "| Metric | N | Mean | CI Lower | CI Upper | Status |",
                    "| --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for metric in report.get("metrics", []):
                lines.append(
                    f"| {escape_md(metric.get('metric', ''))} | {metric.get('n', 0)} | {fmt(metric.get('mean'))} | {fmt(metric.get('ci_lower'))} | {fmt(metric.get('ci_upper'))} | `{metric.get('status', '')}` |"
                )
            lines.append("")
    else:
        lines.append("- none")
    decision = card["claim_decision"]
    claim_ladder = card.get("claim_ladder") or default_claim_ladder()
    lines.extend(
        [
            "",
            "## Claim Ladder",
            "",
            "| Level | Status | Required Claims | Evidence Required | Paper Sentence | Downgrade Rule |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in claim_ladder:
        lines.append(
            f"| {escape_md(row.get('level', ''))} | "
            f"`{escape_md(row.get('status', ''))}` | "
            f"{escape_md(', '.join(str(item) for item in row.get('required_claim_ids', [])))} | "
            f"{escape_md(row.get('evidence_required', ''))} | "
            f"{escape_md(row.get('paper_sentence', ''))} | "
            f"{escape_md(row.get('downgrade_rule', ''))} |"
        )
        non_safe = row.get("missing_or_non_safe_claims", [])
        if non_safe:
            lines.append(
                f"| {escape_md(row.get('level', ''))} blockers |  |  | "
                f"{escape_md('; '.join(str(item) for item in non_safe))} |  |  |"
            )
    lines.extend(
        [
            "",
            "## Claim Decision",
            "",
            f"- Safe claim: {decision.get('safe_claim') or 'none'}",
            f"- Deferred claim: {decision.get('deferred_claim') or 'none'}",
            "",
            "### Claim Discipline",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in decision.get("claim_discipline", [])] or ["- none"])
    lines.extend(["", "### Blockers", ""])
    lines.extend([f"- {item}" for item in decision["blockers"]] or ["- none"])
    lines.extend(["", "### Warnings", ""])
    lines.extend([f"- {item}" for item in decision["warnings"]] or ["- none"])
    if card.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {item}" for item in card["notes"]])
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_csv(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return "none"
    return ",".join(str(item) for item in items)


def classify_handoff_next_command(resume: dict[str, Any]) -> str:
    if resume.get("paid_run_command") and resume.get("next_command") == resume.get("paid_run_command"):
        return "paid_range_run"
    if resume.get("offline_rerun_command") and resume.get("next_command") == resume.get("offline_rerun_command"):
        return "offline_rerun"
    if resume.get("next_command"):
        return "custom"
    return "missing"


if __name__ == "__main__":
    main()
