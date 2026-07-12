#!/usr/bin/env python3
"""Build a submission-readiness report from audit, evidence, and paper files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.build_result_card import build_claim_ladder_status
    from scripts.check_paper_asset_index import semantic_space_contract_blockers, semantic_points_csv_contract_blockers
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from build_result_card import build_claim_ladder_status
    from check_paper_asset_index import semantic_space_contract_blockers, semantic_points_csv_contract_blockers


REQUIRED_PAPER_FILES = [
    "main.tex",
    "sections/abstract.tex",
    "sections/introduction.tex",
    "sections/blindspot_phenomenon.tex",
    "sections/related_work.tex",
    "sections/method.tex",
    "sections/experiments.tex",
    "sections/limitations.tex",
    "sections/conclusion.tex",
]

REQUIRED_SYNCED_TABLES = [
    "tables/main_table.tex",
    "tables/rl_stage_ablation_table.tex",
    "tables/downstream_utility_table.tex",
    "tables/ablation_table.tex",
    "tables/teacher_union_ablation_table.tex",
    "tables/verifier_filter_ablation_table.tex",
    "tables/dimension_transition_table.tex",
]

REQUIRED_SYNCED_FIGURES = [
    "figures/semantic_space.pdf",
    "figures/semantic_space.svg",
]

REQUIRED_SYNCED_DOCS = [
    "asset_index/real_run_dashboard.json",
    "asset_index/real_run_dashboard.md",
    "asset_index/evidence_matrix.json",
    "asset_index/evidence_matrix.csv",
    "asset_index/evidence_matrix.md",
    "asset_index/semantic_space_points.csv",
    "asset_index/semantic_space_summary.json",
    "asset_index/result_card.json",
    "asset_index/result_card.md",
    "asset_index/submission_gap_report.json",
    "asset_index/submission_gap_report.md",
    "asset_index/readiness_report.json",
    "asset_index/readiness_report.md",
    "asset_index/rebuttal_pack.json",
    "asset_index/rebuttal_pack.md",
    "asset_index/rebuttal_pack_manifest.json",
]

SYNCED_DOC_JSON_CONTRACTS = {
    "asset_index/real_run_dashboard.json": ["sections"],
    "asset_index/result_card.json": ["claim_ladder", "dashboard_diagnostics"],
    "asset_index/submission_gap_report.json": ["claim_ladder", "operator_handoff"],
    "asset_index/readiness_report.json": ["claim_ladder", "claim_discipline"],
    "asset_index/rebuttal_pack_manifest.json": ["claim_ladder", "defense_status_counts"],
}

SYNCED_DOC_TEXT_CONTRACTS = {
    "asset_index/real_run_dashboard.md": ["claim_ladder_safe="],
    "asset_index/result_card.md": ["## Claim Ladder", "## Dashboard Diagnostics"],
    "asset_index/submission_gap_report.md": [
        "## Claim Ladder Status",
        "## Execution Sequence",
        "- Summary:",
        "ready_to_run=",
        "- Commands:",
        "- Unlocks:",
        "- Missing prerequisites:",
        "### Paper Asset Warnings",
    ],
    "asset_index/readiness_report.md": ["## Claim Ladder Status", "## Claim Discipline"],
    "asset_index/rebuttal_pack.md": ["## Claim Ladder Status"],
}

REQUIRED_EVIDENCE_CLAIMS = [
    "C0",   # hard-gold contamination audit
    "C1",   # blind-spot motivation diagnostic
    "C2",   # main BSC coverage change
    "C3",   # redundancy/hallucination control
    "C4",   # RewardBench downstream utility
    "C5",   # single-teacher vs multi-teacher union
    "C6",   # BSC threshold robustness
    "C7",   # reward and verifier ablations
    "C9",   # JudgeBench downstream utility
    "C10",  # RewardBench-2 downstream utility
    "C12",  # dimension-transition recovery audit
    "C13",  # semantic-space visualization
    "C14",  # SFT-only vs SFT+GRPO
]

ALLOWED_EVIDENCE_STATUSES = {
    "safe_to_claim",
    "missing_evidence",
    "contradicted",
    "not_yet_supported",
}


def main() -> None:
    args = parse_args()
    report = build_report(
        audit_report=args.audit_report,
        evidence_matrix=args.evidence_matrix,
        paper_dir=args.paper_dir,
        raw_gate_specs=args.raw_gate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Submission readiness ok={report['ok']} report={args.output_json}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BlindSpot-RL submission readiness.")
    parser.add_argument("--audit-report", type=Path, help="audit_experiment.py JSON report.")
    parser.add_argument("--evidence-matrix", type=Path, help="evidence_matrix.json from build_evidence_matrix.py.")
    parser.add_argument("--paper-dir", default=Path("paper"), type=Path)
    parser.add_argument(
        "--raw-gate",
        action="append",
        default=[],
        help="Raw audit gate in the form name|type|path. Repeatable.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if readiness fails.")
    return parser.parse_args()


def build_report(
    audit_report: Path | None,
    evidence_matrix: Path | None,
    paper_dir: Path,
    raw_gate_specs: list[str] | None = None,
) -> dict[str, Any]:
    audit = read_json_if_exists(audit_report)
    evidence_raw = read_json_if_exists(evidence_matrix, default=[])
    evidence_load_error = (
        evidence_raw.get("_load_error") if isinstance(evidence_raw, dict) and evidence_raw.get("_load_error") else ""
    )
    evidence_rows = evidence_raw
    if not isinstance(evidence_rows, list):
        evidence_rows = []

    paper_checks = check_paper_files(paper_dir)
    evidence_summary = summarize_evidence(evidence_rows)
    claim_ladder = build_claim_ladder_status({"claims": evidence_rows})
    raw_gates = [summarize_raw_gate(parse_raw_gate_spec(spec)) for spec in (raw_gate_specs or [])]
    audit_ok = bool(audit.get("ok")) if audit else False
    hard_blockers = []
    warnings = []

    if not audit:
        hard_blockers.append("missing audit report")
    elif audit.get("_load_error"):
        hard_blockers.append(f"audit report is not readable: {audit['_load_error']}")
    elif not audit_ok:
        hard_blockers.append("audit report is not ok")

    if evidence_load_error:
        hard_blockers.append(f"evidence matrix is not readable: {evidence_load_error}")
    elif not evidence_rows:
        hard_blockers.append("missing evidence matrix")
    hard_blockers.extend(evidence_status_blockers(evidence_rows))
    hard_blockers.extend(evidence_required_field_blockers(evidence_rows))
    if evidence_summary["contradicted"] > 0:
        hard_blockers.append("one or more claims are contradicted")
    hard_blockers.extend(required_evidence_blockers(evidence_rows, REQUIRED_EVIDENCE_CLAIMS))
    if evidence_summary["safe_to_claim"] == 0:
        warnings.append("no claim is currently safe_to_claim")
    if evidence_summary["missing_evidence"] > 0:
        warnings.append("some claims still have missing evidence")
    for gate in raw_gates:
        if gate["status"] in {"blocked", "missing"}:
            hard_blockers.append(f"raw gate {gate['name']} is {gate['status']}: {gate['summary']}")
        elif gate["status"] != "pass":
            warnings.append(f"raw gate {gate['name']} is {gate['status']}: {gate['summary']}")

    missing_required_paper = [item["path"] for item in paper_checks["required"] if not item["present"]]
    if missing_required_paper:
        hard_blockers.append("missing required paper files: " + ", ".join(missing_required_paper))
    missing_tables = [item["path"] for item in paper_checks["synced_tables"] if not item["present"]]
    if missing_tables:
        hard_blockers.append("required paper tables not synced: " + ", ".join(missing_tables))
    missing_figures = [item["path"] for item in paper_checks["synced_figures"] if not item["present"]]
    if missing_figures:
        hard_blockers.append("required paper figures not synced: " + ", ".join(missing_figures))
    missing_docs = [item["path"] for item in paper_checks["synced_docs"] if not item["present"]]
    if missing_docs:
        hard_blockers.append("required paper reviewer-facing docs not synced: " + ", ".join(missing_docs))
    blocked_docs = [item for item in paper_checks["synced_docs"] if item.get("status") == "blocked"]
    for item in blocked_docs:
        hard_blockers.append(f"required paper reviewer-facing doc is blocked: {item['path']}: {item.get('summary', '')}")
        hard_blockers.extend(item.get("blockers", []))
    for item in paper_checks["asset_index"]:
        if item.get("status") in {"blocked", "missing"}:
            hard_blockers.append(f"paper asset index is {item['status']}: {item.get('summary', item['path'])}")
            hard_blockers.extend(item.get("blockers", []))
        elif item.get("status") == "warn":
            warnings.append(f"paper asset index has warnings: {item.get('summary', item['path'])}")
            warnings.extend(item.get("warnings", []))

    ok = not hard_blockers
    status = "ready" if ok else "blocked"
    claim_discipline = build_claim_discipline(
        ok=ok,
        evidence_summary=evidence_summary,
        hard_blockers=hard_blockers,
        warnings=warnings,
    )
    hard_blocker_summary = summarize_hard_blockers(hard_blockers, paper_checks)
    return {
        "status": status,
        "ok": ok,
        "summary": {
            "status": status,
            "hard_blocker_count": len(hard_blockers),
            "hard_blocker_category_count": len(hard_blocker_summary),
            "warning_count": len(warnings),
            "safe_to_claim": evidence_summary.get("safe_to_claim", 0),
            "total_claims": evidence_summary.get("total", 0),
        },
        "audit_ok": audit_ok,
        "raw_gates": raw_gates,
        "evidence": evidence_summary,
        "required_evidence_claims": REQUIRED_EVIDENCE_CLAIMS,
        "claim_ladder": claim_ladder,
        "claim_discipline": claim_discipline,
        "paper": paper_checks,
        "blockers": hard_blockers,
        "hard_blockers": hard_blockers,
        "hard_blocker_summary": hard_blocker_summary,
        "warnings": warnings,
    }


def summarize_hard_blockers(hard_blockers: list[str], paper_checks: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable, human-readable blocker categories without hiding details."""
    categories: dict[str, dict[str, Any]] = {}

    def add(category: str, label: str, detail: str) -> None:
        entry = categories.setdefault(category, {"category": category, "label": label, "count": 0, "examples": []})
        entry["count"] += 1
        if detail and len(entry["examples"]) < 3:
            entry["examples"].append(detail)

    for item in hard_blockers:
        if item.startswith("required evidence claim "):
            add("evidence_claims", "required Evidence Matrix rows are not safe_to_claim", item)
        elif item.startswith("required paper tables not synced"):
            add("paper_tables", "paper-facing tables are not synced", item)
        elif item.startswith("required paper figures not synced"):
            add("paper_figures", "paper-facing figures are not synced", item)
        elif item.startswith("paper asset index is blocked"):
            add("paper_asset_index", "paper asset index is blocked", item)
        elif item.startswith("artifact is missing or empty"):
            add("paper_artifacts", "paper artifact files are missing or empty", item)
        elif item.startswith("raw gate "):
            add("raw_gates", "raw audit gates are blocked", item)
        elif item.startswith("audit report"):
            add("audit_report", "experiment audit report is not ready", item)
        elif "contradicted" in item:
            add("contradicted_claims", "one or more claims are contradicted", item)
        else:
            add("other", "other submission-readiness blockers", item)

    for asset_item in paper_checks.get("asset_index", []):
        for summary in asset_item.get("blocker_summary", []):
            category = f"asset_index:{summary.get('category', 'unknown')}"
            add(category, f"asset index: {summary.get('label', summary.get('category', 'unknown'))}", "")

    return sorted(categories.values(), key=lambda row: row["category"])


def build_claim_discipline(
    *,
    ok: bool,
    evidence_summary: dict[str, Any],
    hard_blockers: list[str],
    warnings: list[str],
) -> list[str]:
    if ok:
        return [
            "Submission-readiness gates passed; report only claims whose evidence rows are safe_to_claim and whose synced paper assets are present.",
        ]

    discipline = [
        "Do not treat this package as AAAI-ready while submission readiness is false.",
        "SFT+GRPO coverage, dimension-level recovery, downstream utility, ablation, and semantic-space claims are permitted only after all required C0-C14 evidence rows are safe_to_claim.",
    ]
    if any("C0" in item for item in hard_blockers):
        discipline.append(
            "Do not claim clean hard-gold/proxy/downstream isolation until C0 passes with query-disjoint holdouts and SHA-bound provenance."
        )
    if any(claim_id in " ".join(hard_blockers) for claim_id in ["C2", "C3", "C12", "C14"]):
        discipline.append(
            "Treat BSC changes as metric evidence only; do not describe them as dimension-level recovery until C3, C12, and C14 pass."
        )
    if any(claim_id in " ".join(hard_blockers) for claim_id in ["C4", "C9", "C10"]):
        discipline.append(
            "Do not claim downstream judge-utility support until C4/C9/C10 pass with API/model scorer outputs and paper_claim_eligible summaries."
        )
    if any(claim_id in " ".join(hard_blockers) for claim_id in ["C5", "C7"]):
        discipline.append(
            "Do not claim ablation support until C5/C7 pass with separately trained reward-component variants and verifier-filter evidence."
        )
    if any("C13" in item for item in hard_blockers):
        discipline.append(
            "Treat semantic-space plots as illustrative until C13 verifies SVG/PDF/CSV/JSON assets and point-level provenance."
        )
    if any("tables" in item or "figures" in item or "reviewer-facing docs" in item for item in hard_blockers):
        discipline.append(
            "Do not submit the paper package until required paper-facing tables, figures, and reviewer-facing docs are synced and indexed."
        )
    if evidence_summary.get("safe_to_claim", 0) == 0 or any("no claim is currently safe_to_claim" in item for item in warnings):
        discipline.append(
            "With zero safe_to_claim rows, restrict paper-facing empirical content to planned protocol and clearly marked diagnostic evidence."
        )

    seen = set()
    unique = []
    for item in discipline:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def parse_raw_gate_spec(spec: str) -> dict[str, str]:
    parts = spec.split("|", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError("raw gate must use the form name|type|path")
    return {"name": parts[0].strip(), "type": parts[1].strip(), "path": parts[2].strip()}


def summarize_raw_gate(spec: dict[str, str]) -> dict[str, Any]:
    name = spec["name"]
    gate_type, required_datasets = parse_gate_type(spec["type"])
    raw_path = spec["path"]
    path = Path(raw_path)
    if not path.exists() or path.stat().st_size == 0:
        return {
            "name": name,
            "type": gate_type,
            "path": raw_path,
            "status": "missing",
            "summary": "report missing",
            "blockers": [f"missing required report {raw_path}"],
            "warnings": [],
        }
    if gate_type == "generic":
        return {
            "name": name,
            "type": gate_type,
            "path": raw_path,
            "status": "pass",
            "summary": f"artifact present, bytes={path.stat().st_size}",
            "blockers": [],
            "warnings": [],
        }
    data = read_json_if_exists(path)
    if isinstance(data, dict) and data.get("_load_error"):
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "blocked",
            "summary": f"report is not readable: {data['_load_error']}",
            "blockers": [f"report is not readable: {data['_load_error']}"],
            "warnings": [],
        }
    if gate_type == "data_source_report":
        source_summary = summarize_data_source_report(data, required_datasets)
        return {
            "name": name,
            "type": gate_type,
            "required_datasets": required_datasets,
            "path": str(path),
            **source_summary,
        }
    if gate_type == "data_source_local_config":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": f"{len(blockers)} blockers, source_status={data.get('source_overall_status', 'unknown')}",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type in {"gold_validation", "query_validation"}:
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": f"{data.get('n_records', 0)} records, {len(blockers)} blockers",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "api_budget":
        blockers = data.get("blockers", [])
        total = data.get("total", {})
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": f"{total.get('calls', 0)} calls, {len(blockers)} blockers",
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
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": ", ".join(summary_parts),
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "audit":
        missing = data.get("missing_files", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not missing else "blocked",
            "summary": f"{len(missing)} missing files",
            "blockers": missing,
            "warnings": data.get("warnings", []),
        }
    if gate_type == "preflight":
        blockers = data.get("hard_blockers", [])
        warnings = data.get("warnings", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": f"{len(blockers)} blockers, {len(warnings)} warnings",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "latex_compile":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": (
                f"pdf_bytes={data.get('pdf_bytes', 0)}, "
                f"pages={data.get('page_count', 0)}, "
                f"max_pages={data.get('max_pages', 0)}, "
                f"official_style_active={bool(data.get('official_style_active'))}, "
                f"submission_mode_declared={bool(data.get('submission_mode_declared'))}, "
                f"bibliography_style_active={bool(data.get('bibliography_style_active'))}, "
                f"anonymous_author_declared={bool(data.get('anonymous_author_declared'))}, "
                f"{len(blockers)} blockers"
            ),
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type in {"manual_gate", "schema_contract", "validation"}:
        blockers = data.get("blockers", [])
        if gate_type == "validation" and data.get("failed_records", 0):
            blockers = blockers + [f"{data.get('failed_records')} failed validation records"]
        warnings = data.get("warnings", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": f"{len(blockers)} blockers, {len(warnings)} warnings",
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "bsc_gold_sanity":
        blockers = data.get("blockers", [])
        warnings = data.get("warnings", [])
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": (
                f"{data.get('n_joined', data.get('n', 0))} joined, "
                f"coverage={data.get('mean_coverage', 'missing')}, {len(blockers)} blockers"
            ),
            "blockers": blockers,
            "warnings": warnings,
        }
    if gate_type == "minimal_api_handoff":
        blockers = data.get("blockers", [])
        commands = data.get("commands", {}) if isinstance(data.get("commands"), dict) else {}
        resume = data.get("resume_requirements", {}) if isinstance(data.get("resume_requirements"), dict) else {}
        return {
            "name": name,
            "type": gate_type,
            "path": str(path),
            "status": "pass" if data.get("ok") and not blockers else "blocked",
            "summary": (
                f"status={data.get('status', 'unknown')}, "
                f"{len(blockers)} blockers, commands={len(commands)}, "
                f"missing_env={format_csv(resume.get('missing_env', []))}, "
                f"next={classify_handoff_next_command(resume)}"
            ),
            "blockers": blockers,
            "warnings": [],
        }
    status = normalize_gate_status(data.get("status") or data.get("overall_status") or ("pass" if data.get("ok") else "warn"))
    return {
        "name": name,
        "type": gate_type,
        "path": str(path),
        "status": status,
        "summary": "loaded",
        "blockers": [],
        "warnings": [],
    }


def normalize_gate_status(status: Any) -> str:
    text = str(status).lower()
    if text in {"pass", "ok", "safe_to_claim"}:
        return "pass"
    if text in {"blocked", "fail", "failed", "missing", "contradicted"}:
        return "blocked" if text != "missing" else "missing"
    if text in {"warn", "warning", "deferred"}:
        return "warn"
    return text or "missing"


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


def parse_gate_type(raw_type: str) -> tuple[str, list[str]]:
    if "[" not in raw_type:
        return raw_type, []
    base, rest = raw_type.split("[", 1)
    if not rest.endswith("]"):
        return raw_type, []
    datasets = [item.strip() for item in rest[:-1].split(",") if item.strip()]
    return base.strip(), datasets


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
        status = normalize_gate_status(data.get("overall_status", "pass"))
    else:
        status = "blocked" if blockers else "warn" if warnings else "pass"
    return {
        "status": status,
        "summary": f"{len(selected)} scoped datasets, {len(blockers)} blockers",
        "blockers": blockers,
        "warnings": warnings,
    }


def read_json_if_exists(path: Path | None, default: Any | None = None) -> Any:
    if not path or not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": f"{path}: not valid JSON at line {exc.lineno} column {exc.colno}"}
    except OSError as exc:
        return {"_load_error": f"{path}: {exc}"}


def summarize_evidence(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = {
        "safe_to_claim": 0,
        "missing_evidence": 0,
        "contradicted": 0,
        "not_yet_supported": 0,
    }
    for row in rows:
        status = str(row.get("status", "not_yet_supported")) if isinstance(row, dict) else "not_yet_supported"
        statuses[status] = statuses.get(status, 0) + 1
    statuses["total"] = len(rows)
    return statuses


def required_evidence_blockers(rows: list[dict[str, Any]], required_claim_ids: list[str]) -> list[str]:
    if not rows or not required_claim_ids:
        return []
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not any("claim_id" in row for row in dict_rows):
        return ["evidence matrix has no claim_id fields for required evidence checks"]
    by_id = {str(row.get("claim_id", "")): str(row.get("status", "not_yet_supported")) for row in dict_rows}
    blockers = []
    for claim_id in required_claim_ids:
        status = by_id.get(claim_id)
        if status is None:
            blockers.append(f"required evidence claim {claim_id} is missing from evidence matrix")
        elif status != "safe_to_claim":
            blockers.append(f"required evidence claim {claim_id} is {status}")
    return blockers


def evidence_status_blockers(rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            blockers.append(f"evidence matrix row {index} must be a JSON object")
            continue
        status = str(row.get("status", "not_yet_supported"))
        if status not in ALLOWED_EVIDENCE_STATUSES:
            claim_id = str(row.get("claim_id", f"row {index}"))
            blockers.append(f"evidence matrix claim {claim_id} has unsupported status `{status}`")
    return blockers


def evidence_required_field_blockers(rows: list[dict[str, Any]]) -> list[str]:
    required_fields = ("claim_id", "claim", "paper_section", "status", "evidence")
    blockers: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for key in required_fields:
            if row.get(key) in (None, "", [], {}):
                claim_id = str(row.get("claim_id", f"row {index}"))
                blockers.append(f"evidence matrix claim {claim_id} missing required field `{key}`")
    return blockers


def check_paper_files(paper_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "required": [file_check(paper_dir, path) for path in REQUIRED_PAPER_FILES],
        "synced_tables": [file_check(paper_dir, path) for path in REQUIRED_SYNCED_TABLES],
        "synced_figures": [file_check(paper_dir, path) for path in REQUIRED_SYNCED_FIGURES],
        "synced_docs": [file_check(paper_dir, path) for path in REQUIRED_SYNCED_DOCS],
        "asset_index": [asset_index_check(paper_dir, "asset_index.md")],
    }


def file_check(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    present = path.exists() and path.stat().st_size > 0
    item = {
        "path": relative_path,
        "present": present,
        "bytes": path.stat().st_size if path.exists() else 0,
        "status": "pass" if present else "missing",
    }
    if not present:
        return item
    blockers = synced_doc_contract_blockers(path, relative_path)
    if blockers:
        item.update(
            {
                "status": "blocked",
                "summary": f"{len(blockers)} synced-doc contract blocker(s)",
                "blockers": blockers,
            }
        )
    return item


def synced_doc_contract_blockers(path: Path, relative_path: str) -> list[str]:
    blockers: list[str] = []
    if relative_path == "asset_index/evidence_matrix.json":
        blockers.extend(evidence_matrix_json_contract_blockers(path, relative_path))
    if relative_path == "asset_index/evidence_matrix.csv":
        blockers.extend(evidence_matrix_csv_contract_blockers(path, relative_path))
    if relative_path == "asset_index/evidence_matrix.md":
        blockers.extend(evidence_matrix_markdown_contract_blockers(path, relative_path))
    if relative_path == "asset_index/submission_gap_report.json":
        blockers.extend(submission_gap_report_contract_blockers(path, relative_path))
    if relative_path == "asset_index/rebuttal_pack.json":
        blockers.extend(rebuttal_pack_contract_blockers(path, relative_path))
    if relative_path == "asset_index/semantic_space_summary.json":
        blockers.extend(
            semantic_space_contract_blockers(
                summary_path=path,
                points_path=path.with_name("semantic_space_points.csv"),
                svg_path=path.parent.parent / "figures" / "semantic_space.svg",
                pdf_path=path.parent.parent / "figures" / "semantic_space.pdf",
                paper_path=relative_path,
            )
        )
    if relative_path == "asset_index/semantic_space_points.csv":
        blockers.extend(semantic_points_csv_contract_blockers(path, relative_path))
    required_json_keys = SYNCED_DOC_JSON_CONTRACTS.get(relative_path)
    if required_json_keys:
        data = read_json_if_exists(path)
        if isinstance(data, dict) and data.get("_load_error"):
            return [f"{relative_path} is not valid JSON: {data['_load_error']}"]
        if not isinstance(data, dict):
            return [f"{relative_path} must be a JSON object"]
        for key in required_json_keys:
            value = data.get(key)
            if value in (None, "", [], {}):
                blockers.append(f"{relative_path} missing required JSON field `{key}`")
    required_text = SYNCED_DOC_TEXT_CONTRACTS.get(relative_path)
    if required_text:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return [f"{relative_path} is not readable: {exc}"]
        for needle in required_text:
            if needle not in text:
                blockers.append(f"{relative_path} missing required text `{needle}`")
    return blockers


def submission_gap_report_contract_blockers(path: Path, relative_path: str) -> list[str]:
    data = read_json_if_exists(path)
    if isinstance(data, dict) and data.get("_load_error"):
        return [f"{relative_path} is not valid JSON: {data['_load_error']}"]
    if not isinstance(data, dict):
        return [f"{relative_path} must be a JSON object"]

    blockers: list[str] = []
    phases = data.get("phases")
    if not isinstance(phases, list) or not phases:
        blockers.append(f"{relative_path} phases must be a non-empty list")
    else:
        for index, phase in enumerate(phases):
            if not isinstance(phase, dict):
                blockers.append(f"{relative_path} phase {index} must be a JSON object")
                continue
            if phase.get("summary") in (None, "", [], {}):
                blockers.append(f"{relative_path} phase {index} missing required field `summary`")
            if not isinstance(phase.get("paper_asset_warnings"), list):
                blockers.append(f"{relative_path} phase {index} missing required field `paper_asset_warnings` list")

    execution_sequence = data.get("execution_sequence")
    if not isinstance(execution_sequence, list) or not execution_sequence:
        blockers.append(f"{relative_path} execution_sequence must be a non-empty list")
    else:
        for index, step in enumerate(execution_sequence):
            if not isinstance(step, dict):
                blockers.append(f"{relative_path} execution step {index} must be a JSON object")
                continue
            summary = step.get("summary")
            if summary in (None, "", [], {}):
                blockers.append(f"{relative_path} execution step {index} missing required field `summary`")
            elif "ready_to_run=" not in str(summary):
                blockers.append(f"{relative_path} execution step {index} summary missing ready_to_run")
            for key in ("id", "phase_id"):
                if step.get(key) in (None, "", [], {}):
                    blockers.append(f"{relative_path} execution step {index} missing required field `{key}`")
            for key in ("commands", "unlocks", "evidence_gates"):
                if not isinstance(step.get(key), list) or not step.get(key):
                    blockers.append(
                        f"{relative_path} execution step {index} missing required non-empty field `{key}` list"
                    )
            if not isinstance(step.get("missing_prerequisites"), dict):
                blockers.append(f"{relative_path} execution step {index} missing required field `missing_prerequisites` object")
    return blockers


def evidence_matrix_json_contract_blockers(path: Path, relative_path: str) -> list[str]:
    data = read_json_if_exists(path)
    if isinstance(data, dict) and data.get("_load_error"):
        return [f"{relative_path} is not valid JSON: {data['_load_error']}"]
    if not isinstance(data, list):
        return [f"{relative_path} must be a JSON list"]
    if not data:
        return [f"{relative_path} has no evidence rows"]

    blockers: list[str] = []
    claim_ids: set[str] = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            blockers.append(f"{relative_path} row {index} must be a JSON object")
            continue
        claim_id = row.get("claim_id")
        if isinstance(claim_id, str) and claim_id:
            claim_ids.add(claim_id)
        for key in ("claim_id", "claim", "paper_section", "status", "evidence"):
            if row.get(key) in (None, "", [], {}):
                blockers.append(f"{relative_path} row {index} missing required field `{key}`")
        status = row.get("status")
        if isinstance(status, str) and status and status not in ALLOWED_EVIDENCE_STATUSES:
            blockers.append(f"{relative_path} row {index} has unsupported status `{status}`")
    blockers.extend(missing_required_claim_blockers(claim_ids, relative_path))
    return blockers


def evidence_matrix_csv_contract_blockers(path: Path, relative_path: str) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except (OSError, csv.Error) as exc:
        return [f"{relative_path} is not valid CSV: {exc}"]

    required_fields = {"claim_id", "claim", "paper_section", "status", "evidence"}
    blockers: list[str] = []
    missing_fields = sorted(required_fields - set(fieldnames))
    if missing_fields:
        return [f"{relative_path} missing CSV columns: {', '.join(missing_fields)}"]
    if not rows:
        return [f"{relative_path} has no evidence rows"]

    claim_ids: set[str] = set()
    for index, row in enumerate(rows):
        claim_id = row.get("claim_id", "")
        if claim_id:
            claim_ids.add(claim_id)
        for key in required_fields:
            if row.get(key) in (None, ""):
                blockers.append(f"{relative_path} row {index} missing required field `{key}`")
        status = row.get("status", "")
        if status and status not in ALLOWED_EVIDENCE_STATUSES:
            blockers.append(f"{relative_path} row {index} has unsupported status `{status}`")
    blockers.extend(missing_required_claim_blockers(claim_ids, relative_path))
    return blockers


def evidence_matrix_markdown_contract_blockers(path: Path, relative_path: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{relative_path} is not readable: {exc}"]

    blockers: list[str] = []
    if "| Claim ID | Section | Status | Claim | Evidence |" not in text:
        blockers.append(f"{relative_path} missing Evidence Matrix table header")
    for index, row in enumerate(markdown_evidence_rows(text)):
        status = row.get("status", "")
        if status and status not in ALLOWED_EVIDENCE_STATUSES:
            blockers.append(f"{relative_path} row {index} has unsupported status `{status}`")
    claim_ids = {claim_id for claim_id in REQUIRED_EVIDENCE_CLAIMS if f"| {claim_id} |" in text}
    blockers.extend(missing_required_claim_blockers(claim_ids, relative_path))
    return blockers


def markdown_evidence_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        cells = [cell.strip(" `") for cell in stripped.strip("|").split("|")]
        if len(cells) < 5 or cells[0] in {"Claim ID", ""}:
            continue
        rows.append(
            {
                "claim_id": cells[0],
                "paper_section": cells[1],
                "status": cells[2],
                "claim": cells[3],
                "evidence": cells[4],
            }
        )
    return rows


def missing_required_claim_blockers(claim_ids: set[str], relative_path: str) -> list[str]:
    missing_claims = [claim_id for claim_id in REQUIRED_EVIDENCE_CLAIMS if claim_id not in claim_ids]
    if not missing_claims:
        return []
    return [f"{relative_path} missing required claim ids: {', '.join(missing_claims)}"]


def rebuttal_pack_contract_blockers(path: Path, relative_path: str) -> list[str]:
    data = read_json_if_exists(path)
    if isinstance(data, dict) and data.get("_load_error"):
        return [f"{relative_path} is not valid JSON: {data['_load_error']}"]
    if not isinstance(data, list):
        return [f"{relative_path} must be a JSON list"]
    if not data:
        return [f"{relative_path} has no rebuttal entries"]

    blockers: list[str] = []
    matched_claim_ids: set[str] = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            blockers.append(f"{relative_path} row {index} must be a JSON object")
            continue
        for key in ("id", "topic", "question", "defense_status", "recommended_position", "matched_claims", "readiness_ok"):
            if row.get(key) in (None, "", [], {}):
                blockers.append(f"{relative_path} row {index} missing required field `{key}`")
        matched_claims = row.get("matched_claims")
        if not isinstance(matched_claims, list):
            continue
        for claim in matched_claims:
            if isinstance(claim, dict) and isinstance(claim.get("claim_id"), str):
                matched_claim_ids.add(claim["claim_id"])
    if not matched_claim_ids:
        blockers.append(f"{relative_path} has no matched claim ids")
    if "C0" not in matched_claim_ids:
        blockers.append(f"{relative_path} missing contamination concern claim C0")
    return blockers


def asset_index_check(root: Path, relative_path: str) -> dict[str, Any]:
    item = file_check(root, relative_path)
    if not item["present"]:
        item.update(
            {
                "status": "missing",
                "summary": "paper asset index is missing",
                "blockers": [f"missing paper asset index: {relative_path}"],
                "warnings": [],
            }
        )
        return item

    path = root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        item.update(
            {
                "status": "blocked",
                "summary": f"paper asset index is not readable: {exc}",
                "blockers": [f"paper asset index is not readable: {path}: {exc}"],
                "warnings": [],
            }
        )
        return item

    blockers = section_items(text, "## Blockers")
    warnings = section_items(text, "## Warnings")
    blocker_summary = parse_issue_summary(text, "## Blocker Summary")
    warning_summary = parse_issue_summary(text, "## Warning Summary")
    blocker_count = max(parse_count_line(text, "Blockers"), len(blockers))
    warning_count = max(parse_count_line(text, "Warnings"), len(warnings))
    status = "blocked" if blocker_count else "warn" if warning_count else "pass"
    summary = f"{blocker_count} blockers, {warning_count} warnings"
    if blocker_summary:
        summary += f", {len(blocker_summary)} blocker categories"
    if warning_summary:
        summary += f", {len(warning_summary)} warning categories"
    item.update(
        {
            "status": status,
            "summary": summary,
            "blockers": blockers,
            "blocker_summary": blocker_summary,
            "warnings": warnings,
            "warning_summary": warning_summary,
        }
    )
    return item


def parse_count_line(text: str, label: str) -> int:
    prefix = f"- {label}:"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        raw_value = line[len(prefix):].strip()
        try:
            return int(raw_value)
        except ValueError:
            return 0
    return 0


def section_items(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return []
    items = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        value = stripped[2:].strip()
        if value and value != "none":
            items.append(value)
    return items


def parse_blocker_summary(text: str) -> list[dict[str, Any]]:
    return parse_issue_summary(text, "## Blocker Summary")


def parse_issue_summary(text: str, heading: str) -> list[dict[str, Any]]:
    out = []
    for item in section_items(text, heading):
        if not item.startswith("`") or "`:" not in item:
            continue
        category, rest = item[1:].split("`:", 1)
        raw_count = rest.strip().split(" ", 1)[0]
        try:
            count = int(raw_count)
        except ValueError:
            count = 0
        label = ""
        if "(" in rest and rest.rstrip().endswith(")"):
            label = rest[rest.index("(") + 1:-1].strip()
        out.append({"category": category, "count": count, "label": label})
    return out


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BlindSpot-RL Submission Readiness",
        "",
        f"- Status: `{report.get('status', 'ready' if report.get('ok') else 'blocked')}`",
        f"- Overall ok: `{report['ok']}`",
        f"- Audit ok: `{report['audit_ok']}`",
        f"- Hard blockers: `{len(report.get('hard_blockers', report.get('blockers', [])))}`",
        f"- Hard blocker categories: `{len(report.get('hard_blocker_summary', []))}`",
        f"- Warnings: `{len(report.get('warnings', []))}`",
        f"- Safe claims: `{report['evidence'].get('safe_to_claim', 0)}` / `{report['evidence'].get('total', 0)}`",
        f"- Missing-evidence claims: `{report['evidence'].get('missing_evidence', 0)}`",
        f"- Contradicted claims: `{report['evidence'].get('contradicted', 0)}`",
        "",
    ]
    blocker_summary = report.get("hard_blocker_summary", [])
    if blocker_summary:
        lines.extend(["## Hard Blocker Summary", ""])
        for item in blocker_summary:
            examples = item.get("examples", [])
            example_text = f"; examples: {'; '.join(str(example) for example in examples)}" if examples else ""
            lines.append(
                f"- `{item.get('category', 'unknown')}`: {item.get('count', 0)} "
                f"({item.get('label', 'unknown')}){example_text}"
            )
        lines.append("")
    lines.extend(["## Hard Blockers", ""])
    blockers = report.get("hard_blockers", [])
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend([f"- {item}" for item in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Claim Ladder Status",
            "",
            "| Level | Status | Required Claims | Blocking Claims |",
            "| --- | --- | --- | --- |",
        ]
    )
    claim_ladder = report.get("claim_ladder", [])
    if claim_ladder:
        for row in claim_ladder:
            lines.append(
                f"| {escape_md(row.get('level', ''))} | `{row.get('status', '')}` | "
                f"{escape_md(', '.join(str(item) for item in row.get('required_claim_ids', [])))} | "
                f"{escape_md('; '.join(str(item) for item in row.get('missing_or_non_safe_claims', [])) or 'none')} |"
            )
    else:
        lines.append("| none |  |  |  |")
    lines.extend(["", "## Claim Discipline", ""])
    lines.extend([f"- {item}" for item in report.get("claim_discipline", [])] or ["- none"])
    lines.extend(["", "## Raw Gates", "", "| Gate | Type | Status | Summary |", "| --- | --- | --- | --- |"])
    for gate in report.get("raw_gates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(gate["name"]),
                    f"`{gate['type']}`",
                    f"`{gate['status']}`",
                    escape_md(gate.get("summary", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Paper Files", "", "| File | Present | Status | Bytes |", "| --- | --- | --- | --- |"])
    for group in ["required", "synced_tables", "synced_figures", "synced_docs", "asset_index"]:
        for item in report["paper"].get(group, []):
            lines.append(
                f"| `{item['path']}` | `{item['present']}` | `{item.get('status', '')}` | {item['bytes']} |"
            )
    lines.extend(["", "## Paper Asset Blocker Summary", ""])
    asset_summaries = []
    for item in report["paper"].get("asset_index", []):
        asset_summaries.extend(item.get("blocker_summary", []))
    if not asset_summaries:
        lines.append("- none")
    for item in asset_summaries:
        lines.append(f"- `{item['category']}`: {item['count']} ({item['label']})")
    return "\n".join(lines) + "\n"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
