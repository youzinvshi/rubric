#!/usr/bin/env python3
"""Verify paper asset index paths and SHA256 fingerprints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


INDEXED_JSON_CONTRACTS = {
    "paper/asset_index/real_run_dashboard.json": ["sections"],
    "paper/asset_index/result_card.json": ["claim_ladder", "dashboard_diagnostics"],
    "paper/asset_index/submission_gap_report.json": ["claim_ladder", "operator_handoff"],
    "paper/asset_index/readiness_report.json": ["claim_ladder", "claim_discipline"],
    "paper/asset_index/rebuttal_pack_manifest.json": ["claim_ladder", "defense_status_counts"],
}

INDEXED_TEXT_CONTRACTS = {
    "paper/asset_index/real_run_dashboard.md": ["claim_ladder_safe="],
    "paper/asset_index/result_card.md": ["## Claim Ladder", "## Dashboard Diagnostics"],
    "paper/asset_index/submission_gap_report.md": [
        "## Claim Ladder Status",
        "## Execution Sequence",
        "- Summary:",
        "ready_to_run=",
        "- Commands:",
        "- Unlocks:",
        "- Missing prerequisites:",
        "### Paper Asset Warnings",
    ],
    "paper/asset_index/readiness_report.md": ["## Claim Ladder Status", "## Claim Discipline"],
    "paper/asset_index/rebuttal_pack.md": ["## Claim Ladder Status"],
}

REQUIRED_EVIDENCE_CLAIM_IDS = {
    "C0",  # hard-gold/proxy/downstream isolation
    "C1",  # blind-spot motivation
    "C2",  # BSC coverage
    "C3",  # redundancy/hallucination controls
    "C4",  # RewardBench utility
    "C5",  # teacher-union ablation
    "C6",  # BSC threshold robustness
    "C7",  # reward/verifier ablations
    "C9",  # JudgeBench utility
    "C10",  # RewardBench-2 utility
    "C12",  # dimension-transition audit
    "C13",  # semantic-space visualization
    "C14",  # SFT-only vs SFT+GRPO
}

ALLOWED_EVIDENCE_STATUSES = {
    "safe_to_claim",
    "missing_evidence",
    "contradicted",
    "not_yet_supported",
}

SPECIAL_INDEXED_CONTRACTS = {
    "paper/asset_index/evidence_matrix.json",
    "paper/asset_index/evidence_matrix.csv",
    "paper/asset_index/evidence_matrix.md",
    "paper/asset_index/rebuttal_pack.json",
    "paper/asset_index/semantic_space_summary.json",
    "paper/asset_index/semantic_space_points.csv",
}

CANONICAL_BSC_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CANONICAL_BSC_COVERAGE_TAU = 0.75
SEMANTIC_REQUIRED_METHODS = {"base", "sft_only", "sft_rl"}
SEMANTIC_POINT_CSV_COLUMNS = [
    "point_id",
    "record_idx",
    "method",
    "source_type",
    "category",
    "gold_cluster_id",
    "rubric_idx",
    "x",
    "y",
    "nearest_gold_point_id",
    "nearest_gold_record_idx",
    "nearest_gold_rubric_idx",
    "nearest_gold_category",
    "nearest_gold_cluster_id",
    "nearest_gold_similarity",
    "nearest_gold_same_record",
    "query",
    "text",
    "nearest_gold_text",
]


def main() -> None:
    args = parse_args()
    report = build_report(args.asset_index, root=args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Paper asset index ok={report['ok']} report={args.output}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check paper asset index SHA256 fingerprints.")
    parser.add_argument("--asset-index", default=Path("paper/asset_index.md"), type=Path)
    parser.add_argument("--root", default=Path("."), type=Path)
    parser.add_argument("--output", default=Path("outputs/paper_asset_index_check.json"), type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_report(asset_index: Path, root: Path = Path(".")) -> dict[str, Any]:
    if not asset_index.exists():
        return {
            "ok": False,
            "asset_index": str(asset_index),
            "declared_count": None,
            "actual_count": 0,
            "checks": [],
            "blockers": [f"asset index is missing: {asset_index}"],
        }
    text = asset_index.read_text(encoding="utf-8")
    declared_count = parse_declared_count(text)
    declared_blocker_count = parse_status_count(text, "Blockers")
    declared_warning_count = parse_status_count(text, "Warnings")
    asset_index_blockers = parse_section_items(text, "Blockers")
    asset_index_warnings = parse_section_items(text, "Warnings")
    rows = parse_synced_artifact_rows(text)
    checks = [check_row(row, root=root) for row in rows]
    blockers: list[str] = []
    warnings: list[str] = []
    if declared_count is not None and declared_count != len(rows):
        blockers.append(f"synced artifact count mismatch: declared {declared_count}, found {len(rows)}")
    if declared_blocker_count is not None and declared_blocker_count != len(asset_index_blockers):
        blockers.append(
            f"asset index blocker count mismatch: declared {declared_blocker_count}, found {len(asset_index_blockers)}"
        )
    if declared_warning_count is not None and declared_warning_count != len(asset_index_warnings):
        warnings.append(
            f"asset index warning count mismatch: declared {declared_warning_count}, found {len(asset_index_warnings)}"
        )
    for item in asset_index_blockers:
        blockers.append(f"asset index declares blocker: {item}")
    for item in asset_index_warnings:
        warnings.append(f"asset index declares warning: {item}")
    if not rows:
        blockers.append("asset index has no synced artifact rows")
    for check in checks:
        blockers.extend(check["blockers"])
    blocker_summary = summarize_blockers(blockers)
    warning_summary = summarize_warnings(warnings)
    return {
        "ok": not blockers,
        "asset_index": str(asset_index),
        "declared_count": declared_count,
        "actual_count": len(rows),
        "declared_blocker_count": declared_blocker_count,
        "declared_warning_count": declared_warning_count,
        "blocker_summary": blocker_summary,
        "warning_summary": warning_summary,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
    }


def parse_declared_count(text: str) -> int | None:
    match = re.search(r"^- Synced artifacts:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_status_count(text: str, label: str) -> int | None:
    match = re.search(rf"^- {re.escape(label)}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_section_items(text: str, heading: str) -> list[str]:
    items: list[str] = []
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == f"## {heading}":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.startswith("- "):
            continue
        item = line[2:].strip()
        if item and item != "none":
            items.append(item)
    return items


def parse_synced_artifact_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "| Kind | Source | Paper Path | SHA256 |":
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4 or cells[0] == "---":
            continue
        if cells[0] == "none":
            continue
        rows.append(
            {
                "kind": cells[0],
                "source": strip_ticks(cells[1]),
                "paper_path": strip_ticks(cells[2]),
                "sha256": strip_ticks(cells[3]),
            }
        )
    return rows


def strip_ticks(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def check_row(row: dict[str, str], root: Path) -> dict[str, Any]:
    source = root / row["source"]
    paper_path = root / row["paper_path"]
    expected_sha = row["sha256"]
    blockers: list[str] = []
    source_present = source.exists() and source.is_file() and source.stat().st_size > 0
    paper_present = paper_path.exists() and paper_path.is_file() and paper_path.stat().st_size > 0
    actual_sha = file_sha256(paper_path) if paper_present else ""
    if not source_present:
        blockers.append(f"source artifact is missing or empty: {row['source']}")
    if not paper_present:
        blockers.append(f"paper artifact is missing or empty: {row['paper_path']}")
    if not expected_sha:
        blockers.append(f"missing SHA256 in asset index for {row['paper_path']}")
    elif paper_present and actual_sha != expected_sha:
        blockers.append(f"SHA256 mismatch for {row['paper_path']}: expected {expected_sha}, got {actual_sha}")
    if paper_present:
        blockers.extend(indexed_artifact_contract_blockers(paper_path, row["paper_path"]))
    return {
        **row,
        "source_present": source_present,
        "paper_present": paper_present,
        "actual_sha256": actual_sha,
        "sha256_matches": paper_present and bool(expected_sha) and actual_sha == expected_sha,
        "contract_checked": indexed_artifact_has_contract(row["paper_path"]),
        "blockers": blockers,
    }


def indexed_artifact_has_contract(paper_path: str) -> bool:
    return (
        paper_path in INDEXED_JSON_CONTRACTS
        or paper_path in INDEXED_TEXT_CONTRACTS
        or paper_path in SPECIAL_INDEXED_CONTRACTS
    )


def indexed_artifact_contract_blockers(path: Path, paper_path: str) -> list[str]:
    blockers: list[str] = []
    if paper_path == "paper/asset_index/evidence_matrix.json":
        blockers.extend(evidence_matrix_json_contract_blockers(path, paper_path))
    if paper_path == "paper/asset_index/evidence_matrix.csv":
        blockers.extend(evidence_matrix_csv_contract_blockers(path, paper_path))
    if paper_path == "paper/asset_index/evidence_matrix.md":
        blockers.extend(evidence_matrix_markdown_contract_blockers(path, paper_path))
    if paper_path == "paper/asset_index/submission_gap_report.json":
        blockers.extend(submission_gap_report_json_contract_blockers(path, paper_path))
    if paper_path == "paper/asset_index/rebuttal_pack.json":
        blockers.extend(rebuttal_pack_json_contract_blockers(path, paper_path))
    if paper_path == "paper/asset_index/semantic_space_summary.json":
        blockers.extend(
            semantic_space_contract_blockers(
                summary_path=path,
                points_path=path.with_name("semantic_space_points.csv"),
                svg_path=path.parent.parent / "figures" / "semantic_space.svg",
                pdf_path=path.parent.parent / "figures" / "semantic_space.pdf",
                paper_path=paper_path,
            )
        )
    if paper_path == "paper/asset_index/semantic_space_points.csv":
        blockers.extend(semantic_points_csv_contract_blockers(path, paper_path))
    required_json_keys = INDEXED_JSON_CONTRACTS.get(paper_path)
    if required_json_keys:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"indexed artifact contract failed for {paper_path}: invalid JSON: {exc}"]
        if not isinstance(data, dict):
            return [f"indexed artifact contract failed for {paper_path}: JSON root must be an object"]
        for key in required_json_keys:
            value = data.get(key)
            if value in (None, "", [], {}):
                blockers.append(f"indexed artifact contract failed for {paper_path}: missing JSON field `{key}`")
    required_text = INDEXED_TEXT_CONTRACTS.get(paper_path)
    if required_text:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return [f"indexed artifact contract failed for {paper_path}: not readable: {exc}"]
        for needle in required_text:
            if needle not in text:
                blockers.append(f"indexed artifact contract failed for {paper_path}: missing text `{needle}`")
    return blockers


def semantic_space_contract_blockers(
    *,
    summary_path: Path,
    points_path: Path,
    svg_path: Path,
    pdf_path: Path,
    paper_path: str,
) -> list[str]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"indexed artifact contract failed for {paper_path}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return [f"indexed artifact contract failed for {paper_path}: JSON root must be an object"]

    blockers: list[str] = []
    if data.get("embedding_model") != CANONICAL_BSC_EMBEDDING_MODEL:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: embedding_model must be "
            f"{CANONICAL_BSC_EMBEDDING_MODEL}"
        )
    if data.get("requested_projection") != "umap":
        blockers.append(f"indexed artifact contract failed for {paper_path}: requested_projection must be umap")
    if data.get("projection") not in {"umap", "umap_fallback_pca"}:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: projection must be umap or audited umap_fallback_pca"
        )
    if data.get("gold_cluster_tau") != CANONICAL_BSC_COVERAGE_TAU:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: gold_cluster_tau must be "
            f"{CANONICAL_BSC_COVERAGE_TAU}"
        )
    if data.get("point_csv_schema_version") != 3:
        blockers.append(f"indexed artifact contract failed for {paper_path}: point_csv_schema_version must be 3")
    if data.get("point_csv_columns") != SEMANTIC_POINT_CSV_COLUMNS:
        blockers.append(f"indexed artifact contract failed for {paper_path}: point_csv_columns do not match schema")
    if data.get("output_artifacts_schema_version") != 1:
        blockers.append(f"indexed artifact contract failed for {paper_path}: output_artifacts_schema_version must be 1")
    if data.get("point_csv_rows_match_n_points") is not True:
        blockers.append(f"indexed artifact contract failed for {paper_path}: point CSV row count must match n_points")

    methods = set(data.get("methods") or [])
    missing_methods = sorted(SEMANTIC_REQUIRED_METHODS - methods)
    if missing_methods:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: missing semantic methods "
            f"{', '.join(missing_methods)}"
        )
    for key in (
        "n_gold",
        "n_generated",
        "n_gold_clusters",
        "generated_gold_category_coverage_by_method",
        "nearest_gold_category_coverage_by_method",
        "nearest_gold_cluster_coverage_by_method",
        "nearest_gold_cluster_distribution_by_method",
        "nearest_gold_cluster_entropy_by_method",
        "generated_dispersion_by_method",
        "mean_nearest_gold_similarity_by_method",
    ):
        if data.get(key) in (None, "", [], {}):
            blockers.append(f"indexed artifact contract failed for {paper_path}: missing `{key}`")
    for key in (
        "generated_gold_category_coverage_by_method",
        "nearest_gold_category_coverage_by_method",
        "nearest_gold_cluster_coverage_by_method",
        "nearest_gold_cluster_distribution_by_method",
        "nearest_gold_cluster_entropy_by_method",
        "generated_dispersion_by_method",
        "mean_nearest_gold_similarity_by_method",
    ):
        blockers.extend(semantic_method_metric_blockers(data.get(key), key, paper_path))
    if parse_number(data.get("n_gold")) is not None and parse_number(data.get("n_gold")) <= 0:
        blockers.append(f"indexed artifact contract failed for {paper_path}: n_gold must be positive")
    if parse_number(data.get("n_generated")) is not None and parse_number(data.get("n_generated")) <= 0:
        blockers.append(f"indexed artifact contract failed for {paper_path}: n_generated must be positive")
    if parse_number(data.get("n_gold_clusters")) is not None and parse_number(data.get("n_gold_clusters")) <= 0:
        blockers.append(f"indexed artifact contract failed for {paper_path}: n_gold_clusters must be positive")

    inputs = data.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        blockers.append(f"indexed artifact contract failed for {paper_path}: inputs must be a non-empty list")
    else:
        input_labels = {item.get("label") for item in inputs if isinstance(item, dict)}
        missing_input_labels = sorted(SEMANTIC_REQUIRED_METHODS - input_labels)
        if missing_input_labels:
            blockers.append(
                f"indexed artifact contract failed for {paper_path}: missing input provenance for "
                f"{', '.join(missing_input_labels)}"
            )
        for item in inputs:
            if not isinstance(item, dict):
                blockers.append(f"indexed artifact contract failed for {paper_path}: input provenance row must be an object")
                continue
            label = str(item.get("label") or "<unknown>")
            join_report = item.get("join_report")
            if not isinstance(join_report, dict) or not join_report:
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: input {label} missing join_report provenance"
                )
                continue
            if join_report.get("join_key") != "query":
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: input {label} join_key must be query"
                )
            if join_report.get("gold") != "data/processed/splits/rubricbench_gold_test_main.jsonl":
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: input {label} must bind RubricBench test_main hard-gold"
                )
            if item.get("sha256") != join_report.get("output_sha256"):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: input {label} hash must match join_report output_sha256"
                )

    blockers.extend(check_summary_sha(data, "point_csv_sha256", points_path, paper_path))
    blockers.extend(check_summary_sha(data, "svg_sha256", svg_path, paper_path))
    blockers.extend(check_summary_sha(data, "pdf_sha256", pdf_path, paper_path))
    blockers.extend(semantic_points_csv_contract_blockers(points_path, "paper/asset_index/semantic_space_points.csv"))
    return blockers


def check_summary_sha(data: dict[str, Any], key: str, path: Path, paper_path: str) -> list[str]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return [f"indexed artifact contract failed for {paper_path}: missing artifact for `{key}`: {path}"]
    expected = data.get(key)
    if not expected:
        return [f"indexed artifact contract failed for {paper_path}: missing `{key}`"]
    actual = file_sha256(path)
    if actual != expected:
        return [
            f"indexed artifact contract failed for {paper_path}: `{key}` mismatch for {path}: "
            f"expected {expected}, got {actual}"
        ]
    return []


def semantic_points_csv_contract_blockers(path: Path, paper_path: str) -> list[str]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return [f"indexed artifact contract failed for {paper_path}: point CSV is missing or empty"]
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fieldnames = f.seek(0) or csv.DictReader(f).fieldnames
    except (OSError, csv.Error) as exc:
        return [f"indexed artifact contract failed for {paper_path}: invalid CSV: {exc}"]

    blockers: list[str] = []
    if fieldnames != SEMANTIC_POINT_CSV_COLUMNS:
        blockers.append(f"indexed artifact contract failed for {paper_path}: point CSV columns do not match schema")
        return blockers
    if not rows:
        return [f"indexed artifact contract failed for {paper_path}: point CSV has no rows"]
    if not any(row.get("source_type") == "gold" for row in rows):
        blockers.append(f"indexed artifact contract failed for {paper_path}: point CSV has no human-gold points")
    generated_rows = [row for row in rows if row.get("source_type") == "generated"]
    if not generated_rows:
        blockers.append(f"indexed artifact contract failed for {paper_path}: point CSV has no generated points")
    if not any(row.get("method") == "sft_rl" and row.get("source_type") == "generated" for row in rows):
        blockers.append(f"indexed artifact contract failed for {paper_path}: point CSV has no SFT+GRPO generated points")
    generated_methods = {str(row.get("method") or "") for row in generated_rows}
    missing_generated_methods = sorted(SEMANTIC_REQUIRED_METHODS - generated_methods)
    if missing_generated_methods:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: point CSV missing generated points for "
            f"{', '.join(missing_generated_methods)}"
        )
    for index, row in enumerate(generated_rows):
        for key in (
            "nearest_gold_point_id",
            "nearest_gold_category",
            "nearest_gold_cluster_id",
            "nearest_gold_similarity",
        ):
            if row.get(key) in (None, ""):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: generated row {index} missing `{key}`"
                )
    return blockers


def semantic_method_metric_blockers(value: Any, key: str, paper_path: str) -> list[str]:
    if not isinstance(value, dict) or not value:
        return []
    missing_methods = sorted(SEMANTIC_REQUIRED_METHODS - set(str(method) for method in value))
    if not missing_methods:
        return []
    return [
        f"indexed artifact contract failed for {paper_path}: `{key}` missing methods "
        f"{', '.join(missing_methods)}"
    ]


def parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def submission_gap_report_json_contract_blockers(path: Path, paper_path: str) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"indexed artifact contract failed for {paper_path}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return [f"indexed artifact contract failed for {paper_path}: JSON root must be an object"]

    blockers: list[str] = []
    phases = data.get("phases")
    if not isinstance(phases, list) or not phases:
        blockers.append(f"indexed artifact contract failed for {paper_path}: phases must be a non-empty list")
    else:
        for index, phase in enumerate(phases):
            if not isinstance(phase, dict):
                blockers.append(f"indexed artifact contract failed for {paper_path}: phase {index} must be an object")
                continue
            if phase.get("summary") in (None, "", [], {}):
                blockers.append(f"indexed artifact contract failed for {paper_path}: phase {index} missing `summary`")
            if not isinstance(phase.get("paper_asset_warnings"), list):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: phase {index} missing `paper_asset_warnings` list"
                )

    execution_sequence = data.get("execution_sequence")
    if not isinstance(execution_sequence, list) or not execution_sequence:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: execution_sequence must be a non-empty list"
        )
    else:
        for index, step in enumerate(execution_sequence):
            if not isinstance(step, dict):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: execution step {index} must be an object"
                )
                continue
            summary = step.get("summary")
            if summary in (None, "", [], {}):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: execution step {index} missing `summary`"
                )
            elif "ready_to_run=" not in str(summary):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: execution step {index} summary missing ready_to_run"
                )
            for key in ("id", "phase_id"):
                if step.get(key) in (None, "", [], {}):
                    blockers.append(
                        f"indexed artifact contract failed for {paper_path}: execution step {index} missing `{key}`"
                    )
            for key in ("commands", "unlocks", "evidence_gates"):
                if not isinstance(step.get(key), list) or not step.get(key):
                    blockers.append(
                        f"indexed artifact contract failed for {paper_path}: execution step {index} missing non-empty `{key}` list"
                    )
            if not isinstance(step.get("missing_prerequisites"), dict):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: execution step {index} missing `missing_prerequisites` object"
                )
    return blockers


def rebuttal_pack_json_contract_blockers(path: Path, paper_path: str) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"indexed artifact contract failed for {paper_path}: invalid JSON: {exc}"]
    if not isinstance(data, list):
        return [f"indexed artifact contract failed for {paper_path}: JSON root must be a list"]
    if not data:
        return [f"indexed artifact contract failed for {paper_path}: rebuttal pack has no entries"]

    blockers: list[str] = []
    matched_claim_ids: set[str] = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            blockers.append(f"indexed artifact contract failed for {paper_path}: row {index} must be an object")
            continue
        for key in ("id", "topic", "question", "defense_status", "recommended_position", "matched_claims", "readiness_ok"):
            if row.get(key) in (None, "", [], {}):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: row {index} missing `{key}`"
                )
        matched_claims = row.get("matched_claims")
        if not isinstance(matched_claims, list):
            continue
        for claim in matched_claims:
            if isinstance(claim, dict) and isinstance(claim.get("claim_id"), str):
                matched_claim_ids.add(claim["claim_id"])

    if not matched_claim_ids:
        blockers.append(f"indexed artifact contract failed for {paper_path}: no matched claim ids")
    if "C0" not in matched_claim_ids:
        blockers.append(f"indexed artifact contract failed for {paper_path}: missing contamination concern claim C0")
    return blockers


def evidence_matrix_json_contract_blockers(path: Path, paper_path: str) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"indexed artifact contract failed for {paper_path}: invalid JSON: {exc}"]
    if not isinstance(data, list):
        return [f"indexed artifact contract failed for {paper_path}: JSON root must be a list"]
    if not data:
        return [f"indexed artifact contract failed for {paper_path}: evidence matrix has no rows"]

    blockers: list[str] = []
    claim_ids: set[str] = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            blockers.append(f"indexed artifact contract failed for {paper_path}: row {index} must be an object")
            continue
        claim_id = row.get("claim_id")
        if isinstance(claim_id, str) and claim_id:
            claim_ids.add(claim_id)
        for key in ("claim_id", "claim", "paper_section", "status", "evidence"):
            if row.get(key) in (None, "", [], {}):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: row {index} missing `{key}`"
                )
        status = row.get("status")
        if isinstance(status, str) and status and status not in ALLOWED_EVIDENCE_STATUSES:
            blockers.append(
                f"indexed artifact contract failed for {paper_path}: row {index} has unsupported status `{status}`"
            )

    missing_claims = sorted(REQUIRED_EVIDENCE_CLAIM_IDS - claim_ids, key=claim_sort_key)
    if missing_claims:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: missing required claim ids "
            f"{', '.join(missing_claims)}"
        )
    return blockers


def evidence_matrix_markdown_contract_blockers(path: Path, paper_path: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"indexed artifact contract failed for {paper_path}: not readable: {exc}"]

    blockers: list[str] = []
    for needle in ("| Claim ID | Section | Status | Claim | Evidence |",):
        if needle not in text:
            blockers.append(f"indexed artifact contract failed for {paper_path}: missing text `{needle}`")
    for index, row in enumerate(markdown_evidence_rows(text)):
        status = row.get("status", "")
        if status and status not in ALLOWED_EVIDENCE_STATUSES:
            blockers.append(
                f"indexed artifact contract failed for {paper_path}: row {index} has unsupported status `{status}`"
            )
    missing_claims = [claim_id for claim_id in sorted(REQUIRED_EVIDENCE_CLAIM_IDS, key=claim_sort_key) if f"| {claim_id} |" not in text]
    if missing_claims:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: missing required claim ids "
            f"{', '.join(missing_claims)}"
        )
    return blockers


def evidence_matrix_csv_contract_blockers(path: Path, paper_path: str) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except (OSError, csv.Error) as exc:
        return [f"indexed artifact contract failed for {paper_path}: invalid CSV: {exc}"]

    required_fields = {"claim_id", "claim", "paper_section", "status", "evidence"}
    blockers: list[str] = []
    missing_fields = sorted(required_fields - set(fieldnames))
    if missing_fields:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: missing CSV columns "
            f"{', '.join(missing_fields)}"
        )
        return blockers
    if not rows:
        return [f"indexed artifact contract failed for {paper_path}: evidence matrix has no rows"]

    claim_ids: set[str] = set()
    for index, row in enumerate(rows):
        claim_id = row.get("claim_id", "")
        if claim_id:
            claim_ids.add(claim_id)
        for key in required_fields:
            if row.get(key) in (None, ""):
                blockers.append(
                    f"indexed artifact contract failed for {paper_path}: row {index} missing `{key}`"
                )
        status = row.get("status", "")
        if status and status not in ALLOWED_EVIDENCE_STATUSES:
            blockers.append(
                f"indexed artifact contract failed for {paper_path}: row {index} has unsupported status `{status}`"
            )

    missing_claims = sorted(REQUIRED_EVIDENCE_CLAIM_IDS - claim_ids, key=claim_sort_key)
    if missing_claims:
        blockers.append(
            f"indexed artifact contract failed for {paper_path}: missing required claim ids "
            f"{', '.join(missing_claims)}"
        )
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


def claim_sort_key(claim_id: str) -> tuple[int, str]:
    if len(claim_id) > 1 and claim_id[0] == "C" and claim_id[1:].isdigit():
        return (int(claim_id[1:]), "")
    return (10_000, claim_id)


BLOCKER_CATEGORY_RULES = [
    ("main_bsc_table", "main hard-gold BSC paper table", ["main_table.tex"]),
    ("rl_stage_ablation", "SFT-only vs SFT+GRPO paper table", ["rl_stage_ablation_table.tex"]),
    ("downstream_utility", "RewardBench/JudgeBench/RewardBench-2 utility table", ["downstream_utility_table.tex"]),
    ("reward_ablation", "reward-component ablation table", ["ablation_table.tex"]),
    ("teacher_union_ablation", "teacher-union ablation table", ["teacher_union_ablation_table.tex"]),
    ("verifier_filter_ablation", "verifier-filter ablation table", ["verifier_filter_ablation_table.tex"]),
    ("dimension_transition", "dimension-transition audit paper table", ["dimension_transition_table.tex"]),
    (
        "semantic_space",
        "semantic-space SVG/PDF/CSV/JSON assets",
        ["semantic_space.svg", "semantic_space.pdf", "semantic_space_points.csv", "semantic_space_summary.json"],
    ),
    ("experiment_summary", "paper-facing experiment summary", ["experiment_summary.md"]),
    ("api_handoff", "API handoff reviewer-facing docs", ["api_handoff.json", "api_handoff.md"]),
    ("audit_report", "matrix audit report", ["audit_report.json"]),
]


def summarize_blockers(blockers: list[str]) -> list[dict[str, Any]]:
    return summarize_asset_issues(blockers, issue_key="blockers", uncategorized_label="uncategorized asset-index blockers")


def summarize_warnings(warnings: list[str]) -> list[dict[str, Any]]:
    return summarize_asset_issues(warnings, issue_key="warnings", uncategorized_label="uncategorized asset-index warnings")


def summarize_asset_issues(
    issues: list[str],
    issue_key: str,
    uncategorized_label: str,
) -> list[dict[str, Any]]:
    summaries = []
    for category, label, needles in BLOCKER_CATEGORY_RULES:
        matched = [item for item in issues if blocker_matches_any_artifact(item, needles)]
        if not matched:
            continue
        summaries.append(
            {
                "category": category,
                "label": label,
                "count": len(matched),
                issue_key: matched,
            }
        )
    uncategorized = [
        item
        for item in issues
        if not any(blocker_matches_any_artifact(item, needles) for _, _, needles in BLOCKER_CATEGORY_RULES)
    ]
    if uncategorized:
        summaries.append(
            {
                "category": "uncategorized",
                "label": uncategorized_label,
                "count": len(uncategorized),
                issue_key: uncategorized,
            }
        )
    return summaries


def blocker_matches_any_artifact(blocker: str, artifact_names: list[str]) -> bool:
    return any(Path(token.strip("`:,")).name in artifact_names for token in blocker.split())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Asset Index Check",
        "",
        f"- OK: `{str(report['ok']).lower()}`",
        f"- Asset index: `{report['asset_index']}`",
        f"- Declared count: `{report.get('declared_count')}`",
        f"- Actual count: `{report.get('actual_count')}`",
        f"- Blockers: `{len(report.get('blockers', []))}`",
        f"- Warnings: `{len(report.get('warnings', []))}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in report.get("blockers", [])] or ["- none"])
    lines.extend(
        [
            "",
            "## Blocker Summary",
            "",
        ]
    )
    for item in report.get("blocker_summary", []) or []:
        lines.append(f"- `{item['category']}`: {item['count']} ({item['label']})")
    if not report.get("blocker_summary"):
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in report.get("warnings", [])] or ["- none"])
    lines.extend(
        [
            "",
            "## Warning Summary",
            "",
        ]
    )
    for item in report.get("warning_summary", []) or []:
        lines.append(f"- `{item['category']}`: {item['count']} ({item['label']})")
    if not report.get("warning_summary"):
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Paper Path | Present | SHA Match |",
            "| --- | --- | --- |",
        ]
    )
    for item in report.get("checks", []):
        lines.append(
            f"| `{item['paper_path']}` | `{str(item['paper_present']).lower()}` | "
            f"`{str(item['sha256_matches']).lower()}` |"
        )
    if not report.get("checks"):
        lines.append("| none |  |  |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
