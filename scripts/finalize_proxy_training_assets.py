#!/usr/bin/env python3
"""Finalize generated proxy rubrics into auditable training assets.

This script is intentionally offline. It does not call teacher or verifier APIs.
It normalizes generated rows, merges retry/overlay outputs by (query, teacher),
summarizes generation failures, summarizes rule-verifier outputs when present,
and writes a contamination-aware manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = ["rewardbench_pref", "writingbench", "ifbench", "healthbench", "beir_nq"]
DEFAULT_TEACHERS = ["gpt-4o", "gpt-5", "gpt-5.4"]


def main() -> None:
    args = parse_args()
    sources = args.sources or DEFAULT_SOURCES
    teachers = args.teachers or DEFAULT_TEACHERS
    args.output_root.mkdir(parents=True, exist_ok=True)

    merged_paths: dict[str, Path] = {}
    all_merged: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        rows = collect_rows(source, args.input_roots, include_extra_rounds=args.include_extra_rounds)
        merged, dedup = merge_rows(rows)
        output = args.output_root / f"{source}_teachers.jsonl"
        write_jsonl(output, merged)
        merged_paths[source] = output
        all_merged[source] = merged
        print(f"Merged {source}: raw_rows={len(rows)} deduped={len(merged)} output={output}")
        dedup_reports = args.report_root / "dedup"
        dedup_reports.mkdir(parents=True, exist_ok=True)
        write_json(dedup_reports / f"{source}_dedup.json", dedup)

    shard_report = build_shard_report(args, sources)
    source_teacher_report = build_source_teacher_report(all_merged, sources, teachers)
    failure_report = build_failure_report(all_merged)
    verifier_report = build_verifier_report(args.rule_verified_root, sources)
    api_audit_report = build_api_audit_report(args.api_audit_root, sources)
    manifest = build_manifest(
        args=args,
        sources=sources,
        teachers=teachers,
        merged_paths=merged_paths,
        shard_report=shard_report,
        source_teacher_report=source_teacher_report,
        failure_report=failure_report,
        verifier_report=verifier_report,
        api_audit_report=api_audit_report,
    )

    args.report_root.mkdir(parents=True, exist_ok=True)
    write_json(args.report_root / "shard_completion_report.json", shard_report)
    write_json(args.report_root / "source_teacher_completion_report.json", source_teacher_report)
    write_json(args.report_root / "generation_failure_report.json", failure_report)
    write_json(args.report_root / "rule_verifier_summary.json", verifier_report)
    write_json(args.report_root / "api_audit_summary.json", api_audit_report)
    write_json(args.manifest_output, manifest)
    print(f"Wrote manifest: {args.manifest_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/proxy_generation_parallel/teacher_outputs"),
            Path("outputs/proxy_generation_parallel/gpt5_missing_parallel/teacher_outputs"),
        ],
        help="Teacher output roots. Later roots override earlier roots for duplicate (query, teacher) pairs.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/proxy_generation_parallel/merged"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs/proxy_generation_parallel/asset_closure"))
    parser.add_argument("--manifest-output", type=Path, default=Path("outputs/proxy_generation_parallel/asset_manifest.json"))
    parser.add_argument("--rule-verified-root", type=Path, default=Path("outputs/proxy_generation_parallel/rule_verified"))
    parser.add_argument("--api-audit-root", type=Path, default=Path("outputs/proxy_generation_parallel/api_audit_record"))
    parser.add_argument("--missing-root", type=Path, default=Path("outputs/proxy_generation_parallel/gpt5_missing_parallel"))
    parser.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    parser.add_argument("--teachers", nargs="*", default=DEFAULT_TEACHERS)
    parser.add_argument("--verifier-version", default="rule_meta_verifier_v1")
    parser.add_argument("--dedup-tau", type=float, default=0.90)
    parser.add_argument("--prompt-version", default="teacher_prompt_v1")
    parser.add_argument(
        "--include-extra-rounds",
        action="store_true",
        help="Include extra_round*.jsonl retry outputs. Default excludes them to avoid half-written overlap.",
    )
    return parser.parse_args()


def collect_rows(source: str, roots: list[Path], *, include_extra_rounds: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_idx, root in enumerate(roots):
        source_dir = root / source
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.jsonl")):
            if not include_extra_rounds and path.name.startswith("extra_round"):
                continue
            for row_idx, row in enumerate(read_jsonl(path)):
                row["_asset_source_file"] = str(path)
                row["_asset_source_root_order"] = root_idx
                row["_asset_source_row"] = row_idx
                row["_asset_source_name"] = source
                rows.append(row)
    return rows


def merge_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        query = str(row.get("query") or "")
        teacher = str(row.get("teacher") or row.get("method") or "")
        if not query or not teacher:
            continue
        key = (query, teacher)
        duplicate_counts[key] += 1
        candidate = normalize_row(row)
        current = by_key.get(key)
        if current is None or should_replace(current, candidate):
            by_key[key] = candidate
    merged = sorted(by_key.values(), key=lambda row: (str(row.get("source", "")), int(row.get("query_idx", 10**12)), str(row.get("teacher", "")), str(row.get("query", ""))))
    dedup = {
        "raw_rows": len(rows),
        "merged_rows": len(merged),
        "duplicate_pairs": sum(1 for count in duplicate_counts.values() if count > 1),
        "duplicate_rows_removed": sum(count - 1 for count in duplicate_counts.values() if count > 1),
        "dedup_key": ["query", "teacher"],
        "dedup_policy": "prefer later input root, then non-failed/non-empty rubrics",
    }
    return merged, dedup


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    query = str(row.get("query") or "")
    source = str(row.get("_asset_source_name") or row.get("source") or row.get("data_source") or "")
    teacher = str(row.get("teacher") or row.get("method") or "")
    raw_rubrics = row.get("raw_rubrics", row.get("rubrics", []))
    if raw_rubrics is None:
        raw_rubrics = []
    failed = bool(row.get("generation_failed")) or not bool(raw_rubrics)
    query_id = row.get("query_id") or stable_query_id(source, query)
    timestamp = row.get("timestamp") or file_timestamp(row.get("_asset_source_file"))
    prompt_version = row.get("prompt_version") or f"teacher_prompt_v1:{source or 'unknown'}"
    normalized = dict(row)
    normalized.update(
        {
            "source": source,
            "data_source": source,
            "teacher": teacher,
            "query_id": query_id,
            "generation_failed": failed,
            "raw_rubrics": raw_rubrics,
            "rubrics": raw_rubrics,
            "timestamp": timestamp,
            "prompt_version": prompt_version,
        }
    )
    return normalized


def should_replace(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    current_order = int(current.get("_asset_source_root_order", 0))
    candidate_order = int(candidate.get("_asset_source_root_order", 0))
    if candidate_order != current_order:
        return candidate_order > current_order
    current_ok = not current.get("generation_failed") and bool(current.get("raw_rubrics"))
    candidate_ok = not candidate.get("generation_failed") and bool(candidate.get("raw_rubrics"))
    if candidate_ok != current_ok:
        return candidate_ok
    return int(candidate.get("_asset_source_row", 0)) >= int(current.get("_asset_source_row", 0))


def build_shard_report(args: argparse.Namespace, sources: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": now_iso(), "sources": {}}
    missing_root = args.missing_root
    for source in sources:
        source_report: dict[str, Any] = {"shards": [], "totals": {"query_count": 0, "success_count": 0, "failure_count": 0}}
        shard_dirs = [missing_root / "shards" / source]
        if source == "healthbench" and args.include_extra_rounds:
            shard_dirs.append(missing_root / "shards" / "healthbench_extra_round1")
        for shard_dir in shard_dirs:
            if not shard_dir.exists():
                continue
            for shard_path in sorted(shard_dir.glob("*.jsonl")):
                output_name = output_name_for_shard(shard_path)
                output_path = missing_root / "teacher_outputs" / source / output_name
                input_count = count_jsonl(shard_path)
                output_rows = read_jsonl(output_path) if output_path.exists() else []
                success_count = sum(1 for row in output_rows if not normalize_row(row).get("generation_failed"))
                failure_count = sum(1 for row in output_rows if normalize_row(row).get("generation_failed"))
                item = {
                    "source": source,
                    "shard": str(shard_path),
                    "output": str(output_path),
                    "query_count": input_count,
                    "output_count": len(output_rows),
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "complete": output_path.exists() and len(output_rows) >= input_count,
                }
                source_report["shards"].append(item)
                source_report["totals"]["query_count"] += input_count
                source_report["totals"]["success_count"] += success_count
                source_report["totals"]["failure_count"] += failure_count
        report["sources"][source] = source_report
    return report


def output_name_for_shard(shard_path: Path) -> str:
    name = shard_path.stem
    if name.startswith("extra_round1_"):
        return f"{name}_gpt-5.jsonl"
    return f"{name}_gpt-5.jsonl"


def build_failure_report(all_merged: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total = 0
    failures = 0
    by_source: dict[str, Any] = {}
    by_teacher: dict[str, Counter[str]] = defaultdict(Counter)
    categories: Counter[str] = Counter()
    for source, rows in all_merged.items():
        source_total = len(rows)
        source_failures = 0
        source_categories: Counter[str] = Counter()
        for row in rows:
            total += 1
            teacher = str(row.get("teacher") or "")
            failed = bool(row.get("generation_failed"))
            by_teacher[teacher]["total"] += 1
            if failed:
                failures += 1
                source_failures += 1
                by_teacher[teacher]["failures"] += 1
                category = classify_failure(row)
                categories[category] += 1
                source_categories[category] += 1
        by_source[source] = {
            "total": source_total,
            "failures": source_failures,
            "failure_rate": source_failures / max(source_total, 1),
            "failure_categories": dict(source_categories),
        }
    return {
        "generated_at": now_iso(),
        "total_records": total,
        "total_failure_count": failures,
        "total_failure_rate": failures / max(total, 1),
        "by_source": by_source,
        "by_teacher": {
            teacher: {
                "total": counts["total"],
                "failures": counts["failures"],
                "failure_rate": counts["failures"] / max(counts["total"], 1),
            }
            for teacher, counts in sorted(by_teacher.items())
        },
        "failure_categories": dict(categories),
        "failure_category_rates_among_failures": {
            key: value / max(failures, 1) for key, value in sorted(categories.items())
        },
    }


def build_source_teacher_report(
    all_merged: dict[str, list[dict[str, Any]]],
    sources: list[str],
    teachers: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": now_iso(), "sources": {}}
    for source in sources:
        rows = all_merged.get(source, [])
        source_report: dict[str, Any] = {"teachers": {}, "complete": True}
        for teacher in teachers:
            teacher_rows = [row for row in rows if str(row.get("teacher") or "") == teacher]
            success_count = sum(1 for row in teacher_rows if not row.get("generation_failed"))
            failure_count = sum(1 for row in teacher_rows if row.get("generation_failed"))
            item = {
                "row_count": len(teacher_rows),
                "success_count": success_count,
                "failure_count": failure_count,
                "failure_rate": failure_count / max(len(teacher_rows), 1),
                "has_output": len(teacher_rows) > 0,
            }
            source_report["teachers"][teacher] = item
            source_report["complete"] = bool(source_report["complete"] and item["has_output"])
        report["sources"][source] = source_report
    return report


def classify_failure(row: dict[str, Any]) -> str:
    error = str(row.get("error") or "").lower()
    response = str(row.get("response") or "").lower()
    rubrics = row.get("raw_rubrics") or row.get("rubrics") or []
    text = f"{error}\n{response}"
    if any(term in text for term in ["safety", "refuse", "cannot comply", "can't comply", "unable to assist", "not able to help"]):
        return "safety_refusal"
    if any(term in error for term in ["http", "timeout", "rate", "api", "urlerror", "connection", "server", "quota"]):
        return "api_error"
    if not response and not rubrics:
        return "empty_output"
    if response and not rubrics:
        return "parse_failure"
    return "other_failure"


def build_verifier_report(rule_root: Path, sources: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": now_iso(), "sources": {}}
    for source in sources:
        verified_path = rule_root / f"{source}_verified.jsonl"
        stats_path = rule_root / "stats" / f"{source}_stats.jsonl"
        report_path = rule_root / "reports" / f"{source}_report.json"
        stats_rows = read_jsonl(stats_path) if stats_path.exists() else []
        reason_counts: Counter[str] = Counter()
        n_input = 0
        n_valid = 0
        for row in stats_rows:
            n_input += int(row.get("n_input", 0))
            n_valid += int(row.get("n_valid", 0))
            for reason, count in (row.get("invalid_reason_counts") or {}).items():
                reason_counts[str(reason)] += int(count)
        source_report = read_json(report_path) if report_path.exists() else {}
        report["sources"][source] = {
            "verified_rubrics": str(verified_path) if verified_path.exists() else "",
            "verifier_decisions": str(verified_path) if verified_path.exists() else "",
            "valid_flags_before_filter": str(verified_path) if verified_path.exists() else "",
            "stats": str(stats_path) if stats_path.exists() else "",
            "report": str(report_path) if report_path.exists() else "",
            "records_verified": count_jsonl(verified_path) if verified_path.exists() else 0,
            "n_input_rubrics": n_input,
            "n_valid_rubrics": n_valid,
            "validity": n_valid / max(n_input, 1),
            "dedup_stats": {
                "semantic_duplicate": reason_counts.get("semantic_duplicate", 0),
                "all_invalid_reasons": dict(reason_counts),
            },
            "dedup_tau": source_report.get("dedup_tau"),
            "verifier_version": source_report.get("mode", "rule"),
        }
    return report


def build_api_audit_report(api_root: Path, sources: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": now_iso(), "sources": {}}
    for source in sources:
        summary_path = api_root / source / f"{source}_summary.json"
        stats_path = api_root / source / f"{source}_stats.jsonl"
        record_audit_path = api_root / source / f"{source}_record_audit.jsonl"
        summary = read_json(summary_path) if summary_path.exists() else {}
        pass_rate = summary.get("validity") or summary.get("api_audit_pass_rate")
        if pass_rate is None and stats_path.exists():
            stats_rows = read_jsonl(stats_path)
            n_input = sum(int(row.get("n_input", 0)) for row in stats_rows)
            n_valid = sum(int(row.get("n_valid", 0)) for row in stats_rows)
            pass_rate = n_valid / max(n_input, 1)
        report["sources"][source] = {
            "summary": str(summary_path) if summary_path.exists() else "",
            "stats": str(stats_path) if stats_path.exists() else "",
            "record_audit": str(record_audit_path) if record_audit_path.exists() else "",
            "api_audit_pass_rate": pass_rate,
        }
    return report


def build_manifest(
    *,
    args: argparse.Namespace,
    sources: list[str],
    teachers: list[str],
    merged_paths: dict[str, Path],
    shard_report: dict[str, Any],
    source_teacher_report: dict[str, Any],
    failure_report: dict[str, Any],
    verifier_report: dict[str, Any],
    api_audit_report: dict[str, Any],
) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for source in sources:
        merged_path = merged_paths[source]
        failure_source = failure_report["by_source"].get(source, {})
        verifier_source = verifier_report["sources"].get(source, {})
        api_source = api_audit_report["sources"].get(source, {})
        assets[source] = {
            "path": str(merged_path),
            "sha256": file_sha256(merged_path),
            "num_rows": count_jsonl(merged_path),
            "gold_type": "proxy_teacher",
            "data_source": source,
            "allowed_in_main_bsc_eval": False,
            "is_proxy": True,
            "is_human_gold": False,
            "teacher": teachers,
            "verifier_version": args.verifier_version,
            "dedup_tau": args.dedup_tau,
            "api_audit_pass_rate": api_source.get("api_audit_pass_rate"),
            "generation_failed_count": failure_source.get("failures", 0),
            "generation_failure_rate": failure_source.get("failure_rate", 0.0),
            "rule_verified_path": verifier_source.get("verified_rubrics", ""),
            "rule_verified_rows": verifier_source.get("records_verified", 0),
            "contamination_policy": {
                "training_allowed": True,
                "main_bsc_eval_allowed": False,
                "holdout_human_gold_isolated": True,
            },
        }
    return {
        "schema_version": "proxy_training_asset_manifest.v1",
        "generated_at": now_iso(),
        "input_roots": [str(path) for path in args.input_roots],
        "reports": {
            "shard_completion": str(args.report_root / "shard_completion_report.json"),
            "source_teacher_completion": str(args.report_root / "source_teacher_completion_report.json"),
            "generation_failures": str(args.report_root / "generation_failure_report.json"),
            "rule_verifier": str(args.report_root / "rule_verifier_summary.json"),
            "api_audit": str(args.report_root / "api_audit_summary.json"),
            "data_audit_report": str(args.report_root / "data_audit_report.md"),
        },
        "assets": assets,
        "shard_completion_totals": {
            source: shard_report.get("sources", {}).get(source, {}).get("totals", {}) for source in sources
        },
        "source_teacher_completion": source_teacher_report.get("sources", {}),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            cleaned = {k: v for k, v in row.items() if not k.startswith("_asset_")}
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def stable_query_id(source: str, query: str) -> str:
    digest = hashlib.sha1(f"{source}\n{query}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}" if source else digest


def file_timestamp(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


if __name__ == "__main__":
    main()
