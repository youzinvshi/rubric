#!/usr/bin/env python3
"""Finalize proxy teacher outputs into auditable training-asset metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCES = ["rewardbench_pref", "ifbench", "writingbench", "healthbench", "beir_nq"]
TEACHERS = ["gpt-5.4", "gpt-5", "gpt-4o"]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    teacher_pool_dir = args.output_root / "teacher_pool"
    audit_dir = args.output_root / "generation_audit"
    teacher_pool_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    shard_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    source_summaries: dict[str, dict[str, Any]] = {}

    for source in SOURCES:
        candidates = collect_source_files(args, source)
        rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        provenance_by_key: dict[tuple[str, str], str] = {}

        for priority, path in enumerate(candidates):
            rows = read_jsonl(path)
            shard_summary = summarize_shard(source, path, rows)
            shard_summary["priority"] = priority
            shard_rows.append(shard_summary)
            for row in rows:
                query = str(row.get("query") or "")
                teacher = str(row.get("teacher") or row.get("method") or "")
                if not query or not teacher:
                    continue
                normalized = normalize_row(row, source=source, teacher=teacher, shard_path=path)
                key = (query, teacher)
                old = rows_by_key.get(key)
                if old is None or better_record(normalized, old):
                    rows_by_key[key] = normalized
                    provenance_by_key[key] = str(path)

        output = teacher_pool_dir / f"{source}_teachers.jsonl"
        sorted_rows = sorted(
            rows_by_key.values(),
            key=lambda row: (
                int(row.get("query_idx", 10**12)) if str(row.get("query_idx", "")).isdigit() else 10**12,
                str(row.get("teacher", "")),
                str(row.get("query", "")),
            ),
        )
        write_jsonl(output, sorted_rows)
        failure_rows.extend(extract_failures(source, sorted_rows))
        source_summaries[source] = summarize_source(source, sorted_rows, output, candidates, provenance_by_key)
        print(f"Finalized {source}: {len(sorted_rows)} rows -> {output}")

    shard_summary_path = audit_dir / "shard_summary.jsonl"
    failure_detail_path = audit_dir / "generation_failures.jsonl"
    failure_summary_path = audit_dir / "failure_summary.json"
    manifest_path = args.output_root / "proxy_training_asset_manifest.json"

    write_jsonl(shard_summary_path, shard_rows)
    write_jsonl(failure_detail_path, failure_rows)
    failure_summary = summarize_failures(failure_rows, source_summaries)
    failure_summary_path.write_text(json.dumps(failure_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_root": str(args.output_root),
        "gold_type": "proxy_teacher",
        "is_proxy": True,
        "is_human_gold": False,
        "allowed_in_main_bsc_eval": False,
        "contamination_policy": "Proxy teacher data is excluded from main human-gold BSC evaluation.",
        "teacher_models": TEACHERS,
        "sources": source_summaries,
        "generation_audit": {
            "shard_summary": str(shard_summary_path),
            "generation_failures": str(failure_detail_path),
            "failure_summary": str(failure_summary_path),
            "failure_summary_sha256": sha256(failure_summary_path),
        },
        "rule_verifier": {
            "status": "pending",
            "mode": "rule",
            "script": "scripts/filter_rubrics_with_verifier.py",
            "dedup_tau": args.dedup_tau,
            "max_rubrics": args.max_rubrics,
            "reject_generic_terms": True,
        },
        "relaxed_api_audit": {
            "status": "pending",
            "mode": "record_level_relaxed_api_audit",
            "sample_size_per_source": args.api_audit_sample_size,
            "script": "scripts/run_relaxed_verifier_record_audit.py",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-root", type=Path, default=Path("outputs/proxy_generation_parallel/teacher_outputs"))
    parser.add_argument(
        "--gpt5-missing-root",
        type=Path,
        default=Path("outputs/proxy_generation_parallel/gpt5_missing_parallel/teacher_outputs"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/proxy_generation_parallel/final_asset"))
    parser.add_argument("--dedup-tau", type=float, default=0.90)
    parser.add_argument("--max-rubrics", type=int, default=12)
    parser.add_argument("--api-audit-sample-size", type=int, default=200)
    return parser.parse_args()


def collect_source_files(args: argparse.Namespace, source: str) -> list[Path]:
    files: list[Path] = []
    source_dir = args.parallel_root / source
    if source_dir.exists():
        files.extend(sorted(source_dir.glob("*.jsonl")))
    missing_dir = args.gpt5_missing_root / source
    if missing_dir.exists():
        files.extend(sorted(missing_dir.glob("*.jsonl")))
    return files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row: dict[str, Any], *, source: str, teacher: str, shard_path: Path) -> dict[str, Any]:
    out = dict(row)
    out["data_source"] = out.get("data_source") or source
    out["teacher"] = teacher
    out["gold_type"] = "proxy_teacher"
    out["is_proxy"] = True
    out["is_human_gold"] = False
    out["allowed_in_main_bsc_eval"] = False
    out["source"] = source
    out["source_shard"] = str(shard_path)
    out["raw_rubrics"] = out.get("raw_rubrics") or out.get("rubrics") or out.get("response") or []
    out["prompt_version"] = out.get("prompt_version") or infer_prompt_version(source)
    out["timestamp"] = out.get("timestamp") or out.get("generated_at") or ""
    out["query_id"] = out.get("query_id") or stable_id(source, str(out.get("query", "")))
    out["generation_failed"] = bool(out.get("generation_failed"))
    return out


def infer_prompt_version(source: str) -> str:
    return f"teacher_rubric_generation_domain_v1:{source}"


def stable_id(source: str, query: str) -> str:
    digest = hashlib.sha1(f"{source}\n{query}".encode("utf-8")).hexdigest()[:16]
    return f"{source}_{digest}"


def better_record(new: dict[str, Any], old: dict[str, Any]) -> bool:
    if bool(old.get("generation_failed")) and not bool(new.get("generation_failed")):
        return True
    if bool(new.get("generation_failed")) and not bool(old.get("generation_failed")):
        return False
    return rubric_count(new) > rubric_count(old)


def rubric_count(row: dict[str, Any]) -> int:
    rubrics = row.get("rubrics")
    if isinstance(rubrics, list):
        return len(rubrics)
    if isinstance(rubrics, str):
        return int(bool(rubrics.strip()))
    return 0


def summarize_shard(source: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_counts = Counter(str(row.get("teacher") or row.get("method") or "") for row in rows)
    failed = [row for row in rows if row.get("generation_failed")]
    unique_queries = {str(row.get("query") or "") for row in rows if row.get("query")}
    return {
        "source": source,
        "shard_path": str(path),
        "sha256": sha256(path),
        "records": len(rows),
        "unique_queries": len(unique_queries),
        "success_count": len(rows) - len(failed),
        "failure_count": len(failed),
        "teacher_counts": dict(teacher_counts),
        "failure_reasons": dict(Counter(classify_failure(row) for row in failed)),
    }


def summarize_source(
    source: str,
    rows: list[dict[str, Any]],
    output: Path,
    input_files: list[Path],
    provenance_by_key: dict[tuple[str, str], str],
) -> dict[str, Any]:
    by_teacher = Counter(str(row.get("teacher") or "") for row in rows)
    failed_by_teacher = Counter(str(row.get("teacher") or "") for row in rows if row.get("generation_failed"))
    failed_by_reason = Counter(classify_failure(row) for row in rows if row.get("generation_failed"))
    unique_queries = {str(row.get("query") or "") for row in rows if row.get("query")}
    return {
        "data_source": source,
        "teacher_pool_path": str(output),
        "teacher_pool_sha256": sha256(output),
        "records": len(rows),
        "unique_queries": len(unique_queries),
        "teacher_counts": dict(by_teacher),
        "generation_failed_count": sum(failed_by_teacher.values()),
        "generation_failed_rate": sum(failed_by_teacher.values()) / max(len(rows), 1),
        "failure_by_teacher": dict(failed_by_teacher),
        "failure_by_reason": dict(failed_by_reason),
        "input_shards": [str(path) for path in input_files],
        "selected_record_sources": dict(Counter(provenance_by_key.values())),
    }


def extract_failures(source: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        if not row.get("generation_failed"):
            continue
        failures.append(
            {
                "data_source": source,
                "teacher": row.get("teacher", ""),
                "query_id": row.get("query_id", ""),
                "query": row.get("query", ""),
                "failure_type": classify_failure(row),
                "error": row.get("error") or row.get("error_message") or row.get("response") or "",
                "source_shard": row.get("source_shard", ""),
            }
        )
    return failures


def classify_failure(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ["error", "error_message", "response", "output"]).lower()
    if not row.get("generation_failed"):
        return "success"
    if "refus" in text or "safety" in text:
        return "safety_refusal"
    if "parse" in text or "json" in text:
        return "parse_failure"
    if "empty" in text or not str(row.get("response") or row.get("output") or row.get("rubrics") or "").strip():
        return "empty_output"
    if "http" in text or "api" in text or "timeout" in text or "rate" in text:
        return "api_error"
    return "other_generation_failure"


def summarize_failures(
    failure_rows: list[dict[str, Any]],
    source_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_source = Counter(row["data_source"] for row in failure_rows)
    by_teacher = Counter(row["teacher"] for row in failure_rows)
    by_reason = Counter(row["failure_type"] for row in failure_rows)
    source_rates = {}
    for source, summary in source_summaries.items():
        source_rates[source] = {
            "failure_count": by_source[source],
            "records": summary["records"],
            "failure_rate": by_source[source] / max(int(summary["records"]), 1),
        }
    return {
        "total_failure_count": len(failure_rows),
        "failure_by_source": dict(by_source),
        "failure_rate_by_source": source_rates,
        "failure_by_teacher": dict(by_teacher),
        "failure_by_reason": dict(by_reason),
    }


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
