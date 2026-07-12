#!/usr/bin/env python3
"""Validate generated evaluation-criteria records before BSC/SFT/downstream use."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import TokenOverlapEmbedder, parse_rubrics, redundancy_penalty  # noqa: E402


GENERIC_TERMS = {
    "helpful",
    "clear",
    "clarity",
    "good",
    "bad",
    "quality",
    "accurate",
    "complete",
}


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    rows = [
        validate_record(
            record,
            min_rubrics=args.min_rubrics,
            max_rubrics=args.max_rubrics,
            generic_terms=set(args.generic_term or GENERIC_TERMS),
            redundancy_tau=args.redundancy_tau,
            require_valid_flags=args.require_valid_flags,
            allow_exact_duplicates=args.allow_exact_duplicates,
            allow_generic_terms=args.allow_generic_terms,
            allow_semantic_redundancy=args.allow_semantic_redundancy,
        )
        for record in records
    ]
    report = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "per_record.jsonl", rows)
    (args.output_dir / "validation_report.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"Criteria validation ok={report['ok']} records={report['n_records']} output={args.output_dir}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated evaluation-criteria output quality.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-rubrics", type=int, default=3)
    parser.add_argument("--max-rubrics", type=int, default=12)
    parser.add_argument("--redundancy-tau", type=float, default=0.95)
    parser.add_argument("--generic-term", action="append", help="Generic term to flag. Repeatable.")
    parser.add_argument(
        "--require-valid-flags",
        action="store_true",
        help="Require binary valid_flags aligned one-to-one with parsed rubrics.",
    )
    parser.add_argument("--allow-exact-duplicates", action="store_true")
    parser.add_argument("--allow-generic-terms", action="store_true")
    parser.add_argument("--allow-semantic-redundancy", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def validate_record(
    record: dict[str, Any],
    min_rubrics: int = 3,
    max_rubrics: int = 12,
    generic_terms: set[str] | None = None,
    redundancy_tau: float = 0.95,
    require_valid_flags: bool = False,
    allow_exact_duplicates: bool = False,
    allow_generic_terms: bool = False,
    allow_semantic_redundancy: bool = False,
) -> dict[str, Any]:
    query = str(pick_first(record, "query", "prompt", "instruction") or "")
    method = str(pick_first(record, "method", "teacher", "model", "model_name_or_path") or "unknown")
    raw = pick_first(record, "rubrics", "response", "generated_rubrics", "model_rubrics", "output", "prediction")
    rubrics = parse_rubrics(raw, dedupe=False)
    normalized = [normalize_text(item) for item in rubrics]
    exact_duplicates = len(normalized) - len(set(normalized))
    generic_hits = find_generic_hits(rubrics, generic_terms or GENERIC_TERMS)
    redundancy = redundancy_penalty(rubrics, tau=redundancy_tau, embedder=TokenOverlapEmbedder()) if rubrics else 0.0
    issues = []
    if not rubrics:
        issues.append("no_parseable_rubrics")
    if len(rubrics) < min_rubrics:
        issues.append("too_few_rubrics")
    if len(rubrics) > max_rubrics:
        issues.append("too_many_rubrics")
    if exact_duplicates and not allow_exact_duplicates:
        issues.append("exact_duplicates")
    if generic_hits and not allow_generic_terms:
        issues.append("generic_terms")
    if redundancy > 0 and not allow_semantic_redundancy:
        issues.append("semantic_redundancy")
    valid_flags_status = check_valid_flags(record, rubrics) if require_valid_flags else {"required": False}
    if valid_flags_status.get("issues"):
        issues.extend(valid_flags_status["issues"])
    return {
        "query": query,
        "method": method,
        "n_rubrics": len(rubrics),
        "exact_duplicates": exact_duplicates,
        "semantic_redundancy": redundancy,
        "generic_hits": generic_hits,
        "valid_flags": valid_flags_status,
        "issues": issues,
        "ok": not issues,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n_records": 0,
            "ok_records": 0,
            "parse_failures": 0,
            "valid_flags_failures": 0,
            "too_few": 0,
            "too_many": 0,
            "duplicate_records": 0,
            "generic_records": 0,
            "mean_n_rubrics": 0.0,
            "mean_semantic_redundancy": 0.0,
        }
    )
    for row in rows:
        bucket = by_method[row["method"]]
        bucket["n_records"] += 1
        bucket["ok_records"] += int(row["ok"])
        bucket["parse_failures"] += int("no_parseable_rubrics" in row["issues"])
        bucket["valid_flags_failures"] += int(any(issue.startswith("valid_flags_") for issue in row["issues"]))
        bucket["too_few"] += int("too_few_rubrics" in row["issues"])
        bucket["too_many"] += int("too_many_rubrics" in row["issues"])
        bucket["duplicate_records"] += int("exact_duplicates" in row["issues"] or "semantic_redundancy" in row["issues"])
        bucket["generic_records"] += int("generic_terms" in row["issues"])
        bucket["mean_n_rubrics"] += row["n_rubrics"]
        bucket["mean_semantic_redundancy"] += row["semantic_redundancy"]

    methods = {}
    for method, bucket in by_method.items():
        n = max(bucket["n_records"], 1)
        bucket["ok_rate"] = bucket["ok_records"] / n
        bucket["mean_n_rubrics"] /= n
        bucket["mean_semantic_redundancy"] /= n
        methods[method] = bucket

    n_records = len(rows)
    ok_records = sum(int(row["ok"]) for row in rows)
    return {
        "ok": ok_records == n_records and n_records > 0,
        "n_records": n_records,
        "ok_records": ok_records,
        "failed_records": n_records - ok_records,
        "ok_rate": ok_records / max(n_records, 1),
        "methods": methods,
        "issue_counts": count_issues(rows),
    }


def count_issues(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for issue in row["issues"]:
            counts[issue] += 1
    return dict(counts)


def find_generic_hits(rubrics: list[str], generic_terms: set[str]) -> list[dict[str, Any]]:
    hits = []
    for idx, rubric in enumerate(rubrics):
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", rubric.lower()))
        matched = sorted(tokens & generic_terms)
        if matched:
            hits.append({"idx": idx, "terms": matched, "rubric": rubric})
    return hits


def check_valid_flags(record: dict[str, Any], rubrics: list[str]) -> dict[str, Any]:
    flags = pick_first(record, "valid_flags", "verifier_flags", "validity_flags")
    result: dict[str, Any] = {
        "required": True,
        "present": flags is not None,
        "n_flags": len(flags) if isinstance(flags, list) else 0,
        "n_rubrics": len(rubrics),
        "issues": [],
    }
    if flags is None:
        result["issues"].append("valid_flags_missing")
        return result
    if not isinstance(flags, list):
        result["issues"].append("valid_flags_not_list")
        return result
    if len(flags) != len(rubrics):
        result["issues"].append("valid_flags_length_mismatch")
    if any(not is_binary_flag(flag) for flag in flags):
        result["issues"].append("valid_flags_non_binary")
    source = pick_first(record, "verifier_source", "verifier", "verifier_model")
    result["verifier_source"] = source or ""
    if not source:
        result["issues"].append("valid_flags_missing_source")
    return result


def is_binary_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in {0, 1}
    if isinstance(value, str):
        return value.strip().lower() in {"0", "1", "true", "false", "yes", "no", "valid", "invalid"}
    return False


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("records", "data", "examples"):
        if isinstance(data.get(key), list):
            return data[key]
    return [data]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation-Criteria Output Validation",
        "",
        f"- Overall ok: `{report['ok']}`",
        f"- Records: `{report['n_records']}`",
        f"- OK rate: `{report['ok_rate']:.4f}`",
        "",
        "## Issue Counts",
        "",
    ]
    lines.extend([f"- {key}: `{value}`" for key, value in sorted(report["issue_counts"].items())] or ["- none"])
    lines.extend(
        [
            "",
            "## By Method",
            "",
            "| Method | Records | OK Rate | Mean Criteria | Parse Fail | Duplicate | Generic |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method, row in sorted(report["methods"].items()):
        lines.append(
            f"| `{method}` | {row['n_records']} | {row['ok_rate']:.4f} | "
            f"{row['mean_n_rubrics']:.2f} | {row['parse_failures']} | "
            f"{row['duplicate_records']} | {row['generic_records']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
