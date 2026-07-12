#!/usr/bin/env python3
"""Build a selective third-teacher queue for expensive GPT-5 generation.

The queue is intentionally heuristic and auditable: each selected query records
the exact trigger reasons so GPT-5 can be used only where it adds value.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Any


HIGH_RISK_MEDICAL_TERMS = {
    "dose",
    "dosage",
    "mg",
    "medication",
    "medicine",
    "drug",
    "prescription",
    "antibiotic",
    "opioid",
    "pregnant",
    "postpartum",
    "child",
    "pediatric",
    "diagnosis",
    "symptom",
    "emergency",
    "suicide",
    "harmful thoughts",
    "covid",
    "cancer",
    "sedation",
    "surgery",
    "ivf",
    "pml",
    "感染",
    "药",
    "剂量",
    "处方",
    "诊断",
    "儿童",
    "孕",
    "急诊",
    "手术",
}

AMBIGUOUS_SEARCH_PATTERNS = [
    re.compile(r"^(what|who|when|where|how|why)\b", re.I),
    re.compile(r"\bmeaning\b|\bdefinition\b|\bnear me\b|\bbest\b|\btop\b", re.I),
]


def main() -> None:
    args = parse_args()
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for data_source, verified_path in parse_source_args(args.source):
        stats_path = args.stats_dir / f"{data_source}_stats.jsonl"
        validation_candidates = [
            args.validation_root / f"validate_{data_source}_domain_min1" / "per_record.jsonl",
            args.validation_root / f"validate_{data_source}_domain" / "per_record.jsonl",
            args.validation_root / f"validate_{data_source}_allow_generic" / "per_record.jsonl",
            args.validation_root / f"validate_{data_source}" / "per_record.jsonl",
        ]
        validation_path = next((path for path in validation_candidates if path.exists()), validation_candidates[-1])

        records = load_jsonl(verified_path)
        stats = load_stats(stats_path)
        validation = load_validation(validation_path)
        grouped = group_by_query(records)

        for query, rows in grouped.items():
            reasons: set[str] = set()
            teachers = sorted({str(row.get("teacher") or row.get("method") or "") for row in rows if row})
            source_key = (data_source, query)

            if any(is_borderline_stat(row) for row in stats.get(query, [])):
                reasons.add("verifier_borderline")
            if any(is_low_count_issue(row) for row in validation.get(query, [])):
                reasons.add("low_rubric_count")
            if is_multi_teacher_disagreement(rows, stats.get(query, [])):
                reasons.add("multi_teacher_disagreement")
            if "healthbench" in data_source and is_high_risk_medical(query):
                reasons.add("healthbench_high_risk_medical")
            if ("beir" in data_source or "search" in data_source) and is_ambiguous_search(query):
                reasons.add("beir_ambiguous_search_intent")

            if reasons:
                selected[source_key] = {
                    "query": query,
                    "data_source": data_source,
                    "selective_teacher": "gpt-5",
                    "selection_reasons": sorted(reasons),
                    "teachers_present": teachers,
                    "n_teacher_records": len(rows),
                    "metadata": {
                        "policy": "selective_gpt5_teacher_v1",
                        "source_verified_path": str(verified_path),
                    },
                }

    rows = sorted(selected.values(), key=lambda row: (row["data_source"], row["query"]))
    if args.max_per_source:
        rows = cap_per_source(rows, args.max_per_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = summarize(rows)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} selective GPT-5 queue records to {args.output}")
    print(f"Wrote report to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selective GPT-5 teacher queue.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="DATA_SOURCE=VERIFIED_JSONL. Repeatable.",
    )
    parser.add_argument("--stats-dir", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-source", type=int, help="Optional cap after prioritization.")
    return parser.parse_args()


def parse_source_args(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source must be DATA_SOURCE=PATH, got: {value}")
        name, path = value.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_stats(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query") or "")].append(row)
    return grouped


def load_validation(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query") or "")].append(row)
    return grouped


def group_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query = str(row.get("query") or row.get("prompt") or row.get("instruction") or "")
        if query:
            grouped[query].append(row)
    return grouped


def is_borderline_stat(row: dict[str, Any]) -> bool:
    n_input = int(row.get("n_input") or 0)
    n_valid = int(row.get("n_valid") or 0)
    if n_input <= 0:
        return False
    validity = n_valid / n_input
    substantive_reasons = {
        reason
        for reason in (row.get("invalid_reason_counts") or {})
        if reason not in {"generic_term", "max_rubrics_exceeded"}
    }
    return 0.0 < validity < 0.85 or bool(substantive_reasons)


def is_low_count_issue(row: dict[str, Any]) -> bool:
    issues = set(row.get("issues") or [])
    return "too_few_rubrics" in issues or int(row.get("n_rubrics") or 0) < 3


def is_multi_teacher_disagreement(rows: list[dict[str, Any]], stats: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    counts = [len(row.get("rubrics") or []) for row in rows]
    if counts and (max(counts) - min(counts) >= 4 or pstdev(counts) >= 2.0):
        return True
    validities = []
    for row in stats:
        n_input = int(row.get("n_input") or 0)
        n_valid = int(row.get("n_valid") or 0)
        if n_input:
            validities.append(n_valid / n_input)
    return bool(validities) and max(validities) - min(validities) >= 0.25


def is_high_risk_medical(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in HIGH_RISK_MEDICAL_TERMS)


def is_ambiguous_search(query: str) -> bool:
    q = " ".join(query.lower().split())
    tokens = re.findall(r"[a-z0-9]+", q)
    if len(tokens) <= 5:
        return True
    return any(pattern.search(q) for pattern in AMBIGUOUS_SEARCH_PATTERNS)


def cap_per_source(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    priority = {
        "verifier_borderline": 0,
        "low_rubric_count": 1,
        "healthbench_high_risk_medical": 2,
        "beir_ambiguous_search_intent": 2,
        "multi_teacher_disagreement": 3,
    }

    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        reasons = row.get("selection_reasons") or []
        best = min((priority.get(reason, 9) for reason in reasons), default=9)
        return best, -len(reasons), row["query"]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["data_source"]].append(row)
    capped = []
    for source in sorted(grouped):
        capped.extend(sorted(grouped[source], key=score)[:cap])
    return sorted(capped, key=lambda row: (row["data_source"], row["query"]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, int] = defaultdict(int)
    by_reason: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[row["data_source"]] += 1
        for reason in row.get("selection_reasons") or []:
            by_reason[reason] += 1
    return {
        "n_records": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


if __name__ == "__main__":
    main()
