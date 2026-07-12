#!/usr/bin/env python3
"""Join multi-candidate benchmark records with generated evaluation criteria."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.prepare_downstream_eval import (  # noqa: E402
    build_join_report,
    build_rubric_index,
    load_records,
    make_key,
    occurrence_stats,
    pick_first,
)


def main() -> None:
    args = parse_args()
    benchmark_rows = list(load_records(args.benchmark))
    rubric_records = list(load_records(args.rubrics))
    rubrics_by_query, rubric_stats = build_rubric_index(rubric_records, model=args.model)

    joined_candidates = []
    eligible_keys = []
    source_occurrences: dict[str, int] = {}
    skipped_benchmark = 0
    for record in benchmark_rows:
        query = pick_first(record, "query", "prompt", "instruction")
        candidates = normalize_candidates(pick_first(record, "candidates", "responses", "answers", "choices", "options"))
        label = normalize_label(pick_first(record, "label", "correct", "correct_index", "answer_index", "gold", "winner", "chosen"), candidates)
        if not query or len(candidates) < 2 or label is None:
            skipped_benchmark += 1
            continue
        query_key = make_key(query)
        eligible_keys.append(query_key)
        source_occurrences[query_key] = source_occurrences.get(query_key, 0) + 1
        rubric_entry = rubrics_by_query.get(query_key)
        if rubric_entry is None:
            continue
        joined_candidates.append(
            {
                "query": str(query),
                "candidates": candidates,
                "label": label,
                "rubrics": rubric_entry["rubrics"],
                "model": rubric_entry.get("model", args.model or ""),
                "data_source": args.data_source,
            }
        )

    missing_rubrics = [query_key for query_key in eligible_keys if query_key not in rubrics_by_query]
    unmatched_rubrics = [query_key for query_key in rubrics_by_query if query_key not in set(eligible_keys)]
    rows = joined_candidates[: args.limit] if args.limit is not None else joined_candidates
    output_truncated_by_limit = args.limit is not None and len(joined_candidates) > args.limit

    if not rows:
        raise SystemExit("No joined multi-candidate records found. Check query keys/model filter.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    report = build_join_report(
        source_label="benchmark",
        source_path=args.benchmark,
        rubrics_path=args.rubrics,
        output_path=args.output,
        data_source=args.data_source,
        model=args.model or "",
        counts={
            "n_benchmark": len(benchmark_rows),
            "n_benchmark_raw": len(benchmark_rows),
            "n_source_records_raw": len(benchmark_rows),
            "n_eligible_benchmark": len(eligible_keys),
            "n_source_eligible_records": len(eligible_keys),
            "n_benchmark_join_keys": len(source_occurrences),
            "n_source_join_keys": len(source_occurrences),
            "benchmark_duplicate_join_key_count": occurrence_stats(source_occurrences)["duplicate_join_key_count"],
            "benchmark_duplicate_record_count": occurrence_stats(source_occurrences)["duplicate_record_count"],
            "source_duplicate_join_key_count": occurrence_stats(source_occurrences)["duplicate_join_key_count"],
            "source_duplicate_record_count": occurrence_stats(source_occurrences)["duplicate_record_count"],
            "n_rubric_records_raw": rubric_stats["n_rubric_records_raw"],
            "n_rubric_queries": len(rubrics_by_query),
            "n_rubric_join_keys": len(rubrics_by_query),
            "rubric_duplicate_join_key_count": rubric_stats["duplicate_join_key_count"],
            "rubric_duplicate_record_count": rubric_stats["duplicate_record_count"],
            "rubric_skipped_missing_query_count": rubric_stats["skipped_missing_query_count"],
            "rubric_skipped_empty_criteria_count": rubric_stats["skipped_empty_criteria_count"],
            "rubric_skipped_model_filter_count": rubric_stats["skipped_model_filter_count"],
            "n_joined": len(rows),
            "n_joinable": len(joined_candidates),
            "n_missing_rubrics": len(missing_rubrics),
            "n_unmatched_rubrics": len(unmatched_rubrics),
            "missing_rubric_keys_sample": missing_rubrics[:10],
            "unmatched_rubric_keys_sample": unmatched_rubrics[:10],
            "n_skipped_benchmark": skipped_benchmark,
            "query_alignment_exact": len(missing_rubrics) == 0 and len(unmatched_rubrics) == 0,
            "output_truncated_by_limit": output_truncated_by_limit,
        },
        limit=args.limit,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Prepared {len(rows)} multi-candidate eval records at {args.output}; "
        f"missing_criteria_records={len(missing_rubrics)} skipped_benchmark={skipped_benchmark}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join multi-candidate benchmark records and generated evaluation criteria."
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--rubrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", help="Optional model/teacher filter in rubric records.")
    parser.add_argument("--data-source", default="multicandidate_eval")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def normalize_candidates(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = pick_first(item, "text", "response", "answer", "content", "value")
                if text is None:
                    text = item
                out.append(stringify(text))
            else:
                out.append(stringify(item))
        return [item for item in out if item]
    if isinstance(value, dict):
        return [stringify(value[key]) for key in sorted(value, key=lambda item: str(item)) if value[key] not in (None, "")]
    return []


def normalize_label(value: Any, candidates: list[str]) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        idx = int(value)
        return idx if idx < len(candidates) else None
    if isinstance(value, int):
        return value if 0 <= value < len(candidates) else None
    text = str(value).strip()
    if text.isdigit():
        idx = int(text)
        return idx if 0 <= idx < len(candidates) else None
    for idx, candidate in enumerate(candidates):
        if text == candidate:
            return idx
    return None


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
