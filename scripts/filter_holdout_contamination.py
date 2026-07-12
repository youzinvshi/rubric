#!/usr/bin/env python3
"""Filter training records whose queries overlap a hard-gold holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_holdout_contamination import DEFAULT_QUERY_KEYS, collect_queries, extract_query_candidates, load_records
from scripts.budget_gate import file_sha256


def main() -> None:
    args = parse_args()
    query_keys = tuple(args.query_key or DEFAULT_QUERY_KEYS)
    report = filter_holdout_contamination(
        holdout=args.holdout,
        input_path=args.input,
        output=args.output,
        query_keys=query_keys,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Filtered {report['input_records']} records -> {report['output_records']} "
        f"removed={report['removed_records']} output={args.output}"
    )
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove hard-gold holdout query overlaps from a training JSONL file.")
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--query-key", action="append")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def filter_holdout_contamination(
    *,
    holdout: Path,
    input_path: Path,
    output: Path,
    query_keys: tuple[str, ...] = DEFAULT_QUERY_KEYS,
) -> dict[str, Any]:
    blockers: list[str] = []
    holdout_queries, holdout_raw_queries, holdout_records, holdout_blockers = collect_queries(
        holdout,
        query_keys=query_keys,
    )
    blockers.extend(f"holdout: {item}" for item in holdout_blockers)
    if not input_path.exists() or input_path.stat().st_size == 0:
        blockers.append(f"input missing or empty: {input_path}")
        return build_report(
            holdout,
            input_path,
            output,
            holdout_records,
            len(holdout_raw_queries),
            len(holdout_queries),
            0,
            0,
            [],
            blockers,
            query_keys,
        )

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    input_records = 0
    for record in load_records(input_path):
        input_records += 1
        queries = extract_query_candidates(record, query_keys=query_keys)
        overlap_queries = sorted(query for query in queries if query in holdout_queries)
        if overlap_queries:
            removed.append({"query": overlap_queries[0], "overlap_queries": overlap_queries, "record": record})
        else:
            kept.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return build_report(
        holdout,
        input_path,
        output,
        holdout_records,
        len(holdout_raw_queries),
        len(holdout_queries),
        input_records,
        len(kept),
        removed,
        blockers,
        query_keys,
    )


def build_report(
    holdout: Path,
    input_path: Path,
    output: Path,
    holdout_records: int,
    holdout_raw_unique_queries: int,
    holdout_unique_queries: int,
    input_records: int,
    output_records: int,
    removed: list[dict[str, Any]],
    blockers: list[str],
    query_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "ok": not blockers,
        "holdout": str(holdout),
        "holdout_sha256": file_sha256(holdout) if holdout.exists() and holdout.stat().st_size > 0 else "",
        "input": str(input_path),
        "input_sha256": file_sha256(input_path) if input_path.exists() and input_path.stat().st_size > 0 else "",
        "output": str(output),
        "output_sha256": file_sha256(output) if output.exists() else "",
        "holdout_records": holdout_records,
        "holdout_raw_unique_queries": holdout_raw_unique_queries,
        "holdout_unique_queries": holdout_unique_queries,
        "input_records": input_records,
        "output_records": output_records,
        "removed_records": len(removed),
        "removed_queries": sorted({item["query"] for item in removed}),
        "query_keys": list(query_keys),
        "blockers": blockers,
    }


if __name__ == "__main__":
    main()
