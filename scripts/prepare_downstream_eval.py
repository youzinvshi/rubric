#!/usr/bin/env python3
"""Join preference benchmark records with generated evaluation criteria."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402


def main() -> None:
    args = parse_args()
    preferences = list(load_records(args.preferences))
    rubric_records = list(load_records(args.rubrics))
    rubrics_by_query, rubric_stats = build_rubric_index(rubric_records, model=args.model)

    joined_candidates = []
    eligible_keys = []
    source_occurrences: dict[str, int] = {}
    skipped_preferences = 0
    for record in preferences:
        query = pick_first(record, "query", "prompt", "instruction")
        chosen = pick_first(record, "chosen", "winner", "response_chosen", "answer_chosen")
        rejected = pick_first(record, "rejected", "loser", "response_rejected", "answer_rejected")
        if not query or chosen is None or rejected is None:
            skipped_preferences += 1
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
                "chosen": chosen,
                "rejected": rejected,
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
        raise SystemExit("No joined downstream records found. Check query keys/model filter.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    report = build_join_report(
        source_label="preferences",
        source_path=args.preferences,
        rubrics_path=args.rubrics,
        output_path=args.output,
        data_source=args.data_source,
        model=args.model or "",
        counts={
            "n_preferences": len(preferences),
            "n_preferences_raw": len(preferences),
            "n_source_records_raw": len(preferences),
            "n_eligible_preferences": len(eligible_keys),
            "n_source_eligible_records": len(eligible_keys),
            "n_preference_join_keys": len(source_occurrences),
            "n_source_join_keys": len(source_occurrences),
            "preference_duplicate_join_key_count": occurrence_stats(source_occurrences)["duplicate_join_key_count"],
            "preference_duplicate_record_count": occurrence_stats(source_occurrences)["duplicate_record_count"],
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
            "n_skipped_preferences": skipped_preferences,
            "query_alignment_exact": len(missing_rubrics) == 0 and len(unmatched_rubrics) == 0,
            "output_truncated_by_limit": output_truncated_by_limit,
        },
        limit=args.limit,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Prepared {len(rows)} downstream eval records at {args.output}; "
        f"missing_criteria_records={len(missing_rubrics)} skipped_preferences={skipped_preferences}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join preferences and generated evaluation criteria.")
    parser.add_argument("--preferences", required=True, type=Path)
    parser.add_argument("--rubrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", help="Optional model/teacher filter in rubric records.")
    parser.add_argument("--data-source", default="downstream_eval")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def build_rubric_map(records: list[dict[str, Any]], model: str | None = None) -> dict[str, dict[str, Any]]:
    mapped, _stats = build_rubric_index(records, model=model)
    return mapped


def build_rubric_index(
    records: list[dict[str, Any]],
    model: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    out = {}
    occurrences: dict[str, int] = {}
    skipped_missing_query_count = 0
    skipped_empty_criteria_count = 0
    skipped_model_filter_count = 0
    for record in records:
        record_model = str(record.get("model") or record.get("teacher") or "")
        if model and model not in {record_model, str(record.get("teacher") or "")}:
            skipped_model_filter_count += 1
            continue
        query = pick_first(record, "query", "prompt", "instruction")
        rubrics = parse_rubrics(
            pick_first(record, "rubrics", "generated_rubrics", "model_rubrics", "gold_rubrics", "response", "output"),
            dedupe=True,
        )
        if not query:
            skipped_missing_query_count += 1
            continue
        query_key = make_key(query)
        occurrences[query_key] = occurrences.get(query_key, 0) + 1
        if not rubrics:
            skipped_empty_criteria_count += 1
            continue
        out[query_key] = {"rubrics": rubrics, "model": record_model}
    stats = {
        "n_rubric_records_raw": len(records),
        "skipped_missing_query_count": skipped_missing_query_count,
        "skipped_empty_criteria_count": skipped_empty_criteria_count,
        "skipped_model_filter_count": skipped_model_filter_count,
        **occurrence_stats(occurrences),
    }
    return out, stats


def occurrence_stats(occurrences: dict[str, int]) -> dict[str, int]:
    return {
        "duplicate_join_key_count": sum(1 for count in occurrences.values() if count > 1),
        "duplicate_record_count": sum(count - 1 for count in occurrences.values() if count > 1),
    }


def build_join_report(
    *,
    source_label: str,
    source_path: Path,
    rubrics_path: Path,
    output_path: Path,
    data_source: str,
    model: str,
    counts: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    report = {
        source_label: str(source_path),
        f"{source_label}_sha256": file_sha256(source_path),
        "rubrics": str(rubrics_path),
        "rubrics_sha256": file_sha256(rubrics_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "data_source": data_source,
        "model": model,
        "limit": limit,
        "output_rows_count": count_jsonl_records(output_path),
    }
    report.update(counts)
    report["output_rows_match_n_joined"] = report["output_rows_count"] == report.get("n_joined")
    return report


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("records") or data.get("data") or [data]
        yield from records
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    raise ValueError(f"Unsupported input format: {path}")


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def make_key(value: Any) -> str:
    return " ".join(str(value).strip().split())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
