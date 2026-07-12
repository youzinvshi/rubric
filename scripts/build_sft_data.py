#!/usr/bin/env python3
"""Build SFT and proxy-gold data from multi-teacher evaluation-criteria outputs.

Input JSONL examples:
{"query": "...", "teacher": "gpt-4o", "rubrics": ["..."]}
{"query": "...", "teacher": "claude", "response": "- criterion ..."}

Outputs:
- SFT JSONL: {"instruction": "...", "input": "...", "output": "[...]"}
- Proxy gold JSONL: {"query": "...", "gold_rubrics": [...], "data_source": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import (  # noqa: E402
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    parse_rubrics,
    semantic_dedupe,
)

from scripts.budget_gate import file_sha256  # noqa: E402

DEFAULT_FORBIDDEN_DATA_SOURCE_MARKERS = ("test_main", "holdout", "downstream")
DEFAULT_FORBIDDEN_SPLITS = ("test_main", "holdout", "downstream", "test")


INSTRUCTION = (
    "为给定 query 列出 evaluation criteria。要求：每条 criterion 必须原子化、可判 yes/no、"
    "与 query 直接相关、避免重复，输出 JSON list[str]。"
)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    n_input_records = len(records)
    blockers = holdout_source_blockers(
        records,
        input_path=args.input,
        forbidden_data_source_markers=args.forbid_data_source_marker,
        forbidden_splits=args.forbid_split,
    )
    if blockers:
        raise SystemExit("; ".join(blockers[:3]))
    records = filter_records_by_data_source(records, args.data_source_filter)
    n_filtered_records = len(records)
    grouped = group_teacher_outputs(records)
    if not grouped:
        raise SystemExit("No valid teacher rubric records found.")

    embedder = (
        TokenOverlapEmbedder()
        if args.embedding_model == "token-overlap"
        else SentenceTransformerEmbedder(args.embedding_model)
    )

    sft_rows: list[dict[str, str]] = []
    proxy_rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []

    for query, rubric_items in grouped.items():
        union = []
        teacher_names = sorted({item["teacher"] for item in rubric_items if item["teacher"]})
        if len(teacher_names) < args.min_teachers:
            continue
        for item in rubric_items:
            union.extend(item["rubrics"])
        filtered = semantic_dedupe(union, tau=args.dedupe_tau, embedder=embedder)
        if args.max_rubrics:
            filtered = filtered[: args.max_rubrics]
        if len(filtered) < args.min_rubrics:
            continue

        output = json.dumps(filtered, ensure_ascii=False, indent=2)
        sft_rows.append({"instruction": INSTRUCTION, "input": query, "output": output})
        proxy_rows.append(
            {
                "query": query,
                "gold_rubrics": filtered,
                "data_source": args.data_source,
                "teachers": teacher_names,
            }
        )
        stats.append(
            {
                "query": query,
                "n_teacher_records": len(rubric_items),
                "n_union": len(union),
                "n_after_dedupe": len(filtered),
                "teachers": teacher_names,
            }
        )

    if not sft_rows:
        raise SystemExit("No SFT rows survived filtering.")

    args.sft_output.parent.mkdir(parents=True, exist_ok=True)
    args.proxy_gold_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.sft_output, sft_rows)
    write_jsonl(args.proxy_gold_output, proxy_rows)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.stats_output, stats)
    if args.report_output:
        write_report(
            args.report_output,
            input_path=args.input,
            sft_output=args.sft_output,
            proxy_gold_output=args.proxy_gold_output,
            stats_output=args.stats_output,
            n_input_records=n_input_records,
            n_filtered_records=n_filtered_records,
            n_grouped_queries=len(grouped),
            stats=stats,
            data_source=args.data_source,
            data_source_filter=args.data_source_filter,
            embedding_model=args.embedding_model,
            dedupe_tau=args.dedupe_tau,
            min_rubrics=args.min_rubrics,
            min_teachers=args.min_teachers,
            max_rubrics=args.max_rubrics,
            forbidden_data_source_markers=args.forbid_data_source_marker,
            forbidden_splits=args.forbid_split,
        )

    print(
        "Built SFT data: "
        f"{len(sft_rows)} queries, "
        f"{sum(row['n_union'] for row in stats)} raw rubrics, "
        f"{sum(row['n_after_dedupe'] for row in stats)} after dedupe."
    )
    print(f"SFT output: {args.sft_output}")
    print(f"Proxy gold output: {args.proxy_gold_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT data from teacher evaluation-criteria outputs.")
    parser.add_argument("--input", required=True, type=Path, help="Teacher generations JSONL.")
    parser.add_argument("--sft-output", required=True, type=Path, help="LLaMA-Factory JSONL output.")
    parser.add_argument("--proxy-gold-output", required=True, type=Path, help="Proxy gold JSONL output.")
    parser.add_argument("--stats-output", type=Path, help="Optional per-query stats JSONL.")
    parser.add_argument("--report-output", type=Path, help="Optional JSON report with input/output provenance.")
    parser.add_argument("--data-source", default="multi_teacher_proxy")
    parser.add_argument(
        "--data-source-filter",
        action="append",
        default=[],
        help="Only use teacher records with this data_source. Repeat for multiple domains.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--dedupe-tau", default=0.85, type=float)
    parser.add_argument("--min-rubrics", default=1, type=int)
    parser.add_argument("--min-teachers", default=1, type=int)
    parser.add_argument("--max-rubrics", type=int)
    parser.add_argument(
        "--forbid-data-source-marker",
        action="append",
        default=list(DEFAULT_FORBIDDEN_DATA_SOURCE_MARKERS),
        help="Fail if a record data_source contains this marker. Repeatable.",
    )
    parser.add_argument(
        "--forbid-split",
        action="append",
        default=list(DEFAULT_FORBIDDEN_SPLITS),
        help="Fail if a record split/data_split/subset equals this value. Repeatable.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def filter_records_by_data_source(records: list[dict[str, Any]], filters: list[str]) -> list[dict[str, Any]]:
    if not filters:
        return records
    allowed = set(filters)
    return [record for record in records if str(record.get("data_source", "")) in allowed]


def holdout_source_blockers(
    records: list[dict[str, Any]],
    input_path: Path | None = None,
    forbidden_data_source_markers: list[str] | tuple[str, ...] = DEFAULT_FORBIDDEN_DATA_SOURCE_MARKERS,
    forbidden_splits: list[str] | tuple[str, ...] = DEFAULT_FORBIDDEN_SPLITS,
) -> list[str]:
    forbidden_markers = {str(marker).lower() for marker in forbidden_data_source_markers if str(marker).strip()}
    forbidden_split_values = {str(split).lower() for split in forbidden_splits if str(split).strip()}
    blockers: list[str] = []
    input_text = str(input_path or "").lower()
    if input_text and any(marker in input_text for marker in forbidden_markers):
        blockers.append(f"input path {input_path} is forbidden for SFT/proxy-gold construction")
    for idx, record in enumerate(records, start=1):
        data_source = str(record.get("data_source", "")).lower()
        if data_source and any(marker in data_source for marker in forbidden_markers):
            blockers.append(
                f"record {idx} data_source={record.get('data_source')} is forbidden for SFT/proxy-gold construction"
            )
        split_value = first_split_value(record)
        if split_value and split_value.lower() in forbidden_split_values:
            blockers.append(f"record {idx} split={split_value} is forbidden for SFT/proxy-gold construction")
    return blockers


def first_split_value(record: dict[str, Any]) -> str:
    for key in ("split", "data_split", "subset"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("split", "data_split", "subset"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def group_teacher_outputs(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        query = pick_first(record, "query", "prompt", "instruction")
        raw_rubrics = pick_first(record, "rubrics", "rubric_list", "response", "output", "prediction")
        rubrics = parse_rubrics(raw_rubrics, dedupe=False)
        if not query or not rubrics:
            continue
        grouped[str(query)].append(
            {
                "teacher": str(record.get("teacher") or record.get("model") or ""),
                "rubrics": rubrics,
            }
        )
    return dict(grouped)


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    *,
    input_path: Path,
    sft_output: Path,
    proxy_gold_output: Path,
    stats_output: Path | None,
    n_input_records: int,
    n_filtered_records: int,
    n_grouped_queries: int,
    stats: list[dict[str, Any]],
    data_source: str,
    data_source_filter: list[str],
    embedding_model: str,
    dedupe_tau: float,
    min_rubrics: int,
    min_teachers: int,
    max_rubrics: int | None,
    forbidden_data_source_markers: list[str],
    forbidden_splits: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "sft_output": str(sft_output),
        "sft_output_sha256": file_sha256(sft_output),
        "proxy_gold_output": str(proxy_gold_output),
        "proxy_gold_output_sha256": file_sha256(proxy_gold_output),
        "stats_output": str(stats_output) if stats_output else "",
        "stats_output_sha256": file_sha256(stats_output) if stats_output else "",
        "n_input_records": n_input_records,
        "n_filtered_records": n_filtered_records,
        "n_grouped_queries": n_grouped_queries,
        "n_sft_records": len(stats),
        "n_proxy_gold_records": len(stats),
        "n_raw_rubrics": sum(int(row.get("n_union", 0)) for row in stats),
        "n_after_dedupe": sum(int(row.get("n_after_dedupe", 0)) for row in stats),
        "data_source": data_source,
        "data_source_filter": list(data_source_filter),
        "embedding_model": embedding_model,
        "dedupe_tau": dedupe_tau,
        "min_rubrics": min_rubrics,
        "min_teachers": min_teachers,
        "max_rubrics": max_rubrics,
        "forbidden_data_source_markers": list(forbidden_data_source_markers),
        "forbidden_splits": list(forbidden_splits),
        "columns": {
            "sft": ["instruction", "input", "output"],
            "proxy_gold": ["query", "gold_rubrics", "data_source", "teachers"],
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
