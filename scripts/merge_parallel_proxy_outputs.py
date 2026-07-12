#!/usr/bin/env python3
"""Merge per-teacher parallel proxy-generation shards into per-source JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_names = sorted(
        {
            source_dir.name
            for root in args.parallel_root
            if root.exists()
            for source_dir in root.iterdir()
            if source_dir.is_dir()
        }
    )
    for source_name in source_names:
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for root_order, root in enumerate(args.parallel_root):
            source_dir = root / source_name
            if not source_dir.exists():
                continue
            for shard in sorted(source_dir.glob("*.jsonl")):
                if not args.include_extra_rounds and shard.name.startswith("extra_round"):
                    continue
                for row_idx, row in enumerate(read_jsonl(shard)):
                    row["_merge_root_order"] = root_order
                    row["_merge_row_idx"] = row_idx
                    row["_merge_source_file"] = str(shard)
                    row["_merge_source_name"] = source_name
                    row = normalize_row(row, prompt_version=args.prompt_version)
                    query = str(row.get("query") or "")
                    teacher = str(row.get("teacher") or row.get("method") or "")
                    if query and teacher:
                        key = (query, teacher)
                        current = rows.get(key)
                        if current is None or should_replace(current, row):
                            rows[key] = row
        output = args.output_root / f"{source_name}_teachers.jsonl"
        with output.open("w", encoding="utf-8") as f:
            for _, row in sorted(rows.items(), key=lambda item: (item[1].get("query_idx", 0), item[0][1], item[0][0])):
                row = {key: value for key, value in row.items() if not key.startswith("_merge_")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Merged {len(rows)} rows -> {output}")


def normalize_row(row: dict[str, Any], *, prompt_version: str) -> dict[str, Any]:
    query = str(row.get("query") or "")
    source = str(row.get("_merge_source_name") or row.get("source") or row.get("data_source") or "")
    raw_rubrics = row.get("raw_rubrics", row.get("rubrics", []))
    if raw_rubrics is None:
        raw_rubrics = []
    normalized = dict(row)
    normalized.update(
        {
            "source": source,
            "data_source": source,
            "teacher": str(row.get("teacher") or row.get("method") or ""),
            "query_id": row.get("query_id") or stable_query_id(source, query),
            "generation_failed": bool(row.get("generation_failed")) or not bool(raw_rubrics),
            "raw_rubrics": raw_rubrics,
            "rubrics": raw_rubrics,
            "timestamp": row.get("timestamp") or file_timestamp(row.get("_merge_source_file")),
            "prompt_version": row.get("prompt_version") or f"{prompt_version}:{source or 'unknown'}",
        }
    )
    return normalized


def should_replace(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    current_order = int(current.get("_merge_root_order", 0))
    candidate_order = int(candidate.get("_merge_root_order", 0))
    if candidate_order != current_order:
        return candidate_order > current_order
    current_ok = not current.get("generation_failed") and bool(current.get("raw_rubrics"))
    candidate_ok = not candidate.get("generation_failed") and bool(candidate.get("raw_rubrics"))
    if candidate_ok != current_ok:
        return candidate_ok
    return int(candidate.get("_merge_row_idx", 0)) >= int(current.get("_merge_row_idx", 0))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel-root",
        type=Path,
        action="append",
        default=None,
        help="Teacher output root. May be passed multiple times; later roots override earlier duplicates.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/proxy_generation_parallel/merged"))
    parser.add_argument(
        "--include-extra-rounds",
        action="store_true",
        help="Include extra_round*.jsonl retry outputs. Defaults to excluding overlapping retry rounds.",
    )
    parser.add_argument("--prompt-version", default="teacher_prompt_v1")
    args = parser.parse_args()
    if args.parallel_root is None:
        args.parallel_root = [Path("outputs/proxy_generation_parallel/teacher_outputs")]
    return args


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


if __name__ == "__main__":
    main()
