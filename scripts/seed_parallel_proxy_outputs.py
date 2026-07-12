#!/usr/bin/env python3
"""Seed per-source/per-teacher parallel outputs from existing combined JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    args.parallel_root.mkdir(parents=True, exist_ok=True)
    total_written = 0
    for source in args.source:
        combined = args.combined_root / f"{source}_teachers.jsonl"
        if not combined.exists():
            continue
        rows_by_teacher: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        for row in read_jsonl(combined):
            query = str(row.get("query") or "")
            teacher = str(row.get("teacher") or row.get("method") or "")
            if not query or not teacher:
                continue
            rows_by_teacher.setdefault(teacher, {})[(query, teacher)] = row

        for teacher, keyed_rows in sorted(rows_by_teacher.items()):
            output = args.parallel_root / source / f"{teacher}.jsonl"
            existing = {(str(row.get("query") or ""), str(row.get("teacher") or "")) for row in read_jsonl(output)}
            rows_to_add = [row for key, row in keyed_rows.items() if key not in existing]
            if not rows_to_add:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as f:
                for row in rows_to_add:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_written += len(rows_to_add)
            print(f"Seeded {len(rows_to_add)} rows -> {output}")
    print(f"Seed complete: {total_written} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-root", type=Path, default=Path("outputs/proxy_generation/teacher_outputs"))
    parser.add_argument("--parallel-root", type=Path, default=Path("outputs/proxy_generation_parallel/teacher_outputs"))
    parser.add_argument("--source", action="append", required=True)
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
