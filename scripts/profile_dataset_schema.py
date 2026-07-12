#!/usr/bin/env python3
"""Profile raw dataset schemas before normalization."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.normalize_dataset import (  # noqa: E402
    DEFAULT_CHOSEN_KEYS,
    DEFAULT_GOLD_KEYS,
    DEFAULT_QUERY_KEYS,
    DEFAULT_REJECTED_KEYS,
    load_records,
)


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    profile = profile_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Profiled {profile['n_records']} records from {args.input} -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile raw dataset fields for normalization.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    path_counts: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, Any] = {}

    for record in records:
        for path, value in walk_paths(record):
            path_counts[path] += 1
            type_counts[path][type_name(value)] += 1
            examples.setdefault(path, short_example(value))

    fields = [
        {
            "path": path,
            "count": count,
            "types": dict(type_counts[path]),
            "example": examples.get(path),
        }
        for path, count in sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    paths = [item["path"] for item in fields]
    return {
        "n_records": len(records),
        "fields": fields,
        "suggested_mappings": {
            "query_key": suggest_key(paths, DEFAULT_QUERY_KEYS),
            "gold_key": suggest_key(paths, DEFAULT_GOLD_KEYS),
            "chosen_key": suggest_key(paths, DEFAULT_CHOSEN_KEYS),
            "rejected_key": suggest_key(paths, DEFAULT_REJECTED_KEYS),
        },
    }


def walk_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(walk_paths(child, path))
        return out
    if isinstance(value, list):
        out = [(prefix, value)] if prefix else []
        for idx, child in enumerate(value[:3]):
            path = f"{prefix}.{idx}" if prefix else str(idx)
            out.extend(walk_paths(child, path))
        return out
    return [(prefix, value)] if prefix else []


def suggest_key(paths: list[str], candidates: tuple[str, ...]) -> str | None:
    path_set = set(paths)
    for candidate in candidates:
        if candidate in path_set:
            return candidate
    for candidate in candidates:
        for path in paths:
            if path.endswith(f".{candidate}"):
                return path
    return None


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def short_example(value: Any, max_len: int = 160) -> Any:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


if __name__ == "__main__":
    main()
