#!/usr/bin/env python3
"""Convert query-pool records to policy-RLVR train/validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


PROMPT_TEMPLATE = "{query}"
DEFAULT_QUERY_KEYS = ("query", "prompt", "instruction", "question", "task")


def main() -> None:
    args = parse_args()
    records = convert_records(
        load_records(args.input),
        data_source=args.data_source,
        prompt_template=args.prompt_template,
        query_keys=tuple(args.query_key or DEFAULT_QUERY_KEYS),
        metadata_keys=tuple(args.metadata_key or []),
    )
    if len(records) < args.min_records:
        raise SystemExit(f"Only {len(records)} valid policy-RLVR records found; required at least {args.min_records}.")
    write_records(args.output, records)
    print(f"Wrote {len(records)} policy-RLVR records to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert query-pool records to policy-RLVR data.")
    parser.add_argument("--input", required=True, type=Path, help="Input query-pool JSONL/JSON/parquet.")
    parser.add_argument("--output", required=True, type=Path, help="Output .parquet or .jsonl path.")
    parser.add_argument("--data-source", required=True, help="Data source label for output rows.")
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--query-key", action="append", help="Candidate query field. Repeatable.")
    parser.add_argument("--metadata-key", action="append", help="Extra input field to preserve. Repeatable.")
    parser.add_argument("--min-records", type=int, default=1)
    return parser.parse_args()


def convert_records(
    rows: Iterable[dict[str, Any]],
    data_source: str,
    prompt_template: str = PROMPT_TEMPLATE,
    query_keys: tuple[str, ...] = DEFAULT_QUERY_KEYS,
    metadata_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    out = []
    seen_prompts = set()
    for record in rows:
        query = pick_first(record, query_keys)
        if not query:
            continue
        prompt = prompt_template.format(query=query)
        key = " ".join(prompt.split())
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        row = {
            "prompt": prompt,
            "query": str(query),
            "data_source": str(record.get("data_source") or data_source),
        }
        for metadata_key in metadata_keys:
            if metadata_key in record and record[metadata_key] not in (None, ""):
                row[metadata_key] = json_safe(record[metadata_key])
        out.append(row)
    return out


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield ensure_record(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in unwrap_records(data):
            yield ensure_record(item)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        for item in pd.read_parquet(path).to_dict(orient="records"):
            yield ensure_record(item)
        return
    raise ValueError(f"Unsupported input format: {path}")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas/pyarrow are required to write parquet files.") from exc
    pd.DataFrame(records).to_parquet(path, index=False)


def unwrap_records(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["records", "data", "items", "examples", "rows", "queries"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return [data]


def ensure_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"query": value}


def pick_first(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


if __name__ == "__main__":
    main()
