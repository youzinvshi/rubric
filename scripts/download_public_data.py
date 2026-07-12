#!/usr/bin/env python3
"""Download public benchmark splits to JSONL.

The script keeps dataset acquisition reproducible without hard-coding one
fragile schema. For BSC hard-gold datasets that may not yet have stable HF
releases, use --hf-dataset when an official repo becomes available, then map
fields in the downstream converter/diagnostic scripts.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


PRESETS = {
    "beir_nq": {
        "hf_dataset": "BeIR/nq",
        "name": "queries",
        "split": "queries",
        "output": "data/raw/beir_nq_queries.jsonl",
    },
    "healthbench": {
        "hf_dataset": "openai/healthbench",
        "split": "test",
        "streaming": True,
        "output": "data/raw/healthbench_raw.jsonl",
    },
    "ifbench": {
        "hf_dataset": "allenai/IFBench_test",
        "split": "train",
        "output": "data/raw/ifbench_test.jsonl",
    },
    "rewardbench": {
        "hf_dataset": "allenai/reward-bench",
        "split": "filtered",
        "output": "data/raw/rewardbench_filtered.jsonl",
    },
    "rewardbench2": {
        "hf_dataset": "allenai/reward-bench-2",
        "split": "test",
        "output": "data/raw/rewardbench2_test.jsonl",
    },
    "judgebench": {
        "hf_dataset": "ScalerLab/JudgeBench",
        "splits": ["gpt", "claude"],
        "output": "data/raw/judgebench_test.jsonl",
    },
    "writingbench": {
        "url": "https://raw.githubusercontent.com/X-PLUG/WritingBench/main/benchmark_query/benchmark_all.jsonl",
        "output": "data/raw/writingbench_raw.jsonl",
    },
}


def main() -> None:
    args = parse_args()
    config = PRESETS.get(args.preset, {}) if args.preset else {}

    hf_dataset = args.hf_dataset or config.get("hf_dataset")
    name = args.name or config.get("name")
    url = args.url or config.get("url")
    streaming = args.streaming or bool(config.get("streaming", False))
    splits = resolve_splits(args.split, config)
    output = args.output or config.get("output")
    if url:
        if not output:
            raise SystemExit("Provide --output when using --url.")
        records = load_url_records(url=url, limit=args.limit)
    elif hf_dataset and splits and output:
        records = []
        for split in splits:
            records.extend(
                load_hf_dataset(
                    hf_dataset=hf_dataset,
                    split=split,
                    name=name,
                    streaming=streaming,
                    limit=args.limit,
                )
            )
    else:
        raise SystemExit(
            "Provide --preset, --url with --output, or all of --hf-dataset, --split, and --output. "
            f"Available presets: {', '.join(sorted(PRESETS))}"
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    source = url or f"{hf_dataset}:{'+'.join(splits)}"
    print(f"Wrote {len(records)} records from {source} to {output_path}")


def resolve_splits(cli_split: str | None, config: dict[str, Any]) -> list[str]:
    if cli_split:
        return [cli_split]
    if config.get("splits"):
        return list(config["splits"])
    if config.get("split"):
        return [config["split"]]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public benchmarks to JSONL.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Known benchmark preset.")
    parser.add_argument("--hf-dataset", help="HuggingFace dataset path, e.g. allenai/reward-bench.")
    parser.add_argument("--url", help="Direct URL to a JSONL file or JSON array/object export.")
    parser.add_argument("--name", help="Optional HF dataset config name.")
    parser.add_argument("--split", help="HF split name.")
    parser.add_argument("--output", help="Output JSONL path.")
    parser.add_argument("--limit", type=int, help="Optional record limit for smoke tests.")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming mode.")
    return parser.parse_args()


def load_hf_dataset(
    hf_dataset: str,
    split: str,
    name: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install -r requirements.txt") from exc

    dataset = load_dataset(hf_dataset, name=name, split=split, streaming=streaming)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        item = dict(row)
        item.setdefault("data_source", hf_dataset)
        item.setdefault("split", split)
        records.append(json_safe(item))
    return records


def load_url_records(url: str, limit: int | None = None) -> list[dict[str, Any]]:
    with urlopen(url) as response:  # noqa: S310 - research data URLs are user-supplied config.
        payload = response.read()
    if is_parquet_url(url):
        records = parse_parquet_bytes(payload)
    else:
        records = parse_json_or_jsonl(payload.decode("utf-8"))
    if limit is not None:
        records = records[:limit]
    for record in records:
        record.setdefault("data_source", url)
    return [json_safe(record) for record in records]


def is_parquet_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith((".parquet", ".pq"))


def parse_parquet_bytes(payload: bytes) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas/pyarrow are required to read parquet URLs.") from exc

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            return [ensure_record(item) for item in pd.read_parquet(tmp.name).to_dict(orient="records")]
        except ImportError as exc:
            raise RuntimeError("pyarrow or fastparquet is required to read parquet URLs.") from exc


def parse_json_or_jsonl(payload: str) -> list[dict[str, Any]]:
    text = payload.strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [ensure_record(item) for item in parsed]
        if isinstance(parsed, dict):
            return unwrap_json_object(parsed)

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(ensure_record(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}") from exc
    return records


def unwrap_json_object(value: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ["data", "records", "examples", "items", "rows"]:
        nested = value.get(key)
        if isinstance(nested, list):
            return [ensure_record(item) for item in nested]
    return [ensure_record(value)]


def ensure_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
