#!/usr/bin/env python3
"""Create reproducible pilot/full subsets from normalized BlindSpot-RL data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    sampled = sample_records(
        records=records,
        n=args.n,
        seed=args.seed,
        stratify_key=args.stratify_key,
        dedupe_key=args.dedupe_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, sampled)
    report = build_report(records, sampled, args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Sampled {len(sampled)} / {len(records)} records to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample normalized JSONL/JSON/parquet records reproducibly.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--stratify-key", default="data_source")
    parser.add_argument("--dedupe-key", default="query")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def sample_records(
    records: list[dict[str, Any]],
    n: int,
    seed: int = 13,
    stratify_key: str | None = "data_source",
    dedupe_key: str | None = "query",
) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    deduped = dedupe_records(records, dedupe_key) if dedupe_key else list(records)
    if n >= len(deduped):
        return list(deduped)
    if not stratify_key:
        rng = random.Random(seed)
        return rng.sample(deduped, n)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduped:
        groups[str(record.get(stratify_key) or "unknown")].append(record)
    return stratified_sample(groups, n=n, seed=seed)


def stratified_sample(groups: dict[str, list[dict[str, Any]]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    total = sum(len(items) for items in groups.values())
    allocations = {}
    remainders = []
    for key, items in groups.items():
        exact = n * len(items) / total
        base = min(len(items), int(exact))
        allocations[key] = base
        remainders.append((exact - base, key))
    remaining = n - sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remaining -= 1

    while remaining > 0:
        progressed = False
        for key in sorted(groups):
            if remaining <= 0:
                break
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    sampled = []
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        sampled.extend(items[: allocations[key]])
    rng.shuffle(sampled)
    return sampled


def dedupe_records(records: list[dict[str, Any]], key: str | None) -> list[dict[str, Any]]:
    if not key:
        return list(records)
    seen = set()
    out = []
    for record in records:
        value = record.get(key)
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
        if marker in seen:
            continue
        seen.add(marker)
        out.append(record)
    return out


def build_report(records: list[dict[str, Any]], sampled: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    deduped = dedupe_records(records, args.dedupe_key) if args.dedupe_key else list(records)
    return {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "requested_n": args.n,
        "input_records": len(records),
        "deduped_records": len(deduped),
        "sampled_records": len(sampled),
        "stratify_key": args.stratify_key,
        "dedupe_key": args.dedupe_key,
        "input_strata": count_by_key(deduped, args.stratify_key),
        "sampled_strata": count_by_key(sampled, args.stratify_key),
    }


def count_by_key(records: list[dict[str, Any]], key: str | None) -> dict[str, int]:
    if not key:
        return {"all": len(records)}
    return dict(Counter(str(record.get(key) or "unknown") for record in records))


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        for key in ("records", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BlindSpot-RL Sample Report",
        "",
        f"- Input records: `{report['input_records']}`",
        f"- Deduped records: `{report['deduped_records']}`",
        f"- Sampled records: `{report['sampled_records']}`",
        f"- Seed: `{report['seed']}`",
        f"- Stratify key: `{report['stratify_key']}`",
        f"- Dedupe key: `{report['dedupe_key']}`",
        "",
        "| Stratum | Input | Sampled |",
        "| --- | --- | --- |",
    ]
    strata = sorted(set(report["input_strata"]) | set(report["sampled_strata"]))
    for key in strata:
        lines.append(f"| `{key}` | {report['input_strata'].get(key, 0)} | {report['sampled_strata'].get(key, 0)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
