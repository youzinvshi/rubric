#!/usr/bin/env python3
"""Audit query-level leakage from hard-gold holdout into training artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUERY_KEYS = ("query", "prompt", "input", "question", "instruction")


def main() -> None:
    args = parse_args()
    query_keys = tuple(args.query_key or DEFAULT_QUERY_KEYS)
    report, overlap_rows = audit_holdout_contamination(
        holdout=args.holdout,
        training_specs=args.training,
        query_keys=query_keys,
        max_examples=args.max_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, overlap_rows)
    print(
        "Holdout contamination audit "
        f"ok={report['ok']} holdout_queries={report['holdout_unique_queries']} "
        f"holdout_raw_queries={report['holdout_raw_unique_queries']} "
        f"training_queries={report['training_unique_queries']} overlaps={report['overlap_query_count']}"
    )
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit hard-gold holdout query leakage into training files.")
    parser.add_argument("--holdout", required=True, type=Path, help="Hard-gold holdout JSONL/JSON/parquet file.")
    parser.add_argument(
        "--training",
        required=True,
        action="append",
        help="Training-side file as label=path or path. Repeatable.",
    )
    parser.add_argument("--query-key", action="append", help="Query field to inspect. Repeatable.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def audit_holdout_contamination(
    *,
    holdout: Path,
    training_specs: list[str],
    query_keys: tuple[str, ...] = DEFAULT_QUERY_KEYS,
    max_examples: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blockers: list[str] = []
    warnings: list[str] = []
    holdout_queries, holdout_raw_queries, holdout_records, holdout_blockers = collect_queries(
        holdout,
        query_keys=query_keys,
    )
    blockers.extend(f"holdout: {item}" for item in holdout_blockers)

    training_reports: list[dict[str, Any]] = []
    training_query_sources: dict[str, list[dict[str, Any]]] = {}
    training_raw_queries: set[str] = set()
    for spec in training_specs:
        label, path = parse_labeled_path(spec)
        queries, raw_queries, n_records, source_blockers = collect_queries(path, query_keys=query_keys)
        blockers.extend(f"{label}: {item}" for item in source_blockers)
        training_raw_queries.update(raw_queries)
        training_reports.append(
            {
                "label": label,
                "path": str(path),
                "sha256": file_sha256(path) if path.exists() and path.stat().st_size > 0 else "",
                "records": n_records,
                "raw_unique_queries": len(raw_queries),
                "unique_queries": len(queries),
                "status": "blocked" if source_blockers else "ok",
            }
        )
        for query in queries:
            training_query_sources.setdefault(query, []).append({"label": label, "path": str(path)})

    artifact_blockers = list(blockers)
    artifact_status = "complete" if not artifact_blockers and holdout_queries and training_specs else "blocked"
    overlaps = sorted(set(holdout_queries).intersection(training_query_sources))
    overlap_rows = [
        {
            "query": query,
            "holdout_path": str(holdout),
            "training_labels": ",".join(sorted({item["label"] for item in training_query_sources[query]})),
            "training_paths": ",".join(sorted({item["path"] for item in training_query_sources[query]})),
        }
        for query in overlaps
    ]
    if overlaps:
        blockers.append(f"{len(overlaps)} holdout query(s) overlap with training artifacts")
    if not holdout_queries:
        blockers.append("holdout has no extractable queries")
    if not training_specs:
        blockers.append("no training artifacts configured")

    overlap_status = "not_auditable" if artifact_status != "complete" else ("overlap_found" if overlaps else "clear")
    report = {
        "ok": not blockers,
        "artifact_status": artifact_status,
        "overlap_status": overlap_status,
        "holdout": str(holdout),
        "holdout_sha256": file_sha256(holdout) if holdout.exists() and holdout.stat().st_size > 0 else "",
        "holdout_records": holdout_records,
        "holdout_raw_unique_queries": len(holdout_raw_queries),
        "holdout_unique_queries": len(holdout_queries),
        "training": training_reports,
        "training_raw_unique_queries": len(training_raw_queries),
        "training_unique_queries": len(training_query_sources),
        "overlap_query_count": len(overlaps),
        "overlap_examples": overlap_rows[:max_examples],
        "query_keys": list(query_keys),
        "blockers": blockers,
        "warnings": warnings,
    }
    return report, overlap_rows


def collect_queries(path: Path, *, query_keys: tuple[str, ...]) -> tuple[set[str], set[str], int, list[str]]:
    blockers: list[str] = []
    if not path.exists() or path.stat().st_size == 0:
        return set(), set(), 0, [f"missing or empty file: {path}"]
    queries: set[str] = set()
    raw_queries: set[str] = set()
    n_records = 0
    try:
        for record in load_records(path):
            n_records += 1
            raw_queries.update(extract_raw_query_candidates(record, query_keys=query_keys))
            queries.update(extract_query_candidates(record, query_keys=query_keys))
    except (OSError, ValueError, RuntimeError) as exc:
        blockers.append(str(exc))
    if n_records and not queries:
        blockers.append(f"no query fields found using keys: {', '.join(query_keys)}")
    return queries, raw_queries, n_records, blockers


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
                    raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
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
            raise RuntimeError("pandas/pyarrow are required to read parquet files") from exc
        for item in pd.read_parquet(path).to_dict(orient="records"):
            yield ensure_record(item)
        return
    raise ValueError(f"unsupported input format: {path}")


def unwrap_records(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "data", "items", "examples", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return [data]


def ensure_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def extract_query(record: dict[str, Any], *, query_keys: tuple[str, ...]) -> str:
    candidates = extract_query_candidates(record, query_keys=query_keys)
    return next(iter(candidates), "")


def extract_query_candidates(record: dict[str, Any], *, query_keys: tuple[str, ...]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    for key in query_keys:
        if key not in record:
            continue
        normalized = normalize_query(record[key])
        add(normalized)
    for nested_key in ("extra_info", "metadata", "ground_truth"):
        value = parse_json_object(record.get(nested_key))
        if isinstance(value, dict):
            for nested in extract_query_candidates(value, query_keys=query_keys):
                add(nested)
    return candidates


def extract_raw_query_candidates(record: dict[str, Any], *, query_keys: tuple[str, ...]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    for key in query_keys:
        if key in record:
            add(record[key])
    for nested_key in ("extra_info", "metadata", "ground_truth"):
        value = parse_json_object(record.get(nested_key))
        if isinstance(value, dict):
            for nested in extract_raw_query_candidates(value, query_keys=query_keys):
                add(nested)
    return candidates


def parse_json_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_query(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty training label in spec: {spec}")
        return label, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["query", "holdout_path", "training_labels", "training_paths"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
