#!/usr/bin/env python3
"""Validate hard-gold rubric or query-pool data before BSC claims."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROVENANCE_KEYS = ("provenance", "source_url", "paper_url", "dataset_version", "license", "split")


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    report = validate_gold_records(
        records=records,
        target=args.target,
        min_records=args.min_records,
        min_rubrics_per_query=args.min_rubrics_per_query,
        require_data_source=not args.allow_missing_data_source,
        require_provenance=args.require_provenance,
        required_provenance=parse_required_provenance(args.required_provenance),
        required_data_sources=set(args.required_data_source or []),
        forbidden_data_sources=set(args.forbidden_data_source or []),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    label = "Gold data" if args.target == "gold" else "Query pool"
    print(f"{label} validation ok={report['ok']} report={args.output_json}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate hard-gold rubric or query-pool data for BSC evidence hygiene.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", choices=["gold", "query_pool"], default="gold")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-rubrics-per-query", type=int, default=1)
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument(
        "--required-provenance",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Require an exact provenance field value. Repeatable, e.g. --required-provenance paper_url=https://...",
    )
    parser.add_argument("--allow-missing-data-source", action="store_true")
    parser.add_argument("--required-data-source", action="append", default=[])
    parser.add_argument("--forbidden-data-source", action="append", default=[])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def validate_gold_records(
    records: list[dict[str, Any]],
    target: str = "gold",
    min_records: int = 1,
    min_rubrics_per_query: int = 1,
    require_data_source: bool = True,
    require_provenance: bool = False,
    required_provenance: dict[str, str] | None = None,
    required_data_sources: set[str] | None = None,
    forbidden_data_sources: set[str] | None = None,
) -> dict[str, Any]:
    if target not in {"gold", "query_pool"}:
        raise ValueError(f"Unsupported validation target: {target}")
    required_provenance = required_provenance or {}
    required_data_sources = required_data_sources or set()
    forbidden_data_sources = forbidden_data_sources or set()
    require_gold_rubrics = target == "gold"
    blockers: list[str] = []
    warnings: list[str] = []
    per_record: list[dict[str, Any]] = []
    query_counts = Counter(str(record.get("query", "")) for record in records if record.get("query"))
    duplicate_query_count = sum(1 for count in query_counts.values() if count > 1)
    duplicate_query_record_count = sum(count for count in query_counts.values() if count > 1)

    if len(records) < min_records:
        blockers.append(f"record count {len(records)} < min_records {min_records}")

    for idx, record in enumerate(records):
        issues = validate_one_record(
            idx=idx,
            record=record,
            min_rubrics_per_query=min_rubrics_per_query,
            require_gold_rubrics=require_gold_rubrics,
            require_data_source=require_data_source,
            require_provenance=require_provenance,
            required_provenance=required_provenance,
            required_data_sources=required_data_sources,
            forbidden_data_sources=forbidden_data_sources,
            duplicate_query=query_counts.get(str(record.get("query", "")), 0) > 1,
        )
        per_record.append(issues)
        blockers.extend(f"record {idx}: {item}" for item in issues["blockers"])
        warnings.extend(f"record {idx}: {item}" for item in issues["warnings"])

    return {
        "ok": not blockers,
        "target": target,
        "n_records": len(records),
        "n_unique_queries": len(query_counts),
        "n_duplicate_queries": duplicate_query_count,
        "n_duplicate_query_records": duplicate_query_record_count,
        "min_records": min_records,
        "min_rubrics_per_query": min_rubrics_per_query,
        "require_data_source": require_data_source,
        "require_provenance": require_provenance,
        "required_provenance": dict(sorted(required_provenance.items())),
        "required_data_sources": sorted(required_data_sources),
        "forbidden_data_sources": sorted(forbidden_data_sources),
        "blockers": blockers,
        "warnings": warnings,
        "per_record": per_record,
    }


def validate_one_record(
    idx: int,
    record: dict[str, Any],
    min_rubrics_per_query: int,
    require_gold_rubrics: bool,
    require_data_source: bool,
    require_provenance: bool,
    required_provenance: dict[str, str],
    required_data_sources: set[str],
    forbidden_data_sources: set[str],
    duplicate_query: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    query = record.get("query")
    rubrics = record.get("gold_rubrics")
    data_source = str(record.get("data_source", ""))

    if not isinstance(query, str) or not query.strip():
        blockers.append("missing non-empty query")
    if not require_gold_rubrics and rubrics is None:
        rubrics = []
    elif not isinstance(rubrics, list):
        blockers.append("gold_rubrics must be a list")
        rubrics = []
    else:
        cleaned = [item.strip() for item in rubrics if isinstance(item, str) and item.strip()]
        if len(cleaned) < min_rubrics_per_query:
            blockers.append(f"rubric count {len(cleaned)} < min_rubrics_per_query {min_rubrics_per_query}")
        if len(cleaned) != len(rubrics):
            blockers.append("gold_rubrics contains empty or non-string items")
        if len(set(cleaned)) != len(cleaned):
            warnings.append("gold_rubrics contains exact duplicates")
    if require_data_source and not data_source:
        blockers.append("missing data_source")
    if required_data_sources and data_source not in required_data_sources:
        blockers.append(
            f"data_source {data_source or '<missing>'} not in required_data_sources: "
            f"{', '.join(sorted(required_data_sources))}"
        )
    if data_source in forbidden_data_sources:
        blockers.append(f"forbidden data_source for hard-gold claim: {data_source}")
    if require_provenance and not has_provenance(record):
        blockers.append(f"missing provenance field; expected one of {', '.join(PROVENANCE_KEYS)}")
    for key, expected in required_provenance.items():
        actual = record.get(key)
        if actual in (None, ""):
            blockers.append(f"missing required provenance field: {key}")
        elif str(actual) != expected:
            blockers.append(f"provenance {key} mismatch: expected {expected!r}, got {str(actual)!r}")
    if duplicate_query:
        warnings.append("duplicate query appears in gold data")

    return {
        "idx": idx,
        "ok": not blockers,
        "n_rubrics": len(rubrics) if isinstance(rubrics, list) else 0,
        "data_source": data_source,
        "blockers": blockers,
        "warnings": warnings,
    }


def has_provenance(record: dict[str, Any]) -> bool:
    return any(record.get(key) not in (None, "") for key in PROVENANCE_KEYS)


def parse_required_provenance(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--required-provenance must use KEY=VALUE format: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"--required-provenance must include non-empty KEY and VALUE: {item!r}")
        out[key] = value
    return out


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


def to_markdown(report: dict[str, Any]) -> str:
    title = "Gold Data Validation" if report.get("target", "gold") == "gold" else "Query Pool Validation"
    lines = [
        f"# BlindSpot-RL {title}",
        "",
        f"- Overall ok: `{report['ok']}`",
        f"- Target: `{report.get('target', 'gold')}`",
        f"- Records: `{report['n_records']}`",
        f"- Unique queries: `{report.get('n_unique_queries', report['n_records'])}`",
        f"- Duplicate query groups: `{report.get('n_duplicate_queries', 0)}`",
        f"- Records in duplicate-query groups: `{report.get('n_duplicate_query_records', 0)}`",
        f"- Min records: `{report['min_records']}`",
        f"- Require provenance: `{report['require_provenance']}`",
        f"- Required provenance values: `{format_required_provenance(report.get('required_provenance', {}))}`",
        f"- Required data sources: `{', '.join(report.get('required_data_sources', [])) or 'none'}`",
        "",
        "## Blockers",
        "",
    ]
    if report.get("target", "gold") == "gold":
        lines.insert(6, f"- Min rubrics/query: `{report['min_rubrics_per_query']}`")
    lines.extend([f"- {item}" for item in report["blockers"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- none"])
    if report.get("target", "gold") == "gold":
        lines.extend(
            [
                "",
                "## Per-Record Summary",
                "",
                "| idx | ok | n_rubrics | data_source | blockers | warnings |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Per-Record Summary",
                "",
                "| idx | ok | data_source | blockers | warnings |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
    for row in report["per_record"]:
        if report.get("target", "gold") == "gold":
            cells = [
                str(row["idx"]),
                f"`{row['ok']}`",
                str(row["n_rubrics"]),
                f"`{row['data_source']}`",
                escape_md("; ".join(row["blockers"])),
                escape_md("; ".join(row["warnings"])),
            ]
        else:
            cells = [
                str(row["idx"]),
                f"`{row['ok']}`",
                f"`{row['data_source']}`",
                escape_md("; ".join(row["blockers"])),
                escape_md("; ".join(row["warnings"])),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_required_provenance(values: dict[str, str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in values.items())


if __name__ == "__main__":
    main()
