#!/usr/bin/env python3
"""Check whether a downstream benchmark can be normalized for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.normalize_dataset import load_records, normalize_records  # noqa: E402


def main() -> None:
    args = parse_args()
    report = build_schema_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(
        f"Schema contract ok={report['ok']} selected_target={report.get('selected_target') or ''} "
        f"report={args.output_json}"
    )
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check downstream benchmark schema compatibility.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", action="append", choices=["preference", "multicandidate"], default=[])
    parser.add_argument("--data-source", default="unknown")
    parser.add_argument("--query-key")
    parser.add_argument("--chosen-key")
    parser.add_argument("--rejected-key")
    parser.add_argument("--candidates-key")
    parser.add_argument("--label-key")
    parser.add_argument("--min-records", default=1, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_schema_report(args: argparse.Namespace) -> dict[str, Any]:
    targets = args.target or ["preference", "multicandidate"]
    if not args.input.exists():
        return {
            "ok": False,
            "input": str(args.input),
            "n_records": 0,
            "selected_target": None,
            "targets": [
                {
                    "target": target,
                    "ok": False,
                    "normalized_records": 0,
                    "min_records": args.min_records,
                    "sample_keys": [],
                }
                for target in targets
            ],
            "blockers": [f"Input file is missing: {args.input}"],
            "warnings": [],
        }

    records = list(load_records(args.input))
    target_reports = []
    selected_target = None
    for target in targets:
        rows = normalize_records(records, namespace_for_target(args, target))
        ok = len(rows) >= args.min_records
        if ok and selected_target is None:
            selected_target = target
        target_reports.append(
            {
                "target": target,
                "ok": ok,
                "normalized_records": len(rows),
                "min_records": args.min_records,
                "sample_keys": sorted(rows[0].keys()) if rows else [],
            }
        )

    blockers = []
    warnings = []
    if not records:
        blockers.append(f"No records found in {args.input}")
    if selected_target is None:
        blockers.append(
            "No downstream schema target met the minimum normalized-record threshold. "
            "Provide explicit query/response/label field mappings before making a claim."
        )
    if len(target_reports) > 1 and selected_target:
        warnings.append(f"Selected first compatible target: {selected_target}")
    return {
        "ok": selected_target is not None and not blockers,
        "input": str(args.input),
        "n_records": len(records),
        "selected_target": selected_target,
        "targets": target_reports,
        "blockers": blockers,
        "warnings": warnings,
    }


def namespace_for_target(args: argparse.Namespace, target: str) -> Namespace:
    return Namespace(
        target=target,
        data_source=args.data_source,
        query_key=args.query_key,
        gold_key=None,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        candidates_key=args.candidates_key,
        label_key=args.label_key,
        limit=args.limit,
        dedupe_query=False,
    )


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Downstream Schema Contract",
        "",
        f"- OK: `{report['ok']}`",
        f"- Input: `{report['input']}`",
        f"- Records: `{report['n_records']}`",
        f"- Selected target: `{report.get('selected_target') or ''}`",
        "",
        "| Target | OK | Normalized Records | Min Records |",
        "|---|---:|---:|---:|",
    ]
    for item in report.get("targets", []):
        lines.append(
            f"| {item['target']} | {item['ok']} | {item['normalized_records']} | {item['min_records']} |"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report["warnings"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
