#!/usr/bin/env python3
"""Summarize completed human labels for a BSC audit pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VALID_LABELS = {"match", "non_match", "uncertain"}
DECISIVE_LABELS = {"match", "non_match"}


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    summary = summarize_rows(rows, input_path=args.input)
    summary["ok"] = gate_ok(
        summary,
        min_labeled=args.min_labeled,
        max_invalid_labels=args.max_invalid_labels,
        max_uncertain_rate=args.max_uncertain_rate,
        min_auto_matched_human_match_rate=args.min_auto_matched_human_match_rate,
        min_auto_unmatched_confirmation_rate=args.min_auto_unmatched_confirmation_rate,
    )
    summary["gate"] = {
        "min_labeled": args.min_labeled,
        "max_invalid_labels": args.max_invalid_labels,
        "max_uncertain_rate": args.max_uncertain_rate,
        "min_auto_matched_human_match_rate": args.min_auto_matched_human_match_rate,
        "min_auto_unmatched_confirmation_rate": args.min_auto_unmatched_confirmation_rate,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(summary), encoding="utf-8")
    print(f"Human audit labels ok={summary['ok']} labeled={summary['human_labels_completed']} report={args.output_json}")
    if args.strict and not summary["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize human labels for a BSC audit pack.")
    parser.add_argument("--input", required=True, type=Path, help="audit_items.csv produced by build_bsc_human_audit_pack.py")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-labeled", default=1, type=int)
    parser.add_argument("--max-invalid-labels", default=0, type=int)
    parser.add_argument("--max-uncertain-rate", default=1.0, type=float)
    parser.add_argument("--min-auto-matched-human-match-rate", default=0.0, type=float)
    parser.add_argument("--min-auto-unmatched-confirmation-rate", default=0.0, type=float)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def summarize_rows(rows: list[dict[str, str]], input_path: Path) -> dict[str, Any]:
    labels = [normalize_label(row.get("human_match_label", "")) for row in rows]
    valid_labeled = [label for label in labels if label in VALID_LABELS]
    invalid = [label for label in labels if label and label not in VALID_LABELS]
    empty_count = sum(label == "" for label in labels)
    uncertain_count = sum(label == "uncertain" for label in labels)

    auto_matched = [row for row in rows if row.get("match_status") == "matched"]
    auto_unmatched = [row for row in rows if row.get("match_status") == "unmatched"]
    auto_matched_labels = [normalize_label(row.get("human_match_label", "")) for row in auto_matched]
    auto_unmatched_labels = [normalize_label(row.get("human_match_label", "")) for row in auto_unmatched]

    matched_decisive = [label for label in auto_matched_labels if label in DECISIVE_LABELS]
    unmatched_decisive = [label for label in auto_unmatched_labels if label in DECISIVE_LABELS]
    matched_human_match = sum(label == "match" for label in matched_decisive)
    unmatched_human_match = sum(label == "match" for label in unmatched_decisive)
    unmatched_human_non_match = sum(label == "non_match" for label in unmatched_decisive)

    total = len(rows)
    completed = len(valid_labeled)
    uncertain_rate = uncertain_count / completed if completed else 0.0
    status = "human_audit_complete" if total > 0 and empty_count == 0 and not invalid else "human_audit_incomplete"
    return {
        "input": str(input_path),
        "status": status,
        "total_items": total,
        "human_labels_completed": completed,
        "unlabeled_items": empty_count,
        "invalid_label_count": len(invalid),
        "invalid_labels": sorted(set(invalid)),
        "uncertain_count": uncertain_count,
        "uncertain_rate": uncertain_rate,
        "auto_matched_items": len(auto_matched),
        "auto_unmatched_items": len(auto_unmatched),
        "auto_matched_decisive_labels": len(matched_decisive),
        "auto_unmatched_decisive_labels": len(unmatched_decisive),
        "auto_matched_human_match_rate": safe_rate(matched_human_match, len(matched_decisive)),
        "auto_unmatched_human_match_rate": safe_rate(unmatched_human_match, len(unmatched_decisive)),
        "auto_unmatched_confirmation_rate": safe_rate(unmatched_human_non_match, len(unmatched_decisive)),
    }


def gate_ok(
    summary: dict[str, Any],
    min_labeled: int,
    max_invalid_labels: int,
    max_uncertain_rate: float,
    min_auto_matched_human_match_rate: float = 0.0,
    min_auto_unmatched_confirmation_rate: float = 0.0,
) -> bool:
    return (
        int(summary["human_labels_completed"]) >= min_labeled
        and int(summary["invalid_label_count"]) <= max_invalid_labels
        and float(summary["uncertain_rate"]) <= max_uncertain_rate
        and optional_rate_at_least(summary.get("auto_matched_human_match_rate"), min_auto_matched_human_match_rate)
        and optional_rate_at_least(
            summary.get("auto_unmatched_confirmation_rate"),
            min_auto_unmatched_confirmation_rate,
        )
    )


def normalize_label(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def optional_rate_at_least(value: float | None, threshold: float) -> bool:
    if value is None:
        return threshold <= 0
    return value >= threshold


def to_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ("Status", summary["status"]),
        ("OK", summary["ok"]),
        ("Total items", summary["total_items"]),
        ("Human labels completed", summary["human_labels_completed"]),
        ("Invalid labels", summary["invalid_label_count"]),
        ("Uncertain rate", format_float(summary["uncertain_rate"])),
        ("Auto matched human-match rate", format_optional_float(summary["auto_matched_human_match_rate"])),
        ("Auto unmatched confirmation rate", format_optional_float(summary["auto_unmatched_confirmation_rate"])),
    ]
    lines = ["# BSC Human Audit Label Summary", "", "| Field | Value |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines) + "\n"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else format_float(value)


if __name__ == "__main__":
    main()
