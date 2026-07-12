#!/usr/bin/env python3
"""Summarize downstream policy-RLVR prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CORRECT_KEYS = ("correct", "is_correct", "passed", "pass", "success", "accepted")
DEFAULT_SCORE_KEYS = ("score", "reward", "judge_score", "metric_score", "eval_score", "accuracy")


def main() -> None:
    args = parse_args()
    rows = list(load_records(args.input))
    report, per_item = build_report(
        rows=rows,
        input_path=args.input,
        correct_keys=tuple(args.correct_key or DEFAULT_CORRECT_KEYS),
        score_keys=tuple(args.score_key or DEFAULT_SCORE_KEYS),
        min_score=args.min_score,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    if args.per_item_csv:
        write_csv(args.per_item_csv, per_item)
    print_summary(report)
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize policy-RLVR prediction outputs.")
    parser.add_argument("--input", required=True, type=Path, help="Policy predictions JSONL/JSON/parquet.")
    parser.add_argument("--output", required=True, type=Path, help="Summary JSON output.")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown summary output.")
    parser.add_argument("--per-item-csv", type=Path, help="Optional per-item CSV output.")
    parser.add_argument(
        "--correct-key",
        action="append",
        help="Boolean correctness field to inspect. Repeatable; defaults cover common names.",
    )
    parser.add_argument(
        "--score-key",
        action="append",
        help="Numeric score field to inspect. Repeatable; defaults cover common names.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="If set, numeric scores at or above this threshold count as correct.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when no evaluable records are found.")
    return parser.parse_args()


def build_report(
    rows: list[dict[str, Any]],
    input_path: Path,
    correct_keys: tuple[str, ...] = DEFAULT_CORRECT_KEYS,
    score_keys: tuple[str, ...] = DEFAULT_SCORE_KEYS,
    min_score: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_item: list[dict[str, Any]] = []
    correctness: list[bool] = []
    scores: list[float] = []
    skipped = 0

    for idx, record in enumerate(rows):
        correct, correct_key = pick_correct(record, correct_keys)
        score, score_key = pick_score(record, score_keys)
        if correct is None and score is not None and min_score is not None:
            correct = score >= min_score
            correct_key = f"{score_key}>={min_score:g}" if score_key else f"score>={min_score:g}"
        if correct is None and score is None:
            skipped += 1
            continue
        if correct is not None:
            correctness.append(correct)
        if score is not None:
            scores.append(score)
        per_item.append(
            {
                "idx": idx,
                "query": pick_first(record, "query", "prompt", "instruction"),
                "data_source": record.get("data_source", ""),
                "correct": correct if correct is not None else "",
                "correct_key": correct_key or "",
                "score": score if score is not None else "",
                "score_key": score_key or "",
            }
        )

    blockers = []
    if not rows:
        blockers.append(f"No policy prediction records found in {input_path}")
    if rows and not per_item:
        blockers.append(
            "No evaluable policy records found; add a correctness field "
            f"({', '.join(correct_keys)}) or score field ({', '.join(score_keys)})."
        )

    accuracy = sum(1 for value in correctness if value) / len(correctness) if correctness else None
    mean_score = sum(scores) / len(scores) if scores else None
    report: dict[str, Any] = {
        "ok": not blockers,
        "input": str(input_path),
        "n_records": len(rows),
        "n_evaluable": len(per_item),
        "n_correctness": len(correctness),
        "n_scores": len(scores),
        "skipped": skipped,
        "accuracy": accuracy,
        "mean_score": mean_score,
        "min_score": min_score,
        "blockers": blockers,
        "warnings": [],
    }
    if accuracy is None and mean_score is not None:
        report["warnings"].append("Only score metrics were found; no accuracy/correctness field was available.")
    return report, per_item


def pick_correct(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool | None, str | None]:
    for key in keys:
        if key not in record:
            continue
        value = parse_bool(record[key])
        if value is not None:
            return value, key
    return None, None


def pick_score(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    for key in keys:
        if key not in record:
            continue
        value = parse_number(record[key])
        if value is not None:
            return value, key
    return None, None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "pass", "passed", "correct", "success"}:
        return True
    if text in {"false", "no", "n", "0", "fail", "failed", "incorrect", "error"}:
        return False
    return None


def parse_number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


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


def unwrap_records(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["records", "data", "items", "examples", "rows", "predictions"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return [data]


def ensure_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}


def pick_first(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "query", "data_source", "correct", "correct_key", "score", "score_key"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Policy RLVR Evaluation Report",
        "",
        f"- Status: {'pass' if report['ok'] else 'blocked'}",
        f"- Input: `{report['input']}`",
        f"- Records: {report['n_records']}",
        f"- Evaluable: {report['n_evaluable']}",
        f"- Accuracy records: {report['n_correctness']}",
        f"- Score records: {report['n_scores']}",
        f"- Accuracy: {format_optional(report['accuracy'])}",
        f"- Mean score: {format_optional(report['mean_score'])}",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report["warnings"])
    return "\n".join(lines) + "\n"


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary(report: dict[str, Any]) -> None:
    print("Policy RLVR evaluation summary")
    print(
        f"ok={report['ok']} n={report['n_records']} "
        f"evaluable={report['n_evaluable']} skipped={report['skipped']}"
    )
    print(f"Accuracy={format_optional(report['accuracy'])} MeanScore={format_optional(report['mean_score'])}")


if __name__ == "__main__":
    main()
