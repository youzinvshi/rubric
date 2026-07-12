#!/usr/bin/env python3
"""Merge BSC and downstream summaries into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DOWNSTREAM_NOT_ELIGIBLE_ERROR = (
    "downstream summary is not paper_claim_eligible; use API scorer with bound "
    "provider and budget report"
)


def main() -> None:
    args = parse_args()
    rows: dict[str, dict[str, Any]] = {}
    for spec in args.bsc:
        method, path = parse_spec(spec)
        summary = read_json(path)
        row = rows.setdefault(method, {"method": method})
        if is_load_error(summary):
            row.update({"bsc_status": "blocked", "bsc_error": summary["_load_error"]})
            continue
        rows.setdefault(method, {"method": method}).update(
            {
                "bsc_status": "pass",
                "bsc_n": summary.get("n", 0),
                "cov": summary.get("mean_coverage", 0.0),
                "blind": summary.get("mean_blind", 0.0),
                "red": summary.get("mean_redundancy", 0.0),
                "hall": summary.get("mean_hallucination", 0.0),
                "reward": summary.get("mean_reward", 0.0),
                "mean_n_gen": summary.get("mean_n_gen", ""),
                "gen_to_gold_ratio": summary.get("gen_to_gold_ratio", ""),
                "coverage_per_generated_criterion": summary.get("coverage_per_generated_criterion", ""),
            }
        )
    for spec in args.downstream:
        method, path = parse_spec(spec)
        summary = read_json(path)
        row = rows.setdefault(method, {"method": method})
        if is_load_error(summary):
            row.update({"downstream_status": "blocked", "downstream_error": summary["_load_error"]})
            continue
        apply_downstream_summary(row, summary)
    for spec in args.bsc_ci:
        method, path = parse_spec(spec)
        report = read_json(path)
        if is_load_error(report):
            rows.setdefault(method, {"method": method}).update(
                {"bsc_ci_status": "blocked", "bsc_ci_error": report["_load_error"]}
            )
            continue
        rows.setdefault(method, {"method": method})["bsc_ci_status"] = "pass"
        apply_ci_report(
            rows.setdefault(method, {"method": method}),
            report,
            {
                "coverage": "cov",
                "blind": "blind",
                "redundancy": "red",
                "hallucination": "hall",
                "reward": "reward",
            },
        )
    for spec in args.downstream_ci:
        method, path = parse_spec(spec)
        report = read_json(path)
        if is_load_error(report):
            rows.setdefault(method, {"method": method}).update(
                {"downstream_ci_status": "blocked", "downstream_ci_error": report["_load_error"]}
            )
            continue
        row = rows.setdefault(method, {"method": method})
        if row.get("downstream_status") != "pass":
            row.update(
                {
                    "downstream_ci_status": "blocked",
                    "downstream_ci_error": (
                        "downstream CI cannot be paper-facing without an eligible downstream summary"
                    ),
                }
            )
            continue
        row["downstream_ci_status"] = "pass"
        apply_ci_report(
            row,
            report,
            {
                "correct": "accuracy",
                "tie": "tie_rate",
                "margin": "mean_margin",
            },
        )

    ordered_rows = [rows[name] for name in sorted(rows)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, ordered_rows)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(ordered_rows), encoding="utf-8")
    print(f"Wrote {len(ordered_rows)} methods to {args.output_csv}")
    if args.output_md:
        print(f"Wrote markdown table to {args.output_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize BlindSpot-RL experiment outputs.")
    parser.add_argument(
        "--bsc",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="BSC summary JSON, e.g. base=outputs/base_bsc/summary.json",
    )
    parser.add_argument(
        "--downstream",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Downstream summary JSON, e.g. base=outputs/base_downstream/summary.json",
    )
    parser.add_argument(
        "--bsc-ci",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="BSC bootstrap CI JSON, e.g. base=outputs/base_bsc_ci/bootstrap_ci.json",
    )
    parser.add_argument(
        "--downstream-ci",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Downstream bootstrap CI JSON, e.g. base=outputs/base_downstream_ci/bootstrap_ci.json",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected METHOD=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    if not name:
        raise ValueError(f"Missing method name in spec: {spec}")
    return name, Path(raw_path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": f"{path}: not valid JSON at line {exc.lineno} column {exc.colno}"}
    except OSError as exc:
        return {"_load_error": f"{path}: {exc}"}
    if not isinstance(data, dict):
        return {"_load_error": f"{path}: summary JSON must be an object"}
    return data


def is_load_error(data: Any) -> bool:
    return isinstance(data, dict) and bool(data.get("_load_error"))


def apply_ci_report(row: dict[str, Any], report: dict[str, Any], metric_map: dict[str, str]) -> None:
    for metric in report.get("metrics", []):
        if metric.get("status") != "pass":
            continue
        source_name = str(metric.get("metric", ""))
        target_name = metric_map.get(source_name)
        if not target_name:
            continue
        row[f"{target_name}_ci_lower"] = metric.get("ci_lower", "")
        row[f"{target_name}_ci_upper"] = metric.get("ci_upper", "")
        row[f"{target_name}_ci"] = format_with_ci(
            mean_value=row.get(target_name, metric.get("mean", "")),
            lower=metric.get("ci_lower", ""),
            upper=metric.get("ci_upper", ""),
        )


def apply_downstream_summary(row: dict[str, Any], summary: dict[str, Any]) -> None:
    row.update(
        {
            "downstream_paper_claim_eligible": fmt_bool(summary.get("paper_claim_eligible")),
            "downstream_scorer": summary.get("scorer", ""),
            "downstream_scorer_provider": summary.get("scorer_provider", ""),
            "downstream_budget_report": summary.get("budget_report", ""),
            "downstream_benchmark_format": summary.get("benchmark_format", ""),
        }
    )
    if summary.get("paper_claim_eligible") is not True:
        row.update(
            {
                "downstream_status": "not_paper_eligible",
                "downstream_error": DOWNSTREAM_NOT_ELIGIBLE_ERROR,
                "downstream_n": "",
                "accuracy": "",
                "tie_rate": "",
                "mean_margin": "",
            }
        )
        return
    row.update(
        {
            "downstream_status": "pass",
            "downstream_error": "",
            "downstream_n": summary.get("n", 0),
            "accuracy": summary.get("accuracy", 0.0),
            "tie_rate": summary.get("tie_rate", 0.0),
            "mean_margin": summary.get("mean_margin", 0.0),
        }
    )


def format_with_ci(mean_value: Any, lower: Any, upper: Any) -> str:
    return f"{fmt(mean_value)} [{fmt(lower)}, {fmt(upper)}]"


def fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "bsc_status",
        "bsc_error",
        "bsc_n",
        "cov",
        "cov_ci_lower",
        "cov_ci_upper",
        "cov_ci",
        "blind",
        "blind_ci_lower",
        "blind_ci_upper",
        "blind_ci",
        "red",
        "red_ci_lower",
        "red_ci_upper",
        "red_ci",
        "hall",
        "hall_ci_lower",
        "hall_ci_upper",
        "hall_ci",
        "reward",
        "reward_ci_lower",
        "reward_ci_upper",
        "reward_ci",
        "mean_n_gen",
        "gen_to_gold_ratio",
        "coverage_per_generated_criterion",
        "bsc_ci_status",
        "bsc_ci_error",
        "downstream_status",
        "downstream_error",
        "downstream_paper_claim_eligible",
        "downstream_scorer",
        "downstream_scorer_provider",
        "downstream_budget_report",
        "downstream_benchmark_format",
        "downstream_n",
        "accuracy",
        "accuracy_ci_lower",
        "accuracy_ci_upper",
        "accuracy_ci",
        "tie_rate",
        "tie_rate_ci_lower",
        "tie_rate_ci_upper",
        "tie_rate_ci",
        "mean_margin",
        "mean_margin_ci_lower",
        "mean_margin_ci_upper",
        "mean_margin_ci",
        "downstream_ci_status",
        "downstream_ci_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["Method", "BSC", "Cov↑", "Blind↓", "Red↓", "Hall↓", "N Gen", "Cov/Gen↑", "Downstream", "Acc↑", "Tie↓"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    status_display(row, "bsc"),
                    display_metric(row, "cov"),
                    display_metric(row, "blind"),
                    display_metric(row, "red"),
                    display_metric(row, "hall"),
                    display_metric(row, "mean_n_gen"),
                    display_metric(row, "coverage_per_generated_criterion"),
                    status_display(row, "downstream"),
                    display_metric(row, "accuracy"),
                    display_metric(row, "tie_rate"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def display_metric(row: dict[str, Any], key: str) -> str:
    return str(row.get(f"{key}_ci") or fmt(row.get(key)))


def status_display(row: dict[str, Any], prefix: str) -> str:
    status = str(row.get(f"{prefix}_status", ""))
    error = row.get(f"{prefix}_error")
    if error:
        return f"{status}: {error}"
    return status


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
