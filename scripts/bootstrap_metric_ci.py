#!/usr/bin/env python3
"""Bootstrap confidence intervals for BlindSpot-RL per-item metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ["coverage", "blind", "redundancy", "hallucination", "reward", "correct", "tie", "margin"]


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    metrics = args.metric or [metric for metric in DEFAULT_METRICS if metric in rows[0]]
    if not metrics:
        raise SystemExit("No metric columns found. Use --metric to specify columns.")
    report = build_ci_report(
        rows=rows,
        metrics=metrics,
        n_boot=args.n_boot,
        seed=args.seed,
        confidence=args.confidence,
        input_path=args.input,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, report["metrics"])
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Bootstrap CI report metrics={len(report['metrics'])} n={report['n']} output={args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals for per-item metrics.")
    parser.add_argument("--input", required=True, type=Path, help="Per-item CSV.")
    parser.add_argument("--metric", action="append", help="Metric column to summarize. Repeatable.")
    parser.add_argument("--n-boot", default=1000, type=int)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--confidence", default=0.95, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def build_ci_report(
    rows: list[dict[str, str]],
    metrics: list[str],
    n_boot: int = 1000,
    seed: int = 13,
    confidence: float = 0.95,
    input_path: Path | None = None,
) -> dict[str, Any]:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = random.Random(seed)
    metric_reports = []
    for metric in metrics:
        values = [parse_number(row.get(metric)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            metric_reports.append({"metric": metric, "n": 0, "status": "missing"})
            continue
        boot_means = bootstrap_means(values, n_boot=n_boot, rng=rng)
        lower, upper = percentile_interval(boot_means, confidence=confidence)
        metric_reports.append(
            {
                "metric": metric,
                "n": len(values),
                "status": "pass",
                "mean": mean(values),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence": confidence,
                "n_boot": n_boot,
                "seed": seed,
            }
        )
    report = {
        "n": len(rows),
        "metrics": metric_reports,
        "n_boot": n_boot,
        "seed": seed,
        "confidence": confidence,
    }
    if input_path is not None:
        report["input"] = str(input_path)
        report["input_sha256"] = file_sha256(input_path)
    return report


def bootstrap_means(values: list[float], n_boot: int, rng: random.Random) -> list[float]:
    n = len(values)
    out = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        out.append(mean(sample))
    return sorted(out)


def percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return percentile(values, alpha / 2.0), percentile(values, 1.0 - alpha / 2.0)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip()
    if text.lower() == "true":
        return 1.0
    if text.lower() == "false":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["metric", "n", "status", "mean", "ci_lower", "ci_upper", "confidence", "n_boot", "seed"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bootstrap Metric Confidence Intervals",
        "",
        f"- Rows: `{report['n']}`",
        f"- Bootstrap samples: `{report['n_boot']}`",
        f"- Confidence: `{report['confidence']}`",
        f"- Seed: `{report['seed']}`",
        "",
        "| Metric | N | Mean | CI Lower | CI Upper | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["metrics"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    str(row.get("n", 0)),
                    fmt(row.get("mean")),
                    fmt(row.get("ci_lower")),
                    fmt(row.get("ci_upper")),
                    f"`{row.get('status', '')}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
