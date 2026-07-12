#!/usr/bin/env python3
"""Run BSC diagnostics across coverage/redundancy threshold grids."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import (  # noqa: E402
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    compute_metrics,
    parse_rubrics,
    safe_float,
)
from scripts.bsc_diagnose import build_record_verifier, load_records, pick_first, summarize  # noqa: E402


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    embedder = (
        TokenOverlapEmbedder()
        if args.embedding_model == "token-overlap"
        else SentenceTransformerEmbedder(args.embedding_model)
    )

    rows = []
    for coverage_tau in args.coverage_tau:
        for redundancy_tau in args.redundancy_tau:
            summary = run_one_setting(
                records=records,
                embedder=embedder,
                coverage_tau=coverage_tau,
                redundancy_tau=redundancy_tau,
                weights=(args.w_cov, args.w_valid, args.w_red),
            )
            rows.append(
                {
                    "coverage_tau": coverage_tau,
                    "redundancy_tau": redundancy_tau,
                    **{key: safe_float(value) if isinstance(value, float) else value for key, value in summary.items()},
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "threshold_sweep.csv", rows)
    (args.output_dir / "threshold_sweep.md").write_text(to_markdown(rows), encoding="utf-8")
    (args.output_dir / "threshold_sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} threshold settings to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep BSC coverage/redundancy thresholds.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", nargs="+", type=float, default=[0.70, 0.75, 0.80])
    parser.add_argument("--redundancy-tau", nargs="+", type=float, default=[0.80, 0.85, 0.90])
    parser.add_argument("--w-cov", default=1.0, type=float)
    parser.add_argument("--w-valid", default=0.5, type=float)
    parser.add_argument("--w-red", default=0.5, type=float)
    return parser.parse_args()


def run_one_setting(
    records: list[dict[str, Any]],
    embedder: TokenOverlapEmbedder | SentenceTransformerEmbedder,
    coverage_tau: float,
    redundancy_tau: float,
    weights: tuple[float, float, float],
) -> dict[str, Any]:
    per_item = []
    for idx, record in enumerate(records):
        prompt = pick_first(record, "prompt", "query", "instruction") or ""
        gold = parse_rubrics(
            pick_first(record, "gold_rubrics", "gold", "rubrics_gold"),
            dedupe=True,
        )
        response = pick_first(
            record,
            "response",
            "model_rubrics",
            "generated_rubrics",
            "prediction",
            "output",
        )
        verifier, verifier_source = build_record_verifier(record, response=response, idx=idx)
        metrics = compute_metrics(
            response=response,
            gold_rubrics=gold,
            prompt=prompt,
            verifier=verifier,
            embedder=embedder,
            coverage_tau=coverage_tau,
            redundancy_tau=redundancy_tau,
            weights=weights,
        )
        per_item.append({"idx": idx, "verifier_source": verifier_source, **metrics.as_dict()})
    return summarize(per_item)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["CovTau", "RedTau", "Cov", "Blind", "Red", "Hall", "Reward"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["coverage_tau"]),
                    fmt(row["redundancy_tau"]),
                    fmt(row["mean_coverage"]),
                    fmt(row["mean_blind"]),
                    fmt(row["mean_redundancy"]),
                    fmt(row["mean_hallucination"]),
                    fmt(row["mean_reward"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
