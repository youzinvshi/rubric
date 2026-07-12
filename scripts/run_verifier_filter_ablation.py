#!/usr/bin/env python3
"""Compare proxy-gold quality with and without verifier filtering."""

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

from blindspot_rl.reward_bsc import SentenceTransformerEmbedder, TokenOverlapEmbedder, semantic_dedupe  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402
from scripts.normalize_dataset import load_records  # noqa: E402
from scripts.run_teacher_union_ablation import (  # noqa: E402
    build_gold_map,
    build_teacher_map,
    metric_row,
)


RAW_VARIANT = "no_verifier_filter"
FILTERED_VARIANT = "verifier_filtered"


def main() -> None:
    args = parse_args()
    raw_records = list(load_records(args.raw_teachers))
    filtered_records = list(load_records(args.filtered_teachers))
    gold_records = list(load_records(args.gold))
    embedder = TokenOverlapEmbedder() if args.embedding_model == "token-overlap" else SentenceTransformerEmbedder(args.embedding_model)

    rows = run_ablation(
        raw_teacher_records=raw_records,
        filtered_teacher_records=filtered_records,
        gold_records=gold_records,
        embedder=embedder,
        coverage_tau=args.coverage_tau,
        redundancy_tau=args.redundancy_tau,
        dedupe_tau=args.dedupe_tau,
        min_teachers=args.min_teachers,
    )
    if not rows:
        raise SystemExit("No raw/filtered teacher query overlap with gold records found.")

    summary = summarize(rows)
    attach_protocol_metadata(
        summary,
        raw_teachers=args.raw_teachers,
        filtered_teachers=args.filtered_teachers,
        gold=args.gold,
        embedding_model=args.embedding_model,
        coverage_tau=args.coverage_tau,
        redundancy_tau=args.redundancy_tau,
        dedupe_tau=args.dedupe_tau,
        min_teachers=args.min_teachers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "verifier_filter_per_item.csv", rows)
    write_csv(args.output_dir / "verifier_filter_ablation.csv", summary)
    write_markdown(args.output_dir / "verifier_filter_ablation.md", summary)
    (args.output_dir / "verifier_filter_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote verifier-filter ablation to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verifier-filter vs no-verifier proxy-gold BSC ablation.")
    parser.add_argument("--raw-teachers", required=True, type=Path, help="Unfiltered teacher evaluation-criteria outputs.")
    parser.add_argument("--filtered-teachers", required=True, type=Path, help="Verifier-filtered teacher evaluation-criteria outputs.")
    parser.add_argument("--gold", required=True, type=Path, help="Hard-gold evaluation-criteria records.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--redundancy-tau", default=0.85, type=float)
    parser.add_argument("--dedupe-tau", default=0.85, type=float)
    parser.add_argument("--min-teachers", default=2, type=int)
    return parser.parse_args()


def run_ablation(
    *,
    raw_teacher_records: list[dict[str, Any]],
    filtered_teacher_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    embedder: TokenOverlapEmbedder | SentenceTransformerEmbedder,
    coverage_tau: float,
    redundancy_tau: float,
    dedupe_tau: float,
    min_teachers: int = 2,
) -> list[dict[str, Any]]:
    gold_by_query = build_gold_map(gold_records)
    raw_by_query = build_teacher_map(raw_teacher_records)
    filtered_by_query = build_teacher_map(filtered_teacher_records)
    rows: list[dict[str, Any]] = []

    query_keys = sorted(set(gold_by_query).intersection(raw_by_query).intersection(filtered_by_query))
    for query_key in query_keys:
        gold = gold_by_query[query_key]
        for variant, teacher_map in [
            (RAW_VARIANT, raw_by_query[query_key]),
            (FILTERED_VARIANT, filtered_by_query[query_key]),
        ]:
            if len(teacher_map) < min_teachers:
                continue
            rubrics = union_teacher_rubrics(teacher_map)
            if not rubrics:
                continue
            rows.append(
                metric_row(
                    query=gold["query"],
                    variant=variant,
                    rubrics=semantic_dedupe(rubrics, tau=dedupe_tau, embedder=embedder),
                    gold_rubrics=gold["gold_rubrics"],
                    embedder=embedder,
                    coverage_tau=coverage_tau,
                    redundancy_tau=redundancy_tau,
                )
            )
    return rows


def union_teacher_rubrics(teacher_map: dict[str, list[str]]) -> list[str]:
    rubrics: list[str] = []
    for teacher_rubrics in teacher_map.values():
        rubrics.extend(teacher_rubrics)
    return rubrics


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for variant in [RAW_VARIANT, FILTERED_VARIANT]:
        items = grouped.get(variant, [])
        summary: dict[str, Any] = {"variant": variant, "n": len(items)}
        for key in ["coverage", "blind", "redundancy", "validity", "hallucination", "reward", "n_gen"]:
            values = [float(item[key]) for item in items]
            summary[f"mean_{key}"] = sum(values) / max(len(values), 1)
        summaries.append(summary)

    raw = next(row for row in summaries if row["variant"] == RAW_VARIANT)
    filtered = next(row for row in summaries if row["variant"] == FILTERED_VARIANT)
    filtered["coverage_delta_vs_no_verifier"] = filtered["mean_coverage"] - raw["mean_coverage"]
    filtered["hallucination_delta_vs_no_verifier"] = filtered["mean_hallucination"] - raw["mean_hallucination"]
    raw["coverage_delta_vs_no_verifier"] = 0.0
    raw["hallucination_delta_vs_no_verifier"] = 0.0
    return summaries


def attach_protocol_metadata(
    summaries: list[dict[str, Any]],
    *,
    raw_teachers: Path,
    filtered_teachers: Path,
    gold: Path,
    embedding_model: str,
    coverage_tau: float,
    redundancy_tau: float,
    dedupe_tau: float,
    min_teachers: int,
) -> None:
    for summary in summaries:
        summary["ablation_family"] = "verifier_filter"
        summary["raw_teachers"] = str(raw_teachers)
        summary["raw_teachers_sha256"] = file_sha256(raw_teachers)
        summary["filtered_teachers"] = str(filtered_teachers)
        summary["filtered_teachers_sha256"] = file_sha256(filtered_teachers)
        summary["gold"] = str(gold)
        summary["gold_sha256"] = file_sha256(gold)
        summary["embedding_model"] = embedding_model
        summary["coverage_tau"] = coverage_tau
        summary["redundancy_tau"] = redundancy_tau
        summary["dedupe_tau"] = dedupe_tau
        summary["min_teachers"] = min_teachers


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Variant | Cov | Blind | Red | Hall | Reward | N Gen | Cov Delta | Hall Delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    fmt(row["mean_coverage"]),
                    fmt(row["mean_blind"]),
                    fmt(row["mean_redundancy"]),
                    fmt(row["mean_hallucination"]),
                    fmt(row["mean_reward"]),
                    fmt(row["mean_n_gen"]),
                    fmt(row["coverage_delta_vs_no_verifier"]),
                    fmt(row["hallucination_delta_vs_no_verifier"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
