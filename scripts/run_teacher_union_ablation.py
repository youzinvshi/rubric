#!/usr/bin/env python3
"""Compare single-teacher criteria against multi-teacher criteria unions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    semantic_dedupe,
)
from scripts.normalize_dataset import load_records  # noqa: E402


UNION_VARIANT = "multi_teacher_union"


def main() -> None:
    args = parse_args()
    teacher_records = list(load_records(args.teachers))
    gold_records = list(load_records(args.gold))
    embedder = (
        TokenOverlapEmbedder()
        if args.embedding_model == "token-overlap"
        else SentenceTransformerEmbedder(args.embedding_model)
    )

    rows = run_ablation(
        teacher_records=teacher_records,
        gold_records=gold_records,
        embedder=embedder,
        coverage_tau=args.coverage_tau,
        redundancy_tau=args.redundancy_tau,
        dedupe_tau=args.dedupe_tau,
        min_teachers=args.min_teachers,
    )
    if not rows:
        raise SystemExit("No teacher/gold query overlap found.")

    summaries = summarize_by_variant(rows)
    for summary in summaries:
        summary.update(
            {
                "teachers": str(args.teachers),
                "gold": str(args.gold),
                "embedding_model": args.embedding_model,
                "coverage_tau": args.coverage_tau,
                "redundancy_tau": args.redundancy_tau,
                "dedupe_tau": args.dedupe_tau,
                "min_teachers": args.min_teachers,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "teacher_union_per_item.csv", rows)
    write_csv(args.output_dir / "teacher_union_ablation.csv", summaries)
    write_markdown(args.output_dir / "teacher_union_ablation.md", summaries)
    (args.output_dir / "teacher_union_ablation.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote teacher-union ablation to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-teacher vs multi-teacher union BSC ablation.")
    parser.add_argument("--teachers", required=True, type=Path, help="Teacher generation JSONL/JSON/parquet.")
    parser.add_argument("--gold", required=True, type=Path, help="Gold rubric JSONL/JSON/parquet.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--redundancy-tau", default=0.85, type=float)
    parser.add_argument("--dedupe-tau", default=0.85, type=float)
    parser.add_argument("--min-teachers", default=2, type=int)
    return parser.parse_args()


def run_ablation(
    teacher_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    embedder: TokenOverlapEmbedder | SentenceTransformerEmbedder,
    coverage_tau: float,
    redundancy_tau: float,
    dedupe_tau: float,
    min_teachers: int = 2,
) -> list[dict[str, Any]]:
    gold_by_query = build_gold_map(gold_records)
    teachers_by_query = build_teacher_map(teacher_records)
    rows: list[dict[str, Any]] = []

    for query_key, teacher_map in sorted(teachers_by_query.items()):
        gold = gold_by_query.get(query_key)
        if not gold:
            continue
        if len(teacher_map) < min_teachers:
            continue

        for teacher, rubrics in sorted(teacher_map.items()):
            rows.append(
                metric_row(
                    query=gold["query"],
                    variant=teacher,
                    rubrics=semantic_dedupe(rubrics, tau=dedupe_tau, embedder=embedder),
                    gold_rubrics=gold["gold_rubrics"],
                    embedder=embedder,
                    coverage_tau=coverage_tau,
                    redundancy_tau=redundancy_tau,
                )
            )

        union: list[str] = []
        for rubrics in teacher_map.values():
            union.extend(rubrics)
        rows.append(
            metric_row(
                query=gold["query"],
                variant=UNION_VARIANT,
                rubrics=semantic_dedupe(union, tau=dedupe_tau, embedder=embedder),
                gold_rubrics=gold["gold_rubrics"],
                embedder=embedder,
                coverage_tau=coverage_tau,
                redundancy_tau=redundancy_tau,
            )
        )
    return rows


def build_gold_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for record in records:
        query = pick_first(record, "query", "prompt", "instruction")
        gold = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold", "rubrics"), dedupe=True)
        if query and gold:
            out[make_key(query)] = {"query": str(query), "gold_rubrics": gold}
    return out


def build_teacher_map(records: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        query = pick_first(record, "query", "prompt", "instruction")
        teacher = str(record.get("teacher") or record.get("method") or record.get("model") or "")
        rubrics = parse_rubrics(
            pick_first(record, "rubrics", "rubric_list", "response", "output", "prediction"),
            dedupe=False,
        )
        if query and teacher and rubrics:
            out[make_key(query)][teacher].extend(rubrics)
    return {query: dict(teacher_map) for query, teacher_map in out.items()}


def metric_row(
    query: str,
    variant: str,
    rubrics: list[str],
    gold_rubrics: list[str],
    embedder: TokenOverlapEmbedder | SentenceTransformerEmbedder,
    coverage_tau: float,
    redundancy_tau: float,
) -> dict[str, Any]:
    metrics = compute_metrics(
        response=rubrics,
        gold_rubrics=gold_rubrics,
        prompt=query,
        verifier=None,
        embedder=embedder,
        coverage_tau=coverage_tau,
        redundancy_tau=redundancy_tau,
    )
    return {
        "query": query,
        "variant": variant,
        **{key: safe_float(value) if isinstance(value, float) else value for key, value in metrics.as_dict().items()},
    }


def summarize_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    summaries = []
    for variant, items in sorted(grouped.items(), key=lambda item: (item[0] != UNION_VARIANT, item[0])):
        summary: dict[str, Any] = {"variant": variant, "n": len(items)}
        for key in ["coverage", "blind", "redundancy", "validity", "hallucination", "reward", "n_gen"]:
            values = [float(item[key]) for item in items]
            summary[f"mean_{key}"] = sum(values) / max(len(values), 1)
        summaries.append(summary)

    single_rows = [row for row in summaries if row["variant"] != UNION_VARIANT]
    best_single_row = max(single_rows, key=lambda row: row["mean_coverage"], default=None)
    best_single = best_single_row["mean_coverage"] if best_single_row is not None else None
    for row in summaries:
        row["coverage_gain_vs_best_single"] = (
            row["mean_coverage"] - best_single
            if row["variant"] == UNION_VARIANT and best_single is not None
            else ""
        )
        row["n_single_teacher_variants"] = len(single_rows) if row["variant"] == UNION_VARIANT else ""
        row["best_single_variant"] = (
            best_single_row["variant"]
            if row["variant"] == UNION_VARIANT and best_single_row is not None
            else ""
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Variant | Cov | Blind | Red | Hall | Reward | N Gen | Gain vs Best Single |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    fmt(row["coverage_gain_vs_best_single"]) if row["coverage_gain_vs_best_single"] != "" else "",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def make_key(value: Any) -> str:
    return " ".join(str(value).strip().split())


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
