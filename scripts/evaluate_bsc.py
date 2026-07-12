#!/usr/bin/env python3
"""Evaluate BlindSpot Coverage (BSC) on a hard-gold test split.

This script is kept as the two-file hard-gold evaluator: one JSONL contains
gold evaluation dimensions and another JSONL contains model predictions. Its output schema is
aligned with ``scripts/bsc_diagnose.py`` so that BSC evidence uses the same
threshold, embedding-model, reward-weight, and mean-metric keys throughout the
paper pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from blindspot_rl.reward_bsc import (
    DEFAULT_COVERAGE_TAU,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_REDUNDANCY_TAU,
    DEFAULT_WEIGHTS,
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    compute_metrics,
    parse_rubrics,
    safe_float,
)
from scripts.bsc_diagnose import build_record_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BSC on hard-gold.")
    parser.add_argument("--test-split", required=True, type=Path, help="The hard-gold JSONL (e.g. test_main).")
    parser.add_argument("--predictions", required=True, type=Path, help="Model-generated criteria JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output metrics JSON.")
    parser.add_argument("--per-item-output", type=Path, default=None, help="Optional per-item diagnostics CSV.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--coverage-tau", type=float, default=DEFAULT_COVERAGE_TAU)
    parser.add_argument("--redundancy-tau", type=float, default=DEFAULT_REDUNDANCY_TAU)
    parser.add_argument("--tau", type=float, default=None, help="Deprecated alias for --coverage-tau.")
    parser.add_argument("--w-cov", type=float, default=DEFAULT_WEIGHTS[0])
    parser.add_argument("--w-valid", type=float, default=DEFAULT_WEIGHTS[1])
    parser.add_argument("--w-red", type=float, default=DEFAULT_WEIGHTS[2])
    args = parser.parse_args()

    coverage_tau = args.coverage_tau if args.tau is None else args.tau
    summary, per_item = evaluate_bsc(
        test_split=args.test_split,
        predictions=args.predictions,
        embedding_model=args.embedding_model,
        coverage_tau=coverage_tau,
        redundancy_tau=args.redundancy_tau,
        weights=(args.w_cov, args.w_valid, args.w_red),
    )

    if args.per_item_output is not None:
        write_per_item(per_item, args.per_item_output)
        summary["per_item_output"] = str(args.per_item_output)
        summary["per_item_sha256"] = file_sha256(args.per_item_output)
        summary["per_item_rows"] = len(per_item)
    else:
        summary["per_item_output"] = ""
        summary["per_item_sha256"] = ""
        summary["per_item_rows"] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluating {summary['n']} hard-gold matched queries...")
    print(f"Mean Coverage (Cov): {summary['mean_coverage']:.4f}")
    print(f"Mean Blindspot (Blind): {summary['mean_blind']:.4f}")
    print(f"Results saved to {args.output}")


def evaluate_bsc(
    *,
    test_split: Path,
    predictions: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    coverage_tau: float = DEFAULT_COVERAGE_TAU,
    redundancy_tau: float = DEFAULT_REDUNDANCY_TAU,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_map = load_gold_map(test_split)
    embedder = build_embedder(embedding_model)

    rows: list[dict[str, Any]] = []
    matched_queries = 0
    for idx, record in enumerate(load_jsonl(predictions)):
        query = pick_first(record, "query", "prompt", "input", "question", "instruction")
        if query not in gold_map:
            continue
        matched_queries += 1
        gold = gold_map[query]
        if not gold:
            continue
        response = pick_first(record, "rubrics", "pred_rubrics", "generated_rubrics", "model_rubrics", "prediction", "response", "output")
        verifier, verifier_source = build_record_verifier(record, response=response, idx=idx)
        metrics = compute_metrics(
            response=response,
            gold_rubrics=gold,
            prompt=str(query),
            verifier=verifier,
            embedder=embedder,
            coverage_tau=coverage_tau,
            redundancy_tau=redundancy_tau,
            weights=weights,
        )
        rows.append(
            {
                "query": query,
                "data_source": str(record.get("data_source") or "rubricbench"),
                "verifier_source": verifier_source,
                **{key: safe_float(value) if isinstance(value, float) else value for key, value in metrics.as_dict().items()},
            }
        )

    if not rows:
        raise ValueError("No evaluable matching queries found between predictions and gold split.")

    summary = summarize_rows(
        rows,
        test_split=test_split,
        predictions=predictions,
        embedding_model=embedding_model,
        coverage_tau=coverage_tau,
        redundancy_tau=redundancy_tau,
        weights=weights,
        matched_queries=matched_queries,
    )
    return summary, rows


def load_gold_map(path: Path) -> dict[str, list[str]]:
    gold_map: dict[str, list[str]] = {}
    for record in load_jsonl(path):
        query = pick_first(record, "query", "prompt", "input", "question", "instruction")
        if query is None:
            continue
        gold_map[str(query)] = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold"), dedupe=True)
    return gold_map


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at line {line_no}: {path}")
            records.append(record)
    return records


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            value = record[key]
            return str(value) if key in {"query", "prompt", "input", "question", "instruction"} else value
    return None


def build_embedder(embedding_model: str) -> TokenOverlapEmbedder | SentenceTransformerEmbedder:
    return TokenOverlapEmbedder() if embedding_model == "token-overlap" else SentenceTransformerEmbedder(embedding_model)


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    test_split: Path,
    predictions: Path,
    embedding_model: str,
    coverage_tau: float,
    redundancy_tau: float,
    weights: tuple[float, float, float],
    matched_queries: int,
) -> dict[str, Any]:
    metric_names = ["coverage", "blind", "redundancy", "validity", "hallucination", "reward"]
    data_source_counts: dict[str, int] = {}
    verifier_source_counts: dict[str, int] = {}
    for row in rows:
        data_source = str(row.get("data_source", ""))
        verifier_source = str(row.get("verifier_source", "none") or "none")
        data_source_counts[data_source] = data_source_counts.get(data_source, 0) + 1
        verifier_source_counts[verifier_source] = verifier_source_counts.get(verifier_source, 0) + 1
    summary: dict[str, Any] = {
        "n": len(rows),
        "matched_samples": len(rows),
        "matched_queries": matched_queries,
        "input": str(predictions),
        "input_sha256": file_sha256(predictions),
        "test_split": str(test_split),
        "test_split_sha256": file_sha256(test_split),
        "predictions": str(predictions),
        "predictions_sha256": file_sha256(predictions),
        "embedding_model": embedding_model,
        "coverage_tau": coverage_tau,
        "redundancy_tau": redundancy_tau,
        "tau": coverage_tau,
        "weights": {
            "coverage": weights[0],
            "validity": weights[1],
            "redundancy": weights[2],
        },
        "data_source_counts": data_source_counts,
        "verifier_source_counts": verifier_source_counts,
        "verifier_source": "mixed" if len(verifier_source_counts) > 1 else next(iter(verifier_source_counts)),
        "total_gold": sum(int(row["n_gold"]) for row in rows),
        "total_gen": sum(int(row["n_gen"]) for row in rows),
    }
    summary["mean_n_gold"] = summary["total_gold"] / max(len(rows), 1)
    summary["mean_n_gen"] = summary["total_gen"] / max(len(rows), 1)
    summary["gen_to_gold_ratio"] = summary["total_gen"] / summary["total_gold"] if summary["total_gold"] > 0 else None
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[f"mean_{name}"] = safe_float(sum(values) / len(values))
    summary["mean_blindspot_rate"] = summary["mean_blind"]
    summary["coverage_per_generated_criterion"] = (
        safe_float(summary["mean_coverage"] / summary["mean_n_gen"]) if summary["mean_n_gen"] > 0 else None
    )
    return summary


def write_per_item(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
