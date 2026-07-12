#!/usr/bin/env python3
"""Run a gold-as-prediction sanity check for the BSC diagnostic chain.

This check is not paper evidence. It verifies that the local pipeline can:
- read hard-gold records,
- build model-like evaluation-criteria predictions,
- join predictions back to gold by query,
- compute BSC metrics with ideal predictions.

With predictions equal to gold, mean coverage should be 1.0 and mean blind
should be 0.0 under deterministic token-overlap matching.
"""

from __future__ import annotations

import argparse
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

from blindspot_rl.reward_bsc import TokenOverlapEmbedder, compute_metrics, safe_float  # noqa: E402
from scripts.bsc_diagnose import build_record_verifier, summarize, write_outputs  # noqa: E402
from scripts.prepare_bsc_eval import (  # noqa: E402
    build_gold_map,
    build_prediction_map,
    compact_record,
    join_blockers,
    load_records,
    write_jsonl,
)


def main() -> None:
    args = parse_args()
    report = run_sanity(
        gold_path=args.gold,
        output_dir=args.output_dir,
        data_source=args.data_source,
        min_joined=args.min_joined,
        limit=args.limit,
    )
    print(
        "BSC gold sanity "
        f"ok={report['ok']} n_joined={report['n_joined']} "
        f"Cov={report['mean_coverage']:.4f} Blind={report['mean_blind']:.4f}"
    )
    if not report["ok"]:
        raise SystemExit("; ".join(report["blockers"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BSC gold-as-prediction sanity check.")
    parser.add_argument("--gold", default=ROOT / "data" / "processed" / "rubricbench_gold.jsonl", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "bsc_gold_sanity", type=Path)
    parser.add_argument("--data-source", default="rubricbench")
    parser.add_argument("--min-joined", default=100, type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def run_sanity(
    gold_path: Path,
    output_dir: Path,
    data_source: str = "rubricbench",
    min_joined: int = 100,
    limit: int | None = None,
) -> dict[str, Any]:
    gold_records = list(load_records(gold_path))
    gold_by_query = build_gold_map(gold_records)
    predictions = gold_as_predictions(gold_by_query, model="gold_as_prediction")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "gold_as_prediction.jsonl"
    joined_path = output_dir / "bsc_eval.jsonl"
    report_path = output_dir / "join_report.json"

    write_jsonl(prediction_path, predictions)
    pred_by_query = build_prediction_map(predictions, model="gold_as_prediction")

    rows: list[dict[str, Any]] = []
    for query_key, gold in gold_by_query.items():
        prediction = pred_by_query.get(query_key)
        if prediction is None:
            continue
        rows.append(
            compact_record(
                {
                    "query": gold["query"],
                    "gold_rubrics": gold["gold_rubrics"],
                    "response": prediction["rubrics"],
                    "model": "gold_as_prediction",
                    "data_source": data_source,
                    "valid_flags": prediction.get("valid_flags"),
                    "verifier_source": prediction.get("verifier_source"),
                }
            )
        )
        if limit and len(rows) >= limit:
            break

    blockers = join_blockers(len(rows), min_joined=min_joined)
    join_report = {
        "ok": not blockers,
        "gold": str(gold_path),
        "predictions": str(prediction_path),
        "output": str(joined_path),
        "n_gold": len(gold_by_query),
        "n_predictions": len(pred_by_query),
        "n_joined": len(rows),
        "min_joined": min_joined,
        "blockers": blockers,
    }
    report_path.write_text(json.dumps(join_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        write_jsonl(joined_path, rows)

    per_item = compute_per_item(rows)
    summary = summarize(
        per_item,
        input_path=joined_path,
        embedding_model="token-overlap",
        coverage_tau=0.99,
        redundancy_tau=0.99,
        weights=(1.0, 0.5, 0.5),
    )
    summary.update(
        {
            "sanity_check": "gold_as_prediction",
            "ok": not blockers and summary["mean_coverage"] == 1.0 and summary["mean_blind"] == 0.0,
            "join_report": str(report_path),
            "prediction_path": str(prediction_path),
            "n_gold_records": len(gold_by_query),
            "n_prediction_records": len(pred_by_query),
            "n_joined": len(rows),
        }
    )
    summary["blockers"] = list(blockers)
    if summary["mean_coverage"] != 1.0:
        summary["blockers"].append(f"gold-as-prediction coverage is not 1.0: {summary['mean_coverage']}")
    if summary["mean_blind"] != 0.0:
        summary["blockers"].append(f"gold-as-prediction blind is not 0.0: {summary['mean_blind']}")
    summary["ok"] = not summary["blockers"]
    write_outputs(per_item, summary, output_dir)
    return summary


def gold_as_predictions(gold_by_query: dict[str, dict[str, Any]], model: str) -> list[dict[str, Any]]:
    rows = []
    for gold in gold_by_query.values():
        rubrics = list(gold["gold_rubrics"])
        rows.append(
            {
                "query": gold["query"],
                "model": model,
                "rubrics": rubrics,
                "valid_flags": [1] * len(rubrics),
                "verifier_source": "gold_as_prediction",
            }
        )
    return rows


def compute_per_item(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    embedder = TokenOverlapEmbedder()
    per_item = []
    for idx, record in enumerate(rows):
        response = record["response"]
        verifier, verifier_source = build_record_verifier(record, response=response, idx=idx)
        metrics = compute_metrics(
            response=response,
            gold_rubrics=record["gold_rubrics"],
            prompt=record.get("query", ""),
            verifier=verifier,
            embedder=embedder,
            coverage_tau=0.99,
            redundancy_tau=0.99,
            weights=(1.0, 0.5, 0.5),
        )
        per_item.append(
            {
                "idx": idx,
                "data_source": record.get("data_source", ""),
                "verifier_source": verifier_source,
                "prompt": record.get("query", ""),
                **{key: safe_float(value) if isinstance(value, float) else value for key, value in metrics.as_dict().items()},
            }
        )
    return per_item


if __name__ == "__main__":
    main()
