#!/usr/bin/env python3
"""Build a human-audit pack for BSC matched/unmatched rubric pairs.

The output is intentionally an annotation pack, not an automatic human-audit
claim. Human label columns are left blank so the paper can distinguish
machine-prepared evidence from completed human verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    pairwise_cosine,
    parse_rubrics,
)
from scripts.bsc_diagnose import load_records, pick_first  # noqa: E402


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    embedder = TokenOverlapEmbedder() if args.embedding_model == "token-overlap" else SentenceTransformerEmbedder(args.embedding_model)
    candidates = build_candidates(records=records, embedder=embedder, coverage_tau=args.coverage_tau)
    sampled = sample_candidates(
        candidates,
        matched=args.matched,
        unmatched=args.unmatched,
        seed=args.seed,
    )
    summary = summarize(
        records=records,
        candidates=candidates,
        sampled=sampled,
        input_path=args.input,
        embedding_model=args.embedding_model,
        coverage_tau=args.coverage_tau,
        seed=args.seed,
    )
    write_outputs(sampled=sampled, summary=summary, output_dir=args.output_dir)
    print(
        "Human audit pack "
        f"sampled={summary['sampled_total']} "
        f"matched={summary['sampled_matched']} "
        f"unmatched={summary['sampled_unmatched']} "
        f"output={args.output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BSC matched/unmatched human-audit pack.")
    parser.add_argument("--input", required=True, type=Path, help="BSC eval JSONL/JSON/parquet input.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--matched", default=25, type=int, help="Maximum matched gold dimensions to sample.")
    parser.add_argument("--unmatched", default=25, type=int, help="Maximum unmatched gold dimensions to sample.")
    parser.add_argument("--seed", default=13, type=int)
    return parser.parse_args()


def build_candidates(records: list[dict[str, Any]], embedder: Any, coverage_tau: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        query = str(pick_first(record, "query", "prompt", "instruction") or "")
        gold = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold"), dedupe=True)
        generated = parse_rubrics(
            pick_first(record, "response", "model_rubrics", "generated_rubrics", "prediction", "output"),
            dedupe=True,
        )
        if gold and generated:
            sim = pairwise_cosine(gold, generated, embedder)
            best_indices = sim.argmax(axis=1)
            best_scores = sim.max(axis=1)
        else:
            best_indices = []
            best_scores = []

        for gold_idx, gold_rubric in enumerate(gold):
            if generated:
                best_idx = int(best_indices[gold_idx])
                best_generated = generated[best_idx]
                similarity = float(best_scores[gold_idx])
            else:
                best_idx = -1
                best_generated = ""
                similarity = 0.0
            status = "matched" if similarity >= coverage_tau else "unmatched"
            rows.append(
                {
                    "audit_id": f"r{record_idx:04d}_g{gold_idx:03d}",
                    "record_idx": record_idx,
                    "gold_idx": gold_idx,
                    "match_status": status,
                    "similarity": round(similarity, 6),
                    "coverage_tau": coverage_tau,
                    "query": query,
                    "gold_rubric": gold_rubric,
                    "best_generated_idx": best_idx,
                    "best_generated_rubric": best_generated,
                    "human_match_label": "",
                    "human_notes": "",
                }
            )
    return rows


def sample_candidates(
    candidates: list[dict[str, Any]],
    matched: int,
    unmatched: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    matched_rows = [row for row in candidates if row["match_status"] == "matched"]
    unmatched_rows = [row for row in candidates if row["match_status"] == "unmatched"]
    rng.shuffle(matched_rows)
    rng.shuffle(unmatched_rows)
    sampled = matched_rows[: max(matched, 0)] + unmatched_rows[: max(unmatched, 0)]
    return sorted(sampled, key=lambda row: (str(row["match_status"]), int(row["record_idx"]), int(row["gold_idx"])))


def summarize(
    records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
    input_path: Path,
    embedding_model: str,
    coverage_tau: float,
    seed: int,
) -> dict[str, Any]:
    total_matched = sum(row["match_status"] == "matched" for row in candidates)
    total_unmatched = sum(row["match_status"] == "unmatched" for row in candidates)
    sampled_matched = sum(row["match_status"] == "matched" for row in sampled)
    sampled_unmatched = sum(row["match_status"] == "unmatched" for row in sampled)
    return {
        "input": str(input_path),
        "embedding_model": embedding_model,
        "coverage_tau": coverage_tau,
        "seed": seed,
        "n_records": len(records),
        "total_gold_dimensions": len(candidates),
        "total_matched_candidates": total_matched,
        "total_unmatched_candidates": total_unmatched,
        "sampled_total": len(sampled),
        "sampled_matched": sampled_matched,
        "sampled_unmatched": sampled_unmatched,
        "human_labels_completed": 0,
        "status": "annotation_pack_ready" if sampled else "empty",
    }


def write_outputs(sampled: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "audit_items.jsonl").open("w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "audit_items.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "audit_id",
            "record_idx",
            "gold_idx",
            "match_status",
            "similarity",
            "coverage_tau",
            "query",
            "gold_rubric",
            "best_generated_idx",
            "best_generated_rubric",
            "human_match_label",
            "human_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)


if __name__ == "__main__":
    main()
