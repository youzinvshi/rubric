#!/usr/bin/env python3
"""Evaluate recovered and lost human-gold evaluation dimensions.

The unit of analysis is a human-gold evaluation dimension. A dimension is
counted as recovered only when it is uncovered by the baseline and covered by
the candidate under the same semantic threshold. This is stricter than
comparing aggregate BSC means and makes a dimension-transition statement
auditable before any dimension-recovery claim is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from scripts.blindspot_attribution import CATEGORIES, classify_rubric  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402


def main() -> None:
    args = parse_args()
    baseline_records = list(load_records(args.baseline))
    candidate_records = list(load_records(args.candidate))
    if not baseline_records:
        raise SystemExit(f"No baseline records found in {args.baseline}")
    if not candidate_records:
        raise SystemExit(f"No candidate records found in {args.candidate}")

    embedder = TokenOverlapEmbedder() if args.embedding_model == "token-overlap" else SentenceTransformerEmbedder(args.embedding_model)
    per_item, category_rows, gold_rows, summary = evaluate_blindspot_repair(
        baseline_records,
        candidate_records,
        embedder=embedder,
        coverage_tau=args.coverage_tau,
        join_key=args.join_key,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        respect_valid_flags=not args.ignore_valid_flags,
    )
    summary.update(
        {
            "baseline": str(args.baseline),
            "baseline_sha256": file_sha256(args.baseline),
            "candidate": str(args.candidate),
            "candidate_sha256": file_sha256(args.candidate),
            "embedding_model": args.embedding_model,
            "coverage_tau": args.coverage_tau,
            "join_key": args.join_key,
            "respect_valid_flags": not args.ignore_valid_flags,
        }
    )
    write_outputs(args.output_dir, per_item, category_rows, gold_rows, summary)
    print_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute gold-dimension recovery/loss transition rates.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline BSC eval JSONL/JSON/parquet.")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate BSC eval JSONL/JSON/parquet.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--join-key", default="query", help="Record key used to align baseline and candidate rows.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--ignore-valid-flags",
        action="store_true",
        help="Count all generated criteria even when valid_flags are present.",
    )
    return parser.parse_args()


def evaluate_blindspot_repair(
    baseline_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    *,
    embedder: Any,
    coverage_tau: float = 0.75,
    join_key: str = "query",
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    respect_valid_flags: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_by_key, baseline_join_stats = build_record_index(baseline_records, join_key)
    candidate_by_key, candidate_join_stats = build_record_index(candidate_records, join_key)
    shared_keys = [key for key in baseline_by_key if key in candidate_by_key]
    if not shared_keys:
        raise ValueError(f"No overlapping records for join_key={join_key!r}")

    per_item: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    totals = init_totals()

    for idx, key in enumerate(shared_keys):
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        prompt = str(pick_first(candidate, join_key, "query", "prompt", "instruction") or key)
        gold = parse_rubrics(
            pick_first(candidate, "gold_rubrics", "gold", "rubrics_gold")
            or pick_first(baseline, "gold_rubrics", "gold", "rubrics_gold"),
            dedupe=False,
        )
        baseline_gen = extract_generated_rubrics(baseline, respect_valid_flags=respect_valid_flags)
        candidate_gen = extract_generated_rubrics(candidate, respect_valid_flags=respect_valid_flags)

        baseline_hits, baseline_sims, baseline_match = dimension_hits(gold, baseline_gen, coverage_tau, embedder)
        candidate_hits, candidate_sims, candidate_match = dimension_hits(gold, candidate_gen, coverage_tau, embedder)

        item_counts = init_item_counts()
        for gold_idx, rubric in enumerate(gold):
            category = classify_rubric(rubric)
            baseline_covered = baseline_hits[gold_idx]
            candidate_covered = candidate_hits[gold_idx]
            recovered = (not baseline_covered) and candidate_covered
            lost = baseline_covered and (not candidate_covered)
            retained = baseline_covered and candidate_covered
            still_blind = (not baseline_covered) and (not candidate_covered)

            update_counts(totals["overall"], baseline_covered, candidate_covered, recovered, lost, retained, still_blind)
            update_counts(totals["categories"][category], baseline_covered, candidate_covered, recovered, lost, retained, still_blind)
            update_counts(item_counts, baseline_covered, candidate_covered, recovered, lost, retained, still_blind)

            base_idx = baseline_match[gold_idx]
            cand_idx = candidate_match[gold_idx]
            gold_rows.append(
                {
                    "idx": idx,
                    "join_key": key,
                    "gold_idx": gold_idx,
                    "category": category,
                    "baseline_covered": int(baseline_covered),
                    "candidate_covered": int(candidate_covered),
                    "recovered": int(recovered),
                    "lost": int(lost),
                    "retained": int(retained),
                    "still_blind": int(still_blind),
                    "baseline_max_similarity": round(baseline_sims[gold_idx], 6),
                    "candidate_max_similarity": round(candidate_sims[gold_idx], 6),
                    "baseline_best_generated": baseline_gen[base_idx] if 0 <= base_idx < len(baseline_gen) else "",
                    "candidate_best_generated": candidate_gen[cand_idx] if 0 <= cand_idx < len(candidate_gen) else "",
                    "gold_rubric": rubric,
                }
            )

        per_item.append(
            {
                "idx": idx,
                "join_key": key,
                "prompt": prompt,
                "n_gold": len(gold),
                "n_baseline_gen": len(baseline_gen),
                "n_candidate_gen": len(candidate_gen),
                **rates_from_counts(item_counts),
            }
        )

    category_rows = [
        {"category": category, **rates_from_counts(totals["categories"][category])}
        for category in CATEGORIES
    ]
    overall_rates = rates_from_counts(totals["overall"])
    transition_balance = totals["overall"]["recovered_gold"] - totals["overall"]["lost_gold"]
    n_unmatched_baseline = len(baseline_by_key) - len(shared_keys)
    n_unmatched_candidate = len(candidate_by_key) - len(shared_keys)
    summary = {
        "n_matched_records": len(shared_keys),
        "n_baseline_records": len(baseline_records),
        "n_candidate_records": len(candidate_records),
        "n_baseline_join_keys": len(baseline_by_key),
        "n_candidate_join_keys": len(candidate_by_key),
        "n_unmatched_baseline_records": n_unmatched_baseline,
        "n_unmatched_candidate_records": n_unmatched_candidate,
        "baseline_duplicate_join_key_count": baseline_join_stats["duplicate_join_key_count"],
        "baseline_duplicate_record_count": baseline_join_stats["duplicate_record_count"],
        "candidate_duplicate_join_key_count": candidate_join_stats["duplicate_join_key_count"],
        "candidate_duplicate_record_count": candidate_join_stats["duplicate_record_count"],
        "query_alignment_exact": n_unmatched_baseline == 0 and n_unmatched_candidate == 0,
        "per_item_rows_count": len(per_item),
        "per_item_rows_match_n_matched_records": len(per_item) == len(shared_keys),
        "gold_rows_count": len(gold_rows),
        "gold_rows_match_total_gold": len(gold_rows) == totals["overall"]["total_gold"],
        "category_rows_count": len(category_rows),
        "transition_balance": transition_balance,
        "recovered_exceeds_lost": totals["overall"]["recovered_gold"] > totals["overall"]["lost_gold"],
        "net_positive_transition": transition_balance > 0,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        **overall_rates,
    }
    return per_item, category_rows, gold_rows, summary


def init_totals() -> dict[str, Any]:
    return {
        "overall": init_item_counts(),
        "categories": {category: init_item_counts() for category in CATEGORIES},
    }


def init_item_counts() -> dict[str, int]:
    return {
        "total_gold": 0,
        "baseline_covered_gold": 0,
        "baseline_blind_gold": 0,
        "candidate_covered_gold": 0,
        "candidate_blind_gold": 0,
        "recovered_gold": 0,
        "lost_gold": 0,
        "retained_gold": 0,
        "still_blind_gold": 0,
    }


def update_counts(
    counts: dict[str, int],
    baseline_covered: bool,
    candidate_covered: bool,
    recovered: bool,
    lost: bool,
    retained: bool,
    still_blind: bool,
) -> None:
    counts["total_gold"] += 1
    counts["baseline_covered_gold"] += int(baseline_covered)
    counts["baseline_blind_gold"] += int(not baseline_covered)
    counts["candidate_covered_gold"] += int(candidate_covered)
    counts["candidate_blind_gold"] += int(not candidate_covered)
    counts["recovered_gold"] += int(recovered)
    counts["lost_gold"] += int(lost)
    counts["retained_gold"] += int(retained)
    counts["still_blind_gold"] += int(still_blind)


def rates_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    total = counts["total_gold"]
    baseline_blind = counts["baseline_blind_gold"]
    baseline_covered = counts["baseline_covered_gold"]
    recovered_dimension_rate = counts["recovered_gold"] / baseline_blind if baseline_blind else 0.0
    loss_rate = counts["lost_gold"] / baseline_covered if baseline_covered else 0.0
    net_transition_rate = (counts["recovered_gold"] - counts["lost_gold"]) / total if total else 0.0
    return {
        **counts,
        "baseline_coverage": counts["baseline_covered_gold"] / total if total else 0.0,
        "baseline_blind_rate": baseline_blind / total if total else 0.0,
        "candidate_coverage": counts["candidate_covered_gold"] / total if total else 0.0,
        "candidate_blind_rate": counts["candidate_blind_gold"] / total if total else 0.0,
        "recovered_dimension_rate": recovered_dimension_rate,
        "loss_rate": loss_rate,
        "net_transition_rate": net_transition_rate,
    }


def dimension_hits(
    gold: Sequence[str],
    generated: Sequence[str],
    coverage_tau: float,
    embedder: Any,
) -> tuple[list[bool], list[float], list[int]]:
    if not gold:
        return [], [], []
    if not generated:
        return [False for _ in gold], [0.0 for _ in gold], [-1 for _ in gold]
    sim = pairwise_cosine(gold, generated, embedder)
    max_sims = [float(value) for value in sim.max(axis=1)]
    matches = [int(value) for value in sim.argmax(axis=1)]
    hits = [value >= coverage_tau for value in max_sims]
    return hits, max_sims, matches


def extract_generated_rubrics(record: dict[str, Any], respect_valid_flags: bool = True) -> list[str]:
    response = pick_first(record, "response", "model_rubrics", "generated_rubrics", "prediction", "output")
    generated = parse_rubrics(response, dedupe=True)
    flags = pick_first(record, "valid_flags", "verifier_flags", "validity_flags")
    if not respect_valid_flags or flags is None:
        return generated
    if not isinstance(flags, list):
        raise ValueError("valid_flags must be a list when present")
    raw_generated = parse_rubrics(response, dedupe=False)
    if len(flags) != len(raw_generated):
        raise ValueError(f"valid_flags length mismatch: {len(flags)} flags for {len(raw_generated)} generated criteria")
    valid_items = [
        rubric
        for rubric, flag in zip(raw_generated, flags)
        if parse_valid_flag(flag)
    ]
    return parse_rubrics(valid_items, dedupe=True)


def parse_valid_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "valid"}:
            return True
        if text in {"0", "false", "no", "invalid"}:
            return False
    raise ValueError(f"valid_flags contains non-binary value: {value!r}")


def build_record_index(
    records: Sequence[dict[str, Any]],
    join_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    mapped: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, int] = {}
    for idx, record in enumerate(records):
        base_key = normalize_join_key(pick_first(record, join_key, "query", "prompt", "instruction"))
        if not base_key:
            raise ValueError(f"Missing join key for record {idx}")
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        key = base_key if occurrence == 0 else f"{base_key} [duplicate #{occurrence + 1}]"
        mapped[key] = record
    stats = {
        "duplicate_join_key_count": sum(1 for count in occurrences.values() if count > 1),
        "duplicate_record_count": sum(count - 1 for count in occurrences.values() if count > 1),
    }
    return mapped, stats


def build_record_map(records: Sequence[dict[str, Any]], join_key: str) -> dict[str, dict[str, Any]]:
    mapped, _ = build_record_index(records, join_key)
    return mapped


def normalize_join_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
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
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("records") or data.get("data") or [data]
        yield from records
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    raise ValueError(f"Unsupported input format: {path}")


def write_outputs(
    output_dir: Path,
    per_item: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "transition_per_item.csv", per_item)
    write_csv(output_dir / "transition_by_category.csv", category_rows)
    with (output_dir / "transition_gold_items.jsonl").open("w", encoding="utf-8") as f:
        for row in gold_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    print("Gold-dimension transition summary")
    print(
        f"n={summary['n_matched_records']} total_gold={summary['total_gold']} "
        f"baseline_blind_gold={summary['baseline_blind_gold']}"
    )
    print(
        f"baseline_cov={summary['baseline_coverage']:.4f} "
        f"candidate_cov={summary['candidate_coverage']:.4f} "
        f"recovered_dimension_rate={summary['recovered_dimension_rate']:.4f} "
        f"loss_rate={summary['loss_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
