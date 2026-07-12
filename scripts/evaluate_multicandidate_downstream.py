#!/usr/bin/env python3
"""Evaluate criterion-guided utility on multi-candidate preference benchmarks."""

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

from blindspot_rl.judge_eval import (  # noqa: E402
    APIRubricScorer,
    KeywordRubricScorer,
    aggregate_multicandidate_results,
    evaluate_multicandidate,
)
from blindspot_rl.llm_api import OpenAICompatibleClient, load_llm_configs, sleep_between_calls  # noqa: E402
from blindspot_rl.reward_bsc import parse_rubrics, safe_float  # noqa: E402
from scripts.budget_gate import enforce_budget_report, file_sha256  # noqa: E402
from scripts.evaluate_downstream import paper_claim_eligibility  # noqa: E402
from scripts.prepare_downstream_eval import load_records, pick_first  # noqa: E402
from scripts.prepare_multicandidate_eval import normalize_candidates, normalize_label  # noqa: E402


def main() -> None:
    args = parse_args()
    budget_report = enforce_api_budget_gate(args)
    benchmark_rows = list(load_records(args.input))
    if not benchmark_rows:
        raise SystemExit(f"No multi-candidate records found in {args.input}")

    scorer = build_scorer(args)
    per_item = []
    results = []
    skipped = 0
    for idx, record in enumerate(benchmark_rows):
        query = pick_first(record, "query", "prompt", "instruction")
        candidates = normalize_candidates(pick_first(record, "candidates", "responses", "answers", "choices", "options"))
        label = normalize_label(
            pick_first(record, "label", "correct", "correct_index", "answer_index", "gold", "winner", "chosen"),
            candidates,
        )
        rubrics = pick_first(record, "rubrics", "generated_rubrics", "model_rubrics", "gold_rubrics")
        parsed_rubrics = parse_rubrics(rubrics, dedupe=True)
        if not query or len(candidates) < 2 or label is None or not parsed_rubrics:
            skipped += 1
            continue

        result = evaluate_multicandidate(
            query=str(query),
            candidates=candidates,
            label=label,
            rubrics=parsed_rubrics,
            scorer=scorer,
            tie_epsilon=args.tie_epsilon,
        )
        sleep_between_calls(args.sleep)
        results.append(result)
        row = {
            "idx": idx,
            "data_source": record.get("data_source", ""),
            "query": query,
            **{
                key: json.dumps(value) if isinstance(value, list) else safe_float(value) if isinstance(value, float) else value
                for key, value in result.as_dict().items()
            },
        }
        per_item.append(row)

    summary = aggregate_multicandidate_results(results)
    summary["skipped"] = skipped
    summary["scorer"] = args.scorer
    summary["scorer_provider"] = str(args.provider) if args.provider else ""
    summary["budget_report"] = str(args.require_budget_report) if args.require_budget_report else ""
    eligibility, eligibility_blockers = paper_claim_eligibility(
        args,
        budget_report,
        expected_contract=expected_multicandidate_budget_contract(args),
    )
    summary["paper_claim_eligible"] = eligibility
    summary["paper_claim_eligibility_blockers"] = eligibility_blockers
    summary["input"] = str(args.input)
    summary["input_sha256"] = file_sha256(args.input)
    summary["rubrics_input"] = ""
    summary["scorer_provider_sha256"] = file_sha256(args.provider) if args.provider and args.provider.exists() else ""
    summary["budget_contract"] = budget_report.get("contract", {}) if isinstance(budget_report, dict) else {}
    summary["benchmark_format"] = "multicandidate"
    summary["scorer_contract"] = {
        "unit_field": "rubrics",
        "unit_multiplier_field": "candidates",
        "calls_per_record_per_provider": 1,
    }
    write_outputs(per_item, summary, args.output_dir)
    print_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-candidate accuracy with generated evaluation criteria."
    )
    parser.add_argument("--input", required=True, type=Path, help="Joined multi-candidate JSONL/JSON/parquet.")
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "multicandidate_downstream", type=Path)
    parser.add_argument("--tie-epsilon", default=1e-8, type=float)
    parser.add_argument("--scorer", choices=["keyword", "api"], default="keyword")
    parser.add_argument("--provider", type=Path, help="Single-line provider JSONL for --scorer api.")
    parser.add_argument(
        "--require-budget-report",
        type=Path,
        help="Require an API budget report with ok=true before issuing judge scorer calls.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep after each benchmark item.")
    return parser.parse_args()


def enforce_api_budget_gate(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.scorer == "api":
        if args.require_budget_report is None:
            enforce_budget_report(args.require_budget_report, "API multi-candidate downstream scoring")
        return enforce_budget_report(
            args.require_budget_report,
            "API multi-candidate downstream scoring",
            expected_contract=expected_multicandidate_budget_contract(args),
        )
    return None


def expected_multicandidate_budget_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": args.input,
        "providers": args.provider,
        "unit_field": "rubrics",
        "unit_multiplier_field": "candidates",
        "calls_per_record_per_provider": 1,
    }


def build_scorer(args: argparse.Namespace) -> Any:
    if args.scorer == "keyword":
        return KeywordRubricScorer()
    if not args.provider:
        raise SystemExit("--provider is required when --scorer api")
    configs = load_llm_configs(args.provider)
    if len(configs) != 1:
        raise SystemExit("API scorer provider file must contain exactly one config.")
    return APIRubricScorer(OpenAICompatibleClient(configs[0]))


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_item_path = output_dir / "per_item.csv"
    if rows:
        with per_item_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["per_item_output"] = str(per_item_path)
        summary["per_item_sha256"] = file_sha256(per_item_path)
        summary["per_item_rows"] = len(rows)
    else:
        summary["per_item_output"] = ""
        summary["per_item_sha256"] = ""
        summary["per_item_rows"] = 0
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(summary: dict[str, Any]) -> None:
    print("Criterion-guided multi-candidate downstream summary")
    print(f"n={summary['n']} skipped={summary['skipped']} scorer={summary['scorer']}")
    print(
        f"Accuracy={summary['accuracy']:.4f} "
        f"Tie={summary['tie_rate']:.4f} "
        f"MeanMargin={summary['mean_margin']:.4f} "
        f"MeanCandidates={summary['mean_candidates']:.2f}"
    )


if __name__ == "__main__":
    main()
