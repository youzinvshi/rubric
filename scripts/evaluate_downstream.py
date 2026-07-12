#!/usr/bin/env python3
"""Evaluate criterion-guided utility on chosen-vs-rejected preference benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.judge_eval import (  # noqa: E402
    APIRubricScorer,
    KeywordRubricScorer,
    aggregate_preference_results,
    evaluate_preference,
)
from blindspot_rl.llm_api import OpenAICompatibleClient, load_llm_configs, sleep_between_calls  # noqa: E402
from blindspot_rl.reward_bsc import parse_rubrics, safe_float  # noqa: E402
from scripts.budget_gate import budget_contract_blockers, enforce_budget_report, file_sha256  # noqa: E402


def main() -> None:
    args = parse_args()
    budget_report = enforce_api_budget_gate(args)
    preference_rows = list(load_records(args.input))
    if not preference_rows:
        raise SystemExit(f"No preference records found in {args.input}")

    rubric_by_key = load_rubrics(args.rubrics) if args.rubrics else {}
    scorer = build_scorer(args)

    per_item = []
    results = []
    skipped = 0
    for idx, record in enumerate(preference_rows):
        query = pick_first(record, "query", "prompt", "instruction")
        chosen = pick_first(record, "chosen", "winner", "response_chosen", "answer_chosen")
        rejected = pick_first(record, "rejected", "loser", "response_rejected", "answer_rejected")
        rubrics = pick_first(record, "rubrics", "generated_rubrics", "model_rubrics", "gold_rubrics")
        if rubrics is None and rubric_by_key:
            rubrics = rubric_by_key.get(make_key(query))
        parsed_rubrics = parse_rubrics(rubrics, dedupe=True)
        if not query or chosen is None or rejected is None or not parsed_rubrics:
            skipped += 1
            continue

        result = evaluate_preference(
            query=str(query),
            chosen=str(chosen),
            rejected=str(rejected),
            rubrics=parsed_rubrics,
            scorer=scorer,
            tie_epsilon=args.tie_epsilon,
        )
        sleep_between_calls(args.sleep)
        results.append(result)
        per_item.append(
            {
                "idx": idx,
                "data_source": record.get("data_source", ""),
                "query": query,
                **{
                    key: safe_float(value) if isinstance(value, float) else value
                    for key, value in result.as_dict().items()
                },
            }
        )

    summary = aggregate_preference_results(results)
    summary["skipped"] = skipped
    summary["scorer"] = args.scorer
    summary["scorer_provider"] = str(args.provider) if args.provider else ""
    summary["budget_report"] = str(args.require_budget_report) if args.require_budget_report else ""
    eligibility, eligibility_blockers = paper_claim_eligibility(
        args,
        budget_report,
        expected_contract=expected_downstream_budget_contract(args),
    )
    summary["paper_claim_eligible"] = eligibility
    summary["paper_claim_eligibility_blockers"] = eligibility_blockers
    summary["input"] = str(args.input)
    summary["input_sha256"] = file_sha256(args.input)
    summary["rubrics_input"] = str(args.rubrics) if args.rubrics else ""
    summary["scorer_provider_sha256"] = file_sha256(args.provider) if args.provider and args.provider.exists() else ""
    summary["budget_contract"] = budget_report.get("contract", {}) if isinstance(budget_report, dict) else {}
    summary["benchmark_format"] = "pairwise"
    summary["scorer_contract"] = {
        "unit_field": "rubrics",
        "unit_multiplier_field": None,
        "calls_per_record_per_provider": 2,
    }
    write_outputs(per_item, summary, args.output_dir)
    print_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chosen-vs-rejected accuracy with generated evaluation criteria."
    )
    parser.add_argument("--input", required=True, type=Path, help="Preference benchmark JSONL/JSON/parquet.")
    parser.add_argument(
        "--rubrics",
        type=Path,
        help=(
            "Optional JSONL/JSON/parquet mapping query/prompt to evaluation criteria. "
            "If omitted, input must contain the compatibility rubrics field."
        ),
    )
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "downstream", type=Path)
    parser.add_argument("--tie-epsilon", default=1e-8, type=float)
    parser.add_argument("--scorer", choices=["keyword", "api"], default="keyword")
    parser.add_argument("--provider", type=Path, help="Single-line provider JSONL for --scorer api.")
    parser.add_argument(
        "--require-budget-report",
        type=Path,
        help="Require an API budget report with ok=true before issuing judge scorer calls.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep after each preference item.")
    return parser.parse_args()


def enforce_api_budget_gate(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.scorer == "api":
        if args.require_budget_report is None:
            enforce_budget_report(args.require_budget_report, "API downstream scoring")
        return enforce_budget_report(
            args.require_budget_report,
            "API downstream scoring",
            expected_contract=expected_downstream_budget_contract(args),
        )
    return None


def expected_downstream_budget_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": args.input,
        "providers": args.provider,
        "unit_field": "rubrics",
        "unit_multiplier_field": None,
        "calls_per_record_per_provider": 2,
    }


def paper_claim_eligibility(
    args: argparse.Namespace,
    budget_report: dict[str, Any] | None,
    *,
    expected_contract: dict[str, Any],
) -> tuple[bool, list[str]]:
    blockers = []
    if args.scorer != "api":
        blockers.append("downstream scorer is not api")
    if not args.provider:
        blockers.append("missing downstream scorer provider")
    if not args.require_budget_report:
        blockers.append("missing downstream API budget report")
    if not isinstance(budget_report, dict):
        blockers.append("budget report was not loaded")
    elif budget_report.get("ok") is not True:
        blockers.append("budget report ok is not true")
    else:
        blockers.extend(budget_contract_blockers(budget_report, expected_contract))
    return not blockers, blockers


def build_scorer(args: argparse.Namespace) -> Any:
    if args.scorer == "keyword":
        return KeywordRubricScorer()
    if not args.provider:
        raise SystemExit("--provider is required when --scorer api")
    configs = load_llm_configs(args.provider)
    if len(configs) != 1:
        raise SystemExit("API scorer provider file must contain exactly one config.")
    return APIRubricScorer(OpenAICompatibleClient(configs[0]))


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
        for item in records:
            yield item
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        for item in pd.read_parquet(path).to_dict(orient="records"):
            yield item
        return
    raise ValueError(f"Unsupported input format: {path}")


def load_rubrics(path: Path) -> dict[str, Any]:
    mapping = {}
    for record in load_records(path):
        query = pick_first(record, "query", "prompt", "instruction")
        rubrics = pick_first(record, "rubrics", "generated_rubrics", "model_rubrics", "gold_rubrics", "response")
        if query and rubrics is not None:
            mapping[make_key(query)] = rubrics
    return mapping


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def make_key(value: Any) -> str:
    return " ".join(str(value).strip().split())


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
    print("Criterion-guided downstream preference summary")
    print(f"n={summary['n']} skipped={summary['skipped']} scorer={summary['scorer']}")
    print(
        f"Accuracy={summary['accuracy']:.4f} "
        f"Tie={summary['tie_rate']:.4f} "
        f"MeanMargin={summary['mean_margin']:.4f}"
    )


if __name__ == "__main__":
    main()
