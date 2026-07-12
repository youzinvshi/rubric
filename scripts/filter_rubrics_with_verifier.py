#!/usr/bin/env python3
"""Filter generated criteria with an API Meta-Verifier or deterministic rules."""

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

from blindspot_rl.llm_api import OpenAICompatibleClient, load_llm_configs, sleep_between_calls  # noqa: E402
from blindspot_rl.meta_verifier import (  # noqa: E402
    APIMetaVerifier,
    RuleMetaVerifier,
    filter_proxy_rubrics,
)
from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402
from scripts.budget_gate import enforce_budget_report, enforce_preflight_report, file_sha256  # noqa: E402


def main() -> None:
    args = parse_args()
    enforce_api_budget_gate(args)
    records = load_jsonl(args.input)
    if not records:
        raise SystemExit("No criteria records found.")

    verifier = build_verifier(args)
    kept_records = []
    stats = []
    for record in records:
        query = str(pick_first(record, "query", "prompt", "instruction") or "")
        data_source = str(record.get("data_source") or args.data_source or "")
        rubrics = parse_rubrics(
            pick_first(record, "rubrics", "model_rubrics", "generated_rubrics", "response", "output", "prediction"),
            dedupe=False,
        )
        result = filter_proxy_rubrics(
            query=query,
            candidates=rubrics,
            verifier=verifier,
            data_source=data_source,
            dedup_tau=args.dedup_tau,
            max_rubrics=args.max_rubrics,
            rule_prefilter=not args.no_rule_prefilter,
            reject_generic_terms=args.reject_generic_terms,
        )
        kept = result.verified_rubrics
        if kept or not args.drop_empty or args.annotate_only:
            new_record = dict(record)
            new_record["rubrics_before_filter"] = result.rubrics_before_filter
            if args.annotate_only:
                new_record["rubrics"] = result.rubrics_before_filter
                new_record["verified_rubrics"] = kept
                new_record["valid_flags"] = result.valid_flags
            else:
                new_record["rubrics"] = kept
                new_record["valid_flags"] = [1] * len(kept)
                new_record["valid_flags_before_filter"] = result.valid_flags
            new_record["verifier_decisions"] = result.decisions_as_dicts()
            new_record["verifier_source"] = "valid_flags"
            kept_records.append(new_record)
        invalid_reason_counts: dict[str, int] = {}
        for decision in result.verifier_decisions:
            if not decision.valid:
                invalid_reason_counts[decision.reason] = invalid_reason_counts.get(decision.reason, 0) + 1
        stats.append(
            {
                "query": query,
                "teacher": record.get("teacher", ""),
                "data_source": data_source,
                "n_input": len(result.rubrics_before_filter),
                "n_valid": sum(result.valid_flags),
                "validity": result.validity,
                "invalid_reason_counts": invalid_reason_counts,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, kept_records)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.stats_output, stats)
    if args.report_output:
        write_report(
            args.report_output,
            args=args,
            n_input_records=len(records),
            n_output_records=len(kept_records),
            stats=stats,
        )
    total_in = sum(item["n_input"] for item in stats)
    total_valid = sum(item["n_valid"] for item in stats)
    print(
        f"Filtered {len(records)} records: kept {total_valid}/{total_in} criteria "
        f"({total_valid / max(total_in, 1):.4f}). Output: {args.output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter generated criteria with a Meta-Verifier.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--report-output", type=Path, help="Optional JSON report with filtering provenance.")
    parser.add_argument(
        "--mode",
        choices=["api", "rule"],
        default="rule",
        help="Use API verifier for real filtering; rule mode is for smoke tests.",
    )
    parser.add_argument("--provider", type=Path, help="Single-line provider JSONL for API verifier.")
    parser.add_argument("--data-source", help="Fallback data source/task type when records do not contain data_source.")
    parser.add_argument(
        "--require-budget-report",
        type=Path,
        help="Require an API budget report with ok=true before issuing verifier calls.",
    )
    parser.add_argument(
        "--require-preflight-report",
        type=Path,
        help="Require a preflight report with ok=true before issuing verifier calls.",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--dedup-tau", type=float, default=0.90)
    parser.add_argument("--max-rubrics", type=int, help="Keep at most this many verified criteria per record.")
    parser.add_argument(
        "--no-rule-prefilter",
        action="store_true",
        help="Send even obvious rule failures to the API verifier. Not recommended for production.",
    )
    parser.add_argument(
        "--reject-generic-terms",
        action="store_true",
        help="Reject criteria containing broad terms such as helpful/clear/quality.",
    )
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Attach valid_flags while preserving the original criteria for Hall/BSC evaluation.",
    )
    return parser.parse_args()


def enforce_api_budget_gate(args: argparse.Namespace) -> None:
    if args.mode == "api":
        require_preflight_report = getattr(args, "require_preflight_report", None)
        if require_preflight_report:
            enforce_preflight_report(
                require_preflight_report,
                "API meta-verifier filtering",
                expected_contract={"providers": args.provider},
            )
        if args.require_budget_report is None:
            enforce_budget_report(args.require_budget_report, "API meta-verifier filtering")
        enforce_budget_report(
            args.require_budget_report,
            "API meta-verifier filtering",
            expected_contract={
                "input": args.input,
                "providers": args.provider,
                "unit_field": "rubrics",
                "unit_multiplier_field": None,
                "calls_per_record_per_provider": 1,
            },
        )


def build_verifier(args: argparse.Namespace) -> Any:
    if args.mode == "rule":
        return RuleMetaVerifier(reject_generic_terms=args.reject_generic_terms)
    if not args.provider:
        raise SystemExit("--provider is required when --mode api")
    configs = load_llm_configs(args.provider)
    if len(configs) != 1:
        raise SystemExit("Verifier provider file must contain exactly one config.")
    verifier = APIMetaVerifier(OpenAICompatibleClient(configs[0]))
    if args.sleep <= 0:
        return verifier
    return SleepingVerifier(verifier, sleep=args.sleep)


class SleepingVerifier:
    def __init__(self, verifier: Any, sleep: float):
        self.verifier = verifier
        self.sleep = sleep

    def verify(self, rubric: str, *, query: str = "", data_source: str = "") -> Any:
        decision = self.verifier.verify(rubric, query=query, data_source=data_source)
        sleep_between_calls(self.sleep)
        return decision

    def judge(self, rubric: str, **kwargs: Any) -> int:
        return int(
            self.verify(
                rubric,
                query=str(kwargs.get("prompt") or kwargs.get("query") or ""),
                data_source=str(kwargs.get("data_source") or ""),
            ).valid
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
    return records


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    n_input_records: int,
    n_output_records: int,
    stats: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_input_rubrics = sum(int(item.get("n_input", 0)) for item in stats)
    total_valid_rubrics = sum(int(item.get("n_valid", 0)) for item in stats)
    report = {
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "stats_output": str(args.stats_output) if args.stats_output else "",
        "stats_output_sha256": file_sha256(args.stats_output) if args.stats_output and args.stats_output.exists() else "",
        "mode": args.mode,
        "provider": str(args.provider) if args.provider else "",
        "provider_sha256": file_sha256(args.provider) if args.provider and args.provider.exists() else "",
        "budget_report": str(args.require_budget_report) if args.require_budget_report else "",
        "budget_report_sha256": (
            file_sha256(args.require_budget_report)
            if args.require_budget_report and args.require_budget_report.exists()
            else ""
        ),
        "preflight_report": str(args.require_preflight_report) if args.require_preflight_report else "",
        "preflight_report_sha256": (
            file_sha256(args.require_preflight_report)
            if args.require_preflight_report and args.require_preflight_report.exists()
            else ""
        ),
        "n_input_records": n_input_records,
        "n_output_records": n_output_records,
        "n_stats_records": len(stats),
        "n_input_rubrics": total_input_rubrics,
        "n_valid_rubrics": total_valid_rubrics,
        "validity": total_valid_rubrics / max(total_input_rubrics, 1),
        "drop_empty": bool(args.drop_empty),
        "annotate_only": bool(args.annotate_only),
        "dedup_tau": args.dedup_tau,
        "max_rubrics": args.max_rubrics,
        "rule_prefilter": not args.no_rule_prefilter,
        "reject_generic_terms": bool(args.reject_generic_terms),
        "columns": {
            "output": [
                "query",
                "rubrics",
                "valid_flags",
                "verifier_decisions",
                "verifier_source",
            ],
            "stats": [
                "query",
                "teacher",
                "data_source",
                "n_input",
                "n_valid",
                "validity",
                "invalid_reason_counts",
            ],
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
