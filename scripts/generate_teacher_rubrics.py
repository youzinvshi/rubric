#!/usr/bin/env python3
"""Elicit evaluation-criteria candidates with multiple LLM teachers."""

from __future__ import annotations

import argparse
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

from blindspot_rl.llm_api import (  # noqa: E402
    OpenAICompatibleClient,
    load_llm_configs,
    sleep_between_calls,
)
from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402
from scripts.budget_gate import enforce_budget_report, enforce_preflight_report  # noqa: E402


SYSTEM_PROMPT = (
    "You elicit evaluation criteria. Output JSON only: a list of strings. "
    "Each criterion must be atomic, directly relevant to the query, verifiable as yes/no, "
    "non-overlapping, and must avoid generic criteria such as helpfulness or clarity "
    "unless they are explicitly required by the query."
)


def main() -> None:
    args = parse_args()
    if args.require_preflight_report:
        enforce_preflight_report(
            args.require_preflight_report,
            "teacher evaluation-criteria elicitation",
            expected_contract={
                "input": args.input,
                "providers": args.providers,
            },
        )
    budget_contract: dict[str, Any] = {
        "input": args.input,
        "providers": args.providers,
        "unit_field": None,
        "unit_multiplier_field": None,
        "calls_per_record_per_provider": 1,
    }
    if args.resume:
        budget_contract["resume_output"] = args.output
    enforce_budget_report(
        args.require_budget_report,
        "teacher evaluation-criteria elicitation",
        expected_contract=budget_contract,
    )
    configs = load_llm_configs(args.providers)
    
    # We will need the full records to construct domain-specific prompts
    records = list(load_records(args.input))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No records found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output) if args.resume else set()
    wrote = 0
    with args.output.open("a", encoding="utf-8") as f:
        for q_idx, record in enumerate(records):
            query = pick_first(record, "query", "prompt", "instruction", "question")
            if not query:
                continue
            
            data_source = record.get("data_source", args.data_source)
            system_prompt, user_prompt = construct_prompt(data_source, record, query)

            for config in configs:
                key = (query, config.name)
                if key in done:
                    continue
                client = OpenAICompatibleClient(config)
                try:
                    response = client.chat(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )
                    rubrics = parse_rubrics(response, dedupe=False)
                    error = ""
                except Exception as exc:
                    if args.on_error == "fail":
                        raise
                    response = ""
                    rubrics = []
                    error = f"{type(exc).__name__}: {exc}"
                row = {
                    "query": query,
                    "teacher": config.name,
                    "model": config.model,
                    "response": response,
                    "rubrics": rubrics,
                    "data_source": data_source,
                    "query_idx": q_idx,
                }
                if error:
                    row["generation_failed"] = True
                    row["error"] = error
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                wrote += 1
                sleep_between_calls(args.sleep)
    print(f"Wrote {wrote} teacher evaluation-criteria outputs to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elicit multi-teacher evaluation-criteria candidates.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL/JSON with query/prompt/instruction.")
    parser.add_argument("--providers", required=True, type=Path, help="Provider config JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output teacher JSONL.")
    parser.add_argument("--data-source", default="teacher_generation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Skip query-teacher pairs already in output.")
    parser.add_argument(
        "--on-error",
        choices=("fail", "write_error"),
        default="fail",
        help="Whether to fail immediately or write an auditable error row and continue.",
    )
    parser.add_argument(
        "--require-budget-report",
        type=Path,
        help="Require an API budget report with ok=true before issuing generation calls.",
    )
    parser.add_argument(
        "--require-preflight-report",
        type=Path,
        help="Require a preflight report with ok=true before issuing generation calls.",
    )
    return parser.parse_args()


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("records") or data.get("data") or [data]
    for obj in records:
        yield obj


def load_queries(path: Path) -> list[str]:
    """Load query strings from the same record formats used by teacher generation."""

    queries: list[str] = []
    for record in load_records(path):
        query = pick_first(record, "query", "prompt", "instruction", "question")
        if query:
            queries.append(str(query))
    return queries


def construct_prompt(data_source: str, record: dict[str, Any], query: str) -> tuple[str, str]:
    ds = data_source.lower()
    
    if "rewardbench" in ds:
        system = "You are eliciting evaluation criteria for LLM-as-a-Judge. Output JSON only: a list of strings."
        chosen = record.get("chosen", "")
        rejected = record.get("rejected", "")
        user = (
            f"User query:\n{query}\n\n"
            f"Candidate answer A:\n{chosen}\n\n"
            f"Candidate answer B:\n{rejected}\n\n"
            "Elicit 5-8 atomic evaluation criteria that a judge should use to decide which answer is better.\n"
            "Each criterion must be:\n"
            "- atomic\n"
            "- yes/no decidable\n"
            "- grounded in the user query\n"
            "- not mentioning A or B directly\n"
            "- non-redundant\n"
            "Return a JSON list of strings."
        )
        return system, user

    if "healthbench" in ds:
        system = "You are eliciting medical evaluation criteria. Output JSON only: a list of strings."
        user = (
            f"Medical user query:\n{query}\n\n"
            "Elicit 6-10 atomic criteria for evaluating an assistant's answer.\n"
            "Criteria should cover:\n"
            "- medical factuality\n"
            "- safety and risk disclosure\n"
            "- avoiding overconfident diagnosis\n"
            "- recommending professional care when needed\n"
            "- completeness\n"
            "- user-specific constraints\n"
            "Return a JSON list of strings."
        )
        return system, user

    if "writingbench" in ds:
        system = "You are eliciting criteria for evaluating writing tasks. Output JSON only: a list of strings."
        user = (
            f"Writing instruction:\n{query}\n\n"
            "Elicit 6-10 atomic evaluation criteria.\n"
            "Cover:\n"
            "- task fulfillment\n"
            "- structure\n"
            "- clarity\n"
            "- style/tone\n"
            "- audience fit\n"
            "- factual consistency if relevant\n"
            "- constraint following\n"
            "Return a JSON list of strings."
        )
        return system, user

    if "ifbench" in ds or "advancedif" in ds:
        system = "You are eliciting instruction-following criteria. Output JSON only: a list of strings."
        user = (
            f"Instruction:\n{query}\n\n"
            "Elicit atomic criteria that check whether an answer follows every explicit and implicit constraint.\n"
            "Each criterion should test exactly one constraint.\n"
            "Return a JSON list of strings."
        )
        return system, user

    if "beir" in ds or "serp" in ds or "search" in ds:
        system = "You are eliciting criteria for search result evaluation. Output JSON only: a list of strings."
        user = (
            f"Search query:\n{query}\n\n"
            "Elicit 5-8 atomic criteria for judging whether a document or search snippet satisfies the search intent.\n"
            "Cover:\n"
            "- topical relevance\n"
            "- intent satisfaction\n"
            "- evidence support\n"
            "- freshness if relevant\n"
            "- authority/source quality\n"
            "- completeness\n"
            "- avoiding misleading information\n"
            "Return a JSON list of strings."
        )
        return system, user

    # Default fallback
    return SYSTEM_PROMPT, f"Elicit 6-10 atomic evaluation criteria for the following query.\n\nQuery:\n{query}"

def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            query = obj.get("query")
            teacher = obj.get("teacher")
            if query and teacher:
                done.add((str(query), str(teacher)))
    return done


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


if __name__ == "__main__":
    main()
