#!/usr/bin/env python3
"""Elicit evaluation criteria for one or more evaluation methods/models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.llm_api import (  # noqa: E402
    LLMConfig,
    OpenAICompatibleClient,
    load_llm_configs,
    sleep_between_calls,
)
from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402
from scripts.budget_gate import enforce_budget_report, enforce_preflight_report  # noqa: E402
from scripts.generate_teacher_rubrics import load_queries  # noqa: E402


DEFAULT_SYSTEM_PROMPT = (
    "You elicit evaluation criteria. Output JSON only: a list of strings. "
    "Each criterion must be atomic, directly relevant to the query, verifiable as yes/no, "
    "non-overlapping, and must avoid generic criteria such as helpfulness or clarity "
    "unless explicitly required by the query."
)

DEFAULT_USER_TEMPLATE = (
    "Elicit 6-10 atomic evaluation criteria for the following query.\n\n"
    "Query:\n{query}"
)


def main() -> None:
    args = parse_args()
    if args.require_preflight_report:
        enforce_preflight_report(
            args.require_preflight_report,
            "model evaluation-criteria elicitation",
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
        "model evaluation-criteria elicitation",
        expected_contract=budget_contract,
    )
    configs = load_llm_configs(args.providers)
    queries = list(load_queries(args.input))
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit("No queries found.")

    done = load_done(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    wrote = 0
    with args.output.open(mode, encoding="utf-8") as sink:
        def write_row(row: dict[str, Any]) -> None:
            nonlocal wrote
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            wrote += 1

        generate_rows(
            queries=queries,
            configs=configs,
            done=done,
            data_source=args.data_source,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
            sleep=args.sleep,
            row_sink=write_row,
        )
    print(f"Wrote {wrote} model evaluation-criteria outputs to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elicit model evaluation criteria for BSC/downstream eval.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL/JSON with query/prompt/instruction.")
    parser.add_argument("--providers", required=True, type=Path, help="Method/model config JSONL.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-source", default="model_evaluation_criteria_elicitation")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Skip query-method pairs already in output.")
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


def generate_rows(
    queries: list[str],
    configs: list[LLMConfig],
    done: set[tuple[str, str]] | None = None,
    data_source: str = "model_evaluation_criteria_elicitation",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    sleep: float = 0.0,
    client_factory: Callable[[LLMConfig], Any] = OpenAICompatibleClient,
    row_sink: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    done = done or set()
    for q_idx, query in enumerate(queries):
        for config in configs:
            method = config.name
            if (query, method) in done:
                continue
            client = client_factory(config)
            response = client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.format(query=query)},
                ]
            )
            row = {
                "query": query,
                "method": method,
                "model": method,
                "model_name_or_path": config.model,
                "response": response,
                "rubrics": parse_rubrics(response, dedupe=False),
                "data_source": data_source,
                "query_idx": q_idx,
            }
            if row_sink is not None:
                row_sink(row)
            else:
                rows.append(row)
            sleep_between_calls(sleep)
    return rows


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for record in load_jsonl(path):
        query = record.get("query")
        method = record.get("method") or record.get("model") or record.get("teacher")
        if query and method:
            done.add((str(query), str(method)))
    return done


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]], mode: str = "w") -> int:
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    main()
