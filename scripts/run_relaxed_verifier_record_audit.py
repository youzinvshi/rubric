#!/usr/bin/env python3
"""Record-level relaxed API verifier audit for production proxy criteria.

This is an audit gate, not the full cleaning path. It samples records and asks
the domain-aware API verifier to judge all criteria in one record with a single
call, reducing audit cost from per-criterion calls to per-record calls.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.llm_api import OpenAICompatibleClient, extract_json, load_llm_configs, sleep_between_calls  # noqa: E402
from blindspot_rl.meta_verifier import domain_policy  # noqa: E402
from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402


def main() -> None:
    args = parse_args()
    if args.env_file.exists():
        load_env(args.env_file)

    rows = sample_records(load_jsonl(args.input), args.sample_size, args.seed)
    provider = load_llm_configs(args.provider)[0]
    budget = estimate_budget(rows, provider)
    if budget["calls"] > args.max_calls or budget["total_tokens_upper_bound"] > args.max_total_tokens:
        raise SystemExit(
            "Budget gate blocked record audit: "
            f"calls={budget['calls']} max_calls={args.max_calls}, "
            f"tokens={budget['total_tokens_upper_bound']} max_total_tokens={args.max_total_tokens}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / f"sample_{args.data_source}.jsonl"
    audit_path = args.output_dir / f"{args.data_source}_record_audit.jsonl"
    verified_path = args.output_dir / f"{args.data_source}_verified.jsonl"
    summary_path = args.output_dir / f"{args.data_source}_summary.json"
    budget_path = args.output_dir / f"{args.data_source}_record_audit_budget.json"

    write_jsonl(sample_path, rows)
    budget_path.write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")

    client = OpenAICompatibleClient(provider)
    audit_rows = []
    verified_rows = []
    for idx, row in enumerate(rows, start=1):
        query = str(pick_first(row, "query", "prompt", "instruction", "question") or "")
        rubrics = parse_rubrics(pick_first(row, "rubrics", "model_rubrics", "generated_rubrics", "response", "output"))
        decisions = audit_record(
            client=client,
            query=query,
            rubrics=rubrics,
            data_source=args.data_source,
            max_rubrics=args.max_rubrics,
        )
        valid_flags = [int(decision.get("valid") is True) for decision in decisions]
        kept = [rubric for rubric, flag in zip(rubrics, valid_flags) if flag][: args.max_rubrics]
        audit_row = {
            "query": query,
            "teacher": row.get("teacher") or row.get("method") or row.get("model"),
            "data_source": args.data_source,
            "n_input": len(rubrics),
            "n_valid": len(kept),
            "validity": len(kept) / max(len(rubrics), 1),
            "valid_flags": valid_flags,
            "verifier_source": provider.name,
            "verifier_decisions": decisions,
        }
        audit_rows.append(audit_row)
        if kept or not args.drop_empty:
            out = dict(row)
            out["rubrics"] = kept
            out["valid_flags"] = [1] * len(kept)
            out["audit_valid_flags"] = valid_flags
            out["verifier_source"] = provider.name
            out["verifier_mode"] = "record_level_relaxed_api_audit"
            verified_rows.append(out)
        if args.progress_every and idx % args.progress_every == 0:
            print(f"Audited {idx}/{len(rows)} records for {args.data_source}", flush=True)
        sleep_between_calls(args.sleep)

    write_jsonl(audit_path, audit_rows)
    write_jsonl(verified_path, verified_rows)
    summary = summarize(audit_rows, sample_path, audit_path, verified_path, budget)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run record-level relaxed API verifier audit.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provider", default=Path("configs/verifier.local.jsonl"), type=Path)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rubrics", type=int, default=12)
    parser.add_argument("--max-calls", type=int, default=250)
    parser.add_argument("--max-total-tokens", type=int, default=2_000_000)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--env-file", default=Path(".env"), type=Path)
    return parser.parse_args()


def load_env(path: Path) -> None:
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def sample_records(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def estimate_budget(rows: list[dict[str, Any]], provider: Any) -> dict[str, Any]:
    prompt_tokens = 0
    for row in rows:
        query = str(pick_first(row, "query", "prompt", "instruction", "question") or "")
        rubrics = parse_rubrics(pick_first(row, "rubrics", "model_rubrics", "generated_rubrics", "response", "output"))
        prompt_tokens += rough_tokens(query) + sum(rough_tokens(rubric) for rubric in rubrics) + 500
    completion_tokens = len(rows) * int(getattr(provider, "max_tokens", 1200))
    return {
        "provider": getattr(provider, "name", "unknown"),
        "calls": len(rows),
        "prompt_tokens_estimate": prompt_tokens,
        "completion_tokens_upper_bound": completion_tokens,
        "total_tokens_upper_bound": prompt_tokens + completion_tokens,
    }


def rough_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)


def audit_record(
    *,
    client: OpenAICompatibleClient,
    query: str,
    rubrics: list[str],
    data_source: str,
    max_rubrics: int,
) -> list[dict[str, Any]]:
    if not rubrics:
        return []
    system = (
        "You are a strict but domain-aware meta-verifier for evaluation-criteria training data. "
        "Return JSON only. Evaluate each candidate criterion independently.\n\n"
        "A valid criterion must be atomic, yes/no decidable, relevant to the user query or task type, "
        "and non-hallucinated. Do not reject medical safety, evidence/source quality, freshness, "
        "search intent, writing structure, or instruction-following constraints when they are standard "
        "for the task type.\n\n"
        f"Task-specific policy:\n{domain_policy(data_source)}"
    )
    rubric_lines = "\n".join(f"{idx}. {rubric}" for idx, rubric in enumerate(rubrics))
    user = (
        f"Data source/task type: {data_source}\n\n"
        f"User query:\n{query}\n\n"
        f"Candidate criteria:\n{rubric_lines}\n\n"
        "Return JSON only with this schema:\n"
        '{"rubrics":[{"idx":0,"valid":true,"atomic":true,"decidable":true,'
        '"relevant":true,"non_hallucinated":true,"reason":"short reason"}]}\n'
        f"Return exactly {len(rubrics)} decisions, using idx 0 through {len(rubrics) - 1}. "
        f"Criteria beyond the first {max_rubrics} otherwise-valid items may be marked invalid "
        'with reason "max_rubrics_exceeded".'
    )
    raw = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    parsed = parse_decisions(raw, len(rubrics))
    return parsed


def parse_decisions(raw: str, n_rubrics: int) -> list[dict[str, Any]]:
    candidate = extract_json(raw) or raw.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return [
            {
                "idx": idx,
                "valid": False,
                "atomic": False,
                "decidable": False,
                "relevant": False,
                "non_hallucinated": False,
                "reason": "invalid_json_response",
                "raw_response": raw,
            }
            for idx in range(n_rubrics)
        ]
    rows = parsed.get("rubrics") if isinstance(parsed, dict) else parsed
    by_idx = {}
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict) and isinstance(item.get("idx"), int):
                by_idx[int(item["idx"])] = item
    output = []
    for idx in range(n_rubrics):
        item = by_idx.get(idx)
        if not item:
            output.append(
                {
                    "idx": idx,
                    "valid": False,
                    "atomic": False,
                    "decidable": False,
                    "relevant": False,
                    "non_hallucinated": False,
                    "reason": "missing_decision",
                    "raw_response": raw,
                }
            )
            continue
        valid = all(bool(item.get(key)) for key in ("valid", "atomic", "decidable", "relevant", "non_hallucinated"))
        output.append(
            {
                "idx": idx,
                "valid": valid,
                "atomic": bool(item.get("atomic")),
                "decidable": bool(item.get("decidable")),
                "relevant": bool(item.get("relevant")),
                "non_hallucinated": bool(item.get("non_hallucinated")),
                "reason": str(item.get("reason") or ("api_pass" if valid else "api_reject")),
            }
        )
    return output


def summarize(
    audit_rows: list[dict[str, Any]],
    sample_path: Path,
    audit_path: Path,
    verified_path: Path,
    budget: dict[str, Any],
) -> dict[str, Any]:
    n_input = sum(int(row["n_input"]) for row in audit_rows)
    n_valid = sum(int(row["n_valid"]) for row in audit_rows)
    reason_counts = Counter()
    for row in audit_rows:
        for decision in row["verifier_decisions"]:
            if not decision.get("valid"):
                reason_counts[str(decision.get("reason") or "api_reject")] += 1
    nonempty = sum(1 for row in audit_rows if int(row["n_valid"]) > 0)
    return {
        "records_sampled": len(audit_rows),
        "records_nonempty_after_filter": nonempty,
        "input_rubrics": n_input,
        "valid_rubrics": n_valid,
        "validity": n_valid / max(n_input, 1),
        "invalid_reason_counts": dict(reason_counts),
        "budget": budget,
        "paths": {
            "sample": str(sample_path),
            "audit": str(audit_path),
            "verified": str(verified_path),
        },
    }


def pick_first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
