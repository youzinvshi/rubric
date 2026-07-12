#!/usr/bin/env python3
"""Estimate API calls, tokens, cost, and rate-limit time for evaluation-criteria elicitation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You elicit evaluation criteria. Output JSON only: a list of strings. "
    "Each criterion must be atomic, directly relevant to the query, verifiable as yes/no, "
    "non-overlapping, and must avoid generic criteria unless explicitly required."
)

DEFAULT_USER_TEMPLATE = "Elicit 6-10 atomic evaluation criteria for the following query.\n\nQuery:\n{query}"


def main() -> None:
    args = parse_args()
    try:
        queries, unit_counts = load_query_units(args.input, args.unit_field, args.unit_multiplier_field)
        if args.limit:
            queries = queries[: args.limit]
            unit_counts = unit_counts[: args.limit]
        providers = load_jsonl(args.providers)
        done = load_done(args.resume_output, args.method_key) if args.resume_output else set()
        report = estimate_budget(
            queries=queries,
            unit_counts=unit_counts,
            providers=providers,
            done=done,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
            calls_per_record_per_provider=args.calls_per_record_per_provider,
            default_qpm=args.default_qpm,
            default_tpm=args.default_tpm,
            max_calls=args.max_calls,
            max_total_tokens=args.max_total_tokens,
            max_cost_usd=args.max_cost_usd,
            max_wallclock_minutes_serial=args.max_wallclock_minutes_serial,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = invalid_budget_report(args, exc)
    report["contract"] = budget_contract(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Estimated {report['total']['calls']} calls; ok={report['ok']} report={args.output}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate API budget for real BlindSpot-RL runs.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--providers", required=True, type=Path)
    parser.add_argument("--resume-output", type=Path, help="Existing output JSONL to subtract completed pairs.")
    parser.add_argument("--method-key", default="method", help="Method field in resume output; use teacher for teacher data.")
    parser.add_argument("--unit-field", help="Optional record field whose item count determines API calls, e.g. rubrics.")
    parser.add_argument(
        "--unit-multiplier-field",
        help="Optional second record field multiplied with --unit-field, e.g. candidates.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calls-per-record-per-provider", type=int, default=1)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    parser.add_argument("--default-qpm", type=float, default=60.0)
    parser.add_argument("--default-tpm", type=float, default=60000.0)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--max-wallclock-minutes-serial", type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def budget_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = {
        "input": str(args.input),
        "input_sha256": file_sha256(args.input) if args.input.exists() else "",
        "providers": str(args.providers),
        "providers_sha256": file_sha256(args.providers) if args.providers.exists() else "",
        "resume_output": str(args.resume_output) if args.resume_output else None,
        "resume_output_sha256": file_sha256(args.resume_output) if args.resume_output and args.resume_output.exists() else "",
        "method_key": args.method_key,
        "unit_field": args.unit_field,
        "unit_multiplier_field": args.unit_multiplier_field,
        "limit": args.limit,
        "calls_per_record_per_provider": args.calls_per_record_per_provider,
    }
    return contract


def invalid_budget_report(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "n_queries": 0,
        "n_units": 0,
        "n_providers": 0,
        "calls_per_record_per_provider": args.calls_per_record_per_provider,
        "limits": {
            "max_calls": args.max_calls,
            "max_total_tokens": args.max_total_tokens,
            "max_cost_usd": args.max_cost_usd,
            "max_wallclock_minutes_serial": args.max_wallclock_minutes_serial,
        },
        "blockers": [f"budget input is not readable: {format_load_error(exc)}"],
        "providers": [],
        "total": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "min_minutes_by_rate_limits": 0.0,
            "estimated_wallclock_minutes_serial": 0.0,
        },
    }


def format_load_error(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON at line {exc.lineno} column {exc.colno}"
    return str(exc)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_budget(
    queries: list[str],
    providers: list[dict[str, Any]],
    unit_counts: list[int] | None = None,
    done: set[tuple[str, str]] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    calls_per_record_per_provider: int = 1,
    default_qpm: float = 60.0,
    default_tpm: float = 60000.0,
    max_calls: int | None = None,
    max_total_tokens: int | None = None,
    max_cost_usd: float | None = None,
    max_wallclock_minutes_serial: float | None = None,
) -> dict[str, Any]:
    done = done or set()
    unit_counts = unit_counts or [1] * len(queries)
    if len(unit_counts) != len(queries):
        raise ValueError("unit_counts must have the same length as queries")
    prompt_tokens_by_query = [
        estimate_tokens(system_prompt) + estimate_tokens(user_template.format(query=query)) for query in queries
    ]
    provider_rows = []
    totals = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "min_minutes_by_rate_limits": 0.0,
    }
    for provider in providers:
        name = str(provider.get("name", "unknown"))
        max_tokens = int(provider.get("max_tokens", 1200))
        pending_prompt_tokens = 0
        pending_records = 0
        pending_units = 0
        for query, prompt_tokens, unit_count in zip(queries, prompt_tokens_by_query, unit_counts):
            if (query, name) in done:
                continue
            pending_records += 1
            pending_units += unit_count
            pending_prompt_tokens += prompt_tokens * unit_count
        calls = pending_units * calls_per_record_per_provider
        prompt_tokens = pending_prompt_tokens * calls_per_record_per_provider
        completion_tokens = calls * max_tokens
        total_tokens = prompt_tokens + completion_tokens
        qpm = float(provider.get("qpm", default_qpm) or default_qpm)
        tpm = float(provider.get("tpm", default_tpm) or default_tpm)
        minutes_by_qpm = calls / qpm if qpm > 0 else math.inf
        minutes_by_tpm = total_tokens / tpm if tpm > 0 else math.inf
        min_minutes = max(minutes_by_qpm, minutes_by_tpm)
        input_cost = float(provider.get("input_cost_per_1k", 0.0) or 0.0)
        output_cost = float(provider.get("output_cost_per_1k", 0.0) or 0.0)
        cost = prompt_tokens / 1000.0 * input_cost + completion_tokens / 1000.0 * output_cost
        row = {
            "name": name,
            "model": provider.get("model", ""),
            "pending_records": pending_records,
            "pending_units": pending_units,
            "calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "qpm": qpm,
            "tpm": tpm,
            "min_minutes_by_rate_limits": min_minutes,
            "estimated_cost_usd": cost,
        }
        provider_rows.append(row)
        for key in ["calls", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"]:
            totals[key] += row[key]
        totals["min_minutes_by_rate_limits"] = max(totals["min_minutes_by_rate_limits"], min_minutes)

    totals["estimated_wallclock_minutes_serial"] = sum(row["min_minutes_by_rate_limits"] for row in provider_rows)
    limits = {
        "max_calls": max_calls,
        "max_total_tokens": max_total_tokens,
        "max_cost_usd": max_cost_usd,
        "max_wallclock_minutes_serial": max_wallclock_minutes_serial,
    }
    blockers = budget_blockers(totals, limits)
    return {
        "ok": not blockers,
        "n_queries": len(queries),
        "n_units": sum(unit_counts),
        "n_providers": len(providers),
        "calls_per_record_per_provider": calls_per_record_per_provider,
        "limits": limits,
        "blockers": blockers,
        "providers": provider_rows,
        "total": totals,
    }


def budget_blockers(totals: dict[str, Any], limits: dict[str, Any]) -> list[str]:
    checks = [
        ("calls", "max_calls", "API calls"),
        ("total_tokens", "max_total_tokens", "total tokens"),
        ("estimated_cost_usd", "max_cost_usd", "estimated cost USD"),
        ("estimated_wallclock_minutes_serial", "max_wallclock_minutes_serial", "serial wallclock minutes"),
    ]
    blockers = []
    for total_key, limit_key, label in checks:
        limit = limits.get(limit_key)
        if limit is None:
            continue
        value = float(totals.get(total_key, 0) or 0)
        if value > float(limit):
            blockers.append(f"{label} exceeds {limit_key}: {value:g} > {float(limit):g}")
    return blockers


def estimate_tokens(text: str) -> int:
    # Conservative API planning heuristic for English/Chinese mixed prompts.
    return max(1, math.ceil(len(text) / 3.5))


def load_queries(path: Path) -> list[str]:
    return load_query_units(path)[0]


def load_query_units(
    path: Path,
    unit_field: str | None = None,
    unit_multiplier_field: str | None = None,
) -> tuple[list[str], list[int]]:
    rows = load_json_records(path)
    queries = []
    unit_counts = []
    for row in rows:
        query = pick_first(row, "query", "prompt", "instruction")
        if query:
            queries.append(str(query))
            unit_count = count_units(row, unit_field)
            if unit_multiplier_field:
                unit_count *= count_units(row, unit_multiplier_field)
            unit_counts.append(unit_count)
    return queries, unit_counts


def count_units(record: dict[str, Any], unit_field: str | None = None) -> int:
    if not unit_field:
        return 1
    value = record.get(unit_field)
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, tuple):
        return len(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return 1
        if isinstance(parsed, list):
            return len(parsed)
        return 1
    return 1


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: line {exc.lineno} column {exc.colno}") from exc
    if isinstance(data, list):
        return validate_record_list(data, path)
    if not isinstance(data, dict):
        raise ValueError(f"JSON records file must be an object or list: {path}")
    for key in ("records", "data", "examples"):
        if isinstance(data.get(key), list):
            return validate_record_list(data[key], path)
    return [data]


def load_done(path: Path, method_key: str) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    for row in load_jsonl(path):
        query = row.get("query")
        method = row.get(method_key) or row.get("method") or row.get("teacher") or row.get("model")
        if query and method:
            done.add((str(query), str(method)))
    return done


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record must be an object at line {line_no}: {path}")
            rows.append(row)
    return rows


def validate_record_list(rows: list[Any], path: Path) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"JSON record must be an object at index {index}: {path}")
    return rows


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def to_markdown(report: dict[str, Any]) -> str:
    total = report["total"]
    lines = [
        "# BlindSpot-RL API Budget Estimate",
        "",
        f"- Queries: `{report['n_queries']}`",
        f"- Units: `{report.get('n_units', report['n_queries'])}`",
        f"- Providers: `{report['n_providers']}`",
        f"- Total calls: `{total['calls']}`",
        f"- Total tokens: `{total['total_tokens']}`",
        f"- Estimated cost USD: `${total['estimated_cost_usd']:.4f}`",
        f"- Budget ok: `{report.get('ok', True)}`",
        f"- Min wallclock if providers run in parallel: `{total['min_minutes_by_rate_limits']:.2f}` minutes",
        f"- Min wallclock if providers run serially: `{total['estimated_wallclock_minutes_serial']:.2f}` minutes",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in report.get("blockers", [])] or ["- none"])
    lines.extend(
        [
            "",
            "## Limits",
            "",
        ]
    )
    limits = report.get("limits", {})
    if limits:
        for key in ["max_calls", "max_total_tokens", "max_cost_usd", "max_wallclock_minutes_serial"]:
            lines.append(f"- {key}: `{limits.get(key)}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Contract",
            "",
        ]
    )
    contract = report.get("contract", {})
    if contract:
        for key in [
            "input",
            "input_sha256",
            "providers",
            "providers_sha256",
            "resume_output",
            "resume_output_sha256",
            "method_key",
            "unit_field",
            "unit_multiplier_field",
            "limit",
            "calls_per_record_per_provider",
        ]:
            lines.append(f"- {key}: `{contract.get(key)}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Provider Breakdown",
            "",
            "",
        ]
    )
    lines.extend([
        "| Provider | Pending Records | Pending Units | Calls | Prompt Tok | Completion Tok | Total Tok | QPM | TPM | Min Minutes | Cost USD |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in report["providers"]:
        lines.append(
            f"| `{row['name']}` | {row.get('pending_records', 0)} | {row.get('pending_units', row['calls'])} | "
            f"{row['calls']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {row['total_tokens']} | {row['qpm']} | {row['tpm']} | "
            f"{row['min_minutes_by_rate_limits']:.2f} | ${row['estimated_cost_usd']:.4f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
