#!/usr/bin/env python3
"""Batched, budget-gated model evaluation-criteria elicitation over a full query pool.

Walks the query pool in cumulative ``--limit`` windows so each batch generates at
most ``--target-per-batch`` pending (query, method) pairs. Before every paid
elicitation batch it runs the offline budget estimator with ``--strict`` and the
same ``--max-total-tokens`` ceiling, writing the exact budget report path that
``generate_model_rubrics.py`` requires. This keeps every batch fail-closed under
the per-batch token gate while resuming into a single output file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    pool = load_pool_queries(args.input)
    providers = load_provider_names(args.providers)
    if not providers:
        raise SystemExit(f"No providers found in: {args.providers}")
    if not pool:
        raise SystemExit(f"No queries found in pool: {args.input}")
    total_unique = len(dict.fromkeys(pool))
    print(
        f"Pool: {len(pool)} records, {total_unique} unique queries, "
        f"{len(providers)} providers -> {args.input}"
    )

    # In dry-run mode the real output file never grows, so budget estimates would
    # not reflect earlier planned batches. Mirror the growing output into a temp
    # resume file so each planned batch is estimated against simulated state.
    resume_target = args.output
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="batched_gen_dry_")) / "sim_model_rubrics.jsonl"
        seed_rows = read_rows(args.output)
        with tmp.open("w", encoding="utf-8") as f:
            for row in seed_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        resume_target = tmp

    batch_no = 0
    while True:
        done = load_done(resume_target)
        remaining = count_pending(pool, providers, done, limit=len(pool))
        if remaining == 0:
            print(f"All pool queries already generated (done={len(done)}). Nothing to do.")
            break
        limit = choose_limit(pool, providers, done, target=args.target_per_batch)
        pending = count_pending(pool, providers, done, limit=limit)
        batch_no += 1
        print(
            f"\n=== Batch {batch_no}: limit={limit} pending={pending} "
            f"(done={len(done)}, remaining_total={remaining}) ==="
        )
        run_budget_estimate(args, limit, resume_target)
        report = json.loads(args.budget_report.read_text(encoding="utf-8"))
        if report.get("ok") is not True or report.get("blockers"):
            raise SystemExit(
                f"Budget gate blocked batch {batch_no} at limit={limit}: "
                f"{report.get('blockers')}"
            )
        print(
            f"Budget gate OK: calls={report['total']['calls']} "
            f"tokens={report['total']['total_tokens']}"
        )
        if args.dry_run:
            # Append simulated completions for the pending queries in this window.
            append_simulated(resume_target, pool, providers, done, limit)
            if count_pending(pool, providers, load_done(resume_target), limit=len(pool)) == 0:
                print("\nDry run complete: full pool covered by planned batches.")
                break
            continue
        run_generation(args, limit)

    if not args.dry_run:
        final_done = load_done(args.output)
        print(f"\nGeneration finished: {len(final_done)} (query, method) pairs in {args.output}")


def append_simulated(
    resume_target: Path,
    pool: list[str],
    providers: list[str],
    done: set[tuple[str, str]],
    limit: int,
) -> None:
    seen: set[str] = set()
    with resume_target.open("a", encoding="utf-8") as f:
        for query in pool[:limit]:
            if query in seen:
                continue
            seen.add(query)
            for provider in providers:
                if (query, provider) not in done:
                    f.write(json.dumps({"query": query, "teacher": provider}, ensure_ascii=False) + "\n")


def choose_limit(pool: list[str], providers: list[str], done: set[tuple[str, str]], target: int) -> int:
    """Smallest cumulative limit whose pending count reaches target (or pool end)."""
    seen: set[str] = set()
    pending = 0
    for idx, query in enumerate(pool, start=1):
        if query in seen:
            continue
        seen.add(query)
        pending += sum(1 for provider in providers if (query, provider) not in done)
        if pending >= target:
            return idx
    return len(pool)


def count_pending(pool: list[str], providers: list[str], done: set[tuple[str, str]], limit: int) -> int:
    seen: set[str] = set()
    pending = 0
    for query in pool[:limit]:
        if query in seen:
            continue
        seen.add(query)
        pending += sum(1 for provider in providers if (query, provider) not in done)
    return pending


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_budget_estimate(args: argparse.Namespace, limit: int, resume_target: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "estimate_api_budget.py"),
        "--input", str(args.input),
        "--providers", str(args.providers),
        "--resume-output", str(resume_target),
        "--method-key", "teacher",
        "--calls-per-record-per-provider", "1",
        "--limit", str(limit),
        "--default-qpm", str(args.default_qpm),
        "--default-tpm", str(args.default_tpm),
        "--max-calls", str(args.max_calls),
        "--max-total-tokens", str(args.max_total_tokens),
        "--max-cost-usd", str(args.max_cost_usd),
        "--max-wallclock-minutes-serial", str(args.max_wallclock_minutes_serial),
        "--output", str(args.budget_report),
        "--output-md", str(args.budget_report.with_suffix(".md")),
        "--strict",
    ]
    print("+ " + " ".join(cmd))
    # strict estimator exits 1 when blocked; we surface the report either way.
    subprocess.run(cmd, cwd=ROOT, check=False)


def run_generation(args: argparse.Namespace, limit: int) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_teacher_rubrics.py"),
        "--input", str(args.input),
        "--providers", str(args.providers),
        "--output", str(args.output),
        "--data-source", args.data_source,
        "--limit", str(limit),
        "--resume",
        "--sleep", str(args.sleep),
        "--on-error", args.on_error,
        "--require-budget-report", str(args.budget_report),
        "--require-preflight-report", str(args.preflight_report),
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_pool_queries(path: Path) -> list[str]:
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at line {line_no}: {path}") from exc
            q = row.get("query") or row.get("prompt") or row.get("instruction")
            if q:
                queries.append(str(q))
    return queries


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = row.get("query")
            method = row.get("teacher") or row.get("method") or row.get("model")
            if query and method:
                done.add((str(query), str(method)))
    return done


def load_provider_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid provider JSONL at line {line_no}: {path}") from exc
            name = row.get("name")
            if name:
                names.append(str(name))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched budget-gated evaluation-criteria elicitation.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--providers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-report", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--data-source", default="rubricbench_base_generation")
    parser.add_argument("--method", default="base", help="Provider/method name used in resume dedupe.")
    parser.add_argument("--target-per-batch", type=int, default=170)
    parser.add_argument("--sleep", type=float, default=10.0)
    parser.add_argument("--on-error", choices=("fail", "write_error"), default="fail")
    parser.add_argument("--default-qpm", type=float, default=60.0)
    parser.add_argument("--default-tpm", type=float, default=60000.0)
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--max-total-tokens", type=int, default=2000000)
    parser.add_argument("--max-cost-usd", type=float, default=200.0)
    parser.add_argument("--max-wallclock-minutes-serial", type=float, default=500.0)
    parser.add_argument("--dry-run", action="store_true", help="Only run offline budget estimates per planned batch.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
