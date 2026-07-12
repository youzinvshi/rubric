#!/usr/bin/env python3
"""Evaluate BSC under fixed generated-criteria budgets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import (  # noqa: E402
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    compute_metrics,
    parse_rubrics,
    safe_float,
)


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")
    embedder = TokenOverlapEmbedder() if args.embedding_model == "token-overlap" else SentenceTransformerEmbedder(args.embedding_model)
    rows = evaluate_budget_curve(
        records,
        budgets=args.k,
        embedder=embedder,
        coverage_tau=args.coverage_tau,
        redundancy_tau=args.redundancy_tau,
    )
    write_outputs(args.output_dir, rows, args)
    print(f"Wrote criteria budget curve for K={args.k} to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coverage vs generated-criteria budget K.")
    parser.add_argument("--input", required=True, type=Path, help="BSC eval JSONL/JSON/parquet.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 8, 10, 15])
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--redundancy-tau", default=0.85, type=float)
    return parser.parse_args()


def evaluate_budget_curve(
    records: list[dict[str, Any]],
    budgets: list[int],
    embedder: Any,
    coverage_tau: float,
    redundancy_tau: float,
) -> list[dict[str, Any]]:
    rows = []
    for budget in sorted(set(budgets)):
        per_item = []
        for record in records:
            prompt = str(pick_first(record, "prompt", "query", "instruction") or "")
            gold = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold"), dedupe=True)
            gen = parse_rubrics(
                pick_first(record, "response", "model_rubrics", "generated_rubrics", "prediction", "output"),
                dedupe=False,
            )[:budget]
            metrics = compute_metrics(
                response=gen,
                gold_rubrics=gold,
                prompt=prompt,
                embedder=embedder,
                coverage_tau=coverage_tau,
                redundancy_tau=redundancy_tau,
            )
            per_item.append(metrics.as_dict())
        rows.append(summarize_budget(budget, per_item))
    return rows


def summarize_budget(budget: int, per_item: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["coverage", "blind", "redundancy", "validity", "hallucination", "reward"]
    row: dict[str, Any] = {"k": budget, "n": len(per_item)}
    for name in metric_names:
        values = [float(item[name]) for item in per_item]
        row[f"mean_{name}"] = safe_float(sum(values) / max(len(values), 1))
    row["total_gold"] = sum(int(item["n_gold"]) for item in per_item)
    row["total_gen"] = sum(int(item["n_gen"]) for item in per_item)
    return row


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


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "coverage_by_k.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "input": str(args.input),
        "embedding_model": args.embedding_model,
        "coverage_tau": args.coverage_tau,
        "redundancy_tau": args.redundancy_tau,
        "budgets": args.k,
        "rows": rows,
    }
    (output_dir / "coverage_by_k.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
