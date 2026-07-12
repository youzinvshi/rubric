#!/usr/bin/env python3
"""Offline BSC diagnostic runner.

Input records may be JSONL, JSON list, or parquet. Each record should contain:
- query/prompt/instruction
- gold_rubrics/gold/rubrics_gold
- response/model_rubrics/generated_rubrics/prediction
"""

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

    embedder = (
        TokenOverlapEmbedder()
        if args.embedding_model == "token-overlap"
        else SentenceTransformerEmbedder(args.embedding_model)
    )

    per_item: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt = pick_first(record, "prompt", "query", "instruction") or ""
        gold = parse_rubrics(
            pick_first(record, "gold_rubrics", "gold", "rubrics_gold"),
            dedupe=True,
        )
        response = pick_first(
            record,
            "response",
            "model_rubrics",
            "generated_rubrics",
            "prediction",
            "output",
        )
        verifier, verifier_source = build_record_verifier(record, response=response, idx=idx)

        metrics = compute_metrics(
            response=response,
            gold_rubrics=gold,
            prompt=prompt,
            verifier=verifier,
            embedder=embedder,
            coverage_tau=args.coverage_tau,
            redundancy_tau=args.redundancy_tau,
            weights=(args.w_cov, args.w_valid, args.w_red),
        )
        row = {
            "idx": idx,
            "data_source": record.get("data_source", ""),
            "verifier_source": verifier_source,
            "prompt": prompt,
            **{k: safe_float(v) if isinstance(v, float) else v for k, v in metrics.as_dict().items()},
        }
        per_item.append(row)

    summary = summarize(
        per_item,
        input_path=args.input,
        embedding_model=args.embedding_model,
        coverage_tau=args.coverage_tau,
        redundancy_tau=args.redundancy_tau,
        weights=(args.w_cov, args.w_valid, args.w_red),
    )
    write_outputs(per_item, summary, args.output_dir)
    print_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Blind-Spot Coverage diagnostics.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL/JSON/parquet input file.")
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "bsc", type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--redundancy-tau", default=0.85, type=float)
    parser.add_argument("--w-cov", default=1.0, type=float)
    parser.add_argument("--w-valid", default=0.5, type=float)
    parser.add_argument("--w-red", default=0.5, type=float)
    return parser.parse_args()


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            records = data.get("records") or data.get("data") or [data]
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


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def summarize(
    rows: list[dict[str, Any]],
    input_path: Path | None = None,
    embedding_model: str | None = None,
    coverage_tau: float | None = None,
    redundancy_tau: float | None = None,
    weights: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    metric_names = ["coverage", "blind", "redundancy", "validity", "hallucination", "reward"]
    data_source_counts: dict[str, int] = {}
    verifier_source_counts: dict[str, int] = {}
    for row in rows:
        data_source = str(row.get("data_source", ""))
        data_source_counts[data_source] = data_source_counts.get(data_source, 0) + 1
        verifier_source = str(row.get("verifier_source", "none") or "none")
        verifier_source_counts[verifier_source] = verifier_source_counts.get(verifier_source, 0) + 1

    summary: dict[str, Any] = {"n": len(rows)}
    if input_path is not None:
        summary["input"] = str(input_path)
    if embedding_model is not None:
        summary["embedding_model"] = embedding_model
    if coverage_tau is not None:
        summary["coverage_tau"] = coverage_tau
    if redundancy_tau is not None:
        summary["redundancy_tau"] = redundancy_tau
    if weights is not None:
        summary["weights"] = {
            "coverage": weights[0],
            "validity": weights[1],
            "redundancy": weights[2],
        }
    if data_source_counts:
        summary["data_source_counts"] = data_source_counts
    if verifier_source_counts:
        summary["verifier_source_counts"] = verifier_source_counts
        summary["verifier_source"] = "mixed" if len(verifier_source_counts) > 1 else next(iter(verifier_source_counts))
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[f"mean_{name}"] = sum(values) / max(len(values), 1)
    blind_values = sorted(float(row["blind"]) for row in rows)
    coverage_values = [float(row["coverage"]) for row in rows]
    if blind_values:
        mid = len(blind_values) // 2
        if len(blind_values) % 2:
            summary["median_blind"] = blind_values[mid]
        else:
            summary["median_blind"] = (blind_values[mid - 1] + blind_values[mid]) / 2
    summary["queries_coverage_le_0_5"] = sum(value <= 0.5 for value in coverage_values)
    summary["queries_blind_ge_0_5"] = sum(value >= 0.5 for value in blind_values)
    summary["queries_blind_ge_0_8"] = sum(value >= 0.8 for value in blind_values)
    summary["queries_zero_coverage"] = sum(value == 0.0 for value in coverage_values)
    summary["total_gold"] = sum(int(row["n_gold"]) for row in rows)
    summary["total_gen"] = sum(int(row["n_gen"]) for row in rows)
    summary["mean_n_gold"] = summary["total_gold"] / max(len(rows), 1)
    summary["mean_n_gen"] = summary["total_gen"] / max(len(rows), 1)
    summary["gen_to_gold_ratio"] = (
        summary["total_gen"] / summary["total_gold"] if summary["total_gold"] > 0 else None
    )
    summary["coverage_per_generated_criterion"] = (
        summary.get("mean_coverage", 0.0) / summary["mean_n_gen"] if summary["mean_n_gen"] > 0 else None
    )
    return summary


def build_record_verifier(record: dict[str, Any], response: Any, idx: int) -> tuple[Any | None, str]:
    flags = pick_first(record, "valid_flags", "verifier_flags", "validity_flags")
    if flags is None:
        return None, "none"
    if not isinstance(flags, list):
        raise ValueError(f"valid_flags must be a list for BSC record {idx}")
    gen = parse_rubrics(response, dedupe=False)
    if len(flags) != len(gen):
        raise ValueError(
            f"valid_flags length mismatch for BSC record {idx}: "
            f"{len(flags)} flags for {len(gen)} generated criteria"
        )
    bool_flags = [parse_valid_flag(flag, idx=idx) for flag in flags]
    cursor = {"idx": 0}

    def verifier(_rubric: str, _prompt: str | None = None, **_: Any) -> bool:
        del _rubric, _prompt
        flag = bool_flags[cursor["idx"]]
        cursor["idx"] += 1
        return flag

    return verifier, str(record.get("verifier_source") or "valid_flags")


def parse_valid_flag(value: Any, idx: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "valid"}:
            return True
        if text in {"0", "false", "no", "invalid"}:
            return False
    raise ValueError(f"valid_flags contains non-binary value for BSC record {idx}: {value!r}")


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "per_item.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    print("BSC diagnostic summary")
    print(f"n={summary['n']} total_gold={summary['total_gold']} total_gen={summary['total_gen']}")
    print(f"Cov={summary['mean_coverage']:.4f} Blind={summary['mean_blind']:.4f}")
    print(f"Red={summary['mean_redundancy']:.4f} Hall={summary['mean_hallucination']:.4f}")
    print(f"Reward={summary['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
