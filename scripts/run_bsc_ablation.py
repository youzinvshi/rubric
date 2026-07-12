#!/usr/bin/env python3
"""Run offline BSC reward ablations on model evaluation-criteria outputs."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blindspot_rl.reward_bsc import (  # noqa: E402
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    compute_metrics,
    parse_rubrics,
    safe_float,
)
from scripts.bsc_diagnose import build_record_verifier  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402


ABLATIONS = {
    "full": {"weights": (1.0, 0.5, 0.5), "use_verifier": True},
    "no_red": {"weights": (1.0, 0.5, 0.0), "use_verifier": True},
    "no_valid": {"weights": (1.0, 0.0, 0.5), "use_verifier": False},
    "no_verifier": {"weights": (1.0, 0.5, 0.5), "use_verifier": False},
    "cov_only": {"weights": (1.0, 0.0, 0.0), "use_verifier": False},
}


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

    summary_rows = []
    per_variant_dir = args.output_dir / "variants"
    per_variant_dir.mkdir(parents=True, exist_ok=True)
    for name, variant_config in ABLATIONS.items():
        weights = variant_config["weights"]
        use_verifier = bool(variant_config["use_verifier"])
        rows = []
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
            if not use_verifier:
                verifier = None
                verifier_source = "disabled"
            metrics = compute_metrics(
                response=response,
                gold_rubrics=gold,
                prompt=prompt,
                verifier=verifier,
                embedder=embedder,
                coverage_tau=args.coverage_tau,
                redundancy_tau=args.redundancy_tau,
                weights=weights,
            )
            rows.append(
                {
                    "idx": idx,
                    "variant": name,
                    "verifier_source": verifier_source,
                    **{
                        key: safe_float(value) if isinstance(value, float) else value
                        for key, value in metrics.as_dict().items()
                    },
                }
            )
        summary = summarize(rows)
        summary["variant"] = name
        summary["ablation_family"] = "reward_component"
        summary["input"] = str(args.input)
        summary["input_sha256"] = file_sha256(args.input)
        summary["embedding_model"] = args.embedding_model
        summary["coverage_tau"] = args.coverage_tau
        summary["redundancy_tau"] = args.redundancy_tau
        summary["weights"] = {
            "coverage": weights[0],
            "validity": weights[1],
            "redundancy": weights[2],
        }
        summary["use_verifier"] = use_verifier
        summary_rows.append(summary)
        write_json(per_variant_dir / f"{name}_summary.json", summary)
        write_csv(per_variant_dir / f"{name}_per_item.csv", rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation_summary.csv", summary_rows)
    write_markdown(args.output_dir / "ablation_summary.md", summary_rows)
    print(f"Wrote ablation summaries to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline BSC reward ablations.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "bsc_ablation", type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--redundancy-tau", default=0.85, type=float)
    return parser.parse_args()


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
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
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("records") or data.get("data") or [data]
        yield from records
        return
    raise ValueError(f"Unsupported input format: {path}")


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["coverage", "blind", "redundancy", "validity", "hallucination", "reward"]
    summary: dict[str, Any] = {"n": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[f"mean_{name}"] = sum(values) / max(len(values), 1)
    verifier_source_counts: dict[str, int] = {}
    for row in rows:
        verifier_source = str(row.get("verifier_source", "none") or "none")
        verifier_source_counts[verifier_source] = verifier_source_counts.get(verifier_source, 0) + 1
    summary["verifier_source_counts"] = verifier_source_counts
    summary["verifier_source"] = "mixed" if len(verifier_source_counts) > 1 else next(iter(verifier_source_counts), "none")
    return summary


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Variant | Cov | Blind | Red | Hall | Reward |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    fmt(row["mean_coverage"]),
                    fmt(row["mean_blind"]),
                    fmt(row["mean_redundancy"]),
                    fmt(row["mean_hallucination"]),
                    fmt(row["mean_reward"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
