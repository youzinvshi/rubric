#!/usr/bin/env python3
"""Run an isolated offline smoke test for the post-generation BSC chain.

This script does not create paper evidence and does not touch the real minimal
claim BSC output directory. It uses hard-gold evaluation dimensions as
model-like outputs to exercise the stages that run after API generation:
- generated-criteria validation
- verifier flag attachment
- verified-criteria validation
- gold/prediction join
- BSC diagnostics
- threshold sweep
- bootstrap CI
- main-table summarization
"""

from __future__ import annotations

import argparse
import csv
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

from blindspot_rl.reward_bsc import TokenOverlapEmbedder, compute_metrics, parse_rubrics, safe_float  # noqa: E402
from scripts.bootstrap_metric_ci import build_ci_report, to_markdown as ci_to_markdown, write_csv as write_ci_csv  # noqa: E402
from scripts.bsc_diagnose import build_record_verifier, summarize as summarize_bsc, write_outputs as write_bsc_outputs  # noqa: E402
from scripts.filter_rubrics_with_verifier import RuleMetaVerifier  # noqa: E402
from scripts.prepare_bsc_eval import (  # noqa: E402
    build_gold_map,
    build_prediction_map,
    compact_record,
    join_blockers,
    load_records,
    write_jsonl,
)
from scripts.summarize_experiments import to_markdown as table_to_markdown, write_csv as write_table_csv  # noqa: E402
from scripts.sweep_bsc_thresholds import run_one_setting, to_markdown as sweep_to_markdown, write_csv as write_sweep_csv  # noqa: E402
from scripts.validate_rubric_outputs import (  # noqa: E402
    summarize as summarize_validation,
    to_markdown as validation_to_markdown,
    validate_record,
    write_jsonl as write_validation_jsonl,
)


def main() -> None:
    args = parse_args()
    report = run_smoke(
        gold_path=args.gold,
        output_dir=args.output_dir,
        limit=args.limit,
        min_joined=args.min_joined,
        data_source=args.data_source,
    )
    print(
        "Minimal BSC chain smoke "
        f"ok={report['ok']} n_joined={report['n_joined']} "
        f"Cov={report['bsc']['mean_coverage']:.4f} Blind={report['bsc']['mean_blind']:.4f}"
    )
    if not report["ok"]:
        raise SystemExit("; ".join(report["blockers"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated offline BSC chain smoke.")
    parser.add_argument("--gold", default=ROOT / "data" / "processed" / "rubricbench_gold.jsonl", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "outputs" / "minimal_bsc_chain_smoke", type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--min-joined", default=100, type=int)
    parser.add_argument("--data-source", default="rubricbench")
    return parser.parse_args()


def run_smoke(
    gold_path: Path,
    output_dir: Path,
    limit: int | None = 100,
    min_joined: int = 100,
    data_source: str = "rubricbench",
) -> dict[str, Any]:
    paths = smoke_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)

    gold_by_query = build_gold_map(list(load_records(gold_path)))
    predictions = make_gold_model_predictions(gold_by_query, limit=limit)
    write_jsonl(paths["model_rubrics"], predictions)

    model_validation = write_validation(
        predictions,
        paths["model_validation_dir"],
        require_valid_flags=False,
    )
    verified = attach_rule_verifier_flags(predictions)
    write_jsonl(paths["verified_model_rubrics"], verified)
    write_jsonl(paths["verifier_stats"], verifier_stats(verified))
    verified_validation = write_validation(
        verified,
        paths["verified_validation_dir"],
        require_valid_flags=True,
    )

    joined_rows, join_report = join_for_bsc(
        gold_by_query=gold_by_query,
        predictions=verified,
        output_path=paths["bsc_eval"],
        report_path=paths["join_report"],
        data_source=data_source,
        min_joined=min_joined,
        limit=limit,
    )
    per_item, bsc_summary = compute_bsc(joined_rows, paths["bsc_dir"])
    sweep_rows = write_sweep(joined_rows, paths["sweep_dir"])
    ci_report = write_ci(paths["bsc_per_item"], paths["ci_dir"])
    table_rows = write_main_table(bsc_summary, ci_report, paths["main_table_csv"], paths["main_table_md"])

    blockers = []
    if not model_validation["ok"]:
        blockers.append("model rubric validation failed")
    if not verified_validation["ok"]:
        blockers.append("verified rubric validation failed")
    blockers.extend(join_report["blockers"])
    if bsc_summary["mean_coverage"] != 1.0:
        blockers.append(f"gold-as-model BSC coverage is not 1.0: {bsc_summary['mean_coverage']}")
    if bsc_summary["mean_blind"] != 0.0:
        blockers.append(f"gold-as-model BSC blind is not 0.0: {bsc_summary['mean_blind']}")

    report = {
        "ok": not blockers,
        "smoke": "minimal_bsc_post_generation_chain",
        "gold": str(gold_path),
        "output_dir": str(output_dir),
        "n_joined": join_report["n_joined"],
        "blockers": blockers,
        "artifacts": {name: str(path) for name, path in paths.items() if path.suffix},
        "model_validation": model_validation,
        "verified_validation": verified_validation,
        "join_report": join_report,
        "bsc": bsc_summary,
        "sweep_settings": len(sweep_rows),
        "ci_metrics": len(ci_report["metrics"]),
        "main_table_rows": len(table_rows),
    }
    paths["smoke_report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def smoke_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "model_rubrics": output_dir / "data" / "model_rubrics.jsonl",
        "verified_model_rubrics": output_dir / "data" / "model_rubrics_verified.jsonl",
        "verifier_stats": output_dir / "verifier" / "model_rubrics_stats.jsonl",
        "model_validation_dir": output_dir / "validation" / "model_rubrics",
        "verified_validation_dir": output_dir / "validation" / "model_rubrics_verified",
        "bsc_eval": output_dir / "data" / "bsc_eval.jsonl",
        "join_report": output_dir / "bsc_join_report.json",
        "bsc_dir": output_dir / "bsc",
        "bsc_per_item": output_dir / "bsc" / "per_item.csv",
        "bsc_summary": output_dir / "bsc" / "summary.json",
        "sweep_dir": output_dir / "bsc_sweep",
        "ci_dir": output_dir / "bsc_ci",
        "main_table_csv": output_dir / "main_table.csv",
        "main_table_md": output_dir / "main_table.md",
        "smoke_report": output_dir / "smoke_report.json",
    }


def make_gold_model_predictions(gold_by_query: dict[str, dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for gold in gold_by_query.values():
        rows.append({"query": gold["query"], "model": "gold_as_model", "rubrics": list(gold["gold_rubrics"])})
        if limit and len(rows) >= limit:
            break
    return rows


def write_validation(records: list[dict[str, Any]], output_dir: Path, require_valid_flags: bool) -> dict[str, Any]:
    rows = [
        validate_record(
            record,
            min_rubrics=1,
            max_rubrics=50,
            require_valid_flags=require_valid_flags,
            allow_exact_duplicates=True,
            allow_generic_terms=True,
            allow_semantic_redundancy=True,
        )
        for record in records
    ]
    report = summarize_validation(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "validation_report.md").write_text(validation_to_markdown(report), encoding="utf-8")
    write_validation_jsonl(output_dir / "per_record.jsonl", rows)
    return report


def attach_rule_verifier_flags(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verifier = RuleMetaVerifier()
    out = []
    for record in records:
        rubrics = parse_rubrics(record.get("rubrics"), dedupe=False)
        flags = [int(verifier.judge(rubric, prompt=record.get("query", ""))) for rubric in rubrics]
        new_record = dict(record)
        new_record["rubrics_before_filter"] = rubrics
        new_record["verified_rubrics"] = [rubric for rubric, flag in zip(rubrics, flags) if flag]
        new_record["valid_flags"] = flags
        new_record["verifier_source"] = "valid_flags"
        out.append(new_record)
    return out


def verifier_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        flags = record.get("valid_flags", [])
        rows.append(
            {
                "query": record.get("query", ""),
                "teacher": record.get("teacher", ""),
                "n_input": len(flags),
                "n_valid": sum(int(flag) for flag in flags),
                "validity": sum(int(flag) for flag in flags) / max(len(flags), 1),
            }
        )
    return rows


def join_for_bsc(
    gold_by_query: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_path: Path,
    report_path: Path,
    data_source: str,
    min_joined: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_by_query = build_prediction_map(predictions, model="gold_as_model")
    rows = []
    missing_predictions = []
    for query_key, gold in gold_by_query.items():
        prediction = pred_by_query.get(query_key)
        if prediction is None:
            missing_predictions.append(query_key)
            continue
        rows.append(
            compact_record(
                {
                    "query": gold["query"],
                    "gold_rubrics": gold["gold_rubrics"],
                    "response": prediction["rubrics"],
                    "model": prediction.get("model", "gold_as_model"),
                    "data_source": data_source,
                    "valid_flags": prediction.get("valid_flags"),
                    "verifier_source": prediction.get("verifier_source"),
                }
            )
        )
        if limit and len(rows) >= limit:
            break
    blockers = join_blockers(len(rows), min_joined=min_joined)
    report = {
        "ok": not blockers,
        "n_gold": len(gold_by_query),
        "n_predictions": len(pred_by_query),
        "n_joined": len(rows),
        "n_missing_predictions": len(missing_predictions),
        "min_joined": min_joined,
        "model": "gold_as_model",
        "blockers": blockers,
    }
    write_jsonl(output_path, rows)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, report


def compute_bsc(rows: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    embedder = TokenOverlapEmbedder()
    per_item = []
    for idx, record in enumerate(rows):
        verifier, verifier_source = build_record_verifier(record, response=record["response"], idx=idx)
        metrics = compute_metrics(
            response=record["response"],
            gold_rubrics=record["gold_rubrics"],
            prompt=record.get("query", ""),
            verifier=verifier,
            embedder=embedder,
            coverage_tau=0.99,
            redundancy_tau=0.99,
        )
        per_item.append(
            {
                "idx": idx,
                "data_source": record.get("data_source", ""),
                "verifier_source": verifier_source,
                "prompt": record.get("query", ""),
                **{key: safe_float(value) if isinstance(value, float) else value for key, value in metrics.as_dict().items()},
            }
        )
    summary = summarize_bsc(
        per_item,
        input_path=output_dir.parent / "data" / "bsc_eval.jsonl",
        embedding_model="token-overlap",
        coverage_tau=0.99,
        redundancy_tau=0.99,
        weights=(1.0, 0.5, 0.5),
    )
    write_bsc_outputs(per_item, summary, output_dir)
    return per_item, summary


def write_sweep(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    settings = []
    embedder = TokenOverlapEmbedder()
    for coverage_tau in [0.70, 0.75, 0.80]:
        for redundancy_tau in [0.80, 0.85, 0.90]:
            summary = run_one_setting(rows, embedder, coverage_tau, redundancy_tau, (1.0, 0.5, 0.5))
            settings.append({"coverage_tau": coverage_tau, "redundancy_tau": redundancy_tau, **summary})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sweep_csv(output_dir / "threshold_sweep.csv", settings)
    (output_dir / "threshold_sweep.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "threshold_sweep.md").write_text(sweep_to_markdown(settings), encoding="utf-8")
    return settings


def write_ci(per_item_csv: Path, output_dir: Path) -> dict[str, Any]:
    with per_item_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    report = build_ci_report(
        rows,
        metrics=["coverage", "blind", "redundancy", "hallucination", "reward"],
        n_boot=100,
        seed=13,
        confidence=0.95,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bootstrap_ci.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_ci_csv(output_dir / "bootstrap_ci.csv", report["metrics"])
    (output_dir / "bootstrap_ci.md").write_text(ci_to_markdown(report), encoding="utf-8")
    return report


def write_main_table(
    bsc_summary: dict[str, Any],
    ci_report: dict[str, Any],
    output_csv: Path,
    output_md: Path,
) -> list[dict[str, Any]]:
    row = {
        "method": "gold_as_model",
        "bsc_status": "pass",
        "bsc_n": bsc_summary["n"],
        "cov": bsc_summary["mean_coverage"],
        "blind": bsc_summary["mean_blind"],
        "red": bsc_summary["mean_redundancy"],
        "hall": bsc_summary["mean_hallucination"],
        "reward": bsc_summary["mean_reward"],
        "bsc_ci_status": "pass",
    }
    for metric in ci_report["metrics"]:
        prefix = {
            "coverage": "cov",
            "blind": "blind",
            "redundancy": "red",
            "hallucination": "hall",
            "reward": "reward",
        }.get(metric["metric"])
        if prefix:
            row[f"{prefix}_ci_lower"] = metric.get("ci_lower")
            row[f"{prefix}_ci_upper"] = metric.get("ci_upper")
    rows = [row]
    write_table_csv(output_csv, rows)
    output_md.write_text(table_to_markdown(rows), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
