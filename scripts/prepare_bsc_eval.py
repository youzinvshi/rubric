#!/usr/bin/env python3
"""Join gold dimensions with model-generated criteria for BSC diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402

try:
    from scripts.budget_gate import file_sha256  # type: ignore  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - used when running this file as a script.
    from budget_gate import file_sha256  # type: ignore  # noqa: E402


def main() -> None:
    args = parse_args()
    gold_records = list(load_records(args.gold))
    prediction_records = list(load_records(args.predictions))
    gold_by_query, gold_stats = build_gold_index(gold_records)
    pred_by_query, prediction_stats = build_prediction_index(prediction_records, model=args.model)

    rows = []
    missing_predictions = [query_key for query_key in gold_by_query if query_key not in pred_by_query]
    unmatched_predictions = [query_key for query_key in pred_by_query if query_key not in gold_by_query]
    joinable_keys = [query_key for query_key in gold_by_query if query_key in pred_by_query]
    valid_flags_mismatch = []
    for query_key, gold in gold_by_query.items():
        prediction = pred_by_query.get(query_key)
        if prediction is None:
            continue
        if args.limit and len(rows) >= args.limit:
            continue
        valid_flags = prediction.get("valid_flags")
        if valid_flags is not None and len(valid_flags) != len(prediction["rubrics"]):
            valid_flags_mismatch.append(query_key)
        rows.append(
            compact_record(
                {
                    "query": gold["query"],
                    "gold_rubrics": gold["gold_rubrics"],
                    "response": prediction["rubrics"],
                    "model": prediction.get("model", args.model or ""),
                    "data_source": args.data_source,
                    "valid_flags": prediction.get("valid_flags"),
                    "verifier_source": prediction.get("verifier_source"),
                }
            )
        )

    query_alignment_exact = not missing_predictions and not unmatched_predictions
    output_truncated_by_limit = args.limit is not None and len(joinable_keys) > len(rows)
    joined_records_with_valid_flags = sum(1 for row in rows if "valid_flags" in row)

    report = {
        "gold": str(args.gold),
        "gold_sha256": file_sha256(args.gold),
        "predictions": str(args.predictions),
        "predictions_sha256": file_sha256(args.predictions),
        "output": str(args.output),
        "output_sha256": "",
        "output_written": False,
        "output_rows_count": 0,
        "output_rows_match_n_joined": False,
        "n_gold_records_raw": len(gold_records),
        "n_prediction_records_raw": len(prediction_records),
        "n_gold": len(gold_by_query),
        "n_predictions": len(pred_by_query),
        "n_gold_join_keys": len(gold_by_query),
        "n_prediction_join_keys": len(pred_by_query),
        "n_joinable": len(joinable_keys),
        "n_joined": len(rows),
        "n_missing_predictions": len(missing_predictions),
        "n_unmatched_predictions": len(unmatched_predictions),
        "missing_prediction_keys_sample": missing_predictions[:10],
        "unmatched_prediction_keys_sample": unmatched_predictions[:10],
        "query_alignment_exact": query_alignment_exact,
        "limit": args.limit,
        "output_truncated_by_limit": output_truncated_by_limit,
        "gold_duplicate_join_key_count": gold_stats["duplicate_join_key_count"],
        "gold_duplicate_record_count": gold_stats["duplicate_record_count"],
        "prediction_duplicate_join_key_count": prediction_stats["duplicate_join_key_count"],
        "prediction_duplicate_record_count": prediction_stats["duplicate_record_count"],
        "gold_skipped_missing_query_count": gold_stats["skipped_missing_query_count"],
        "gold_skipped_empty_rubrics_count": gold_stats["skipped_empty_rubrics_count"],
        "prediction_skipped_missing_query_count": prediction_stats["skipped_missing_query_count"],
        "prediction_skipped_empty_rubrics_count": prediction_stats["skipped_empty_rubrics_count"],
        "prediction_skipped_model_filter_count": prediction_stats["skipped_model_filter_count"],
        "joined_records_with_valid_flags": joined_records_with_valid_flags,
        "all_joined_records_have_valid_flags": joined_records_with_valid_flags == len(rows),
        "valid_flags_length_mismatch_count": len(valid_flags_mismatch),
        "valid_flags_length_mismatch_keys_sample": valid_flags_mismatch[:10],
        "min_joined": args.min_joined,
        "model": args.model or "",
        "blockers": join_blockers(len(rows), args.min_joined),
    }
    report["ok"] = not report["blockers"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        raise SystemExit("No joined BSC records found. Check query keys/model filter.")
    if report["blockers"]:
        raise SystemExit("; ".join(report["blockers"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    report["output_sha256"] = file_sha256(args.output)
    report["output_written"] = True
    report["output_rows_count"] = count_jsonl_records(args.output)
    report["output_rows_match_n_joined"] = report["output_rows_count"] == report["n_joined"]
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Prepared {len(rows)} BSC eval records at {args.output}; "
        f"missing_predictions={len(missing_predictions)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join gold dimensions and generated criteria for BSC eval.")
    parser.add_argument("--gold", required=True, type=Path, help="Gold JSONL/JSON/parquet.")
    parser.add_argument("--predictions", required=True, type=Path, help="Generated criteria JSONL/JSON/parquet.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", help="Optional model/teacher filter in predictions.")
    parser.add_argument("--data-source", default="bsc_eval")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-joined", type=int, help="Fail unless at least this many records join.")
    return parser.parse_args()


def join_blockers(n_joined: int, min_joined: int | None = None) -> list[str]:
    blockers = []
    if n_joined == 0:
        blockers.append("no joined BSC records")
    if min_joined is not None and n_joined < min_joined:
        blockers.append(f"joined records below required minimum: {n_joined} < {min_joined}")
    return blockers


def build_gold_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return build_gold_index(records)[0]


def build_gold_index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    out = {}
    occurrences: dict[str, int] = {}
    skipped_missing_query_count = 0
    skipped_empty_rubrics_count = 0
    for record in records:
        query = pick_first(record, "query", "prompt", "instruction")
        gold = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold", "rubrics"), dedupe=True)
        if not query:
            skipped_missing_query_count += 1
            continue
        if not gold:
            skipped_empty_rubrics_count += 1
            continue
        key = occurrence_key(query, occurrences)
        out[key] = {"query": str(query), "gold_rubrics": gold}
    stats = {
        "input_record_count": len(records),
        "join_key_count": len(out),
        "skipped_missing_query_count": skipped_missing_query_count,
        "skipped_empty_rubrics_count": skipped_empty_rubrics_count,
        **occurrence_stats(occurrences),
    }
    return out, stats


def build_prediction_map(records: list[dict[str, Any]], model: str | None = None) -> dict[str, dict[str, Any]]:
    return build_prediction_index(records, model=model)[0]


def build_prediction_index(
    records: list[dict[str, Any]],
    model: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    out = {}
    occurrences: dict[str, int] = {}
    skipped_model_filter_count = 0
    skipped_missing_query_count = 0
    skipped_empty_rubrics_count = 0
    for record in records:
        record_model = str(record.get("model") or record.get("teacher") or "")
        if model and model not in {record_model, str(record.get("teacher") or "")}:
            skipped_model_filter_count += 1
            continue
        query = pick_first(record, "query", "prompt", "instruction")
        rubrics = parse_rubrics(
            pick_first(record, "rubrics", "response", "model_rubrics", "generated_rubrics", "prediction", "output"),
            dedupe=False,
        )
        if not query:
            skipped_missing_query_count += 1
            continue
        if not rubrics:
            skipped_empty_rubrics_count += 1
            continue
        key = occurrence_key(query, occurrences)
        out[key] = compact_record(
            {
                "rubrics": rubrics,
                "model": record_model,
                "valid_flags": pick_first(record, "valid_flags", "verifier_flags", "validity_flags"),
                "verifier_source": pick_first(record, "verifier_source", "verifier", "verifier_model"),
            }
        )
    stats = {
        "input_record_count": len(records),
        "join_key_count": len(out),
        "skipped_model_filter_count": skipped_model_filter_count,
        "skipped_missing_query_count": skipped_missing_query_count,
        "skipped_empty_rubrics_count": skipped_empty_rubrics_count,
        **occurrence_stats(occurrences),
    }
    return out, stats


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


def make_key(value: Any) -> str:
    return " ".join(str(value).strip().split())


def occurrence_key(value: Any, occurrences: dict[str, int]) -> str:
    base = make_key(value)
    occurrence = occurrences.get(base, 0)
    occurrences[base] = occurrence + 1
    if occurrence == 0:
        return base
    return f"{base} [occurrence={occurrence + 1}]"


def occurrence_stats(occurrences: dict[str, int]) -> dict[str, int]:
    return {
        "duplicate_join_key_count": sum(1 for count in occurrences.values() if count > 1),
        "duplicate_record_count": sum(count - 1 for count in occurrences.values() if count > 1),
    }


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if value not in (None, "", [])}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


if __name__ == "__main__":
    main()
