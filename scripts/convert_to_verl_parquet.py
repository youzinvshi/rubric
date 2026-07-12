#!/usr/bin/env python3
"""Convert rubric-gold records to verl-friendly parquet.

Output columns:
- prompt: visible model input
- gold_rubrics: hidden reward-side gold dimensions
- data_source: dataset name
- ground_truth: nested reward-side gold for reward hooks that receive ground_truth
- extra_info: nested reward-side metadata for reward hooks that receive extra_info
"""

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

from blindspot_rl.reward_bsc import parse_rubrics  # noqa: E402

from scripts.budget_gate import file_sha256  # noqa: E402


PROMPT_TEMPLATE = "为以下query生成评估rubric(原子化/可判yes-no/去冗余):\n{query}"
DEFAULT_FORBIDDEN_SOURCE_MARKERS = ("test_main", "holdout", "downstream")
DEFAULT_FORBIDDEN_SPLITS = ("test_main", "holdout", "downstream", "test")


def main() -> None:
    args = parse_args()
    rows = list(load_json_records(args.input))
    blockers = holdout_source_blockers(
        rows,
        input_path=args.input,
        data_source=args.data_source,
        forbidden_source_markers=args.forbid_source_marker,
        forbidden_splits=args.forbid_split,
    )
    if blockers:
        raise SystemExit("; ".join(blockers[:3]))
    records = convert_records(rows, data_source=args.data_source, prompt_template=args.prompt_template)

    if not records:
        raise SystemExit("No valid records with query and gold evaluation dimensions were found.")
    if len(records) < args.min_records:
        raise SystemExit(f"Too few valid verl records: {len(records)} < {args.min_records}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".jsonl", ".json"}:
        write_jsonl(args.output, records)
    else:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to write parquet files.") from exc
        pd.DataFrame(records).to_parquet(args.output, index=False)
    if args.report_output:
        write_report(
            args.report_output,
            input_path=args.input,
            output_path=args.output,
            records=records,
            data_source=args.data_source,
            forbidden_source_markers=args.forbid_source_marker,
            forbidden_splits=args.forbid_split,
        )
    print(f"Wrote {len(records)} records to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert rubric records to verl parquet.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL/JSON file with query+gold.")
    parser.add_argument("--output", required=True, type=Path, help="Output .parquet or .jsonl path.")
    parser.add_argument("--data-source", default="rubricbench")
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--min-records", type=int, default=1, help="Fail unless at least this many records are converted.")
    parser.add_argument("--report-output", type=Path, help="Optional JSON report with input/output provenance.")
    parser.add_argument(
        "--forbid-source-marker",
        action="append",
        default=list(DEFAULT_FORBIDDEN_SOURCE_MARKERS),
        help="Fail if input path, --data-source, or record data_source contains this marker. Repeatable.",
    )
    parser.add_argument(
        "--forbid-split",
        action="append",
        default=list(DEFAULT_FORBIDDEN_SPLITS),
        help="Fail if a record split/data_split/subset equals this value. Repeatable.",
    )
    return parser.parse_args()


def convert_records(
    rows: Iterable[dict[str, Any]],
    data_source: str = "rubricbench",
    prompt_template: str = PROMPT_TEMPLATE,
) -> list[dict[str, Any]]:
    records = []
    for record in rows:
        prompt = pick_first(record, "prompt", "query", "instruction")
        query = pick_first(record, "query", "instruction", "task", "prompt")
        raw_gold = parse_rubrics(
            pick_first(record, "response", "gold_rubrics", "gold", "rubrics", "criteria"),
            dedupe=True,
        )
        gold = [normalize_text(item) for item in raw_gold]
        if not prompt or not gold:
            continue
        prompt_text = normalize_text(str(prompt))
        query_text = normalize_text(str(query or prompt))
        source = str(record.get("source") or record.get("data_source") or data_source)
        visible_prompt = prompt_text if record.get("prompt") else prompt_template.format(query=query_text)
        records.append(
            {
                "prompt": visible_prompt,
                "gold_rubrics": gold,
                "data_source": source,
                "ground_truth": {"gold_rubrics": gold},
                "extra_info": {
                    "gold_rubrics": gold,
                    "query": query_text,
                    "data_source": source,
                },
            }
        )
    return records


def normalize_text(value: str) -> str:
    return " ".join(str(value).splitlines()).strip()


def holdout_source_blockers(
    rows: list[dict[str, Any]],
    input_path: Path | None = None,
    data_source: str = "",
    forbidden_source_markers: list[str] | tuple[str, ...] = DEFAULT_FORBIDDEN_SOURCE_MARKERS,
    forbidden_splits: list[str] | tuple[str, ...] = DEFAULT_FORBIDDEN_SPLITS,
) -> list[str]:
    forbidden_markers = {str(marker).lower() for marker in forbidden_source_markers if str(marker).strip()}
    forbidden_split_values = {str(split).lower() for split in forbidden_splits if str(split).strip()}
    blockers: list[str] = []
    input_text = str(input_path or "").lower()
    if input_text and any(marker in input_text for marker in forbidden_markers):
        blockers.append(f"input path {input_path} is forbidden for GRPO/RLVR training conversion")
    data_source_text = str(data_source or "").lower()
    if data_source_text and any(marker in data_source_text for marker in forbidden_markers):
        blockers.append(f"data_source={data_source} is forbidden for GRPO/RLVR training conversion")
    for idx, record in enumerate(rows, start=1):
        record_source = str(record.get("data_source", "")).lower()
        if record_source and any(marker in record_source for marker in forbidden_markers):
            blockers.append(
                f"record {idx} data_source={record.get('data_source')} is forbidden for GRPO/RLVR training conversion"
            )
        split_value = first_split_value(record)
        if split_value and split_value.lower() in forbidden_split_values:
            blockers.append(f"record {idx} split={split_value} is forbidden for GRPO/RLVR training conversion")
    return blockers


def first_split_value(record: dict[str, Any]) -> str:
    for key in ("split", "data_split", "subset"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("split", "data_split", "subset"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def load_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        records = data.get("records") or data.get("data") or [data]
        yield from records
    else:
        raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    input_path: Path,
    output_path: Path,
    records: list[dict[str, Any]],
    data_source: str,
    forbidden_source_markers: list[str],
    forbidden_splits: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "n_records": len(records),
        "data_source": data_source,
        "forbidden_source_markers": list(forbidden_source_markers),
        "forbidden_splits": list(forbidden_splits),
        "columns": ["prompt", "gold_rubrics", "data_source", "ground_truth", "extra_info"],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
