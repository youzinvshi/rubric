#!/usr/bin/env python3
"""Normalize raw benchmark files into BlindSpot-RL schemas.

Targets:
- gold:       {"query", "gold_rubrics", "data_source", ...provenance}
- query_pool: {"query", "data_source", ...provenance}
- preference: {"query", "chosen", "rejected", "data_source", ...provenance}
- multicandidate: {"query", "candidates", "label", "data_source", ...provenance}
"""

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


MISSING = object()


DEFAULT_QUERY_KEYS = (
    "query",
    "prompt",
    "instruction",
    "question",
    "task",
    "input",
    "problem",
)
DEFAULT_GOLD_KEYS = (
    "gold_rubrics",
    "rubrics",
    "rubric",
    "criteria",
    "criterion",
    "human_rubrics",
    "reference_rubrics",
)
DEFAULT_CHOSEN_KEYS = ("chosen", "winner", "response_chosen", "answer_chosen", "chosen_response")
DEFAULT_REJECTED_KEYS = ("rejected", "loser", "response_rejected", "answer_rejected", "rejected_response")
DEFAULT_CANDIDATES_KEYS = ("candidates", "responses", "answers", "choices", "options")
DEFAULT_LABEL_KEYS = ("label", "correct", "correct_index", "answer_index", "gold", "winner", "chosen")
DEFAULT_PROVENANCE_KEYS = ("provenance", "source_url", "paper_url", "dataset_version", "license", "split")
STATIC_PROVENANCE_FIELDS = ("provenance", "source_url", "paper_url", "dataset_version", "license", "split")

# Named two-way comparison pairs (e.g. JudgeBench response_A/response_B + label="A>B").
COMPARATIVE_RESPONSE_KEYS = (("response_A", "response_B"), ("answer_A", "answer_B"), ("response_a", "response_b"))

# Split correct/incorrect lists (e.g. RewardBench-2 chosen + rejected as separate lists).
SPLIT_CHOSEN_KEYS = ("chosen", "correct", "correct_responses", "positive")
SPLIT_REJECTED_KEYS = ("rejected", "incorrect", "incorrect_responses", "negative")


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    rows = normalize_records(records, args)
    if not rows:
        raise SystemExit(f"No records matched target={args.target}. Check field mappings.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} normalized {args.target} records to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw datasets for BlindSpot-RL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=["gold", "query_pool", "preference", "multicandidate"])
    parser.add_argument("--data-source", default="unknown")
    parser.add_argument("--query-key")
    parser.add_argument("--gold-key")
    parser.add_argument("--chosen-key")
    parser.add_argument("--rejected-key")
    parser.add_argument("--candidates-key")
    parser.add_argument("--label-key")
    parser.add_argument("--provenance-key")
    parser.add_argument("--provenance")
    parser.add_argument("--source-url")
    parser.add_argument("--paper-url")
    parser.add_argument("--dataset-version")
    parser.add_argument("--license")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dedupe-query", action="store_true")
    return parser.parse_args()


def normalize_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    seen_queries = set()
    for record in records:
        query = pick_field(record, [args.query_key] if args.query_key else DEFAULT_QUERY_KEYS)
        if not query:
            continue
        query = stringify(query)
        if args.dedupe_query and query in seen_queries:
            continue
        seen_queries.add(query)

        if args.target == "query_pool":
            rows.append(with_provenance({"query": query, "data_source": args.data_source}, record, args))
        elif args.target == "gold":
            raw_gold = pick_field(record, [args.gold_key] if args.gold_key else DEFAULT_GOLD_KEYS)
            gold = parse_rubrics(raw_gold, dedupe=True)
            if not gold:
                continue
            rows.append(
                with_provenance(
                    {"query": query, "gold_rubrics": gold, "data_source": args.data_source},
                    record,
                    args,
                )
            )
        elif args.target == "preference":
            chosen = pick_field(record, [args.chosen_key] if args.chosen_key else DEFAULT_CHOSEN_KEYS)
            rejected = pick_field(record, [args.rejected_key] if args.rejected_key else DEFAULT_REJECTED_KEYS)
            if chosen is None or rejected is None:
                chosen, rejected = resolve_comparative_pair(record, args)
            if chosen is None or rejected is None:
                continue
            rows.append(
                with_provenance(
                    {
                        "query": query,
                        "chosen": stringify(chosen),
                        "rejected": stringify(rejected),
                        "data_source": args.data_source,
                    },
                    record,
                    args,
                )
            )
        elif args.target == "multicandidate":
            raw_candidates = pick_field(
                record,
                [args.candidates_key] if args.candidates_key else DEFAULT_CANDIDATES_KEYS,
            )
            candidates = normalize_candidates(raw_candidates)
            label_value = pick_field(record, [args.label_key] if args.label_key else DEFAULT_LABEL_KEYS)
            label = normalize_label(label_value, candidates)
            if len(candidates) < 2 or label is None:
                candidates, label = resolve_split_candidates(record, args)
            if len(candidates) < 2 or label is None:
                continue
            rows.append(
                with_provenance(
                    {
                        "query": query,
                        "candidates": candidates,
                        "label": label,
                        "data_source": args.data_source,
                    },
                    record,
                    args,
                )
            )
        if args.limit and len(rows) >= args.limit:
            break
    return rows


def with_provenance(row: dict[str, Any], record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    provenance_key = getattr(args, "provenance_key", None)
    raw_keys = [provenance_key] if provenance_key else DEFAULT_PROVENANCE_KEYS
    for key in raw_keys:
        value = pick_field(record, [key])
        if value not in (None, ""):
            row[key] = stringify(value)
    for field in STATIC_PROVENANCE_FIELDS:
        value = getattr(args, field, None)
        if value not in (None, ""):
            row[field] = str(value)
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


def pick_field(record: dict[str, Any], keys: Iterable[str | None]) -> Any:
    for key in keys:
        if not key:
            continue
        value = get_path(record, key)
        if value not in (MISSING, None, ""):
            return value
    return None


def get_path(record: Any, path: str) -> Any:
    if isinstance(record, dict) and path in record:
        return record[path]
    current = record
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return MISSING
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx >= len(current):
                return MISSING
            current = current[idx]
            continue
        return MISSING
    return current


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def normalize_candidates(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = pick_field(item, ("text", "response", "answer", "content", "value"))
                if text is None:
                    text = item
                out.append(stringify(text))
            else:
                out.append(stringify(item))
        return [item for item in out if item]
    if isinstance(value, dict):
        ordered_keys = sorted(value, key=lambda item: str(item))
        return [stringify(value[key]) for key in ordered_keys if value[key] not in (None, "")]
    return []


def resolve_comparative_pair(record: dict[str, Any], args: argparse.Namespace) -> tuple[Any, Any]:
    """Map a named A/B response pair plus a comparative label to (chosen, rejected).

    Handles JudgeBench-style records: {response_A, response_B, label="A>B"|"B>A"}.
    Returns (None, None) when the pattern does not apply.
    """
    for key_a, key_b in COMPARATIVE_RESPONSE_KEYS:
        resp_a = get_path(record, key_a)
        resp_b = get_path(record, key_b)
        if resp_a in (MISSING, None, "") or resp_b in (MISSING, None, ""):
            continue
        label = pick_field(record, [args.label_key] if args.label_key else DEFAULT_LABEL_KEYS)
        winner = comparative_winner(label)
        if winner == "A":
            return resp_a, resp_b
        if winner == "B":
            return resp_b, resp_a
        return None, None
    return None, None


def comparative_winner(label: Any) -> str | None:
    """Return "A" or "B" for comparative labels like "A>B"/"B>A"; None if ambiguous."""
    if not isinstance(label, str):
        return None
    text = label.strip().upper().replace(" ", "")
    if text in {"A>B", "A"}:
        return "A"
    if text in {"B>A", "B"}:
        return "B"
    return None


def resolve_split_candidates(record: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], int | None]:
    """Build a single candidate list from separate correct/incorrect lists.

    Handles RewardBench-2-style records where ``chosen`` holds the gold
    completion(s) and ``rejected`` holds distractors. The first gold completion
    is placed at index 0 (label=0) followed by all distractors, matching the
    best-of-N downstream selection setup. Returns ([], None) when the pattern
    does not apply.
    """
    chosen_raw = pick_field(record, [args.chosen_key] if args.chosen_key else SPLIT_CHOSEN_KEYS)
    rejected_raw = pick_field(record, [args.rejected_key] if args.rejected_key else SPLIT_REJECTED_KEYS)
    chosen = normalize_candidates(chosen_raw) if isinstance(chosen_raw, list) else ([stringify(chosen_raw)] if chosen_raw not in (None, "") else [])
    rejected = normalize_candidates(rejected_raw) if isinstance(rejected_raw, list) else ([stringify(rejected_raw)] if rejected_raw not in (None, "") else [])
    if not chosen or not rejected:
        return [], None
    candidates = [chosen[0]] + rejected
    if len(candidates) < 2:
        return [], None
    return candidates, 0


def normalize_label(value: Any, candidates: list[str]) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value) if int(value) < len(candidates) else None
    if isinstance(value, int):
        return value if 0 <= value < len(candidates) else None
    text = str(value).strip()
    if text.isdigit():
        idx = int(text)
        return idx if 0 <= idx < len(candidates) else None
    for idx, candidate in enumerate(candidates):
        if text == candidate:
            return idx
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
