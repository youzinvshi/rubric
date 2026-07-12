#!/usr/bin/env python3
"""Deterministically partition a normalized JSONL dataset into query-disjoint splits.

Partitioning happens at the *query group* level: every record sharing a query stays
in the same split. This is what protects the main BSC evaluation from training
contamination -- a query assigned to the ``test_main`` holdout can never appear in
any training split.

Each emitted row is stamped with governance fields (``split``, ``gold_type``,
``allowed_in_main_bsc_eval``) so contamination policy travels with the data. Only
human-gold rows in a designated main-eval split are marked
``allowed_in_main_bsc_eval=True``.

The CLI writes one JSONL per split plus a ``--manifest`` JSON that records the seed,
per-split group/record counts, per-split sha256, and an asserted cross-split
group-disjointness check (fail-closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Functional core (imported by tests)
# ---------------------------------------------------------------------------

def group_by_query(records: list[dict[str, Any]], group_key: str = "query") -> "dict[str, list[dict[str, Any]]]":
    """Group records by ``group_key``. Raises if any record lacks a usable key."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(group_key)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"Record missing group key '{group_key}': {record!r}")
        groups[group_marker(value)].append(record)
    return groups


def group_marker(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def detect_overlap(assigned_queries: "dict[str, list[str]]") -> set:
    """Return the set of queries that appear in more than one split."""
    counts: Counter = Counter()
    for queries in assigned_queries.values():
        counts.update(set(queries))
    return {query for query, n in counts.items() if n > 1}


def split_records(
    records: list[dict[str, Any]],
    counts: dict[str, int],
    seed: int = 13,
    group_key: str = "query",
    gold_type: str = "human_gold",
    main_eval_splits: "tuple[str, ...]" = ("test_main",),
) -> "tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]":
    """Assign whole query-groups to named splits targeting per-split record counts.

    Returns ``(splits, report)``. ``splits`` maps split name -> stamped rows. The last
    split in ``counts`` absorbs any remaining groups so record totals reconcile.
    """
    split_names = list(counts)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}

    try:
        groups = group_by_query(records, group_key)
    except ValueError as exc:
        return splits, {"ok": False, "blockers": [str(exc)], "counts": {}, "total_records": len(records)}

    total_records = len(records)
    requested = sum(counts.values())
    blockers: list[str] = []
    if requested > total_records:
        blockers.append(
            f"Requested {requested} records across splits but only {total_records} available."
        )
    if blockers:
        return splits, {"ok": False, "blockers": blockers, "counts": {}, "total_records": total_records}

    query_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(query_ids)

    idx = 0
    for i, name in enumerate(split_names):
        is_last = i == len(split_names) - 1
        target = counts[name]
        current = 0
        if is_last:
            while idx < len(query_ids):
                current += _absorb(splits[name], groups[query_ids[idx]])
                idx += 1
        else:
            while idx < len(query_ids) and current < target:
                current += _absorb(splits[name], groups[query_ids[idx]])
                idx += 1

    allowed_names = set(main_eval_splits) if gold_type == "human_gold" else set()
    for name in split_names:
        allowed = name in allowed_names
        for row in splits[name]:
            row["split"] = name
            row["gold_type"] = gold_type
            row["allowed_in_main_bsc_eval"] = allowed

    assigned_queries = {name: [group_key_value(row, group_key) for row in rows] for name, rows in splits.items()}
    overlap = detect_overlap(assigned_queries)
    if overlap:
        blockers.append(f"Query overlap across splits: {sorted(overlap)[:5]}")

    report = {
        "ok": not blockers,
        "blockers": blockers,
        "seed": seed,
        "group_key": group_key,
        "gold_type": gold_type,
        "total_records": total_records,
        "total_groups": len(groups),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "group_counts": {name: len({group_key_value(r, group_key) for r in rows}) for name, rows in splits.items()},
    }
    return splits, report


def group_key_value(row: dict[str, Any], group_key: str) -> str:
    return group_marker(row.get(group_key))


def _absorb(target_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> int:
    target_rows.extend(group_rows)
    return len(group_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    counts = resolve_counts(args.split, len(records))
    splits, report = split_records(
        records=records,
        counts=counts,
        seed=args.seed,
        group_key=args.group_key,
        gold_type=args.gold_type,
        main_eval_splits=tuple(args.main_eval_split),
    )
    if not report["ok"]:
        raise SystemExit("Split failed: " + "; ".join(report["blockers"]))

    written = write_splits(splits, args)
    manifest = build_manifest(records, splits, written, report, args)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = ", ".join(f"{name}={len(rows)}" for name, rows in splits.items())
    print(f"Split {len(records)} records / {report['total_groups']} groups -> {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query-disjoint deterministic dataset splitter.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        action="append",
        help="Split spec 'name:count', 'name:0.6' (fraction of records), or 'name:rest'. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for split files.")
    parser.add_argument("--output-prefix", default="", help="Filename prefix, e.g. 'rubricbench_'.")
    parser.add_argument("--group-key", default="query", help="Field defining disjoint groups.")
    parser.add_argument("--stratify-key", default="data_source", help="Field summarized per split in the manifest.")
    parser.add_argument("--gold-type", default="human_gold", help="human_gold | proxy_teacher | synthetic.")
    parser.add_argument(
        "--main-eval-split",
        action="append",
        default=[],
        help="Split name(s) that back the main BSC eval (only honored for human_gold).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def resolve_counts(raw_specs: list[str], total: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    rest_name = None
    allocated = 0
    for raw in raw_specs:
        if ":" not in raw:
            raise SystemExit(f"Invalid --split '{raw}'. Expected 'name:count', 'name:frac', or 'name:rest'.")
        name, value = raw.split(":", 1)
        name = name.strip()
        value = value.strip().lower()
        if not name:
            raise SystemExit(f"Invalid --split '{raw}': empty name.")
        if name in counts or name == rest_name:
            raise SystemExit(f"Duplicate split name '{name}'.")
        if value == "rest":
            if rest_name is not None:
                raise SystemExit("Only one split may use ':rest'.")
            rest_name = name
            continue
        number = float(value)
        if number <= 0:
            raise SystemExit(f"Invalid --split '{raw}': value must be positive.")
        if 0 < number < 1 or (number == 1 and "." in value):
            resolved = int(round(number * total))
        else:
            if number != int(number):
                raise SystemExit(f"Invalid --split '{raw}': non-integer count.")
            resolved = int(number)
        counts[name] = resolved
        allocated += resolved

    if rest_name is not None:
        counts[rest_name] = max(0, total - allocated)
    return counts


def write_splits(splits: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, rows in splits.items():
        path = output_dir / f"{args.output_prefix}{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[name] = path
    return written


def build_manifest(
    records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    written: dict[str, Path],
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_splits: dict[str, Any] = {}
    for name, rows in splits.items():
        groups = {group_key_value(r, args.group_key) for r in rows}
        manifest_splits[name] = {
            "path": str(written[name]),
            "groups": len(groups),
            "records": len(rows),
            "sha256": sha256_file(written[name]),
            "allowed_in_main_bsc_eval": rows[0]["allowed_in_main_bsc_eval"] if rows else False,
            "strata": strata_counts(rows, args.stratify_key),
        }
    return {
        "input": str(args.input),
        "seed": args.seed,
        "group_key": args.group_key,
        "stratify_key": args.stratify_key or None,
        "gold_type": args.gold_type,
        "main_eval_splits": list(args.main_eval_split),
        "total_records": len(records),
        "total_groups": report["total_groups"],
        "group_disjoint": report["ok"],
        "splits": manifest_splits,
    }


def strata_counts(rows: list[dict[str, Any]], stratify_key: str | None) -> dict[str, int]:
    if not stratify_key:
        return {"all": len(rows)}
    counter: Counter = Counter(str(row.get(stratify_key) or "unknown") for row in rows)
    return dict(counter)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
    return rows


if __name__ == "__main__":
    main()
