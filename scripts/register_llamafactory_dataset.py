#!/usr/bin/env python3
"""Register BlindSpot-RL SFT JSONL in LLaMA-Factory dataset_info.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = {"prompt": "prompt", "response": "response"}


def main() -> None:
    args = parse_args()
    dataset_info = load_dataset_info(args.dataset_info)
    entry = build_entry(args.file_name, args.formatting, DEFAULT_COLUMNS)
    if args.name in dataset_info and not args.overwrite:
        raise SystemExit(f"Dataset {args.name!r} already exists. Pass --overwrite to replace it.")
    dataset_info[args.name] = entry
    args.dataset_info.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_info.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Registered {args.name} -> {args.file_name} in {args.dataset_info}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a BlindSpot-RL SFT JSONL for LLaMA-Factory.")
    parser.add_argument("--dataset-info", required=True, type=Path)
    parser.add_argument("--name", default="blindspot_sft")
    parser.add_argument("--file-name", default="blindspot_sft.jsonl")
    parser.add_argument("--formatting", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dataset_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in dataset_info.json, got {type(data).__name__}")
    return data


def build_entry(file_name: str, formatting: str, columns: dict[str, str]) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": formatting,
        "columns": columns,
    }


if __name__ == "__main__":
    main()
