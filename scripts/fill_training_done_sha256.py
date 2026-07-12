#!/usr/bin/env python3
"""Fill SHA256 provenance fields in training_done JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TOP_LEVEL_SHA_FIELDS = {
    "sft_config": "sft_config_sha256",
    "grpo_config": "grpo_config_sha256",
    "sft_data": "sft_data_sha256",
    "rl_data": "rl_data_sha256",
    "rl_data_report": "rl_data_report_sha256",
    "reward_function": "reward_function_sha256",
}

VARIANT_SHA_FIELDS = {
    "grpo_config": "grpo_config_sha256",
    "rl_data": "rl_data_sha256",
    "rl_data_report": "rl_data_report_sha256",
    "reward_function": "reward_function_sha256",
}


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    filled, report = fill_training_done_sha256(data, root=args.root, allow_missing=args.allow_missing)
    output_path = args.output or args.input
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["input"] = str(args.input)
    report["output"] = str(output_path)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Filled training_done SHA256 fields ok={report['ok']} output={output_path}")
    if not report["ok"] and not args.allow_missing:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill SHA256 fields in a training_done JSON file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Output path. Defaults to overwriting --input.")
    parser.add_argument("--root", default=Path("."), type=Path, help="Repository root for relative recorded paths.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep going and leave missing SHA fields unchanged when referenced files are absent.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"training_done JSON is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"training_done JSON is invalid: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"training_done JSON must contain an object: {path}")
    return data


def fill_training_done_sha256(
    data: dict[str, Any],
    *,
    root: Path,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    filled = json.loads(json.dumps(data))
    updates: list[dict[str, Any]] = []
    blockers: list[str] = []

    fill_container(
        filled,
        prefix="",
        field_map=TOP_LEVEL_SHA_FIELDS,
        root=root,
        updates=updates,
        blockers=blockers,
        allow_missing=allow_missing,
    )
    variants = filled.get("variants")
    if isinstance(variants, dict):
        for variant_name, variant in variants.items():
            if not isinstance(variant, dict):
                blockers.append(f"variants.{variant_name} must be an object")
                continue
            fill_container(
                variant,
                prefix=f"variants.{variant_name}.",
                field_map=VARIANT_SHA_FIELDS,
                root=root,
                updates=updates,
                blockers=blockers,
                allow_missing=allow_missing,
            )

    return filled, {"ok": not blockers, "updates": updates, "blockers": blockers}


def fill_container(
    container: dict[str, Any],
    *,
    prefix: str,
    field_map: dict[str, str],
    root: Path,
    updates: list[dict[str, Any]],
    blockers: list[str],
    allow_missing: bool,
) -> None:
    for source_key, sha_key in field_map.items():
        if source_key not in container and sha_key not in container:
            continue
        source_value = str(container.get(source_key, "") or "").strip()
        json_key = f"{prefix}{sha_key}"
        if not source_value:
            blockers.append(f"missing source path for {prefix}{source_key}")
            continue
        source_path = resolve_recorded_path(source_key, source_value, root=root)
        if not source_path.exists() or not source_path.is_file():
            message = f"missing SHA256 source file for {json_key}: {source_path}"
            if allow_missing:
                updates.append(
                    {
                        "json_key": json_key,
                        "source_key": f"{prefix}{source_key}",
                        "source_path": str(source_path),
                        "status": "missing",
                    }
                )
                continue
            blockers.append(message)
            continue
        digest = file_sha256(source_path)
        container[sha_key] = digest
        updates.append(
            {
                "json_key": json_key,
                "source_key": f"{prefix}{source_key}",
                "source_path": str(source_path),
                "sha256": digest,
                "status": "filled",
            }
        )


def resolve_recorded_path(source_key: str, value: str, *, root: Path) -> Path:
    path_text = value.split(":", 1)[0] if source_key == "reward_function" else value
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
