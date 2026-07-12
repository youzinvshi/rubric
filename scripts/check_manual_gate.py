#!/usr/bin/env python3
"""Check a manual evidence gate before continuing a real run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    report = build_report(
        name=args.name,
        required_path=args.required_path,
        required_json=args.required_json,
        required_json_contains=args.required_json_contains,
        required_json_equals=args.required_json_equals,
        required_json_sha256=args.required_json_sha256,
        instructions=args.instructions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Manual gate ok={report['ok']} report={args.output}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a manual real-run gate.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--required-path", action="append", default=[], type=Path)
    parser.add_argument(
        "--required-json",
        action="append",
        default=[],
        help="JSON content check in the form path:key1,key2,...",
    )
    parser.add_argument(
        "--required-json-contains",
        action="append",
        default=[],
        help="JSON list containment check in the form path:list_key=value1,value2,...",
    )
    parser.add_argument(
        "--required-json-equals",
        action="append",
        default=[],
        help="JSON exact-value check in the form path:key=value",
    )
    parser.add_argument(
        "--required-json-sha256",
        action="append",
        default=[],
        help="JSON SHA256 check in the form json_path:key=file_path.",
    )
    parser.add_argument("--instructions", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_report(
    name: str,
    required_path: list[Path],
    instructions: list[str],
    required_json: list[str] | None = None,
    required_json_contains: list[str] | None = None,
    required_json_equals: list[str] | None = None,
    required_json_sha256: list[str] | None = None,
) -> dict[str, Any]:
    checks = [check_path(path) for path in required_path]
    json_checks = [check_json_contract(spec) for spec in (required_json or [])]
    json_contains_checks = [check_json_contains_contract(spec) for spec in (required_json_contains or [])]
    json_equals_checks = [check_json_equals_contract(spec) for spec in (required_json_equals or [])]
    json_sha256_checks = [check_json_sha256_contract(spec) for spec in (required_json_sha256 or [])]
    blockers = [f"missing required path: {item['path']}" for item in checks if not item["present"]]
    for item in json_checks:
        blockers.extend(item["blockers"])
    for item in json_contains_checks:
        blockers.extend(item["blockers"])
    for item in json_equals_checks:
        blockers.extend(item["blockers"])
    for item in json_sha256_checks:
        blockers.extend(item["blockers"])
    return {
        "name": name,
        "ok": not blockers,
        "checks": checks,
        "json_checks": json_checks,
        "json_contains_checks": json_contains_checks,
        "json_equals_checks": json_equals_checks,
        "json_sha256_checks": json_sha256_checks,
        "blockers": blockers,
        "instructions": instructions,
    }


def check_path(path: Path) -> dict[str, Any]:
    present = path.exists()
    return {
        "path": str(path),
        "present": present,
        "type": "dir" if present and path.is_dir() else "file" if present and path.is_file() else "missing",
        "bytes": path.stat().st_size if present and path.is_file() else 0,
    }


def check_json_contract(spec: str) -> dict[str, Any]:
    path_text, keys = parse_json_spec(spec)
    path = Path(path_text)
    result: dict[str, Any] = {
        "path": path_text,
        "required_keys": keys,
        "present": path.exists(),
        "valid_json": False,
        "missing_keys": [],
        "blockers": [],
    }
    if not path.exists():
        result["blockers"].append(f"missing required JSON file: {path_text}")
        result["missing_keys"] = keys
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["blockers"].append(f"invalid JSON file: {path_text} ({exc.msg})")
        result["missing_keys"] = keys
        return result
    result["valid_json"] = True
    if not isinstance(data, dict):
        result["blockers"].append(f"required JSON file must contain an object: {path_text}")
        result["missing_keys"] = keys
        return result
    missing = [key for key in keys if is_missing_value(nested_get(data, key))]
    result["missing_keys"] = missing
    result["blockers"].extend(f"missing required JSON key in {path_text}: {key}" for key in missing)
    return result


def check_json_contains_contract(spec: str) -> dict[str, Any]:
    path_text, key, required_values = parse_json_contains_spec(spec)
    path = Path(path_text)
    result: dict[str, Any] = {
        "path": path_text,
        "key": key,
        "required_values": required_values,
        "present": path.exists(),
        "valid_json": False,
        "actual_values": [],
        "missing_values": [],
        "blockers": [],
    }
    if not path.exists():
        result["blockers"].append(f"missing required JSON file: {path_text}")
        result["missing_values"] = required_values
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["blockers"].append(f"invalid JSON file: {path_text} ({exc.msg})")
        result["missing_values"] = required_values
        return result
    result["valid_json"] = True
    if not isinstance(data, dict):
        result["blockers"].append(f"required JSON file must contain an object: {path_text}")
        result["missing_values"] = required_values
        return result
    value = nested_get(data, key)
    if isinstance(value, list):
        actual = [str(item) for item in value]
    else:
        actual = [] if is_missing_value(value) else [str(value)]
    result["actual_values"] = actual
    missing = [item for item in required_values if item not in actual]
    result["missing_values"] = missing
    result["blockers"].extend(f"missing required JSON list value in {path_text}: {key} must contain {item}" for item in missing)
    return result


def check_json_equals_contract(spec: str) -> dict[str, Any]:
    path_text, key, expected_value = parse_json_equals_spec(spec)
    path = Path(path_text)
    result: dict[str, Any] = {
        "path": path_text,
        "key": key,
        "expected_value": expected_value,
        "present": path.exists(),
        "valid_json": False,
        "actual_value": None,
        "matches": False,
        "blockers": [],
    }
    if not path.exists():
        result["blockers"].append(f"missing required JSON file: {path_text}")
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["blockers"].append(f"invalid JSON file: {path_text} ({exc.msg})")
        return result
    result["valid_json"] = True
    if not isinstance(data, dict):
        result["blockers"].append(f"required JSON file must contain an object: {path_text}")
        return result
    actual = nested_get(data, key)
    actual_text = stringify_json_value(actual)
    result["actual_value"] = actual_text
    result["matches"] = actual_text == expected_value
    if not result["matches"]:
        result["blockers"].append(
            f"JSON value mismatch in {path_text}: {key} expected {expected_value}, got {actual_text}"
        )
    return result


def check_json_sha256_contract(spec: str) -> dict[str, Any]:
    path_text, key, target_path_text = parse_json_equals_spec(spec)
    path = Path(path_text)
    target_path = Path(target_path_text)
    result: dict[str, Any] = {
        "path": path_text,
        "key": key,
        "target_path": target_path_text,
        "present": path.exists(),
        "target_present": target_path.exists(),
        "valid_json": False,
        "expected_sha256": "",
        "actual_value": None,
        "matches": False,
        "blockers": [],
    }
    if not path.exists():
        result["blockers"].append(f"missing required JSON file: {path_text}")
        return result
    if not target_path.exists():
        result["blockers"].append(f"missing SHA256 target file: {target_path_text}")
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["blockers"].append(f"invalid JSON file: {path_text} ({exc.msg})")
        return result
    result["valid_json"] = True
    if not isinstance(data, dict):
        result["blockers"].append(f"required JSON file must contain an object: {path_text}")
        return result
    actual = nested_get(data, key)
    actual_text = stringify_json_value(actual)
    expected = file_sha256(target_path)
    result["actual_value"] = actual_text
    result["expected_sha256"] = expected
    result["matches"] = actual_text == expected
    if not result["matches"]:
        result["blockers"].append(
            f"JSON sha256 mismatch in {path_text}: {key} expected current sha256 of {target_path_text} "
            f"({expected}), got {actual_text}"
        )
    return result


def parse_json_spec(spec: str) -> tuple[str, list[str]]:
    if ":" not in spec:
        return spec, []
    path, raw_keys = spec.split(":", 1)
    keys = [item.strip() for item in raw_keys.split(",") if item.strip()]
    return path, keys


def parse_json_contains_spec(spec: str) -> tuple[str, str, list[str]]:
    if ":" not in spec or "=" not in spec:
        raise ValueError(f"Invalid required-json-contains spec: {spec}")
    path, rest = spec.split(":", 1)
    key, raw_values = rest.split("=", 1)
    values = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not key.strip() or not values:
        raise ValueError(f"Invalid required-json-contains spec: {spec}")
    return path, key.strip(), values


def parse_json_equals_spec(spec: str) -> tuple[str, str, str]:
    if ":" not in spec or "=" not in spec:
        raise ValueError(f"Invalid required-json-equals spec: {spec}")
    path, rest = spec.split(":", 1)
    key, value = rest.split("=", 1)
    if not key.strip() or value == "":
        raise ValueError(f"Invalid required-json-equals spec: {spec}")
    return path, key.strip(), value


def nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx < len(current):
                current = current[idx]
                continue
        return None
    return current


def is_missing_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def stringify_json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Manual Gate: {report['name']}",
        "",
        f"- Overall ok: `{report['ok']}`",
        "",
        "## Required Paths",
        "",
        "| Path | Present | Type | Bytes |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["checks"]:
        lines.append(f"| `{item['path']}` | `{item['present']}` | `{item['type']}` | {item['bytes']} |")
    if report.get("json_checks"):
        lines.extend(["", "## Required JSON Contracts", "", "| Path | Present | Valid JSON | Missing Keys |", "| --- | --- | --- | --- |"])
        for item in report["json_checks"]:
            missing = ", ".join(item["missing_keys"]) if item["missing_keys"] else "none"
            lines.append(f"| `{item['path']}` | `{item['present']}` | `{item['valid_json']}` | {missing} |")
    if report.get("json_contains_checks"):
        lines.extend(
            [
                "",
                "## Required JSON Contains Contracts",
                "",
                "| Path | Key | Present | Valid JSON | Missing Values |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in report["json_contains_checks"]:
            missing = ", ".join(item["missing_values"]) if item["missing_values"] else "none"
            lines.append(f"| `{item['path']}` | `{item['key']}` | `{item['present']}` | `{item['valid_json']}` | {missing} |")
    if report.get("json_equals_checks"):
        lines.extend(
            [
                "",
                "## Required JSON Equals Contracts",
                "",
                "| Path | Key | Expected | Actual | Matches |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in report["json_equals_checks"]:
            lines.append(
                f"| `{item['path']}` | `{item['key']}` | `{item['expected_value']}` | "
                f"`{item['actual_value']}` | `{item['matches']}` |"
            )
    if report.get("json_sha256_checks"):
        lines.extend(
            [
                "",
                "## Required JSON SHA256 Contracts",
                "",
                "| Path | Key | Target | Expected SHA256 | Actual | Matches |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in report["json_sha256_checks"]:
            lines.append(
                f"| `{item['path']}` | `{item['key']}` | `{item['target_path']}` | "
                f"`{item['expected_sha256']}` | `{item['actual_value']}` | `{item['matches']}` |"
            )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in report["blockers"]] or ["- none"])
    lines.extend(["", "## Instructions", ""])
    lines.extend([f"- {item}" for item in report["instructions"]] or ["- none"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
