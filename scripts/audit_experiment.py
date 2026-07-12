#!/usr/bin/env python3
"""Audit BlindSpot-RL experiment artifacts against a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    manifest, manifest_error = load_manifest(args.manifest)
    report = invalid_manifest_report(manifest_error) if manifest_error else audit_manifest(manifest, root=args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, args.output)
    if not args.non_strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BlindSpot-RL experiment outputs.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", default=Path("."), type=Path)
    parser.add_argument("--output", default=Path("outputs/audit_report.json"), type=Path)
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Write the audit report but exit successfully when report ok=false.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"manifest is missing: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"manifest is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
    if not isinstance(data, dict):
        return {}, f"manifest must be a JSON object: {path}"
    return data, None


def invalid_manifest_report(error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "present_files": [],
        "missing_files": [],
        "summary_checks": [],
        "manifest_error": error,
    }


def audit_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    missing_files = []
    present_files = []
    for raw_path in manifest.get("required_files", []):
        path = root / raw_path
        if path.exists() and path.stat().st_size > 0:
            present_files.append(raw_path)
        else:
            missing_files.append(raw_path)

    summary_checks = []
    for item in manifest.get("summaries", []):
        summary_checks.append(audit_summary(item, root))

    ok = not missing_files and all(item["ok"] for item in summary_checks)
    return {
        "ok": ok,
        "present_files": present_files,
        "missing_files": missing_files,
        "summary_checks": summary_checks,
    }


def audit_summary(item: dict[str, Any], root: Path) -> dict[str, Any]:
    path = root / item["path"]
    result = {
        "name": item.get("name", item["path"]),
        "path": item["path"],
        "ok": False,
        "missing_keys": [],
        "non_numeric_keys": [],
        "error": "",
    }
    if not path.exists():
        result["missing_keys"] = list(item.get("required_keys", []))
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["error"] = f"summary is not valid JSON: line {exc.lineno} column {exc.colno}"
        return result
    if not isinstance(data, dict):
        result["error"] = "summary must be a JSON object"
        return result
    missing = []
    non_numeric = []
    for key in item.get("required_keys", []):
        if key not in data:
            missing.append(key)
            continue
        if isinstance(data[key], (int, float)):
            continue
        non_numeric.append(key)
    result["missing_keys"] = missing
    result["non_numeric_keys"] = non_numeric
    result["ok"] = not missing and not non_numeric
    return result


def print_report(report: dict[str, Any], output: Path) -> None:
    print(f"Audit ok={report['ok']} report={output}")
    print(f"present_files={len(report['present_files'])} missing_files={len(report['missing_files'])}")
    if report.get("manifest_error"):
        print(f"Manifest error: {report['manifest_error']}")
    if report["missing_files"]:
        print("Missing files:")
        for path in report["missing_files"]:
            print(f"- {path}")
    failed = [item for item in report["summary_checks"] if not item["ok"]]
    if failed:
        print("Failed summaries:")
        for item in failed:
            print(
                f"- {item['name']}: missing={item['missing_keys']} "
                f"non_numeric={item['non_numeric_keys']} error={item.get('error', '')}"
            )


if __name__ == "__main__":
    main()
