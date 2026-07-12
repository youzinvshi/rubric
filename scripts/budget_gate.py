"""Helpers for enforcing API budget reports before paid generation stages."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any


def require_budget_report(path: Path, expected_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a budget report and fail unless it explicitly permits execution."""
    if not path.exists():
        raise SystemExit(f"Required API budget report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Required API budget report is not valid JSON: {path}") from exc
    blockers = report.get("blockers", [])
    if report.get("ok") is not True or blockers:
        blocker_text = "; ".join(str(item) for item in blockers) or "budget report ok is not true"
        raise SystemExit(f"Required API budget report blocks execution: {path}: {blocker_text}")
    if expected_contract:
        contract_blockers = budget_contract_blockers(report, expected_contract)
        if contract_blockers:
            blocker_text = "; ".join(contract_blockers)
            raise SystemExit(f"Required API budget report contract mismatch: {path}: {blocker_text}")
    return report


def enforce_budget_report(
    path: Path | None,
    context: str = "paid API calls",
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require an explicit budget report path, then validate the report."""
    if path is None:
        raise SystemExit(f"--require-budget-report is required before {context}")
    return require_budget_report(path, expected_contract=expected_contract)


def require_preflight_report(path: Path, expected_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a preflight report and fail unless it explicitly permits execution."""
    if not path.exists():
        raise SystemExit(f"Required preflight report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Required preflight report is not valid JSON: {path}") from exc
    blockers = report.get("hard_blockers", [])
    if report.get("ok") is not True or blockers:
        blocker_text = "; ".join(str(item) for item in blockers) or "preflight report ok is not true"
        raise SystemExit(f"Required preflight report blocks execution: {path}: {blocker_text}")
    if expected_contract:
        contract_blockers = preflight_contract_blockers(report, expected_contract)
        if contract_blockers:
            blocker_text = "; ".join(contract_blockers)
            raise SystemExit(f"Required preflight report contract mismatch: {path}: {blocker_text}")
    return report


def enforce_preflight_report(
    path: Path | None,
    context: str = "paid API calls",
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require an explicit preflight report path, then validate the report."""
    if path is None:
        raise SystemExit(f"--require-preflight-report is required before {context}")
    return require_preflight_report(path, expected_contract=expected_contract)


def budget_contract_blockers(report: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    actual = report.get("contract")
    if not isinstance(actual, dict):
        return ["budget report is missing contract"]
    blockers = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if normalize_contract_value(actual_value) != normalize_contract_value(expected_value):
            blockers.append(
                f"{key} mismatch: expected {normalize_contract_value(expected_value)!r}, "
                f"got {normalize_contract_value(actual_value)!r}"
            )
    for path_key, sha_key in [
        ("input", "input_sha256"),
        ("providers", "providers_sha256"),
        ("resume_output", "resume_output_sha256"),
    ]:
        if path_key not in expected:
            continue
        path = normalize_contract_value(expected[path_key])
        expected_sha = normalize_contract_value(actual.get(sha_key))
        if not expected_sha:
            if path_key == "resume_output" and path is not None and not Path(path).exists():
                continue
            blockers.append(f"budget report is missing {sha_key}")
            continue
        if path is None:
            blockers.append(f"{path_key} path is missing from expected budget contract")
            continue
        current_path = Path(path)
        if not current_path.exists():
            blockers.append(f"{path_key} file from budget report is missing now: {path!r}")
            continue
        current_sha = file_sha256(current_path)
        if current_sha != expected_sha:
            blockers.append(
                f"{path_key} sha256 mismatch for {path!r}: expected {expected_sha!r}, got {current_sha!r}"
            )
    return blockers


def preflight_contract_blockers(report: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    blockers = []
    if "providers" in expected:
        for path in normalize_contract_list(expected["providers"]):
            group = find_report_path(report.get("providers", []), path)
            if group is None:
                blockers.append(f"providers missing from preflight report: {path!r}")
                continue
            blockers.extend(file_hash_blockers(group, path, label="providers"))
            blockers.extend(provider_env_blockers(group, path))
    if "input" in expected:
        for path in normalize_contract_list(expected["input"]):
            item = find_report_path(report.get("inputs", []), path)
            if item is None:
                blockers.append(f"input missing from preflight report: {path!r}")
                continue
            blockers.extend(file_hash_blockers(item, path, label="input"))
    return blockers


def find_report_path(items: Any, path: str | None) -> dict[str, Any] | None:
    for item in items or []:
        if isinstance(item, dict) and normalize_contract_value(item.get("path")) == path:
            return item
    return None


def file_hash_blockers(item: dict[str, Any], path: str | None, label: str) -> list[str]:
    blockers = []
    expected_sha = str(item.get("sha256", "") or "")
    if not expected_sha:
        return [f"{label} preflight report is missing sha256: {path!r}"]
    if path is None:
        return [f"{label} path is missing from expected preflight contract"]
    current_path = Path(path)
    if not current_path.exists():
        return [f"{label} file from preflight report is missing now: {path!r}"]
    current_sha = file_sha256(current_path)
    if current_sha != expected_sha:
        blockers.append(
            f"{label} sha256 mismatch for {path!r}: expected {expected_sha!r}, got {current_sha!r}"
        )
    return blockers


def provider_env_blockers(group: dict[str, Any], path: str | None) -> list[str]:
    blockers = []
    for provider in group.get("providers", []):
        if not isinstance(provider, dict):
            continue
        name = str(provider.get("name") or "")
        api_key_env = str(provider.get("api_key_env") or "")
        if not api_key_env:
            blockers.append(f"provider in preflight report is missing api_key_env: {path!r}:{name or 'unknown'}")
            continue
        if not os.environ.get(api_key_env):
            provider_name = name or "unknown"
            blockers.append(f"missing current API env {api_key_env} for provider {provider_name}")
    return blockers


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_contract_list(value: Any) -> list[str | None]:
    if isinstance(value, (list, tuple, set)):
        return [normalize_contract_value(item) for item in value]
    return [normalize_contract_value(value)]


def normalize_contract_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
