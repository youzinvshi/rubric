#!/usr/bin/env python3
"""Preflight checks before launching real BlindSpot-RL data/model runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def main() -> None:
    args = parse_args()
    report = build_preflight_report(
        inputs=args.input,
        providers=args.providers,
        training_config=args.training_config,
        required_env=args.required_env,
        required_provider=args.required_provider,
        required_provider_in=args.required_provider_in,
        min_records=args.min_records,
        check_local_provider_health=args.check_local_provider_health,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Preflight ok={report['ok']} report={args.output}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline preflight checks for real BlindSpot-RL runs.")
    parser.add_argument("--input", action="append", default=[], type=Path, help="Input data file to check. Repeatable.")
    parser.add_argument("--providers", action="append", default=[], type=Path, help="LLM provider JSONL. Repeatable.")
    parser.add_argument("--required-provider", action="append", default=[], help="Required provider/generator name. Repeatable.")
    parser.add_argument(
        "--required-provider-in",
        action="append",
        default=[],
        help="Required provider names bound to one file, in the form path:name1,name2,...",
    )
    parser.add_argument("--training-config", type=Path, help="training_commands JSON config.")
    parser.add_argument("--required-env", action="append", default=[], help="Extra required env var. Repeatable.")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument(
        "--check-local-provider-health",
        action="store_true",
        help="Require localhost/loopback provider endpoints to accept a TCP connection.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_preflight_report(
    inputs: list[Path],
    providers: list[Path],
    training_config: Path | None = None,
    required_env: list[str] | None = None,
    required_provider: list[str] | None = None,
    required_provider_in: list[str] | None = None,
    min_records: int = 1,
    check_local_provider_health: bool = False,
) -> dict[str, Any]:
    input_checks = [check_input_file(path, min_records=min_records) for path in inputs]
    provider_checks = [
        check_provider_file(path, check_local_provider_health=check_local_provider_health)
        for path in providers
    ]
    provider_name_check = check_required_providers(provider_checks, required_provider or [])
    provider_file_checks = [check_required_providers_in_file(provider_checks, spec) for spec in (required_provider_in or [])]
    training_check = check_training_config(training_config) if training_config else None
    env_checks = [check_env_var(name) for name in (required_env or [])]

    hard_blockers = []
    warnings = []
    for item in input_checks:
        if not item["present"]:
            hard_blockers.append(f"missing input file: {item['path']}")
        elif item.get("read_error"):
            hard_blockers.append(f"input file is not readable: {item['path']}: {item['read_error']}")
        elif item["records"] < min_records:
            hard_blockers.append(f"input file has too few records: {item['path']} ({item['records']})")
    for group in provider_checks:
        hard_blockers.extend(group["hard_blockers"])
        warnings.extend(group["warnings"])
    hard_blockers.extend(provider_name_check["hard_blockers"])
    for item in provider_file_checks:
        hard_blockers.extend(item["hard_blockers"])
    if training_check:
        hard_blockers.extend(training_check["hard_blockers"])
        warnings.extend(training_check["warnings"])
    for item in env_checks:
        if not item["present"]:
            hard_blockers.append(f"missing required env var: {item['name']}")

    return {
        "ok": not hard_blockers,
        "inputs": input_checks,
        "providers": provider_checks,
        "provider_names": provider_name_check,
        "provider_file_requirements": provider_file_checks,
        "training": training_check,
        "env": env_checks,
        "blockers": hard_blockers,
        "hard_blockers": hard_blockers,
        "warnings": warnings,
    }


def check_input_file(path: Path, min_records: int = 1) -> dict[str, Any]:
    present = path.exists() and path.is_file()
    records = 0
    read_error = ""
    sha256 = ""
    if present:
        sha256 = file_sha256(path)
        try:
            records = count_records(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            read_error = format_json_error(exc)
    return {
        "path": str(path),
        "present": present,
        "bytes": path.stat().st_size if present else 0,
        "sha256": sha256,
        "records": records,
        "min_records": min_records,
        "read_error": read_error,
    }


def count_records(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        for key in ("records", "data", "examples"):
            if isinstance(data.get(key), list):
                return len(data[key])
        return 1
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return 0
        return int(len(pd.read_parquet(path)))
    return 1


def check_provider_file(path: Path, check_local_provider_health: bool = False) -> dict[str, Any]:
    result = {
        "path": str(path),
        "present": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": file_sha256(path) if path.exists() else "",
        "providers": [],
        "hard_blockers": [],
        "warnings": [],
    }
    if not path.exists():
        result["hard_blockers"].append(f"missing provider config: {path}")
        return result

    rows, read_errors = read_jsonl_objects(path)
    for error in read_errors:
        result["hard_blockers"].append(f"invalid provider JSONL at {path}:{error}")
    for line_no, obj in rows:
        provider = check_provider(line_no, obj, check_local_provider_health=check_local_provider_health)
        result["providers"].append(provider)
        if not provider["valid_schema"]:
            result["hard_blockers"].append(f"invalid provider schema at {path}:{line_no}")
        if not provider["api_key_present"]:
            result["hard_blockers"].append(
                f"missing API env {provider['api_key_env']} for provider {provider['name']}"
            )
        if not provider["base_url_valid"]:
            result["hard_blockers"].append(f"invalid base_url for provider {provider['name']}: {provider['base_url']}")
        if check_local_provider_health and provider.get("local_health", {}).get("ok") is False:
            result["hard_blockers"].append(
                f"local provider endpoint is not reachable for provider {provider['name']}: {provider['base_url']}"
            )
        if provider["model_looks_like_path"] and not provider["model_path_exists"]:
            result["warnings"].append(f"model path does not exist yet for provider {provider['name']}: {provider['model']}")
    if not result["providers"]:
        result["hard_blockers"].append(f"provider config has no records: {path}")
    return result


def check_required_providers(provider_checks: list[dict[str, Any]], required: list[str]) -> dict[str, Any]:
    present = sorted(
        {
            str(provider.get("name", ""))
            for group in provider_checks
            for provider in group.get("providers", [])
            if provider.get("name")
        }
    )
    missing = [name for name in required if name not in present]
    return {
        "required": required,
        "present": present,
        "missing": missing,
        "hard_blockers": [f"missing required provider: {name}" for name in missing],
    }


def check_required_providers_in_file(provider_checks: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    try:
        path_text, required = parse_required_provider_in_spec(spec)
    except ValueError as exc:
        return {
            "path": "",
            "required": [],
            "present": [],
            "missing": [],
            "hard_blockers": [str(exc)],
        }
    group = next((item for item in provider_checks if item.get("path") == path_text), None)
    present = sorted(
        str(provider.get("name", ""))
        for provider in (group or {}).get("providers", [])
        if provider.get("name")
    )
    missing = [name for name in required if name not in present]
    return {
        "path": path_text,
        "required": required,
        "present": present,
        "missing": missing,
        "hard_blockers": [f"missing required provider in {path_text}: {name}" for name in missing],
    }


def parse_required_provider_in_spec(spec: str) -> tuple[str, list[str]]:
    if ":" not in spec:
        raise ValueError(f"Invalid required-provider-in spec: {spec}")
    path, raw_names = spec.split(":", 1)
    names = [item.strip() for item in raw_names.split(",") if item.strip()]
    if not path.strip() or not names:
        raise ValueError(f"Invalid required-provider-in spec: {spec}")
    return path.strip(), names


def read_jsonl_objects(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{line_no}: line {exc.lineno} column {exc.colno}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"{line_no}: provider record must be a JSON object")
                continue
            rows.append((line_no, obj))
    return rows, errors


def check_provider(
    line_no: int,
    obj: dict[str, Any],
    check_local_provider_health: bool = False,
) -> dict[str, Any]:
    required = ["name", "model", "base_url", "api_key_env"]
    valid_schema = all(obj.get(key) for key in required)
    base_url = str(obj.get("base_url", ""))
    parsed = urlparse(base_url)
    api_key_env = str(obj.get("api_key_env", ""))
    model = str(obj.get("model", ""))
    model_path = Path(model)
    model_looks_like_path = model.startswith(".") or model.startswith("/") or "/" in model and model.startswith("outputs")
    result = {
        "line": line_no,
        "name": obj.get("name", ""),
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "valid_schema": valid_schema,
        "api_key_present": bool(os.environ.get(api_key_env)) if api_key_env else False,
        "base_url_valid": parsed.scheme in {"http", "https"} and bool(parsed.netloc),
        "model_looks_like_path": model_looks_like_path,
        "model_path_exists": model_path.exists() if model_looks_like_path else None,
        "local_health": local_provider_health(parsed, enabled=check_local_provider_health),
    }
    return result


def local_provider_health(parsed: Any, enabled: bool = False, timeout_seconds: float = 0.5) -> dict[str, Any]:
    host = parsed.hostname
    port = parsed.port
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    if not enabled or not is_local:
        return {"checked": False, "is_local": bool(is_local), "ok": None, "error": ""}
    if not port:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return {"checked": True, "is_local": True, "ok": True, "host": host, "port": port, "error": ""}
    except OSError as exc:
        return {
            "checked": True,
            "is_local": True,
            "ok": False,
            "host": host,
            "port": port,
            "error": str(exc),
        }


def check_training_config(path: Path | None) -> dict[str, Any]:
    result = {
        "path": str(path) if path else "",
        "present": bool(path and path.exists()),
        "sft": {},
        "grpo": {},
        "hard_blockers": [],
        "warnings": [],
    }
    if not path or not path.exists():
        result["hard_blockers"].append(f"missing training config: {path}")
        return result
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["hard_blockers"].append(f"training config is not readable: {path}: {format_json_error(exc)}")
        return result
    if not isinstance(config, dict):
        result["hard_blockers"].append(f"training config must be a JSON object: {path}")
        return result
    for section in ("sft", "grpo"):
        section_config = config.get(section, {})
        yaml_path = Path(str(section_config.get("config", "")))
        output_dir = Path(str(section_config.get("output_dir", ""))) if section_config.get("output_dir") else None
        env = section_config.get("env", {})
        result[section] = {
            "config": str(yaml_path),
            "config_exists": yaml_path.exists(),
            "output_dir": str(output_dir) if output_dir else "",
            "env": [check_env_var(name) for name in env.keys()],
        }
        if not yaml_path.exists():
            result["warnings"].append(f"{section} config does not exist yet: {yaml_path}")
        for env_check in result[section]["env"]:
            if not env_check["present"]:
                result["warnings"].append(f"{section} env is not set: {env_check['name']}")
    return result


def format_json_error(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"not valid JSON at line {exc.lineno} column {exc.colno}"
    return str(exc)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_env_var(name: str) -> dict[str, Any]:
    value = os.environ.get(name)
    return {
        "name": name,
        "present": bool(value),
        "length": len(value) if value else 0,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BlindSpot-RL Real-Run Preflight",
        "",
        f"- Overall ok: `{report['ok']}`",
        "",
        "## Hard Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in report["hard_blockers"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- none"])
    lines.extend(["", "## Inputs", "", "| Path | Present | Records | Bytes | SHA256 |", "| --- | --- | --- | --- | --- |"])
    for item in report["inputs"]:
        lines.append(
            f"| `{item['path']}` | `{item['present']}` | {item['records']} | "
            f"{item['bytes']} | `{item.get('sha256', '')}` |"
        )
    lines.extend(["", "## Providers", ""])
    if report.get("provider_names"):
        names = report["provider_names"]
        lines.extend(
            [
                f"- Required names: `{', '.join(names.get('required', [])) or 'none'}`",
                f"- Present names: `{', '.join(names.get('present', [])) or 'none'}`",
                f"- Missing names: `{', '.join(names.get('missing', [])) or 'none'}`",
                "",
            ]
        )
    for item in report.get("provider_file_requirements", []):
        lines.extend(
            [
                f"### Required names in `{item['path']}`",
                "",
                f"- Required: `{', '.join(item.get('required', [])) or 'none'}`",
                f"- Present: `{', '.join(item.get('present', [])) or 'none'}`",
                f"- Missing: `{', '.join(item.get('missing', [])) or 'none'}`",
                "",
            ]
        )
    for group in report["providers"]:
        lines.append(f"### `{group['path']}`")
        lines.append("")
        lines.append(f"- SHA256: `{group.get('sha256', '')}`")
        lines.append("")
        lines.append("| Name | Base URL Valid | API Key Present | Local Health | Model Path Exists |")
        lines.append("| --- | --- | --- | --- | --- |")
        for provider in group["providers"]:
            health = provider.get("local_health", {})
            health_status = health.get("ok") if health.get("checked") else "not_checked"
            lines.append(
                f"| `{provider['name']}` | `{provider['base_url_valid']}` | "
                f"`{provider['api_key_present']}` | `{health_status}` | `{provider['model_path_exists']}` |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
