#!/usr/bin/env python3
"""Build a data-source readiness report for BlindSpot-RL datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PLACEHOLDER_HOSTS = {"example.com", "www.example.com", "example.org", "example.net"}
PLACEHOLDER_URL_TOKENS = ("todo", "tbd", "placeholder", "replace-me")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def main() -> None:
    args = parse_args()
    if args.config.exists():
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report = build_invalid_config_report(args.config, exc)
        else:
            report = build_report(config, required_datasets=args.required_dataset)
    else:
        report = build_missing_config_report(args.config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Data source report status={report['overall_status']} report={args.output_json}")
    if args.strict and report["overall_status"] == "blocked":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BlindSpot-RL data source readiness report.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--required-dataset",
        action="append",
        default=[],
        help="Limit blocking status and next actions to required dataset names. Repeatable.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_report(config: dict[str, Any], required_datasets: list[str] | None = None) -> dict[str, Any]:
    datasets = [summarize_dataset(dataset) for dataset in config.get("datasets", [])]
    scoped_datasets, missing = select_required_datasets(datasets, required_datasets or [])
    blockers = []
    warnings = []
    blockers.extend(missing)
    for dataset in scoped_datasets:
        blockers.extend(dataset["blockers"])
        warnings.extend(dataset["warnings"])
    overall_status = "blocked" if blockers else "warn" if warnings else "pass"
    return {
        "title": config.get("title", "BlindSpot-RL Data Source Report"),
        "overall_status": overall_status,
        "datasets": datasets,
        "required_datasets": required_datasets or [],
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": build_next_actions(scoped_datasets, config),
    }


def select_required_datasets(
    datasets: list[dict[str, Any]],
    required_datasets: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not required_datasets:
        return datasets, []
    by_name = {dataset["name"]: dataset for dataset in datasets}
    selected = []
    missing = []
    for name in required_datasets:
        dataset = by_name.get(name)
        if dataset is None:
            missing.append(f"data source report is missing required dataset: {name}")
        else:
            selected.append(dataset)
    return selected, missing


def build_missing_config_report(path: Path) -> dict[str, Any]:
    path_text = str(path)
    template_hint = path_text.replace(".local.", ".template.")
    actions = [f"Create local data-source config at {path_text}"]
    if template_hint != path_text:
        actions.insert(0, f"Copy {template_hint} to {path_text} and fill official URLs/field mappings")
    return {
        "title": "BlindSpot-RL Data Source Report",
        "overall_status": "blocked",
        "datasets": [],
        "blockers": [f"data source config is missing: {path_text}"],
        "warnings": [],
        "next_actions": actions,
    }


def build_invalid_config_report(path: Path, exc: json.JSONDecodeError) -> dict[str, Any]:
    path_text = str(path)
    return {
        "title": "BlindSpot-RL Data Source Report",
        "overall_status": "blocked",
        "datasets": [],
        "blockers": [f"data source config is not valid JSON: {path_text}: line {exc.lineno} column {exc.colno}"],
        "warnings": [],
        "next_actions": [f"Fix JSON syntax in {path_text} before running data acquisition/normalization"],
    }


def summarize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    name = dataset["name"]
    source = dataset["source"]
    source_type = source.get("type", "unknown")
    raw_path = Path(source.get("raw_path") or source.get("output") or "")
    raw_present = bool(raw_path and raw_path.exists() and raw_path.stat().st_size > 0)
    require_official_url = bool(source.get("require_official_url", False))
    require_raw_sha256 = bool(source.get("require_raw_sha256", False))
    expected_raw_sha256 = str(source.get("raw_sha256", "") or "").strip().lower()
    actual_raw_sha256 = sha256_file(raw_path) if raw_present else ""
    normalizations = [summarize_output(item["output"], item["target"]) for item in dataset.get("normalizations", [])]
    profiles = []
    if dataset.get("profile"):
        profiles.append(summarize_output(dataset["profile"]["output"], "profile"))

    blockers = []
    warnings = []
    if source_type == "manual" and not raw_present:
        blockers.append(f"{name}: manual raw source is missing at {raw_path}")
    if source_type == "manual" and require_official_url and not source.get("official_url"):
        blockers.append(f"{name}: manual source requires official_url before hard-gold claims")
    if source_type == "manual" and require_official_url and source.get("official_url"):
        if not is_valid_official_url(str(source["official_url"])):
            blockers.append(f"{name}: official_url must be a non-placeholder http(s) URL")
    if source_type == "manual" and require_raw_sha256 and not expected_raw_sha256:
        blockers.append(f"{name}: manual source requires raw_sha256 before hard-gold claims")
    if source_type == "manual" and expected_raw_sha256 and not is_valid_sha256(expected_raw_sha256):
        blockers.append(f"{name}: raw_sha256 must be a 64-character hex digest")
    if (
        source_type == "manual"
        and is_valid_sha256(expected_raw_sha256)
        and raw_present
        and actual_raw_sha256 != expected_raw_sha256
    ):
        blockers.append(f"{name}: raw_sha256 mismatch for {raw_path}")
    if source_type == "hf" and not raw_present:
        warnings.append(f"{name}: raw HF export not present yet at {raw_path}")
    if source_type == "url" and not raw_present:
        warnings.append(f"{name}: raw URL export not present yet at {raw_path}")
    missing_normalized = [item for item in normalizations if not item["present"]]
    if missing_normalized:
        warnings.append(f"{name}: {len(missing_normalized)} normalized output(s) not present yet")
    missing_profiles = [item for item in profiles if not item["present"]]
    if missing_profiles:
        warnings.append(f"{name}: schema profile not present yet")

    status = "blocked" if blockers else "warn" if warnings else "pass"
    return {
        "name": name,
        "source_type": source_type,
        "raw_path": str(raw_path),
        "raw_present": raw_present,
        "require_official_url": require_official_url,
        "official_url": source.get("official_url", ""),
        "require_raw_sha256": require_raw_sha256,
        "raw_sha256": expected_raw_sha256,
        "actual_raw_sha256": actual_raw_sha256,
        "paper_url": source.get("paper_url", ""),
        "note": source.get("note", ""),
        "expected_fields": dataset.get("expected_fields", []),
        "profiles": profiles,
        "normalizations": normalizations,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
    }


def summarize_output(path: str, label: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "label": label,
        "path": path,
        "present": p.exists() and p.stat().st_size > 0,
        "bytes": p.stat().st_size if p.exists() else 0,
    }


def build_next_actions(datasets: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    actions = []
    for dataset in datasets:
        if dataset["source_type"] == "manual" and dataset.get("require_official_url") and not dataset.get("official_url"):
            actions.append(f"Set {dataset['name']} official_url in the local data-source config")
        elif dataset["source_type"] == "manual" and dataset.get("require_official_url"):
            if not is_valid_official_url(str(dataset.get("official_url", ""))):
                actions.append(f"Replace {dataset['name']} official_url with the non-placeholder official release URL")
        if dataset["source_type"] == "manual" and dataset.get("require_raw_sha256") and not dataset.get("raw_sha256"):
            actions.append(f"Set {dataset['name']} raw_sha256 from the official release file")
        elif dataset["source_type"] == "manual" and dataset.get("require_raw_sha256"):
            if not is_valid_sha256(str(dataset.get("raw_sha256", ""))):
                actions.append(f"Set {dataset['name']} raw_sha256 to a 64-character hex digest")
        if dataset["source_type"] == "manual" and not dataset["raw_present"]:
            if dataset.get("official_url"):
                actions.append(f"Download {dataset['name']} official release: {dataset['official_url']}")
                actions.append(
                    "Run "
                    f"`python3 scripts/download_public_data.py --url {dataset['official_url']} "
                    f"--output {dataset['raw_path']}`"
                )
            else:
                actions.append(f"Place {dataset['name']} raw JSONL at {dataset['raw_path']}")
        elif dataset["source_type"] == "hf" and not dataset["raw_present"]:
            actions.append(f"Run the generated download stage for {dataset['name']}")
        elif dataset["source_type"] == "url" and not dataset["raw_present"]:
            actions.append(f"Run the generated URL download stage for {dataset['name']}")
        if dataset["raw_present"] and any(not item["present"] for item in dataset["profiles"]):
            actions.append(f"Profile {dataset['name']} schema before setting field mappings")
        if dataset["raw_present"] and any(not item["present"] for item in dataset["normalizations"]):
            actions.append(f"Normalize {dataset['name']} into project schema")
    actions.extend(config.get("next_actions", []))
    return dedupe(actions)


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Blockers: `{len(report['blockers'])}`",
        f"- Warnings: `{len(report['warnings'])}`",
        "",
        "| Dataset | Source | Status | Raw Present | Raw Path | Expected Fields |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in report["datasets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(dataset["name"]),
                    code(dataset["source_type"]),
                    code(dataset["status"]),
                    code(dataset["raw_present"]),
                    code(dataset["raw_path"]),
                    escape_md(", ".join(dataset.get("expected_fields", []))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in report["blockers"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- none"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend([f"- {item}" for item in report["next_actions"]] or ["- none"])
    lines.extend(["", "## Source Notes", ""])
    for dataset in report["datasets"]:
        lines.append(f"### {dataset['name']}")
        if dataset.get("official_url"):
            lines.append(f"- Official URL: {dataset['official_url']}")
        if dataset.get("require_official_url"):
            lines.append("- Requires official URL: true")
        if dataset.get("require_raw_sha256"):
            lines.append("- Requires raw SHA256: true")
        if dataset.get("raw_sha256"):
            lines.append(f"- Expected raw SHA256: `{dataset['raw_sha256']}`")
        if dataset.get("actual_raw_sha256"):
            lines.append(f"- Actual raw SHA256: `{dataset['actual_raw_sha256']}`")
        if dataset.get("paper_url"):
            lines.append(f"- Paper URL: {dataset['paper_url']}")
        if dataset.get("note"):
            lines.append(f"- Note: {dataset['note']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def code(value: Any) -> str:
    return f"`{value}`"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_sha256(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(value.strip().lower()))


def is_valid_official_url(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    if host in PLACEHOLDER_HOSTS:
        return False
    lowered = text.lower()
    return not any(token in lowered for token in PLACEHOLDER_URL_TOKENS)


if __name__ == "__main__":
    main()
