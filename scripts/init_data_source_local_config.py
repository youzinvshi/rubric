#!/usr/bin/env python3
"""Initialize and audit the local real-data source config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_data_source_report import build_report, is_valid_official_url, is_valid_sha256, sha256_file


def main() -> None:
    args = parse_args()
    template = load_json(args.template)
    output_existed = args.output.exists()
    config = load_json(args.output) if output_existed else template

    changed = False
    if not output_existed:
        changed = True
    if output_existed and args.update_existing:
        changed = sync_template_official_urls(config, template) or changed
    if args.fill_present_sha256:
        changed = fill_present_sha256(config) or changed

    report = build_init_report(
        config=config,
        template_path=args.template,
        output_path=args.output,
        output_existed=output_existed,
        wrote_config=changed,
        required_datasets=args.required_dataset,
    )

    if changed:
        if output_existed and not args.update_existing:
            report["blockers"].append(
                f"local config exists and --update-existing was not set, so pending changes were not written: {args.output}"
            )
            report["wrote_config"] = False
            report["ok"] = False
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report["wrote_config"] = True

    write_report(report, args.report_json, args.report_md)
    print(f"Data source local config ok={report['ok']} output={args.output}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize/audit configs/data_sources_real.local.json.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument(
        "--fill-present-sha256",
        action="store_true",
        help="Fill missing manual raw_sha256 values when the configured raw_path exists.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Allow writing updates to an existing local config. Without this, existing configs are audited only.",
    )
    parser.add_argument(
        "--required-dataset",
        action="append",
        default=[],
        help="Limit blocking status and next actions to required dataset names. Repeatable.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return data


def fill_present_sha256(config: dict[str, Any]) -> bool:
    changed = False
    for dataset in config.get("datasets", []):
        source = dataset.get("source", {})
        if source.get("type") != "manual" or source.get("raw_sha256"):
            continue
        raw_path = Path(source.get("raw_path", ""))
        if raw_path.exists() and raw_path.is_file() and raw_path.stat().st_size > 0:
            source["raw_sha256"] = sha256_file(raw_path)
            changed = True
    return changed


def sync_template_official_urls(config: dict[str, Any], template: dict[str, Any]) -> bool:
    changed = False
    template_by_name = {dataset.get("name"): dataset for dataset in template.get("datasets", [])}
    for dataset in config.get("datasets", []):
        name = dataset.get("name")
        source = dataset.get("source", {})
        template_source = template_by_name.get(name, {}).get("source", {})
        template_url = str(template_source.get("official_url", "") or "")
        current_url = str(source.get("official_url", "") or "")
        if template_url and is_valid_official_url(template_url) and not is_valid_official_url(current_url):
            source["official_url"] = template_url
            changed = True
        if template_source.get("download_enabled") and source.get("download_enabled") != template_source.get("download_enabled"):
            source["download_enabled"] = template_source["download_enabled"]
            changed = True
        template_note = template_source.get("note")
        current_note = source.get("note")
        if template_note and (not current_note or "Replace with the official" in str(current_note)):
            source["note"] = template_note
            changed = True
    return changed


def build_init_report(
    config: dict[str, Any],
    template_path: Path,
    output_path: Path,
    output_existed: bool,
    wrote_config: bool,
    required_datasets: list[str] | None = None,
) -> dict[str, Any]:
    source_report = build_report(config, required_datasets=required_datasets or [])
    hard_gold_actions = hard_gold_next_actions(source_report, required_datasets=required_datasets or [])
    warnings = list(source_report["warnings"])
    if not output_existed:
        warnings.append(f"local config was missing and has been initialized from template: {output_path}")
    return {
        "ok": source_report["overall_status"] != "blocked",
        "template": str(template_path),
        "output": str(output_path),
        "output_existed": output_existed,
        "wrote_config": wrote_config,
        "required_datasets": required_datasets or [],
        "source_overall_status": source_report["overall_status"],
        "source_blockers": source_report["blockers"],
        "source_warnings": source_report["warnings"],
        "hard_gold_next_actions": hard_gold_actions,
        "blockers": source_report["blockers"],
        "warnings": warnings,
    }


def hard_gold_next_actions(source_report: dict[str, Any], required_datasets: list[str] | None = None) -> list[str]:
    actions: list[str] = []
    required = set(required_datasets or [])
    for dataset in source_report.get("datasets", []):
        if required and dataset.get("name") not in required:
            continue
        if not dataset.get("require_official_url") and not dataset.get("require_raw_sha256"):
            continue
        actions.extend(item for item in dataset.get("blockers", []) if item not in actions)
        name = dataset["name"]
        if dataset.get("require_official_url") and not dataset.get("official_url"):
            append_once(actions, f"Set {name} official_url in the local data-source config")
        elif dataset.get("require_official_url") and not is_valid_official_url(str(dataset.get("official_url", ""))):
            append_once(actions, f"Replace {name} official_url with the non-placeholder official release URL")
        if dataset.get("require_raw_sha256") and not dataset.get("raw_sha256"):
            append_once(actions, f"Set {name} raw_sha256 from the official release file")
        elif dataset.get("require_raw_sha256") and not is_valid_sha256(str(dataset.get("raw_sha256", ""))):
            append_once(actions, f"Set {name} raw_sha256 to a 64-character hex digest")
        if not dataset.get("raw_present"):
            if dataset.get("official_url"):
                append_once(actions, f"Download {name} official release: {dataset['official_url']}")
                append_once(
                    actions,
                    "Run "
                    f"`python3 scripts/download_public_data.py --url {dataset['official_url']} "
                    f"--output {dataset['raw_path']}`",
                )
            else:
                append_once(actions, f"Place {name} raw JSONL at {dataset['raw_path']}")
    return actions


def append_once(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def write_report(report: dict[str, Any], report_json: Path | None, report_md: Path | None) -> None:
    if report_json:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report_md:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(to_markdown(report), encoding="utf-8")


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BlindSpot-RL Local Data Source Config",
        "",
        f"- OK: `{report['ok']}`",
        f"- Output: `{report['output']}`",
        f"- Output existed: `{report['output_existed']}`",
        f"- Wrote config: `{report['wrote_config']}`",
        f"- Source status: `{report['source_overall_status']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in report["blockers"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- none"])
    lines.extend(["", "## Hard-Gold Next Actions", ""])
    lines.extend([f"- {item}" for item in report["hard_gold_next_actions"]] or ["- none"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
