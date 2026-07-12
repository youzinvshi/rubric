#!/usr/bin/env python3
"""Sync generated experiment artifacts into the paper workspace."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

try:
    from scripts.check_paper_asset_index import (
        semantic_space_contract_blockers,
        summarize_blockers,
        summarize_warnings,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from check_paper_asset_index import semantic_space_contract_blockers, summarize_blockers, summarize_warnings


TABLE_FILES = [
    "main_table.tex",
    "rl_stage_ablation_table.tex",
    "downstream_utility_table.tex",
    "ablation_table.tex",
    "teacher_union_ablation_table.tex",
    "verifier_filter_ablation_table.tex",
    "dimension_transition_table.tex",
]

FIGURE_FILES = [
    "semantic_space.svg",
    "semantic_space.pdf",
]

DOC_FILES = [
    "api_handoff.json",
    "api_handoff.md",
    "audit_report.json",
    "evidence_matrix.csv",
    "evidence_matrix.json",
    "evidence_matrix.md",
    "experiment_summary.md",
    "readiness_report.json",
    "readiness_report.md",
    "real_run_dashboard.json",
    "real_run_dashboard.md",
    "rebuttal_pack_manifest.json",
    "rebuttal_pack.json",
    "rebuttal_pack.md",
    "result_card.md",
    "result_card.json",
    "sprint_plan.json",
    "sprint_plan.csv",
    "sprint_plan.md",
    "submission_gap_report.json",
    "submission_gap_report.md",
    "semantic_space_points.csv",
    "semantic_space_summary.json",
]

DEPRECATED_DOC_FILES = [
    "paper_asset_index_check.json",
    "paper_asset_index_check.md",
]

DEPRECATED_TABLE_FILES = [
    "repair_table.tex",
]

REQUIRED_FILES = {
    *TABLE_FILES,
    "semantic_space.svg",
    "semantic_space.pdf",
    "semantic_space_points.csv",
    "semantic_space_summary.json",
    "experiment_summary.md",
}

TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}

NARRATIVE_BLOCKLIST = [
    "better rubric generator",
    "rubric-generator",
    "rubric generator",
    "engineering choice",
    "engineering scaling choice",
    "state-of-the-art",
    "state of the art",
    "sota",
    "achieving sota",
    "achieves sota",
    "state-of-the-art results",
    "guarantee acceptance",
    "guaranteed acceptance",
    "significantly improves",
    "significantly outperforms",
    "sft+grpo improves",
    "rlvr stage improves",
    "grpo/rlvr stage improves",
    "downstream utility improves",
    "accuracy improves",
    "higher bsc proves downstream",
    "higher bsc proves judge utility",
    "higher bsc alone supports judge utility",
    "bsc alone supports judge utility",
    "bsc gain supports judge utility",
    "bsc gains support judge utility",
    "coverage gain proves blind-spot reduction",
    "coverage gain demonstrates blind-spot reduction",
    "higher bsc proves blind-spot reduction",
    "sft/rl reduction",
    "we prove",
    "promote empirical claims",
    "promoting empirical claims",
    "promoting",
    "promotable",
    "do not promote",
    "we show that sft+grpo",
    "not synced yet",
    "main table not synced",
    "run scripts/sync_paper_artifacts.py",
    "repairs baseline blind spots",
    "repair evaluation blind spots",
    "repairing evaluation blind spots",
    "blind-spot repair",
    "blind-spot repair rate",
    "repair rate",
    "repairrate",
    "criteria-text improvement",
    "criteria-text improvement problem",
]


def main() -> None:
    args = parse_args()
    required_files = set(args.required_file) if args.required_file is not None else None
    blockers, warnings = inspect_artifacts(
        args.artifacts_dir,
        required_files=required_files,
        supplied_doc_names={path.name for path in args.extra_doc},
    )
    extra_blockers = inspect_extra_docs(args.extra_doc)
    blockers.extend(extra_blockers)
    copied = sync_artifacts(args.artifacts_dir, args.paper_dir, extra_docs=args.extra_doc)
    write_asset_index(args.paper_dir, args.artifacts_dir, copied, blockers=blockers, warnings=warnings)
    print(f"Synced {len(copied)} artifacts into {args.paper_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync generated artifacts into paper/ workspace.")
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--paper-dir", default=Path("paper"), type=Path)
    parser.add_argument("--extra-doc", action="append", default=[], type=Path, help="Additional document to copy into paper/asset_index by basename.")
    parser.add_argument(
        "--required-file",
        action="append",
        default=None,
        help="Artifact basename that must exist. Repeatable. Defaults to the full paper-facing real-run set.",
    )
    return parser.parse_args()


def sync_artifacts(artifacts_dir: Path, paper_dir: Path, extra_docs: list[Path] | None = None) -> list[dict[str, str]]:
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    asset_dir = paper_dir / "asset_index"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    for name in TABLE_FILES:
        source = artifacts_dir / name
        target = tables_dir / name
        if source.exists() and source.stat().st_size > 0:
            shutil.copyfile(source, target)
            copied.append({"kind": "table", "source": str(source), "target": str(target), "sha256": file_sha256(target)})
        elif target.exists():
            target.unlink()
    for name in DEPRECATED_TABLE_FILES:
        target = tables_dir / name
        if target.exists():
            target.unlink()

    for name in FIGURE_FILES:
        source = artifacts_dir / name
        target = figures_dir / name
        if source.exists() and source.stat().st_size > 0:
            shutil.copyfile(source, target)
            copied.append({"kind": "figure", "source": str(source), "target": str(target), "sha256": file_sha256(target)})
        elif target.exists():
            target.unlink()

    for name in DOC_FILES:
        source = artifacts_dir / name
        target = asset_dir / name
        if source.exists() and source.stat().st_size > 0:
            shutil.copyfile(source, target)
            copied.append({"kind": "doc", "source": str(source), "target": str(target), "sha256": file_sha256(target)})
        elif target.exists():
            target.unlink()
    for name in DEPRECATED_DOC_FILES:
        target = asset_dir / name
        if target.exists():
            target.unlink()
    for source in extra_docs or []:
        artifact_target = artifacts_dir / source.name
        target = asset_dir / source.name
        if source.exists() and source.stat().st_size > 0:
            if source.resolve() != artifact_target.resolve():
                shutil.copyfile(source, artifact_target)
            shutil.copyfile(source, target)
            copied = [item for item in copied if item.get("target") != str(target)]
            copied.append({"kind": "doc", "source": str(source), "target": str(target), "sha256": file_sha256(target)})
        elif target.exists():
            target.unlink()
            copied = [item for item in copied if item.get("target") != str(target)]
    return copied


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_artifacts(
    artifacts_dir: Path,
    required_files: set[str] | None = None,
    supplied_doc_names: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    required_files = REQUIRED_FILES if required_files is None else required_files
    supplied_doc_names = supplied_doc_names or set()
    if not artifacts_dir.exists():
        blockers.append(f"artifacts directory is missing: {artifacts_dir}")
        return blockers, warnings
    for name in TABLE_FILES + FIGURE_FILES + DOC_FILES:
        path = artifacts_dir / name
        required = name in required_files
        if not path.exists() or path.stat().st_size == 0:
            if name in supplied_doc_names:
                continue
            message = f"artifact is missing or empty: {path}"
            if required:
                blockers.append(message)
            else:
                warnings.append(message)
        else:
            blockers.extend(inspect_narrative_text(path))
    blockers.extend(inspect_semantic_space_contract(artifacts_dir))
    return blockers, warnings


def inspect_semantic_space_contract(artifacts_dir: Path) -> list[str]:
    required = [
        artifacts_dir / "semantic_space.svg",
        artifacts_dir / "semantic_space.pdf",
        artifacts_dir / "semantic_space_points.csv",
        artifacts_dir / "semantic_space_summary.json",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return []
    return semantic_space_contract_blockers(
        summary_path=artifacts_dir / "semantic_space_summary.json",
        points_path=artifacts_dir / "semantic_space_points.csv",
        svg_path=artifacts_dir / "semantic_space.svg",
        pdf_path=artifacts_dir / "semantic_space.pdf",
        paper_path="outputs/paper_artifacts/semantic_space_summary.json",
    )


def inspect_extra_docs(extra_docs: list[Path]) -> list[str]:
    blockers: list[str] = []
    for path in extra_docs:
        if not path.exists() or path.stat().st_size == 0:
            blockers.append(f"extra doc is missing or empty: {path}")
        else:
            blockers.extend(inspect_narrative_text(path))
    return blockers


def inspect_narrative_text(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    normalized = normalize_narrative_text(lowered)
    blockers: list[str] = []
    for phrase in NARRATIVE_BLOCKLIST:
        if phrase in lowered or phrase in normalized:
            line_no = first_matching_line(lowered, phrase)
            blockers.append(f"narrative blocker in {path}: unsupported phrase `{phrase}` at line {line_no}")
    return blockers


def normalize_narrative_text(text: str) -> str:
    normalized = text.replace("\\_", "_")
    normalized = re.sub(r"\\texttt\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def first_matching_line(text: str, phrase: str) -> int:
    for idx, line in enumerate(text.splitlines(), start=1):
        if phrase in line:
            return idx
    return 1


def write_asset_index(
    paper_dir: Path,
    artifacts_dir: Path,
    copied: list[dict[str, str]],
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    blockers = blockers or []
    warnings = warnings or []
    path = paper_dir / "asset_index.md"
    lines = [
        "# BlindSpot-RL Paper Asset Index",
        "",
        f"- Source artifact directory: `{artifacts_dir}`",
        f"- Synced artifacts: {len(copied)}",
        f"- Blockers: {len(blockers)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    blocker_summary = summarize_blockers([f"asset index declares blocker: {item}" for item in blockers])
    lines.extend(
        [
            "",
            "## Blocker Summary",
            "",
        ]
    )
    for item in blocker_summary:
        lines.append(f"- `{item['category']}`: {item['count']} ({item['label']})")
    if not blocker_summary:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in warnings] or ["- none"])
    warning_summary = summarize_warnings([f"asset index declares warning: {item}" for item in warnings])
    lines.extend(
        [
            "",
            "## Warning Summary",
            "",
        ]
    )
    for item in warning_summary:
        lines.append(f"- `{item['category']}`: {item['count']} ({item['label']})")
    if not warning_summary:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Synced Artifacts",
            "",
            "| Kind | Source | Paper Path | SHA256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in copied:
        lines.append(f"| {item['kind']} | `{item['source']}` | `{item['target']}` | `{item['sha256']}` |")
    if not copied:
        lines.append("| none |  |  |  |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
