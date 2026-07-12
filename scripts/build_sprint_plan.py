#!/usr/bin/env python3
"""Build a day-by-day sprint plan for BlindSpot-RL experiments and paper writing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.build_result_card import default_claim_ladder
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from build_result_card import default_claim_ladder


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = build_plan(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sprint_plan.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "sprint_plan.csv", rows)
    (args.output_dir / "sprint_plan.md").write_text(to_markdown(rows, config), encoding="utf-8")
    print(f"Wrote {len(rows)} sprint days to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BlindSpot-RL sprint plan artifacts.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Sprint plan config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Sprint plan config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Sprint plan config must be a JSON object: {path}")
    return data


def build_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    current_day = 1
    total_days = int(config.get("total_days", 20))
    for phase in config.get("phases", []):
        duration = int(phase["days"])
        for offset, task in enumerate(expand_tasks(phase, duration), start=0):
            day = current_day + offset
            if day > total_days:
                break
            rows.append(
                {
                    "day": day,
                    "phase": phase["name"],
                    "goal": task.get("goal", phase.get("goal", "")),
                    "commands": task.get("commands", []),
                    "artifacts": task.get("artifacts", []),
                    "evidence_gates": task.get("evidence_gates", phase.get("evidence_gates", [])),
                    "claim_ladder_levels": task.get(
                        "claim_ladder_levels",
                        phase.get("claim_ladder_levels", config.get("claim_ladder_levels", [])),
                    ),
                    "paper_claims": task.get("paper_claims", phase.get("paper_claims", [])),
                    "claim_discipline": task.get(
                        "claim_discipline",
                        phase.get(
                            "claim_discipline",
                            config.get(
                                "claim_discipline",
                                [
                                    "Do not write paper claims from this task unless the relevant evidence gates pass."
                                ],
                            ),
                        ),
                    ),
                    "exit_criteria": task.get("exit_criteria", phase.get("exit_criteria", "")),
                }
            )
        current_day += duration
        if current_day > total_days:
            break
    return rows


def expand_tasks(phase: dict[str, Any], duration: int) -> list[dict[str, Any]]:
    tasks = list(phase.get("tasks", []))
    if len(tasks) >= duration:
        return tasks[:duration]
    while len(tasks) < duration:
        tasks.append(
            {
                "goal": phase.get("goal", ""),
                "commands": phase.get("commands", []),
                "artifacts": phase.get("artifacts", []),
            }
        )
    return tasks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "day",
        "phase",
        "goal",
        "commands",
        "artifacts",
        "evidence_gates",
        "claim_ladder_levels",
        "paper_claims",
        "claim_discipline",
        "exit_criteria",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "; ".join(row[key]) if isinstance(row.get(key), list) else row.get(key, "")
                    for key in fieldnames
                }
            )


def to_markdown(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    lines = [
        f"# {config.get('title', 'BlindSpot-RL Sprint Plan')}",
        "",
        f"- Total days: {config.get('total_days', len(rows))}",
        f"- Owner note: {config.get('owner_note', 'Keep claims evidence-gated.')}",
        "",
        "| Day | Phase | Goal | Commands | Artifacts | Evidence Gates | Claim Ladder | Paper Claims |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["day"]),
                    escape_md(row["phase"]),
                    escape_md(row["goal"]),
                    escape_md("<br>".join(code(item) for item in row["commands"])),
                    escape_md("<br>".join(code(item) for item in row["artifacts"])),
                    escape_md("<br>".join(row["evidence_gates"])),
                    escape_md("<br>".join(row.get("claim_ladder_levels", [])) or "none"),
                    escape_md("<br>".join(row["paper_claims"]) if row["paper_claims"] else "none"),
                ]
            )
            + " |"
        )
    global_discipline = config.get("claim_discipline", [])
    lines.extend(["", "## Global Claim Discipline", ""])
    lines.extend([f"- {item}" for item in global_discipline] or ["- Do not write paper claims unless evidence gates pass."])
    claim_ladder = config.get("claim_ladder") or default_claim_ladder()
    lines.extend(
        [
            "",
            "## Claim Ladder Milestones",
            "",
            "| Level | Required Claims | Evidence Required | Downgrade Rule |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in claim_ladder:
        lines.append(
            f"| {escape_md(row.get('level', ''))} | "
            f"{escape_md(', '.join(str(item) for item in row.get('required_claim_ids', [])))} | "
            f"{escape_md(row.get('evidence_required', ''))} | "
            f"{escape_md(row.get('downgrade_rule', ''))} |"
        )
    lines.extend(["", "## Claim Discipline By Day", ""])
    for row in rows:
        discipline = row.get("claim_discipline", [])
        if isinstance(discipline, str):
            discipline = [discipline]
        lines.append(f"- Day {row['day']}: {'; '.join(discipline) if discipline else 'No paper claim may be written from this day alone.'}")
    lines.extend(["", "## Exit Criteria By Day", ""])
    for row in rows:
        lines.append(f"- Day {row['day']}: {row['exit_criteria'] or 'No explicit criterion configured.'}")
    return "\n".join(lines) + "\n"


def code(value: str) -> str:
    return f"`{value}`"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
