#!/usr/bin/env python3
"""Fail unless the minimal API handoff explicitly permits the paid range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    status = check_handoff_ready(args.handoff)
    print(status["message"])
    if status.get("paid_run_command"):
        print(status["paid_run_command"])
    if not status["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check minimal API handoff readiness.")
    parser.add_argument("--handoff", required=True, type=Path)
    return parser.parse_args()


def check_handoff_ready(path: Path) -> dict[str, Any]:
    data = load_handoff(path)
    resume = data.get("resume_requirements", {}) if isinstance(data.get("resume_requirements"), dict) else {}
    blockers = data.get("blockers", []) if isinstance(data.get("blockers"), list) else []
    ready = (
        data.get("ok") is True
        and data.get("status") == "ready_for_paid_run"
        and resume.get("ready") is True
        and not blockers
    )
    if ready:
        return {
            "ok": True,
            "message": f"Minimal API handoff is ready: {path}",
            "paid_run_command": resume.get("paid_run_command"),
        }
    missing_env = resume.get("missing_env", [])
    failed_local = resume.get("failed_local_providers", [])
    details = [
        f"Minimal API handoff is not ready: {path}",
        f"status={data.get('status', 'unknown')} ok={data.get('ok')}",
        f"missing_env={','.join(str(item) for item in missing_env) if missing_env else 'none'}",
        f"failed_local_providers={len(failed_local) if isinstance(failed_local, list) else 'unknown'}",
    ]
    next_command = resume.get("next_command")
    if next_command:
        details.append(f"next_command={next_command}")
    blocked_report_refresh_command = resume.get("blocked_report_refresh_command")
    if blocked_report_refresh_command:
        details.append(f"blocked_report_refresh_command={blocked_report_refresh_command}")
    return {
        "ok": False,
        "message": "\n".join(details),
        "paid_run_command": None,
    }


def load_handoff(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Minimal API handoff is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Minimal API handoff is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Minimal API handoff root must be an object: {path}")
    return data


if __name__ == "__main__":
    main()
