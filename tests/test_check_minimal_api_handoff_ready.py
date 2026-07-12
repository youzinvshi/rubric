from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_minimal_api_handoff_ready import check_handoff_ready


class CheckMinimalApiHandoffReadyTest(unittest.TestCase):
    def test_ready_handoff_returns_paid_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "api_handoff.json"
            write_json(
                handoff,
                {
                    "ok": True,
                    "status": "ready_for_paid_run",
                    "blockers": [],
                    "resume_requirements": {
                        "ready": True,
                        "paid_run_command": "python3 scripts/run_experiment_pipeline.py --from-stage generate_model_rubrics",
                    },
                },
            )

            status = check_handoff_ready(handoff)

        self.assertTrue(status["ok"])
        self.assertIn("ready", status["message"])
        self.assertIn("run_experiment_pipeline.py", status["paid_run_command"])

    def test_blocked_handoff_reports_missing_env_and_next_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "api_handoff.json"
            write_json(
                handoff,
                {
                    "ok": False,
                    "status": "blocked",
                    "blockers": ["preflight: ok=false"],
                    "resume_requirements": {
                        "ready": False,
                        "missing_env": ["LOCAL_OPENAI_API_KEY", "OPENAI_API_KEY"],
                        "failed_local_providers": [],
                        "next_command": "python3 scripts/run_experiment_pipeline.py --only preflight",
                        "blocked_report_refresh_command": "python3 scripts/run_experiment_pipeline.py --from-stage audit --to-stage paper_asset_index_check_post_sync",
                    },
                },
            )

            status = check_handoff_ready(handoff)

        self.assertFalse(status["ok"])
        self.assertIn("status=blocked", status["message"])
        self.assertIn("missing_env=LOCAL_OPENAI_API_KEY,OPENAI_API_KEY", status["message"])
        self.assertIn("next_command=python3 scripts/run_experiment_pipeline.py --only preflight", status["message"])
        self.assertIn(
            "blocked_report_refresh_command=python3 scripts/run_experiment_pipeline.py --from-stage audit --to-stage paper_asset_index_check_post_sync",
            status["message"],
        )
        self.assertIsNone(status["paid_run_command"])


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
