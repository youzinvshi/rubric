from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_real_api_handoff import build_handoff
from scripts.check_real_api_handoff_ready import check_handoff_ready


class BuildRealApiHandoffTest(unittest.TestCase):
    def test_real_handoff_blocks_on_preflight_and_missing_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            providers = write_text(root / "generators.jsonl", '{"name": "base"}\n')
            queries = write_text(root / "queries.jsonl", '{"query": "q"}\n')
            preflight = write_json(
                root / "preflight.json",
                {
                    "ok": False,
                    "hard_blockers": ["missing required env var: LOCAL_OPENAI_API_KEY"],
                    "env": [{"name": "LOCAL_OPENAI_API_KEY", "present": False, "length": 0}],
                    "inputs": [{"path": str(queries), "present": True, "sha256": sha256(queries), "records": 1}],
                    "providers": [
                        {
                            "path": str(providers),
                            "present": True,
                            "sha256": sha256(providers),
                            "providers": [
                                {
                                    "name": "base",
                                    "base_url": "http://localhost:8000/v1",
                                    "api_key_env": "LOCAL_OPENAI_API_KEY",
                                    "api_key_present": False,
                                    "local_health": {"checked": True, "ok": True},
                                }
                            ],
                        }
                    ],
                },
            )
            pipeline = write_json(
                root / "pipeline.json",
                {
                    "stages": [
                        {"name": "real_run_preflight", "type": "preflight"},
                        {
                            "name": "api_budget_model_rubrics",
                            "type": "api_budget",
                            "args": {"output": str(root / "missing_budget.json")},
                        },
                        {
                            "name": "training_completion_gate",
                            "type": "manual_gate",
                            "args": {"output": str(root / "missing_training_gate.json")},
                        },
                        {
                            "name": "generate_model_rubrics_rubricbench",
                            "type": "generate_model_rubrics",
                            "args": {
                                "input": str(queries),
                                "providers": str(providers),
                                "output": str(root / "model_rubrics.jsonl"),
                                "require_budget_report": str(root / "missing_budget.json"),
                                "require_preflight_report": str(preflight),
                            },
                        },
                        {"name": "result_card_real", "type": "result_card"},
                    ]
                },
            )

            report = build_handoff(
                pipeline_path=pipeline,
                preflight_path=preflight,
                output_json=root / "api_handoff.json",
                output_md=root / "api_handoff.md",
            )

            self.assertFalse(report["ok"])
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["scope"], "full_real_run_api_handoff")
            self.assertIn("preflight: ok=false", report["blockers"])
            self.assertTrue(any("missing_budget.json" in item for item in report["blockers"]))
            self.assertEqual(report["resume_requirements"]["missing_env"], ["LOCAL_OPENAI_API_KEY"])
            self.assertIn("--only real_run_preflight", report["resume_requirements"]["next_command"])
            self.assertIn("check_real_api_handoff_ready.py", report["commands"]["handoff_ready_check"])
            self.assertIn("not experimental evidence", (root / "api_handoff.md").read_text(encoding="utf-8"))

            ready = check_handoff_ready(root / "api_handoff.json")
            self.assertFalse(ready["ok"])
            self.assertIn("missing_env=LOCAL_OPENAI_API_KEY", ready["message"])

    def test_real_handoff_allows_paid_run_only_when_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            providers = write_text(root / "generators.jsonl", '{"name": "base"}\n')
            queries = write_text(root / "queries.jsonl", '{"query": "q"}\n')
            output = root / "model_rubrics.jsonl"
            preflight = write_json(
                root / "preflight.json",
                {
                    "ok": True,
                    "hard_blockers": [],
                    "env": [{"name": "LOCAL_OPENAI_API_KEY", "present": True, "length": 8}],
                    "inputs": [{"path": str(queries), "present": True, "sha256": sha256(queries), "records": 1}],
                    "providers": [
                        {
                            "path": str(providers),
                            "present": True,
                            "sha256": sha256(providers),
                            "providers": [
                                {
                                    "name": "base",
                                    "base_url": "http://localhost:8000/v1",
                                    "api_key_env": "LOCAL_OPENAI_API_KEY",
                                    "api_key_present": True,
                                    "local_health": {"checked": True, "ok": True, "host": "localhost", "port": 8000},
                                }
                            ],
                        }
                    ],
                },
            )
            budget = write_json(
                root / "budget.json",
                {
                    "ok": True,
                    "blockers": [],
                    "total": {"calls": 1, "total_tokens": 100, "estimated_cost_usd": 0.01},
                    "contract": {
                        "input": str(queries),
                        "input_sha256": sha256(queries),
                        "providers": str(providers),
                        "providers_sha256": sha256(providers),
                        "resume_output": str(output),
                        "resume_output_sha256": "",
                    },
                },
            )
            training_gate = write_json(root / "training_gate.json", {"ok": True, "blockers": []})
            pipeline = write_json(
                root / "pipeline.json",
                {
                    "stages": [
                        {"name": "real_run_preflight", "type": "preflight"},
                        {
                            "name": "api_budget_model_rubrics",
                            "type": "api_budget",
                            "args": {"output": str(budget)},
                        },
                        {
                            "name": "training_completion_gate",
                            "type": "manual_gate",
                            "args": {"output": str(training_gate)},
                        },
                        {
                            "name": "generate_model_rubrics_rubricbench",
                            "type": "generate_model_rubrics",
                            "args": {
                                "input": str(queries),
                                "providers": str(providers),
                                "output": str(output),
                                "require_budget_report": str(budget),
                                "require_preflight_report": str(preflight),
                            },
                        },
                        {"name": "result_card_real", "type": "result_card"},
                    ]
                },
            )

            report = build_handoff(
                pipeline_path=pipeline,
                preflight_path=preflight,
                output_json=root / "api_handoff.json",
                output_md=root / "api_handoff.md",
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["status"], "ready_for_paid_run")
            self.assertEqual(report["blockers"], [])
            self.assertEqual(report["api_budget_count"], 1)
            self.assertEqual(report["paid_api_stage_count"], 1)
            self.assertEqual(report["resume_requirements"]["missing_env"], [])
            self.assertEqual(report["resume_requirements"]["paid_run_command"], report["commands"]["paid_range_run"])
            self.assertIn("--from-stage generate_model_rubrics_rubricbench --to-stage result_card_real", report["commands"]["paid_range_run"])

            ready = check_handoff_ready(root / "api_handoff.json")
            self.assertTrue(ready["ok"])
            self.assertEqual(ready["paid_run_command"], report["commands"]["paid_range_run"])


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_text(path: Path, data: str) -> Path:
    path.write_text(data, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
