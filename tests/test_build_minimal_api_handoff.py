from __future__ import annotations

import json
import tempfile
import unittest
import hashlib
from pathlib import Path

from scripts.build_minimal_api_handoff import build_handoff, load_blockers


class BuildMinimalApiHandoffTest(unittest.TestCase):
    def test_load_blockers_deduplicates_hard_blockers_alias(self) -> None:
        blockers = load_blockers(
            "preflight",
            {
                "ok": False,
                "hard_blockers": ["missing env"],
                "blockers": ["missing env"],
            },
        )

        self.assertEqual(blockers, ["preflight: ok=false", "preflight: missing env"])

    def test_handoff_blocks_when_preflight_is_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generator_providers = write_text(root / "generator_providers.jsonl", '{"name": "base"}\n')
            verifier_providers = write_text(root / "verifier_providers.jsonl", '{"name": "verifier"}\n')
            pipeline = write_json(
                root / "pipeline.json",
                {
                    "stages": [
                        {"name": "validate_rubricbench_gold", "type": "validate_gold"},
                        {"name": "sample_queries", "type": "sample_records"},
                        {"name": "preflight", "type": "preflight"},
                        {"name": "api_budget", "type": "api_budget"},
                        {"name": "minimal_api_handoff", "type": "minimal_api_handoff"},
                        {
                            "name": "generate_model_rubrics",
                            "args": {
                                "providers": str(generator_providers),
                                "output": str(root / "model_rubrics.jsonl"),
                            },
                        },
                        {
                            "name": "verifier_api_budget",
                            "type": "api_budget",
                            "args": {
                                "input": str(root / "model_rubrics.jsonl"),
                                "output": str(root / "verifier_budget.json"),
                            },
                        },
                        {"name": "result_card"},
                    ]
                },
            )
            preflight = write_json(
                root / "preflight.json",
                {
                    "ok": False,
                    "hard_blockers": [
                        "missing required env var: LOCAL_OPENAI_API_KEY",
                        "missing required env var: OPENAI_API_KEY",
                    ],
                    "env": [
                        {"name": "LOCAL_OPENAI_API_KEY", "present": False, "length": 0},
                        {"name": "OPENAI_API_KEY", "present": False, "length": 0},
                    ],
                    "providers": [
                        {
                            "path": str(generator_providers),
                            "present": True,
                            "sha256": sha256(generator_providers),
                            "providers": [
                                {
                                    "name": "base",
                                    "base_url": "http://localhost:8000/v1",
                                    "api_key_env": "LOCAL_OPENAI_API_KEY",
                                    "api_key_present": False,
                                    "local_health": {"checked": True, "ok": True},
                                }
                            ],
                        },
                        {
                            "path": str(verifier_providers),
                            "present": True,
                            "sha256": sha256(verifier_providers),
                            "providers": [
                                {
                                    "name": "verifier",
                                    "base_url": "https://api.openai.com/v1",
                                    "api_key_env": "OPENAI_API_KEY",
                                    "api_key_present": False,
                                    "local_health": {"checked": False, "ok": None},
                                }
                            ],
                        },
                    ],
                },
            )
            budget = write_json(
                root / "budget.json",
                {
                    "ok": True,
                    "total": {"calls": 100, "tokens": 500},
                    "contract": {"providers": str(generator_providers)},
                },
            )
            sanity = write_json(root / "sanity.json", {"ok": True, "n_joined": 1147, "mean_coverage": 1.0})

            report = build_handoff(
                pipeline_path=pipeline,
                preflight_path=preflight,
                api_budget_path=budget,
                bsc_gold_sanity_path=sanity,
                output_json=root / "handoff.json",
                output_md=root / "handoff.md",
            )

            self.assertFalse(report["ok"])
            self.assertEqual(report["status"], "blocked")
            self.assertIn("preflight: ok=false", report["blockers"])
            self.assertTrue(any("LOCAL_OPENAI_API_KEY" in item for item in report["blockers"]))
            self.assertTrue(report["commands"]["paid_range_run"].startswith("python3 "))
            self.assertIn("--from-stage generate_model_rubrics --to-stage result_card", report["commands"]["paid_range_run"])
            self.assertIn("--require-ready-handoff", report["commands"]["paid_range_run"])
            self.assertIn("--require-ready-handoff", report["commands"]["paid_range_dry_run"])
            self.assertIn("--dry-run", report["commands"]["paid_range_dry_run"])
            self.assertIn("check_minimal_api_handoff_ready.py", report["commands"]["handoff_ready_check"])
            self.assertIn("--handoff", report["commands"]["handoff_ready_check"])
            self.assertIn("--only validate_rubricbench_gold", report["commands"]["rerun_offline_gates"])
            self.assertIn("--only minimal_api_handoff", report["commands"]["rerun_offline_gates"])
            self.assertNotIn("--only generate_model_rubrics", report["commands"]["rerun_offline_gates"])
            self.assertIn(
                "--from-stage audit --to-stage paper_asset_index_check_post_sync",
                report["commands"]["refresh_blocked_reports"],
            )
            self.assertFalse(report["resume_requirements"]["ready"])
            self.assertEqual(report["resume_requirements"]["missing_env"], ["LOCAL_OPENAI_API_KEY", "OPENAI_API_KEY"])
            self.assertEqual(
                report["resume_requirements"]["env_export_templates"],
                [
                    "export LOCAL_OPENAI_API_KEY=<set-local-openai-api-key>",
                    "export OPENAI_API_KEY=<set-openai-api-key>",
                ],
            )
            self.assertFalse(report["resume_requirements"]["partial_paid_run_allowed"])
            plan = {item["stage"]: item for item in report["resume_requirements"]["paid_stage_plan"]}
            self.assertEqual(plan["generate_model_rubrics"]["missing_env"], ["LOCAL_OPENAI_API_KEY"])
            self.assertEqual(plan["verifier_api_budget"]["status"], "blocked_until_model_rubrics_exist")
            self.assertEqual(plan["verifier_api_budget"]["input"], str(root / "model_rubrics.jsonl"))
            self.assertEqual(plan["verify_model_rubrics"]["missing_env"], ["OPENAI_API_KEY"])
            self.assertIn("verifier_api_budget", [stage["name"] for stage in report["paid_range_stages"]])
            self.assertEqual(
                report["resume_requirements"]["next_command"],
                report["commands"]["rerun_offline_gates"],
            )
            self.assertEqual(
                report["resume_requirements"]["blocked_report_refresh_command"],
                report["commands"]["refresh_blocked_reports"],
            )
            self.assertIsNone(report["resume_requirements"]["paid_run_command"])
            self.assertTrue((root / "handoff.json").exists())
            self.assertTrue((root / "handoff.md").exists())
            markdown = (root / "handoff.md").read_text(encoding="utf-8")
            self.assertIn("Partial paid run allowed: `false`", markdown)
            self.assertIn("### Env Export Template", markdown)
            self.assertIn("export LOCAL_OPENAI_API_KEY=<set-local-openai-api-key>", markdown)
            self.assertIn("export OPENAI_API_KEY=<set-openai-api-key>", markdown)
            self.assertIn("`generate_model_rubrics` status=`missing_env`", markdown)
            self.assertIn("`verifier_api_budget` status=`blocked_until_model_rubrics_exist`", markdown)
            self.assertIn("`verify_model_rubrics` status=`missing_env`", markdown)
            self.assertIn("### refresh_blocked_reports", markdown)

    def test_handoff_is_ready_when_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = write_text(root / "queries.jsonl", '{"query": "q"}\n')
            providers = write_text(root / "providers.jsonl", '{"name": "base"}\n')
            pipeline = write_json(
                root / "pipeline.json",
                {
                    "stages": [
                        {"name": "preflight"},
                        {
                            "name": "generate_model_rubrics",
                            "args": {
                                "input": str(queries),
                                "providers": str(providers),
                                "output": str(root / "model_rubrics.jsonl"),
                            },
                        },
                        {"name": "result_card"},
                    ]
                },
            )
            preflight = write_json(
                root / "preflight.json",
                {
                    "ok": True,
                    "hard_blockers": [],
                    "inputs": [{"path": str(queries), "present": True, "sha256": sha256(queries), "records": 1}],
                    "providers": [
                        {
                            "path": str(providers),
                            "present": True,
                            "sha256": sha256(providers),
                            "records": 1,
                            "providers": [
                                {
                                    "name": "base",
                                    "model": "qwen",
                                    "base_url": "http://localhost:8000/v1",
                                    "api_key_env": "TEST_KEY",
                                    "api_key_present": True,
                                    "base_url_valid": True,
                                    "local_health": {
                                        "checked": True,
                                        "is_local": True,
                                        "ok": True,
                                        "host": "localhost",
                                        "port": 8000,
                                        "error": "",
                                    },
                                }
                            ],
                        }
                    ],
                    "env": [{"name": "OPENAI_API_KEY", "present": True, "length": 8}],
                },
            )
            budget = write_json(
                root / "budget.json",
                {
                    "ok": True,
                    "blockers": [],
                    "total": {"calls": 2, "total_tokens": 100, "estimated_cost_usd": 0.01},
                    "contract": {
                        "input": str(queries),
                        "input_sha256": sha256(queries),
                        "providers": str(providers),
                        "providers_sha256": sha256(providers),
                        "resume_output": str(root / "model_rubrics.jsonl"),
                        "resume_output_sha256": "",
                        "method_key": "method",
                        "calls_per_record_per_provider": 1,
                    },
                },
            )
            sanity = write_json(root / "sanity.json", {"ok": True, "blockers": [], "n_joined": 2, "mean_coverage": 1.0})

            report = build_handoff(
                pipeline_path=pipeline,
                preflight_path=preflight,
                api_budget_path=budget,
                bsc_gold_sanity_path=sanity,
                output_md=root / "handoff.md",
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["status"], "ready_for_paid_run")
            self.assertEqual(report["blockers"], [])
            self.assertEqual(report["stage_range"]["start_index"], 1)
            self.assertEqual(report["stage_range"]["end_index"], 2)
            self.assertEqual(report["api_budget"]["total_tokens"], 100)
            self.assertEqual(report["api_budget"]["estimated_cost_usd"], 0.01)
            self.assertTrue(report["resume_requirements"]["ready"])
            self.assertEqual(report["resume_requirements"]["missing_env"], [])
            self.assertEqual(report["resume_requirements"]["env_export_templates"], [])
            self.assertEqual(
                report["resume_requirements"]["next_command"],
                report["commands"]["paid_range_run"],
            )
            self.assertEqual(
                report["resume_requirements"]["paid_run_command"],
                report["commands"]["paid_range_run"],
            )
            provider_summary = report["preflight"]["providers"][0]["providers"][0]
            self.assertEqual(provider_summary["name"], "base")
            self.assertTrue(provider_summary["local_health"]["checked"])
            self.assertTrue(provider_summary["local_health"]["ok"])

            markdown = (root / "handoff.md").read_text(encoding="utf-8")
            self.assertIn("## Provider Preflight", markdown)
            self.assertIn("http://localhost:8000/v1", markdown)
            self.assertIn("TEST_KEY", markdown)
            self.assertIn("ok localhost:8000", markdown)
            self.assertIn("## Resume Requirements", markdown)
            self.assertIn("Missing env: `none`", markdown)
            self.assertIn("### Next Command", markdown)

    def test_handoff_blocks_when_budget_input_sha_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = write_text(root / "queries.jsonl", '{"query": "old"}\n')
            old_sha = sha256(queries)
            providers = write_text(root / "providers.jsonl", '{"name": "base"}\n')
            pipeline = write_json(
                root / "pipeline.json",
                {
                    "stages": [
                        {
                            "name": "generate_model_rubrics",
                            "args": {
                                "input": str(queries),
                                "providers": str(providers),
                                "output": str(root / "model_rubrics.jsonl"),
                            },
                        },
                        {"name": "result_card"},
                    ]
                },
            )
            write_text(queries, '{"query": "new"}\n')
            preflight = write_json(
                root / "preflight.json",
                {
                    "ok": True,
                    "hard_blockers": [],
                    "inputs": [{"path": str(queries), "present": True, "sha256": old_sha, "records": 1}],
                    "providers": [{"path": str(providers), "present": True, "sha256": sha256(providers), "records": 1}],
                },
            )
            budget = write_json(
                root / "budget.json",
                {
                    "ok": True,
                    "blockers": [],
                    "total": {"calls": 1, "total_tokens": 10},
                    "contract": {
                        "input": str(queries),
                        "input_sha256": old_sha,
                        "providers": str(providers),
                        "providers_sha256": sha256(providers),
                        "resume_output": str(root / "model_rubrics.jsonl"),
                        "resume_output_sha256": "",
                        "method_key": "method",
                        "calls_per_record_per_provider": 1,
                    },
                },
            )
            sanity = write_json(root / "sanity.json", {"ok": True, "blockers": [], "n_joined": 1})

            report = build_handoff(pipeline, preflight, budget, sanity)

            self.assertFalse(report["ok"])
            self.assertTrue(any("preflight input sha256 changed" in item for item in report["blockers"]))
            self.assertTrue(any("api_budget contract input sha256" in item for item in report["blockers"]))

    def test_handoff_blocks_when_budget_contract_does_not_match_generation_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = write_text(root / "queries.jsonl", '{"query": "q"}\n')
            other_queries = write_text(root / "other_queries.jsonl", '{"query": "q"}\n')
            providers = write_text(root / "providers.jsonl", '{"name": "base"}\n')
            pipeline = write_json(
                root / "pipeline.json",
                {
                    "stages": [
                        {
                            "name": "generate_model_rubrics",
                            "args": {
                                "input": str(queries),
                                "providers": str(providers),
                                "output": str(root / "model_rubrics.jsonl"),
                            },
                        },
                        {"name": "result_card"},
                    ]
                },
            )
            preflight = write_json(root / "preflight.json", {"ok": True, "hard_blockers": []})
            budget = write_json(
                root / "budget.json",
                {
                    "ok": True,
                    "blockers": [],
                    "total": {"calls": 1, "total_tokens": 10},
                    "contract": {
                        "input": str(other_queries),
                        "input_sha256": sha256(other_queries),
                        "providers": str(providers),
                        "providers_sha256": sha256(providers),
                        "resume_output": str(root / "model_rubrics.jsonl"),
                        "resume_output_sha256": "",
                        "method_key": "method",
                        "calls_per_record_per_provider": 1,
                    },
                },
            )
            sanity = write_json(root / "sanity.json", {"ok": True, "blockers": [], "n_joined": 1})

            report = build_handoff(pipeline, preflight, budget, sanity)

            self.assertFalse(report["ok"])
            self.assertTrue(any("contract input does not match" in item for item in report["blockers"]))


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
