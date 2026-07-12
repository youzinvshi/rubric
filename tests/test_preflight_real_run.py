from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.preflight_real_run import (
    build_preflight_report,
    check_training_config,
    check_input_file,
    check_provider_file,
    check_required_providers_in_file,
    parse_required_provider_in_spec,
    check_required_providers,
    to_markdown,
)


class PreflightRealRunTest(unittest.TestCase):
    def test_check_input_file_counts_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
            check = check_input_file(path, min_records=2)
        self.assertTrue(check["present"])
        self.assertEqual(check["records"], 2)
        self.assertEqual(len(check["sha256"]), 64)

    def test_check_input_file_reports_invalid_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text("{bad", encoding="utf-8")

            report = build_preflight_report([path], [], min_records=1)

        self.assertFalse(report["ok"])
        self.assertIn("input file is not readable", report["hard_blockers"][0])
        self.assertIn("not valid JSON", report["inputs"][0]["read_error"])

    def test_check_provider_file_reports_missing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "qwen",
                        "base_url": "http://localhost:8000/v1",
                        "api_key_env": "MISSING_TEST_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                report = check_provider_file(path)
        self.assertFalse(report["providers"][0]["api_key_present"])
        self.assertIn("missing API env MISSING_TEST_KEY", report["hard_blockers"][0])

    def test_check_provider_file_reports_invalid_jsonl_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.jsonl"
            path.write_text("{bad\n[]\n", encoding="utf-8")

            report = check_provider_file(path)

        self.assertEqual(report["providers"], [])
        self.assertIn("invalid provider JSONL", report["hard_blockers"][0])
        self.assertIn("provider config has no records", report["hard_blockers"][-1])

    def test_check_training_config_reports_invalid_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.json"
            path.write_text("{bad", encoding="utf-8")

            report = check_training_config(path)

        self.assertFalse(report["hard_blockers"] == [])
        self.assertIn("training config is not readable", report["hard_blockers"][0])

    def test_check_training_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.json"
            path.write_text("[]", encoding="utf-8")

            report = check_training_config(path)

        self.assertIn("training config must be a JSON object", report["hard_blockers"][0])

    def test_build_preflight_report_passes_with_env_and_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            providers = root / "providers.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "qwen",
                        "base_url": "http://localhost:8000/v1",
                        "api_key_env": "TEST_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_KEY": "secret"}, clear=True):
                report = build_preflight_report([data], [providers])
        self.assertTrue(report["ok"])
        self.assertEqual(report["hard_blockers"], [])
        self.assertEqual(report["blockers"], report["hard_blockers"])
        self.assertEqual(len(report["inputs"][0]["sha256"]), 64)
        self.assertEqual(len(report["providers"][0]["sha256"]), 64)

    def test_build_preflight_report_exposes_blockers_alias(self) -> None:
        report = build_preflight_report([], [], required_env=["MISSING_TEST_KEY"])

        self.assertFalse(report["ok"])
        self.assertEqual(report["blockers"], report["hard_blockers"])
        self.assertIn("missing required env var: MISSING_TEST_KEY", report["blockers"])

    def test_local_provider_health_is_optional_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            providers = Path(tmp) / "providers.jsonl"
            providers.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "qwen",
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key_env": "TEST_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TEST_KEY": "secret"}, clear=True):
                report = build_preflight_report([], [providers])

        self.assertTrue(report["ok"])
        self.assertFalse(report["providers"][0]["providers"][0]["local_health"]["checked"])

    def test_local_provider_health_blocks_unreachable_loopback_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            providers = Path(tmp) / "providers.jsonl"
            providers.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "qwen",
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key_env": "TEST_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TEST_KEY": "secret"}, clear=True):
                report = build_preflight_report(
                    [],
                    [providers],
                    check_local_provider_health=True,
                )

        self.assertFalse(report["ok"])
        provider = report["providers"][0]["providers"][0]
        self.assertTrue(provider["local_health"]["checked"])
        self.assertFalse(provider["local_health"]["ok"])
        self.assertIn("local provider endpoint is not reachable", report["hard_blockers"][0])

    def test_local_provider_health_skips_external_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            providers = Path(tmp) / "providers.jsonl"
            providers.write_text(
                json.dumps(
                    {
                        "name": "meta-verifier",
                        "model": "gpt-4o-mini",
                        "base_url": "https://api.openai.com/v1",
                        "api_key_env": "TEST_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"TEST_KEY": "secret"}, clear=True):
                report = build_preflight_report(
                    [],
                    [providers],
                    check_local_provider_health=True,
                )

        self.assertTrue(report["ok"])
        self.assertFalse(report["providers"][0]["providers"][0]["local_health"]["checked"])

    def test_required_provider_names_block_missing_generators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "qwen",
                        "base_url": "http://localhost:8000/v1",
                        "api_key_env": "TEST_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_KEY": "secret"}, clear=True):
                report = build_preflight_report(
                    [],
                    [path],
                    required_provider=["base", "sft_only", "sft_rl"],
                )

        self.assertFalse(report["ok"])
        self.assertEqual(report["provider_names"]["missing"], ["sft_only", "sft_rl"])
        self.assertIn("missing required provider: sft_rl", report["hard_blockers"])

    def test_check_required_providers_collects_names_across_files(self) -> None:
        check = check_required_providers(
            [
                {"providers": [{"name": "base"}]},
                {"providers": [{"name": "sft_rl"}]},
            ],
            ["base", "sft_only", "sft_rl"],
        )

        self.assertEqual(check["present"], ["base", "sft_rl"])
        self.assertEqual(check["missing"], ["sft_only"])

    def test_required_provider_in_blocks_missing_name_in_specific_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generators = root / "generators.jsonl"
            teachers = root / "providers.jsonl"
            generators.write_text(
                json.dumps({"name": "gpt-4o", "model": "m", "base_url": "https://api.example.com/v1", "api_key_env": "KEY"})
                + "\n",
                encoding="utf-8",
            )
            teachers.write_text(
                json.dumps({"name": "deepseek", "model": "m", "base_url": "https://api.example.com/v1", "api_key_env": "KEY"})
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"KEY": "secret"}, clear=True):
                report = build_preflight_report(
                    [],
                    [generators, teachers],
                    required_provider_in=[f"{teachers}:gpt-4o,deepseek,qwen"],
                )

        self.assertFalse(report["ok"])
        self.assertEqual(report["provider_file_requirements"][0]["missing"], ["gpt-4o", "qwen"])
        self.assertIn(f"missing required provider in {teachers}: qwen", report["hard_blockers"])

    def test_check_required_providers_in_file_uses_exact_file_path(self) -> None:
        check = check_required_providers_in_file(
            [{"path": "configs/providers.local.jsonl", "providers": [{"name": "gpt-4o"}]}],
            "configs/providers.local.jsonl:gpt-4o,deepseek",
        )

        self.assertEqual(check["present"], ["gpt-4o"])
        self.assertEqual(check["missing"], ["deepseek"])

    def test_parse_required_provider_in_spec(self) -> None:
        path, names = parse_required_provider_in_spec("configs/providers.local.jsonl:gpt-4o,deepseek")

        self.assertEqual(path, "configs/providers.local.jsonl")
        self.assertEqual(names, ["gpt-4o", "deepseek"])

    def test_invalid_required_provider_in_spec_becomes_preflight_blocker(self) -> None:
        report = build_preflight_report([], [], required_provider_in=["bad-spec"])

        self.assertFalse(report["ok"])
        self.assertIn("Invalid required-provider-in spec", report["hard_blockers"][0])

    def test_to_markdown_includes_blockers(self) -> None:
        text = to_markdown(
            {
                "ok": False,
                "hard_blockers": ["missing input"],
                "warnings": [],
                "inputs": [],
                "providers": [],
                "provider_names": {"required": ["base"], "present": [], "missing": ["base"]},
                "provider_file_requirements": [
                    {
                        "path": "configs/providers.local.jsonl",
                        "required": ["gpt-4o"],
                        "present": [],
                        "missing": ["gpt-4o"],
                    }
                ],
            }
        )
        self.assertIn("missing input", text)
        self.assertIn("Missing names", text)
        self.assertIn("SHA256", text)
        self.assertIn("Required names in `configs/providers.local.jsonl`", text)


if __name__ == "__main__":
    unittest.main()
