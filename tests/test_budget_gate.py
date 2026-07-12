from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts import generate_model_rubrics, generate_teacher_rubrics
from scripts.budget_gate import (
    budget_contract_blockers,
    preflight_contract_blockers,
    require_budget_report,
    require_preflight_report,
)
from scripts.evaluate_downstream import enforce_api_budget_gate as enforce_pairwise_api_budget_gate
from scripts.evaluate_multicandidate_downstream import enforce_api_budget_gate as enforce_multi_api_budget_gate
from scripts.filter_rubrics_with_verifier import enforce_api_budget_gate as enforce_verifier_api_budget_gate


class BudgetGateTest(unittest.TestCase):
    def test_require_budget_report_accepts_ok_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.json"
            path.write_text(json.dumps({"ok": True, "blockers": []}), encoding="utf-8")

            self.assertTrue(require_budget_report(path)["ok"])

    def test_require_budget_report_blocks_missing_or_failed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            failed = Path(tmp) / "budget.json"
            failed.write_text(json.dumps({"ok": False, "blockers": ["too expensive"]}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                require_budget_report(missing)
            with self.assertRaises(SystemExit):
                require_budget_report(failed)

    def test_require_preflight_report_accepts_only_ok_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "preflight_ok.json"
            failed = Path(tmp) / "preflight_failed.json"
            ok.write_text(json.dumps({"ok": True, "hard_blockers": []}), encoding="utf-8")
            failed.write_text(json.dumps({"ok": False, "hard_blockers": ["missing env"]}), encoding="utf-8")

            self.assertTrue(require_preflight_report(ok)["ok"])
            with self.assertRaises(SystemExit):
                require_preflight_report(failed)

    def test_preflight_contract_blocks_stale_provider_or_input_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "current.jsonl"
            providers = root / "current_providers.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text('{"name": "base"}\n', encoding="utf-8")
            report = {
                "ok": True,
                "hard_blockers": [],
                "inputs": [{"path": str(data), "sha256": file_sha256(data)}],
                "providers": [{"path": str(providers), "sha256": file_sha256(providers)}],
            }

            self.assertEqual(
                preflight_contract_blockers(
                    report,
                    {"input": str(data), "providers": str(providers)},
                ),
                [],
            )
            blockers = preflight_contract_blockers(
                report,
                {"input": str(root / "other.jsonl"), "providers": str(root / "other_providers.jsonl")},
            )

        self.assertEqual(len(blockers), 2)
        self.assertTrue(any("input missing" in blocker for blocker in blockers))
        self.assertTrue(any("providers missing" in blocker for blocker in blockers))

    def test_require_preflight_report_enforces_expected_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight_ok.json"
            data = Path(tmp) / "current.jsonl"
            providers = Path(tmp) / "current_providers.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text('{"name": "base"}\n', encoding="utf-8")
            path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "hard_blockers": [],
                        "inputs": [{"path": str(data), "sha256": file_sha256(data)}],
                        "providers": [{"path": str(providers), "sha256": file_sha256(providers)}],
                    }
                ),
                encoding="utf-8",
            )

            require_preflight_report(
                path,
                expected_contract={"input": str(data), "providers": str(providers)},
            )
            with self.assertRaises(SystemExit):
                require_preflight_report(
                    path,
                    expected_contract={"input": str(data), "providers": str(Path(tmp) / "stale.jsonl")},
                )

    def test_require_preflight_report_blocks_when_current_file_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight_ok.json"
            data = Path(tmp) / "current.jsonl"
            providers = Path(tmp) / "current_providers.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text('{"name": "base"}\n', encoding="utf-8")
            path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "hard_blockers": [],
                        "inputs": [{"path": str(data), "sha256": file_sha256(data)}],
                        "providers": [{"path": str(providers), "sha256": file_sha256(providers)}],
                    }
                ),
                encoding="utf-8",
            )
            data.write_text('{"query": "changed"}\n', encoding="utf-8")

            with self.assertRaises(SystemExit):
                require_preflight_report(
                    path,
                    expected_contract={"input": str(data), "providers": str(providers)},
                )

    def test_preflight_contract_rechecks_current_api_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            providers = root / "providers.jsonl"
            providers.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "model",
                        "base_url": "https://api.example.com/v1",
                        "api_key_env": "TEST_PREFLIGHT_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = {
                "ok": True,
                "hard_blockers": [],
                "providers": [
                    {
                        "path": str(providers),
                        "sha256": file_sha256(providers),
                        "providers": [
                            {
                                "name": "base",
                                "api_key_env": "TEST_PREFLIGHT_KEY",
                            }
                        ],
                    }
                ],
            }

            with patch.dict(os.environ, {"TEST_PREFLIGHT_KEY": "secret"}, clear=True):
                self.assertEqual(preflight_contract_blockers(report, {"providers": str(providers)}), [])
            with patch.dict(os.environ, {}, clear=True):
                blockers = preflight_contract_blockers(report, {"providers": str(providers)})

        self.assertTrue(any("missing current API env TEST_PREFLIGHT_KEY" in blocker for blocker in blockers))

    def test_api_downstream_requires_budget_report(self) -> None:
        args = Namespace(scorer="api", require_budget_report=None)

        with self.assertRaises(SystemExit):
            enforce_pairwise_api_budget_gate(args)
        with self.assertRaises(SystemExit):
            enforce_multi_api_budget_gate(args)

    def test_keyword_downstream_does_not_require_budget_report(self) -> None:
        args = Namespace(scorer="keyword", require_budget_report=None)

        enforce_pairwise_api_budget_gate(args)
        enforce_multi_api_budget_gate(args)

    def test_api_downstream_accepts_valid_budget_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            joined = root / "joined.jsonl"
            judge = root / "judge.jsonl"
            joined.write_text('{"query": "q", "rubrics": ["r"], "candidates": ["a", "b"]}\n', encoding="utf-8")
            judge.write_text('{"name": "judge"}\n', encoding="utf-8")
            pairwise = Path(tmp) / "pairwise_budget.json"
            pairwise.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "blockers": [],
                        "contract": {
                            "input": str(joined),
                            "input_sha256": file_sha256(joined),
                            "providers": str(judge),
                            "providers_sha256": file_sha256(judge),
                            "unit_field": "rubrics",
                            "unit_multiplier_field": None,
                            "calls_per_record_per_provider": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            multi = Path(tmp) / "multi_budget.json"
            multi.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "blockers": [],
                        "contract": {
                            "input": str(joined),
                            "input_sha256": file_sha256(joined),
                            "providers": str(judge),
                            "providers_sha256": file_sha256(judge),
                            "unit_field": "rubrics",
                            "unit_multiplier_field": "candidates",
                            "calls_per_record_per_provider": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            enforce_pairwise_api_budget_gate(
                Namespace(
                    scorer="api",
                    input=joined,
                    provider=judge,
                    require_budget_report=pairwise,
                )
            )
            enforce_multi_api_budget_gate(
                Namespace(
                    scorer="api",
                    input=joined,
                    provider=judge,
                    require_budget_report=multi,
                )
            )

    def test_budget_contract_blocks_mismatched_report(self) -> None:
        report = {
            "ok": True,
            "blockers": [],
            "contract": {
                "input": "a.jsonl",
                "providers": "providers.jsonl",
                "unit_field": "rubrics",
            },
        }

        blockers = budget_contract_blockers(
            report,
            {"input": "b.jsonl", "providers": "providers.jsonl", "unit_field": "rubrics"},
        )

        self.assertTrue(any("input mismatch" in blocker for blocker in blockers))

    def test_budget_contract_blocks_when_current_file_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "input.jsonl"
            providers = root / "providers.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text('{"name": "base"}\n', encoding="utf-8")
            report = {
                "ok": True,
                "blockers": [],
                "contract": {
                    "input": str(data),
                    "input_sha256": file_sha256(data),
                    "providers": str(providers),
                    "providers_sha256": file_sha256(providers),
                    "unit_field": None,
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 1,
                },
            }
            data.write_text('{"query": "changed"}\n', encoding="utf-8")

            blockers = budget_contract_blockers(
                report,
                {
                    "input": data,
                    "providers": providers,
                    "unit_field": None,
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 1,
                },
            )

        self.assertTrue(any("input sha256 mismatch" in blocker for blocker in blockers))

    def test_budget_contract_blocks_when_resume_output_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "input.jsonl"
            providers = root / "providers.jsonl"
            resume_output = root / "out.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text('{"name": "base"}\n', encoding="utf-8")
            resume_output.write_text('{"query": "q", "method": "base"}\n', encoding="utf-8")
            report = {
                "ok": True,
                "blockers": [],
                "contract": {
                    "input": str(data),
                    "input_sha256": file_sha256(data),
                    "providers": str(providers),
                    "providers_sha256": file_sha256(providers),
                    "resume_output": str(resume_output),
                    "resume_output_sha256": file_sha256(resume_output),
                    "unit_field": None,
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 1,
                },
            }
            resume_output.write_text("", encoding="utf-8")

            blockers = budget_contract_blockers(
                report,
                {
                    "input": data,
                    "providers": providers,
                    "resume_output": resume_output,
                    "unit_field": None,
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 1,
                },
            )

        self.assertTrue(any("resume_output sha256 mismatch" in blocker for blocker in blockers))

    def test_budget_contract_blocks_when_resume_output_appears_after_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "input.jsonl"
            providers = root / "providers.jsonl"
            resume_output = root / "out.jsonl"
            data.write_text('{"query": "q"}\n', encoding="utf-8")
            providers.write_text('{"name": "base"}\n', encoding="utf-8")
            report = {
                "ok": True,
                "blockers": [],
                "contract": {
                    "input": str(data),
                    "input_sha256": file_sha256(data),
                    "providers": str(providers),
                    "providers_sha256": file_sha256(providers),
                    "resume_output": str(resume_output),
                    "resume_output_sha256": "",
                    "unit_field": None,
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 1,
                },
            }
            self.assertEqual(
                budget_contract_blockers(
                    report,
                    {
                        "input": data,
                        "providers": providers,
                        "resume_output": resume_output,
                        "unit_field": None,
                        "unit_multiplier_field": None,
                        "calls_per_record_per_provider": 1,
                    },
                ),
                [],
            )
            resume_output.write_text('{"query": "q", "method": "base"}\n', encoding="utf-8")

            blockers = budget_contract_blockers(
                report,
                {
                    "input": data,
                    "providers": providers,
                    "resume_output": resume_output,
                    "unit_field": None,
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 1,
                },
            )

        self.assertTrue(any("budget report is missing resume_output_sha256" in blocker for blocker in blockers))

    def test_budget_contract_requires_contract_when_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.json"
            path.write_text(json.dumps({"ok": True, "blockers": []}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                require_budget_report(path, expected_contract={"input": "data.jsonl"})

    def test_api_verifier_requires_budget_report(self) -> None:
        with self.assertRaises(SystemExit):
            enforce_verifier_api_budget_gate(Namespace(mode="api", require_budget_report=None))

    def test_rule_verifier_does_not_require_budget_report(self) -> None:
        enforce_verifier_api_budget_gate(Namespace(mode="rule", require_budget_report=None))

    def test_paid_generation_requires_budget_report_before_file_io(self) -> None:
        model_argv = [
            "generate_model_rubrics.py",
            "--input",
            "missing_input.jsonl",
            "--providers",
            "missing_providers.jsonl",
            "--output",
            "missing_output.jsonl",
        ]
        teacher_argv = [
            "generate_teacher_rubrics.py",
            "--input",
            "missing_input.jsonl",
            "--providers",
            "missing_providers.jsonl",
            "--output",
            "missing_output.jsonl",
        ]

        with patch.object(sys, "argv", model_argv), self.assertRaises(SystemExit):
            generate_model_rubrics.main()
        with patch.object(sys, "argv", teacher_argv), self.assertRaises(SystemExit):
            generate_teacher_rubrics.main()

    def test_paid_generation_rejects_preflight_report_for_wrong_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "queries.jsonl"
            providers = root / "providers.jsonl"
            budget = root / "budget.json"
            preflight = root / "preflight.json"
            output = root / "out.jsonl"
            data.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")
            providers.write_text(
                json.dumps(
                    {
                        "name": "base",
                        "model": "model",
                        "base_url": "http://localhost:8000/v1",
                        "api_key_env": "LOCAL_OPENAI_API_KEY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            budget.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "blockers": [],
                        "contract": {
                            "input": str(data),
                            "input_sha256": file_sha256(data),
                            "providers": str(providers),
                            "providers_sha256": file_sha256(providers),
                            "unit_field": None,
                            "unit_multiplier_field": None,
                            "calls_per_record_per_provider": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            preflight.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "hard_blockers": [],
                        "inputs": [{"path": str(data), "sha256": file_sha256(data)}],
                        "providers": [{"path": str(root / "other_providers.jsonl"), "sha256": file_sha256(providers)}],
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "generate_model_rubrics.py",
                "--input",
                str(data),
                "--providers",
                str(providers),
                "--output",
                str(output),
                "--require-budget-report",
                str(budget),
                "--require-preflight-report",
                str(preflight),
            ]

            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                generate_model_rubrics.main()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
