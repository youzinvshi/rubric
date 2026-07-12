from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts import evaluate_downstream as pairwise_module
from scripts import evaluate_multicandidate_downstream as multicandidate_module
from scripts.evaluate_downstream import main as pairwise_main, paper_claim_eligibility
from scripts.evaluate_multicandidate_downstream import main as multicandidate_main


class EvaluateDownstreamScriptTest(unittest.TestCase):
    def test_downstream_public_descriptions_use_criteria_framing(self) -> None:
        pairwise_public_text = "\n".join(
            [
                pairwise_module.__doc__ or "",
                inspect.getsource(pairwise_module.parse_args),
                inspect.getsource(pairwise_module.print_summary),
            ]
        )
        multicandidate_public_text = "\n".join(
            [
                multicandidate_module.__doc__ or "",
                inspect.getsource(multicandidate_module.parse_args),
                inspect.getsource(multicandidate_module.print_summary),
            ]
        )
        combined = pairwise_public_text + "\n" + multicandidate_public_text

        self.assertIn("criterion-guided utility", combined)
        self.assertIn("generated evaluation criteria", combined)
        self.assertIn("Criterion-guided downstream preference summary", combined)
        self.assertIn("Criterion-guided multi-candidate downstream summary", combined)
        self.assertNotIn("rubric usefulness", combined)
        self.assertNotIn("using rubrics", combined)

    def test_paper_claim_eligibility_requires_current_budget_contract_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "joined.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            input_path.write_text('{"query": "q"}\n', encoding="utf-8")
            provider.write_text('{"name": "judge"}\n', encoding="utf-8")
            args = Namespace(
                scorer="api",
                input=input_path,
                provider=provider,
                require_budget_report=budget,
            )
            budget_report = {
                "ok": True,
                "contract": {
                    "input": str(input_path),
                    "input_sha256": "0" * 64,
                    "providers": str(provider),
                    "providers_sha256": file_sha256(provider),
                    "unit_field": "rubrics",
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 2,
                },
            }

            eligible, blockers = paper_claim_eligibility(
                args,
                budget_report,
                expected_contract={
                    "input": input_path,
                    "providers": provider,
                    "unit_field": "rubrics",
                    "unit_multiplier_field": None,
                    "calls_per_record_per_provider": 2,
                },
            )

        self.assertFalse(eligible)
        self.assertTrue(any("input sha256 mismatch" in blocker for blocker in blockers))

    def test_pairwise_summary_records_scorer_provider_and_budget_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "pairwise.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            output_dir = root / "out"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "chosen": "uses evidence",
                        "rejected": "ignores source",
                        "rubrics": ["uses evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider.write_text('{"name": "judge"}\n', encoding="utf-8")
            budget.write_text('{"ok": true}\n', encoding="utf-8")
            expected_input_sha = file_sha256(input_path)
            expected_provider_sha = file_sha256(provider)

            with patch(
                "sys.argv",
                [
                    "evaluate_downstream.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--scorer",
                    "keyword",
                    "--provider",
                    str(provider),
                    "--require-budget-report",
                    str(budget),
                ],
            ):
                pairwise_main()

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["scorer"], "keyword")
        self.assertEqual(summary["scorer_provider"], str(provider))
        self.assertEqual(summary["budget_report"], str(budget))
        self.assertFalse(summary["paper_claim_eligible"])
        self.assertIn("downstream scorer is not api", summary["paper_claim_eligibility_blockers"])
        self.assertIn("budget report was not loaded", summary["paper_claim_eligibility_blockers"])
        self.assertEqual(summary["input"], str(input_path))
        self.assertEqual(summary["input_sha256"], expected_input_sha)
        self.assertEqual(summary["rubrics_input"], "")
        self.assertEqual(summary["scorer_provider_sha256"], expected_provider_sha)
        self.assertEqual(summary["budget_contract"], {})
        self.assertEqual(summary["benchmark_format"], "pairwise")
        self.assertEqual(
            summary["scorer_contract"],
            {
                "unit_field": "rubrics",
                "unit_multiplier_field": None,
                "calls_per_record_per_provider": 2,
            },
        )

    def test_multicandidate_summary_records_scorer_provider_and_budget_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "multi.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            output_dir = root / "out"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "candidates": ["ignores source", "uses evidence"],
                        "label": 1,
                        "rubrics": ["uses evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider.write_text('{"name": "judge"}\n', encoding="utf-8")
            budget.write_text('{"ok": true}\n', encoding="utf-8")
            expected_input_sha = file_sha256(input_path)
            expected_provider_sha = file_sha256(provider)

            with patch(
                "sys.argv",
                [
                    "evaluate_multicandidate_downstream.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--scorer",
                    "keyword",
                    "--provider",
                    str(provider),
                    "--require-budget-report",
                    str(budget),
                ],
            ):
                multicandidate_main()

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["scorer"], "keyword")
        self.assertEqual(summary["scorer_provider"], str(provider))
        self.assertEqual(summary["budget_report"], str(budget))
        self.assertFalse(summary["paper_claim_eligible"])
        self.assertIn("downstream scorer is not api", summary["paper_claim_eligibility_blockers"])
        self.assertIn("budget report was not loaded", summary["paper_claim_eligibility_blockers"])
        self.assertEqual(summary["input"], str(input_path))
        self.assertEqual(summary["input_sha256"], expected_input_sha)
        self.assertEqual(summary["rubrics_input"], "")
        self.assertEqual(summary["scorer_provider_sha256"], expected_provider_sha)
        self.assertEqual(summary["budget_contract"], {})
        self.assertEqual(summary["benchmark_format"], "multicandidate")
        self.assertEqual(
            summary["scorer_contract"],
            {
                "unit_field": "rubrics",
                "unit_multiplier_field": "candidates",
                "calls_per_record_per_provider": 1,
            },
        )

    def test_pairwise_api_summary_marks_paper_claim_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "pairwise.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            output_dir = root / "out"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "chosen": "uses evidence",
                        "rejected": "ignores source",
                        "rubrics": ["uses evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider.write_text('{"name": "judge", "api_key_env": "X"}\n', encoding="utf-8")
            budget.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "contract": {
                            "input": str(input_path),
                            "input_sha256": file_sha256(input_path),
                            "providers": str(provider),
                            "providers_sha256": file_sha256(provider),
                            "unit_field": "rubrics",
                            "unit_multiplier_field": None,
                            "calls_per_record_per_provider": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            expected_input_sha = file_sha256(input_path)
            expected_provider_sha = file_sha256(provider)

            class FakeScorer:
                def score(self, query: str, answer: str, rubric: str) -> float:
                    del query, rubric
                    return 1.0 if "uses evidence" in answer else 0.0

            with patch("scripts.evaluate_downstream.build_scorer", return_value=FakeScorer()):
                with patch(
                    "sys.argv",
                    [
                        "evaluate_downstream.py",
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(output_dir),
                        "--scorer",
                        "api",
                        "--provider",
                        str(provider),
                        "--require-budget-report",
                        str(budget),
                    ],
                ):
                    pairwise_main()

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            per_item = output_dir / "per_item.csv"
            expected_per_item_sha = file_sha256(per_item)

        self.assertEqual(summary["scorer"], "api")
        self.assertTrue(summary["paper_claim_eligible"])
        self.assertEqual(summary["paper_claim_eligibility_blockers"], [])
        self.assertEqual(summary["input_sha256"], expected_input_sha)
        self.assertEqual(summary["per_item_output"], str(per_item))
        self.assertEqual(summary["per_item_sha256"], expected_per_item_sha)
        self.assertEqual(summary["per_item_rows"], 1)
        self.assertEqual(summary["scorer_provider_sha256"], expected_provider_sha)
        self.assertEqual(summary["budget_contract"]["input_sha256"], expected_input_sha)
        self.assertEqual(summary["budget_contract"]["providers_sha256"], expected_provider_sha)

    def test_multicandidate_api_summary_marks_paper_claim_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "multi.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            output_dir = root / "out"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "candidates": ["ignores source", "uses evidence"],
                        "label": 1,
                        "rubrics": ["uses evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider.write_text('{"name": "judge", "api_key_env": "X"}\n', encoding="utf-8")
            budget.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "contract": {
                            "input": str(input_path),
                            "input_sha256": file_sha256(input_path),
                            "providers": str(provider),
                            "providers_sha256": file_sha256(provider),
                            "unit_field": "rubrics",
                            "unit_multiplier_field": "candidates",
                            "calls_per_record_per_provider": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            expected_input_sha = file_sha256(input_path)
            expected_provider_sha = file_sha256(provider)

            class FakeScorer:
                def score(self, query: str, answer: str, rubric: str) -> float:
                    del query, rubric
                    return 1.0 if "uses evidence" in answer else 0.0

            with patch("scripts.evaluate_multicandidate_downstream.build_scorer", return_value=FakeScorer()):
                with patch(
                    "sys.argv",
                    [
                        "evaluate_multicandidate_downstream.py",
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(output_dir),
                        "--scorer",
                        "api",
                        "--provider",
                        str(provider),
                        "--require-budget-report",
                        str(budget),
                    ],
                ):
                    multicandidate_main()

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            per_item = output_dir / "per_item.csv"
            expected_per_item_sha = file_sha256(per_item)

        self.assertEqual(summary["scorer"], "api")
        self.assertTrue(summary["paper_claim_eligible"])
        self.assertEqual(summary["paper_claim_eligibility_blockers"], [])
        self.assertEqual(summary["input_sha256"], expected_input_sha)
        self.assertEqual(summary["per_item_output"], str(per_item))
        self.assertEqual(summary["per_item_sha256"], expected_per_item_sha)
        self.assertEqual(summary["per_item_rows"], 1)
        self.assertEqual(summary["scorer_provider_sha256"], expected_provider_sha)
        self.assertEqual(summary["budget_contract"]["input_sha256"], expected_input_sha)
        self.assertEqual(summary["budget_contract"]["providers_sha256"], expected_provider_sha)

    def test_pairwise_api_budget_contract_mismatch_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "pairwise.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            output_dir = root / "out"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "chosen": "uses evidence",
                        "rejected": "ignores source",
                        "rubrics": ["uses evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider.write_text('{"name": "judge", "api_key_env": "X"}\n', encoding="utf-8")
            budget.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "contract": {
                            "input": str(input_path),
                            "input_sha256": file_sha256(input_path),
                            "providers": str(provider),
                            "providers_sha256": file_sha256(provider),
                            "unit_field": "rubrics",
                            "unit_multiplier_field": "candidates",
                            "calls_per_record_per_provider": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "evaluate_downstream.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--scorer",
                    "api",
                    "--provider",
                    str(provider),
                    "--require-budget-report",
                    str(budget),
                ],
            ):
                with self.assertRaisesRegex(SystemExit, "contract mismatch"):
                    pairwise_main()

    def test_multicandidate_api_budget_contract_mismatch_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "multi.jsonl"
            provider = root / "judge.jsonl"
            budget = root / "budget.json"
            output_dir = root / "out"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "candidates": ["ignores source", "uses evidence"],
                        "label": 1,
                        "rubrics": ["uses evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider.write_text('{"name": "judge", "api_key_env": "X"}\n', encoding="utf-8")
            budget.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "contract": {
                            "input": str(input_path),
                            "input_sha256": file_sha256(input_path),
                            "providers": str(provider),
                            "providers_sha256": file_sha256(provider),
                            "unit_field": "rubrics",
                            "unit_multiplier_field": None,
                            "calls_per_record_per_provider": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "evaluate_multicandidate_downstream.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--scorer",
                    "api",
                    "--provider",
                    str(provider),
                    "--require-budget-report",
                    str(budget),
                ],
            ):
                with self.assertRaisesRegex(SystemExit, "contract mismatch"):
                    multicandidate_main()


if __name__ == "__main__":
    unittest.main()
