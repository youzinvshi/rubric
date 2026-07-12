from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.estimate_api_budget import (
    budget_contract,
    budget_blockers,
    count_units,
    estimate_budget,
    estimate_tokens,
    load_json_records,
    load_query_units,
    main,
    to_markdown,
)


class EstimateAPIBudgetTest(unittest.TestCase):
    def test_estimate_budget_counts_calls_and_tokens(self) -> None:
        report = estimate_budget(
            queries=["short query", "another query"],
            providers=[
                {
                    "name": "gpt",
                    "model": "gpt",
                    "max_tokens": 100,
                    "qpm": 10,
                    "tpm": 1000,
                    "input_cost_per_1k": 0.01,
                    "output_cost_per_1k": 0.02,
                }
            ],
            system_prompt="sys",
            user_template="Q: {query}",
        )
        self.assertEqual(report["total"]["calls"], 2)
        self.assertEqual(report["providers"][0]["completion_tokens"], 200)
        self.assertGreater(report["total"]["estimated_cost_usd"], 0)
        self.assertTrue(report["ok"])

    def test_budget_contract_records_cli_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "in.jsonl"
            providers_path = root / "providers.jsonl"
            resume_path = root / "out.jsonl"
            input_path.write_text('{"query": "q"}\n', encoding="utf-8")
            providers_path.write_text('{"name": "base"}\n', encoding="utf-8")
            resume_path.write_text('{"query": "q", "method": "base"}\n', encoding="utf-8")

            contract = budget_contract(
                Namespace(
                    input=input_path,
                    providers=providers_path,
                    resume_output=resume_path,
                    method_key="teacher",
                    unit_field="rubrics",
                    unit_multiplier_field="candidates",
                    limit=10,
                    calls_per_record_per_provider=2,
                )
            )
        self.assertEqual(contract["input"], str(input_path))
        self.assertEqual(len(contract["input_sha256"]), 64)
        self.assertEqual(contract["providers"], str(providers_path))
        self.assertEqual(len(contract["providers_sha256"]), 64)
        self.assertEqual(contract["resume_output"], str(resume_path))
        self.assertEqual(len(contract["resume_output_sha256"]), 64)
        self.assertEqual(contract["method_key"], "teacher")
        self.assertEqual(contract["unit_field"], "rubrics")
        self.assertEqual(contract["unit_multiplier_field"], "candidates")
        self.assertEqual(contract["limit"], 10)
        self.assertEqual(contract["calls_per_record_per_provider"], 2)

    def test_estimate_budget_subtracts_resume_done_pairs(self) -> None:
        report = estimate_budget(
            queries=["q1", "q2"],
            providers=[{"name": "gpt", "model": "gpt", "max_tokens": 10}],
            done={("q1", "gpt")},
            system_prompt="s",
            user_template="{query}",
        )
        self.assertEqual(report["providers"][0]["pending_records"], 1)
        self.assertEqual(report["total"]["calls"], 1)

    def test_estimate_budget_counts_units_for_rubric_level_api_calls(self) -> None:
        report = estimate_budget(
            queries=["q1", "q2"],
            unit_counts=[3, 2],
            providers=[{"name": "verifier", "model": "v", "max_tokens": 5}],
            system_prompt="s",
            user_template="{query}",
        )

        self.assertEqual(report["n_units"], 5)
        self.assertEqual(report["providers"][0]["pending_units"], 5)
        self.assertEqual(report["total"]["calls"], 5)

    def test_count_units_accepts_lists_and_json_strings(self) -> None:
        self.assertEqual(count_units({"rubrics": ["a", "b"]}, "rubrics"), 2)
        self.assertEqual(count_units({"rubrics": "[\"a\", \"b\", \"c\"]"}, "rubrics"), 3)
        self.assertEqual(count_units({"rubrics": "plain rubric"}, "rubrics"), 1)

    def test_estimate_budget_counts_multiplied_units(self) -> None:
        report = estimate_budget(
            queries=["q1"],
            unit_counts=[6],
            providers=[{"name": "judge", "model": "judge", "max_tokens": 5}],
            system_prompt="s",
            user_template="{query}",
        )

        self.assertEqual(report["n_units"], 6)
        self.assertEqual(report["total"]["calls"], 6)

    def test_load_query_units_multiplies_unit_fields_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joined.jsonl"
            rows = [
                {"query": "q1", "rubrics": ["r1", "r2"], "candidates": ["a", "b", "c"]},
                {"query": "q2", "rubrics": json.dumps(["r3"]), "candidates": ["a", "b"]},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            queries, counts = load_query_units(path, "rubrics", "candidates")

        self.assertEqual(queries, ["q1", "q2"])
        self.assertEqual(counts, [6, 2])

    def test_load_json_records_rejects_non_object_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text("[1]", encoding="utf-8")

            with self.assertRaises(ValueError) as context:
                load_json_records(path)

        self.assertIn("JSON record must be an object", str(context.exception))

    def test_main_writes_failed_report_for_invalid_input_json_before_strict_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "bad_input.json"
            providers = root / "providers.jsonl"
            output = root / "budget.json"
            output_md = root / "budget.md"
            input_path.write_text("{bad", encoding="utf-8")
            providers.write_text(json.dumps({"name": "gpt", "model": "gpt"}) + "\n", encoding="utf-8")

            argv = [
                "estimate_api_budget.py",
                "--input",
                str(input_path),
                "--providers",
                str(providers),
                "--output",
                str(output),
                "--output-md",
                str(output_md),
                "--strict",
            ]
            with patch("sys.argv", argv), self.assertRaises(SystemExit) as context:
                main()

            report = json.loads(output.read_text(encoding="utf-8"))
            md_text = output_md.read_text(encoding="utf-8")

        self.assertEqual(context.exception.code, 1)
        self.assertFalse(report["ok"])
        self.assertIn("budget input is not readable", report["blockers"][0])
        self.assertEqual(report["contract"]["input"], str(input_path))
        self.assertIn("budget input is not readable", md_text)

    def test_estimate_tokens_has_minimum_one(self) -> None:
        self.assertEqual(estimate_tokens(""), 1)

    def test_estimate_budget_blocks_when_limits_are_exceeded(self) -> None:
        report = estimate_budget(
            queries=["q1", "q2"],
            providers=[{"name": "gpt", "model": "gpt", "max_tokens": 100}],
            system_prompt="s",
            user_template="{query}",
            max_calls=1,
            max_total_tokens=10,
        )

        self.assertFalse(report["ok"])
        self.assertIn("API calls exceeds max_calls", report["blockers"][0])
        self.assertIn("total tokens exceeds max_total_tokens", "\n".join(report["blockers"]))

    def test_budget_blockers_accepts_unset_limits(self) -> None:
        blockers = budget_blockers({"calls": 100}, {"max_calls": None})

        self.assertEqual(blockers, [])

    def test_to_markdown_contains_provider_row(self) -> None:
        text = to_markdown(
            {
                "n_queries": 1,
                "n_providers": 1,
                "ok": False,
                "blockers": ["too expensive"],
                "limits": {
                    "max_calls": 1,
                    "max_total_tokens": 10,
                    "max_cost_usd": 0.01,
                    "max_wallclock_minutes_serial": 1,
                },
                "total": {
                    "calls": 1,
                    "total_tokens": 10,
                    "estimated_cost_usd": 0.1,
                    "min_minutes_by_rate_limits": 0.5,
                    "estimated_wallclock_minutes_serial": 0.5,
                },
                "providers": [
                    {
                        "name": "gpt",
                        "calls": 1,
                        "prompt_tokens": 2,
                        "completion_tokens": 8,
                        "total_tokens": 10,
                        "qpm": 60,
                        "tpm": 60000,
                        "min_minutes_by_rate_limits": 0.5,
                        "estimated_cost_usd": 0.1,
                    }
                ],
            }
        )
        self.assertIn("`gpt`", text)
        self.assertIn("Budget ok: `False`", text)
        self.assertIn("too expensive", text)
        self.assertIn("max_calls: `1`", text)
        self.assertIn("## Contract", text)


if __name__ == "__main__":
    unittest.main()
