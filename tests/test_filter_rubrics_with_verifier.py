from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts import filter_rubrics_with_verifier


class FilterRubricsWithVerifierTest(unittest.TestCase):
    def test_annotate_only_preserves_original_rubrics_and_adds_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rubrics.jsonl"
            output_path = root / "verified.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q1",
                        "rubrics": ["must cite evidence", "good"],
                        "method": "base",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "filter_rubrics_with_verifier.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--mode",
                "rule",
                "--annotate-only",
            ]
            with patch.object(sys, "argv", argv):
                filter_rubrics_with_verifier.main()

            record = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["rubrics"], ["must cite evidence", "good"])
            self.assertEqual(record["rubrics_before_filter"], ["must cite evidence", "good"])
            self.assertEqual(record["verified_rubrics"], ["must cite evidence"])
            self.assertEqual(record["valid_flags"], [1, 0])
            self.assertEqual(record["verifier_source"], "valid_flags")

    def test_accepts_common_generated_rubric_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rubrics.jsonl"
            output_path = root / "verified.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "model_rubrics": ["must cite evidence"]}),
                        json.dumps({"query": "q2", "generated_rubrics": ["good"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "filter_rubrics_with_verifier.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--mode",
                "rule",
                "--annotate-only",
            ]
            with patch.object(sys, "argv", argv):
                filter_rubrics_with_verifier.main()

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["rubrics"], ["must cite evidence"])
        self.assertEqual(rows[0]["valid_flags"], [1])
        self.assertEqual(rows[1]["rubrics"], ["good"])
        self.assertEqual(rows[1]["valid_flags"], [0])

    def test_filter_mode_aligns_valid_flags_to_filtered_rubrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rubrics.jsonl"
            output_path = root / "verified.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q1",
                        "rubrics": ["must cite evidence", "good"],
                        "method": "base",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "filter_rubrics_with_verifier.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--mode",
                "rule",
            ]
            with patch.object(sys, "argv", argv):
                filter_rubrics_with_verifier.main()

            record = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["rubrics"], ["must cite evidence"])
        self.assertEqual(record["valid_flags"], [1])
        self.assertEqual(record["valid_flags_before_filter"], [1, 0])

    def test_report_output_records_filtering_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rubrics.jsonl"
            output_path = root / "verified.jsonl"
            stats_path = root / "stats.jsonl"
            report_path = root / "report.json"
            provider_path = root / "verifier.jsonl"
            budget_path = root / "budget.json"
            preflight_path = root / "preflight.json"
            input_path.write_text(
                json.dumps({"query": "q1", "rubrics": ["must cite evidence", "good"]}) + "\n",
                encoding="utf-8",
            )
            provider_path.write_text('{"name":"meta-verifier"}\n', encoding="utf-8")
            budget_path.write_text('{"ok":true}\n', encoding="utf-8")
            preflight_path.write_text('{"ok":true}\n', encoding="utf-8")

            argv = [
                "filter_rubrics_with_verifier.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--stats-output",
                str(stats_path),
                "--report-output",
                str(report_path),
                "--mode",
                "rule",
                "--provider",
                str(provider_path),
                "--require-budget-report",
                str(budget_path),
                "--require-preflight-report",
                str(preflight_path),
            ]
            with patch.object(sys, "argv", argv):
                filter_rubrics_with_verifier.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input"], str(input_path))
            self.assertEqual(report["input_sha256"], file_sha256(input_path))
            self.assertEqual(report["output"], str(output_path))
            self.assertEqual(report["output_sha256"], file_sha256(output_path))
            self.assertEqual(report["stats_output"], str(stats_path))
            self.assertEqual(report["stats_output_sha256"], file_sha256(stats_path))
            self.assertEqual(report["provider"], str(provider_path))
            self.assertEqual(report["provider_sha256"], file_sha256(provider_path))
            self.assertEqual(report["budget_report"], str(budget_path))
            self.assertEqual(report["budget_report_sha256"], file_sha256(budget_path))
            self.assertEqual(report["preflight_report"], str(preflight_path))
            self.assertEqual(report["preflight_report_sha256"], file_sha256(preflight_path))
            self.assertEqual(report["n_input_records"], 1)
            self.assertEqual(report["n_output_records"], 1)
            self.assertEqual(report["n_input_rubrics"], 2)
            self.assertEqual(report["n_valid_rubrics"], 1)
            self.assertEqual(report["mode"], "rule")


if __name__ == "__main__":
    unittest.main()
