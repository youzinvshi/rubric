from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts.filter_holdout_contamination import filter_holdout_contamination, main


class FilterHoldoutContaminationTest(unittest.TestCase):
    def test_filter_removes_training_records_overlapping_holdout_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "train.jsonl"
            output = root / "train.clean.jsonl"
            holdout.write_text(json.dumps({"query": " Evaluate   Answer A "}) + "\n", encoding="utf-8")
            train.write_text(
                json.dumps({"query": "evaluate answer a", "id": "drop"}) + "\n"
                + json.dumps({"query": "evaluate answer b", "id": "keep"}) + "\n",
                encoding="utf-8",
            )

            report = filter_holdout_contamination(holdout=holdout, input_path=train, output=output)
            kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            expected_holdout_sha256 = file_sha256(holdout)
            expected_input_sha256 = file_sha256(train)
            expected_output_sha256 = file_sha256(output)

        self.assertTrue(report["ok"])
        self.assertEqual(report["input_records"], 2)
        self.assertEqual(report["holdout_raw_unique_queries"], 1)
        self.assertEqual(report["holdout_unique_queries"], 1)
        self.assertEqual(report["output_records"], 1)
        self.assertEqual(report["removed_records"], 1)
        self.assertEqual(report["removed_queries"], ["evaluate answer a"])
        self.assertEqual(kept[0]["id"], "keep")
        self.assertEqual(report["holdout_sha256"], expected_holdout_sha256)
        self.assertEqual(report["input_sha256"], expected_input_sha256)
        self.assertEqual(report["output_sha256"], expected_output_sha256)

    def test_filter_reads_sft_input_before_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "sft.jsonl"
            output = root / "sft.clean.jsonl"
            holdout.write_text(json.dumps({"query": "hard gold query"}) + "\n", encoding="utf-8")
            train.write_text(
                json.dumps({"instruction": "fixed rubric instruction", "input": "hard gold query"}) + "\n",
                encoding="utf-8",
            )

            report = filter_holdout_contamination(holdout=holdout, input_path=train, output=output)
            filtered_text = output.read_text(encoding="utf-8")

        self.assertEqual(report["removed_records"], 1)
        self.assertEqual(filtered_text, "")

    def test_filter_removes_verl_records_overlapping_via_extra_info_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "proxy_gold_verl.jsonl"
            output = root / "proxy_gold_verl.clean.jsonl"
            holdout.write_text(json.dumps({"query": "hard gold query"}) + "\n", encoding="utf-8")
            train.write_text(
                json.dumps(
                    {
                        "prompt": "为以下query生成评估rubric:\nhard gold query",
                        "extra_info": {"query": "hard gold query"},
                        "id": "drop",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                + json.dumps({"prompt": "为以下query生成评估rubric:\nclean query", "id": "keep"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            report = filter_holdout_contamination(holdout=holdout, input_path=train, output=output)
            kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["removed_records"], 1)
        self.assertEqual(report["removed_queries"], ["hard gold query"])
        self.assertEqual([row["id"] for row in kept], ["keep"])

    def test_cli_writes_filtered_file_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "train.jsonl"
            output = root / "train.clean.jsonl"
            report_path = root / "report.json"
            holdout.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")
            train.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            argv = [
                "filter_holdout_contamination.py",
                "--holdout",
                str(holdout),
                "--input",
                str(train),
                "--output",
                str(output),
                "--report",
                str(report_path),
                "--strict",
            ]
            with patch("sys.argv", argv):
                main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            filtered_text = output.read_text(encoding="utf-8")
            expected_holdout_sha256 = file_sha256(holdout)
            expected_input_sha256 = file_sha256(train)
            expected_output_sha256 = file_sha256(output)

        self.assertTrue(report["ok"])
        self.assertEqual(report["removed_records"], 1)
        self.assertEqual(filtered_text, "")
        self.assertEqual(report["holdout_sha256"], expected_holdout_sha256)
        self.assertEqual(report["input_sha256"], expected_input_sha256)
        self.assertEqual(report["output_sha256"], expected_output_sha256)


if __name__ == "__main__":
    unittest.main()
