from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts.audit_holdout_contamination import audit_holdout_contamination, extract_query, extract_query_candidates, main


class AuditHoldoutContaminationTest(unittest.TestCase):
    def test_audit_passes_when_holdout_and_training_queries_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "train.jsonl"
            holdout.write_text(json.dumps({"query": "Evaluate answer A"}) + "\n", encoding="utf-8")
            train.write_text(json.dumps({"query": "Generate rubric for answer B"}) + "\n", encoding="utf-8")

            report, rows = audit_holdout_contamination(
                holdout=holdout,
                training_specs=[f"sft={train}"],
            )
            expected_holdout_sha256 = file_sha256(holdout)
            expected_training_sha256 = file_sha256(train)

        self.assertTrue(report["ok"])
        self.assertEqual(report["artifact_status"], "complete")
        self.assertEqual(report["overlap_status"], "clear")
        self.assertEqual(report["overlap_query_count"], 0)
        self.assertEqual(report["holdout_raw_unique_queries"], 1)
        self.assertEqual(report["holdout_unique_queries"], 1)
        self.assertEqual(report["training_raw_unique_queries"], 1)
        self.assertEqual(report["training_unique_queries"], 1)
        self.assertEqual(report["training"][0]["raw_unique_queries"], 1)
        self.assertEqual(report["training"][0]["unique_queries"], 1)
        self.assertEqual(rows, [])
        self.assertEqual(report["holdout_sha256"], expected_holdout_sha256)
        self.assertEqual(report["training"][0]["sha256"], expected_training_sha256)

    def test_audit_reports_raw_and_normalized_unique_query_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "train.jsonl"
            holdout.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "query with spaces"}),
                        json.dumps({"query": "query  with   spaces"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            train.write_text(json.dumps({"query": "different query"}) + "\n", encoding="utf-8")

            report, rows = audit_holdout_contamination(
                holdout=holdout,
                training_specs=[f"sft={train}"],
            )

        self.assertTrue(report["ok"])
        self.assertEqual(rows, [])
        self.assertEqual(report["holdout_raw_unique_queries"], 2)
        self.assertEqual(report["holdout_unique_queries"], 1)

    def test_audit_blocks_query_overlap_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "train.jsonl"
            holdout.write_text(json.dumps({"query": "Evaluate   Answer A"}) + "\n", encoding="utf-8")
            train.write_text(json.dumps({"prompt": " evaluate answer a "}) + "\n", encoding="utf-8")

            report, rows = audit_holdout_contamination(
                holdout=holdout,
                training_specs=[f"proxy={train}"],
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["artifact_status"], "complete")
        self.assertEqual(report["overlap_status"], "overlap_found")
        self.assertEqual(report["overlap_query_count"], 1)
        self.assertIn("overlap", report["blockers"][0])
        self.assertEqual(rows[0]["training_labels"], "proxy")

    def test_audit_blocks_missing_training_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            holdout.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            report, _ = audit_holdout_contamination(
                holdout=holdout,
                training_specs=[f"sft={root / 'missing.jsonl'}"],
            )
            expected_holdout_sha256 = file_sha256(holdout)

        self.assertFalse(report["ok"])
        self.assertEqual(report["artifact_status"], "blocked")
        self.assertEqual(report["overlap_status"], "not_auditable")
        self.assertTrue(any("missing or empty file" in item for item in report["blockers"]))
        self.assertEqual(report["holdout_sha256"], expected_holdout_sha256)
        self.assertEqual(report["training"][0]["sha256"], "")

    def test_extract_query_reads_nested_extra_info(self) -> None:
        self.assertEqual(
            extract_query({"extra_info": {"query": "  A Query\n"}}, query_keys=("query",)),
            "a query",
        )

    def test_extract_query_candidates_keeps_verl_extra_info_query(self) -> None:
        candidates = extract_query_candidates(
            {
                "prompt": "为以下query生成评估rubric:\nhard gold query",
                "extra_info": {"query": "hard gold query"},
            },
            query_keys=("query", "prompt"),
        )

        self.assertEqual(
            candidates,
            ["为以下query生成评估rubric: hard gold query", "hard gold query"],
        )

    def test_audit_catches_verl_record_overlap_via_extra_info_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "proxy_gold_verl.jsonl"
            holdout.write_text(json.dumps({"query": "hard gold query"}) + "\n", encoding="utf-8")
            train.write_text(
                json.dumps(
                    {
                        "prompt": "为以下query生成评估rubric:\nhard gold query",
                        "extra_info": {"query": "hard gold query"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report, rows = audit_holdout_contamination(
                holdout=holdout,
                training_specs=[f"verl={train}"],
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlap_query_count"], 1)
        self.assertEqual(rows[0]["query"], "hard gold query")
        self.assertEqual(rows[0]["training_labels"], "verl")

    def test_audit_reads_json_encoded_extra_info_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "proxy_gold_verl.jsonl"
            holdout.write_text(json.dumps({"query": "hard gold query"}) + "\n", encoding="utf-8")
            train.write_text(
                json.dumps(
                    {
                        "prompt": "为以下query生成评估rubric:\nhard gold query",
                        "extra_info": json.dumps({"query": "hard gold query"}),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report, _ = audit_holdout_contamination(
                holdout=holdout,
                training_specs=[f"verl={train}"],
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlap_query_count"], 1)

    def test_cli_writes_json_and_overlap_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.jsonl"
            train = root / "train.jsonl"
            output = root / "audit.json"
            output_csv = root / "overlap.csv"
            holdout.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")
            train.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            argv = [
                "audit_holdout_contamination.py",
                "--holdout",
                str(holdout),
                "--training",
                f"train={train}",
                "--output",
                str(output),
                "--output-csv",
                str(output_csv),
            ]
            with patch("sys.argv", argv):
                main()

            report = json.loads(output.read_text(encoding="utf-8"))
            overlap_csv = output_csv.read_text(encoding="utf-8")
            expected_holdout_sha256 = file_sha256(holdout)
            expected_training_sha256 = file_sha256(train)

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlap_query_count"], 1)
        self.assertIn("training_labels", overlap_csv)
        self.assertEqual(report["holdout_sha256"], expected_holdout_sha256)
        self.assertEqual(report["training"][0]["sha256"], expected_training_sha256)


if __name__ == "__main__":
    unittest.main()
