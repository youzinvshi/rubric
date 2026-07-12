from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts.prepare_bsc_eval import build_gold_map, build_prediction_map, join_blockers, main


class PrepareBSCEvalTest(unittest.TestCase):
    def test_build_gold_map_normalizes_query(self) -> None:
        records = [{"query": "  q  one ", "gold_rubrics": ["a", "a", "b"]}]
        result = build_gold_map(records)
        self.assertEqual(result["q one"]["gold_rubrics"], ["a", "b"])

    def test_build_prediction_map_filters_model_or_teacher(self) -> None:
        records = [
            {"query": "q one", "teacher": "gpt-4o", "rubrics": ["a"]},
            {"query": "q one", "teacher": "claude", "rubrics": ["b"]},
        ]
        result = build_prediction_map(records, model="gpt-4o")
        self.assertEqual(result["q one"]["rubrics"], ["a"])

    def test_build_prediction_map_preserves_verifier_flags(self) -> None:
        records = [
            {
                "query": "q one",
                "model": "base",
                "rubrics": ["criterion a", "criterion b"],
                "valid_flags": [1, 0],
                "verifier_source": "valid_flags",
            }
        ]

        result = build_prediction_map(records, model="base")

        self.assertEqual(result["q one"]["valid_flags"], [1, 0])
        self.assertEqual(result["q one"]["verifier_source"], "valid_flags")

    def test_duplicate_queries_are_preserved_by_occurrence(self) -> None:
        gold = build_gold_map(
            [
                {"query": "q one", "gold_rubrics": ["gold first"]},
                {"query": "q one", "gold_rubrics": ["gold second"]},
            ]
        )
        predictions = build_prediction_map(
            [
                {"query": "q one", "model": "base", "rubrics": ["pred first"]},
                {"query": "q one", "model": "base", "rubrics": ["pred second"]},
            ],
            model="base",
        )

        self.assertEqual(list(gold), ["q one", "q one [occurrence=2]"])
        self.assertEqual(gold["q one"]["gold_rubrics"], ["gold first"])
        self.assertEqual(gold["q one [occurrence=2]"]["gold_rubrics"], ["gold second"])
        self.assertEqual(predictions["q one"]["rubrics"], ["pred first"])
        self.assertEqual(predictions["q one [occurrence=2]"]["rubrics"], ["pred second"])

    def test_join_blockers_require_min_joined(self) -> None:
        self.assertEqual(join_blockers(2, min_joined=2), [])
        self.assertIn("2 < 3", join_blockers(2, min_joined=3)[0])

    def test_main_writes_report_before_failing_min_joined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.jsonl"
            output = root / "bsc.jsonl"
            report = root / "report.json"
            gold.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "gold_rubrics": ["g1"]}),
                        json.dumps({"query": "q2", "gold_rubrics": ["g2"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pred.write_text(json.dumps({"query": "q1", "model": "base", "rubrics": ["r1"]}) + "\n", encoding="utf-8")
            expected_gold_sha256 = file_sha256(gold)
            expected_predictions_sha256 = file_sha256(pred)

            argv = [
                "prepare_bsc_eval.py",
                "--gold",
                str(gold),
                "--predictions",
                str(pred),
                "--output",
                str(output),
                "--report",
                str(report),
                "--model",
                "base",
                "--min-joined",
                "2",
            ]
            with patch("sys.argv", argv), self.assertRaises(SystemExit):
                main()

            data = json.loads(report.read_text(encoding="utf-8"))

        self.assertFalse(data["ok"])
        self.assertEqual(data["gold"], str(gold))
        self.assertEqual(data["gold_sha256"], expected_gold_sha256)
        self.assertEqual(data["predictions"], str(pred))
        self.assertEqual(data["predictions_sha256"], expected_predictions_sha256)
        self.assertEqual(data["output"], str(output))
        self.assertEqual(data["output_sha256"], "")
        self.assertFalse(data["output_written"])
        self.assertEqual(data["output_rows_count"], 0)
        self.assertFalse(data["output_rows_match_n_joined"])
        self.assertEqual(data["n_gold_records_raw"], 2)
        self.assertEqual(data["n_prediction_records_raw"], 1)
        self.assertEqual(data["n_joinable"], 1)
        self.assertEqual(data["n_joined"], 1)
        self.assertEqual(data["n_missing_predictions"], 1)
        self.assertEqual(data["missing_prediction_keys_sample"], ["q2"])
        self.assertEqual(data["n_unmatched_predictions"], 0)
        self.assertFalse(data["query_alignment_exact"])
        self.assertEqual(data["gold_duplicate_record_count"], 0)
        self.assertEqual(data["prediction_duplicate_record_count"], 0)
        self.assertEqual(data["joined_records_with_valid_flags"], 0)
        self.assertFalse(data["all_joined_records_have_valid_flags"])
        self.assertIn("joined records below required minimum", data["blockers"][0])

    def test_main_records_output_hash_after_successful_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.jsonl"
            output = root / "bsc.jsonl"
            report = root / "report.json"
            gold.write_text(json.dumps({"query": "q1", "gold_rubrics": ["g1"]}) + "\n", encoding="utf-8")
            pred.write_text(json.dumps({"query": "q1", "model": "base", "rubrics": ["r1"]}) + "\n", encoding="utf-8")

            argv = [
                "prepare_bsc_eval.py",
                "--gold",
                str(gold),
                "--predictions",
                str(pred),
                "--output",
                str(output),
                "--report",
                str(report),
                "--model",
                "base",
                "--min-joined",
                "1",
            ]
            with patch("sys.argv", argv):
                main()

            data = json.loads(report.read_text(encoding="utf-8"))

            self.assertTrue(data["ok"])
            self.assertEqual(data["output"], str(output))
            self.assertEqual(data["output_sha256"], file_sha256(output))
            self.assertTrue(data["output_written"])
            self.assertEqual(data["output_rows_count"], 1)
            self.assertTrue(data["output_rows_match_n_joined"])
            self.assertEqual(data["n_joinable"], 1)
            self.assertEqual(data["n_joined"], 1)
            self.assertEqual(data["n_missing_predictions"], 0)
            self.assertEqual(data["n_unmatched_predictions"], 0)
            self.assertTrue(data["query_alignment_exact"])
            self.assertFalse(data["output_truncated_by_limit"])

    def test_main_reports_duplicate_and_unmatched_query_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.jsonl"
            output = root / "bsc.jsonl"
            report = root / "report.json"
            gold.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "gold_rubrics": ["g1"]}),
                        json.dumps({"query": "q1", "gold_rubrics": ["g1 duplicate"]}),
                        json.dumps({"query": "q2", "gold_rubrics": ["g2"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pred.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "model": "base", "rubrics": ["r1"], "valid_flags": [1]}),
                        json.dumps({"query": "q1", "model": "base", "rubrics": ["r1 duplicate"], "valid_flags": [1]}),
                        json.dumps({"query": "q3", "model": "base", "rubrics": ["r3"], "valid_flags": [1]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "prepare_bsc_eval.py",
                "--gold",
                str(gold),
                "--predictions",
                str(pred),
                "--output",
                str(output),
                "--report",
                str(report),
                "--model",
                "base",
            ]
            with patch("sys.argv", argv):
                main()

            data = json.loads(report.read_text(encoding="utf-8"))

        self.assertTrue(data["ok"])
        self.assertEqual(data["gold_duplicate_join_key_count"], 1)
        self.assertEqual(data["gold_duplicate_record_count"], 1)
        self.assertEqual(data["prediction_duplicate_join_key_count"], 1)
        self.assertEqual(data["prediction_duplicate_record_count"], 1)
        self.assertEqual(data["n_joinable"], 2)
        self.assertEqual(data["n_joined"], 2)
        self.assertEqual(data["n_missing_predictions"], 1)
        self.assertEqual(data["missing_prediction_keys_sample"], ["q2"])
        self.assertEqual(data["n_unmatched_predictions"], 1)
        self.assertEqual(data["unmatched_prediction_keys_sample"], ["q3"])
        self.assertFalse(data["query_alignment_exact"])
        self.assertEqual(data["joined_records_with_valid_flags"], 2)
        self.assertTrue(data["all_joined_records_have_valid_flags"])

    def test_main_reports_limit_truncation_without_hiding_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            pred = root / "pred.jsonl"
            output = root / "bsc.jsonl"
            report = root / "report.json"
            gold.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "gold_rubrics": ["g1"]}),
                        json.dumps({"query": "q2", "gold_rubrics": ["g2"]}),
                        json.dumps({"query": "q3", "gold_rubrics": ["g3"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pred.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "model": "base", "rubrics": ["r1"]}),
                        json.dumps({"query": "q2", "model": "base", "rubrics": ["r2"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "prepare_bsc_eval.py",
                "--gold",
                str(gold),
                "--predictions",
                str(pred),
                "--output",
                str(output),
                "--report",
                str(report),
                "--model",
                "base",
                "--limit",
                "1",
            ]
            with patch("sys.argv", argv):
                main()

            data = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(data["n_joinable"], 2)
        self.assertEqual(data["n_joined"], 1)
        self.assertEqual(data["n_missing_predictions"], 1)
        self.assertEqual(data["missing_prediction_keys_sample"], ["q3"])
        self.assertTrue(data["output_truncated_by_limit"])


if __name__ == "__main__":
    unittest.main()
