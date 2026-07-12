from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.budget_gate import file_sha256
from scripts import prepare_downstream_eval
from scripts.prepare_downstream_eval import build_join_report, build_rubric_map


class PrepareDownstreamEvalTest(unittest.TestCase):
    def test_public_descriptions_use_criteria_framing(self) -> None:
        public_text = "\n".join(
            [
                prepare_downstream_eval.__doc__ or "",
                inspect.getsource(prepare_downstream_eval.parse_args),
                inspect.getsource(prepare_downstream_eval.main),
            ]
        )

        self.assertIn("generated evaluation criteria", public_text)
        self.assertIn("missing_criteria_records", public_text)
        self.assertNotIn("generated rubrics", public_text)
        self.assertNotIn("Join preferences and generated rubrics", public_text)

    def test_build_rubric_map_normalizes_and_dedupes(self) -> None:
        records = [{"query": " q   one ", "rubrics": ["a", "a", "b"], "model": "base"}]
        result = build_rubric_map(records)
        self.assertEqual(result["q one"]["rubrics"], ["a", "b"])
        self.assertEqual(result["q one"]["model"], "base")

    def test_build_rubric_map_filters_model_or_teacher(self) -> None:
        records = [
            {"query": "q one", "teacher": "gpt-4o", "rubrics": ["a"]},
            {"query": "q one", "teacher": "claude", "rubrics": ["b"]},
        ]
        result = build_rubric_map(records, model="claude")
        self.assertEqual(result["q one"]["rubrics"], ["b"])

    def test_join_report_records_pairwise_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preferences = root / "preferences.jsonl"
            rubrics = root / "rubrics.jsonl"
            output = root / "downstream_eval.jsonl"
            preferences.write_text('{"query":"q","chosen":"a","rejected":"b"}\n', encoding="utf-8")
            rubrics.write_text('{"query":"q","rubrics":["r"],"model":"base"}\n', encoding="utf-8")
            output.write_text('{"query":"q","rubrics":["r"]}\n', encoding="utf-8")

            report = build_join_report(
                source_label="preferences",
                source_path=preferences,
                rubrics_path=rubrics,
                output_path=output,
                data_source="rewardbench",
                model="base",
                counts={"n_joined": 1},
                limit=None,
            )

            self.assertEqual(report["preferences"], str(preferences))
            self.assertEqual(report["preferences_sha256"], file_sha256(preferences))
            self.assertEqual(report["rubrics"], str(rubrics))
            self.assertEqual(report["rubrics_sha256"], file_sha256(rubrics))
            self.assertEqual(report["output"], str(output))
            self.assertEqual(report["output_sha256"], file_sha256(output))
            self.assertEqual(report["data_source"], "rewardbench")
            self.assertEqual(report["model"], "base")
            self.assertEqual(report["n_joined"], 1)
            self.assertEqual(report["output_rows_count"], 1)
            self.assertTrue(report["output_rows_match_n_joined"])

    def test_main_reports_join_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preferences = root / "preferences.jsonl"
            rubrics = root / "rubrics.jsonl"
            output = root / "downstream_eval.jsonl"
            report = root / "join_report.json"
            preferences.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "chosen": "a", "rejected": "b"}),
                        json.dumps({"query": "q1", "chosen": "c", "rejected": "d"}),
                        json.dumps({"query": "q2", "chosen": "e", "rejected": "f"}),
                        json.dumps({"query": "", "chosen": "skip", "rejected": "skip"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rubrics.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "rubrics": ["r1"], "model": "base"}),
                        json.dumps({"query": "q1", "rubrics": ["r1 duplicate"], "model": "base"}),
                        json.dumps({"query": "q3", "rubrics": ["r3"], "model": "base"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "prepare_downstream_eval.py",
                    "--preferences",
                    str(preferences),
                    "--rubrics",
                    str(rubrics),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--model",
                    "base",
                    "--data-source",
                    "rewardbench",
                    "--limit",
                    "1",
                ],
            ):
                prepare_downstream_eval.main()

            data = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(data["n_source_records_raw"], 4)
        self.assertEqual(data["n_source_eligible_records"], 3)
        self.assertEqual(data["n_missing_rubrics"], 1)
        self.assertEqual(data["missing_rubric_keys_sample"], ["q2"])
        self.assertEqual(data["n_unmatched_rubrics"], 1)
        self.assertEqual(data["unmatched_rubric_keys_sample"], ["q3"])
        self.assertEqual(data["source_duplicate_record_count"], 1)
        self.assertEqual(data["rubric_duplicate_record_count"], 1)
        self.assertFalse(data["query_alignment_exact"])
        self.assertTrue(data["output_truncated_by_limit"])
        self.assertEqual(data["output_rows_count"], 1)
        self.assertTrue(data["output_rows_match_n_joined"])


if __name__ == "__main__":
    unittest.main()
