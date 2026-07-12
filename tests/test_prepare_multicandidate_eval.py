from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.budget_gate import file_sha256
from scripts import prepare_multicandidate_eval
from scripts.prepare_downstream_eval import build_join_report
from scripts.prepare_multicandidate_eval import normalize_candidates, normalize_label


class PrepareMultiCandidateEvalTest(unittest.TestCase):
    def test_public_descriptions_use_criteria_framing(self) -> None:
        public_text = "\n".join(
            [
                prepare_multicandidate_eval.__doc__ or "",
                inspect.getsource(prepare_multicandidate_eval.parse_args),
                inspect.getsource(prepare_multicandidate_eval.main),
            ]
        )

        self.assertIn("generated evaluation criteria", public_text)
        self.assertIn("missing_criteria_records", public_text)
        self.assertNotIn("generated rubrics", public_text)
        self.assertNotIn("benchmark records and generated rubrics", public_text)

    def test_normalize_candidates_accepts_object_list(self) -> None:
        self.assertEqual(
            normalize_candidates([{"text": "answer a"}, {"response": "answer b"}]),
            ["answer a", "answer b"],
        )

    def test_normalize_label_accepts_index_or_text(self) -> None:
        candidates = ["answer a", "answer b"]
        self.assertEqual(normalize_label("1", candidates), 1)
        self.assertEqual(normalize_label("answer b", candidates), 1)
        self.assertIsNone(normalize_label("3", candidates))

    def test_join_report_records_multicandidate_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.jsonl"
            rubrics = root / "rubrics.jsonl"
            output = root / "downstream_eval.jsonl"
            benchmark.write_text('{"query":"q","candidates":["a","b"],"label":0}\n', encoding="utf-8")
            rubrics.write_text('{"query":"q","rubrics":["r"],"model":"sft_rl"}\n', encoding="utf-8")
            output.write_text('{"query":"q","candidates":["a","b"],"rubrics":["r"]}\n', encoding="utf-8")

            report = build_join_report(
                source_label="benchmark",
                source_path=benchmark,
                rubrics_path=rubrics,
                output_path=output,
                data_source="rewardbench2",
                model="sft_rl",
                counts={"n_joined": 1},
                limit=100,
            )

            self.assertEqual(report["benchmark"], str(benchmark))
            self.assertEqual(report["benchmark_sha256"], file_sha256(benchmark))
            self.assertEqual(report["rubrics"], str(rubrics))
            self.assertEqual(report["rubrics_sha256"], file_sha256(rubrics))
            self.assertEqual(report["output"], str(output))
            self.assertEqual(report["output_sha256"], file_sha256(output))
            self.assertEqual(report["data_source"], "rewardbench2")
            self.assertEqual(report["model"], "sft_rl")
            self.assertEqual(report["limit"], 100)
            self.assertEqual(report["output_rows_count"], 1)
            self.assertTrue(report["output_rows_match_n_joined"])

    def test_main_reports_join_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.jsonl"
            rubrics = root / "rubrics.jsonl"
            output = root / "downstream_eval.jsonl"
            report = root / "join_report.json"
            benchmark.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "candidates": ["a", "b"], "label": 0}),
                        json.dumps({"query": "q1", "candidates": ["c", "d"], "label": 1}),
                        json.dumps({"query": "q2", "candidates": ["e", "f"], "label": 0}),
                        json.dumps({"query": "bad", "candidates": ["only one"], "label": 0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rubrics.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "rubrics": ["r1"], "model": "sft_rl"}),
                        json.dumps({"query": "q1", "rubrics": ["r1 duplicate"], "model": "sft_rl"}),
                        json.dumps({"query": "q3", "rubrics": ["r3"], "model": "sft_rl"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "prepare_multicandidate_eval.py",
                    "--benchmark",
                    str(benchmark),
                    "--rubrics",
                    str(rubrics),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--model",
                    "sft_rl",
                    "--data-source",
                    "rewardbench2",
                    "--limit",
                    "1",
                ],
            ):
                prepare_multicandidate_eval.main()

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
