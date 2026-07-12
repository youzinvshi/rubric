from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_policy_outputs import build_report, load_records, to_markdown


class EvaluatePolicyOutputsTest(unittest.TestCase):
    def test_build_report_uses_boolean_correctness(self) -> None:
        report, rows = build_report(
            rows=[
                {"query": "q1", "correct": True, "score": 0.9},
                {"query": "q2", "correct": False, "score": 0.2},
                {"query": "q3", "passed": "yes"},
            ],
            input_path=Path("predictions.jsonl"),
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["n_evaluable"], 3)
        self.assertEqual(report["n_correctness"], 3)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(rows[0]["correct_key"], "correct")

    def test_build_report_can_threshold_numeric_scores(self) -> None:
        report, rows = build_report(
            rows=[
                {"query": "q1", "judge_score": 0.8},
                {"query": "q2", "judge_score": 0.4},
            ],
            input_path=Path("predictions.jsonl"),
            min_score=0.5,
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["n_scores"], 2)
        self.assertAlmostEqual(report["mean_score"], 0.6)
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertEqual(rows[0]["correct_key"], "judge_score>=0.5")

    def test_build_report_blocks_when_no_evaluable_fields_exist(self) -> None:
        report, _ = build_report(
            rows=[{"query": "q1", "response": "answer"}],
            input_path=Path("predictions.jsonl"),
        )

        self.assertFalse(report["ok"])
        self.assertIn("No evaluable policy records", report["blockers"][0])
        self.assertIn("blocked", to_markdown(report))

    def test_load_records_accepts_json_prediction_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text(json.dumps({"predictions": [{"query": "q", "correct": True}]}), encoding="utf-8")

            rows = list(load_records(path))

        self.assertEqual(rows, [{"query": "q", "correct": True}])


if __name__ == "__main__":
    unittest.main()
