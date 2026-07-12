from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_metric_ci import build_ci_report, file_sha256, parse_number, percentile


class BootstrapMetricCITest(unittest.TestCase):
    def test_parse_number_handles_booleans_and_floats(self) -> None:
        self.assertEqual(parse_number("True"), 1.0)
        self.assertEqual(parse_number("false"), 0.0)
        self.assertEqual(parse_number("0.25"), 0.25)
        self.assertIsNone(parse_number("not-a-number"))

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 1.0], 0.5), 0.5)

    def test_build_ci_report_for_numeric_and_boolean_columns(self) -> None:
        rows = [
            {"coverage": "0.5", "correct": "True"},
            {"coverage": "1.0", "correct": "False"},
            {"coverage": "0.0", "correct": "True"},
        ]

        report = build_ci_report(rows, metrics=["coverage", "correct"], n_boot=50, seed=7, confidence=0.8)

        self.assertEqual(report["n"], 3)
        coverage = report["metrics"][0]
        correct = report["metrics"][1]
        self.assertEqual(coverage["status"], "pass")
        self.assertAlmostEqual(coverage["mean"], 0.5)
        self.assertAlmostEqual(correct["mean"], 2 / 3)
        self.assertEqual(coverage["confidence"], 0.8)

    def test_missing_metric_is_reported(self) -> None:
        report = build_ci_report([{"coverage": "0.5"}], metrics=["accuracy"], n_boot=10)
        self.assertEqual(report["metrics"][0]["status"], "missing")

    def test_build_ci_report_records_input_sha_when_path_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "per_item.csv"
            input_path.write_text("coverage\n0.5\n", encoding="utf-8")

            report = build_ci_report(
                [{"coverage": "0.5"}],
                metrics=["coverage"],
                n_boot=10,
                input_path=input_path,
            )

            self.assertEqual(report["input"], str(input_path))
            self.assertEqual(report["input_sha256"], file_sha256(input_path))


if __name__ == "__main__":
    unittest.main()
