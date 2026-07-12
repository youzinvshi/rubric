from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from scripts.summarize_experiments import (
    DOWNSTREAM_NOT_ELIGIBLE_ERROR,
    apply_ci_report,
    apply_downstream_summary,
    display_metric,
    format_with_ci,
    read_json,
    to_markdown,
    write_csv,
)


class SummarizeExperimentsTest(unittest.TestCase):
    def test_apply_ci_report_adds_display_and_bounds(self) -> None:
        row = {"method": "base", "cov": 0.6}
        report = {
            "metrics": [
                {"metric": "coverage", "status": "pass", "mean": 0.6, "ci_lower": 0.5, "ci_upper": 0.7},
                {"metric": "blind", "status": "missing"},
            ]
        }

        apply_ci_report(row, report, {"coverage": "cov", "blind": "blind"})

        self.assertEqual(row["cov_ci_lower"], 0.5)
        self.assertEqual(row["cov_ci_upper"], 0.7)
        self.assertEqual(row["cov_ci"], "0.6000 [0.5000, 0.7000]")
        self.assertNotIn("blind_ci", row)

    def test_format_with_ci(self) -> None:
        self.assertEqual(format_with_ci(0.5, 0.4, 0.6), "0.5000 [0.4000, 0.6000]")

    def test_markdown_prefers_ci_display(self) -> None:
        md = to_markdown(
            [
                {
                    "method": "base",
                    "cov": 0.6,
                    "cov_ci": "0.6000 [0.5000, 0.7000]",
                    "blind": 0.4,
                    "mean_n_gen": 6.0,
                    "coverage_per_generated_criterion": 0.1,
                }
            ]
        )

        self.assertIn("0.6000 [0.5000, 0.7000]", md)
        self.assertIn("0.4000", md)
        self.assertIn("6.0000", md)
        self.assertIn("0.1000", md)

    def test_display_metric_falls_back_to_point_estimate(self) -> None:
        self.assertEqual(display_metric({"cov": 0.5}, "cov"), "0.5000")

    def test_read_json_returns_load_error_for_invalid_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_summary.json"
            path.write_text("{bad", encoding="utf-8")

            data = read_json(path)

        self.assertIn("_load_error", data)
        self.assertIn("not valid JSON", data["_load_error"])

    def test_markdown_exposes_blocked_summary_status(self) -> None:
        md = to_markdown(
            [
                {
                    "method": "base",
                    "bsc_status": "blocked",
                    "bsc_error": "summary.json: not valid JSON at line 1 column 2",
                    "downstream_status": "pass",
                    "accuracy": 0.5,
                }
            ]
        )

        self.assertIn("blocked: summary.json", md)
        self.assertIn("pass", md)

    def test_write_csv_includes_status_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"

            write_csv(
                path,
                [
                    {
                        "method": "base",
                        "bsc_status": "blocked",
                        "bsc_error": "bad json",
                        "mean_n_gen": 4.0,
                        "gen_to_gold_ratio": 1.25,
                        "coverage_per_generated_criterion": 0.2,
                        "downstream_status": "pass",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("bsc_status", text)
        self.assertIn("mean_n_gen", text)
        self.assertIn("gen_to_gold_ratio", text)
        self.assertIn("coverage_per_generated_criterion", text)
        self.assertIn("downstream_status", text)
        self.assertIn("bad json", text)
        self.assertIn("1.25", text)
        self.assertIn("0.2", text)

    def test_downstream_summary_requires_paper_claim_eligible_for_metrics(self) -> None:
        row = {"method": "base"}

        apply_downstream_summary(
            row,
            {
                "paper_claim_eligible": False,
                "scorer": "keyword",
                "n": 10,
                "accuracy": 0.8,
                "tie_rate": 0.1,
                "mean_margin": 0.2,
            },
        )

        self.assertEqual(row["downstream_status"], "not_paper_eligible")
        self.assertEqual(row["downstream_error"], DOWNSTREAM_NOT_ELIGIBLE_ERROR)
        self.assertEqual(row["downstream_paper_claim_eligible"], "false")
        self.assertEqual(row["downstream_scorer"], "keyword")
        self.assertEqual(row["accuracy"], "")
        self.assertEqual(row["tie_rate"], "")
        self.assertEqual(row["mean_margin"], "")

    def test_downstream_summary_keeps_metrics_when_paper_claim_eligible(self) -> None:
        row = {"method": "base"}

        apply_downstream_summary(
            row,
            {
                "paper_claim_eligible": True,
                "scorer": "api",
                "scorer_provider": "configs/judge.jsonl",
                "budget_report": "outputs/budget.json",
                "benchmark_format": "pairwise",
                "n": 10,
                "accuracy": 0.8,
                "tie_rate": 0.1,
                "mean_margin": 0.2,
            },
        )

        self.assertEqual(row["downstream_status"], "pass")
        self.assertEqual(row["downstream_paper_claim_eligible"], "true")
        self.assertEqual(row["downstream_scorer"], "api")
        self.assertEqual(row["downstream_n"], 10)
        self.assertEqual(row["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
