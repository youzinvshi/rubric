from __future__ import annotations

import unittest

from scripts.validate_rubric_outputs import summarize, to_markdown, validate_record
import scripts.validate_rubric_outputs as validate_rubric_outputs


class ValidateRubricOutputsTest(unittest.TestCase):
    def test_module_and_report_use_criteria_framing(self) -> None:
        self.assertIn("evaluation-criteria records", validate_rubric_outputs.__doc__ or "")
        self.assertNotIn("generated rubric records", validate_rubric_outputs.__doc__ or "")

    def test_validate_record_accepts_good_rubrics(self) -> None:
        row = validate_record(
            {
                "query": "Evaluate an answer.",
                "method": "base",
                "rubrics": [
                    "Mentions the required equation",
                    "Computes the final numeric answer",
                    "Explains the reasoning steps",
                ],
            },
            min_rubrics=3,
            max_rubrics=5,
        )
        self.assertTrue(row["ok"])
        self.assertEqual(row["issues"], [])

    def test_validate_record_can_require_verifier_flags(self) -> None:
        row = validate_record(
            {
                "query": "Evaluate an answer.",
                "method": "base",
                "rubrics": [
                    "Mentions the required equation",
                    "Computes the final numeric answer",
                    "Explains the reasoning steps",
                ],
                "valid_flags": [1, 1, 0],
                "verifier_source": "valid_flags",
            },
            min_rubrics=3,
            max_rubrics=5,
            require_valid_flags=True,
        )

        self.assertTrue(row["ok"])
        self.assertEqual(row["valid_flags"]["n_flags"], 3)

    def test_validate_record_blocks_missing_or_misaligned_verifier_flags(self) -> None:
        missing = validate_record(
            {
                "query": "Evaluate an answer.",
                "method": "base",
                "rubrics": [
                    "Mentions the required equation",
                    "Computes the final numeric answer",
                    "Explains the reasoning steps",
                ],
            },
            min_rubrics=3,
            max_rubrics=5,
            require_valid_flags=True,
        )
        mismatch = validate_record(
            {
                "query": "Evaluate an answer.",
                "method": "base",
                "rubrics": [
                    "Mentions the required equation",
                    "Computes the final numeric answer",
                    "Explains the reasoning steps",
                ],
                "valid_flags": [1, 0],
                "verifier_source": "valid_flags",
            },
            min_rubrics=3,
            max_rubrics=5,
            require_valid_flags=True,
        )

        self.assertIn("valid_flags_missing", missing["issues"])
        self.assertIn("valid_flags_length_mismatch", mismatch["issues"])

    def test_validate_record_flags_too_few_duplicate_and_generic(self) -> None:
        row = validate_record(
            {
                "query": "Evaluate an answer.",
                "method": "base",
                "rubrics": ["Helpful", "Helpful"],
            },
            min_rubrics=3,
            max_rubrics=5,
        )
        self.assertFalse(row["ok"])
        self.assertIn("too_few_rubrics", row["issues"])
        self.assertIn("exact_duplicates", row["issues"])
        self.assertIn("generic_terms", row["issues"])

    def test_validate_record_can_allow_diagnostic_quality_issues(self) -> None:
        row = validate_record(
            {
                "query": "Evaluate an answer.",
                "method": "base",
                "rubrics": ["Helpful", "Helpful"],
            },
            min_rubrics=1,
            max_rubrics=5,
            allow_exact_duplicates=True,
            allow_generic_terms=True,
            allow_semantic_redundancy=True,
        )

        self.assertTrue(row["ok"])
        self.assertEqual(row["issues"], [])

    def test_summarize_groups_by_method(self) -> None:
        rows = [
            {"method": "a", "ok": True, "issues": [], "n_rubrics": 3, "semantic_redundancy": 0.0},
            {
                "method": "a",
                "ok": False,
                "issues": ["too_few_rubrics"],
                "n_rubrics": 1,
                "semantic_redundancy": 0.0,
            },
        ]
        report = summarize(rows)
        self.assertFalse(report["ok"])
        self.assertEqual(report["methods"]["a"]["n_records"], 2)
        self.assertEqual(report["failed_records"], 1)
        self.assertEqual(report["issue_counts"]["too_few_rubrics"], 1)

    def test_to_markdown_includes_issue_counts(self) -> None:
        text = to_markdown(
            {
                "ok": False,
                "n_records": 1,
                "ok_rate": 0.0,
                "issue_counts": {"too_few_rubrics": 1},
                "methods": {
                    "base": {
                        "n_records": 1,
                        "ok_rate": 0.0,
                        "mean_n_rubrics": 1.0,
                        "parse_failures": 0,
                        "duplicate_records": 0,
                        "generic_records": 0,
                    }
                },
            }
        )
        self.assertIn("too_few_rubrics", text)
        self.assertIn("# Evaluation-Criteria Output Validation", text)
        self.assertIn("Mean Criteria", text)
        self.assertNotIn("# Rubric Output Validation", text)
        self.assertNotIn("Mean Rubrics", text)


if __name__ == "__main__":
    unittest.main()
