from __future__ import annotations

import unittest

from scripts.validate_gold_data import parse_required_provenance, validate_gold_records, to_markdown


class ValidateGoldDataTest(unittest.TestCase):
    def test_valid_gold_records_pass(self) -> None:
        report = validate_gold_records(
            records=[
                {
                    "query": "q1",
                    "gold_rubrics": ["criterion a", "criterion b"],
                    "data_source": "rubricbench",
                    "provenance": "official_release",
                }
            ],
            min_records=1,
            min_rubrics_per_query=1,
            require_provenance=True,
            forbidden_data_sources={"toy"},
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["blockers"], [])

    def test_too_few_records_and_forbidden_source_block(self) -> None:
        report = validate_gold_records(
            records=[{"query": "q1", "gold_rubrics": ["a"], "data_source": "toy"}],
            min_records=2,
            forbidden_data_sources={"toy"},
        )

        self.assertFalse(report["ok"])
        self.assertIn("record count 1 < min_records 2", report["blockers"])
        self.assertTrue(any("forbidden data_source" in item for item in report["blockers"]))

    def test_missing_provenance_blocks_when_required(self) -> None:
        report = validate_gold_records(
            records=[{"query": "q1", "gold_rubrics": ["a"], "data_source": "rubricbench"}],
            require_provenance=True,
        )

        self.assertFalse(report["ok"])
        self.assertTrue(any("missing provenance field" in item for item in report["blockers"]))

    def test_required_data_source_blocks_mislabeled_hard_gold(self) -> None:
        report = validate_gold_records(
            records=[
                {
                    "query": "q1",
                    "gold_rubrics": ["a"],
                    "data_source": "other_gold",
                    "provenance": "official_release",
                }
            ],
            require_provenance=True,
            required_data_sources={"rubricbench"},
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["required_data_sources"], ["rubricbench"])
        self.assertTrue(any("not in required_data_sources" in item for item in report["blockers"]))

    def test_required_provenance_values_block_wrong_official_source(self) -> None:
        report = validate_gold_records(
            records=[
                {
                    "query": "q1",
                    "gold_rubrics": ["a"],
                    "data_source": "rubricbench",
                    "provenance": "official_release",
                    "source_url": "https://example.com/proxy.jsonl",
                    "paper_url": "https://arxiv.org/abs/2603.01562",
                }
            ],
            require_provenance=True,
            required_provenance={
                "source_url": "https://official.example/rubricbench.jsonl",
                "paper_url": "https://arxiv.org/abs/2603.01562",
            },
        )

        self.assertFalse(report["ok"])
        self.assertEqual(
            report["required_provenance"]["source_url"],
            "https://official.example/rubricbench.jsonl",
        )
        self.assertTrue(any("source_url mismatch" in item for item in report["blockers"]))

    def test_required_provenance_parser_requires_key_value(self) -> None:
        self.assertEqual(parse_required_provenance(["paper_url=https://arxiv.org/abs/2603.01562"]), {
            "paper_url": "https://arxiv.org/abs/2603.01562"
        })
        with self.assertRaises(ValueError):
            parse_required_provenance(["paper_url"])

    def test_bad_rubrics_and_duplicate_query_are_flagged(self) -> None:
        report = validate_gold_records(
            records=[
                {"query": "q1", "gold_rubrics": ["a", "a", ""], "data_source": "rubricbench"},
                {"query": "q1", "gold_rubrics": "not-list", "data_source": "rubricbench"},
            ],
            min_rubrics_per_query=2,
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["n_records"], 2)
        self.assertEqual(report["n_unique_queries"], 1)
        self.assertEqual(report["n_duplicate_queries"], 1)
        self.assertEqual(report["n_duplicate_query_records"], 2)
        self.assertTrue(any("gold_rubrics must be a list" in item for item in report["blockers"]))
        self.assertTrue(any("duplicate query" in item for item in report["warnings"]))

    def test_query_pool_target_does_not_require_gold_rubrics(self) -> None:
        report = validate_gold_records(
            records=[
                {
                    "query": "q1",
                    "data_source": "rubricbench",
                    "paper_url": "https://arxiv.org/abs/2603.01562",
                }
            ],
            target="query_pool",
            min_records=1,
            require_provenance=True,
            required_provenance={"paper_url": "https://arxiv.org/abs/2603.01562"},
            required_data_sources={"rubricbench"},
            forbidden_data_sources={"toy", "proxy"},
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["target"], "query_pool")
        self.assertEqual(report["blockers"], [])

    def test_query_pool_markdown_omits_rubric_column(self) -> None:
        report = validate_gold_records(
            records=[{"query": "q1", "data_source": "rubricbench"}],
            target="query_pool",
            min_records=1,
        )
        md = to_markdown(report)

        self.assertIn("Query Pool Validation", md)
        self.assertNotIn("Min rubrics/query", md)
        self.assertNotIn("n_rubrics", md)

    def test_markdown_lists_blockers(self) -> None:
        report = validate_gold_records(records=[], min_records=1)
        md = to_markdown(report)
        self.assertIn("Gold Data Validation", md)
        self.assertIn("Unique queries", md)
        self.assertIn("Duplicate query groups", md)
        self.assertIn("record count 0", md)


if __name__ == "__main__":
    unittest.main()
