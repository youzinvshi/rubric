from __future__ import annotations

import unittest

from scripts.sample_records import build_report, sample_records, stratified_sample, to_markdown


class SampleRecordsTest(unittest.TestCase):
    def test_sample_records_is_reproducible(self) -> None:
        records = [{"query": f"q{i}", "data_source": "a"} for i in range(10)]
        first = sample_records(records, n=4, seed=7)
        second = sample_records(records, n=4, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_sample_records_dedupes_query(self) -> None:
        records = [
            {"query": "q1", "data_source": "a"},
            {"query": "q1", "data_source": "a"},
            {"query": "q2", "data_source": "a"},
        ]
        sampled = sample_records(records, n=10, seed=1, dedupe_key="query")
        self.assertEqual(len(sampled), 2)

    def test_stratified_sample_allocates_across_groups(self) -> None:
        groups = {
            "a": [{"query": f"a{i}", "data_source": "a"} for i in range(8)],
            "b": [{"query": f"b{i}", "data_source": "b"} for i in range(2)],
        }
        sampled = stratified_sample(groups, n=5, seed=1)
        counts = {key: sum(row["data_source"] == key for row in sampled) for key in ["a", "b"]}
        self.assertEqual(counts["a"], 4)
        self.assertEqual(counts["b"], 1)

    def test_to_markdown_contains_strata(self) -> None:
        class Args:
            input = "in.jsonl"
            output = "out.jsonl"
            seed = 13
            n = 1
            stratify_key = "data_source"
            dedupe_key = "query"

        report = build_report(
            [{"query": "q", "data_source": "rubricbench"}],
            [{"query": "q", "data_source": "rubricbench"}],
            Args(),
        )
        text = to_markdown(report)
        self.assertIn("rubricbench", text)


if __name__ == "__main__":
    unittest.main()
