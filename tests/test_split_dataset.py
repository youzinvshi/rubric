from __future__ import annotations

import unittest

from scripts.split_dataset import detect_overlap, group_by_query, split_records


def make_records(n: int, rows_per_query: int = 1) -> list[dict]:
    records = []
    for i in range(n):
        for r in range(rows_per_query):
            records.append({"query": f"q{i}", "gold_rubrics": [f"crit-{i}-{r}"], "data_source": "rubricbench"})
    return records


class SplitDatasetTest(unittest.TestCase):
    def test_counts_and_stamps(self) -> None:
        records = make_records(100)
        splits, report = split_records(
            records=records,
            counts={"train_seed": 50, "dev": 20, "test_main": 30},
            seed=7,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(len(splits["train_seed"]), 50)
        self.assertEqual(len(splits["dev"]), 20)
        self.assertEqual(len(splits["test_main"]), 30)
        for row in splits["test_main"]:
            self.assertEqual(row["split"], "test_main")
            self.assertTrue(row["allowed_in_main_bsc_eval"])
            self.assertEqual(row["gold_type"], "human_gold")
        for row in splits["train_seed"] + splits["dev"]:
            self.assertFalse(row["allowed_in_main_bsc_eval"])

    def test_no_query_overlap_across_splits(self) -> None:
        records = make_records(100)
        splits, report = split_records(
            records=records,
            counts={"train_seed": 50, "dev": 20, "test_main": 30},
            seed=7,
        )
        assigned_queries = {name: sorted({row["query"] for row in rows}) for name, rows in splits.items()}
        self.assertEqual(detect_overlap(assigned_queries), set())
        # The three query sets must be pairwise disjoint.
        train = set(assigned_queries["train_seed"])
        dev = set(assigned_queries["dev"])
        test = set(assigned_queries["test_main"])
        self.assertEqual(train & dev, set())
        self.assertEqual(train & test, set())
        self.assertEqual(dev & test, set())

    def test_deterministic_given_seed(self) -> None:
        records = make_records(60)
        counts = {"train_seed": 30, "dev": 15, "test_main": 15}
        splits_a, _ = split_records(records=records, counts=counts, seed=123)
        splits_b, _ = split_records(records=records, counts=counts, seed=123)
        splits_c, _ = split_records(records=records, counts=counts, seed=999)
        self.assertEqual(
            [r["query"] for r in splits_a["test_main"]],
            [r["query"] for r in splits_b["test_main"]],
        )
        self.assertNotEqual(
            [r["query"] for r in splits_a["test_main"]],
            [r["query"] for r in splits_c["test_main"]],
        )

    def test_multi_row_query_never_splits(self) -> None:
        # Each query has 2 rows; grouping keeps them together.
        records = make_records(30, rows_per_query=2)
        splits, report = split_records(
            records=records,
            counts={"train_seed": 20, "dev": 20, "test_main": 20},
            seed=5,
        )
        assigned_queries = {name: {row["query"] for row in rows} for name, rows in splits.items()}
        self.assertEqual(detect_overlap({k: sorted(v) for k, v in assigned_queries.items()}), set())
        # Every split's record count is even because rows come in query-pairs.
        for rows in splits.values():
            self.assertEqual(len(rows) % 2, 0)

    def test_insufficient_data_blocks(self) -> None:
        records = make_records(10)
        _, report = split_records(
            records=records,
            counts={"train_seed": 50, "dev": 20, "test_main": 30},
            seed=1,
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["blockers"])

    def test_group_by_query_raises_on_missing_query(self) -> None:
        with self.assertRaises(ValueError):
            group_by_query([{"gold_rubrics": ["a"]}], "query")


if __name__ == "__main__":
    unittest.main()
