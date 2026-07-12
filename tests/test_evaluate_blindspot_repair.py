from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.evaluate_blindspot_repair import (
    evaluate_blindspot_repair,
    extract_generated_rubrics,
    main,
)


class EvaluateBlindspotRepairTest(unittest.TestCase):
    def test_evaluate_transition_counts_dimension_level_recoveries_and_losses(self) -> None:
        baseline = [
            {
                "query": "q1",
                "gold_rubrics": [
                    "Uses citations and evidence",
                    "Follows JSON format",
                    "Follows safety policy",
                ],
                "response": ["Follows JSON format"],
            }
        ]
        candidate = [
            {
                "query": "q1",
                "gold_rubrics": [
                    "Uses citations and evidence",
                    "Follows JSON format",
                    "Follows safety policy",
                ],
                "response": ["Uses citations and evidence", "Follows safety policy"],
            }
        ]

        per_item, category_rows, gold_rows, summary = evaluate_blindspot_repair(
            baseline,
            candidate,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )

        self.assertEqual(summary["total_gold"], 3)
        self.assertEqual(summary["baseline_covered_gold"], 1)
        self.assertEqual(summary["baseline_blind_gold"], 2)
        self.assertEqual(summary["candidate_covered_gold"], 2)
        self.assertEqual(summary["recovered_gold"], 2)
        self.assertNotIn("repaired_gold", summary)
        self.assertEqual(summary["lost_gold"], 1)
        self.assertAlmostEqual(summary["recovered_dimension_rate"], 1.0)
        self.assertNotIn("repair_rate", summary)
        self.assertAlmostEqual(summary["loss_rate"], 1.0)
        self.assertAlmostEqual(summary["net_transition_rate"], 1 / 3)
        self.assertNotIn("net_repair_rate", summary)
        self.assertEqual(summary["gold_rows_count"], 3)
        self.assertTrue(summary["gold_rows_match_total_gold"])
        self.assertEqual(summary["per_item_rows_count"], 1)
        self.assertTrue(summary["per_item_rows_match_n_matched_records"])
        self.assertEqual(summary["transition_balance"], 1)
        self.assertTrue(summary["recovered_exceeds_lost"])
        self.assertTrue(summary["net_positive_transition"])
        self.assertTrue(summary["query_alignment_exact"])
        self.assertEqual(per_item[0]["recovered_gold"], 2)
        self.assertNotIn("repaired_gold", per_item[0])
        self.assertEqual(len(gold_rows), 3)
        self.assertEqual(gold_rows[0]["recovered"], 1)
        self.assertNotIn("repaired", gold_rows[0])

        by_category = {row["category"]: row for row in category_rows}
        self.assertEqual(by_category["evidence_grounding"]["recovered_gold"], 1)
        self.assertEqual(by_category["constraint_following"]["lost_gold"], 1)
        self.assertEqual(by_category["safety"]["recovered_gold"], 1)

    def test_valid_flags_prevent_invalid_rubrics_from_repairing_blind_spots(self) -> None:
        baseline = [
            {
                "query": "q1",
                "gold_rubrics": ["Uses citations and evidence"],
                "response": [],
            }
        ]
        candidate = [
            {
                "query": "q1",
                "gold_rubrics": ["Uses citations and evidence"],
                "response": ["Uses citations and evidence"],
                "valid_flags": [0],
            }
        ]

        _, _, _, summary = evaluate_blindspot_repair(
            baseline,
            candidate,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )

        self.assertEqual(summary["recovered_gold"], 0)
        self.assertNotIn("repaired_gold", summary)
        self.assertEqual(summary["still_blind_gold"], 1)
        self.assertEqual(extract_generated_rubrics(candidate[0]), [])

    def test_duplicate_queries_align_by_occurrence(self) -> None:
        baseline = [
            {"query": "q", "gold_rubrics": ["first gold"], "response": []},
            {"query": "q", "gold_rubrics": ["second gold"], "response": []},
        ]
        candidate = [
            {"query": "q", "gold_rubrics": ["first gold"], "response": ["first gold"]},
            {"query": "q", "gold_rubrics": ["second gold"], "response": ["second gold"]},
        ]

        per_item, _, _, summary = evaluate_blindspot_repair(
            baseline,
            candidate,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )

        self.assertEqual(summary["n_matched_records"], 2)
        self.assertEqual(summary["recovered_gold"], 2)
        self.assertNotIn("repaired_gold", summary)
        self.assertEqual(summary["baseline_duplicate_join_key_count"], 1)
        self.assertEqual(summary["baseline_duplicate_record_count"], 1)
        self.assertEqual(summary["candidate_duplicate_join_key_count"], 1)
        self.assertEqual(summary["candidate_duplicate_record_count"], 1)
        self.assertTrue(summary["query_alignment_exact"])
        self.assertEqual(per_item[0]["join_key"], "q")
        self.assertEqual(per_item[1]["join_key"], "q [duplicate #2]")

    def test_cli_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "baseline.jsonl"
            candidate_path = tmp_path / "candidate.jsonl"
            output_dir = tmp_path / "repair"
            baseline_path.write_text(
                json.dumps(
                    {
                        "query": "q1",
                        "gold_rubrics": ["Uses citations and evidence"],
                        "response": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        "query": "q1",
                        "gold_rubrics": ["Uses citations and evidence"],
                        "response": ["Uses citations and evidence"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_blindspot_repair.py",
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output-dir",
                    str(output_dir),
                    "--embedding-model",
                    "token-overlap",
                    "--coverage-tau",
                    "0.99",
                ]
                main()
            finally:
                sys.argv = old_argv

            summary = json.loads((output_dir / "transition_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["recovered_dimension_rate"], 1.0)
            self.assertNotIn("repair_rate", summary)
            self.assertNotIn("net_repair_rate", summary)
            self.assertEqual(summary["net_transition_rate"], 1.0)
            self.assertEqual(len(summary["baseline_sha256"]), 64)
            self.assertEqual(len(summary["candidate_sha256"]), 64)
            self.assertTrue(summary["query_alignment_exact"])
            self.assertTrue(summary["gold_rows_match_total_gold"])
            self.assertTrue(summary["per_item_rows_match_n_matched_records"])
            self.assertEqual(summary["transition_balance"], 1)
            self.assertTrue(summary["recovered_exceeds_lost"])
            self.assertTrue(summary["net_positive_transition"])
            self.assertTrue((output_dir / "transition_per_item.csv").exists())
            self.assertTrue((output_dir / "transition_by_category.csv").exists())
            self.assertTrue((output_dir / "transition_gold_items.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
