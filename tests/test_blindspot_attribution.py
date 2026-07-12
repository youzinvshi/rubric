from __future__ import annotations

import unittest

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.blindspot_attribution import build_blindspot_map, classify_rubric


class BlindspotAttributionTest(unittest.TestCase):
    def test_classify_rubric_uses_stable_categories(self) -> None:
        self.assertEqual(classify_rubric("Uses citations and evidence from the source."), "evidence_grounding")
        self.assertEqual(classify_rubric("Follows the requested JSON format."), "constraint_following")
        self.assertEqual(classify_rubric("Explains the user's intent and ambiguity."), "intent_reasoning")

    def test_build_blindspot_map_extracts_uncovered_gold_and_category_summary(self) -> None:
        rows, category_rows, summary = build_blindspot_map(
            [
                {
                    "query": "q",
                    "gold_rubrics": ["Uses citations and evidence", "Follows safety policy"],
                    "response": ["Follows safety policy"],
                    "model": "base",
                }
            ],
            coverage_tau=0.99,
            embedder=TokenOverlapEmbedder(),
            model="base",
        )

        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["total_gold"], 2)
        self.assertEqual(summary["uncovered_gold"], 1)
        self.assertEqual(rows[0]["n_uncovered_gold"], 1)
        self.assertEqual(rows[0]["uncovered_gold_rubrics"][0]["category"], "evidence_grounding")

        by_category = {row["category"]: row for row in category_rows}
        self.assertEqual(by_category["evidence_grounding"]["uncovered_gold"], 1)
        self.assertEqual(by_category["safety"]["covered_gold"], 1)


if __name__ == "__main__":
    unittest.main()
