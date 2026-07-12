from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.sweep_bsc_thresholds import run_one_setting, to_markdown


class SweepBSCThresholdsTest(unittest.TestCase):
    def test_run_one_setting_summarizes_metrics(self) -> None:
        records = [
            {
                "query": "q",
                "gold_rubrics": ["mentions sunlight", "oxygen"],
                "response": ["mentions sunlight"],
                "valid_flags": [1],
            }
        ]
        summary = run_one_setting(
            records=records,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.5,
            redundancy_tau=0.9,
            weights=(1.0, 0.5, 0.5),
        )
        self.assertEqual(summary["n"], 1)
        self.assertAlmostEqual(summary["mean_coverage"], 0.5)
        self.assertAlmostEqual(summary["mean_blind"], 0.5)
        self.assertEqual(summary["verifier_source"], "valid_flags")
        self.assertEqual(summary["verifier_source_counts"], {"valid_flags": 1})

    def test_run_one_setting_uses_valid_flags_for_hallucination(self) -> None:
        summary = run_one_setting(
            records=[
                {
                    "query": "q",
                    "gold_rubrics": ["mentions sunlight"],
                    "response": ["mentions sunlight", "generic quality"],
                    "valid_flags": [1, 0],
                }
            ],
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.5,
            redundancy_tau=0.9,
            weights=(1.0, 0.5, 0.5),
        )

        self.assertAlmostEqual(summary["mean_validity"], 0.5)
        self.assertAlmostEqual(summary["mean_hallucination"], 0.5)

    def test_to_markdown_contains_thresholds(self) -> None:
        text = to_markdown(
            [
                {
                    "coverage_tau": 0.5,
                    "redundancy_tau": 0.9,
                    "mean_coverage": 1.0,
                    "mean_blind": 0.0,
                    "mean_redundancy": 0.0,
                    "mean_hallucination": 0.0,
                    "mean_reward": 1.5,
                }
            ]
        )
        self.assertIn("| 0.5000 | 0.9000 | 1.0000 |", text)


if __name__ == "__main__":
    unittest.main()
