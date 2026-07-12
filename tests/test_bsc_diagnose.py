from __future__ import annotations

import unittest
from pathlib import Path

from scripts.bsc_diagnose import build_record_verifier, summarize
from blindspot_rl.reward_bsc import TokenOverlapEmbedder, compute_metrics


class BscDiagnoseTest(unittest.TestCase):
    def test_summarize_keeps_auditable_metric_context(self) -> None:
        summary = summarize(
            rows=[
                {
                    "data_source": "rubricbench",
                    "coverage": 0.6,
                    "blind": 0.4,
                    "redundancy": 0.1,
                    "validity": 1.0,
                    "hallucination": 0.0,
                    "reward": 1.05,
                    "n_gold": 5,
                    "n_gen": 7,
                },
                {
                    "data_source": "researchrubrics",
                    "coverage": 0.8,
                    "blind": 0.2,
                    "redundancy": 0.0,
                    "validity": 0.5,
                    "hallucination": 0.5,
                    "reward": 1.15,
                    "n_gold": 4,
                    "n_gen": 5,
                },
            ],
            input_path=Path("data/processed/bsc_eval.jsonl"),
            embedding_model="BAAI/bge-large-en-v1.5",
            coverage_tau=0.75,
            redundancy_tau=0.85,
            weights=(1.0, 0.5, 0.5),
        )

        self.assertEqual(summary["input"], "data/processed/bsc_eval.jsonl")
        self.assertEqual(summary["embedding_model"], "BAAI/bge-large-en-v1.5")
        self.assertEqual(summary["coverage_tau"], 0.75)
        self.assertEqual(summary["redundancy_tau"], 0.85)
        self.assertEqual(summary["weights"], {"coverage": 1.0, "validity": 0.5, "redundancy": 0.5})
        self.assertEqual(summary["data_source_counts"], {"rubricbench": 1, "researchrubrics": 1})
        self.assertEqual(summary["verifier_source_counts"], {"none": 2})
        self.assertEqual(summary["verifier_source"], "none")
        self.assertAlmostEqual(summary["mean_coverage"], 0.7)
        self.assertAlmostEqual(summary["median_blind"], 0.30000000000000004)
        self.assertEqual(summary["queries_coverage_le_0_5"], 0)
        self.assertEqual(summary["queries_blind_ge_0_5"], 0)
        self.assertEqual(summary["queries_blind_ge_0_8"], 0)
        self.assertEqual(summary["queries_zero_coverage"], 0)
        self.assertEqual(summary["total_gold"], 9)
        self.assertEqual(summary["total_gen"], 12)
        self.assertEqual(summary["mean_n_gold"], 4.5)
        self.assertEqual(summary["mean_n_gen"], 6.0)
        self.assertAlmostEqual(summary["gen_to_gold_ratio"], 12 / 9)
        self.assertAlmostEqual(summary["coverage_per_generated_criterion"], 0.7 / 6.0)

    def test_summarize_reports_query_level_blindspot_distribution(self) -> None:
        summary = summarize(
            rows=[
                {
                    "data_source": "rubricbench",
                    "coverage": 0.0,
                    "blind": 1.0,
                    "redundancy": 0.0,
                    "validity": 1.0,
                    "hallucination": 0.0,
                    "reward": 0.5,
                    "n_gold": 3,
                    "n_gen": 2,
                },
                {
                    "data_source": "rubricbench",
                    "coverage": 0.5,
                    "blind": 0.5,
                    "redundancy": 0.0,
                    "validity": 1.0,
                    "hallucination": 0.0,
                    "reward": 1.0,
                    "n_gold": 4,
                    "n_gen": 3,
                },
                {
                    "data_source": "rubricbench",
                    "coverage": 0.75,
                    "blind": 0.25,
                    "redundancy": 0.0,
                    "validity": 1.0,
                    "hallucination": 0.0,
                    "reward": 1.25,
                    "n_gold": 4,
                    "n_gen": 4,
                },
            ]
        )

        self.assertEqual(summary["median_blind"], 0.5)
        self.assertEqual(summary["queries_coverage_le_0_5"], 2)
        self.assertEqual(summary["queries_blind_ge_0_5"], 2)
        self.assertEqual(summary["queries_blind_ge_0_8"], 1)
        self.assertEqual(summary["queries_zero_coverage"], 1)

    def test_record_valid_flags_drive_hallucination_metrics(self) -> None:
        record = {
            "response": ["match gold", "invalid rubric"],
            "valid_flags": [1, 0],
            "verifier_source": "valid_flags",
        }
        verifier, source = build_record_verifier(record, response=record["response"], idx=0)

        metrics = compute_metrics(
            response=record["response"],
            gold_rubrics=["match gold"],
            verifier=verifier,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )

        self.assertEqual(source, "valid_flags")
        self.assertEqual(metrics.validity, 0.5)
        self.assertEqual(metrics.hallucination, 0.5)

    def test_record_valid_flags_must_align_with_generated_rubrics(self) -> None:
        with self.assertRaises(ValueError):
            build_record_verifier({"valid_flags": [1]}, response=["criterion a", "criterion b"], idx=3)


if __name__ == "__main__":
    unittest.main()
