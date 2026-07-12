from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_bsc import evaluate_bsc, file_sha256, main


class EvaluateBSCTest(unittest.TestCase):
    def test_evaluate_bsc_reports_auditable_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "test_main.jsonl"
            predictions = root / "predictions.jsonl"
            gold.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query": "q1",
                                "gold_rubrics": ["match evidence", "check final answer"],
                            }
                        ),
                        json.dumps(
                            {
                                "query": "q2",
                                "gold_rubrics": ["safety policy"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query": "q1",
                                "rubrics": ["match evidence", "extra dimension"],
                                "valid_flags": [1, 0],
                                "verifier_source": "valid_flags",
                            }
                        ),
                        json.dumps(
                            {
                                "query": "q2",
                                "rubrics": ["safety policy"],
                                "valid_flags": [1],
                                "verifier_source": "valid_flags",
                            }
                        ),
                        json.dumps({"query": "unmatched", "rubrics": ["ignored"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary, per_item = evaluate_bsc(
                test_split=gold,
                predictions=predictions,
                embedding_model="token-overlap",
                coverage_tau=0.99,
                redundancy_tau=0.99,
            )
            expected_predictions_sha = file_sha256(predictions)
            expected_gold_sha = file_sha256(gold)

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["matched_samples"], 2)
        self.assertEqual(summary["matched_queries"], 2)
        self.assertEqual(summary["input"], str(predictions))
        self.assertEqual(summary["input_sha256"], expected_predictions_sha)
        self.assertEqual(summary["test_split"], str(gold))
        self.assertEqual(summary["test_split_sha256"], expected_gold_sha)
        self.assertEqual(summary["predictions"], str(predictions))
        self.assertEqual(summary["predictions_sha256"], expected_predictions_sha)
        self.assertEqual(summary["embedding_model"], "token-overlap")
        self.assertEqual(summary["coverage_tau"], 0.99)
        self.assertEqual(summary["redundancy_tau"], 0.99)
        self.assertEqual(summary["weights"], {"coverage": 1.0, "validity": 0.5, "redundancy": 0.5})
        self.assertEqual(summary["data_source_counts"], {"rubricbench": 2})
        self.assertEqual(summary["verifier_source_counts"], {"valid_flags": 2})
        self.assertEqual(summary["verifier_source"], "valid_flags")
        self.assertAlmostEqual(summary["mean_coverage"], 0.75)
        self.assertAlmostEqual(summary["mean_blind"], 0.25)
        self.assertAlmostEqual(summary["mean_validity"], 0.75)
        self.assertAlmostEqual(summary["mean_hallucination"], 0.25)
        self.assertAlmostEqual(summary["mean_blindspot_rate"], summary["mean_blind"])
        self.assertEqual(summary["total_gold"], 3)
        self.assertEqual(summary["total_gen"], 3)
        self.assertEqual(summary["mean_n_gold"], 1.5)
        self.assertEqual(summary["mean_n_gen"], 1.5)
        self.assertEqual(summary["gen_to_gold_ratio"], 1.0)
        self.assertAlmostEqual(summary["coverage_per_generated_criterion"], 0.5)
        self.assertEqual(len(per_item), 2)

    def test_main_writes_summary_and_per_item_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "test_main.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "summary.json"
            per_item = root / "per_item.csv"
            gold.write_text(
                json.dumps({"query": "q1", "gold_rubrics": ["match evidence"]}) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps({"query": "q1", "rubrics": ["match evidence"]}) + "\n",
                encoding="utf-8",
            )

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "evaluate_bsc.py",
                    "--test-split",
                    str(gold),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--per-item-output",
                    str(per_item),
                    "--embedding-model",
                    "token-overlap",
                    "--tau",
                    "0.99",
                ]
                main()
            finally:
                sys.argv = old_argv

            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["mean_blind"], 0.0)
            self.assertEqual(summary["coverage_tau"], 0.99)
            self.assertEqual(summary["per_item_output"], str(per_item))
            self.assertEqual(summary["per_item_sha256"], file_sha256(per_item))
            self.assertEqual(summary["per_item_rows"], 1)
            self.assertIn("query,data_source,verifier_source,coverage,blind,redundancy", per_item.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
