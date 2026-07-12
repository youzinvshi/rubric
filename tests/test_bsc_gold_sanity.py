from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_bsc_gold_sanity import run_sanity

ROOT = Path(__file__).resolve().parents[1]


class BscGoldSanityTest(unittest.TestCase):
    def test_real_rubricbench_gold_as_prediction_reaches_full_coverage(self) -> None:
        gold = ROOT / "data" / "processed" / "rubricbench_gold.jsonl"
        self.assertTrue(gold.exists(), "rubricbench_gold.jsonl must exist for the offline BSC sanity check")

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_sanity(
                gold_path=gold,
                output_dir=Path(tmp),
                min_joined=25,
                limit=25,
            )

            self.assertTrue(summary["ok"])
            self.assertEqual(summary["sanity_check"], "gold_as_prediction")
            self.assertEqual(summary["n"], 25)
            self.assertEqual(summary["data_source_counts"], {"rubricbench": 25})
            self.assertEqual(summary["verifier_source_counts"], {"gold_as_prediction": 25})
            self.assertEqual(summary["mean_coverage"], 1.0)
            self.assertEqual(summary["mean_blind"], 0.0)
            self.assertEqual(summary["mean_validity"], 1.0)
            self.assertEqual(summary["mean_hallucination"], 0.0)
            self.assertTrue((Path(tmp) / "gold_as_prediction.jsonl").exists())
            self.assertTrue((Path(tmp) / "bsc_eval.jsonl").exists())
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "per_item.csv").exists())

    def test_sanity_writes_blocked_join_report_when_min_joined_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold.jsonl"
            gold.write_text(
                json.dumps({"query": "q1", "gold_rubrics": ["criterion one"]}) + "\n",
                encoding="utf-8",
            )

            summary = run_sanity(
                gold_path=gold,
                output_dir=root / "out",
                min_joined=2,
            )
            join_report = json.loads((root / "out" / "join_report.json").read_text(encoding="utf-8"))

            self.assertFalse(summary["ok"])
            self.assertFalse(join_report["ok"])
            self.assertEqual(join_report["n_joined"], 1)
            self.assertIn("joined records below required minimum", join_report["blockers"][0])


if __name__ == "__main__":
    unittest.main()
