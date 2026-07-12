from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_minimal_bsc_chain_smoke import run_smoke
import scripts.run_minimal_bsc_chain_smoke as minimal_bsc_chain_smoke

ROOT = Path(__file__).resolve().parents[1]


class MinimalBSCChainSmokeTest(unittest.TestCase):
    def test_smoke_docstring_uses_criteria_framing(self) -> None:
        doc = minimal_bsc_chain_smoke.__doc__ or ""

        self.assertIn("hard-gold evaluation dimensions", doc)
        self.assertIn("generated-criteria validation", doc)
        self.assertIn("verified-criteria validation", doc)
        self.assertNotIn("hard-gold rubrics", doc)
        self.assertNotIn("generated-rubric validation", doc)

    def test_real_rubricbench_gold_runs_post_generation_chain(self) -> None:
        gold = ROOT / "data" / "processed" / "rubricbench_gold.jsonl"
        self.assertTrue(gold.exists())

        with tempfile.TemporaryDirectory() as tmp:
            report = run_smoke(gold_path=gold, output_dir=Path(tmp), limit=12, min_joined=12)
            root = Path(tmp)

            self.assertTrue(report["ok"])
            self.assertEqual(report["smoke"], "minimal_bsc_post_generation_chain")
            self.assertEqual(report["n_joined"], 12)
            self.assertEqual(report["bsc"]["mean_coverage"], 1.0)
            self.assertEqual(report["bsc"]["mean_blind"], 0.0)
            self.assertEqual(report["sweep_settings"], 9)
            self.assertEqual(report["ci_metrics"], 5)
            self.assertEqual(report["main_table_rows"], 1)
            self.assertTrue((root / "data" / "model_rubrics.jsonl").exists())
            self.assertTrue((root / "validation" / "model_rubrics" / "validation_report.json").exists())
            self.assertTrue((root / "data" / "model_rubrics_verified.jsonl").exists())
            self.assertTrue((root / "bsc" / "summary.json").exists())
            self.assertTrue((root / "bsc_sweep" / "threshold_sweep.json").exists())
            self.assertTrue((root / "bsc_ci" / "bootstrap_ci.json").exists())
            self.assertTrue((root / "main_table.csv").exists())
            self.assertTrue((root / "smoke_report.json").exists())


if __name__ == "__main__":
    unittest.main()
