from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts.run_bsc_ablation import main, summarize


class RunBSCAblationTest(unittest.TestCase):
    def test_summarize_reports_verifier_source_counts(self) -> None:
        summary = summarize(
            [
                {
                    "coverage": 1.0,
                    "blind": 0.0,
                    "redundancy": 0.0,
                    "validity": 0.5,
                    "hallucination": 0.5,
                    "reward": 1.25,
                    "verifier_source": "valid_flags",
                }
            ]
        )

        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["verifier_source"], "valid_flags")
        self.assertEqual(summary["verifier_source_counts"], {"valid_flags": 1})
        self.assertAlmostEqual(summary["mean_hallucination"], 0.5)

    def test_main_writes_component_weights_for_each_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            output_dir = root / "ablation"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "gold_rubrics": ["Use evidence", "Follow JSON format"],
                        "response": ["Use evidence", "Follow JSON format"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected_input_sha256 = file_sha256(input_path)
            argv = [
                "run_bsc_ablation.py",
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--embedding-model",
                "token-overlap",
            ]

            with patch("sys.argv", argv):
                main()

            expected = {
                "full": {"coverage": 1.0, "validity": 0.5, "redundancy": 0.5},
                "no_red": {"coverage": 1.0, "validity": 0.5, "redundancy": 0.0},
                "no_valid": {"coverage": 1.0, "validity": 0.0, "redundancy": 0.5},
                "no_verifier": {"coverage": 1.0, "validity": 0.5, "redundancy": 0.5},
                "cov_only": {"coverage": 1.0, "validity": 0.0, "redundancy": 0.0},
            }
            for variant, weights in expected.items():
                summary = json.loads(
                    (output_dir / "variants" / f"{variant}_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(summary["variant"], variant)
                self.assertEqual(summary["weights"], weights)
                self.assertEqual(summary["ablation_family"], "reward_component")
                self.assertEqual(summary["input"], str(input_path))
                self.assertEqual(summary["input_sha256"], expected_input_sha256)
                self.assertEqual(summary["embedding_model"], "token-overlap")
                self.assertEqual(summary["coverage_tau"], 0.75)
                self.assertEqual(summary["redundancy_tau"], 0.85)
                self.assertEqual(summary["use_verifier"], variant in {"full", "no_red"})

            no_verifier = json.loads(
                (output_dir / "variants" / "no_verifier_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(no_verifier["verifier_source"], "disabled")


if __name__ == "__main__":
    unittest.main()
