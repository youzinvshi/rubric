from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.build_bsc_human_audit_pack import build_candidates, sample_candidates, summarize


class BuildBSCHumanAuditPackTest(unittest.TestCase):
    def test_build_candidates_marks_matched_and_unmatched_gold_dimensions(self) -> None:
        rows = build_candidates(
            records=[
                {
                    "query": "q1",
                    "gold_rubrics": ["mentions sunlight", "mentions oxygen"],
                    "response": ["mentions sunlight"],
                }
            ],
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )

        self.assertEqual([row["match_status"] for row in rows], ["matched", "unmatched"])
        self.assertEqual(rows[0]["best_generated_rubric"], "mentions sunlight")
        self.assertEqual(rows[1]["best_generated_rubric"], "mentions sunlight")
        self.assertEqual(rows[0]["human_match_label"], "")
        self.assertEqual(rows[0]["human_notes"], "")

    def test_build_candidates_handles_empty_generated_rubrics_as_unmatched(self) -> None:
        rows = build_candidates(
            records=[
                {
                    "query": "q1",
                    "gold_rubrics": ["mentions sunlight"],
                    "response": [],
                }
            ],
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.75,
        )

        self.assertEqual(rows[0]["match_status"], "unmatched")
        self.assertEqual(rows[0]["best_generated_idx"], -1)
        self.assertEqual(rows[0]["best_generated_rubric"], "")
        self.assertEqual(rows[0]["similarity"], 0.0)

    def test_sample_candidates_is_deterministic_and_balanced_by_status(self) -> None:
        candidates = [
            {"match_status": "matched", "record_idx": 2, "gold_idx": 0},
            {"match_status": "matched", "record_idx": 1, "gold_idx": 0},
            {"match_status": "unmatched", "record_idx": 4, "gold_idx": 0},
            {"match_status": "unmatched", "record_idx": 3, "gold_idx": 0},
        ]

        sampled_a = sample_candidates(candidates, matched=1, unmatched=1, seed=13)
        sampled_b = sample_candidates(candidates, matched=1, unmatched=1, seed=13)

        self.assertEqual(sampled_a, sampled_b)
        self.assertEqual(len(sampled_a), 2)
        self.assertEqual({row["match_status"] for row in sampled_a}, {"matched", "unmatched"})

    def test_summarize_keeps_annotation_pack_status_pending_human_labels(self) -> None:
        summary = summarize(
            records=[{"query": "q1"}],
            candidates=[
                {"match_status": "matched"},
                {"match_status": "unmatched"},
            ],
            sampled=[{"match_status": "matched"}],
            input_path=Path("bsc_eval.jsonl"),
            embedding_model="token-overlap",
            coverage_tau=0.75,
            seed=13,
        )

        self.assertEqual(summary["status"], "annotation_pack_ready")
        self.assertEqual(summary["human_labels_completed"], 0)
        self.assertEqual(summary["total_gold_dimensions"], 2)
        self.assertEqual(summary["total_unmatched_candidates"], 1)
        self.assertEqual(summary["sampled_matched"], 1)
        self.assertEqual(summary["sampled_unmatched"], 0)

    def test_cli_writes_csv_jsonl_and_summary(self) -> None:
        from scripts.build_bsc_human_audit_pack import main
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_dir = root / "audit"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q1",
                        "gold_rubrics": ["mentions sunlight", "mentions oxygen"],
                        "response": ["mentions sunlight"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_bsc_human_audit_pack.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--embedding-model",
                    "token-overlap",
                    "--matched",
                    "1",
                    "--unmatched",
                    "1",
                ]
                main()
            finally:
                sys.argv = old_argv

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            rows = list(csv.DictReader((output_dir / "audit_items.csv").open(encoding="utf-8")))
            jsonl_exists = (output_dir / "audit_items.jsonl").exists()

        self.assertEqual(summary["sampled_total"], 2)
        self.assertEqual(len(rows), 2)
        self.assertIn("human_match_label", rows[0])
        self.assertTrue(jsonl_exists)


if __name__ == "__main__":
    unittest.main()
