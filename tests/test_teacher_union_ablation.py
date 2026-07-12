from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.run_teacher_union_ablation import UNION_VARIANT, main, run_ablation, summarize_by_variant


class TeacherUnionAblationTest(unittest.TestCase):
    def test_run_ablation_compares_single_teachers_and_union(self) -> None:
        teacher_records = [
            {"query": "q1", "teacher": "teacher_a", "rubrics": ["alpha"]},
            {"query": "q1", "teacher": "teacher_b", "rubrics": ["beta"]},
        ]
        gold_records = [{"query": "q1", "gold_rubrics": ["alpha", "beta"]}]
        rows = run_ablation(
            teacher_records=teacher_records,
            gold_records=gold_records,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
            redundancy_tau=0.99,
            dedupe_tau=0.99,
        )
        by_variant = {row["variant"]: row for row in rows}
        self.assertEqual(by_variant["teacher_a"]["coverage"], 0.5)
        self.assertEqual(by_variant["teacher_b"]["coverage"], 0.5)
        self.assertEqual(by_variant[UNION_VARIANT]["coverage"], 1.0)

    def test_summarize_by_variant_reports_union_gain(self) -> None:
        rows = [
            {"variant": "a", "coverage": 0.5, "blind": 0.5, "redundancy": 0, "validity": 1, "hallucination": 0, "reward": 1, "n_gen": 1},
            {"variant": UNION_VARIANT, "coverage": 1.0, "blind": 0, "redundancy": 0, "validity": 1, "hallucination": 0, "reward": 1.5, "n_gen": 2},
        ]
        summary = summarize_by_variant(rows)
        union = next(row for row in summary if row["variant"] == UNION_VARIANT)
        self.assertEqual(union["coverage_gain_vs_best_single"], 0.5)
        self.assertEqual(union["n_single_teacher_variants"], 1)
        self.assertEqual(union["best_single_variant"], "a")

    def test_main_writes_protocol_metadata_for_evidence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teachers = root / "teachers.jsonl"
            gold = root / "gold.jsonl"
            output_dir = root / "out"
            teachers.write_text(
                json.dumps({"query": "q", "teacher": "teacher_a", "rubrics": ["alpha"]}) + "\n"
                + json.dumps({"query": "q", "teacher": "teacher_b", "rubrics": ["beta"]}) + "\n",
                encoding="utf-8",
            )
            gold.write_text(
                json.dumps({"query": "q", "gold_rubrics": ["alpha", "beta"]}) + "\n",
                encoding="utf-8",
            )
            argv = [
                "run_teacher_union_ablation.py",
                "--teachers",
                str(teachers),
                "--gold",
                str(gold),
                "--output-dir",
                str(output_dir),
                "--embedding-model",
                "token-overlap",
                "--coverage-tau",
                "0.99",
                "--redundancy-tau",
                "0.88",
                "--dedupe-tau",
                "0.77",
                "--min-teachers",
                "2",
            ]

            with patch("sys.argv", argv):
                main()

            rows = json.loads((output_dir / "teacher_union_ablation.json").read_text(encoding="utf-8"))
            union = rows[0]
            self.assertEqual(union["variant"], UNION_VARIANT)
            self.assertEqual(union["teachers"], str(teachers))
            self.assertEqual(union["gold"], str(gold))
            self.assertEqual(union["embedding_model"], "token-overlap")
            self.assertEqual(union["coverage_tau"], 0.99)
            self.assertEqual(union["redundancy_tau"], 0.88)
            self.assertEqual(union["dedupe_tau"], 0.77)
            self.assertEqual(union["min_teachers"], 2)
            self.assertEqual(union["n_single_teacher_variants"], 2)


if __name__ == "__main__":
    unittest.main()
