from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.budget_gate import file_sha256
from scripts.run_verifier_filter_ablation import (
    FILTERED_VARIANT,
    RAW_VARIANT,
    attach_protocol_metadata,
    main,
    run_ablation,
    summarize,
)


class VerifierFilterAblationTest(unittest.TestCase):
    def test_run_ablation_compares_raw_and_filtered_teacher_unions(self) -> None:
        gold_records = [{"query": "q", "gold_rubrics": ["must cite evidence", "must be concise"]}]
        raw_records = [
            {"query": "q", "teacher": "a", "rubrics": ["must cite evidence", "generic preference"]},
            {"query": "q", "teacher": "b", "rubrics": ["must be concise"]},
        ]
        filtered_records = [
            {"query": "q", "teacher": "a", "rubrics": ["must cite evidence"]},
            {"query": "q", "teacher": "b", "rubrics": ["must be concise"]},
        ]

        rows = run_ablation(
            raw_teacher_records=raw_records,
            filtered_teacher_records=filtered_records,
            gold_records=gold_records,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.5,
            redundancy_tau=0.9,
            dedupe_tau=0.95,
            min_teachers=2,
        )
        summary = summarize(rows)

        self.assertEqual([row["variant"] for row in rows], [RAW_VARIANT, FILTERED_VARIANT])
        self.assertEqual([row["variant"] for row in summary], [RAW_VARIANT, FILTERED_VARIANT])
        self.assertIn("coverage_delta_vs_no_verifier", summary[1])
        self.assertIn("hallucination_delta_vs_no_verifier", summary[1])
        self.assertAlmostEqual(
            summary[1]["coverage_delta_vs_no_verifier"],
            summary[1]["mean_coverage"] - summary[0]["mean_coverage"],
        )
        self.assertAlmostEqual(
            summary[1]["hallucination_delta_vs_no_verifier"],
            summary[1]["mean_hallucination"] - summary[0]["mean_hallucination"],
        )

    def test_attach_protocol_metadata_records_ablation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            filtered = root / "filtered.jsonl"
            gold = root / "gold.jsonl"
            raw.write_text('{"query":"q","rubrics":["raw"]}\n', encoding="utf-8")
            filtered.write_text('{"query":"q","rubrics":["filtered"]}\n', encoding="utf-8")
            gold.write_text('{"query":"q","gold_rubrics":["gold"]}\n', encoding="utf-8")
            summary = [{"variant": RAW_VARIANT}, {"variant": FILTERED_VARIANT}]

            attach_protocol_metadata(
                summary,
                raw_teachers=raw,
                filtered_teachers=filtered,
                gold=gold,
                embedding_model="token-overlap",
                coverage_tau=0.75,
                redundancy_tau=0.85,
                dedupe_tau=0.9,
                min_teachers=2,
            )

            for row in summary:
                self.assertEqual(row["ablation_family"], "verifier_filter")
                self.assertEqual(row["raw_teachers"], str(raw))
                self.assertEqual(row["raw_teachers_sha256"], file_sha256(raw))
                self.assertEqual(row["filtered_teachers"], str(filtered))
                self.assertEqual(row["filtered_teachers_sha256"], file_sha256(filtered))
                self.assertEqual(row["gold"], str(gold))
                self.assertEqual(row["gold_sha256"], file_sha256(gold))
                self.assertEqual(row["embedding_model"], "token-overlap")
                self.assertEqual(row["coverage_tau"], 0.75)
                self.assertEqual(row["redundancy_tau"], 0.85)
                self.assertEqual(row["dedupe_tau"], 0.9)
                self.assertEqual(row["min_teachers"], 2)

    def test_cli_writes_ablation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            filtered = root / "filtered.jsonl"
            gold = root / "gold.jsonl"
            output_dir = root / "out"
            gold.write_text(json.dumps({"query": "q", "gold_rubrics": ["must cite evidence"]}) + "\n", encoding="utf-8")
            raw.write_text(
                json.dumps({"query": "q", "teacher": "a", "rubrics": ["must cite evidence"]}) + "\n"
                + json.dumps({"query": "q", "teacher": "b", "rubrics": ["must cite evidence clearly"]}) + "\n",
                encoding="utf-8",
            )
            filtered.write_text(raw.read_text(encoding="utf-8"), encoding="utf-8")

            argv = [
                "run_verifier_filter_ablation.py",
                "--raw-teachers",
                str(raw),
                "--filtered-teachers",
                str(filtered),
                "--gold",
                str(gold),
                "--output-dir",
                str(output_dir),
                "--embedding-model",
                "token-overlap",
                "--coverage-tau",
                "0.5",
            ]
            with patch("sys.argv", argv):
                main()

            self.assertTrue((output_dir / "verifier_filter_per_item.csv").exists())
            self.assertTrue((output_dir / "verifier_filter_ablation.csv").exists())
            self.assertTrue((output_dir / "verifier_filter_ablation.md").exists())
            self.assertTrue((output_dir / "verifier_filter_ablation.json").exists())
            summary = json.loads((output_dir / "verifier_filter_ablation.json").read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["embedding_model"], "token-overlap")
            self.assertEqual(summary[0]["coverage_tau"], 0.5)


if __name__ == "__main__":
    unittest.main()
