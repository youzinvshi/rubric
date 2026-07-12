from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_bsc_human_audit_labels import gate_ok, summarize_rows


ROOT = Path(__file__).resolve().parents[1]


class SummarizeBSCHumanAuditLabelsTest(unittest.TestCase):
    def test_summarize_rows_reports_completed_human_audit(self) -> None:
        summary = summarize_rows(
            [
                {"match_status": "matched", "human_match_label": "match"},
                {"match_status": "matched", "human_match_label": "non_match"},
                {"match_status": "unmatched", "human_match_label": "non_match"},
                {"match_status": "unmatched", "human_match_label": "match"},
            ],
            input_path=Path("audit_items.csv"),
        )

        self.assertEqual(summary["status"], "human_audit_complete")
        self.assertEqual(summary["human_labels_completed"], 4)
        self.assertEqual(summary["invalid_label_count"], 0)
        self.assertAlmostEqual(summary["auto_matched_human_match_rate"], 0.5)
        self.assertAlmostEqual(summary["auto_unmatched_human_match_rate"], 0.5)
        self.assertAlmostEqual(summary["auto_unmatched_confirmation_rate"], 0.5)

    def test_summarize_rows_keeps_incomplete_status_for_empty_labels(self) -> None:
        summary = summarize_rows(
            [
                {"match_status": "matched", "human_match_label": ""},
                {"match_status": "unmatched", "human_match_label": "uncertain"},
            ],
            input_path=Path("audit_items.csv"),
        )

        self.assertEqual(summary["status"], "human_audit_incomplete")
        self.assertEqual(summary["human_labels_completed"], 1)
        self.assertEqual(summary["unlabeled_items"], 1)
        self.assertEqual(summary["uncertain_count"], 1)
        self.assertAlmostEqual(summary["uncertain_rate"], 1.0)

    def test_summarize_rows_reports_invalid_labels(self) -> None:
        summary = summarize_rows(
            [{"match_status": "matched", "human_match_label": "maybe"}],
            input_path=Path("audit_items.csv"),
        )

        self.assertEqual(summary["status"], "human_audit_incomplete")
        self.assertEqual(summary["human_labels_completed"], 0)
        self.assertEqual(summary["invalid_label_count"], 1)
        self.assertEqual(summary["invalid_labels"], ["maybe"])

    def test_gate_ok_applies_label_count_invalid_and_uncertain_thresholds(self) -> None:
        summary = {
            "human_labels_completed": 10,
            "invalid_label_count": 0,
            "uncertain_rate": 0.2,
        }

        self.assertTrue(gate_ok(summary, min_labeled=10, max_invalid_labels=0, max_uncertain_rate=0.2))
        self.assertFalse(gate_ok(summary, min_labeled=11, max_invalid_labels=0, max_uncertain_rate=0.2))
        self.assertFalse(gate_ok(summary, min_labeled=10, max_invalid_labels=0, max_uncertain_rate=0.1))

    def test_gate_ok_can_require_human_confirmation_rates(self) -> None:
        summary = {
            "human_labels_completed": 10,
            "invalid_label_count": 0,
            "uncertain_rate": 0.0,
            "auto_matched_human_match_rate": 0.8,
            "auto_unmatched_confirmation_rate": 0.6,
        }

        self.assertTrue(
            gate_ok(
                summary,
                min_labeled=10,
                max_invalid_labels=0,
                max_uncertain_rate=0.0,
                min_auto_matched_human_match_rate=0.8,
                min_auto_unmatched_confirmation_rate=0.6,
            )
        )
        self.assertFalse(
            gate_ok(
                summary,
                min_labeled=10,
                max_invalid_labels=0,
                max_uncertain_rate=0.0,
                min_auto_matched_human_match_rate=0.9,
                min_auto_unmatched_confirmation_rate=0.6,
            )
        )
        self.assertFalse(
            gate_ok(
                summary,
                min_labeled=10,
                max_invalid_labels=0,
                max_uncertain_rate=0.0,
                min_auto_matched_human_match_rate=0.8,
                min_auto_unmatched_confirmation_rate=0.7,
            )
        )

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "audit_items.csv"
            output_json = root / "summary.json"
            output_md = root / "summary.md"
            write_csv(
                input_csv,
                [
                    {"match_status": "matched", "human_match_label": "match"},
                    {"match_status": "unmatched", "human_match_label": "non_match"},
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_bsc_human_audit_labels.py"),
                    "--input",
                    str(input_csv),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                    "--min-labeled",
                    "2",
                    "--min-auto-matched-human-match-rate",
                    "1.0",
                    "--min-auto-unmatched-confirmation-rate",
                    "1.0",
                    "--strict",
                ],
                cwd=ROOT,
                check=True,
            )

            summary = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["status"], "human_audit_complete")
        self.assertEqual(summary["gate"]["min_auto_matched_human_match_rate"], 1.0)
        self.assertEqual(summary["gate"]["min_auto_unmatched_confirmation_rate"], 1.0)
        self.assertIn("BSC Human Audit Label Summary", markdown)

    def test_cli_strict_fails_when_labels_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "audit_items.csv"
            output_json = root / "summary.json"
            write_csv(input_csv, [{"match_status": "matched", "human_match_label": ""}])

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_bsc_human_audit_labels.py"),
                    "--input",
                    str(input_csv),
                    "--output-json",
                    str(output_json),
                    "--min-labeled",
                    "1",
                    "--strict",
                ],
                cwd=ROOT,
                check=False,
            )

            summary = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(summary["ok"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_status", "human_match_label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
