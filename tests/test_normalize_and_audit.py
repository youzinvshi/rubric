from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.audit_experiment import audit_manifest, load_manifest, main as audit_main
from scripts.audit_raw_data import assert_contamination_policy
from scripts.normalize_dataset import comparative_winner, get_path, normalize_records


class NormalizeAndAuditTest(unittest.TestCase):
    def test_normalize_gold_records(self) -> None:
        args = Namespace(
            target="gold",
            data_source="toy",
            query_key=None,
            gold_key=None,
            chosen_key=None,
            rejected_key=None,
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance=None,
            source_url=None,
            paper_url=None,
            dataset_version=None,
            license=None,
            split=None,
            limit=None,
            dedupe_query=True,
        )
        rows = normalize_records(
            [{"prompt": "q1", "rubrics": ["a", "a", "b"], "dataset_version": "v1"}],
            args,
        )
        self.assertEqual(
            rows,
            [{"query": "q1", "gold_rubrics": ["a", "b"], "data_source": "toy", "dataset_version": "v1"}],
        )

    def test_normalize_gold_can_inject_static_provenance(self) -> None:
        args = Namespace(
            target="gold",
            data_source="rubricbench",
            query_key=None,
            gold_key=None,
            chosen_key=None,
            rejected_key=None,
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance="official_release",
            source_url="https://example.com/rubricbench.jsonl",
            paper_url="https://arxiv.org/abs/2603.01562",
            dataset_version="2026-03",
            license="research",
            split="test",
            limit=None,
            dedupe_query=True,
        )
        rows = normalize_records(
            [{"prompt": "q1", "rubrics": ["a"]}],
            args,
        )

        self.assertEqual(rows[0]["provenance"], "official_release")
        self.assertEqual(rows[0]["source_url"], "https://example.com/rubricbench.jsonl")
        self.assertEqual(rows[0]["paper_url"], "https://arxiv.org/abs/2603.01562")
        self.assertEqual(rows[0]["dataset_version"], "2026-03")
        self.assertEqual(rows[0]["license"], "research")
        self.assertEqual(rows[0]["split"], "test")

    def test_normalize_preference_records(self) -> None:
        args = Namespace(
            target="preference",
            data_source="pref",
            query_key=None,
            gold_key=None,
            chosen_key=None,
            rejected_key=None,
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance=None,
            source_url=None,
            paper_url=None,
            dataset_version=None,
            license=None,
            split=None,
            limit=None,
            dedupe_query=False,
        )
        rows = normalize_records(
            [{"prompt": "q1", "chosen": "good", "rejected": "bad"}],
            args,
        )
        self.assertEqual(
            rows,
            [{"query": "q1", "chosen": "good", "rejected": "bad", "data_source": "pref"}],
        )

    def test_normalize_comparative_pair_preference(self) -> None:
        args = Namespace(
            target="preference",
            data_source="judgebench",
            query_key=None,
            gold_key=None,
            chosen_key=None,
            rejected_key=None,
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance=None,
            source_url=None,
            paper_url=None,
            dataset_version=None,
            license=None,
            split=None,
            limit=None,
            dedupe_query=False,
        )
        rows = normalize_records(
            [
                {"question": "q1", "response_A": "good", "response_B": "bad", "label": "A>B"},
                {"question": "q2", "response_A": "bad", "response_B": "good", "label": "B>A"},
                {"question": "q3", "response_A": "x", "response_B": "y", "label": "A=B"},
            ],
            args,
        )
        self.assertEqual(
            rows,
            [
                {"query": "q1", "chosen": "good", "rejected": "bad", "data_source": "judgebench"},
                {"query": "q2", "chosen": "good", "rejected": "bad", "data_source": "judgebench"},
            ],
        )

    def test_comparative_winner_handles_directions_and_ambiguity(self) -> None:
        self.assertEqual(comparative_winner("A>B"), "A")
        self.assertEqual(comparative_winner("B>A"), "B")
        self.assertEqual(comparative_winner("a > b"), "A")
        self.assertIsNone(comparative_winner("A=B"))
        self.assertIsNone(comparative_winner("tie"))
        self.assertIsNone(comparative_winner(None))

    def test_contamination_policy_passes_for_valid_manifest(self) -> None:
        manifest = {
            "rubricbench": {"gold_type": "human_gold", "allowed_in_main_bsc_eval": True, "has_hard_gold": True},
            "rewardbench": {"gold_type": "proxy_teacher", "allowed_in_main_bsc_eval": False, "has_hard_gold": False},
        }
        assert_contamination_policy(manifest)  # should not raise

    def test_contamination_policy_blocks_proxy_in_main_eval(self) -> None:
        manifest = {
            "rewardbench": {"gold_type": "proxy_teacher", "allowed_in_main_bsc_eval": True, "has_hard_gold": False},
        }
        with self.assertRaises(ValueError) as ctx:
            assert_contamination_policy(manifest)
        self.assertIn("proxy_teacher", str(ctx.exception))

    def test_contamination_policy_blocks_hard_gold_without_human_gold_type(self) -> None:
        manifest = {
            "weird": {"gold_type": "proxy_teacher", "allowed_in_main_bsc_eval": False, "has_hard_gold": True},
        }
        with self.assertRaises(ValueError) as ctx:
            assert_contamination_policy(manifest)
        self.assertIn("human_gold", str(ctx.exception))


    def test_normalize_nested_field_paths(self) -> None:
        args = Namespace(
            target="preference",
            data_source="nested",
            query_key="conversation.0.content",
            gold_key=None,
            chosen_key="answers.chosen.text",
            rejected_key="answers.rejected.text",
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance=None,
            source_url=None,
            paper_url=None,
            dataset_version=None,
            license=None,
            split=None,
            limit=None,
            dedupe_query=False,
        )
        rows = normalize_records(
            [
                {
                    "conversation": [{"content": "q1"}],
                    "answers": {"chosen": {"text": "good"}, "rejected": {"text": "bad"}},
                }
            ],
            args,
        )
        self.assertEqual(rows[0]["query"], "q1")
        self.assertEqual(rows[0]["chosen"], "good")
        self.assertEqual(get_path({"a": [{"b": 1}]}, "a.0.b"), 1)

    def test_normalize_multicandidate_records(self) -> None:
        args = Namespace(
            target="multicandidate",
            data_source="rb2",
            query_key=None,
            gold_key=None,
            chosen_key=None,
            rejected_key=None,
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance=None,
            source_url=None,
            paper_url=None,
            dataset_version=None,
            license=None,
            split=None,
            limit=None,
            dedupe_query=False,
        )
        rows = normalize_records(
            [{"prompt": "q1", "responses": [{"text": "bad"}, {"text": "good"}], "correct_index": 1}],
            args,
        )
        self.assertEqual(
            rows,
            [{"query": "q1", "candidates": ["bad", "good"], "label": 1, "data_source": "rb2"}],
        )

    def test_normalize_multicandidate_from_split_lists(self) -> None:
        args = Namespace(
            target="multicandidate",
            data_source="rb2",
            query_key=None,
            gold_key=None,
            chosen_key=None,
            rejected_key=None,
            candidates_key=None,
            label_key=None,
            provenance_key=None,
            provenance=None,
            source_url=None,
            paper_url=None,
            dataset_version=None,
            license=None,
            split=None,
            limit=None,
            dedupe_query=False,
        )
        rows = normalize_records(
            [{"prompt": "q1", "chosen": ["gold"], "rejected": ["bad1", "bad2", "bad3"]}],
            args,
        )
        self.assertEqual(
            rows,
            [
                {
                    "query": "q1",
                    "candidates": ["gold", "bad1", "bad2", "bad3"],
                    "label": 0,
                    "data_source": "rb2",
                }
            ],
        )

    def test_audit_manifest_passes_for_present_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            summary.write_text('{"n": 1, "accuracy": 1.0}', encoding="utf-8")
            manifest = {
                "required_files": ["summary.json"],
                "summaries": [
                    {"name": "s", "path": "summary.json", "required_keys": ["n", "accuracy"]}
                ],
            }
            report = audit_manifest(manifest, root=root)
            self.assertTrue(report["ok"])

    def test_load_manifest_reports_missing_invalid_and_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, error = load_manifest(root / "missing.json")
            self.assertIn("manifest is missing", error)

            bad = root / "bad.json"
            bad.write_text("{bad", encoding="utf-8")
            _, error = load_manifest(bad)
            self.assertIn("manifest is not valid JSON", error)

            list_path = root / "list.json"
            list_path.write_text("[]", encoding="utf-8")
            _, error = load_manifest(list_path)
            self.assertIn("manifest must be a JSON object", error)

    def test_audit_manifest_reports_invalid_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("{bad", encoding="utf-8")
            manifest = {
                "required_files": ["summary.json"],
                "summaries": [{"name": "s", "path": "summary.json", "required_keys": ["n"]}],
            }
            report = audit_manifest(manifest, root=root)

        self.assertFalse(report["ok"])
        self.assertIn("summary is not valid JSON", report["summary_checks"][0]["error"])

    def test_audit_manifest_reports_non_object_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("[]", encoding="utf-8")
            manifest = {
                "required_files": ["summary.json"],
                "summaries": [{"name": "s", "path": "summary.json", "required_keys": ["n"]}],
            }
            report = audit_manifest(manifest, root=root)

        self.assertFalse(report["ok"])
        self.assertEqual(report["summary_checks"][0]["error"], "summary must be a JSON object")

    def test_audit_main_non_strict_writes_failed_report_without_exiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            output = root / "audit.json"
            manifest.write_text('{"required_files": ["missing.json"], "summaries": []}', encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "audit_experiment.py",
                    "--manifest",
                    str(manifest),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--non-strict",
                ],
            ):
                audit_main()

            self.assertTrue(output.exists())

    def test_audit_main_defaults_to_strict_failed_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            output = root / "audit.json"
            manifest.write_text('{"required_files": ["missing.json"], "summaries": []}', encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "audit_experiment.py",
                    "--manifest",
                    str(manifest),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
            ), self.assertRaises(SystemExit):
                audit_main()

            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
