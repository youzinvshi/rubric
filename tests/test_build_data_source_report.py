from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_data_source_report import build_invalid_config_report, build_missing_config_report, build_report, to_markdown


class BuildDataSourceReportTest(unittest.TestCase):
    def test_missing_local_config_builds_blocked_report(self) -> None:
        report = build_missing_config_report(Path("configs/data_sources_real.local.json"))

        self.assertEqual(report["overall_status"], "blocked")
        self.assertIn("data source config is missing", report["blockers"][0])
        self.assertIn("configs/data_sources_real.template.json", report["next_actions"][0])

    def test_invalid_local_config_builds_blocked_report(self) -> None:
        try:
            json.loads("{bad")
        except json.JSONDecodeError as exc:
            report = build_invalid_config_report(Path("configs/data_sources_real.local.json"), exc)
        else:  # pragma: no cover
            self.fail("expected JSONDecodeError")

        self.assertEqual(report["overall_status"], "blocked")
        self.assertIn("not valid JSON", report["blockers"][0])
        self.assertIn("Fix JSON syntax", report["next_actions"][0])

    def test_manual_missing_source_blocks_report(self) -> None:
        report = build_report(
            {
                "datasets": [
                    {
                        "name": "rubricbench",
                        "source": {
                            "type": "manual",
                            "raw_path": "/tmp/missing_rubricbench.jsonl",
                            "official_url": "https://example.com/rubricbench",
                        },
                        "expected_fields": ["query", "gold_rubrics"],
                        "normalizations": [
                            {"target": "gold", "output": "/tmp/missing_gold.jsonl"},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(report["overall_status"], "blocked")
        self.assertIn("manual raw source is missing", report["blockers"][0])
        self.assertIn("Download rubricbench official release", report["next_actions"][0])
        self.assertTrue(any("download_public_data.py --url https://example.com/rubricbench" in item for item in report["next_actions"]))

    def test_manual_hard_gold_requires_official_url_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            raw.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {
                                "type": "manual",
                                "raw_path": str(raw),
                                "official_url": "",
                                "require_official_url": True,
                            },
                            "normalizations": [],
                        }
                    ]
                }
            )

        self.assertEqual(report["overall_status"], "blocked")
        self.assertTrue(any("requires official_url" in item for item in report["blockers"]))
        self.assertTrue(any("Set rubricbench official_url" in item for item in report["next_actions"]))

    def test_manual_hard_gold_blocks_placeholder_official_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            raw.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {
                                "type": "manual",
                                "raw_path": str(raw),
                                "official_url": "https://example.com/rubricbench",
                                "require_official_url": True,
                            },
                            "normalizations": [],
                        }
                    ]
                }
            )

        self.assertEqual(report["overall_status"], "blocked")
        self.assertTrue(any("official_url must be" in item for item in report["blockers"]))
        self.assertTrue(any("Replace rubricbench official_url" in item for item in report["next_actions"]))

    def test_manual_hard_gold_requires_raw_sha256_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            raw.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {
                                "type": "manual",
                                "raw_path": str(raw),
                                "official_url": "https://example.com/rubricbench",
                                "require_raw_sha256": True,
                                "raw_sha256": "",
                            },
                            "normalizations": [],
                        }
                    ]
                }
            )

        self.assertEqual(report["overall_status"], "blocked")
        self.assertTrue(any("requires raw_sha256" in item for item in report["blockers"]))
        self.assertTrue(any("Set rubricbench raw_sha256" in item for item in report["next_actions"]))

    def test_manual_hard_gold_blocks_invalid_raw_sha256_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            raw.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {
                                "type": "manual",
                                "raw_path": str(raw),
                                "official_url": "https://huggingface.co/datasets/org/rubricbench",
                                "require_raw_sha256": True,
                                "raw_sha256": "not-a-sha",
                            },
                            "normalizations": [],
                        }
                    ]
                }
            )

        self.assertEqual(report["overall_status"], "blocked")
        self.assertTrue(any("64-character hex digest" in item for item in report["blockers"]))
        self.assertTrue(any("64-character hex digest" in item for item in report["next_actions"]))

    def test_manual_hard_gold_blocks_raw_sha256_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            raw.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {
                                "type": "manual",
                                "raw_path": str(raw),
                                "official_url": "https://example.com/rubricbench",
                                "require_raw_sha256": True,
                                "raw_sha256": "0" * 64,
                            },
                            "normalizations": [],
                        }
                    ]
                }
            )

        self.assertEqual(report["overall_status"], "blocked")
        self.assertTrue(any("raw_sha256 mismatch" in item for item in report["blockers"]))

    def test_manual_hard_gold_accepts_matching_raw_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            content = json.dumps({"query": "q"}) + "\n"
            raw.write_text(content, encoding="utf-8")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {
                                "type": "manual",
                                "raw_path": str(raw),
                                "official_url": "https://example.com/rubricbench",
                                "require_raw_sha256": True,
                                "raw_sha256": digest,
                            },
                            "normalizations": [],
                        }
                    ]
                }
            )

        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual(report["datasets"][0]["actual_raw_sha256"], digest)

    def test_hf_missing_source_warns_without_blocking(self) -> None:
        report = build_report(
            {
                "datasets": [
                    {
                        "name": "rewardbench",
                        "source": {"type": "hf", "preset": "rewardbench", "output": "/tmp/missing_rewardbench.jsonl"},
                        "normalizations": [{"target": "preference", "output": "/tmp/missing_pref.jsonl"}],
                    }
                ]
            }
        )

        self.assertEqual(report["overall_status"], "warn")
        self.assertEqual(report["blockers"], [])
        self.assertIn("raw HF export not present", report["warnings"][0])

    def test_required_dataset_scope_ignores_unrelated_blockers(self) -> None:
        report = build_report(
            {
                "datasets": [
                    {
                        "name": "rubricbench",
                        "source": {"type": "hf", "output": "/tmp/missing_rubricbench.jsonl"},
                        "normalizations": [],
                    },
                    {
                        "name": "researchrubrics",
                        "source": {
                            "type": "manual",
                            "raw_path": "/tmp/missing_researchrubrics.jsonl",
                            "require_official_url": True,
                            "official_url": "",
                        },
                        "normalizations": [],
                    },
                ]
            },
            required_datasets=["rubricbench"],
        )

        self.assertEqual(report["overall_status"], "warn")
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["required_datasets"], ["rubricbench"])
        self.assertFalse(any("researchrubrics" in item for item in report["next_actions"]))

    def test_required_dataset_scope_blocks_missing_dataset(self) -> None:
        report = build_report({"datasets": []}, required_datasets=["rubricbench"])

        self.assertEqual(report["overall_status"], "blocked")
        self.assertIn("missing required dataset", report["blockers"][0])

    def test_url_missing_source_warns_without_blocking(self) -> None:
        report = build_report(
            {
                "datasets": [
                    {
                        "name": "healthbench",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/healthbench.jsonl",
                            "output": "/tmp/missing_healthbench.jsonl",
                        },
                        "normalizations": [{"target": "query_pool", "output": "/tmp/missing_queries.jsonl"}],
                    }
                ]
            }
        )

        self.assertEqual(report["overall_status"], "warn")
        self.assertEqual(report["blockers"], [])
        self.assertIn("raw URL export not present", report["warnings"][0])
        self.assertIn("Run the generated URL download stage", report["next_actions"][0])

    def test_present_source_with_outputs_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            profile = root / "profile.json"
            normalized = root / "gold.jsonl"
            raw.write_text(json.dumps({"query": "q"}) + "\n", encoding="utf-8")
            profile.write_text("{}", encoding="utf-8")
            normalized.write_text(json.dumps({"query": "q", "gold_rubrics": ["g"]}) + "\n", encoding="utf-8")

            report = build_report(
                {
                    "datasets": [
                        {
                            "name": "rubricbench",
                            "source": {"type": "manual", "raw_path": str(raw)},
                            "profile": {"output": str(profile)},
                            "normalizations": [{"target": "gold", "output": str(normalized)}],
                        }
                    ]
                }
            )

            self.assertEqual(report["overall_status"], "pass")
            self.assertTrue(report["datasets"][0]["raw_present"])

    def test_markdown_contains_source_notes(self) -> None:
        report = {
            "title": "Data Sources",
            "overall_status": "pass",
            "blockers": [],
            "warnings": [],
            "next_actions": [],
            "datasets": [
                {
                    "name": "rubricbench",
                    "source_type": "manual",
                    "status": "pass",
                    "raw_present": True,
                    "raw_path": "data/raw/rubricbench.jsonl",
                    "expected_fields": ["query"],
                    "official_url": "https://example.com",
                    "paper_url": "",
                    "note": "note",
                }
            ],
        }
        md = to_markdown(report)
        self.assertIn("Source Notes", md)
        self.assertIn("https://example.com", md)


if __name__ == "__main__":
    unittest.main()
