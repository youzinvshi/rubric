from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.init_data_source_local_config import fill_present_sha256, main, sync_template_official_urls


class InitDataSourceLocalConfigTest(unittest.TestCase):
    def test_initializes_missing_local_config_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.json"
            output = root / "local.json"
            report = root / "report.json"
            template.write_text(json.dumps(sample_config()), encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                run_main(
                    [
                        "--template",
                        str(template),
                        "--output",
                        str(output),
                        "--report-json",
                        str(report),
                        "--strict",
                    ]
                )

            self.assertEqual(context.exception.code, 1)
            self.assertTrue(output.exists())
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["datasets"][0]["name"], "rubricbench")
            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(audit["ok"])
            self.assertTrue(audit["wrote_config"])
            self.assertTrue(any("official_url" in item for item in audit["blockers"]))
            self.assertTrue(any("Set rubricbench official_url" in item for item in audit["hard_gold_next_actions"]))
            self.assertFalse(any("rewardbench" in item.lower() for item in audit["hard_gold_next_actions"]))

    def test_existing_local_config_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.json"
            output = root / "local.json"
            report = root / "report.json"
            template.write_text(json.dumps(sample_config()), encoding="utf-8")
            existing = sample_config()
            existing["datasets"][0]["source"]["official_url"] = "https://huggingface.co/datasets/org/rubricbench"
            output.write_text(json.dumps(existing), encoding="utf-8")

            run_main(["--template", str(template), "--output", str(output), "--report-json", str(report)])

            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["datasets"][0]["source"]["official_url"], "https://huggingface.co/datasets/org/rubricbench")
            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(audit["output_existed"])
            self.assertFalse(audit["wrote_config"])

    def test_first_initialization_passes_when_template_contracts_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "rubricbench.jsonl"
            content = '{"query": "q"}\n'
            raw.write_text(content, encoding="utf-8")
            template = root / "template.json"
            output = root / "local.json"
            report = root / "report.json"
            config = sample_config(raw_path=str(raw))
            config["datasets"][0]["source"]["official_url"] = "https://huggingface.co/datasets/org/rubricbench"
            config["datasets"][0]["source"]["raw_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            template.write_text(json.dumps(config), encoding="utf-8")

            run_main(
                [
                    "--template",
                    str(template),
                    "--output",
                    str(output),
                    "--report-json",
                    str(report),
                    "--strict",
                ]
            )

            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(audit["ok"])
            self.assertEqual(audit["blockers"], [])
            self.assertTrue(audit["wrote_config"])
            self.assertTrue(any("has been initialized" in item for item in audit["warnings"]))

    def test_pending_update_without_update_existing_does_not_claim_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "rubricbench.jsonl"
            raw.write_text('{"query": "q"}\n', encoding="utf-8")
            template = root / "template.json"
            output = root / "local.json"
            report = root / "report.json"
            template.write_text(json.dumps(sample_config(raw_path=str(raw))), encoding="utf-8")
            output.write_text(json.dumps(sample_config(raw_path=str(raw))), encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                run_main(
                    [
                        "--template",
                        str(template),
                        "--output",
                        str(output),
                        "--report-json",
                        str(report),
                        "--fill-present-sha256",
                        "--strict",
                    ]
                )

            self.assertEqual(context.exception.code, 1)
            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(audit["ok"])
            self.assertFalse(audit["wrote_config"])
            self.assertTrue(any("--update-existing was not set" in item for item in audit["blockers"]))

    def test_fill_present_sha256_updates_missing_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "rubricbench.jsonl"
            content = '{"query": "q"}\n'
            raw.write_text(content, encoding="utf-8")
            config = sample_config(raw_path=str(raw))

            changed = fill_present_sha256(config)

            self.assertTrue(changed)
            self.assertEqual(
                config["datasets"][0]["source"]["raw_sha256"],
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )

    def test_sync_template_official_urls_fills_missing_local_url(self) -> None:
        local = sample_config()
        template = sample_config()
        template["datasets"][0]["source"]["official_url"] = (
            "https://huggingface.co/datasets/org/rubricbench/resolve/main/data.parquet"
        )
        template["datasets"][0]["source"]["download_enabled"] = True
        template["datasets"][0]["source"]["note"] = "Official release."

        changed = sync_template_official_urls(local, template)

        self.assertTrue(changed)
        self.assertEqual(
            local["datasets"][0]["source"]["official_url"],
            "https://huggingface.co/datasets/org/rubricbench/resolve/main/data.parquet",
        )
        self.assertTrue(local["datasets"][0]["source"]["download_enabled"])
        self.assertEqual(local["datasets"][0]["source"]["note"], "Official release.")

    def test_sync_template_official_urls_keeps_existing_valid_url(self) -> None:
        local = sample_config()
        local["datasets"][0]["source"]["official_url"] = "https://huggingface.co/datasets/local/rubricbench"
        template = sample_config()
        template["datasets"][0]["source"]["official_url"] = "https://huggingface.co/datasets/template/rubricbench"

        changed = sync_template_official_urls(local, template)

        self.assertFalse(changed)
        self.assertEqual(
            local["datasets"][0]["source"]["official_url"],
            "https://huggingface.co/datasets/local/rubricbench",
        )

    def test_required_dataset_scope_ignores_unrelated_hard_gold_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "rubricbench.jsonl"
            content = '{"query": "q"}\n'
            raw.write_text(content, encoding="utf-8")
            template = root / "template.json"
            output = root / "local.json"
            report = root / "report.json"
            config = sample_config(raw_path=str(raw))
            config["datasets"][0]["source"]["official_url"] = "https://huggingface.co/datasets/org/rubricbench"
            config["datasets"][0]["source"]["raw_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            config["datasets"].append(
                {
                    "name": "researchrubrics",
                    "source": {
                        "type": "manual",
                        "raw_path": str(root / "missing_researchrubrics.jsonl"),
                        "official_url": "",
                        "require_official_url": True,
                        "raw_sha256": "",
                        "require_raw_sha256": True,
                    },
                    "normalizations": [],
                }
            )
            template.write_text(json.dumps(config), encoding="utf-8")

            run_main(
                [
                    "--template",
                    str(template),
                    "--output",
                    str(output),
                    "--report-json",
                    str(report),
                    "--required-dataset",
                    "rubricbench",
                    "--strict",
                ]
            )

            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(audit["ok"])
            self.assertEqual(audit["required_datasets"], ["rubricbench"])
            self.assertEqual(audit["blockers"], [])
            self.assertFalse(any("researchrubrics" in item for item in audit["hard_gold_next_actions"]))

    def test_hard_gold_next_actions_include_download_command_when_official_url_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.json"
            output = root / "local.json"
            report = root / "report.json"
            config = sample_config(raw_path=str(root / "missing_rubricbench.jsonl"))
            config["datasets"][0]["source"]["official_url"] = "https://huggingface.co/datasets/org/rubricbench/resolve/main/data.jsonl"
            template.write_text(json.dumps(config), encoding="utf-8")

            run_main(["--template", str(template), "--output", str(output), "--report-json", str(report)])

            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(
                any(
                    "download_public_data.py --url https://huggingface.co/datasets/org/rubricbench/resolve/main/data.jsonl"
                    in item
                    for item in audit["hard_gold_next_actions"]
                )
            )


def run_main(argv: list[str]) -> None:
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["init_data_source_local_config.py", *argv]
        main()
    finally:
        sys.argv = old_argv


def sample_config(raw_path: str = "data/raw/rubricbench_raw.jsonl") -> dict:
    return {
        "datasets": [
            {
                "name": "rubricbench",
                "source": {
                    "type": "manual",
                    "raw_path": raw_path,
                    "official_url": "",
                    "require_official_url": True,
                    "raw_sha256": "",
                    "require_raw_sha256": True,
                },
                "normalizations": [],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
