from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.fill_training_done_sha256 import (
    file_sha256,
    fill_training_done_sha256,
    load_json,
    resolve_recorded_path,
)


class FillTrainingDoneSha256Test(unittest.TestCase):
    def test_fills_main_training_done_sha256_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft_config = write_file(root / "configs" / "sft.yaml", "sft")
            grpo_config = write_file(root / "configs" / "grpo.yaml", "grpo")
            sft_data = write_file(root / "data" / "sft.jsonl", "{}\n")
            rl_data = write_file(root / "data" / "rl.parquet", "rl")
            rl_report = write_file(root / "outputs" / "rl_report.json", "{}")
            reward = write_file(root / "src" / "reward.py", "def compute_score():\n    return 1\n")
            data = {
                "sft_config": "configs/sft.yaml",
                "grpo_config": "configs/grpo.yaml",
                "sft_data": "data/sft.jsonl",
                "rl_data": "data/rl.parquet",
                "rl_data_report": "outputs/rl_report.json",
                "reward_function": "src/reward.py:compute_score",
                "sft_config_sha256": "",
                "grpo_config_sha256": "",
                "sft_data_sha256": "",
                "rl_data_sha256": "",
                "rl_data_report_sha256": "",
                "reward_function_sha256": "",
            }

            filled, report = fill_training_done_sha256(data, root=root)
            expected = {
                "sft_config_sha256": file_sha256(sft_config),
                "grpo_config_sha256": file_sha256(grpo_config),
                "sft_data_sha256": file_sha256(sft_data),
                "rl_data_sha256": file_sha256(rl_data),
                "rl_data_report_sha256": file_sha256(rl_report),
                "reward_function_sha256": file_sha256(reward),
            }

        self.assertTrue(report["ok"])
        self.assertEqual(filled["sft_config_sha256"], expected["sft_config_sha256"])
        self.assertEqual(filled["grpo_config_sha256"], expected["grpo_config_sha256"])
        self.assertEqual(filled["sft_data_sha256"], expected["sft_data_sha256"])
        self.assertEqual(filled["rl_data_sha256"], expected["rl_data_sha256"])
        self.assertEqual(filled["rl_data_report_sha256"], expected["rl_data_report_sha256"])
        self.assertEqual(filled["reward_function_sha256"], expected["reward_function_sha256"])
        self.assertEqual(len([item for item in report["updates"] if item["status"] == "filled"]), 6)

    def test_fills_reward_component_variant_sha256_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_file(root / "configs" / "grpo.yaml", "grpo")
            write_file(root / "data" / "rl.parquet", "rl")
            write_file(root / "outputs" / "rl_report.json", "{}")
            write_file(root / "src" / "reward.py", "def compute_score():\n    return 1\n")
            data = {
                "rl_data": "data/rl.parquet",
                "rl_data_report": "outputs/rl_report.json",
                "reward_function": "src/reward.py:compute_score",
                "rl_data_sha256": "",
                "rl_data_report_sha256": "",
                "reward_function_sha256": "",
                "variants": {
                    "full": {
                        "grpo_config": "configs/grpo.yaml",
                        "rl_data": "data/rl.parquet",
                        "rl_data_report": "outputs/rl_report.json",
                        "reward_function": "src/reward.py:compute_score",
                        "grpo_config_sha256": "",
                        "rl_data_sha256": "",
                        "rl_data_report_sha256": "",
                        "reward_function_sha256": "",
                    },
                    "no_verifier": {
                        "grpo_config": "configs/grpo.yaml",
                        "rl_data": "data/rl.parquet",
                        "rl_data_report": "outputs/rl_report.json",
                        "reward_function": "src/reward.py:compute_score",
                        "grpo_config_sha256": "",
                        "rl_data_sha256": "",
                        "rl_data_report_sha256": "",
                        "reward_function_sha256": "",
                    },
                },
            }

            filled, report = fill_training_done_sha256(data, root=root)

        self.assertTrue(report["ok"])
        self.assertEqual(filled["variants"]["full"]["grpo_config_sha256"], filled["variants"]["no_verifier"]["grpo_config_sha256"])
        self.assertEqual(filled["variants"]["full"]["reward_function_sha256"], filled["reward_function_sha256"])
        self.assertEqual(len([item for item in report["updates"] if item["status"] == "filled"]), 11)

    def test_missing_source_blocks_by_default(self) -> None:
        filled, report = fill_training_done_sha256(
            {"rl_data": "missing.parquet", "rl_data_sha256": ""},
            root=Path("/tmp/nonexistent_training_done_sha_root"),
        )

        self.assertFalse(report["ok"])
        self.assertEqual(filled["rl_data_sha256"], "")
        self.assertIn("missing SHA256 source file", report["blockers"][0])

    def test_allow_missing_reports_without_blocking(self) -> None:
        filled, report = fill_training_done_sha256(
            {"rl_data": "missing.parquet", "rl_data_sha256": ""},
            root=Path("/tmp/nonexistent_training_done_sha_root"),
            allow_missing=True,
        )

        self.assertTrue(report["ok"])
        self.assertEqual(filled["rl_data_sha256"], "")
        self.assertEqual(report["updates"][0]["status"], "missing")

    def test_reward_function_resolves_to_source_file(self) -> None:
        path = resolve_recorded_path("reward_function", "src/reward.py:compute_score", root=Path("/repo"))

        self.assertEqual(path, Path("/repo/src/reward.py"))

    def test_load_json_requires_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_done.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit):
                load_json(path)


def write_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
