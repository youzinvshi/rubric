from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.register_llamafactory_dataset import build_entry, load_dataset_info


class RegisterLlamaFactoryDatasetTest(unittest.TestCase):
    def test_build_entry_uses_alpaca_columns(self) -> None:
        entry = build_entry(
            "blindspot_sft.jsonl",
            "alpaca",
            {"prompt": "instruction", "query": "input", "response": "output"},
        )
        self.assertEqual(entry["file_name"], "blindspot_sft.jsonl")
        self.assertEqual(entry["columns"]["response"], "output")

    def test_load_dataset_info_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_dataset_info(Path(tmp) / "dataset_info.json"), {})

    def test_load_dataset_info_preserves_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_info.json"
            path.write_text('{"old": {"file_name": "old.jsonl"}}', encoding="utf-8")
            self.assertEqual(load_dataset_info(path)["old"]["file_name"], "old.jsonl")


if __name__ == "__main__":
    unittest.main()
