from __future__ import annotations

import unittest
from unittest import mock

from scripts.download_public_data import (
    PRESETS,
    is_parquet_url,
    parse_json_or_jsonl,
    parse_parquet_bytes,
    resolve_splits,
)


class DownloadPublicDataTest(unittest.TestCase):
    def test_parse_jsonl_payload(self) -> None:
        records = parse_json_or_jsonl('{"query": "q1"}\n{"query": "q2"}\n')

        self.assertEqual([record["query"] for record in records], ["q1", "q2"])

    def test_parse_json_array_payload(self) -> None:
        records = parse_json_or_jsonl('[{"query": "q1"}, {"query": "q2"}]')

        self.assertEqual([record["query"] for record in records], ["q1", "q2"])

    def test_parse_wrapped_json_payload(self) -> None:
        records = parse_json_or_jsonl('{"data": [{"query": "q1"}]}')

        self.assertEqual(records, [{"query": "q1"}])

    def test_parse_scalar_items_as_value_records(self) -> None:
        records = parse_json_or_jsonl('["a", "b"]')

        self.assertEqual(records, [{"value": "a"}, {"value": "b"}])

    def test_rewardbench2_preset_uses_public_hf_dataset(self) -> None:
        self.assertEqual(PRESETS["rewardbench2"]["hf_dataset"], "allenai/reward-bench-2")
        self.assertEqual(PRESETS["rewardbench2"]["split"], "test")

    def test_healthbench_preset_uses_public_hf_dataset(self) -> None:
        self.assertEqual(PRESETS["healthbench"]["hf_dataset"], "openai/healthbench")
        self.assertEqual(PRESETS["healthbench"]["split"], "test")

    def test_beir_nq_preset_uses_queries_config(self) -> None:
        self.assertEqual(PRESETS["beir_nq"]["hf_dataset"], "BeIR/nq")
        self.assertEqual(PRESETS["beir_nq"]["name"], "queries")
        self.assertEqual(PRESETS["beir_nq"]["split"], "queries")

    def test_ifbench_preset_uses_public_hf_dataset(self) -> None:
        self.assertEqual(PRESETS["ifbench"]["hf_dataset"], "allenai/IFBench_test")
        self.assertEqual(PRESETS["ifbench"]["split"], "train")

    def test_judgebench_preset_uses_gpt_and_claude_splits(self) -> None:
        self.assertEqual(PRESETS["judgebench"]["hf_dataset"], "ScalerLab/JudgeBench")
        self.assertEqual(PRESETS["judgebench"]["splits"], ["gpt", "claude"])
        self.assertNotIn("split", PRESETS["judgebench"])

    def test_writingbench_preset_uses_public_raw_query_url(self) -> None:
        self.assertEqual(
            PRESETS["writingbench"]["url"],
            "https://raw.githubusercontent.com/X-PLUG/WritingBench/main/benchmark_query/benchmark_all.jsonl",
        )
        self.assertEqual(PRESETS["writingbench"]["output"], "data/raw/writingbench_raw.jsonl")

    def test_resolve_splits_precedence(self) -> None:
        self.assertEqual(resolve_splits("custom", {"splits": ["gpt", "claude"]}), ["custom"])
        self.assertEqual(resolve_splits(None, {"splits": ["gpt", "claude"]}), ["gpt", "claude"])
        self.assertEqual(resolve_splits(None, {"split": "test"}), ["test"])
        self.assertEqual(resolve_splits(None, {}), [])

    def test_detects_parquet_urls(self) -> None:
        self.assertTrue(is_parquet_url("https://huggingface.co/datasets/org/ds/resolve/main/data/train.parquet"))
        self.assertTrue(is_parquet_url("https://example.com/data/file.pq?download=1"))
        self.assertFalse(is_parquet_url("https://example.com/data/file.jsonl"))

    def test_parquet_parser_reports_missing_engine_or_invalid_payload(self) -> None:
        with mock.patch("pandas.read_parquet", side_effect=ImportError("missing engine")):
            with self.assertRaises(RuntimeError) as context:
                parse_parquet_bytes(b"not parquet")

        self.assertIn("parquet", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()
