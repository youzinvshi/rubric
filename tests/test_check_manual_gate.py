from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from scripts.check_manual_gate import (
    build_report,
    check_json_contains_contract,
    check_json_contract,
    check_json_equals_contract,
    check_json_sha256_contract,
    file_sha256,
    nested_get,
    parse_json_contains_spec,
    parse_json_equals_spec,
    parse_json_spec,
    to_markdown,
)


class CheckManualGateTest(unittest.TestCase):
    def test_build_report_passes_when_required_paths_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            path.mkdir()
            report = build_report("training", [path], ["run training"])

        self.assertTrue(report["ok"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["checks"][0]["type"], "dir")

    def test_build_report_blocks_missing_required_paths(self) -> None:
        report = build_report("training", [Path("missing-checkpoint")], ["run training"])

        self.assertFalse(report["ok"])
        self.assertIn("missing required path: missing-checkpoint", report["blockers"])

    def test_to_markdown_lists_instructions(self) -> None:
        report = build_report("training", [Path("missing-checkpoint")], ["run SFT first"])
        markdown = to_markdown(report)

        self.assertIn("Manual Gate: training", markdown)
        self.assertIn("run SFT first", markdown)

    def test_build_report_blocks_missing_required_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "training_done.json"
            done.write_text(json.dumps({"sft_checkpoint": "sft"}), encoding="utf-8")

            report = build_report(
                "training",
                [done],
                ["write done metadata"],
                required_json=[f"{done}:sft_checkpoint,rl_checkpoint,operator,date"],
            )

        self.assertFalse(report["ok"])
        self.assertIn("missing required JSON key", report["blockers"][0])
        self.assertEqual(report["json_checks"][0]["missing_keys"], ["rl_checkpoint", "operator", "date"])

    def test_json_contract_blocks_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text("{bad", encoding="utf-8")

            check = check_json_contract(f"{path}:operator")

        self.assertFalse(check["valid_json"])
        self.assertIn("invalid JSON file", check["blockers"][0])

    def test_json_contract_treats_empty_containers_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text(
                json.dumps({"served_generators": [], "metadata": {"operator": ""}}),
                encoding="utf-8",
            )

            check = check_json_contract(f"{path}:served_generators,metadata.operator")

        self.assertEqual(check["missing_keys"], ["served_generators", "metadata.operator"])

    def test_json_contract_supports_dotted_keys_and_list_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text(
                json.dumps({"served_generators": ["base", "sft_only"], "reports": [{"path": "eval.json"}]}),
                encoding="utf-8",
            )

            check = check_json_contract(f"{path}:served_generators.1,reports.0.path")

        self.assertEqual(check["missing_keys"], [])

    def test_json_contains_contract_blocks_missing_list_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text(json.dumps({"served_generators": ["base"]}), encoding="utf-8")

            check = check_json_contains_contract(f"{path}:served_generators=base,sft_only,sft_rl")

        self.assertEqual(check["actual_values"], ["base"])
        self.assertEqual(check["missing_values"], ["sft_only", "sft_rl"])
        self.assertIn("must contain sft_only", check["blockers"][0])

    def test_build_report_includes_json_contains_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text(json.dumps({"served_generators": ["base", "sft_only", "sft_rl"]}), encoding="utf-8")

            report = build_report(
                "training",
                [path],
                [],
                required_json_contains=[f"{path}:served_generators=base,sft_only,sft_rl"],
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["json_contains_checks"][0]["missing_values"], [])

    def test_json_equals_contract_blocks_mismatched_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text(
                json.dumps({"rl_data": "data/processed/other.parquet"}),
                encoding="utf-8",
            )

            check = check_json_equals_contract(f"{path}:rl_data=data/processed/proxy_gold_verl.parquet")

        self.assertFalse(check["matches"])
        self.assertIn("JSON value mismatch", check["blockers"][0])

    def test_build_report_includes_json_equals_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.json"
            path.write_text(
                json.dumps({"reward_function": "src/blindspot_rl/verl_reward.py:compute_score"}),
                encoding="utf-8",
            )

            report = build_report(
                "training",
                [path],
                [],
                required_json_equals=[f"{path}:reward_function=src/blindspot_rl/verl_reward.py:compute_score"],
            )

        self.assertTrue(report["ok"])
        self.assertTrue(report["json_equals_checks"][0]["matches"])

    def test_json_sha256_contract_matches_current_file_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "proxy_gold_verl.parquet"
            data.write_text("current rl data", encoding="utf-8")
            report_path = root / "proxy_gold_verl_report.json"
            report_path.write_text(
                json.dumps({"output_sha256": file_sha256(data)}),
                encoding="utf-8",
            )

            check = check_json_sha256_contract(f"{report_path}:output_sha256={data}")

        self.assertTrue(check["matches"])
        self.assertEqual(check["blockers"], [])

    def test_json_sha256_contract_blocks_stale_report_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "proxy_gold_verl.parquet"
            data.write_text("old rl data", encoding="utf-8")
            stale_sha = file_sha256(data)
            data.write_text("new rl data", encoding="utf-8")
            report_path = root / "proxy_gold_verl_report.json"
            report_path.write_text(
                json.dumps({"output_sha256": stale_sha}),
                encoding="utf-8",
            )

            report = build_report(
                "training",
                [data, report_path],
                [],
                required_json_sha256=[f"{report_path}:output_sha256={data}"],
            )

        self.assertFalse(report["ok"])
        self.assertFalse(report["json_sha256_checks"][0]["matches"])
        self.assertTrue(any("JSON sha256 mismatch" in item for item in report["blockers"]))

    def test_training_done_provenance_contract_blocks_placeholder_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_done.json"
            path.write_text(
                json.dumps(
                    {
                        "sft_checkpoint": "outputs/checkpoints/evaluation_criteria_policy_sft",
                        "rl_checkpoint": "outputs/checkpoints/evaluation_criteria_policy_rl",
                        "served_generators": ["base", "sft_only", "sft_rl"],
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(
                "trained_method_gate",
                [path],
                [],
                required_json=[
                    f"{path}:sft_checkpoint,rl_checkpoint,served_generators,serving.base,serving.sft_only,serving.sft_rl,operator,date,sft_config,grpo_config,sft_data,rl_data,reward_function"
                ],
                required_json_contains=[f"{path}:served_generators=base,sft_only,sft_rl"],
                required_json_equals=[f"{path}:reward_function=src/blindspot_rl/verl_reward.py:compute_score"],
            )

        self.assertFalse(report["ok"])
        self.assertIn("serving.base", report["json_checks"][0]["missing_keys"])
        self.assertIn("operator", report["json_checks"][0]["missing_keys"])
        self.assertIn("reward_function", report["json_checks"][0]["missing_keys"])
        self.assertFalse(report["json_equals_checks"][0]["matches"])

    def test_nested_get_returns_none_for_missing_nested_keys(self) -> None:
        self.assertIsNone(nested_get({"a": []}, "a.0.x"))

    def test_parse_json_spec(self) -> None:
        path, keys = parse_json_spec("outputs/training_done.json:sft,rl")

        self.assertEqual(path, "outputs/training_done.json")
        self.assertEqual(keys, ["sft", "rl"])

    def test_parse_json_contains_spec(self) -> None:
        path, key, values = parse_json_contains_spec("done.json:served_generators=base,sft_rl")

        self.assertEqual(path, "done.json")
        self.assertEqual(key, "served_generators")
        self.assertEqual(values, ["base", "sft_rl"])

    def test_parse_json_equals_spec(self) -> None:
        path, key, value = parse_json_equals_spec("done.json:reward.path=src/reward.py:compute_score")

        self.assertEqual(path, "done.json")
        self.assertEqual(key, "reward.path")
        self.assertEqual(value, "src/reward.py:compute_score")


if __name__ == "__main__":
    unittest.main()
