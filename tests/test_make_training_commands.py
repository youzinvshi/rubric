from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.make_training_commands import (
    build_grpo_variant_section,
    build_grpo_script,
    build_manifest,
    build_reward_component_ablation_done_template,
    build_sft_script,
    build_training_done_template,
    load_config,
    shell_quote,
    write_reward_component_ablation_outputs,
)


class MakeTrainingCommandsTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_training_commands_config.json"))

        self.assertIn("Training command config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Training command config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Training command config must be a JSON object", str(context.exception))

    def test_shell_quote_handles_spaces_and_plain_values(self) -> None:
        self.assertEqual(shell_quote("abc/def"), "abc/def")
        self.assertEqual(shell_quote("a b"), "'a b'")

    def test_build_sft_script_contains_env_and_command(self) -> None:
        script = build_sft_script(
            {
                "sft": {
                    "config": "sft.yaml",
                    "command": "llamafactory-cli train",
                    "log": "logs/sft.log",
                    "env": {"TOKENIZERS_PARALLELISM": "false"},
                },
                "grpo": {"config": "grpo.yaml"},
            }
        )
        self.assertIn("export TOKENIZERS_PARALLELISM=false", script)
        self.assertIn("llamafactory-cli train sft.yaml", script)

    def test_build_grpo_script_contains_reward_env(self) -> None:
        script = build_grpo_script(
            {
                "sft": {"config": "sft.yaml"},
                "grpo": {
                    "config": "grpo.yaml",
                    "command": "python3 -m verl.trainer.main_ppo",
                    "log": "logs/grpo.log",
                    "env": {"BSC_W_RED": "0.5"},
                    "overrides": ["trainer.total_epochs=1"],
                },
            }
        )
        self.assertIn("export BSC_W_RED=0.5", script)
        self.assertIn("--config-path grpo.yaml trainer.total_epochs=1", script)

    def test_build_manifest_records_outputs(self) -> None:
        manifest = build_manifest(
            {
                "sft": {"config": "sft.yaml", "output_dir": "sft_ckpt", "sft_data": "sft.jsonl"},
                "grpo": {
                    "config": "grpo.yaml",
                    "output_dir": "rl_ckpt",
                    "rl_data": "rl.parquet",
                    "rl_data_report": "rl_report.json",
                    "reward_function": "src/reward.py:score",
                },
            },
            Path("run_sft.sh"),
            Path("run_grpo.sh"),
            Path("training_done.template.json"),
        )
        self.assertEqual(manifest["sft_output_dir"], "sft_ckpt")
        self.assertEqual(manifest["grpo_script"], "run_grpo.sh")
        self.assertEqual(manifest["training_done_template"], "training_done.template.json")
        self.assertEqual(manifest["expected_training_done"], "training_done.json")
        self.assertIn("scripts/fill_training_done_sha256.py --input training_done.json", manifest["training_done_sha256_command"])
        self.assertEqual(manifest["rl_data_report"], "rl_report.json")
        self.assertEqual(manifest["reward_function"], "src/reward.py:score")

    def test_build_training_done_template_records_bound_inputs_and_reward(self) -> None:
        template = build_training_done_template(
            {
                "sft": {
                    "config": "configs/llamafactory_sft.local.yaml",
                    "output_dir": "outputs/checkpoints/sft",
                    "sft_data": "data/processed/blindspot_sft.jsonl",
                },
                "grpo": {
                    "config": "configs/verl_grpo_bsc.local.yaml",
                    "output_dir": "outputs/checkpoints/rl",
                    "rl_data": "data/processed/proxy_gold_verl.parquet",
                    "rl_data_report": "outputs/sft_data/proxy_gold_verl_report.json",
                    "reward_function": "src/blindspot_rl/verl_reward.py:compute_score",
                },
            }
        )

        self.assertEqual(template["sft_data"], "data/processed/blindspot_sft.jsonl")
        self.assertEqual(template["rl_data"], "data/processed/proxy_gold_verl.parquet")
        self.assertEqual(template["rl_data_report"], "outputs/sft_data/proxy_gold_verl_report.json")
        self.assertEqual(template["reward_function"], "src/blindspot_rl/verl_reward.py:compute_score")
        self.assertEqual(template["served_methods"], ["base", "sft_only", "sft_rl"])
        self.assertEqual(template["served_generators"], ["base", "sft_only", "sft_rl"])
        for key in [
            "sft_config_sha256",
            "grpo_config_sha256",
            "sft_data_sha256",
            "rl_data_sha256",
            "rl_data_report_sha256",
            "reward_function_sha256",
        ]:
            with self.subTest(key=key):
                self.assertEqual(template[key], "")

    def test_reward_component_ablation_variant_merges_reward_env(self) -> None:
        config = minimal_ablation_config()
        variant = build_grpo_variant_section(
            config,
            "no_valid",
            config["reward_component_ablation"]["variants"]["no_valid"],
        )

        self.assertEqual(variant["env"]["BSC_W_COV"], "1.0")
        self.assertEqual(variant["env"]["BSC_W_VALID"], "0.0")
        self.assertEqual(variant["env"]["BSC_VERIFIER"], "none")
        self.assertEqual(variant["output_dir"], "outputs/checkpoints/rl_no_valid")
        self.assertIn("trainer.default_local_dir=outputs/checkpoints/rl_no_valid", variant["overrides"])

    def test_build_reward_component_ablation_done_template_records_variants(self) -> None:
        template = build_reward_component_ablation_done_template(minimal_ablation_config())

        self.assertEqual(template["reward_variants"], ["full", "no_red", "no_valid", "no_verifier", "cov_only"])
        self.assertEqual(template["variants"]["full"]["checkpoint"], "outputs/checkpoints/rl")
        self.assertEqual(template["variants"]["full"]["env"]["BSC_VERIFIER"], "rule")
        self.assertEqual(template["variants"]["no_red"]["env"]["BSC_VERIFIER"], "rule")
        self.assertEqual(template["variants"]["no_red"]["env"]["BSC_W_RED"], "0.0")
        self.assertEqual(template["variants"]["no_valid"]["env"]["BSC_W_VALID"], "0.0")
        self.assertEqual(template["variants"]["no_valid"]["env"]["BSC_VERIFIER"], "none")
        self.assertEqual(template["variants"]["no_verifier"]["env"]["BSC_W_VALID"], "0.5")
        self.assertEqual(template["variants"]["no_verifier"]["env"]["BSC_W_RED"], "0.5")
        self.assertEqual(template["variants"]["no_verifier"]["env"]["BSC_VERIFIER"], "none")
        self.assertEqual(template["variants"]["cov_only"]["env"]["BSC_W_VALID"], "0.0")
        self.assertEqual(template["variants"]["cov_only"]["env"]["BSC_W_RED"], "0.0")
        self.assertEqual(template["variants"]["cov_only"]["env"]["BSC_VERIFIER"], "none")
        self.assertEqual(template["rl_data"], "data/processed/proxy_gold_verl.parquet")
        for key in ["sft_data_sha256", "rl_data_sha256", "rl_data_report_sha256", "reward_function_sha256"]:
            with self.subTest(key=key):
                self.assertEqual(template[key], "")
        for variant_name, variant in template["variants"].items():
            with self.subTest(variant=variant_name):
                self.assertEqual(variant["grpo_config_sha256"], "")
                self.assertEqual(variant["rl_data_sha256"], "")
                self.assertEqual(variant["rl_data_report_sha256"], "")
                self.assertEqual(variant["reward_function_sha256"], "")

    def test_write_reward_component_ablation_outputs_writes_scripts_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "training_commands"
            output_dir.mkdir()
            config = minimal_ablation_config()
            config["reward_component_ablation"]["output_dir"] = str(Path(tmp) / "reward_component_training_ablation")

            result = write_reward_component_ablation_outputs(config, output_dir)

            assert result is not None
            self.assertEqual(result["reward_variants"], ["full", "no_red", "no_valid", "no_verifier", "cov_only"])
            self.assertIn("scripts/fill_training_done_sha256.py", result["sha256_fill_command"])
            self.assertIn("reward_component_training_ablation", result["sha256_fill_command"])
            no_red_script = output_dir / "run_grpo_no_red.sh"
            no_valid_script = output_dir / "run_grpo_no_valid.sh"
            no_verifier_script = output_dir / "run_grpo_no_verifier.sh"
            cov_only_script = output_dir / "run_grpo_cov_only.sh"
            self.assertTrue(no_red_script.exists())
            self.assertTrue(no_valid_script.exists())
            self.assertTrue(no_verifier_script.exists())
            self.assertTrue(cov_only_script.exists())
            self.assertIn("export BSC_W_RED=0.0", no_red_script.read_text(encoding="utf-8"))
            self.assertIn("export BSC_W_VALID=0.0", no_valid_script.read_text(encoding="utf-8"))
            self.assertIn("export BSC_W_VALID=0.5", no_verifier_script.read_text(encoding="utf-8"))
            self.assertIn("export BSC_VERIFIER=none", no_verifier_script.read_text(encoding="utf-8"))
            self.assertIn("export BSC_VERIFIER=none", cov_only_script.read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "reward_component_training_ablation" / "training_done.template.json").exists())


def minimal_ablation_config() -> dict:
    return {
        "sft": {
            "config": "configs/llamafactory_sft.local.yaml",
            "output_dir": "outputs/checkpoints/sft",
            "sft_data": "data/processed/blindspot_sft.jsonl",
        },
        "grpo": {
            "config": "configs/verl_grpo_bsc.local.yaml",
            "command": "python3 -m verl.trainer.main_ppo",
            "log": "outputs/logs/grpo.log",
            "output_dir": "outputs/checkpoints/rl",
            "rl_data": "data/processed/proxy_gold_verl.parquet",
            "rl_data_report": "outputs/sft_data/proxy_gold_verl_report.json",
            "reward_function": "src/blindspot_rl/verl_reward.py:compute_score",
            "env": {
                "BSC_W_COV": "1.0",
                "BSC_W_VALID": "0.5",
                "BSC_W_RED": "0.5",
                "BSC_VERIFIER": "rule",
            },
            "overrides": [],
        },
        "reward_component_ablation": {
            "enabled": True,
            "output_dir": "outputs/reward_component_training_ablation",
            "variants": {
                "no_red": {
                    "log": "outputs/logs/grpo_no_red.log",
                    "output_dir": "outputs/checkpoints/rl_no_red",
                    "env": {"BSC_W_RED": "0.0"},
                    "overrides": [
                        "trainer.experiment_name=bsc_grpo_no_red",
                        "trainer.default_local_dir=outputs/checkpoints/rl_no_red",
                    ],
                },
                "no_valid": {
                    "log": "outputs/logs/grpo_no_valid.log",
                    "output_dir": "outputs/checkpoints/rl_no_valid",
                    "env": {"BSC_W_VALID": "0.0", "BSC_VERIFIER": "none"},
                    "overrides": [
                        "trainer.experiment_name=bsc_grpo_no_valid",
                        "trainer.default_local_dir=outputs/checkpoints/rl_no_valid",
                    ],
                },
                "no_verifier": {
                    "log": "outputs/logs/grpo_no_verifier.log",
                    "output_dir": "outputs/checkpoints/rl_no_verifier",
                    "env": {"BSC_VERIFIER": "none"},
                    "overrides": [
                        "trainer.experiment_name=bsc_grpo_no_verifier",
                        "trainer.default_local_dir=outputs/checkpoints/rl_no_verifier",
                    ],
                },
                "cov_only": {
                    "log": "outputs/logs/grpo_cov_only.log",
                    "output_dir": "outputs/checkpoints/rl_cov_only",
                    "env": {"BSC_W_VALID": "0.0", "BSC_W_RED": "0.0", "BSC_VERIFIER": "none"},
                    "overrides": [
                        "trainer.experiment_name=bsc_grpo_cov_only",
                        "trainer.default_local_dir=outputs/checkpoints/rl_cov_only",
                    ],
                },
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
