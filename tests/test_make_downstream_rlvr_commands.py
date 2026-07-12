from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.make_downstream_rlvr_commands import build_done_template, build_manifest, build_rlvr_script, build_scripts, load_config


class MakeDownstreamRLVRCommandsTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_downstream_rlvr_commands_config.json"))

        self.assertIn("Downstream RLVR command config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Downstream RLVR command config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Downstream RLVR command config must be a JSON object", str(context.exception))

    def test_build_rlvr_script_contains_reward_source_and_overrides(self) -> None:
        script = build_rlvr_script(
            {"env": {"BSC_REWARD_SOURCE": "criteria_policy"}, "overrides": ["algorithm.adv_estimator=grpo"]},
            {
                "name": "healthbench_hard",
                "config": "policy.yaml",
                "train_data": "train.parquet",
                "val_data": "val.parquet",
                "criteria_policy_checkpoint": "rubric_ckpt",
                "rubric_file": "rubrics.jsonl",
                "output_dir": "policy_ckpt",
            },
        )

        self.assertIn("export BSC_REWARD_SOURCE=criteria_policy", script)
        self.assertIn("export BSC_POLICY_CRITERIA_POLICY_CHECKPOINT=rubric_ckpt", script)
        self.assertIn("export BSC_POLICY_RUBRIC_FILE=rubrics.jsonl", script)
        self.assertIn("--config-path policy.yaml", script)
        self.assertIn("data.train_files=train.parquet", script)
        self.assertIn("reward_model.rubric_generator=rubric_ckpt", script)

    def test_legacy_rubric_generator_key_is_still_accepted(self) -> None:
        script = build_rlvr_script(
            {},
            {
                "name": "healthbench_hard",
                "config": "policy.yaml",
                "train_data": "train.parquet",
                "val_data": "val.parquet",
                "rubric_generator": "legacy_ckpt",
                "rubric_file": "rubrics.jsonl",
            },
        )

        self.assertIn("export BSC_POLICY_CRITERIA_POLICY_CHECKPOINT=legacy_ckpt", script)
        self.assertIn("reward_model.rubric_generator=legacy_ckpt", script)

    def test_build_scripts_and_manifest_records_eval_outputs(self) -> None:
        config = {
            "benchmarks": [
                {
                    "name": "arenahard",
                    "config": "arena.yaml",
                    "output_dir": "outputs/policy_rlvr/arenahard_policy",
                    "eval_command": "python eval.py",
                    "eval_output": "outputs/policy_rlvr/arenahard_eval.json",
                }
            ]
        }
        scripts = build_scripts(config)
        manifest = build_manifest(config, Path("outputs/downstream_rlvr_commands"), scripts)

        self.assertIn("run_arenahard_rlvr.sh", scripts)
        self.assertIn("run_arenahard_eval.sh", scripts)
        self.assertEqual(manifest["benchmarks"][0]["eval_output"], "outputs/policy_rlvr/arenahard_eval.json")
        self.assertEqual(
            manifest["downstream_rlvr_done_template"],
            "outputs/downstream_rlvr_commands/downstream_rlvr_done.template.json",
        )
        self.assertFalse(manifest["ok"])
        self.assertTrue(manifest["blockers"])

    def test_build_done_template_records_policy_rlvr_bindings(self) -> None:
        template = build_done_template(
            {
                "benchmarks": [
                    {
                        "name": "healthbench_hard",
                        "config": "configs/verl_policy_grpo_healthbench_hard.local.yaml",
                        "train_data": "data/processed/healthbench_hard_policy_rlvr.parquet",
                        "val_data": "data/processed/healthbench_hard_policy_rlvr.parquet",
                        "criteria_policy_checkpoint": "outputs/checkpoints/evaluation_criteria_policy_rl",
                        "rubric_file": "data/processed/healthbench_model_rubrics.jsonl",
                        "reward_function": "src/blindspot_rl/policy_reward.py:compute_score",
                        "output_dir": "outputs/policy_rlvr/healthbench_hard_policy",
                        "eval_output": "outputs/policy_rlvr/healthbench_hard_eval.json",
                    }
                ]
            }
        )

        self.assertEqual(template["healthbench_hard_policy"], "outputs/policy_rlvr/healthbench_hard_policy")
        self.assertEqual(template["healthbench_hard_eval"], "outputs/policy_rlvr/healthbench_hard_eval.json")
        self.assertEqual(
            template["benchmarks"]["healthbench_hard"]["reward_function"],
            "src/blindspot_rl/policy_reward.py:compute_score",
        )
        self.assertEqual(
            template["benchmarks"]["healthbench_hard"]["criteria_policy_checkpoint"],
            "outputs/checkpoints/evaluation_criteria_policy_rl",
        )

    def test_manifest_passes_when_required_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel_path in [
                "configs/policy.yaml",
                "data/train.parquet",
                "data/val.parquet",
                "outputs/checkpoints/evaluation_criteria_policy_rl",
                "data/rubrics.jsonl",
            ]:
                path = root / rel_path
                if path.suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)
            config = {
                "benchmarks": [
                    {
                        "name": "healthbench_hard",
                        "config": "configs/policy.yaml",
                        "train_data": "data/train.parquet",
                        "val_data": "data/val.parquet",
                        "criteria_policy_checkpoint": "outputs/checkpoints/evaluation_criteria_policy_rl",
                        "rubric_file": "data/rubrics.jsonl",
                    }
                ]
            }

            manifest = build_manifest(
                config,
                Path("outputs/downstream_rlvr_commands"),
                build_scripts(config),
                workspace=root,
            )

        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["blockers"], [])
        checks = manifest["benchmarks"][0]["checks"]
        self.assertTrue(all(item["present"] for item in checks))


if __name__ == "__main__":
    unittest.main()
