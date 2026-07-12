from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_real_run_pipeline import build_manifest, build_pipeline, load_pipeline_stages, read_json_object


class BuildRealRunPipelineTest(unittest.TestCase):
    def test_read_json_object_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            read_json_object(Path("/tmp/missing_real_run_assembly_config.json"), "Real-run assembly config")

        self.assertIn("Real-run assembly config is missing", str(context.exception))

    def test_read_json_object_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                read_json_object(path, "Real-run assembly config")

        self.assertIn("Real-run assembly config is not valid JSON", str(context.exception))

    def test_read_json_object_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                read_json_object(path, "Real-run assembly config")

        self.assertIn("Real-run assembly config must be a JSON object", str(context.exception))

    def test_load_pipeline_stages_reports_bad_component_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid_json = root / "bad_pipeline.json"
            invalid_json.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_pipeline_stages(invalid_json)

            self.assertIn("Real-run component pipeline is not valid JSON", str(context.exception))

            non_object = root / "list_pipeline.json"
            non_object.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_pipeline_stages(non_object)

            self.assertIn("Real-run component pipeline must be a JSON object", str(context.exception))

            bad_stages = root / "bad_stages.json"
            write_json(bad_stages, {"stages": {}})

            with self.assertRaises(SystemExit) as context:
                load_pipeline_stages(bad_stages)

            self.assertIn("Real-run component pipeline stages must be a list", str(context.exception))

    def test_build_manifest_reports_bad_component_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_component_files(root)
            bad_manifest = root / "bad_manifest.json"
            bad_manifest.write_text("{bad", encoding="utf-8")
            config["data_manifest"] = str(bad_manifest)

            with self.assertRaises(SystemExit) as context:
                build_manifest(config)

        self.assertIn("Real-run component manifest is not valid JSON", str(context.exception))

    def test_build_pipeline_assembles_ordered_real_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_component_files(root)
            pipeline = build_pipeline(config)
            names = [stage["name"] for stage in pipeline["stages"]]

            self.assertLess(names.index("normalize_data"), names.index("audit_data_readiness"))
            self.assertLess(names.index("audit_data_readiness"), names.index("filter_rewardbench_sft_proxy_train_holdout_overlap"))
            self.assertLess(
                names.index("filter_rewardbench_sft_proxy_train_holdout_overlap"),
                names.index("audit_clean_proxy_train_vs_hard_gold"),
            )
            self.assertLess(names.index("audit_clean_proxy_train_vs_hard_gold"), names.index("preflight"))
            self.assertLess(names.index("filter_rewardbench_sft_proxy_train_holdout_overlap"), names.index("preflight"))
            self.assertLess(names.index("audit_data_readiness"), names.index("preflight"))
            self.assertLess(names.index("preflight"), names.index("api_budget"))
            self.assertLess(names.index("api_budget"), names.index("build_sft_all_domains"))
            self.assertLess(names.index("build_sft_all_domains"), names.index("training_commands"))
            self.assertLess(names.index("filter_rewardbench_sft_proxy_train_holdout_overlap"), names.index("holdout_contamination_audit"))
            self.assertLess(names.index("convert_proxy_gold_to_verl"), names.index("holdout_contamination_audit"))
            self.assertLess(names.index("holdout_contamination_audit"), names.index("training_commands"))
            self.assertLess(names.index("holdout_contamination_audit"), names.index("audit_rewardbench_downstream_holdout_contamination"))
            self.assertLess(names.index("audit_rewardbench_downstream_holdout_contamination"), names.index("training_commands"))
            self.assertLess(names.index("training_commands"), names.index("training_completion_gate"))
            self.assertLess(names.index("training_completion_gate"), names.index("generate_model_rubrics_rubricbench"))
            self.assertLess(names.index("training_completion_gate"), names.index("reward_component_ablation_training_gate"))
            self.assertLess(
                names.index("reward_component_ablation_training_gate"),
                names.index("api_budget_reward_component_ablation_rubrics"),
            )
            self.assertLess(
                names.index("api_budget_reward_component_ablation_rubrics"),
                names.index("generate_reward_component_ablation_rubrics"),
            )
            self.assertLess(
                names.index("generate_reward_component_ablation_rubrics"),
                names.index("reward_ablation_no_red_prepare_bsc"),
            )
            self.assertLess(names.index("reward_ablation_no_red_prepare_bsc"), names.index("reward_ablation_no_red_bsc"))
            self.assertLess(names.index("reward_ablation_cov_only_bsc"), names.index("generate_model_rubrics_rubricbench"))
            self.assertLess(names.index("generate_model_rubrics_rubricbench"), names.index("api_budget_rubricbench_model_rubrics_verifier"))
            self.assertLess(names.index("api_budget_rubricbench_model_rubrics_verifier"), names.index("verify_rubricbench_model_rubrics"))
            self.assertLess(names.index("verify_rubricbench_model_rubrics"), names.index("validate_rubricbench_model_rubrics"))
            self.assertLess(names.index("generate_model_rubrics_healthbench"), names.index("verify_healthbench_model_rubrics"))
            self.assertLess(names.index("verify_healthbench_model_rubrics"), names.index("validate_healthbench_model_rubrics"))
            self.assertLess(names.index("validate_rubricbench_model_rubrics"), names.index("base_bsc"))
            self.assertLess(names.index("validate_healthbench_model_rubrics"), names.index("healthbench_base_bsc"))
            self.assertLess(names.index("base_bsc"), names.index("healthbench_base_bsc"))
            self.assertLess(names.index("base_bsc"), names.index("judgebench_base_downstream"))
            self.assertLess(names.index("generate_model_rubrics_healthbench"), names.index("healthbench_base_bsc"))
            self.assertLess(names.index("judgebench_base_downstream"), names.index("healthbench_base_bsc"))
            self.assertLess(names.index("healthbench_base_bsc"), names.index("evidence_real"))
            self.assertLess(names.index("healthbench_base_bsc"), names.index("convert_healthbench_hard_policy_rlvr_data"))
            self.assertLess(names.index("convert_healthbench_hard_policy_rlvr_data"), names.index("downstream_rlvr_commands"))
            self.assertLess(names.index("convert_arenahard_policy_rlvr_data"), names.index("downstream_rlvr_commands"))
            self.assertLess(names.index("healthbench_base_bsc"), names.index("downstream_rlvr_commands"))
            self.assertLess(names.index("downstream_rlvr_commands"), names.index("downstream_rlvr_completion_gate"))
            self.assertLess(names.index("downstream_rlvr_completion_gate"), names.index("evidence_real"))
            self.assertLess(names.index("evidence_real"), names.index("final_paper_export"))
            self.assertLess(names.index("final_paper_export"), names.index("sync_paper_artifacts"))
            self.assertLess(names.index("sync_paper_artifacts"), names.index("paper_asset_index_check_real"))
            self.assertLess(names.index("paper_asset_index_check_real"), names.index("submission_readiness"))
            self.assertLess(names.index("submission_readiness"), names.index("rebuttal_pack_real"))
            self.assertLess(names.index("rebuttal_pack_real"), names.index("submission_gap_report_real"))
            self.assertLess(names.index("submission_gap_report_real"), names.index("dashboard_real"))
            self.assertLess(names.index("dashboard_real"), names.index("result_card_real"))
            self.assertLess(names.index("result_card_real"), names.index("sync_result_card_real"))
            self.assertLess(names.index("sync_result_card_real"), names.index("paper_asset_index_check_final_real"))

            preflight = next(stage for stage in pipeline["stages"] if stage["name"] == "preflight")
            self.assertTrue(preflight["args"]["strict"])
            convert_verl = next(stage for stage in pipeline["stages"] if stage["name"] == "convert_proxy_gold_to_verl")
            self.assertEqual(convert_verl["type"], "convert_verl")
            self.assertGreaterEqual(convert_verl["args"]["min_records"], 1000)
            generation = next(stage for stage in pipeline["stages"] if stage["name"] == "generate_model_rubrics_rubricbench")
            self.assertEqual(generation["args"]["output"], "data/processed/rubricbench_model_rubrics.jsonl")
            self.assertTrue(generation["args"]["resume"])
            self.assertEqual(
                generation["args"]["require_preflight_report"],
                "outputs/preflight/real_run_preflight.json",
            )
            health_generation = next(stage for stage in pipeline["stages"] if stage["name"] == "generate_model_rubrics_healthbench")
            self.assertEqual(health_generation["args"]["input"], "data/processed/healthbench_queries.jsonl")
            self.assertEqual(health_generation["args"]["output"], "data/processed/healthbench_model_rubrics.jsonl")
            verifier = next(stage for stage in pipeline["stages"] if stage["name"] == "verify_rubricbench_model_rubrics")
            self.assertEqual(verifier["type"], "filter_verifier")
            self.assertTrue(verifier["args"]["annotate_only"])
            self.assertEqual(verifier["args"]["input"], "data/processed/rubricbench_model_rubrics.jsonl")
            self.assertEqual(verifier["args"]["output"], "data/processed/rubricbench_model_rubrics.jsonl")
            self.assertEqual(
                verifier["args"]["require_budget_report"],
                "outputs/api_budget/rubricbench_model_rubrics_verifier_budget.json",
            )
            self.assertEqual(
                verifier["args"]["require_preflight_report"],
                "outputs/preflight/real_run_preflight.json",
            )
            training = next(stage for stage in pipeline["stages"] if stage["name"] == "training_commands")
            self.assertEqual(training["args"]["config"], "configs/training_commands.example.json")
            self.assertEqual(training["args"]["output_dir"], "outputs/training_commands")
            holdout_audit = next(stage for stage in pipeline["stages"] if stage["name"] == "holdout_contamination_audit")
            self.assertEqual(holdout_audit["type"], "holdout_contamination_audit")
            self.assertEqual(holdout_audit["args"]["holdout"], "data/processed/splits/rubricbench_gold_test_main.jsonl")
            self.assertEqual(holdout_audit["args"]["output"], "outputs/contamination_audit/hard_gold_holdout_contamination.json")
            self.assertIn(
                "rewardbench_sft_proxy_train=data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
                holdout_audit["args"]["training"],
            )
            self.assertTrue(holdout_audit["args"]["strict"])
            downstream_holdout_audit = next(
                stage for stage in pipeline["stages"] if stage["name"] == "audit_rewardbench_downstream_holdout_contamination"
            )
            self.assertEqual(downstream_holdout_audit["type"], "holdout_contamination_audit")
            self.assertEqual(
                downstream_holdout_audit["args"]["holdout"],
                "data/processed/splits/rewardbench_pref_downstream_holdout.jsonl",
            )
            self.assertEqual(
                downstream_holdout_audit["args"]["output"],
                "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
            )
            self.assertIn(
                "rewardbench_sft_proxy_train=data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
                downstream_holdout_audit["args"]["training"],
            )
            self.assertTrue(downstream_holdout_audit["args"]["strict"])
            gate = next(stage for stage in pipeline["stages"] if stage["name"] == "training_completion_gate")
            self.assertEqual(gate["type"], "manual_gate")
            self.assertTrue(gate["args"]["strict"])
            self.assertIn("outputs/training_commands/training_done.json:sft_checkpoint,rl_checkpoint", gate["args"]["required_json"][0])
            self.assertIn("serving.base,serving.sft_only,serving.sft_rl", gate["args"]["required_json"][0])
            self.assertIn("sft_config_sha256,grpo_config_sha256,sft_data_sha256", gate["args"]["required_json"][0])
            self.assertIn("rl_data_sha256,rl_data_report_sha256,reward_function_sha256", gate["args"]["required_json"][0])
            self.assertIn("served_methods,served_generators", gate["args"]["required_json"][0])
            self.assertIn(
                "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
                gate["args"]["required_json_contains"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl",
                gate["args"]["required_json_contains"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:rl_data=data/processed/proxy_gold_verl.parquet",
                gate["args"]["required_json_equals"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:sft_config_sha256=configs/llamafactory_sft.local.yaml",
                gate["args"]["required_json_sha256"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:grpo_config_sha256=configs/verl_grpo_bsc.local.yaml",
                gate["args"]["required_json_sha256"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:sft_data_sha256=data/processed/blindspot_sft.jsonl",
                gate["args"]["required_json_sha256"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:rl_data_sha256=data/processed/proxy_gold_verl.parquet",
                gate["args"]["required_json_sha256"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:rl_data_report_sha256=outputs/sft_data/proxy_gold_verl_report.json",
                gate["args"]["required_json_sha256"],
            )
            self.assertIn(
                "outputs/training_commands/training_done.json:reward_function_sha256=src/blindspot_rl/verl_reward.py",
                gate["args"]["required_json_sha256"],
            )
            self.assertIn(
                "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
                gate["args"]["required_json_sha256"],
            )
            instructions = " ".join(gate["args"]["instructions"])
            self.assertIn("holdout-clean teacher input and .unfiltered SFT/proxy-gold outputs", instructions)
            self.assertIn("final post-SFT clean proxy-gold file", instructions)
            self.assertNotIn("current clean proxy-gold files", instructions)
            downstream_rlvr = next(stage for stage in pipeline["stages"] if stage["name"] == "downstream_rlvr_commands")
            self.assertEqual(downstream_rlvr["type"], "downstream_rlvr_commands")
            self.assertEqual(downstream_rlvr["args"]["config"], "configs/downstream_rlvr_commands.example.json")
            downstream_gate = next(stage for stage in pipeline["stages"] if stage["name"] == "downstream_rlvr_completion_gate")
            self.assertEqual(downstream_gate["type"], "manual_gate")
            self.assertIn("outputs/policy_rlvr/downstream_rlvr_done.json", downstream_gate["args"]["required_path"])
            self.assertIn("outputs/policy_rlvr/downstream_rlvr_done.json:healthbench_hard_policy", downstream_gate["args"]["required_json"][0])
            self.assertIn(
                "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.train_data=data/processed/healthbench_hard_policy_rlvr.parquet",
                downstream_gate["args"]["required_json_equals"],
            )
            self.assertIn(
                "benchmarks.healthbench_hard.criteria_policy_checkpoint",
                downstream_gate["args"]["required_json"][0],
            )
            self.assertIn(
                "benchmarks.arenahard.criteria_policy_checkpoint",
                downstream_gate["args"]["required_json"][0],
            )
            self.assertIn(
                "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
                downstream_gate["args"]["required_json_equals"],
            )
            self.assertIn(
                "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.arenahard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
                downstream_gate["args"]["required_json_equals"],
            )
            self.assertNotIn("benchmarks.healthbench_hard.rubric_generator", downstream_gate["args"]["required_json"][0])
            self.assertNotIn("benchmarks.arenahard.rubric_generator", downstream_gate["args"]["required_json"][0])
            self.assertFalse(
                any("benchmarks.healthbench_hard.rubric_generator" in item for item in downstream_gate["args"]["required_json_equals"])
            )
            self.assertFalse(
                any("benchmarks.arenahard.rubric_generator" in item for item in downstream_gate["args"]["required_json_equals"])
            )
            sync_paper = next(stage for stage in pipeline["stages"] if stage["name"] == "sync_paper_artifacts")
            final_export = next(stage for stage in pipeline["stages"] if stage["name"] == "final_paper_export")
            self.assertEqual(final_export["type"], "export")
            self.assertEqual(final_export["args"]["output_dir"], "outputs/paper_artifacts")
            self.assertEqual(final_export["args"]["ablation_csv"], "outputs/bsc_ablation/ablation_summary.csv")
            self.assertEqual(
                final_export["args"]["teacher_union_csv"],
                "outputs/teacher_union_ablation/teacher_union_ablation.csv",
            )
            self.assertEqual(
                final_export["args"]["verifier_filter_csv"],
                "outputs/verifier_filter_ablation/verifier_filter_ablation.csv",
            )
            self.assertEqual(
                final_export["args"]["downstream_table_csv"],
                [
                    "RewardBench=outputs/matrix_real/main_table.csv",
                    "JudgeBench=outputs/matrix_judgebench/main_table.csv",
                    "RewardBench-2=outputs/matrix_rewardbench2/main_table.csv",
                ],
            )
            self.assertEqual(
                final_export["args"]["transition_summary_json"],
                [
                    "outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json",
                    "outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json",
                ],
            )
            self.assertEqual(final_export["args"]["semantic_space_dir"], "outputs/matrix_real/semantic_space")
            self.assertEqual(sync_paper["type"], "sync_paper")
            self.assertEqual(sync_paper["args"]["artifacts_dir"], "outputs/paper_artifacts")
            self.assertEqual(sync_paper["args"]["paper_dir"], "paper")
            self.assertNotIn("required_file", sync_paper["args"])
            asset_check = next(stage for stage in pipeline["stages"] if stage["name"] == "paper_asset_index_check_real")
            self.assertEqual(asset_check["type"], "paper_asset_index_check")
            self.assertEqual(asset_check["args"]["asset_index"], "paper/asset_index.md")
            self.assertEqual(asset_check["args"]["output"], "outputs/paper_artifacts/paper_asset_index_check.json")
            self.assertTrue(asset_check["args"]["strict"])
            result_card_sync = next(stage for stage in pipeline["stages"] if stage["name"] == "sync_result_card_real")
            self.assertEqual(result_card_sync["type"], "sync_paper")
            self.assertEqual(result_card_sync["args"]["artifacts_dir"], "outputs/paper_artifacts")
            self.assertEqual(
                result_card_sync["args"]["extra_doc"],
                [
                    "outputs/dashboard/real_run_dashboard.json",
                    "outputs/dashboard/real_run_dashboard.md",
                    "outputs/evidence_real/evidence_matrix.json",
                    "outputs/evidence_real/evidence_matrix.csv",
                    "outputs/evidence_real/evidence_matrix.md",
                    "outputs/result_card/result_card.json",
                    "outputs/result_card/result_card.md",
                    "outputs/submission_readiness/readiness_report.json",
                    "outputs/submission_readiness/readiness_report.md",
                    "outputs/rebuttal_pack/rebuttal_pack.json",
                    "outputs/rebuttal_pack/rebuttal_pack.md",
                    "outputs/rebuttal_pack/rebuttal_pack_manifest.json",
                    "outputs/submission_readiness/gap_report/submission_gap_report.json",
                    "outputs/submission_readiness/gap_report/submission_gap_report.md",
                ],
            )
            self.assertNotIn("required_file", result_card_sync["args"])
            rebuttal = next(stage for stage in pipeline["stages"] if stage["name"] == "rebuttal_pack_real")
            self.assertEqual(rebuttal["type"], "rebuttal_pack")
            self.assertEqual(rebuttal["args"]["evidence_matrix"], "outputs/evidence_real/evidence_matrix.json")
            self.assertEqual(rebuttal["args"]["readiness_report"], "outputs/submission_readiness/readiness_report.json")
            self.assertEqual(rebuttal["args"]["output_dir"], "outputs/rebuttal_pack")
            gap_report = next(stage for stage in pipeline["stages"] if stage["name"] == "submission_gap_report_real")
            self.assertEqual(gap_report["type"], "submission_gap_report")
            self.assertEqual(gap_report["args"]["readiness_report"], "outputs/submission_readiness/readiness_report.json")
            self.assertEqual(gap_report["args"]["evidence_matrix"], "outputs/evidence_real/evidence_matrix.json")
            self.assertEqual(gap_report["args"]["rebuttal_manifest"], "outputs/rebuttal_pack/rebuttal_pack_manifest.json")
            self.assertEqual(
                gap_report["args"]["preflight_report"],
                [
                    "outputs/preflight/real_run_preflight.json",
                    "outputs/preflight/sft_data_preflight.json",
                ],
            )
            self.assertEqual(
                gap_report["args"]["gate_report"],
                [
                    "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                    "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
                    "outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
                    "outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
                ],
            )
            self.assertEqual(gap_report["args"]["output_dir"], "outputs/submission_readiness/gap_report")
            final_asset_check = next(stage for stage in pipeline["stages"] if stage["name"] == "paper_asset_index_check_final_real")
            self.assertEqual(final_asset_check["type"], "paper_asset_index_check")
            self.assertEqual(final_asset_check["args"]["output_md"], "outputs/paper_artifacts/paper_asset_index_check.md")
            readiness = next(stage for stage in pipeline["stages"] if stage["name"] == "submission_readiness")
            self.assertEqual(readiness["type"], "submission_readiness")
            self.assertTrue(readiness["args"]["strict"])
            result_card = next(stage for stage in pipeline["stages"] if stage["name"] == "result_card_real")
            self.assertEqual(result_card["type"], "result_card")
            self.assertTrue(result_card["args"]["strict"])

    def test_build_manifest_merges_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = write_component_files(root)
            manifest = build_manifest(config)
            required = manifest["required_files"]

            self.assertIn("data/processed/rubricbench_gold.jsonl", required)
            self.assertIn("outputs/data_readiness_audit.json", required)
            self.assertIn("outputs/preflight/real_run_preflight.json", required)
            self.assertIn("outputs/api_budget/model_rubrics_budget.json", required)
            self.assertIn("data/processed/blindspot_sft.jsonl", required)
            self.assertIn("data/processed/healthbench_proxy_gold.jsonl", required)
            self.assertIn("data/processed/writingbench_proxy_gold.jsonl", required)
            self.assertIn("outputs/verifier_stats.jsonl", required)
            self.assertIn("data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl", required)
            self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json", required)
            self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json", required)
            self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_hard_gold_overlaps.csv", required)
            self.assertIn("data/processed/proxy_gold_verl.parquet", required)
            self.assertIn("data/processed/rubricbench_model_rubrics.jsonl", required)
            self.assertIn("data/processed/healthbench_model_rubrics.jsonl", required)
            self.assertIn("outputs/api_budget/rubricbench_model_rubrics_verifier_budget.json", required)
            self.assertIn("outputs/api_budget/rubricbench_model_rubrics_verifier_budget.md", required)
            self.assertIn("outputs/verifier/rubricbench_model_rubrics_stats.jsonl", required)
            self.assertIn("outputs/validation/rubricbench_model_rubrics/validation_report.json", required)
            self.assertIn("outputs/validation/healthbench_model_rubrics/validation_report.json", required)
            self.assertIn("outputs/training_commands/run_sft.sh", required)
            self.assertIn("outputs/training_commands/run_grpo.sh", required)
            self.assertIn("outputs/training_commands/run_grpo_no_red.sh", required)
            self.assertIn("outputs/training_commands/run_grpo_no_valid.sh", required)
            self.assertIn("outputs/training_commands/run_grpo_no_verifier.sh", required)
            self.assertIn("outputs/training_commands/run_grpo_cov_only.sh", required)
            self.assertIn("outputs/training_commands/training_done.template.json", required)
            self.assertIn("outputs/reward_component_training_ablation/training_done.template.json", required)
            self.assertIn("outputs/training_commands/training_manifest.json", required)
            self.assertIn("outputs/reward_component_training_ablation/training_gate.json", required)
            self.assertIn("outputs/reward_component_training_ablation/training_gate.md", required)
            self.assertIn("outputs/api_budget/reward_component_ablation_rubrics_budget.json", required)
            self.assertIn("outputs/api_budget/reward_component_ablation_rubrics_budget.md", required)
            self.assertIn("data/processed/reward_component_training_ablation/model_rubrics.jsonl", required)
            for variant in ["no_red", "no_valid", "no_verifier", "cov_only"]:
                with self.subTest(reward_ablation_variant=variant):
                    self.assertIn(f"data/processed/reward_component_training_ablation/{variant}/bsc_eval.jsonl", required)
                    self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc_join_report.json", required)
                    self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc/summary.json", required)
                    self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc/per_item.csv", required)
                    self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc/per_item.jsonl", required)
            self.assertIn("outputs/contamination_audit/hard_gold_holdout_contamination.json", required)
            self.assertIn("outputs/contamination_audit/hard_gold_holdout_overlaps.csv", required)
            self.assertIn("outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json", required)
            self.assertIn("outputs/contamination_audit/rewardbench_downstream_holdout_overlaps.csv", required)
            self.assertIn("outputs/training_commands/training_completion_gate.json", required)
            self.assertIn("outputs/training_commands/training_completion_gate.md", required)
            self.assertIn("data/processed/healthbench_hard_policy_rlvr.parquet", required)
            self.assertIn("data/processed/arenahard_policy_rlvr.parquet", required)
            self.assertIn("outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json", required)
            self.assertIn("outputs/downstream_rlvr_commands/downstream_rlvr_done.template.json", required)
            self.assertIn("outputs/downstream_rlvr_commands/run_healthbench_hard_rlvr.sh", required)
            self.assertIn("outputs/downstream_rlvr_commands/run_healthbench_hard_eval.sh", required)
            self.assertIn("outputs/downstream_rlvr_commands/run_arenahard_rlvr.sh", required)
            self.assertIn("outputs/downstream_rlvr_commands/run_arenahard_eval.sh", required)
            self.assertIn("outputs/policy_rlvr/downstream_rlvr_completion_gate.json", required)
            self.assertIn("outputs/policy_rlvr/downstream_rlvr_completion_gate.md", required)
            self.assertIn("outputs/matrix_real/base/bsc/summary.json", required)
            self.assertIn("outputs/matrix_judgebench/judgebench_base/downstream/summary.json", required)
            self.assertIn("outputs/generalization_matrix/healthbench_base/bsc/summary.json", required)
            self.assertIn("outputs/evidence_real/evidence_matrix.json", required)
            self.assertIn("outputs/paper_artifacts/main_table.tex", required)
            self.assertIn("outputs/paper_artifacts/rl_stage_ablation_table.tex", required)
            self.assertIn("outputs/paper_artifacts/downstream_utility_table.tex", required)
            self.assertIn("outputs/paper_artifacts/ablation_table.tex", required)
            self.assertIn("outputs/paper_artifacts/teacher_union_ablation_table.tex", required)
            self.assertIn("outputs/paper_artifacts/verifier_filter_ablation_table.tex", required)
            self.assertIn("outputs/paper_artifacts/dimension_transition_table.tex", required)
            self.assertNotIn("outputs/paper_artifacts/repair_table.tex", required)
            self.assertIn("outputs/paper_artifacts/semantic_space.svg", required)
            self.assertIn("outputs/paper_artifacts/semantic_space.pdf", required)
            self.assertIn("outputs/paper_artifacts/semantic_space_points.csv", required)
            self.assertIn("outputs/paper_artifacts/semantic_space_summary.json", required)
            self.assertIn("paper/asset_index.md", required)
            self.assertIn("outputs/paper_artifacts/paper_asset_index_check.json", required)
            self.assertIn("outputs/paper_artifacts/paper_asset_index_check.md", required)
            self.assertIn("outputs/submission_readiness/readiness_report.json", required)
            self.assertIn("outputs/dashboard/real_run_dashboard.json", required)
            self.assertIn("outputs/result_card/result_card.json", required)
            self.assertIn("outputs/rebuttal_pack/rebuttal_pack.json", required)
            self.assertIn("outputs/rebuttal_pack/rebuttal_pack.md", required)
            self.assertIn("outputs/rebuttal_pack/rebuttal_pack_manifest.json", required)
            self.assertIn("outputs/submission_readiness/gap_report/submission_gap_report.json", required)
            self.assertIn("outputs/submission_readiness/gap_report/submission_gap_report.md", required)
            self.assertEqual(len(required), len(set(required)))
            self.assertEqual(manifest["summaries"][0]["name"], "base_bsc")


def write_component_files(root: Path) -> dict[str, object]:
    data_pipeline = write_json(root / "data_pipeline.json", {"stages": [{"name": "normalize_data", "type": "normalize", "args": {}}]})
    data_manifest = write_json(root / "data_manifest.json", {"required_files": ["data/processed/rubricbench_gold.jsonl"], "summaries": []})
    preflight = write_json(
        root / "preflight.json",
        {
            "stages": [
                {
                    "name": "preflight",
                    "type": "preflight",
                    "args": {
                        "output": "outputs/preflight/real_run_preflight.json",
                        "output_md": "outputs/preflight/real_run_preflight.md",
                        "strict": True,
                    },
                }
            ]
        },
    )
    api_budget = write_json(
        root / "api_budget.json",
        {
            "stages": [
                {
                    "name": "api_budget",
                    "type": "api_budget",
                    "args": {
                        "output": "outputs/api_budget/model_rubrics_budget.json",
                        "output_md": "outputs/api_budget/model_rubrics_budget.md",
                    },
                }
            ]
        },
    )
    sft_data_pipeline = write_json(
        root / "sft_data.json",
        {
            "stages": [
                {
                    "name": "build_sft_all_domains",
                    "type": "build_sft",
                    "args": {"sft_output": "data/processed/blindspot_sft.jsonl"},
                },
                {
                    "name": "convert_proxy_gold_to_verl",
                    "type": "convert_verl",
                    "args": {
                        "input": "data/processed/proxy_gold.jsonl",
                        "output": "data/processed/proxy_gold_verl.parquet",
                        "data_source": "multi_teacher_proxy",
                        "min_records": 1000,
                    },
                }
            ]
        },
    )
    sft_data_manifest = write_json(
        root / "sft_manifest.json",
        {
            "required_files": [
                "data/processed/blindspot_sft.jsonl",
                "data/processed/healthbench_proxy_gold.jsonl",
                "data/processed/writingbench_proxy_gold.jsonl",
                "data/processed/proxy_gold_verl.parquet",
                "outputs/verifier_stats.jsonl",
            ],
            "summaries": [],
        },
    )
    validation = write_json(
        root / "validation.json",
        {
            "stages": [
                {
                    "name": "validate_rubricbench_model_rubrics",
                    "type": "validate_rubrics",
                    "args": {"output_dir": "outputs/validation/rubricbench_model_rubrics"},
                },
                {
                    "name": "validate_healthbench_model_rubrics",
                    "type": "validate_rubrics",
                    "args": {"output_dir": "outputs/validation/healthbench_model_rubrics"},
                }
            ]
        },
    )
    matrix_pipeline = write_json(root / "matrix_pipeline.json", {"stages": [{"name": "base_bsc", "type": "bsc", "args": {}}]})
    matrix_manifest = write_json(
        root / "matrix_manifest.json",
        {
            "required_files": ["outputs/matrix_real/base/bsc/summary.json"],
            "summaries": [{"name": "base_bsc", "path": "outputs/matrix_real/base/bsc/summary.json"}],
        },
    )
    generalization_pipeline = write_json(
        root / "generalization_pipeline.json",
        {"stages": [{"name": "healthbench_base_bsc", "type": "bsc", "args": {}}]},
    )
    judgebench_pipeline = write_json(
        root / "judgebench_pipeline.json",
        {"stages": [{"name": "judgebench_base_downstream", "type": "downstream", "args": {}}]},
    )
    judgebench_manifest = write_json(
        root / "judgebench_manifest.json",
        {
            "required_files": ["outputs/matrix_judgebench/judgebench_base/downstream/summary.json"],
            "summaries": [
                {
                    "name": "judgebench_base_downstream",
                    "path": "outputs/matrix_judgebench/judgebench_base/downstream/summary.json",
                }
            ],
        },
    )
    generalization_manifest = write_json(
        root / "generalization_manifest.json",
        {
            "required_files": ["outputs/generalization_matrix/healthbench_base/bsc/summary.json"],
            "summaries": [{"name": "healthbench_base_bsc", "path": "outputs/generalization_matrix/healthbench_base/bsc/summary.json"}],
        },
    )
    downstream_rlvr_data_pipeline = write_json(
        root / "downstream_rlvr_data.json",
        {
            "stages": [
                {
                    "name": "convert_healthbench_hard_policy_rlvr_data",
                    "type": "convert_policy_rlvr",
                    "args": {"output": "data/processed/healthbench_hard_policy_rlvr.parquet"},
                },
                {
                    "name": "convert_arenahard_policy_rlvr_data",
                    "type": "convert_policy_rlvr",
                    "args": {"output": "data/processed/arenahard_policy_rlvr.parquet"},
                },
            ]
        },
    )
    readiness = write_json(
        root / "readiness.json",
        {
            "stages": [
                {
                    "name": "submission_readiness",
                    "type": "submission_readiness",
                    "args": {
                        "output_json": "outputs/submission_readiness/readiness_report.json",
                        "output_md": "outputs/submission_readiness/readiness_report.md",
                        "strict": True,
                    },
                }
            ]
        },
    )
    return {
        "data_pipeline": str(data_pipeline),
        "data_manifest": str(data_manifest),
        "data_audit": {"enabled": True, "manifest": str(data_manifest), "output": "outputs/data_readiness_audit.json"},
        "holdout_contamination_filters": [
            {
                "enabled": True,
                "name": "filter_rewardbench_sft_proxy_train_holdout_overlap",
                "holdout": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                "input": "data/processed/splits/rewardbench_pref_sft_proxy_train.jsonl",
                "output": "data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
                "report": "outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json",
                "query_keys": ["query", "prompt", "input", "question", "instruction"],
                "strict": True,
            }
        ],
        "preflight_pipeline": str(preflight),
        "api_budget_pipeline": str(api_budget),
        "sft_data_pipeline": str(sft_data_pipeline),
        "sft_data_manifest": str(sft_data_manifest),
        "pre_sft_holdout_contamination_audits": [
            {
                "enabled": True,
                "name": "audit_clean_proxy_train_vs_hard_gold",
                "holdout": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                "training": [
                    "rewardbench_sft_proxy_train_clean=data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl"
                ],
                "query_keys": ["query", "prompt", "input", "question", "instruction"],
                "output": "outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json",
                "output_csv": "outputs/contamination_audit/clean_proxy_train_vs_hard_gold_overlaps.csv",
                "strict": True,
            }
        ],
        "holdout_contamination_audit": {
            "enabled": True,
            "holdout": "data/processed/splits/rubricbench_gold_test_main.jsonl",
            "training": [
                "rubricbench_train_seed=data/processed/splits/rubricbench_gold_train_seed.jsonl",
                "rewardbench_sft_proxy_train=data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
                "blindspot_sft=data/processed/blindspot_sft.jsonl",
                "proxy_gold=data/processed/proxy_gold.jsonl",
            ],
            "query_keys": ["query", "prompt", "input", "question", "instruction"],
            "output": "outputs/contamination_audit/hard_gold_holdout_contamination.json",
            "output_csv": "outputs/contamination_audit/hard_gold_holdout_overlaps.csv",
            "strict": True,
        },
        "holdout_contamination_audits": [
            {
                "enabled": True,
                "name": "audit_rewardbench_downstream_holdout_contamination",
                "holdout": "data/processed/splits/rewardbench_pref_downstream_holdout.jsonl",
                "training": [
                    "rubricbench_train_seed=data/processed/splits/rubricbench_gold_train_seed.jsonl",
                    "rewardbench_sft_proxy_train=data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
                    "blindspot_sft=data/processed/blindspot_sft.jsonl",
                    "proxy_gold=data/processed/proxy_gold.jsonl",
                ],
                "query_keys": ["query", "prompt", "input", "question", "instruction"],
                "output": "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
                "output_csv": "outputs/contamination_audit/rewardbench_downstream_holdout_overlaps.csv",
                "strict": True,
            }
        ],
        "model_generation": {
            "enabled": True,
            "providers": "configs/generators.local.jsonl",
            "output": "data/processed/rubricbench_model_rubrics.jsonl",
            "resume": True,
            "domains": [
                {
                    "name": "generate_model_rubrics_rubricbench",
                    "input": "data/processed/rubricbench_queries.jsonl",
                    "output": "data/processed/rubricbench_model_rubrics.jsonl",
                    "data_source": "rubricbench",
                },
                {
                    "name": "generate_model_rubrics_healthbench",
                    "input": "data/processed/healthbench_queries.jsonl",
                    "output": "data/processed/healthbench_model_rubrics.jsonl",
                    "data_source": "healthbench",
                },
            ],
        },
        "model_verification": {
            "enabled": True,
            "provider": "configs/verifier.local.jsonl",
            "mode": "api",
            "annotate_only": True,
            "budget_output_template": "outputs/api_budget/{stem}_verifier_budget.json",
            "budget_output_md_template": "outputs/api_budget/{stem}_verifier_budget.md",
            "stats_output_template": "outputs/verifier/{stem}_stats.jsonl",
        },
        "rubric_validation_pipeline": str(validation),
        "training_commands": {
            "enabled": True,
            "config": "configs/training_commands.example.json",
            "output_dir": "outputs/training_commands",
        },
        "training_completion_gate": {
            "enabled": True,
            "name": "training_completion_gate",
            "required_paths": [
                "outputs/checkpoints/evaluation_criteria_policy_sft",
                "outputs/sft_data/proxy_gold_verl_report.json",
            ],
            "required_json": [
                "outputs/training_commands/training_done.json:sft_checkpoint,rl_checkpoint,served_methods,served_generators,serving.base,serving.sft_only,serving.sft_rl,operator,date,sft_config,grpo_config,sft_data,rl_data,rl_data_report,reward_function,sft_config_sha256,grpo_config_sha256,sft_data_sha256,rl_data_sha256,rl_data_report_sha256,reward_function_sha256",
                "outputs/sft_data/proxy_gold_verl_report.json:input,input_sha256,output,output_sha256,n_records,data_source,forbidden_source_markers,forbidden_splits",
            ],
            "required_json_contains": [
                "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
                "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl"
            ],
            "required_json_equals": [
                "outputs/training_commands/training_done.json:rl_data=data/processed/proxy_gold_verl.parquet",
                "outputs/training_commands/training_done.json:rl_data_report=outputs/sft_data/proxy_gold_verl_report.json",
                "outputs/sft_data/proxy_gold_verl_report.json:input=data/processed/proxy_gold.jsonl",
                "outputs/sft_data/proxy_gold_verl_report.json:output=data/processed/proxy_gold_verl.parquet",
                "outputs/training_commands/training_done.json:reward_function=src/blindspot_rl/verl_reward.py:compute_score",
            ],
            "required_json_sha256": [
                "outputs/training_commands/training_done.json:sft_config_sha256=configs/llamafactory_sft.local.yaml",
                "outputs/training_commands/training_done.json:grpo_config_sha256=configs/verl_grpo_bsc.local.yaml",
                "outputs/training_commands/training_done.json:sft_data_sha256=data/processed/blindspot_sft.jsonl",
                "outputs/training_commands/training_done.json:rl_data_sha256=data/processed/proxy_gold_verl.parquet",
                "outputs/training_commands/training_done.json:rl_data_report_sha256=outputs/sft_data/proxy_gold_verl_report.json",
                "outputs/training_commands/training_done.json:reward_function_sha256=src/blindspot_rl/verl_reward.py",
                "outputs/sft_data/proxy_gold_verl_report.json:input_sha256=data/processed/proxy_gold.jsonl",
                "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
            ],
            "instructions": [
                "Keep outputs/sft_data/proxy_gold_build_report.json bound to the holdout-clean teacher input and .unfiltered SFT/proxy-gold outputs; keep outputs/sft_data/proxy_gold_verl_report.json bound to the final post-SFT clean proxy-gold file."
            ],
            "output": "outputs/training_commands/training_completion_gate.json",
            "output_md": "outputs/training_commands/training_completion_gate.md",
            "strict": True,
        },
        "reward_component_ablation_eval": {
            "enabled": True,
            "input": "data/processed/rubricbench_queries.jsonl",
            "gold": "data/processed/splits/rubricbench_gold_test_main.jsonl",
            "providers": "configs/generators_reward_ablation.local.jsonl",
            "predictions": "data/processed/reward_component_training_ablation/model_rubrics.jsonl",
            "eval_data_dir": "data/processed/reward_component_training_ablation",
            "output_dir": "outputs/reward_component_training_ablation",
            "variants": ["no_red", "no_valid", "no_verifier", "cov_only"],
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "coverage_tau": 0.75,
            "redundancy_tau": 0.85,
            "budget": {
                "enabled": True,
                "name": "api_budget_reward_component_ablation_rubrics",
                "output": "outputs/api_budget/reward_component_ablation_rubrics_budget.json",
                "output_md": "outputs/api_budget/reward_component_ablation_rubrics_budget.md",
                "strict": True,
            },
            "gate": {
                "enabled": True,
                "name": "reward_component_ablation_training_gate",
                "required_paths": [
                    "outputs/checkpoints/evaluation_criteria_policy_rl_no_red",
                    "outputs/reward_component_training_ablation/training_done.json",
                ],
                "required_json": [
                    "outputs/reward_component_training_ablation/training_done.json:reward_variants,variants.no_red.checkpoint,variants.no_red.serving,variants.no_red.env.BSC_W_RED"
                ],
                "required_json_contains": [
                    "outputs/reward_component_training_ablation/training_done.json:reward_variants=full,no_red,no_valid,no_verifier,cov_only"
                ],
                "required_json_equals": [
                    "outputs/reward_component_training_ablation/training_done.json:variants.no_red.env.BSC_W_RED=0.0"
                ],
                "required_json_sha256": [
                    "outputs/reward_component_training_ablation/training_done.json:variants.no_red.rl_data_sha256=data/processed/proxy_gold_verl.parquet"
                ],
                "output": "outputs/reward_component_training_ablation/training_gate.json",
                "output_md": "outputs/reward_component_training_ablation/training_gate.md",
                "strict": True,
            },
        },
        "matrix_pipeline": str(matrix_pipeline),
        "matrix_manifest": str(matrix_manifest),
        "extra_pipelines": [
            {
                "name": "judgebench_downstream_matrix",
                "pipeline": str(judgebench_pipeline),
                "manifest": str(judgebench_manifest),
            }
        ],
        "generalization_pipeline": str(generalization_pipeline),
        "generalization_manifest": str(generalization_manifest),
        "downstream_rlvr_data_pipeline": str(downstream_rlvr_data_pipeline),
        "downstream_rlvr_commands": {
            "enabled": True,
            "config": "configs/downstream_rlvr_commands.example.json",
            "output_dir": "outputs/downstream_rlvr_commands",
        },
        "downstream_rlvr_completion_gate": {
            "enabled": True,
            "name": "downstream_rlvr_completion_gate",
            "required_paths": [
                "outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json",
                "outputs/policy_rlvr/healthbench_hard_policy",
                "outputs/policy_rlvr/healthbench_hard_eval.json",
                "outputs/policy_rlvr/arenahard_policy",
                "outputs/policy_rlvr/arenahard_eval.json",
                "outputs/policy_rlvr/downstream_rlvr_done.json",
            ],
            "required_json": [
                "outputs/policy_rlvr/downstream_rlvr_done.json:healthbench_hard_policy,healthbench_hard_eval,benchmarks.healthbench_hard.criteria_policy_checkpoint,benchmarks.arenahard.criteria_policy_checkpoint"
            ],
            "required_json_equals": [
                "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.train_data=data/processed/healthbench_hard_policy_rlvr.parquet",
                "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
                "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.arenahard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
            ],
            "output": "outputs/policy_rlvr/downstream_rlvr_completion_gate.json",
            "output_md": "outputs/policy_rlvr/downstream_rlvr_completion_gate.md",
            "strict": True,
        },
        "evidence": {"config": "configs/evidence.json", "output_dir": "outputs/evidence_real"},
        "final_paper_export": {
            "enabled": True,
            "name": "final_paper_export",
            "main_table_csv": "outputs/matrix_real/main_table.csv",
            "main_table_md": "outputs/matrix_real/main_table.md",
            "downstream_table_csv": [
                "RewardBench=outputs/matrix_real/main_table.csv",
                "JudgeBench=outputs/matrix_judgebench/main_table.csv",
                "RewardBench-2=outputs/matrix_rewardbench2/main_table.csv",
            ],
            "ablation_csv": "outputs/bsc_ablation/ablation_summary.csv",
            "teacher_union_csv": "outputs/teacher_union_ablation/teacher_union_ablation.csv",
            "verifier_filter_csv": "outputs/verifier_filter_ablation/verifier_filter_ablation.csv",
            "transition_summary_json": [
                "outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json",
                "outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json",
            ],
            "semantic_space_dir": "outputs/matrix_real/semantic_space",
            "output_dir": "outputs/paper_artifacts",
        },
        "paper_sync": {
            "enabled": True,
            "name": "sync_paper_artifacts",
            "artifacts_dir": "outputs/paper_artifacts",
            "paper_dir": "paper",
        },
        "submission_readiness_pipeline": str(readiness),
        "dashboard": {
            "config": "configs/dashboard.json",
            "output_json": "outputs/dashboard/real_run_dashboard.json",
            "output_md": "outputs/dashboard/real_run_dashboard.md",
        },
        "result_card": {
            "config": "configs/result_card.json",
            "output_json": "outputs/result_card/result_card.json",
            "output_md": "outputs/result_card/result_card.md",
        },
        "rebuttal_pack": {
            "enabled": True,
            "name": "rebuttal_pack_real",
            "evidence_matrix": "outputs/evidence_real/evidence_matrix.json",
            "readiness_report": "outputs/submission_readiness/readiness_report.json",
            "output_dir": "outputs/rebuttal_pack",
        },
        "submission_gap_report": {
            "enabled": True,
            "name": "submission_gap_report_real",
            "readiness_report": "outputs/submission_readiness/readiness_report.json",
            "evidence_matrix": "outputs/evidence_real/evidence_matrix.json",
            "rebuttal_manifest": "outputs/rebuttal_pack/rebuttal_pack_manifest.json",
            "preflight_report": [
                "outputs/preflight/real_run_preflight.json",
                "outputs/preflight/sft_data_preflight.json",
            ],
            "gate_report": [
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
                "outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
                "outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
            ],
            "output_dir": "outputs/submission_readiness/gap_report",
        },
    }


def write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
