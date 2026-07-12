from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_matrix import build_manifest, build_pipeline, load_methods_config


class BuildExperimentMatrixTest(unittest.TestCase):
    def test_load_methods_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_methods_config(Path("/tmp/missing_experiment_matrix_methods.json"))

        self.assertIn("Experiment matrix methods config is missing", str(context.exception))

    def test_load_methods_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_methods_config(path)

        self.assertIn("Experiment matrix methods config is not valid JSON", str(context.exception))

    def test_load_methods_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_methods_config(path)

        self.assertIn("Experiment matrix methods config must be a JSON object", str(context.exception))

    def test_build_pipeline_expands_method_stages(self) -> None:
        config = sample_config()
        pipeline = build_pipeline(config, manifest_path="configs/m.json")
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("base_prepare_bsc", names)
        self.assertIn("base_downstream", names)
        self.assertIn("summarize", names)
        audit = next(stage for stage in pipeline["stages"] if stage["name"] == "audit")
        self.assertEqual(audit["args"]["manifest"], "configs/m.json")
        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertNotIn("ablation_csv", export["args"])

    def test_build_manifest_contains_method_outputs(self) -> None:
        manifest = build_manifest(sample_config())
        required = manifest["required_files"]
        self.assertIn("data/processed/base/bsc_eval.jsonl", required)
        self.assertIn("outputs/base/downstream/summary.json", required)
        self.assertEqual(len(manifest["summaries"]), 2)
        self.assertIn("required_keys", manifest["summaries"][0])
        self.assertIn("embedding_model", manifest["summaries"][0]["required_keys"])
        self.assertIn("coverage_tau", manifest["summaries"][0]["required_keys"])
        self.assertIn("data_source_counts", manifest["summaries"][0]["required_keys"])
        self.assertIn("verifier_source", manifest["summaries"][0]["required_keys"])
        self.assertIn("verifier_source_counts", manifest["summaries"][0]["required_keys"])

    def test_build_pipeline_uses_shared_model_rubrics(self) -> None:
        config = sample_config()
        config["common"]["model_rubrics"] = "all_methods.jsonl"
        del config["methods"][0]["bsc_rubrics"]
        del config["methods"][0]["downstream_rubrics"]
        pipeline = build_pipeline(config)
        prepare_bsc = next(stage for stage in pipeline["stages"] if stage["name"] == "base_prepare_bsc")
        prepare_downstream = next(
            stage for stage in pipeline["stages"] if stage["name"] == "base_prepare_downstream"
        )
        self.assertEqual(prepare_bsc["args"]["predictions"], "all_methods.jsonl")
        self.assertEqual(prepare_downstream["args"]["rubrics"], "all_methods.jsonl")

    def test_build_pipeline_export_includes_optional_paper_artifacts(self) -> None:
        config = sample_config()
        config["common"]["teacher_union_csv"] = "teacher_union.csv"
        config["common"]["transition_summary_json"] = [
            "outputs/repair_sft/transition_summary.json",
            "outputs/repair_grpo/transition_summary.json",
        ]
        config["common"]["evidence_json"] = "evidence.json"
        config["common"]["evidence_csv"] = "evidence.csv"
        config["common"]["evidence_md"] = "evidence.md"
        config["common"]["downstream_table_csv"] = [
            "RewardBench=outputs/matrix_real/main_table.csv",
            "JudgeBench=outputs/matrix_judgebench/main_table.csv",
        ]
        pipeline = build_pipeline(config)
        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertEqual(export["args"]["teacher_union_csv"], "teacher_union.csv")
        self.assertEqual(
            export["args"]["transition_summary_json"],
            ["outputs/repair_sft/transition_summary.json", "outputs/repair_grpo/transition_summary.json"],
        )
        self.assertEqual(export["args"]["evidence_json"], "evidence.json")
        self.assertEqual(export["args"]["evidence_csv"], "evidence.csv")
        self.assertEqual(export["args"]["evidence_md"], "evidence.md")
        self.assertEqual(
            export["args"]["downstream_table_csv"],
            [
                "RewardBench=outputs/matrix_real/main_table.csv",
                "JudgeBench=outputs/matrix_judgebench/main_table.csv",
            ],
        )

    def test_build_pipeline_can_add_reward_component_ablation(self) -> None:
        config = sample_config()
        config["common"]["reward_ablation"] = {
            "enabled": True,
            "method": "base",
            "output_dir": "outputs/abl",
        }
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("reward_component_ablation", names)
        self.assertLess(names.index("base_bsc"), names.index("reward_component_ablation"))
        self.assertLess(names.index("reward_component_ablation"), names.index("summarize"))

        ablation = next(stage for stage in pipeline["stages"] if stage["name"] == "reward_component_ablation")
        self.assertEqual(ablation["type"], "ablation")
        self.assertEqual(ablation["args"]["input"], "data/processed/base/bsc_eval.jsonl")
        self.assertEqual(ablation["args"]["output_dir"], "outputs/abl")
        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertEqual(export["args"]["ablation_csv"], "outputs/abl/ablation_summary.csv")

    def test_build_pipeline_can_add_teacher_union_ablation(self) -> None:
        config = sample_config()
        config["common"]["teacher_union_ablation"] = {
            "enabled": True,
            "teachers": "teachers.jsonl",
            "output_dir": "outputs/teacher_union",
            "dedupe_tau": 0.9,
        }
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("teacher_union_ablation", names)
        self.assertLess(names.index("teacher_union_ablation"), names.index("summarize"))

        ablation = next(stage for stage in pipeline["stages"] if stage["name"] == "teacher_union_ablation")
        self.assertEqual(ablation["type"], "teacher_union_ablation")
        self.assertEqual(ablation["args"]["teachers"], "teachers.jsonl")
        self.assertEqual(ablation["args"]["gold"], "gold.jsonl")
        self.assertEqual(ablation["args"]["dedupe_tau"], 0.9)
        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertEqual(export["args"]["teacher_union_csv"], "outputs/teacher_union/teacher_union_ablation.csv")

    def test_build_pipeline_can_add_verifier_filter_ablation(self) -> None:
        config = sample_config()
        config["common"]["verifier_filter_ablation"] = {
            "enabled": True,
            "raw_teachers": "raw_teachers.jsonl",
            "filtered_teachers": "filtered_teachers.jsonl",
            "output_dir": "outputs/verifier_filter",
            "dedupe_tau": 0.9,
        }
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("verifier_filter_ablation", names)
        self.assertLess(names.index("verifier_filter_ablation"), names.index("summarize"))

        ablation = next(stage for stage in pipeline["stages"] if stage["name"] == "verifier_filter_ablation")
        self.assertEqual(ablation["type"], "verifier_filter_ablation")
        self.assertEqual(ablation["args"]["raw_teachers"], "raw_teachers.jsonl")
        self.assertEqual(ablation["args"]["filtered_teachers"], "filtered_teachers.jsonl")
        self.assertEqual(ablation["args"]["gold"], "gold.jsonl")
        self.assertEqual(ablation["args"]["dedupe_tau"], 0.9)

        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertEqual(export["args"]["verifier_filter_csv"], "outputs/verifier_filter/verifier_filter_ablation.csv")

        manifest = build_manifest(config)
        self.assertIn("outputs/verifier_filter/verifier_filter_ablation.csv", manifest["required_files"])
        self.assertIn("outputs/verifier_filter/verifier_filter_ablation.json", manifest["required_files"])

    def test_build_pipeline_can_add_dimension_transition_prefixed_stages(self) -> None:
        config = sample_config()
        config["methods"].append(
            {
                "name": "sft_rl",
                "model_filter": "sft_rl",
                "bsc_rubrics": "sft_rl_bsc.jsonl",
                "downstream_rubrics": "sft_rl_down.jsonl",
                "skip_downstream": True,
            }
        )
        config["common"]["dimension_transition"] = {
            "enabled": True,
            "baseline": "base",
            "candidates": ["sft_rl"],
            "name_prefix": "dimension_transition",
            "output_dir": "outputs/dimension_transition",
        }

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertIn("dimension_transition_base_to_sft_rl", names)
        repair = next(stage for stage in pipeline["stages"] if stage["name"] == "dimension_transition_base_to_sft_rl")
        self.assertEqual(repair["type"], "dimension_transition")
        self.assertEqual(repair["args"]["baseline"], "data/processed/base/bsc_eval.jsonl")
        self.assertEqual(repair["args"]["candidate"], "data/processed/sft_rl/bsc_eval.jsonl")
        self.assertEqual(repair["args"]["output_dir"], "outputs/dimension_transition/base_to_sft_rl")
        self.assertLess(names.index("sft_rl_bsc"), names.index("dimension_transition_base_to_sft_rl"))
        self.assertLess(names.index("dimension_transition_base_to_sft_rl"), names.index("summarize"))

        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertEqual(
            export["args"]["transition_summary_json"],
            ["outputs/dimension_transition/base_to_sft_rl/transition_summary.json"],
        )

        manifest = build_manifest(config)
        self.assertIn(
            "outputs/dimension_transition/base_to_sft_rl/transition_summary.json",
            manifest["required_files"],
        )
        self.assertIn(
            "outputs/dimension_transition/base_to_sft_rl/transition_by_category.csv",
            manifest["required_files"],
        )

    def test_build_pipeline_can_add_semantic_space_stage(self) -> None:
        config = sample_config()
        config["methods"].append(
            {
                "name": "sft_rl",
                "model_filter": "sft_rl",
                "bsc_rubrics": "sft_rl_bsc.jsonl",
                "downstream_rubrics": "sft_rl_down.jsonl",
                "skip_downstream": True,
            }
        )
        config["common"]["semantic_space"] = {
            "enabled": True,
            "methods": ["base", "sft_rl"],
            "output_dir": "outputs/semantic_space",
            "max_points": 2000,
        }

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertIn("semantic_space", names)
        semantic = next(stage for stage in pipeline["stages"] if stage["name"] == "semantic_space")
        self.assertEqual(semantic["type"], "semantic_space")
        self.assertEqual(
            semantic["args"]["input"],
            ["base=data/processed/base/bsc_eval.jsonl", "sft_rl=data/processed/sft_rl/bsc_eval.jsonl"],
        )
        self.assertEqual(
            semantic["args"]["join_report"],
            ["base=outputs/base/bsc_join_report.json", "sft_rl=outputs/sft_rl/bsc_join_report.json"],
        )
        self.assertEqual(semantic["args"]["output_dir"], "outputs/semantic_space")
        self.assertEqual(semantic["args"]["projection"], "pca")
        self.assertEqual(semantic["args"]["gold_cluster_tau"], 0.75)
        self.assertEqual(semantic["args"]["max_points"], 2000)
        self.assertLess(names.index("sft_rl_bsc"), names.index("semantic_space"))
        self.assertLess(names.index("semantic_space"), names.index("summarize"))

        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")
        self.assertEqual(export["args"]["semantic_space_dir"], "outputs/semantic_space")

        manifest = build_manifest(config)
        self.assertIn("outputs/semantic_space/semantic_space.svg", manifest["required_files"])
        self.assertIn("outputs/semantic_space/semantic_space.pdf", manifest["required_files"])
        self.assertIn("outputs/semantic_space/semantic_space_summary.json", manifest["required_files"])

    def test_method_can_override_domain_inputs(self) -> None:
        config = sample_config()
        config["methods"][0].update(
            {
                "bsc_gold": "data/processed/healthbench_proxy_gold.jsonl",
                "bsc_data_source": "healthbench",
                "downstream_preferences": "data/processed/healthbench_pref.jsonl",
                "downstream_data_source": "healthbench",
            }
        )
        pipeline = build_pipeline(config)

        prepare_bsc = next(stage for stage in pipeline["stages"] if stage["name"] == "base_prepare_bsc")
        prepare_downstream = next(stage for stage in pipeline["stages"] if stage["name"] == "base_prepare_downstream")
        self.assertEqual(prepare_bsc["args"]["gold"], "data/processed/healthbench_proxy_gold.jsonl")
        self.assertEqual(prepare_bsc["args"]["data_source"], "healthbench")
        self.assertEqual(prepare_downstream["args"]["preferences"], "data/processed/healthbench_pref.jsonl")
        self.assertEqual(prepare_downstream["args"]["data_source"], "healthbench")

    def test_common_multicandidate_downstream_uses_dedicated_stages(self) -> None:
        config = sample_config()
        config["common"]["downstream_type"] = "multicandidate"
        pipeline = build_pipeline(config)

        prepare = next(stage for stage in pipeline["stages"] if stage["name"] == "base_prepare_downstream")
        evaluate = next(stage for stage in pipeline["stages"] if stage["name"] == "base_downstream")
        self.assertEqual(prepare["type"], "prepare_multicandidate")
        self.assertEqual(prepare["args"]["benchmark"], "pref.jsonl")
        self.assertNotIn("preferences", prepare["args"])
        self.assertEqual(evaluate["type"], "multicandidate_downstream")

    def test_api_downstream_inserts_budget_gate_before_evaluation(self) -> None:
        config = sample_config()
        config["common"]["downstream_scorer"] = "api"
        config["common"]["judge_provider"] = "configs/judge.jsonl"
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]

        budget = next(stage for stage in pipeline["stages"] if stage["name"] == "base_downstream_api_budget")
        evaluate = next(stage for stage in pipeline["stages"] if stage["name"] == "base_downstream")
        self.assertLess(names.index("base_prepare_downstream"), names.index("base_downstream_api_budget"))
        self.assertLess(names.index("base_downstream_api_budget"), names.index("base_downstream"))
        self.assertEqual(budget["type"], "api_budget")
        self.assertEqual(budget["args"]["unit_field"], "rubrics")
        self.assertEqual(budget["args"]["calls_per_record_per_provider"], 2)
        self.assertEqual(evaluate["args"]["require_budget_report"], "outputs/base/downstream_api_budget/budget.json")

        manifest = build_manifest(config)
        self.assertIn("outputs/base/downstream_api_budget/budget.json", manifest["required_files"])
        self.assertIn("outputs/base/downstream_api_budget/budget.md", manifest["required_files"])

    def test_multicandidate_api_downstream_budget_multiplies_candidates(self) -> None:
        config = sample_config()
        config["common"]["downstream_type"] = "multicandidate"
        config["common"]["downstream_scorer"] = "api"
        config["common"]["judge_provider"] = "configs/judge.jsonl"
        pipeline = build_pipeline(config)

        budget = next(stage for stage in pipeline["stages"] if stage["name"] == "base_downstream_api_budget")
        self.assertEqual(budget["args"]["unit_field"], "rubrics")
        self.assertEqual(budget["args"]["unit_multiplier_field"], "candidates")
        self.assertEqual(budget["args"]["calls_per_record_per_provider"], 1)

    def test_api_downstream_rejects_budget_contract_overrides_that_do_not_match_evaluator(self) -> None:
        config = sample_config()
        config["common"]["downstream_scorer"] = "api"
        config["common"]["judge_provider"] = "configs/judge.jsonl"
        config["common"]["judge_api_budget"] = {"calls_per_record_per_provider": 1}

        with self.assertRaises(ValueError):
            build_pipeline(config)

        config = sample_config()
        config["common"]["downstream_scorer"] = "api"
        config["common"]["judge_provider"] = "configs/judge.jsonl"
        config["common"]["judge_api_budget"] = {"unit_multiplier_field": "candidates"}

        with self.assertRaises(ValueError):
            build_pipeline(config)

    def test_multicandidate_api_downstream_rejects_budget_contract_overrides_that_do_not_match_evaluator(self) -> None:
        config = sample_config()
        config["common"]["downstream_type"] = "multicandidate"
        config["common"]["downstream_scorer"] = "api"
        config["common"]["judge_provider"] = "configs/judge.jsonl"
        config["common"]["judge_api_budget"] = {"unit_multiplier_field": "answers"}

        with self.assertRaises(ValueError):
            build_pipeline(config)

        config = sample_config()
        config["common"]["downstream_type"] = "multicandidate"
        config["common"]["downstream_scorer"] = "api"
        config["common"]["judge_provider"] = "configs/judge.jsonl"
        config["common"]["judge_api_budget"] = {"calls_per_record_per_provider": 2}

        with self.assertRaises(ValueError):
            build_pipeline(config)

    def test_common_skip_downstream_builds_bsc_only_matrix(self) -> None:
        config = sample_config()
        config["common"]["skip_downstream"] = True
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertIn("base_prepare_bsc", names)
        self.assertNotIn("base_prepare_downstream", names)
        summarize = next(stage for stage in pipeline["stages"] if stage["name"] == "summarize")
        self.assertIn("bsc", summarize["args"])
        self.assertNotIn("downstream", summarize["args"])

        manifest = build_manifest(config)
        required = manifest["required_files"]
        self.assertIn("outputs/base/bsc/summary.json", required)
        self.assertNotIn("outputs/base/downstream/summary.json", required)
        self.assertEqual([item["name"] for item in manifest["summaries"]], ["base_bsc"])

    def test_build_pipeline_can_add_bootstrap_ci_for_main_table(self) -> None:
        config = sample_config()
        config["common"]["bootstrap_ci"] = {
            "enabled": True,
            "bsc_metrics": ["coverage", "blind"],
            "downstream_metrics": ["correct"],
            "n_boot": 50,
            "seed": 7,
            "confidence": 0.9,
        }
        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertLess(names.index("base_bsc"), names.index("base_bsc_bootstrap_ci"))
        self.assertLess(names.index("base_downstream"), names.index("base_downstream_bootstrap_ci"))
        self.assertLess(names.index("base_bsc_bootstrap_ci"), names.index("summarize"))

        bsc_ci = next(stage for stage in pipeline["stages"] if stage["name"] == "base_bsc_bootstrap_ci")
        self.assertEqual(bsc_ci["args"]["input"], "outputs/base/bsc/per_item.csv")
        self.assertEqual(bsc_ci["args"]["metric"], ["coverage", "blind"])
        self.assertEqual(bsc_ci["args"]["n_boot"], 50)
        self.assertEqual(bsc_ci["args"]["seed"], 7)
        self.assertEqual(bsc_ci["args"]["confidence"], 0.9)

        summarize = next(stage for stage in pipeline["stages"] if stage["name"] == "summarize")
        self.assertEqual(summarize["args"]["bsc_ci"], ["base=outputs/base/bsc_ci/bootstrap_ci.json"])
        self.assertEqual(
            summarize["args"]["downstream_ci"],
            ["base=outputs/base/downstream_ci/bootstrap_ci.json"],
        )

    def test_build_manifest_contains_bootstrap_ci_outputs_when_enabled(self) -> None:
        config = sample_config()
        config["common"]["bootstrap_ci"] = {"enabled": True}
        manifest = build_manifest(config)
        required = manifest["required_files"]
        self.assertIn("outputs/base/bsc_ci/bootstrap_ci.json", required)
        self.assertIn("outputs/base/bsc_ci/bootstrap_ci.csv", required)
        self.assertIn("outputs/base/bsc_ci/bootstrap_ci.md", required)
        self.assertIn("outputs/base/downstream_ci/bootstrap_ci.json", required)
        self.assertIn("outputs/base/downstream_ci/bootstrap_ci.csv", required)
        self.assertIn("outputs/base/downstream_ci/bootstrap_ci.md", required)

    def test_build_pipeline_can_add_selected_bsc_threshold_sweeps(self) -> None:
        config = sample_config()
        config["common"]["bsc_sweep"] = {
            "enabled": True,
            "methods": ["base"],
            "coverage_tau": [0.7, 0.8],
            "redundancy_tau": [0.85],
        }
        config["methods"].append(
            {
                "name": "sft_rl",
                "model_filter": "sft_rl",
                "bsc_rubrics": "rl_bsc.jsonl",
                "downstream_rubrics": "rl_down.jsonl",
            }
        )

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("base_bsc_sweep", names)
        self.assertNotIn("sft_rl_bsc_sweep", names)
        self.assertLess(names.index("base_bsc"), names.index("base_bsc_sweep"))
        self.assertLess(names.index("base_bsc_sweep"), names.index("base_prepare_downstream"))

        sweep = next(stage for stage in pipeline["stages"] if stage["name"] == "base_bsc_sweep")
        self.assertEqual(sweep["args"]["input"], "data/processed/base/bsc_eval.jsonl")
        self.assertEqual(sweep["args"]["coverage_tau"], [0.7, 0.8])
        self.assertEqual(sweep["args"]["redundancy_tau"], [0.85])

    def test_build_pipeline_can_add_selected_bsc_human_audit_packs(self) -> None:
        config = sample_config()
        config["common"]["bsc_human_audit_pack"] = {
            "enabled": True,
            "methods": ["base"],
            "matched": 3,
            "unmatched": 4,
            "seed": 17,
        }
        config["methods"].append(
            {
                "name": "sft_rl",
                "model_filter": "sft_rl",
                "bsc_rubrics": "rl_bsc.jsonl",
                "downstream_rubrics": "rl_down.jsonl",
            }
        )

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("base_bsc_human_audit_pack", names)
        self.assertNotIn("sft_rl_bsc_human_audit_pack", names)
        self.assertLess(names.index("base_bsc"), names.index("base_bsc_human_audit_pack"))
        self.assertLess(names.index("base_bsc_human_audit_pack"), names.index("base_prepare_downstream"))

        stage = next(stage for stage in pipeline["stages"] if stage["name"] == "base_bsc_human_audit_pack")
        self.assertEqual(stage["type"], "bsc_human_audit_pack")
        self.assertEqual(stage["args"]["input"], "data/processed/base/bsc_eval.jsonl")
        self.assertEqual(stage["args"]["matched"], 3)
        self.assertEqual(stage["args"]["unmatched"], 4)
        self.assertEqual(stage["args"]["seed"], 17)
        self.assertEqual(stage["args"]["output_dir"], "outputs/base/bsc_human_audit_pack")

    def test_trained_method_gate_blocks_sft_methods_before_matrix_stages(self) -> None:
        config = sample_config()
        config["common"]["trained_method_gate"] = {
            "enabled": True,
            "name": "matrix_trained_method_gate",
            "gate_name": "matrix_trained_method_gate",
            "model_filters": ["sft_only", "sft_rl"],
            "served_methods": "outputs/training_commands/training_done.json:served_methods",
            "served_generators": "outputs/training_commands/training_done.json:served_generators",
            "required_json": [
                "outputs/training_commands/training_done.json:sft_checkpoint,rl_checkpoint,served_methods,served_generators,serving.sft_rl,operator,date"
            ],
            "required_json_contains": [
                "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
                "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl"
            ],
            "required_json_equals": [
                "outputs/training_commands/training_done.json:reward_function=src/blindspot_rl/verl_reward.py:compute_score"
            ],
            "required_json_sha256": [
                "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet"
            ],
            "output": "outputs/matrix/trained_method_gate.json",
            "output_md": "outputs/matrix/trained_method_gate.md",
            "strict": True,
        }
        config["methods"].append(
            {
                "name": "sft_rl",
                "model_filter": "sft_rl",
                "bsc_rubrics": "rl_bsc.jsonl",
                "downstream_rubrics": "rl_down.jsonl",
            }
        )

        pipeline = build_pipeline(config)
        names = [stage["name"] for stage in pipeline["stages"]]
        gate = pipeline["stages"][0]

        self.assertEqual(gate["name"], "matrix_trained_method_gate")
        self.assertEqual(gate["type"], "manual_gate")
        self.assertLess(names.index("matrix_trained_method_gate"), names.index("sft_rl_prepare_bsc"))
        self.assertEqual(gate["args"]["output"], "outputs/matrix/trained_method_gate.json")
        self.assertIn(
            "outputs/training_commands/training_done.json:served_methods=sft_rl",
            gate["args"]["required_json_contains"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:served_generators=sft_rl",
            gate["args"]["required_json_contains"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
            gate["args"]["required_json_contains"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl",
            gate["args"]["required_json_contains"],
        )
        self.assertIn("served_methods,served_generators", gate["args"]["required_json"][0])
        self.assertIn("serving.sft_rl", gate["args"]["required_json"][0])
        self.assertIn("operator,date", gate["args"]["required_json"][0])
        self.assertIn(
            "outputs/training_commands/training_done.json:reward_function=src/blindspot_rl/verl_reward.py:compute_score",
            gate["args"]["required_json_equals"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
            gate["args"]["required_json_sha256"],
        )
        self.assertTrue(gate["args"]["strict"])

        manifest = build_manifest(config)
        self.assertIn("outputs/matrix/trained_method_gate.json", manifest["required_files"])
        self.assertIn("outputs/matrix/trained_method_gate.md", manifest["required_files"])

    def test_trained_method_gate_is_skipped_when_no_trained_methods_are_present(self) -> None:
        config = sample_config()
        config["common"]["trained_method_gate"] = {
            "enabled": True,
            "model_filters": ["sft_rl"],
            "output": "outputs/matrix/trained_method_gate.json",
        }

        pipeline = build_pipeline(config)

        self.assertNotEqual(pipeline["stages"][0]["type"], "manual_gate")

    def test_build_manifest_contains_selected_bsc_threshold_sweeps(self) -> None:
        config = sample_config()
        config["common"]["bsc_sweep"] = {"enabled": True, "methods": ["base"]}
        manifest = build_manifest(config)
        required = manifest["required_files"]
        self.assertIn("outputs/base/bsc_sweep/threshold_sweep.csv", required)
        self.assertIn("outputs/base/bsc_sweep/threshold_sweep.json", required)
        self.assertIn("outputs/base/bsc_sweep/threshold_sweep.md", required)

    def test_build_manifest_contains_selected_bsc_human_audit_packs(self) -> None:
        config = sample_config()
        config["common"]["bsc_human_audit_pack"] = {"enabled": True, "methods": ["base"]}
        manifest = build_manifest(config)
        required = manifest["required_files"]

        self.assertIn("outputs/base/bsc_human_audit_pack/summary.json", required)
        self.assertIn("outputs/base/bsc_human_audit_pack/audit_items.csv", required)
        self.assertIn("outputs/base/bsc_human_audit_pack/audit_items.jsonl", required)
        self.assertIn("outputs/base/bsc_human_audit_pack/human_label_summary.json", required)
        self.assertIn("outputs/base/bsc_human_audit_pack/human_label_summary.md", required)

    def test_build_manifest_contains_ablation_outputs_when_enabled(self) -> None:
        config = sample_config()
        config["common"]["reward_ablation"] = {"enabled": True, "output_dir": "outputs/abl"}
        config["common"]["teacher_union_ablation"] = {"enabled": True, "output_dir": "outputs/teacher_union"}
        config["common"]["verifier_filter_ablation"] = {"enabled": True, "output_dir": "outputs/verifier_filter"}
        config["common"]["transition_summary_json"] = "outputs/repair/transition_summary.json"
        manifest = build_manifest(config)
        required = manifest["required_files"]
        self.assertIn("outputs/abl/ablation_summary.csv", required)
        self.assertIn("outputs/abl/variants/no_red_summary.json", required)
        self.assertIn("outputs/abl/variants/no_valid_summary.json", required)
        self.assertIn("outputs/teacher_union/teacher_union_ablation.csv", required)
        self.assertIn("outputs/teacher_union/teacher_union_ablation.json", required)
        self.assertIn("outputs/verifier_filter/verifier_filter_ablation.csv", required)
        self.assertIn("outputs/verifier_filter/verifier_filter_ablation.json", required)
        self.assertIn("outputs/repair/transition_summary.json", required)


def sample_config():
    return {
        "common": {
            "bsc_gold": "gold.jsonl",
            "downstream_preferences": "pref.jsonl",
            "processed_root": "data/processed",
            "output_root": "outputs",
        },
        "methods": [
            {
                "name": "base",
                "model_filter": None,
                "bsc_rubrics": "base_bsc.jsonl",
                "downstream_rubrics": "base_down.jsonl",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
