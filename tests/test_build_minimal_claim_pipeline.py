from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_minimal_claim_pipeline import (
    build_evidence_config,
    build_manifest,
    build_pipeline,
    build_result_card_config,
    load_config,
    resolve_paths,
)


class BuildMinimalClaimPipelineTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_minimal_claim_config.json"))

        self.assertIn("Minimal claim config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Minimal claim config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Minimal claim config must be a JSON object", str(context.exception))

    def test_build_pipeline_includes_generation_and_bsc_gate(self) -> None:
        config = sample_config()
        pipeline = build_pipeline(
            config,
            evidence_config_path="configs/minimal_evidence.json",
            manifest_path="configs/minimal_manifest.json",
        )
        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertEqual(names[:3], ["generate_model_rubrics", "prepare_bsc", "bsc"])
        self.assertIn("bsc_sweep", names)
        self.assertLess(names.index("bsc"), names.index("blindspot_map"))
        self.assertLess(names.index("bsc"), names.index("budget_curve"))
        self.assertEqual(pipeline["stages"][0]["args"]["resume"], True)
        self.assertEqual(pipeline["stages"][1]["args"]["model"], "base")
        self.assertEqual(pipeline["stages"][1]["args"]["min_joined"], 10)

    def test_build_pipeline_adds_section_2_diagnostic_stages(self) -> None:
        pipeline = build_pipeline(
            sample_config(),
            evidence_config_path="configs/minimal_evidence.json",
            manifest_path="configs/minimal_manifest.json",
        )
        stages = {stage["name"]: stage for stage in pipeline["stages"]}

        blindspot_map = stages["blindspot_map"]
        self.assertEqual(blindspot_map["type"], "blindspot_map")
        self.assertEqual(blindspot_map["args"]["input"], "data/processed/minimal/base/bsc_eval.jsonl")
        self.assertEqual(blindspot_map["args"]["embedding_model"], "token-overlap")
        self.assertEqual(blindspot_map["args"]["coverage_tau"], 0.5)
        self.assertEqual(blindspot_map["args"]["model"], "base")
        self.assertEqual(blindspot_map["args"]["output_dir"], "outputs/minimal/base/blindspot_map")

        budget_curve = stages["budget_curve"]
        self.assertEqual(budget_curve["type"], "budget_curve")
        self.assertEqual(budget_curve["args"]["input"], "data/processed/minimal/base/bsc_eval.jsonl")
        self.assertEqual(budget_curve["args"]["embedding_model"], "token-overlap")
        self.assertEqual(budget_curve["args"]["coverage_tau"], 0.5)
        self.assertEqual(budget_curve["args"]["redundancy_tau"], 0.9)
        self.assertEqual(budget_curve["args"]["k"], [3, 5, 8, 10, 15])
        self.assertEqual(budget_curve["args"]["output_dir"], "outputs/minimal/base/budget_curve")

    def test_build_pipeline_can_add_bsc_human_audit_pack(self) -> None:
        config = sample_config()
        config["human_audit_pack"] = {"enabled": True, "matched": 3, "unmatched": 4, "seed": 17}
        pipeline = build_pipeline(
            config,
            evidence_config_path="configs/minimal_evidence.json",
            manifest_path="configs/minimal_manifest.json",
        )

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertLess(names.index("bsc"), names.index("bsc_human_audit_pack"))
        stage = next(stage for stage in pipeline["stages"] if stage["name"] == "bsc_human_audit_pack")
        self.assertEqual(stage["type"], "bsc_human_audit_pack")
        self.assertEqual(stage["args"]["input"], "data/processed/minimal/base/bsc_eval.jsonl")
        self.assertEqual(stage["args"]["embedding_model"], "token-overlap")
        self.assertEqual(stage["args"]["coverage_tau"], 0.5)
        self.assertEqual(stage["args"]["matched"], 3)
        self.assertEqual(stage["args"]["unmatched"], 4)
        self.assertEqual(stage["args"]["seed"], 17)
        self.assertEqual(stage["args"]["output_dir"], "outputs/minimal/base/bsc_human_audit_pack")

    def test_build_pipeline_can_add_real_run_startup_stages(self) -> None:
        config = sample_config()
        config["pilot_sample"] = {"enabled": True, "n": 5, "seed": 7}
        config["preflight"] = {"enabled": True, "required_env": ["LOCAL_OPENAI_API_KEY"]}
        config["api_budget"] = {"enabled": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertEqual(names[:4], ["sample_queries", "preflight", "api_budget", "generate_model_rubrics"])
        self.assertEqual(
            pipeline["stages"][0]["args"]["output"],
            "data/processed/minimal/base/queries_pilot.jsonl",
        )
        self.assertEqual(
            pipeline["stages"][2]["args"]["input"],
            "data/processed/minimal/base/queries_pilot.jsonl",
        )
        self.assertEqual(
            pipeline["stages"][3]["args"]["input"],
            "data/processed/minimal/base/queries_pilot.jsonl",
        )
        self.assertEqual(
            pipeline["stages"][3]["args"]["require_budget_report"],
            pipeline["stages"][2]["args"]["output"],
        )
        self.assertEqual(
            pipeline["stages"][3]["args"]["require_preflight_report"],
            "outputs/minimal/base/preflight/preflight.json",
        )
        prepare = next(stage for stage in pipeline["stages"] if stage["name"] == "prepare_bsc")
        self.assertEqual(prepare["args"]["min_joined"], 5)
        self.assertEqual(pipeline["stages"][2]["args"]["providers"], pipeline["stages"][3]["args"]["providers"])
        self.assertEqual(pipeline["stages"][2]["args"]["calls_per_record_per_provider"], 1)
        self.assertNotIn("unit_field", pipeline["stages"][2]["args"])
        self.assertNotIn("unit_multiplier_field", pipeline["stages"][2]["args"])

    def test_build_pipeline_can_add_bsc_gold_sanity_before_api_boundary(self) -> None:
        config = sample_config()
        config["bsc_gold_sanity"] = {"enabled": True, "min_joined": 10}
        config["preflight"] = {"enabled": True}
        config["api_budget"] = {"enabled": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
        )
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertLess(names.index("bsc_gold_sanity"), names.index("preflight"))
        sanity = pipeline["stages"][names.index("bsc_gold_sanity")]
        self.assertEqual(sanity["type"], "bsc_gold_sanity")
        self.assertEqual(sanity["args"]["gold"], "gold.jsonl")
        self.assertEqual(sanity["args"]["output_dir"], "outputs/minimal/base/bsc_gold_sanity")
        self.assertEqual(sanity["args"]["min_joined"], 10)

    def test_build_pipeline_can_add_minimal_api_handoff_before_paid_generation(self) -> None:
        config = sample_config()
        config["bsc_gold_sanity"] = {"enabled": True, "min_joined": 10}
        config["preflight"] = {"enabled": True}
        config["api_budget"] = {"enabled": True}
        config["minimal_api_handoff"] = {"enabled": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            pipeline_config_path="pipeline.json",
        )
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertLess(names.index("api_budget"), names.index("minimal_api_handoff"))
        self.assertLess(names.index("minimal_api_handoff"), names.index("generate_model_rubrics"))
        handoff = pipeline["stages"][names.index("minimal_api_handoff")]
        self.assertEqual(handoff["type"], "minimal_api_handoff")
        self.assertEqual(handoff["args"]["pipeline"], "pipeline.json")
        self.assertEqual(handoff["args"]["preflight"], "outputs/minimal/base/preflight/preflight.json")
        self.assertEqual(handoff["args"]["api_budget"], "outputs/minimal/base/api_budget/model_rubrics_budget.json")
        self.assertEqual(handoff["args"]["bsc_gold_sanity"], "outputs/minimal/base/bsc_gold_sanity/summary.json")
        self.assertEqual(handoff["args"]["output_json"], "outputs/minimal/base/handoff/api_handoff.json")
        self.assertEqual(handoff["args"]["output_md"], "outputs/minimal/base/handoff/api_handoff.md")

    def test_build_pipeline_can_add_scoped_source_report_stage(self) -> None:
        config = sample_config()
        config["local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "output": "configs/data_sources_real.local.json",
            "report_json": "outputs/data_sources/local_config_init_rubricbench.json",
            "report_md": "outputs/data_sources/local_config_init_rubricbench.md",
            "required_datasets": ["rubricbench"],
        }
        config["result_card"] = {
            "enabled": True,
            "data_source_reports": [
                {
                    "name": "Data Source Report",
                    "config": "configs/data_sources_real.local.json",
                    "path": "outputs/data_sources/source_report_rubricbench.json",
                    "output_md": "outputs/data_sources/source_report_rubricbench.md",
                    "required_datasets": ["rubricbench"],
                    "strict": True,
                }
            ],
        }
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        init_stage = pipeline["stages"][0]
        source_stage = pipeline["stages"][1]
        self.assertEqual(init_stage["type"], "init_data_source_config")
        self.assertEqual(init_stage["args"]["report_json"], "outputs/data_sources/local_config_init_rubricbench.json")
        self.assertEqual(init_stage["args"]["required_dataset"], ["rubricbench"])
        self.assertFalse(init_stage["args"]["strict"])
        self.assertEqual(source_stage["type"], "data_source_report")
        self.assertEqual(source_stage["args"]["config"], "configs/data_sources_real.local.json")
        self.assertEqual(source_stage["args"]["output_json"], "outputs/data_sources/source_report_rubricbench.json")
        self.assertEqual(source_stage["args"]["required_dataset"], ["rubricbench"])
        self.assertTrue(source_stage["args"]["strict"])

    def test_build_pipeline_can_add_gold_validation_before_sampling(self) -> None:
        config = sample_config()
        config["pilot_sample"] = {"enabled": True}
        config["gold_validation"] = {
            "enabled": True,
            "input": "data/processed/rubricbench_gold.jsonl",
            "output_json": "outputs/data_validation/rubricbench_gold.json",
            "output_md": "outputs/data_validation/rubricbench_gold.md",
            "min_records": 100,
            "require_provenance": True,
            "required_provenance_values": {"paper_url": "https://arxiv.org/abs/2603.01562"},
            "required_data_source": ["rubricbench"],
            "forbidden_data_source": ["toy", "proxy"],
            "strict": True,
        }
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertLess(names.index("validate_rubricbench_gold"), names.index("sample_queries"))
        stage = next(stage for stage in pipeline["stages"] if stage["name"] == "validate_rubricbench_gold")
        self.assertEqual(stage["type"], "validate_gold")
        self.assertEqual(stage["args"]["input"], "data/processed/rubricbench_gold.jsonl")
        self.assertEqual(stage["args"]["required_data_source"], ["rubricbench"])
        self.assertEqual(stage["args"]["forbidden_data_source"], ["toy", "proxy"])
        self.assertEqual(stage["args"]["required_provenance"], ["paper_url=https://arxiv.org/abs/2603.01562"])
        self.assertTrue(stage["args"]["require_provenance"])
        self.assertTrue(stage["args"]["strict"])

    def test_build_pipeline_can_add_query_validation_before_sampling(self) -> None:
        config = sample_config()
        config["pilot_sample"] = {"enabled": True}
        config["query_validation"] = {
            "enabled": True,
            "input": "data/processed/rubricbench_queries.jsonl",
            "output_json": "outputs/data_validation/rubricbench_queries.json",
            "output_md": "outputs/data_validation/rubricbench_queries.md",
            "min_records": 100,
            "require_provenance": True,
            "required_provenance_values": {"paper_url": "https://arxiv.org/abs/2603.01562"},
            "required_data_source": ["rubricbench"],
            "forbidden_data_source": ["toy", "proxy"],
            "strict": True,
        }
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertLess(names.index("validate_rubricbench_queries"), names.index("sample_queries"))
        stage = next(stage for stage in pipeline["stages"] if stage["name"] == "validate_rubricbench_queries")
        self.assertEqual(stage["type"], "validate_gold")
        self.assertEqual(stage["args"]["target"], "query_pool")
        self.assertEqual(stage["args"]["input"], "data/processed/rubricbench_queries.jsonl")
        self.assertEqual(stage["args"]["required_data_source"], ["rubricbench"])
        self.assertEqual(stage["args"]["forbidden_data_source"], ["toy", "proxy"])
        self.assertEqual(stage["args"]["required_provenance"], ["paper_url=https://arxiv.org/abs/2603.01562"])
        self.assertTrue(stage["args"]["require_provenance"])
        self.assertTrue(stage["args"]["strict"])

    def test_build_pipeline_can_add_verifier_annotation_before_bsc(self) -> None:
        config = sample_config()
        config["model_output_validation"] = {"enabled": True, "strict": True}
        config["model_verification"] = {
            "enabled": True,
            "provider": "verifier.jsonl",
            "mode": "api",
            "annotate_only": True,
            "strict": True,
        }
        config["verified_output_validation"] = {"enabled": True, "require_valid_flags": True, "strict": True}
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertLess(names.index("generate_model_rubrics"), names.index("verifier_api_budget"))
        self.assertLess(names.index("generate_model_rubrics"), names.index("validate_model_rubrics"))
        self.assertLess(names.index("validate_model_rubrics"), names.index("verifier_api_budget"))
        self.assertLess(names.index("verifier_api_budget"), names.index("verify_model_rubrics"))
        self.assertLess(names.index("verify_model_rubrics"), names.index("validate_verified_model_rubrics"))
        self.assertLess(names.index("validate_verified_model_rubrics"), names.index("prepare_bsc"))
        budget = next(stage for stage in pipeline["stages"] if stage["name"] == "verifier_api_budget")
        verifier = next(stage for stage in pipeline["stages"] if stage["name"] == "verify_model_rubrics")
        raw_validation = next(stage for stage in pipeline["stages"] if stage["name"] == "validate_model_rubrics")
        verified_validation = next(
            stage for stage in pipeline["stages"] if stage["name"] == "validate_verified_model_rubrics"
        )
        self.assertEqual(budget["args"]["input"], "data/processed/minimal/base/model_rubrics.jsonl")
        self.assertEqual(budget["args"]["providers"], "verifier.jsonl")
        self.assertEqual(budget["args"]["unit_field"], "rubrics")
        self.assertEqual(budget["args"]["calls_per_record_per_provider"], 1)
        self.assertTrue(budget["args"]["strict"])
        self.assertEqual(raw_validation["type"], "validate_rubrics")
        self.assertEqual(raw_validation["args"]["input"], "data/processed/minimal/base/model_rubrics.jsonl")
        self.assertTrue(raw_validation["args"]["strict"])
        self.assertEqual(verifier["type"], "filter_verifier")
        self.assertTrue(verifier["args"]["annotate_only"])
        self.assertEqual(verifier["args"]["input"], "data/processed/minimal/base/model_rubrics.jsonl")
        self.assertEqual(verifier["args"]["output"], "data/processed/minimal/base/model_rubrics_verified.jsonl")
        self.assertEqual(verifier["args"]["require_budget_report"], budget["args"]["output"])
        self.assertNotIn("require_preflight_report", verifier["args"])
        self.assertEqual(verified_validation["args"]["input"], "data/processed/minimal/base/model_rubrics_verified.jsonl")
        self.assertTrue(verified_validation["args"]["require_valid_flags"])
        prepare = next(stage for stage in pipeline["stages"] if stage["name"] == "prepare_bsc")
        self.assertEqual(prepare["args"]["predictions"], "data/processed/minimal/base/model_rubrics_verified.jsonl")

    def test_verifier_paid_stage_requires_preflight_when_enabled(self) -> None:
        config = sample_config()
        config["preflight"] = {"enabled": True}
        config["model_verification"] = {"enabled": True, "provider": "verifier.jsonl"}
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

        verifier = next(stage for stage in pipeline["stages"] if stage["name"] == "verify_model_rubrics")

        self.assertEqual(verifier["args"]["require_preflight_report"], "outputs/minimal/base/preflight/preflight.json")

    def test_preflight_can_enable_local_provider_health_check(self) -> None:
        config = sample_config()
        config["preflight"] = {"enabled": True, "check_local_provider_health": True}
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

        preflight = next(stage for stage in pipeline["stages"] if stage["name"] == "preflight")

        self.assertTrue(preflight["args"]["check_local_provider_health"])

    def test_api_budget_contract_must_match_generation(self) -> None:
        cases = [
            ("input", "other.jsonl"),
            ("providers", "other_providers.jsonl"),
            ("unit_field", "rubrics"),
            ("unit_multiplier_field", "candidates"),
            ("calls_per_record_per_provider", 2),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                config = sample_config()
                config["api_budget"] = {"enabled": True, key: value}
                with self.assertRaises(ValueError):
                    build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

    def test_build_pipeline_can_add_paper_sync_and_readiness_after_export(self) -> None:
        config = sample_config()
        config["paper_sync"] = {"enabled": True, "paper_dir": "paper"}
        config["readiness"] = {"enabled": True, "paper_dir": "paper", "strict": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertEqual(names[-4:], ["export", "sync_paper", "paper_asset_index_check", "submission_readiness"])
        self.assertEqual(pipeline["stages"][-3]["args"]["artifacts_dir"], "outputs/minimal/base/paper_artifacts")
        self.assertEqual(pipeline["stages"][-3]["args"]["required_file"], ["main_table.tex", "experiment_summary.md"])
        self.assertEqual(pipeline["stages"][-2]["args"]["asset_index"], "paper/asset_index.md")
        self.assertEqual(pipeline["stages"][-2]["args"]["output"], "outputs/minimal/base/paper_artifacts/paper_asset_index_check.json")
        self.assertNotIn("strict", pipeline["stages"][-2]["args"])
        self.assertEqual(pipeline["stages"][-1]["args"]["audit_report"], "outputs/minimal/base/audit_report.json")
        self.assertEqual(pipeline["stages"][-1]["args"]["evidence_matrix"], "outputs/minimal/base/evidence/evidence_matrix.json")
        self.assertTrue(pipeline["stages"][-1]["args"]["strict"])

    def test_build_pipeline_adds_minimal_readiness_raw_gates(self) -> None:
        config = sample_config()
        config["local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "output": "configs/data_sources_real.local.json",
            "report_json": "outputs/data_sources/local_config_init_rubricbench.json",
        }
        config["preflight"] = {"enabled": True}
        config["api_budget"] = {"enabled": True}
        config["readiness"] = {
            "enabled": True,
            "raw_gate": ["Custom Gate|generic|outputs/custom_gate.json"],
        }
        config["result_card"] = {
            "enabled": True,
            "data_source_reports": [
                {
                    "name": "Data Source Report",
                    "path": "outputs/data_sources/source_report_rubricbench.json",
                    "required_datasets": ["rubricbench"],
                }
            ],
            "gold_validations": [{"name": "RubricBench Gold Validation", "path": "outputs/data_validation/rubricbench_gold.json"}],
            "query_validations": [
                {"name": "RubricBench Query Pool Validation", "path": "outputs/data_validation/rubricbench_queries.json"}
            ],
        }
        config["query_validation"] = {
            "enabled": True,
            "output_json": "outputs/data_validation/rubricbench_queries.json",
        }
        config["bsc_gold_sanity"] = {"enabled": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        readiness_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "submission_readiness")
        self.assertEqual(
            readiness_stage["args"]["raw_gate"],
            [
                "Data Source Local Config|data_source_local_config|outputs/data_sources/local_config_init_rubricbench.json",
                "Data Source Report|data_source_report[rubricbench]|outputs/data_sources/source_report_rubricbench.json",
                "RubricBench Gold Validation|gold_validation|outputs/data_validation/rubricbench_gold.json",
                "RubricBench Query Pool Validation|query_validation|outputs/data_validation/rubricbench_queries.json",
                "Minimal Claim Preflight|preflight|outputs/minimal/base/preflight/preflight.json",
                "Minimal Claim API Budget|api_budget|outputs/minimal/base/api_budget/model_rubrics_budget.json",
                "BSC Gold-as-Prediction Sanity|bsc_gold_sanity|outputs/minimal/base/bsc_gold_sanity/summary.json",
                "Custom Gate|generic|outputs/custom_gate.json",
            ],
        )

    def test_minimal_readiness_uses_gold_validation_output_path(self) -> None:
        config = sample_config()
        config["readiness"] = {"enabled": True}
        config["gold_validation"] = {
            "enabled": True,
            "output_json": "outputs/custom/rubricbench_gold_validation.json",
        }
        config["result_card"] = {
            "enabled": True,
            "gold_validations": [
                {
                    "name": "Stale Duplicate",
                    "path": "outputs/custom/rubricbench_gold_validation.json",
                }
            ],
        }
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        readiness_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "submission_readiness")
        gold_gates = [gate for gate in readiness_stage["args"]["raw_gate"] if "gold_validation" in gate]
        self.assertEqual(
            gold_gates,
            [
                "RubricBench Gold Validation|gold_validation|outputs/custom/rubricbench_gold_validation.json",
            ],
        )

    def test_minimal_readiness_uses_query_validation_output_path(self) -> None:
        config = sample_config()
        config["readiness"] = {"enabled": True}
        config["query_validation"] = {
            "enabled": True,
            "output_json": "outputs/custom/rubricbench_query_validation.json",
        }
        config["result_card"] = {
            "enabled": True,
            "query_validations": [
                {
                    "name": "Stale Duplicate",
                    "path": "outputs/custom/rubricbench_query_validation.json",
                }
            ],
        }
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        readiness_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "submission_readiness")
        query_gates = [gate for gate in readiness_stage["args"]["raw_gate"] if "query_validation" in gate]
        self.assertEqual(
            query_gates,
            [
                "RubricBench Query Pool Validation|query_validation|outputs/custom/rubricbench_query_validation.json",
            ],
        )

    def test_minimal_readiness_includes_verifier_raw_gates_when_enabled(self) -> None:
        config = sample_config()
        config["readiness"] = {"enabled": True}
        config["model_output_validation"] = {"enabled": True}
        config["model_verification"] = {"enabled": True, "provider": "verifier.jsonl"}
        config["verified_output_validation"] = {"enabled": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        readiness_stage = next(stage for stage in pipeline["stages"] if stage["name"] == "submission_readiness")
        self.assertIn(
            "Minimal Claim Verifier API Budget|api_budget|outputs/minimal/base/api_budget/model_rubrics_verifier_budget.json",
            readiness_stage["args"]["raw_gate"],
        )
        self.assertIn(
            "Minimal Claim Model Evaluation-Criteria Validation|validation|outputs/minimal/base/validation/model_rubrics/validation_report.json",
            readiness_stage["args"]["raw_gate"],
        )
        self.assertIn(
            "Minimal Claim Verified Model Evaluation-Criteria|generic|data/processed/minimal/base/model_rubrics_verified.jsonl",
            readiness_stage["args"]["raw_gate"],
        )
        self.assertIn(
            "Minimal Claim Verifier Stats|generic|outputs/minimal/base/verifier/model_rubrics_stats.jsonl",
            readiness_stage["args"]["raw_gate"],
        )
        self.assertIn(
            "Minimal Claim Verified Evaluation-Criteria Validation|validation|outputs/minimal/base/validation/model_rubrics_verified/validation_report.json",
            readiness_stage["args"]["raw_gate"],
        )

    def test_build_pipeline_can_add_result_card_after_readiness(self) -> None:
        config = sample_config()
        config["local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "report_json": "outputs/data_sources/local_config_init_rubricbench.json",
        }
        config["readiness"] = {"enabled": True}
        config["result_card"] = {"enabled": True, "strict": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertEqual(names[-2:], ["submission_readiness", "result_card"])
        self.assertEqual(pipeline["stages"][-1]["args"]["config"], "result_card_config.json")
        self.assertEqual(pipeline["stages"][-1]["args"]["output_json"], "outputs/minimal/base/paper_artifacts/result_card.json")
        self.assertTrue(pipeline["stages"][-1]["args"]["strict"])

        card = build_result_card_config(config, manifest_path="manifest.json")
        self.assertEqual(card["raw_audit_gates"][0]["type"], "data_source_local_config")
        self.assertEqual(card["raw_audit_gates"][0]["path"], "outputs/data_sources/local_config_init_rubricbench.json")

    def test_build_pipeline_can_add_bootstrap_ci_after_bsc(self) -> None:
        config = sample_config()
        config["bootstrap_ci"] = {"enabled": True, "metrics": ["coverage", "blind"], "n_boot": 50}
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertIn("bsc_bootstrap_ci", names)
        ci_stage = pipeline["stages"][names.index("bsc_bootstrap_ci")]
        self.assertLess(names.index("bsc"), names.index("bsc_bootstrap_ci"))
        self.assertLess(names.index("bsc_bootstrap_ci"), names.index("summarize"))
        self.assertEqual(ci_stage["args"]["input"], "outputs/minimal/base/bsc/per_item.csv")
        self.assertEqual(ci_stage["args"]["metric"], ["coverage", "blind"])
        self.assertEqual(ci_stage["args"]["n_boot"], 50)

    def test_build_pipeline_syncs_result_card_when_paper_sync_enabled(self) -> None:
        config = sample_config()
        config["paper_sync"] = {"enabled": True, "paper_dir": "paper"}
        config["readiness"] = {"enabled": True}
        config["result_card"] = {"enabled": True}
        pipeline = build_pipeline(
            config,
            evidence_config_path="evidence.json",
            manifest_path="manifest.json",
            result_card_config_path="result_card_config.json",
        )

        names = [stage["name"] for stage in pipeline["stages"]]
        self.assertEqual(
            names[-4:],
            ["result_card", "sync_result_card", "paper_asset_index_check_post_sync", "paper_asset_index_check_final"],
        )
        self.assertEqual(pipeline["stages"][-3]["args"]["artifacts_dir"], "outputs/minimal/base/paper_artifacts")
        self.assertEqual(pipeline["stages"][-3]["args"]["required_file"], ["main_table.tex", "experiment_summary.md"])
        self.assertEqual(pipeline["stages"][-2]["args"]["asset_index"], "paper/asset_index.md")
        self.assertEqual(pipeline["stages"][-2]["args"]["output_md"], "outputs/minimal/base/paper_artifacts/paper_asset_index_check.md")
        self.assertNotIn("strict", pipeline["stages"][-2]["args"])
        self.assertEqual(pipeline["stages"][-1]["args"]["asset_index"], "paper/asset_index.md")
        self.assertEqual(pipeline["stages"][-1]["args"]["output_md"], "outputs/minimal/base/paper_artifacts/paper_asset_index_check.md")
        self.assertTrue(pipeline["stages"][-1]["args"]["strict"])

    def test_result_card_requires_config_path_when_enabled(self) -> None:
        config = sample_config()
        config["result_card"] = {"enabled": True}
        with self.assertRaises(ValueError):
            build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

    def test_build_pipeline_can_skip_generation_when_predictions_exist(self) -> None:
        config = sample_config()
        del config["providers"]
        pipeline = build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")
        self.assertEqual(pipeline["stages"][0]["name"], "prepare_bsc")

    def test_build_manifest_requires_minimal_outputs(self) -> None:
        manifest = build_manifest(sample_config())
        self.assertIn("outputs/minimal/base/bsc/summary.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/evidence/evidence_matrix.md", manifest["required_files"])
        self.assertEqual(manifest["summaries"][0]["name"], "minimal_bsc")
        self.assertIn("coverage_tau", manifest["summaries"][0]["required_keys"])
        self.assertIn("mean_blind", manifest["summaries"][0]["required_keys"])
        self.assertIn("median_blind", manifest["summaries"][0]["required_keys"])
        self.assertIn("queries_coverage_le_0_5", manifest["summaries"][0]["required_keys"])
        self.assertIn("queries_blind_ge_0_5", manifest["summaries"][0]["required_keys"])
        self.assertIn("queries_zero_coverage", manifest["summaries"][0]["required_keys"])
        self.assertNotIn("embedding_model", manifest["summaries"][0]["required_keys"])
        self.assertNotIn("data_source_counts", manifest["summaries"][0]["required_keys"])

    def test_build_manifest_includes_human_audit_pack_when_enabled(self) -> None:
        config = sample_config()
        config["human_audit_pack"] = {"enabled": True}
        manifest = build_manifest(config)

        self.assertIn("outputs/minimal/base/bsc_human_audit_pack/summary.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_human_audit_pack/audit_items.csv", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_human_audit_pack/audit_items.jsonl", manifest["required_files"])

    def test_export_stage_includes_machine_readable_evidence_artifacts(self) -> None:
        pipeline = build_pipeline(sample_config(), evidence_config_path="evidence.json", manifest_path="manifest.json")
        export = next(stage for stage in pipeline["stages"] if stage["name"] == "export")

        self.assertEqual(export["args"]["handoff_json"], "outputs/minimal/base/handoff/api_handoff.json")
        self.assertEqual(export["args"]["handoff_md"], "outputs/minimal/base/handoff/api_handoff.md")
        self.assertEqual(export["args"]["evidence_json"], "outputs/minimal/base/evidence/evidence_matrix.json")
        self.assertEqual(export["args"]["evidence_csv"], "outputs/minimal/base/evidence/evidence_matrix.csv")
        self.assertEqual(export["args"]["evidence_md"], "outputs/minimal/base/evidence/evidence_matrix.md")

    def test_audit_stage_is_non_strict_so_blocked_result_card_can_be_built(self) -> None:
        pipeline = build_pipeline(sample_config(), evidence_config_path="evidence.json", manifest_path="manifest.json")
        audit = next(stage for stage in pipeline["stages"] if stage["name"] == "audit")

        self.assertTrue(audit["args"]["non_strict"])

    def test_build_manifest_includes_real_run_startup_artifacts_when_enabled(self) -> None:
        config = sample_config()
        config["local_config_init"] = {
            "enabled": True,
            "template": "configs/data_sources_real.template.json",
            "output": "configs/data_sources_real.local.json",
            "report_json": "outputs/data_sources/local_config_init_rubricbench.json",
        }
        config["result_card"] = {
            "enabled": True,
            "data_source_reports": [{"name": "Data Source Report", "path": "outputs/data_sources/source_report_rubricbench.json"}],
        }
        config["gold_validation"] = {
            "enabled": True,
            "output_json": "outputs/data_validation/rubricbench_gold.json",
            "output_md": "outputs/data_validation/rubricbench_gold.md",
        }
        config["query_validation"] = {
            "enabled": True,
            "output_json": "outputs/data_validation/rubricbench_queries.json",
            "output_md": "outputs/data_validation/rubricbench_queries.md",
        }
        config["bsc_gold_sanity"] = {"enabled": True}
        config["minimal_api_handoff"] = {"enabled": True}
        config["pilot_sample"] = {"enabled": True}
        config["preflight"] = {"enabled": True}
        config["api_budget"] = {"enabled": True}
        manifest = build_manifest(config)
        self.assertIn("configs/data_sources_real.local.json", manifest["required_files"])
        self.assertIn("outputs/data_sources/local_config_init_rubricbench.json", manifest["required_files"])
        self.assertIn("outputs/data_sources/source_report_rubricbench.json", manifest["required_files"])
        self.assertIn("outputs/data_validation/rubricbench_gold.json", manifest["required_files"])
        self.assertIn("outputs/data_validation/rubricbench_gold.md", manifest["required_files"])
        self.assertIn("outputs/data_validation/rubricbench_queries.json", manifest["required_files"])
        self.assertIn("outputs/data_validation/rubricbench_queries.md", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_gold_sanity/summary.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_gold_sanity/join_report.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_gold_sanity/bsc_eval.jsonl", manifest["required_files"])
        self.assertIn("data/processed/minimal/base/queries_pilot.jsonl", manifest["required_files"])
        self.assertIn("outputs/minimal/base/preflight/preflight.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/api_budget/model_rubrics_budget.md", manifest["required_files"])
        self.assertIn("outputs/minimal/base/handoff/api_handoff.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/handoff/api_handoff.md", manifest["required_files"])

    def test_build_manifest_includes_verifier_artifacts_when_enabled(self) -> None:
        config = sample_config()
        config["model_output_validation"] = {"enabled": True}
        config["model_verification"] = {"enabled": True, "provider": "verifier.jsonl"}
        config["verified_output_validation"] = {"enabled": True}
        manifest = build_manifest(config)

        self.assertIn("outputs/minimal/base/validation/model_rubrics/validation_report.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/validation/model_rubrics/validation_report.md", manifest["required_files"])
        self.assertIn("outputs/minimal/base/api_budget/model_rubrics_verifier_budget.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/api_budget/model_rubrics_verifier_budget.md", manifest["required_files"])
        self.assertIn("outputs/minimal/base/verifier/model_rubrics_stats.jsonl", manifest["required_files"])
        self.assertIn("data/processed/minimal/base/model_rubrics_verified.jsonl", manifest["required_files"])
        self.assertIn(
            "outputs/minimal/base/validation/model_rubrics_verified/validation_report.json",
            manifest["required_files"],
        )
        self.assertIn(
            "outputs/minimal/base/validation/model_rubrics_verified/validation_report.md",
            manifest["required_files"],
        )
        self.assertNotIn("verifier_source", manifest["summaries"][0]["required_keys"])
        self.assertNotIn("verifier_source_counts", manifest["summaries"][0]["required_keys"])

    def test_build_manifest_includes_result_card_outputs_when_enabled(self) -> None:
        config = sample_config()
        config["result_card"] = {"enabled": True}
        manifest = build_manifest(config)
        self.assertIn("outputs/minimal/base/paper_artifacts/result_card.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/paper_artifacts/result_card.md", manifest["required_files"])

    def test_build_manifest_includes_bootstrap_ci_outputs_when_enabled(self) -> None:
        config = sample_config()
        config["bootstrap_ci"] = {"enabled": True}
        manifest = build_manifest(config)
        self.assertIn("outputs/minimal/base/bsc_ci/bootstrap_ci.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_ci/bootstrap_ci.csv", manifest["required_files"])
        self.assertIn("outputs/minimal/base/bsc_ci/bootstrap_ci.md", manifest["required_files"])

    def test_build_manifest_includes_section_2_diagnostic_outputs_by_default(self) -> None:
        manifest = build_manifest(sample_config())

        self.assertIn("outputs/minimal/base/blindspot_map/summary.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/blindspot_map/category_summary.csv", manifest["required_files"])
        self.assertIn("outputs/minimal/base/blindspot_map/blindspots.jsonl", manifest["required_files"])
        self.assertIn("outputs/minimal/base/budget_curve/coverage_by_k.json", manifest["required_files"])
        self.assertIn("outputs/minimal/base/budget_curve/coverage_by_k.csv", manifest["required_files"])

    def test_api_budget_requires_providers(self) -> None:
        config = sample_config()
        del config["providers"]
        config["api_budget"] = {"enabled": True}
        with self.assertRaises(ValueError):
            build_pipeline(config, evidence_config_path="evidence.json", manifest_path="manifest.json")

    def test_build_evidence_config_uses_claim_gates(self) -> None:
        evidence = build_evidence_config(sample_config())
        self.assertEqual(
            evidence["claims"][0]["claim"],
            "A single-model evaluation-criteria policy leaves measurable blind spots against human-gold evaluation dimensions.",
        )
        metrics = evidence["claims"][0]["metrics"]
        self.assertEqual(metrics[0]["value"], 10)
        self.assertEqual(metrics[1]["value"], 0.1)
        self.assertIn(
            {
                "label": "BSC coverage threshold matches protocol",
                "path": "outputs/minimal/base/bsc/summary.json",
                "metric": "coverage_tau",
                "op": "==",
                "value": 0.5,
            },
            metrics,
        )
        self.assertIn(
            {
                "label": "Hard-gold source has enough evaluated records (rubricbench)",
                "path": "outputs/minimal/base/bsc/summary.json",
                "metric": "data_source_counts.rubricbench",
                "op": ">=",
                "value": 10,
            },
            metrics,
        )
        self.assertIn("median_blind", {item["metric"] for item in metrics})
        self.assertIn("queries_coverage_le_0_5", {item["metric"] for item in metrics})
        self.assertIn("outputs/minimal/base/blindspot_map/summary.json", {item["path"] for item in evidence["claims"][0]["artifacts"]})
        self.assertIn("outputs/minimal/base/budget_curve/coverage_by_k.json", {item["path"] for item in evidence["claims"][0]["artifacts"]})
        self.assertIn("total_gold", {item["metric"] for item in metrics})
        csv_paths = {item["path"] for item in evidence["claims"][0]["csv_checks"]}
        self.assertIn("outputs/minimal/base/blindspot_map/category_summary.csv", csv_paths)
        self.assertIn("outputs/minimal/base/budget_curve/coverage_by_k.csv", csv_paths)
        robustness_metrics = {item["metric"] for item in evidence["claims"][1]["metrics"]}
        self.assertIn("0.mean_blind", robustness_metrics)
        self.assertIn("6.mean_blind", robustness_metrics)
        self.assertIn(
            {
                "label": "BSC embedding model matches protocol",
                "path": "outputs/minimal/base/bsc/summary.json",
                "key": "embedding_model",
                "op": "==",
                "value": "token-overlap",
            },
            evidence["claims"][0]["values"],
        )

    def test_build_evidence_config_can_freeze_diagnostic_snapshot(self) -> None:
        config = sample_config()
        config["bootstrap_ci"] = {"enabled": True}
        config["claim_gates"]["frozen_diagnostic"] = {
            "n": 10,
            "mean_coverage": 0.36919517010450364,
            "mean_blind": 0.6308048298954964,
            "median_blind": 0.6666666567325592,
            "queries_coverage_le_0_5": 7,
            "queries_zero_coverage": 2,
            "mean_redundancy": 0.07675396825396825,
            "mean_hallucination": 0.12273412698412695,
            "blind_ci_lower": 0.5784969914257526,
            "blind_ci_upper": 0.6822689984422177,
        }
        config["claim_gates"]["frozen_blindspot_attribution"] = {
            "total_gold": 569,
            "uncovered_gold": 367,
            "mean_blind_over_gold": 0.6449912126537786,
            "categories": {
                "constraint_following": {
                    "total_gold": 124,
                    "coverage": 0.1774193548387097,
                    "blind_rate": 0.8225806451612904,
                }
            },
        }
        config["claim_gates"]["frozen_budget_curve"] = {
            "rows": [
                {
                    "k": 10,
                    "n": 100,
                    "mean_coverage": 0.36919517010450364,
                    "mean_blind": 0.6308048298954964,
                    "mean_redundancy": 0.07675396825396825,
                    "total_gen": 842,
                }
            ]
        }
        evidence = build_evidence_config(config)
        c1_metrics = {
            item["label"]: item
            for item in evidence["claims"][0]["metrics"]
            if item["label"].startswith("Frozen diagnostic")
        }
        table_values = {(item["row_value"], item["key"]): item for item in evidence["claims"][0]["table_values"]}

        self.assertEqual(c1_metrics["Frozen diagnostic mean blind-spot rate"]["op"], "==")
        self.assertEqual(c1_metrics["Frozen diagnostic mean blind-spot rate"]["value"], 0.6308048298954964)
        self.assertEqual(c1_metrics["Frozen diagnostic mean coverage"]["value"], 0.36919517010450364)
        self.assertEqual(c1_metrics["Frozen diagnostic zero-coverage queries"]["value"], 2)
        self.assertEqual(
            c1_metrics["Frozen diagnostic blind CI lower bound"]["metric"],
            "metrics[metric=blind].ci_lower",
        )
        self.assertEqual(
            c1_metrics["Frozen diagnostic blind CI upper bound"]["metric"],
            "metrics[metric=blind].ci_upper",
        )
        self.assertEqual(table_values[("constraint_following", "blind_rate")]["value"], 0.8225806451612904)
        self.assertEqual(table_values[(10, "mean_coverage")]["value"], 0.36919517010450364)
        self.assertEqual(table_values[(10, "total_gen")]["value"], 842)

    def test_build_evidence_config_gates_human_audit_pack_when_enabled(self) -> None:
        config = sample_config()
        config["human_audit_pack"] = {"enabled": True, "matched": 3, "unmatched": 4}
        evidence = build_evidence_config(config)
        c2 = evidence["claims"][1]
        metrics = {item["metric"]: item for item in c2["metrics"]}
        value_keys = {item["key"] for item in c2["values"]}
        artifact_paths = {item["path"] for item in c2["artifacts"]}

        self.assertIn("outputs/minimal/base/bsc_human_audit_pack/audit_items.csv", artifact_paths)
        self.assertIn("status", value_keys)
        self.assertEqual(metrics["sampled_matched"]["value"], 3)
        self.assertEqual(metrics["sampled_unmatched"]["value"], 4)

    def test_build_evidence_config_can_require_blind_ci_lower_bound(self) -> None:
        config = sample_config()
        config["bootstrap_ci"] = {"enabled": True}
        config["claim_gates"] = {"min_n": 10, "min_blind": 0.1, "min_blind_ci_lower": 0.15}
        evidence = build_evidence_config(config)

        metrics = evidence["claims"][0]["metrics"]
        self.assertEqual(metrics[-1]["metric"], "metrics[metric=blind].ci_lower")
        self.assertEqual(metrics[-1]["value"], 0.15)
        self.assertEqual(evidence["claims"][0]["artifacts"][-1]["label"], "BSC bootstrap CI")

    def test_build_evidence_config_requires_verifier_source_when_enabled(self) -> None:
        config = sample_config()
        config["model_verification"] = {"enabled": True, "provider": "verifier.jsonl"}
        evidence = build_evidence_config(config)

        metrics = {item["metric"] for item in evidence["claims"][0]["metrics"]}
        values = {item["key"] for item in evidence["claims"][0]["values"]}
        self.assertIn("verifier_source_counts.valid_flags", metrics)
        self.assertIn("verifier_source", values)

    def test_resolve_paths_defaults_are_method_scoped(self) -> None:
        paths = resolve_paths({"method": "base"})
        self.assertEqual(paths["bsc_summary"], "outputs/minimal_claim/base/bsc/summary.json")

    def test_build_result_card_config_uses_minimal_outputs(self) -> None:
        config = sample_config()
        config["readiness"] = {"enabled": True}
        config["result_card"] = {
            "enabled": True,
            "gold_validations": [{"name": "Gold", "path": "outputs/gold.json"}],
            "dashboard": "outputs/dashboard.json",
        }
        config["bootstrap_ci"] = {"enabled": True}
        card_config = build_result_card_config(config, manifest_path="manifest.json")

        self.assertEqual(
            card_config["scope"],
            "Minimal motivation experiment: single-model BSC against hard-gold evaluation dimensions.",
        )
        self.assertEqual(
            card_config["safe_claim"],
            "A single-model evaluation-criteria policy leaves measurable blind spots only if Raw Audit Gates and evidence gates pass.",
        )
        self.assertEqual(
            card_config["deferred_claim"],
            "Keep the blind-spot magnitude claim deferred until hard-gold data, criteria elicitation, BSC diagnostics, and evidence gates pass.",
        )
        self.assertEqual(card_config["bsc_summaries"][0]["path"], "outputs/minimal/base/bsc/summary.json")
        self.assertEqual(card_config["confidence_intervals"][0]["path"], "outputs/minimal/base/bsc_ci/bootstrap_ci.json")
        self.assertEqual(card_config["evidence_matrix"], "outputs/minimal/base/evidence/evidence_matrix.json")
        self.assertEqual(card_config["readiness_report"], "outputs/minimal/base/readiness/readiness_report.json")
        self.assertEqual(card_config["gold_validations"][0]["path"], "outputs/gold.json")
        self.assertEqual(card_config["dashboard"], "outputs/dashboard.json")

    def test_build_result_card_config_uses_gold_validation_output_path(self) -> None:
        config = sample_config()
        config["gold_validation"] = {
            "enabled": True,
            "output_json": "outputs/custom/rubricbench_gold_validation.json",
        }
        config["result_card"] = {
            "enabled": True,
            "gold_validations": [
                {"name": "Duplicate", "path": "outputs/custom/rubricbench_gold_validation.json"},
                {"name": "Extra Gold Check", "path": "outputs/extra_gold.json"},
            ],
        }

        card_config = build_result_card_config(config, manifest_path="manifest.json")

        self.assertEqual(
            card_config["gold_validations"],
            [
                {
                    "name": "RubricBench Gold Validation",
                    "path": "outputs/custom/rubricbench_gold_validation.json",
                },
                {"name": "Extra Gold Check", "path": "outputs/extra_gold.json"},
            ],
        )

    def test_build_result_card_config_uses_query_validation_output_path(self) -> None:
        config = sample_config()
        config["query_validation"] = {
            "enabled": True,
            "output_json": "outputs/custom/rubricbench_query_validation.json",
        }
        config["result_card"] = {
            "enabled": True,
            "query_validations": [
                {"name": "Duplicate", "path": "outputs/custom/rubricbench_query_validation.json"},
                {"name": "Extra Query Check", "path": "outputs/extra_queries.json"},
            ],
        }

        card_config = build_result_card_config(config, manifest_path="manifest.json")

        self.assertEqual(
            card_config["query_validations"],
            [
                {
                    "name": "RubricBench Query Pool Validation",
                    "path": "outputs/custom/rubricbench_query_validation.json",
                },
                {"name": "Extra Query Check", "path": "outputs/extra_queries.json"},
            ],
        )
        self.assertNotIn("query_validation", {gate["type"] for gate in card_config["raw_audit_gates"]})

    def test_build_result_card_config_tracks_minimal_api_budget(self) -> None:
        config = sample_config()
        config["api_budget"] = {"enabled": True}
        config["result_card"] = {"enabled": True}
        card_config = build_result_card_config(config, manifest_path="manifest.json")

        budget_gate = next(gate for gate in card_config["raw_audit_gates"] if gate["type"] == "api_budget")
        self.assertEqual(budget_gate["name"], "Minimal Claim API Budget")
        self.assertEqual(budget_gate["path"], "outputs/minimal/base/api_budget/model_rubrics_budget.json")

    def test_build_result_card_config_tracks_minimal_api_handoff(self) -> None:
        config = sample_config()
        config["minimal_api_handoff"] = {"enabled": True}
        config["result_card"] = {"enabled": True}
        card_config = build_result_card_config(config, manifest_path="manifest.json")

        handoff_gate = next(gate for gate in card_config["raw_audit_gates"] if gate["type"] == "minimal_api_handoff")
        self.assertEqual(handoff_gate["name"], "Minimal Claim API Handoff")
        self.assertEqual(handoff_gate["path"], "outputs/minimal/base/handoff/api_handoff.json")

    def test_build_result_card_config_tracks_minimal_verifier_gates(self) -> None:
        config = sample_config()
        config["model_output_validation"] = {"enabled": True}
        config["model_verification"] = {"enabled": True, "provider": "verifier.jsonl"}
        config["verified_output_validation"] = {"enabled": True}
        config["result_card"] = {"enabled": True}
        card_config = build_result_card_config(config, manifest_path="manifest.json")

        paths = {gate["path"] for gate in card_config["raw_audit_gates"]}
        self.assertIn("outputs/minimal/base/validation/model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/minimal/base/api_budget/model_rubrics_verifier_budget.json", paths)
        self.assertIn("data/processed/minimal/base/model_rubrics_verified.jsonl", paths)
        self.assertIn("outputs/minimal/base/verifier/model_rubrics_stats.jsonl", paths)
        self.assertIn("outputs/minimal/base/validation/model_rubrics_verified/validation_report.json", paths)

    def test_build_result_card_config_tracks_minimal_preflight_as_hard_gate(self) -> None:
        config = sample_config()
        config["preflight"] = {"enabled": True}
        config["result_card"] = {"enabled": True}
        card_config = build_result_card_config(config, manifest_path="manifest.json")

        preflight_gate = next(gate for gate in card_config["raw_audit_gates"] if gate["name"] == "Minimal Claim Preflight")
        self.assertEqual(preflight_gate["type"], "preflight")
        self.assertEqual(preflight_gate["path"], "outputs/minimal/base/preflight/preflight.json")


def sample_config() -> dict:
    return {
        "method": "base",
        "model_filter": "base",
        "gold": "gold.jsonl",
        "queries": "queries.jsonl",
        "providers": "providers.jsonl",
        "processed_root": "data/processed/minimal/base",
        "output_root": "outputs/minimal/base",
        "model_rubrics": "data/processed/minimal/base/model_rubrics.jsonl",
        "embedding_model": "token-overlap",
        "coverage_tau": 0.5,
        "redundancy_tau": 0.9,
        "resume": True,
        "claim_gates": {"min_n": 10, "min_blind": 0.1},
    }


if __name__ == "__main__":
    unittest.main()
