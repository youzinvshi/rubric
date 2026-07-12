from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from scripts.build_semantic_space_visualization import POINT_CSV_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
MAIN_MATRIX_METHODS = {"base", "gpt4o", "claude", "sft_only", "sft_rl"}
JUDGEBENCH_METHODS = {
    "judgebench_base",
    "judgebench_gpt4o",
    "judgebench_claude",
    "judgebench_sft_only",
    "judgebench_sft_rl",
}
REWARDBENCH2_METHODS = {
    "rewardbench2_base",
    "rewardbench2_gpt4o",
    "rewardbench2_claude",
    "rewardbench2_sft_only",
    "rewardbench2_sft_rl",
}


class RealTemplateConsistencyTest(unittest.TestCase):
    def test_minimal_provider_examples_match_preflight_contract(self) -> None:
        generator = read_jsonl("configs/generators_minimal.example.jsonl")
        verifier = read_jsonl("configs/verifier_minimal.example.jsonl")
        config = read_json("configs/minimal_claim_real.template.json")
        required_env = set(config["preflight"]["required_env"])

        self.assertEqual([item["name"] for item in generator], ["base"])
        self.assertEqual(generator[0]["api_key_env"], "LOCAL_OPENAI_API_KEY")
        self.assertEqual([item["name"] for item in verifier], ["meta-verifier"])
        self.assertEqual(verifier[0]["api_key_env"], "OPENAI_API_KEY")
        self.assertIn(generator[0]["api_key_env"], required_env)
        self.assertIn(verifier[0]["api_key_env"], required_env)

    def test_minimal_result_card_does_not_depend_on_real_run_dashboard(self) -> None:
        template = read_json("configs/minimal_claim_real.template.json")
        generated = read_json("configs/result_card_minimal_claim.generated.json")

        self.assertNotIn("dashboard", template["result_card"])
        self.assertIsNone(generated.get("dashboard"))

    def test_result_card_real_uses_main_matrix_outputs(self) -> None:
        config = read_json("configs/result_card_real.template.json")
        paths = list(collect_paths(config))

        self.assertIn("C5 single-teacher vs multi-teacher union", config["scope"])
        self.assertIn("full required evidence set", config["safe_claim"])
        self.assertIn("C0, C1, C2, C3, C4, C5, C6, C7, C9, C10, C12, C13, C14", config["safe_claim"])
        self.assertIn("teacher-union effects", config["deferred_claim"])
        self.assertIn("RL-stage coverage-change support", config["deferred_claim"])
        self.assertNotIn("SFT/RL reduction", config["deferred_claim"])
        self.assertFalse(any(path.startswith("outputs/bsc/") for path in paths))
        self.assertFalse(any(path.startswith("outputs/downstream/") for path in paths))
        self.assertIn("outputs/rebuttal_pack/rebuttal_pack_manifest.json", paths)

        bsc_methods = {item["method"] for item in config["bsc_summaries"]}
        downstream_methods = {item["method"] for item in config["downstream_summaries"]}
        self.assertEqual(bsc_methods, MAIN_MATRIX_METHODS)
        self.assertTrue(MAIN_MATRIX_METHODS.issubset(downstream_methods))
        self.assertTrue(JUDGEBENCH_METHODS.issubset(downstream_methods))
        self.assertTrue(REWARDBENCH2_METHODS.issubset(downstream_methods))

        for method in MAIN_MATRIX_METHODS:
            self.assertIn(f"outputs/matrix_real/{method}/bsc/summary.json", paths)
            self.assertIn(f"outputs/matrix_real/{method}/downstream/summary.json", paths)
            self.assertIn(f"outputs/matrix_real/{method}/bsc_ci/bootstrap_ci.json", paths)
            self.assertIn(f"outputs/matrix_real/{method}/downstream_ci/bootstrap_ci.json", paths)
        for method in JUDGEBENCH_METHODS:
            self.assertIn(f"outputs/matrix_judgebench/{method}/downstream/summary.json", paths)
        for method in REWARDBENCH2_METHODS:
            self.assertIn(f"outputs/matrix_rewardbench2/{method}/downstream/summary.json", paths)

    def test_real_evidence_method_claim_titles_are_hypothesis_framed(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        hypothesis_claims = {
            "C2": "tested for hard-gold evaluation-dimension coverage changes",
            "C3": "tested for whether coverage changes remain reportable under redundancy and hallucination controls",
            "C4": "tested for downstream chosen-vs-rejected utility",
            "C5": "evaluated against the best single teacher",
            "C8": "tested for cross-domain coverage preservation",
            "C9": "tested for chosen-vs-rejected utility",
            "C10": "tested for multi-candidate utility",
            "C12": "evaluated with a dimension-transition audit",
            "C14": "tested against SFT-only",
        }

        for claim_id, phrase in hypothesis_claims.items():
            with self.subTest(claim_id=claim_id):
                self.assertIn(phrase, claims[claim_id]["claim"])

        claim_titles = "\n".join(claim["claim"] for claim in claims.values())
        claim_notes = "\n".join(str(claim.get("notes", "")) for claim in claims.values())
        self.assertIn("hard-gold domain evaluation dimensions", claim_notes)
        self.assertNotIn("hard-gold domain rubrics", claim_notes)
        for overclaim in [
            "BSC reward training improves",
            "preserve or improve downstream",
            "criteria union covers more",
            "GRPO/RLVR stage improves",
            "coverage gains that are not explained",
            "coverage gains over the base policy",
            "policy preserves cross-domain coverage",
            "preserve or improve chosen-vs-rejected",
            "preserve or improve multi-candidate",
        ]:
            with self.subTest(overclaim=overclaim):
                self.assertNotIn(overclaim, claim_titles)

    def test_downstream_rlvr_manual_gate_failure_is_missing_evidence_not_contradiction(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        manual_gate = next(
            metric for metric in claims["C11"]["metrics"] if metric["label"] == "Downstream RLVR manual gate ok"
        )

        self.assertEqual(manual_gate["fail_status"], "missing")

    def test_dashboard_real_tracks_main_matrix_confidence_intervals(self) -> None:
        config = read_json("configs/dashboard_real.template.json")
        paths = list(collect_paths(config))
        section_names = {section["name"] for section in config["sections"]}
        section_types = {section["name"]: section["type"] for section in config["sections"]}

        for method in MAIN_MATRIX_METHODS:
            self.assertIn(f"outputs/matrix_real/{method}/bsc_ci/bootstrap_ci.json", paths)
            self.assertIn(f"outputs/matrix_real/{method}/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("Minimal Claim Evidence", section_names)
        self.assertEqual(section_types["Minimal Claim Evidence"], "evidence")
        self.assertNotIn("Minimal Claim Readiness", section_names)
        self.assertNotIn("outputs/minimal_claim/base/readiness/readiness_report.json", paths)
        self.assertIn("C5 Teacher-Union Ablation JSON", section_names)
        self.assertIn("C5 Teacher-Union Ablation Table", section_names)
        self.assertIn("C5 Teacher-Union Per-Item Audit", section_names)
        compile_sections = {
            section["name"]: section
            for section in config["sections"]
            if section["name"] == "AAAI LaTeX Compile"
        }
        self.assertEqual(compile_sections["AAAI LaTeX Compile"]["type"], "latex_compile")
        self.assertIn("outputs/teacher_union_ablation/teacher_union_ablation.json", paths)
        self.assertIn("outputs/teacher_union_ablation/teacher_union_ablation.csv", paths)
        self.assertIn("outputs/teacher_union_ablation/teacher_union_per_item.csv", paths)
        self.assertIn("outputs/submission_readiness/latex_compile_report.json", paths)
        self.assertIn("outputs/submission_readiness/gap_report/submission_gap_report.json", paths)
        self.assertIn("outputs/rebuttal_pack/rebuttal_pack_manifest.json", paths)
        self.assertIn("Submission Gap Report", section_names)
        self.assertEqual(
            next(section for section in config["sections"] if section["name"] == "Submission Gap Report")["type"],
            "submission_gap_report",
        )
        self.assertIn("Rebuttal Pack Manifest", section_names)
        self.assertEqual(
            next(section for section in config["sections"] if section["name"] == "Rebuttal Pack Manifest")["type"],
            "rebuttal_manifest",
        )
        self.assertIn("outputs/preflight/real_run_preflight.json", paths)
        self.assertIn("outputs/api_budget/model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/judgebench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench2_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/rmbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/teacher_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/healthbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/writingbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/healthbench_teacher_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/writingbench_teacher_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/meta_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rubricbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/judgebench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench2_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rmbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/healthbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/writingbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/verifier/rubricbench_model_rubrics_stats.jsonl", paths)
        self.assertIn("outputs/validation/rubricbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/rewardbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/judgebench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/rewardbench2_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/rmbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/healthbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/writingbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/teacher_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/teacher_rubrics_filtered/validation_report.json", paths)
        self.assertIn("outputs/matrix_real/trained_method_gate.json", paths)
        self.assertIn("outputs/matrix_judgebench/trained_method_gate.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/trained_method_gate.json", paths)
        self.assertIn("outputs/generalization_matrix/trained_method_gate.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json", paths)
        self.assertIn("outputs/contamination_audit/hard_gold_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_judgebench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_queries_rubricbench_train_seed_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_queries_researchrubrics_train_seed_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_multicandidate_rubricbench_train_seed_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_rewardbench_downstream_audit.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_judgebench_downstream_audit.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_rewardbench2_downstream_audit.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/judgebench_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/teacher_rubrics_hard_gold_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/teacher_rubrics_rewardbench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/teacher_rubrics_judgebench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/blindspot_sft_hard_gold_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/blindspot_sft_rewardbench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/blindspot_sft_judgebench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/proxy_gold_hard_gold_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/proxy_gold_rewardbench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/proxy_gold_judgebench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json", paths)
        for section_name in [
            "RewardBench Proxy-Train Holdout Filter",
            "Hard-Gold Holdout Contamination Audit",
            "RewardBench Downstream Holdout Contamination Audit",
            "RewardBench Proxy-Train vs RewardBench Holdout Filter",
            "RewardBench Proxy-Train vs JudgeBench Holdout Filter",
            "RewardBench Proxy-Train vs RewardBench-2 Holdout Filter",
            "RewardBench-2 Query Pool vs RubricBench Train-Seed Filter",
            "RewardBench-2 Query Pool vs ResearchRubrics Train-Seed Filter",
            "RewardBench-2 Multicandidate vs RubricBench Train-Seed Filter",
            "RewardBench-2 Multicandidate vs ResearchRubrics Train-Seed Filter",
            "Pre-SFT Clean Proxy-Train vs Hard-Gold Audit",
            "Pre-SFT Clean Proxy-Train vs RewardBench Holdout Audit",
            "Pre-SFT Clean Proxy-Train vs JudgeBench Holdout Audit",
            "Pre-SFT Clean Proxy-Train vs RewardBench-2 Holdout Audit",
            "JudgeBench Downstream Holdout Contamination Audit",
            "RewardBench-2 Downstream Holdout Contamination Audit",
            "Teacher Rubrics vs Hard-Gold Holdout Filter",
            "Teacher Rubrics vs RewardBench Holdout Filter",
            "Teacher Rubrics vs JudgeBench Holdout Filter",
            "Teacher Rubrics vs RewardBench-2 Holdout Filter",
            "BlindSpot SFT vs Hard-Gold Holdout Filter",
            "BlindSpot SFT vs RewardBench Holdout Filter",
            "BlindSpot SFT vs JudgeBench Holdout Filter",
            "BlindSpot SFT vs RewardBench-2 Holdout Filter",
            "Proxy-Gold vs Hard-Gold Holdout Filter",
            "Proxy-Gold vs RewardBench Holdout Filter",
            "Proxy-Gold vs JudgeBench Holdout Filter",
            "Proxy-Gold vs RewardBench-2 Holdout Filter",
        ]:
            with self.subTest(section_name=section_name):
                self.assertEqual(section_types[section_name], "contamination_audit")
        self.assertIn(
            "Keep every claim with a non-safe Evidence Matrix row out of the paper claim surface.",
            config["next_actions"],
        )
        self.assertNotIn(
            "Keep contradicted or missing-evidence rows out of the paper claim surface.",
            config["next_actions"],
        )

    def test_synced_real_run_dashboard_tracks_clean_contamination_audits(self) -> None:
        for path in [
            "outputs/dashboard/real_run_dashboard.json",
            "outputs/paper_artifacts/real_run_dashboard.json",
            "paper/asset_index/real_run_dashboard.json",
        ]:
            with self.subTest(path=path):
                dashboard = read_json(path)
                sections = {section["name"]: section for section in dashboard["sections"]}
                proxy_filter = sections["RewardBench Proxy-Train Holdout Filter"]
                hard_gold = sections["Hard-Gold Holdout Contamination Audit"]
                sft_rewardbench2_filter = sections["BlindSpot SFT vs RewardBench-2 Holdout Filter"]
                proxy_gold_rewardbench2_filter = sections["Proxy-Gold vs RewardBench-2 Holdout Filter"]

                self.assertEqual(proxy_filter["type"], "contamination_audit")
                self.assertEqual(hard_gold["type"], "contamination_audit")
                self.assertEqual(hard_gold["status"], "pass")
                self.assertIn("overlap_status=clear", hard_gold["summary"])
                self.assertEqual(sft_rewardbench2_filter["type"], "contamination_audit")
                self.assertEqual(sft_rewardbench2_filter["status"], "pass")
                self.assertIn("removed=1", sft_rewardbench2_filter["summary"])
                self.assertEqual(proxy_gold_rewardbench2_filter["type"], "contamination_audit")
                self.assertEqual(proxy_gold_rewardbench2_filter["status"], "pass")
                self.assertIn("removed=1", proxy_gold_rewardbench2_filter["summary"])

    def test_api_budget_real_stages_are_strict_and_bounded(self) -> None:
        config = read_json("configs/api_budget_real.template.json")

        for stage in config["stages"]:
            with self.subTest(stage=stage["name"]):
                args = stage["args"]
                self.assertTrue(args["strict"])
                self.assertGreater(args["max_calls"], 0)
                self.assertGreater(args["max_total_tokens"], 0)
                self.assertGreater(args["max_cost_usd"], 0)
                self.assertGreater(args["max_wallclock_minutes_serial"], 0)

    def test_paid_generation_stages_require_matching_budget_reports(self) -> None:
        assembly = read_json("configs/real_run_assembly.template.json")
        sft_data = read_json("configs/sft_data_real.template.json")
        expected_preflight_report = "outputs/preflight/sft_data_preflight.json"
        budget_outputs = {
            stage["args"]["output"]
            for stage in read_json("configs/api_budget_real.template.json")["stages"]
        }
        api_budget_stages = {
            stage["name"]: stage
            for stage in read_json("configs/api_budget_real.template.json")["stages"]
        }

        self.assertEqual(
            api_budget_stages["api_budget_teacher_rubrics"]["args"]["input"],
            "data/processed/splits/rubricbench_gold_train_seed.jsonl",
        )

        for domain in assembly["model_generation"]["domains"]:
            with self.subTest(stage=domain["name"]):
                self.assertIn(domain["require_budget_report"], budget_outputs)

        teacher_stages = [stage for stage in sft_data["stages"] if stage["type"] == "generate_teachers"]
        self.assertGreater(len(teacher_stages), 0)
        rubricbench_teacher = {
            stage["name"]: stage for stage in teacher_stages
        }["generate_teacher_rubrics_rubricbench"]
        self.assertEqual(
            rubricbench_teacher["args"]["input"],
            "data/processed/splits/rubricbench_gold_train_seed.jsonl",
        )
        self.assertNotEqual(rubricbench_teacher["args"]["input"], "data/processed/rubricbench_queries.jsonl")
        for stage in teacher_stages:
            with self.subTest(stage=stage["name"]):
                self.assertIn(stage["args"]["require_budget_report"], budget_outputs)
                self.assertEqual(
                    stage["args"]["require_preflight_report"],
                    expected_preflight_report,
                )

    def test_real_run_api_stages_use_assembly_preflight_report(self) -> None:
        sft_data = read_json("configs/sft_data_real.template.json")
        generated = read_json("configs/pipeline_real_run.generated.json")
        expected_sft_preflight_report = "outputs/preflight/sft_data_preflight.json"
        expected_preflight_report = expected_real_preflight_report_path()

        sft_api_stages = [
            stage
            for stage in sft_data["stages"]
            if stage["type"] in {"generate_teachers", "filter_verifier"}
        ]
        self.assertGreater(len(sft_api_stages), 0)
        for stage in sft_api_stages:
            with self.subTest(config="sft_data_real", stage=stage["name"]):
                self.assertEqual(stage["args"].get("require_preflight_report"), expected_sft_preflight_report)

        generated_api_stages = [
            stage
            for stage in generated["stages"]
            if stage["type"] == "generate_model_rubrics"
            or (stage["type"] == "filter_verifier" and stage["name"].startswith("verify_"))
        ]
        self.assertGreater(len(generated_api_stages), 0)
        for stage in generated_api_stages:
            with self.subTest(config="pipeline_real_run", stage=stage["name"]):
                self.assertEqual(stage["args"].get("require_preflight_report"), expected_preflight_report)

    def test_generated_api_stages_have_matching_budget_contracts(self) -> None:
        for relative_path in [
            "configs/pipeline_minimal_claim.generated.json",
            "configs/pipeline_real_run.generated.json",
            "configs/pipeline_matrix_real.generated.json",
            "configs/pipeline_matrix_judgebench.generated.json",
            "configs/pipeline_matrix_rewardbench2.generated.json",
        ]:
            with self.subTest(pipeline=relative_path):
                assert_pipeline_budget_contracts(self, read_json(relative_path))

    def test_real_run_generated_includes_pre_sft_clean_proxy_audits(self) -> None:
        pipeline = read_json("configs/pipeline_real_run.generated.json")
        manifest = read_json("configs/manifest_real_run.generated.json")
        names = [stage["name"] for stage in pipeline["stages"]]
        required = set(manifest["required_files"])

        expected = [
            "filter_rewardbench2_queries_rubricbench_train_seed_overlap",
            "filter_rewardbench2_queries_researchrubrics_train_seed_overlap",
            "filter_rewardbench2_multicandidate_rubricbench_train_seed_overlap",
            "filter_rewardbench2_multicandidate_researchrubrics_train_seed_overlap",
            "audit_clean_proxy_train_vs_hard_gold",
            "audit_clean_proxy_train_vs_rewardbench_downstream",
            "audit_clean_proxy_train_vs_judgebench_downstream",
            "audit_clean_proxy_train_vs_rewardbench2_downstream",
        ]
        for name in expected:
            self.assertIn(name, names)
        for name in [
            "split_rubricbench_gold_1",
            "split_researchrubrics_gold_1",
            "split_rewardbench_preference_1",
        ]:
            self.assertIn(name, names)
        self.assertLess(names.index("split_rubricbench_gold_1"), names.index("filter_rewardbench_sft_proxy_train_holdout_overlap"))
        self.assertLess(names.index("split_rewardbench_preference_1"), names.index("filter_rewardbench_sft_proxy_train_holdout_overlap"))
        self.assertLess(
            names.index("filter_rewardbench_sft_proxy_train_rewardbench2_downstream_overlap"),
            names.index("audit_clean_proxy_train_vs_hard_gold"),
        )
        self.assertLess(
            names.index("filter_rewardbench2_multicandidate_researchrubrics_train_seed_overlap"),
            names.index("filter_rewardbench_sft_proxy_train_rewardbench2_downstream_overlap"),
        )
        preflight_idx = next(idx for idx, stage in enumerate(pipeline["stages"]) if stage["type"] == "preflight")
        self.assertLess(
            names.index("filter_rewardbench2_queries_researchrubrics_train_seed_overlap"),
            preflight_idx,
        )
        self.assertLess(names.index("audit_clean_proxy_train_vs_rewardbench2_downstream"), preflight_idx)
        for stem in [
            "hard_gold",
            "rewardbench_downstream",
            "judgebench_downstream",
            "rewardbench2_downstream",
        ]:
            self.assertIn(f"outputs/contamination_audit/clean_proxy_train_vs_{stem}_audit.json", required)
            self.assertIn(f"outputs/contamination_audit/clean_proxy_train_vs_{stem}_overlaps.csv", required)
        for path in [
            "outputs/data_splits/rubricbench_gold_split.json",
            "data/processed/splits/rubricbench_gold_train_seed.jsonl",
            "data/processed/splits/rubricbench_gold_test_main.jsonl",
            "outputs/data_splits/researchrubrics_gold_split.json",
            "data/processed/splits/researchrubrics_gold_train_seed.jsonl",
            "outputs/data_splits/rewardbench_pref_split.json",
            "data/processed/splits/rewardbench_pref_sft_proxy_train.jsonl",
            "data/processed/splits/rewardbench_pref_downstream_holdout.jsonl",
            "data/processed/rewardbench2_queries.clean.jsonl",
            "data/processed/rewardbench2_multicandidate.clean.jsonl",
            "outputs/contamination_audit/rewardbench2_queries_rubricbench_train_seed_filter.json",
            "outputs/contamination_audit/rewardbench2_queries_researchrubrics_train_seed_filter.json",
            "outputs/contamination_audit/rewardbench2_multicandidate_rubricbench_train_seed_filter.json",
            "outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json",
        ]:
            self.assertIn(path, required)

    def test_real_run_generated_includes_endpoint_bsc_human_audit_packs(self) -> None:
        pipeline = read_json("configs/pipeline_real_run.generated.json")
        manifest = read_json("configs/manifest_real_run.generated.json")
        names = [stage["name"] for stage in pipeline["stages"]]
        required = set(manifest["required_files"])

        self.assertIn("base_bsc_human_audit_pack", names)
        self.assertIn("sft_rl_bsc_human_audit_pack", names)
        self.assertLess(names.index("base_bsc"), names.index("base_bsc_human_audit_pack"))
        self.assertLess(names.index("sft_rl_bsc"), names.index("sft_rl_bsc_human_audit_pack"))
        for method in ["base", "sft_rl"]:
            self.assertIn(f"outputs/matrix_real/{method}/bsc_human_audit_pack/summary.json", required)
            self.assertIn(f"outputs/matrix_real/{method}/bsc_human_audit_pack/audit_items.csv", required)
            self.assertIn(f"outputs/matrix_real/{method}/bsc_human_audit_pack/audit_items.jsonl", required)
            self.assertIn(f"outputs/matrix_real/{method}/bsc_human_audit_pack/human_label_summary.json", required)
            self.assertIn(f"outputs/matrix_real/{method}/bsc_human_audit_pack/human_label_summary.md", required)

    def test_meta_verifier_filter_requires_rubric_level_budget_report(self) -> None:
        sft_data = read_json("configs/sft_data_real.template.json")
        stages = {stage["name"]: stage for stage in sft_data["stages"]}
        expected_preflight_report = "outputs/preflight/sft_data_preflight.json"
        budget = stages["api_budget_meta_verifier"]
        verifier = stages["filter_teacher_rubrics"]

        self.assertEqual(budget["type"], "api_budget")
        self.assertEqual(budget["args"]["unit_field"], "rubrics")
        self.assertTrue(budget["args"]["strict"])
        self.assertEqual(verifier["args"]["require_budget_report"], budget["args"]["output"])
        self.assertEqual(verifier["args"]["require_preflight_report"], expected_preflight_report)

    def test_model_rubric_verifier_stages_are_budgeted_and_annotate_only(self) -> None:
        pipeline = read_json("configs/pipeline_real_run.generated.json")
        stages = {stage["name"]: stage for stage in pipeline["stages"]}
        budget = stages["api_budget_rubricbench_model_rubrics_verifier"]
        verifier = stages["verify_rubricbench_model_rubrics"]

        self.assertEqual(budget["type"], "api_budget")
        self.assertEqual(budget["args"]["input"], "data/processed/rubricbench_model_rubrics.jsonl")
        self.assertEqual(budget["args"]["providers"], "configs/verifier.local.jsonl")
        self.assertEqual(budget["args"]["unit_field"], "rubrics")
        self.assertTrue(budget["args"]["strict"])
        self.assertEqual(verifier["type"], "filter_verifier")
        self.assertTrue(verifier["args"]["annotate_only"])
        self.assertEqual(verifier["args"]["input"], "data/processed/rubricbench_model_rubrics.jsonl")
        self.assertEqual(verifier["args"]["output"], "data/processed/rubricbench_model_rubrics.jsonl")
        self.assertEqual(verifier["args"]["require_budget_report"], budget["args"]["output"])

    def test_sft_data_requires_filtered_teacher_validation_and_multi_teacher_rows(self) -> None:
        sft_data = read_json("configs/sft_data_real.template.json")
        sft_manifest = read_json("configs/manifest_sft_data_real.template.json")
        generated_pipeline = read_json("configs/pipeline_real_run.generated.json")
        generated_manifest = read_json("configs/manifest_real_run.generated.json")
        stages = {stage["name"]: stage for stage in sft_data["stages"]}
        generated_stages = {stage["name"]: stage for stage in generated_pipeline["stages"]}
        names = [stage["name"] for stage in sft_data["stages"]]

        sft_preflight = stages["sft_data_preflight"]
        self.assertEqual(sft_preflight["type"], "preflight")
        self.assertEqual(sft_preflight["args"]["output"], "outputs/preflight/sft_data_preflight.json")
        self.assertTrue(sft_preflight["args"]["strict"])
        self.assertLess(names.index("sft_data_preflight"), names.index("generate_teacher_rubrics_rubricbench"))
        self.assertLess(names.index("filter_teacher_rubrics"), names.index("filter_teacher_rubrics_hard_gold_holdout_overlap"))
        self.assertLess(
            names.index("filter_teacher_rubrics_hard_gold_holdout_overlap"),
            names.index("filter_teacher_rubrics_rewardbench_downstream_overlap"),
        )
        self.assertLess(
            names.index("filter_teacher_rubrics_rewardbench_downstream_overlap"),
            names.index("filter_teacher_rubrics_judgebench_downstream_overlap"),
        )
        self.assertLess(
            names.index("filter_teacher_rubrics_judgebench_downstream_overlap"),
            names.index("filter_teacher_rubrics_rewardbench2_downstream_overlap"),
        )
        self.assertLess(
            names.index("filter_teacher_rubrics_rewardbench2_downstream_overlap"),
            names.index("validate_filtered_teacher_rubrics"),
        )
        self.assertLess(names.index("validate_filtered_teacher_rubrics"), names.index("build_sft_all_domains"))

        validation = stages["validate_filtered_teacher_rubrics"]
        self.assertEqual(validation["type"], "validate_rubrics")
        self.assertEqual(validation["args"]["input"], "data/processed/teacher_rubrics_training_clean.jsonl")
        self.assertEqual(validation["args"]["output_dir"], "outputs/validation/teacher_rubrics_filtered")
        self.assertTrue(validation["args"]["strict"])

        for stage_name in ["build_sft_all_domains", "build_healthbench_proxy_gold", "build_writingbench_proxy_gold"]:
            self.assertEqual(stages[stage_name]["args"]["input"], "data/processed/teacher_rubrics_training_clean.jsonl")
            self.assertEqual(stages[stage_name]["args"]["min_teachers"], 2)
            self.assertIn("report_output", stages[stage_name]["args"])
        self.assertEqual(
            stages["build_sft_all_domains"]["args"]["report_output"],
            "outputs/sft_data/proxy_gold_build_report.json",
        )
        self.assertEqual(
            stages["build_healthbench_proxy_gold"]["args"]["report_output"],
            "outputs/sft_data/healthbench_proxy_gold_build_report.json",
        )
        self.assertEqual(
            stages["build_writingbench_proxy_gold"]["args"]["report_output"],
            "outputs/sft_data/writingbench_proxy_gold_build_report.json",
        )

        convert_verl = stages["convert_proxy_gold_to_verl"]
        self.assertEqual(convert_verl["type"], "convert_verl")
        self.assertEqual(convert_verl["args"]["input"], "data/processed/proxy_gold.jsonl")
        self.assertEqual(convert_verl["args"]["output"], "data/processed/proxy_gold_verl.parquet")
        self.assertEqual(convert_verl["args"]["report_output"], "outputs/sft_data/proxy_gold_verl_report.json")
        self.assertGreaterEqual(convert_verl["args"]["min_records"], 1000)
        self.assertEqual(
            generated_stages["convert_proxy_gold_to_verl"]["args"]["report_output"],
            "outputs/sft_data/proxy_gold_verl_report.json",
        )

        required = set(sft_manifest["required_files"])
        generated_required = set(generated_manifest["required_files"])
        self.assertIn("outputs/preflight/sft_data_preflight.json", required)
        self.assertIn("outputs/preflight/sft_data_preflight.md", required)
        self.assertIn("outputs/preflight/sft_data_preflight.json", generated_required)
        for path in [
            "data/processed/teacher_rubrics_filtered.hard_gold_clean.jsonl",
            "outputs/contamination_audit/teacher_rubrics_hard_gold_holdout_filter.json",
            "data/processed/teacher_rubrics_filtered.rewardbench_clean.jsonl",
            "outputs/contamination_audit/teacher_rubrics_rewardbench_holdout_filter.json",
            "data/processed/teacher_rubrics_filtered.judgebench_clean.jsonl",
            "outputs/contamination_audit/teacher_rubrics_judgebench_holdout_filter.json",
            "data/processed/teacher_rubrics_training_clean.jsonl",
            "outputs/contamination_audit/teacher_rubrics_rewardbench2_holdout_filter.json",
        ]:
            with self.subTest(path=path):
                self.assertIn(path, required)
                self.assertIn(path, generated_required)
        self.assertIn("outputs/validation/teacher_rubrics_filtered/validation_report.json", required)
        self.assertIn("outputs/validation/teacher_rubrics_filtered/validation_report.md", required)
        self.assertIn("outputs/validation/teacher_rubrics_filtered/per_record.jsonl", required)
        self.assertIn("outputs/sft_data/proxy_gold_build_report.json", required)
        self.assertIn("outputs/sft_data/proxy_gold_verl_report.json", required)
        self.assertIn("outputs/sft_data/proxy_gold_build_report.json", generated_required)
        self.assertIn("outputs/sft_data/proxy_gold_verl_report.json", generated_required)

    def test_evidence_matrix_real_gates_main_claims_with_ci(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        c1_metrics = collect_metric_names(claims["C1"])
        c2_metrics = collect_metric_names(claims["C2"])
        c3_metrics = collect_metric_names(claims["C3"])
        c4_metrics = collect_metric_names(claims["C4"])
        c4_metric_specs = collect_metric_specs(claims["C4"])
        c4_value_keys = collect_value_keys(claims["C4"])
        c4_value_specs = collect_value_specs(claims["C4"])
        c4_value_comparison_keys = collect_value_comparison_keys(claims["C4"])
        c4_table_value_specs = collect_table_value_specs(claims["C4"])
        c1_value_keys = collect_value_keys(claims["C1"])
        c2_value_keys = collect_value_keys(claims["C2"])
        c2_value_specs = collect_value_specs(claims["C2"])
        c3_value_keys = collect_value_keys(claims["C3"])
        paths = list(collect_paths(config))

        self.assertIn("coverage_tau", c1_metrics)
        self.assertIn("redundancy_tau", c1_metrics)
        self.assertIn("median_blind", c1_metrics)
        self.assertIn("queries_coverage_le_0_5", c1_metrics)
        self.assertIn("data_source_counts.rubricbench", c1_metrics)
        self.assertIn("verifier_source_counts.valid_flags", c1_metrics)
        self.assertIn("embedding_model", c1_value_keys)
        self.assertIn("weights.coverage", c1_value_keys)
        self.assertIn("weights.validity", c1_value_keys)
        self.assertIn("weights.redundancy", c1_value_keys)
        self.assertIn("verifier_source", c1_value_keys)
        self.assertIn("metrics[metric=coverage].ci_lower", c2_metrics)
        self.assertIn("metrics[metric=coverage].ci_lower", c2_metrics)
        self.assertIn("metrics[metric=coverage].ci_upper", c2_metrics)
        self.assertIn("coverage_tau", c2_metrics)
        self.assertIn("data_source_counts.rubricbench", c2_metrics)
        self.assertIn("embedding_model", c2_value_keys)
        self.assertIn(
            (
                "outputs/matrix_real/base/bsc_join_report.json",
                "gold",
                "data/processed/splits/rubricbench_gold_test_main.jsonl",
            ),
            c2_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/sft_rl/bsc_join_report.json",
                "gold",
                "data/processed/splits/rubricbench_gold_test_main.jsonl",
            ),
            c2_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/base/bsc_join_report.json",
                "output",
                "data/processed/matrix_real/base/bsc_eval.jsonl",
            ),
            c2_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/sft_rl/bsc_join_report.json",
                "output",
                "data/processed/matrix_real/sft_rl/bsc_eval.jsonl",
            ),
            c2_value_specs,
        )
        self.assertIn("metrics[metric=redundancy].ci_upper", c3_metrics)
        self.assertIn("metrics[metric=hallucination].ci_upper", c3_metrics)
        self.assertIn("verifier_source_counts.valid_flags", c3_metrics)
        self.assertIn("mean_n_gen", c3_metrics)
        self.assertIn("gen_to_gold_ratio", c3_metrics)
        self.assertIn("coverage_per_generated_criterion", c3_metrics)
        self.assertIn("verifier_source", c3_value_keys)
        self.assertIn("metrics[metric=correct].ci_lower", c4_metrics)
        self.assertIn("metrics[metric=correct].ci_upper", c4_metrics)
        self.assertIn("n", c4_metrics)
        self.assertIn("overlap_query_count", c4_metrics)
        self.assertIn("holdout_unique_queries", c4_metrics)
        for path in [
            "outputs/matrix_real/base/downstream_join_report.json",
            "outputs/matrix_real/sft_rl/downstream_join_report.json",
        ]:
            with self.subTest(path=path):
                self.assertIn((path, "n_missing_rubrics", "0"), c4_metric_specs)
                self.assertIn((path, "n_unmatched_rubrics", "0"), c4_metric_specs)
                self.assertIn((path, "source_duplicate_record_count", "0"), c4_metric_specs)
                self.assertIn((path, "rubric_duplicate_record_count", "0"), c4_metric_specs)
        self.assertIn("scorer", c4_value_keys)
        self.assertIn("paper_claim_eligible", c4_value_keys)
        self.assertIn("paper_claim_eligibility_blockers", c4_value_keys)
        self.assertIn("scorer_provider", c4_value_keys)
        self.assertIn("budget_report", c4_value_keys)
        self.assertIn("input", c4_value_keys)
        self.assertIn("input_sha256", c4_value_keys)
        self.assertIn("per_item_output", c4_value_keys)
        self.assertIn("per_item_sha256", c4_value_keys)
        self.assertIn("preferences", c4_value_keys)
        self.assertIn("rubrics", c4_value_keys)
        self.assertIn("output", c4_value_keys)
        self.assertIn("output_sha256", c4_value_keys)
        self.assertIn("data_source", c4_value_keys)
        self.assertIn("model", c4_value_keys)
        self.assertIn("query_alignment_exact", c4_value_keys)
        self.assertIn("output_rows_match_n_joined", c4_value_keys)
        self.assertIn("output_truncated_by_limit", c4_value_keys)
        self.assertIn("scorer_provider_sha256", c4_value_keys)
        self.assertIn("benchmark_format", c4_value_keys)
        self.assertIn("scorer_contract.calls_per_record_per_provider", c4_value_keys)
        self.assertIn("scorer_contract.unit_field", c4_value_keys)
        self.assertIn("ok", c4_value_keys)
        self.assertIn("contract.input", c4_value_keys)
        self.assertIn("contract.providers", c4_value_keys)
        self.assertIn(("outputs/matrix_real/main_table.csv", "base", "downstream_status", "pass"), c4_table_value_specs)
        self.assertIn(
            ("outputs/matrix_real/main_table.csv", "base", "downstream_paper_claim_eligible", "true"),
            c4_table_value_specs,
        )
        self.assertIn(("outputs/matrix_real/main_table.csv", "sft_rl", "downstream_status", "pass"), c4_table_value_specs)
        self.assertIn(
            ("outputs/matrix_real/main_table.csv", "sft_rl", "downstream_paper_claim_eligible", "true"),
            c4_table_value_specs,
        )
        self.assertIn("contract.calls_per_record_per_provider", c4_value_keys)
        self.assertIn("contract.unit_field", c4_value_keys)
        self.assertIn(("input_sha256", "contract.input_sha256"), c4_value_comparison_keys)
        self.assertIn(("input_sha256", "output_sha256"), c4_value_comparison_keys)
        self.assertIn(("scorer_provider_sha256", "contract.providers_sha256"), c4_value_comparison_keys)
        self.assertIn(("per_item_sha256", "input_sha256"), c4_value_comparison_keys)
        self.assertIn(("per_item_rows", "n"), c4_value_comparison_keys)
        self.assertIn(
            (
                "outputs/matrix_real/base/downstream_join_report.json",
                "preferences",
                "data/processed/splits/rewardbench_pref_downstream_holdout.jsonl",
            ),
            c4_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/base/downstream_join_report.json",
                "rubrics",
                "data/processed/rewardbench_model_rubrics.jsonl",
            ),
            c4_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/sft_rl/downstream_join_report.json",
                "output",
                "data/processed/matrix_real/sft_rl/downstream_eval.jsonl",
            ),
            c4_value_specs,
        )
        for path in [
            "outputs/matrix_real/base/downstream_join_report.json",
            "outputs/matrix_real/sft_rl/downstream_join_report.json",
        ]:
            with self.subTest(path=path):
                self.assertIn((path, "query_alignment_exact", "True"), c4_value_specs)
                self.assertIn((path, "output_rows_match_n_joined", "True"), c4_value_specs)
                self.assertIn((path, "output_truncated_by_limit", "False"), c4_value_specs)

        self.assertIn("outputs/matrix_real/base/bsc_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_real/base/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_real/base/downstream_join_report.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/downstream_join_report.json", paths)
        self.assertIn("outputs/matrix_real/base/downstream/summary.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/downstream/summary.json", paths)
        self.assertIn("outputs/matrix_real/base/downstream_api_budget/budget.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/downstream_api_budget/budget.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_base/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_sft_rl/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_base/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_ci/bootstrap_ci.json", paths)

    def test_evidence_matrix_real_claims_use_blindspot_framing_not_generator_framing(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claim_text = "\n".join(str(item.get("claim", "")) for item in config["claims"])
        config_text = json.dumps(config, ensure_ascii=False)
        normalized = " ".join(claim_text.split()).lower()

        self.assertIn("evaluation-dimension coverage", normalized)
        self.assertIn("bsc-optimized evaluation criteria", normalized)
        self.assertIn("evaluation-criteria policy", normalized)
        self.assertIn("Teacher evaluation-criteria verifier-filtering provenance report", config_text)
        self.assertIn("Teacher verifier report writes filtered teacher evaluation criteria", config_text)
        self.assertNotIn("base generator", normalized)
        self.assertNotIn("trained generator", normalized)
        self.assertNotIn("rubrics generated by sft+rl", normalized)
        self.assertNotIn("better generator", normalized)
        self.assertNotIn("Teacher rubric verifier-filtering provenance report", config_text)
        self.assertNotIn("Teacher verifier report writes filtered teacher rubrics", config_text)

    def test_minimal_claim_freezes_section_2_diagnostic_snapshot(self) -> None:
        config = read_json("configs/minimal_claim_real.template.json")
        generated = read_json("configs/evidence_minimal_claim.generated.json")
        result_card = read_json("configs/result_card_minimal_claim.generated.json")
        frozen = config["claim_gates"]["frozen_diagnostic"]
        c1_metrics = {
            item["label"]: item
            for item in generated["claims"][0]["metrics"]
            if item["label"].startswith("Frozen diagnostic")
        }

        self.assertEqual(
            generated["claims"][0]["claim"],
            "A single-model evaluation-criteria policy leaves measurable blind spots against human-gold evaluation dimensions.",
        )
        self.assertEqual(
            result_card["scope"],
            "Minimal motivation experiment: single-model BSC against hard-gold evaluation dimensions.",
        )
        self.assertIn("evaluation-criteria policy", result_card["safe_claim"])
        self.assertIn("criteria elicitation", result_card["deferred_claim"])
        self.assertIn(
            "Minimal claim card for the first paper gate: single-model BSC against hard-gold evaluation dimensions.",
            result_card["notes"],
        )
        self.assertNotIn(
            "Minimal claim card for the first paper gate: single-model BSC against RubricBench hard gold.",
            result_card["notes"],
        )
        self.assertEqual(frozen["n"], 100)
        self.assertEqual(frozen["mean_coverage"], 0.36919517010450364)
        self.assertEqual(frozen["mean_blind"], 0.6308048298954964)
        self.assertEqual(frozen["median_blind"], 0.6666666567325592)
        self.assertEqual(frozen["queries_coverage_le_0_5"], 74)
        self.assertEqual(frozen["queries_zero_coverage"], 21)
        self.assertEqual(frozen["mean_redundancy"], 0.07675396825396825)
        self.assertEqual(frozen["mean_hallucination"], 0.12273412698412695)
        self.assertEqual(frozen["blind_ci_lower"], 0.5784969914257526)
        self.assertEqual(frozen["blind_ci_upper"], 0.6822689984422177)
        self.assertEqual(c1_metrics["Frozen diagnostic mean blind-spot rate"]["op"], "==")
        self.assertEqual(c1_metrics["Frozen diagnostic mean blind-spot rate"]["value"], frozen["mean_blind"])
        self.assertEqual(c1_metrics["Frozen diagnostic mean coverage"]["value"], frozen["mean_coverage"])
        self.assertEqual(c1_metrics["Frozen diagnostic zero-coverage queries"]["value"], frozen["queries_zero_coverage"])

    def test_evidence_matrix_real_requires_endpoint_threshold_sweeps(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(config))
        c6_metrics = collect_metric_names(claims["C6"])
        c6_value_keys = collect_value_keys(claims["C6"])

        self.assertIn("outputs/matrix_real/base/bsc_sweep/threshold_sweep.csv", paths)
        self.assertIn("outputs/matrix_real/base/bsc_sweep/threshold_sweep.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.csv", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_sweep/threshold_sweep.json", paths)
        self.assertIn("outputs/matrix_real/base/bsc_human_audit_pack/audit_items.csv", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_human_audit_pack/audit_items.csv", paths)
        self.assertIn("outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.json", paths)
        self.assertIn("outputs/matrix_real/base/bsc_human_audit_pack/human_label_summary.md", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_human_audit_pack/human_label_summary.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_human_audit_pack/human_label_summary.md", paths)
        self.assertIn("0.mean_blind", c6_metrics)
        self.assertIn("6.mean_blind", c6_metrics)
        self.assertIn("sampled_matched", c6_metrics)
        self.assertIn("sampled_unmatched", c6_metrics)
        self.assertIn("human_labels_completed", c6_metrics)
        self.assertIn("invalid_label_count", c6_metrics)
        self.assertIn("uncertain_rate", c6_metrics)
        self.assertIn("auto_matched_human_match_rate", c6_metrics)
        self.assertIn("auto_unmatched_confirmation_rate", c6_metrics)
        self.assertIn("status", c6_value_keys)
        self.assertIn("ok", c6_value_keys)

    def test_evidence_matrix_real_requires_hard_gold_contamination_audit(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C0"]))
        metrics = collect_metric_names(claims["C0"])
        value_keys = collect_value_keys(claims["C0"])
        value_specs = collect_value_specs(claims["C0"])
        file_sha256_specs = collect_file_sha256_specs(claims["C0"])
        value_comparison_keys = collect_value_comparison_keys(claims["C0"])

        self.assertIn("outputs/contamination_audit/hard_gold_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/hard_gold_holdout_overlaps.csv", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_judgebench_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_rewardbench_downstream_audit.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_judgebench_downstream_audit.json", paths)
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_rewardbench2_downstream_audit.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/judgebench_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json", paths)
        self.assertIn("outputs/verifier/teacher_rubrics_filtered_report.json", paths)
        self.assertIn("outputs/sft_data/proxy_gold_build_report.json", paths)
        self.assertIn("outputs/sft_data/proxy_gold_verl_report.json", paths)
        self.assertIn("overlap_query_count", metrics)
        self.assertIn("holdout_unique_queries", metrics)
        self.assertIn("training_unique_queries", metrics)
        self.assertIn("output_records", metrics)
        self.assertIn("ok", value_keys)
        self.assertIn("artifact_status", value_keys)
        self.assertIn("overlap_status", value_keys)
        self.assertIn("blockers", value_keys)
        self.assertIn("holdout", value_keys)
        self.assertIn("holdout_sha256", value_keys)
        self.assertIn("input", value_keys)
        self.assertIn("input_sha256", value_keys)
        self.assertIn("output", value_keys)
        self.assertIn("output_sha256", value_keys)
        self.assertIn("training[label=rewardbench_sft_proxy_train].path", value_keys)
        self.assertIn("training[label=rewardbench_sft_proxy_train].sha256", value_keys)
        self.assertIn("training[label=blindspot_sft].path", value_keys)
        self.assertIn("training[label=blindspot_sft].sha256", value_keys)
        self.assertIn("training[label=proxy_gold].sha256", value_keys)
        self.assertIn("training[label=proxy_gold_verl].sha256", value_keys)
        self.assertIn("provider_sha256", value_keys)
        self.assertIn("budget_report_sha256", value_keys)
        self.assertIn("preflight_report_sha256", value_keys)
        self.assertIn("sft_output_sha256", value_keys)
        self.assertIn("proxy_gold_output_sha256", value_keys)
        self.assertIn(("output_sha256", "input_sha256"), value_comparison_keys)
        self.assertIn(("output_sha256", "training[label=rewardbench_sft_proxy_train].sha256"), value_comparison_keys)
        self.assertIn(("output_sha256", "training[label=rewardbench_sft_proxy_train_clean].sha256"), value_comparison_keys)
        self.assertIn(("output_sha256", "training[label=proxy_gold_verl].sha256"), value_comparison_keys)
        self.assertIn(("holdout_sha256", "holdout_sha256"), value_comparison_keys)
        c0_missing_when_not_auditable = {
            "Hard-gold contamination audit is complete",
            "Hard-gold contamination audit overlap status is clear",
            "Hard-gold contamination audit has no blockers",
            "Clean proxy-train hard-gold audit is complete",
            "Clean proxy-train hard-gold audit overlap status is clear",
            "Clean proxy-train RewardBench audit is complete",
            "Clean proxy-train RewardBench audit overlap status is clear",
            "Clean proxy-train JudgeBench audit is complete",
            "Clean proxy-train JudgeBench audit overlap status is clear",
            "Clean proxy-train RewardBench-2 audit is complete",
            "Clean proxy-train RewardBench-2 audit overlap status is clear",
            "RewardBench downstream contamination audit is complete",
            "RewardBench downstream contamination audit overlap status is clear",
            "RewardBench downstream contamination audit has no blockers",
            "JudgeBench downstream contamination audit is complete",
            "JudgeBench downstream contamination audit overlap status is clear",
            "JudgeBench downstream contamination audit has no blockers",
            "RewardBench-2 downstream contamination audit is complete",
            "RewardBench-2 downstream contamination audit overlap status is clear",
            "RewardBench-2 downstream contamination audit has no blockers",
            "Proxy-gold build report binds holdout-clean teacher input",
            "Proxy-gold build report writes unfiltered SFT data",
            "Proxy-gold build report writes unfiltered proxy-gold data",
        }
        c0_value_items = {
            item["label"]: item
            for item in claims["C0"].get("values", [])
            if item.get("label") in c0_missing_when_not_auditable
        }
        self.assertEqual(set(c0_value_items), c0_missing_when_not_auditable)
        for item in c0_value_items.values():
            self.assertEqual(item.get("fail_status"), "missing")
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json",
                "holdout",
                "data/processed/splits/rubricbench_gold_test_main.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json",
                "output",
                "data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench2_queries_researchrubrics_train_seed_filter.json",
                "output",
                "data/processed/rewardbench2_queries.clean.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json",
                "output",
                "data/processed/rewardbench2_multicandidate.clean.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "artifact_status",
                "complete",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "overlap_status",
                "clear",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "blockers",
                "[]",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
                "artifact_status",
                "complete",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
                "overlap_status",
                "clear",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
                "artifact_status",
                "complete",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
                "overlap_status",
                "clear",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
                "artifact_status",
                "complete",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
                "overlap_status",
                "clear",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json",
                "blockers",
                "[]",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json",
                "blockers",
                "[]",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "holdout_sha256",
                "data/processed/splits/rubricbench_gold_test_main.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "training[label=blindspot_sft].sha256",
                "data/processed/blindspot_sft.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "training[label=proxy_gold_verl].sha256",
                "data/processed/proxy_gold_verl.parquet",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json",
                "output_sha256",
                "data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json",
                "output_sha256",
                "data/processed/rewardbench2_multicandidate.clean.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/sft_data/proxy_gold_build_report.json",
                "sft_output_sha256",
                "data/processed/blindspot_sft.unfiltered.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json",
                "output_sha256",
                "data/processed/blindspot_sft.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/sft_data/proxy_gold_verl_report.json",
                "output_sha256",
                "data/processed/proxy_gold_verl.parquet",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "output_sha256",
                "data/processed/teacher_rubrics_filtered.jsonl",
            ),
            file_sha256_specs,
        )
        self.assertIn(
            (
                "outputs/contamination_audit/hard_gold_holdout_contamination.json",
                "training[label=proxy_gold_verl].path",
                "data/processed/proxy_gold_verl.parquet",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "input",
                "data/processed/teacher_rubrics_raw.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "output",
                "data/processed/teacher_rubrics_filtered.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "mode",
                "api",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "provider",
                "configs/verifier.local.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "budget_report",
                "outputs/api_budget/meta_verifier_budget.json",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/verifier/teacher_rubrics_filtered_report.json",
                "preflight_report",
                "outputs/preflight/sft_data_preflight.json",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/sft_data/proxy_gold_build_report.json",
                "sft_output",
                "data/processed/blindspot_sft.unfiltered.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/sft_data/proxy_gold_build_report.json",
                "proxy_gold_output",
                "data/processed/proxy_gold.unfiltered.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/sft_data/proxy_gold_verl_report.json",
                "output",
                "data/processed/proxy_gold_verl.parquet",
            ),
            value_specs,
        )

    def test_evidence_matrix_real_requires_reward_component_ablation(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C7"]))
        metrics = collect_metric_names(claims["C7"])
        value_keys = collect_value_keys(claims["C7"])
        value_specs = collect_value_specs(claims["C7"])
        file_sha_specs = collect_file_sha256_specs(claims["C7"])

        self.assertIn("outputs/bsc_ablation/ablation_summary.csv", paths)
        self.assertIn("outputs/bsc_ablation/variants/full_summary.json", paths)
        self.assertIn("outputs/bsc_ablation/variants/no_red_summary.json", paths)
        self.assertIn("outputs/bsc_ablation/variants/no_valid_summary.json", paths)
        self.assertIn("outputs/bsc_ablation/variants/no_verifier_summary.json", paths)
        self.assertIn("outputs/bsc_ablation/variants/cov_only_summary.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_join_report.json", paths)
        self.assertIn("outputs/reward_component_training_ablation/training_done.json", paths)
        for variant in ["no_red", "no_valid", "no_verifier", "cov_only"]:
            with self.subTest(trained_reward_ablation=variant):
                self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc/summary.json", paths)
                self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc_join_report.json", paths)
        self.assertIn("outputs/verifier_filter_ablation/verifier_filter_ablation.csv", paths)
        self.assertIn("outputs/verifier_filter_ablation/verifier_filter_ablation.json", paths)
        self.assertIn("outputs/verifier_filter_ablation/verifier_filter_per_item.csv", paths)
        self.assertIn("n", metrics)
        self.assertIn("data_source_counts.rubricbench", metrics)
        self.assertIn("mean_redundancy", metrics)
        self.assertIn("mean_hallucination", metrics)
        self.assertIn("verifier_source_counts.valid_flags", metrics)
        self.assertIn("0.n", metrics)
        self.assertIn("1.n", metrics)
        self.assertIn("1.hallucination_delta_vs_no_verifier", metrics)
        self.assertIn("1.coverage_delta_vs_no_verifier", metrics)
        self.assertIn("reward_variants", value_keys)
        self.assertIn("variants.full.env.BSC_VERIFIER", value_keys)
        self.assertIn("variants.no_red.env.BSC_VERIFIER", value_keys)
        self.assertIn("variants.no_valid.env.BSC_W_VALID", value_keys)
        self.assertIn("variants.no_valid.env.BSC_VERIFIER", value_keys)
        self.assertIn("variants.no_verifier.env.BSC_W_VALID", value_keys)
        self.assertIn("variants.no_verifier.env.BSC_W_RED", value_keys)
        self.assertIn("variants.no_verifier.env.BSC_VERIFIER", value_keys)
        self.assertIn("variants.cov_only.env.BSC_W_VALID", value_keys)
        self.assertIn("variants.cov_only.env.BSC_W_RED", value_keys)
        self.assertIn("variants.cov_only.env.BSC_VERIFIER", value_keys)
        self.assertIn("variant", value_keys)
        self.assertIn("embedding_model", value_keys)
        self.assertIn("coverage_tau", value_keys)
        self.assertIn("redundancy_tau", value_keys)
        self.assertIn("weights.coverage", value_keys)
        self.assertIn("weights.validity", value_keys)
        self.assertIn("weights.redundancy", value_keys)
        self.assertIn("use_verifier", value_keys)
        self.assertIn("input", value_keys)
        self.assertIn("input_sha256", value_keys)
        self.assertIn("0.variant", value_keys)
        self.assertIn("0.embedding_model", value_keys)
        self.assertIn("0.coverage_tau", value_keys)
        self.assertIn("0.redundancy_tau", value_keys)
        self.assertIn("0.raw_teachers_sha256", value_keys)
        self.assertIn("0.filtered_teachers_sha256", value_keys)
        self.assertIn("0.gold_sha256", value_keys)
        self.assertIn("1.variant", value_keys)
        for variant in ["full", "no_red", "no_valid", "no_verifier", "cov_only"]:
            with self.subTest(variant=variant):
                self.assertIn(
                    (
                        f"outputs/bsc_ablation/variants/{variant}_summary.json",
                        "input",
                        "data/processed/matrix_real/sft_rl/bsc_eval.jsonl",
                    ),
                    value_specs,
                )
        self.assertIn(
            (
                "outputs/matrix_real/sft_rl/bsc_join_report.json",
                "gold",
                "data/processed/splits/rubricbench_gold_test_main.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/sft_rl/bsc_join_report.json",
                "output",
                "data/processed/matrix_real/sft_rl/bsc_eval.jsonl",
            ),
            value_specs,
        )
        for variant in ["no_red", "no_valid", "no_verifier", "cov_only"]:
            with self.subTest(trained_reward_ablation_join=variant):
                self.assertIn(
                    (
                        f"outputs/reward_component_training_ablation/{variant}/bsc_join_report.json",
                        "gold",
                        "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/reward_component_training_ablation/{variant}/bsc_join_report.json",
                        "output",
                        f"data/processed/reward_component_training_ablation/{variant}/bsc_eval.jsonl",
                    ),
                    value_specs,
                )
        self.assertIn(
            (
                "outputs/reward_component_training_ablation/training_done.json",
                "reward_variants",
                "['full', 'no_red', 'no_valid', 'no_verifier', 'cov_only']",
            ),
            value_specs,
        )
        for key, value in [
            ("variants.full.env.BSC_VERIFIER", "rule"),
            ("variants.no_red.env.BSC_VERIFIER", "rule"),
            ("variants.no_valid.env.BSC_W_VALID", "0.0"),
            ("variants.no_valid.env.BSC_VERIFIER", "none"),
            ("variants.no_verifier.env.BSC_W_VALID", "0.5"),
            ("variants.no_verifier.env.BSC_W_RED", "0.5"),
            ("variants.no_verifier.env.BSC_VERIFIER", "none"),
            ("variants.cov_only.env.BSC_W_VALID", "0.0"),
            ("variants.cov_only.env.BSC_W_RED", "0.0"),
            ("variants.cov_only.env.BSC_VERIFIER", "none"),
        ]:
            with self.subTest(reward_ablation_training_env=key):
                self.assertIn(
                    (
                        "outputs/reward_component_training_ablation/training_done.json",
                        key,
                        value,
                    ),
                    value_specs,
                )
        for key, path in [
            ("sft_data_sha256", "data/processed/blindspot_sft.jsonl"),
            ("rl_data_sha256", "data/processed/proxy_gold_verl.parquet"),
            ("rl_data_report_sha256", "outputs/sft_data/proxy_gold_verl_report.json"),
            ("reward_function_sha256", "src/blindspot_rl/verl_reward.py"),
        ]:
            with self.subTest(reward_ablation_top_level_sha=key):
                self.assertIn(
                    (
                        "outputs/reward_component_training_ablation/training_done.json",
                        key,
                        path,
                    ),
                    file_sha_specs,
                )
        for variant in ["full", "no_red", "no_valid", "no_verifier", "cov_only"]:
            for key, path in [
                ("grpo_config_sha256", "configs/verl_grpo_bsc.local.yaml"),
                ("rl_data_sha256", "data/processed/proxy_gold_verl.parquet"),
                ("rl_data_report_sha256", "outputs/sft_data/proxy_gold_verl_report.json"),
                ("reward_function_sha256", "src/blindspot_rl/verl_reward.py"),
            ]:
                with self.subTest(reward_ablation_variant_sha=f"{variant}.{key}"):
                    self.assertIn(
                        (
                            "outputs/reward_component_training_ablation/training_done.json",
                            f"variants.{variant}.{key}",
                            path,
                        ),
                        file_sha_specs,
                    )
        values = {str(item.get("value")) for item in claims["C7"].get("values", [])}
        self.assertIn("full", values)
        self.assertIn("no_red", values)
        self.assertIn("no_valid", values)
        self.assertIn("no_verifier", values)
        self.assertIn("cov_only", values)
        self.assertIn("no_verifier_filter", values)
        self.assertIn("verifier_filtered", values)
        self.assertIn("['full', 'no_red', 'no_valid', 'no_verifier', 'cov_only']", values)

    def test_generated_real_pipeline_runs_trained_reward_component_ablation_eval(self) -> None:
        pipeline = read_json("configs/pipeline_real_run.generated.json")
        manifest = read_json("configs/manifest_real_run.generated.json")
        stages = {stage["name"]: stage for stage in pipeline["stages"]}
        names = [stage["name"] for stage in pipeline["stages"]]
        required = set(manifest["required_files"])

        self.assertIn("reward_component_ablation_training_gate", stages)
        gate = stages["reward_component_ablation_training_gate"]
        self.assertEqual(gate["type"], "manual_gate")
        self.assertTrue(gate["args"]["strict"])
        self.assertIn(
            "outputs/reward_component_training_ablation/training_done.json:reward_variants=full,no_red,no_valid,no_verifier,cov_only",
            gate["args"]["required_json_contains"],
        )
        budget = stages["api_budget_reward_component_ablation_rubrics"]
        self.assertEqual(budget["type"], "api_budget")
        self.assertEqual(budget["args"]["input"], "data/processed/rubricbench_queries.jsonl")
        self.assertEqual(budget["args"]["providers"], "configs/generators_reward_ablation.local.jsonl")
        self.assertEqual(
            budget["args"]["output"],
            "outputs/api_budget/reward_component_ablation_rubrics_budget.json",
        )
        self.assertTrue(budget["args"]["strict"])
        generation = stages["generate_reward_component_ablation_rubrics"]
        self.assertEqual(generation["type"], "generate_model_rubrics")
        self.assertEqual(generation["args"]["input"], "data/processed/rubricbench_queries.jsonl")
        self.assertEqual(generation["args"]["providers"], "configs/generators_reward_ablation.local.jsonl")
        self.assertEqual(
            generation["args"]["output"],
            "data/processed/reward_component_training_ablation/model_rubrics.jsonl",
        )
        self.assertEqual(
            generation["args"]["require_budget_report"],
            "outputs/api_budget/reward_component_ablation_rubrics_budget.json",
        )
        self.assertEqual(generation["args"]["require_preflight_report"], expected_real_preflight_report_path())
        self.assertLess(names.index("reward_component_ablation_training_gate"), names.index("generate_reward_component_ablation_rubrics"))
        self.assertLess(names.index("reward_component_ablation_training_gate"), names.index("api_budget_reward_component_ablation_rubrics"))
        self.assertLess(names.index("api_budget_reward_component_ablation_rubrics"), names.index("generate_reward_component_ablation_rubrics"))
        self.assertIn("outputs/api_budget/reward_component_ablation_rubrics_budget.json", required)
        self.assertIn("outputs/api_budget/reward_component_ablation_rubrics_budget.md", required)

        for variant in ["no_red", "no_valid", "no_verifier", "cov_only"]:
            with self.subTest(variant=variant):
                prepare = stages[f"reward_ablation_{variant}_prepare_bsc"]
                bsc = stages[f"reward_ablation_{variant}_bsc"]
                self.assertEqual(prepare["type"], "prepare_bsc")
                self.assertEqual(prepare["args"]["gold"], "data/processed/splits/rubricbench_gold_test_main.jsonl")
                self.assertEqual(
                    prepare["args"]["predictions"],
                    "data/processed/reward_component_training_ablation/model_rubrics.jsonl",
                )
                self.assertEqual(prepare["args"]["model"], variant)
                self.assertEqual(
                    prepare["args"]["output"],
                    f"data/processed/reward_component_training_ablation/{variant}/bsc_eval.jsonl",
                )
                self.assertEqual(
                    prepare["args"]["report"],
                    f"outputs/reward_component_training_ablation/{variant}/bsc_join_report.json",
                )
                self.assertEqual(bsc["type"], "bsc")
                self.assertEqual(
                    bsc["args"]["input"],
                    f"data/processed/reward_component_training_ablation/{variant}/bsc_eval.jsonl",
                )
                self.assertEqual(bsc["args"]["embedding_model"], "BAAI/bge-large-en-v1.5")
                self.assertEqual(bsc["args"]["coverage_tau"], 0.75)
                self.assertEqual(bsc["args"]["redundancy_tau"], 0.85)
                self.assertEqual(
                    bsc["args"]["output_dir"],
                    f"outputs/reward_component_training_ablation/{variant}/bsc",
                )
                self.assertLess(names.index("generate_reward_component_ablation_rubrics"), names.index(prepare["name"]))
                self.assertLess(names.index(prepare["name"]), names.index(bsc["name"]))
                self.assertIn(f"data/processed/reward_component_training_ablation/{variant}/bsc_eval.jsonl", required)
                self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc_join_report.json", required)
                self.assertIn(f"outputs/reward_component_training_ablation/{variant}/bsc/summary.json", required)

    def test_evidence_matrix_real_requires_teacher_union_ablation_protocol(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C5"]))
        metrics = collect_metric_names(claims["C5"])
        value_keys = collect_value_keys(claims["C5"])

        self.assertIn("outputs/teacher_union_ablation/teacher_union_ablation.csv", paths)
        self.assertIn("outputs/teacher_union_ablation/teacher_union_ablation.json", paths)
        self.assertIn("outputs/teacher_union_ablation/teacher_union_per_item.csv", paths)
        self.assertIn("0.coverage_gain_vs_best_single", metrics)
        self.assertIn("0.n", metrics)
        self.assertIn("0.n_single_teacher_variants", metrics)
        self.assertIn("0.coverage_tau", metrics)
        self.assertIn("0.redundancy_tau", metrics)
        self.assertIn("0.dedupe_tau", metrics)
        self.assertIn("0.min_teachers", metrics)
        self.assertIn("0.variant", value_keys)
        self.assertIn("0.teachers", value_keys)
        self.assertIn("0.gold", value_keys)
        self.assertIn("0.embedding_model", value_keys)
        gold_checks = [
            item
            for item in claims["C5"]["values"]
            if item.get("key") == "0.gold"
        ]
        self.assertEqual(len(gold_checks), 1)
        self.assertEqual(
            gold_checks[0]["value"],
            "data/processed/splits/rubricbench_gold_train_seed.jsonl",
        )

    def test_evidence_matrix_real_requires_sft_only_vs_rl_ablation(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C14"]))
        metrics = collect_metric_names(claims["C14"])
        values = collect_value_keys(claims["C14"])
        value_specs = collect_value_specs(claims["C14"])
        value_comparison_keys = collect_value_comparison_keys(claims["C14"])
        file_sha256_specs = collect_file_sha256_specs(claims["C14"])
        csv_checks = claims["C14"].get("csv_checks", [])

        self.assertIn("outputs/matrix_real/trained_method_gate.json", paths)
        self.assertIn("outputs/training_commands/training_done.json", paths)
        self.assertIn("outputs/matrix_real/sft_only/bsc/summary.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc/summary.json", paths)
        self.assertIn("outputs/matrix_real/sft_only/bsc_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_real/sft_only/bsc_join_report.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_join_report.json", paths)
        self.assertIn("mean_coverage", metrics)
        self.assertIn("metrics[metric=coverage].ci_lower", metrics)
        self.assertIn("metrics[metric=coverage].ci_upper", metrics)
        self.assertIn("mean_redundancy", metrics)
        trained_gate_checks = [
            item
            for item in claims["C14"].get("values", [])
            if item.get("label") == "Matrix trained-method gate passed"
        ]
        self.assertEqual(len(trained_gate_checks), 1)
        self.assertEqual(trained_gate_checks[0].get("fail_status"), "missing")
        self.assertIn("mean_hallucination", metrics)
        self.assertIn("data_source_counts.rubricbench", metrics)
        self.assertIn("coverage_tau", metrics)
        self.assertIn("redundancy_tau", metrics)
        self.assertIn("verifier_source_counts.valid_flags", metrics)
        self.assertIn("mean_n_gen", metrics)
        self.assertIn("gen_to_gold_ratio", metrics)
        self.assertIn("coverage_per_generated_criterion", metrics)
        self.assertIn("n_missing_predictions", metrics)
        self.assertIn("n_unmatched_predictions", metrics)
        self.assertIn("gold_duplicate_record_count", metrics)
        self.assertIn("prediction_duplicate_record_count", metrics)
        self.assertIn("embedding_model", values)
        self.assertIn("verifier_source", values)
        self.assertIn("query_alignment_exact", values)
        self.assertIn("output_rows_match_n_joined", values)
        self.assertIn("output_truncated_by_limit", values)
        self.assertIn("input_sha256", values)
        self.assertIn("per_item_output", values)
        self.assertIn("per_item_sha256", values)
        self.assertIn("per_item_rows", values)
        self.assertIn(
            (
                "outputs/matrix_real/trained_method_gate.json",
                "ok",
                "True",
            ),
            value_specs,
        )
        for key, expected in [
            ("served_methods", "['base', 'sft_only', 'sft_rl']"),
            ("sft_config", "configs/llamafactory_sft.local.yaml"),
            ("grpo_config", "configs/verl_grpo_bsc.local.yaml"),
            ("sft_data", "data/processed/blindspot_sft.jsonl"),
            ("rl_data", "data/processed/proxy_gold_verl.parquet"),
            ("rl_data_report", "outputs/sft_data/proxy_gold_verl_report.json"),
            ("reward_function", "src/blindspot_rl/verl_reward.py:compute_score"),
        ]:
            with self.subTest(training_done_key=key):
                self.assertIn(
                    (
                        "outputs/training_commands/training_done.json",
                        key,
                        expected,
                    ),
                    value_specs,
                )
        for key, file_path in [
            ("sft_config_sha256", "configs/llamafactory_sft.local.yaml"),
            ("grpo_config_sha256", "configs/verl_grpo_bsc.local.yaml"),
            ("sft_data_sha256", "data/processed/blindspot_sft.jsonl"),
            ("rl_data_sha256", "data/processed/proxy_gold_verl.parquet"),
            ("rl_data_report_sha256", "outputs/sft_data/proxy_gold_verl_report.json"),
            ("reward_function_sha256", "src/blindspot_rl/verl_reward.py"),
        ]:
            with self.subTest(training_done_sha_key=key):
                self.assertIn(
                    (
                        "outputs/training_commands/training_done.json",
                        key,
                        file_path,
                    ),
                    file_sha256_specs,
                )
        self.assertTrue(
            any(
                item.get("path") == "outputs/matrix_real/main_table.csv"
                and {
                    "method",
                    "bsc_status",
                    "mean_n_gen",
                    "gen_to_gold_ratio",
                    "coverage_per_generated_criterion",
                }.issubset(
                    set(item.get("columns", []))
                )
                for item in csv_checks
            )
        )
        for method in ["sft_only", "sft_rl"]:
            with self.subTest(main_table_method=method):
                self.assertTrue(
                    any(
                        item.get("path") == "outputs/matrix_real/main_table.csv"
                        and item.get("where") == {"method": method}
                        and {
                            "mean_n_gen",
                            "gen_to_gold_ratio",
                            "coverage_per_generated_criterion",
                        }.issubset(set(item.get("non_empty", [])))
                        and {
                            "mean_n_gen",
                            "gen_to_gold_ratio",
                            "coverage_per_generated_criterion",
                        }.issubset(set(item.get("numeric", [])))
                        for item in csv_checks
                    )
                )
        for method in ["sft_only", "sft_rl"]:
            with self.subTest(method=method):
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "gold",
                        "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "output",
                        f"data/processed/matrix_real/{method}/bsc_eval.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc/summary.json",
                        "input",
                        f"data/processed/matrix_real/{method}/bsc_eval.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc/summary.json",
                        "per_item_output",
                        f"outputs/matrix_real/{method}/bsc/per_item.csv",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_ci/bootstrap_ci.json",
                        "input",
                        f"outputs/matrix_real/{method}/bsc/per_item.csv",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "query_alignment_exact",
                        "True",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "output_rows_match_n_joined",
                        "True",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "output_truncated_by_limit",
                        "False",
                    ),
                    value_specs,
                )
                self.assertIn(("input_sha256", "output_sha256"), value_comparison_keys)
                self.assertIn(("per_item_sha256", "input_sha256"), value_comparison_keys)
                self.assertIn(("per_item_rows", "n"), value_comparison_keys)

    def test_evidence_matrix_real_requires_dimension_transition_analysis(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C12"]))
        metrics = collect_metric_names(claims["C12"])
        values = collect_value_keys(claims["C12"])
        value_specs = collect_value_specs(claims["C12"])
        value_comparison_keys = collect_value_comparison_keys(claims["C12"])
        c12_text = json.dumps(claims["C12"], ensure_ascii=False)

        self.assertIn("evaluated with a dimension-transition audit", claims["C12"]["claim"])
        self.assertIn("does not by itself permit a dimension-level recovery conclusion", claims["C12"]["notes"])
        self.assertNotIn("SFT+RL repairs baseline blind spots", claims["C12"]["claim"])
        self.assertNotIn("training repairs baseline blind dimensions directly", claims["C12"]["notes"])
        self.assertNotIn("repairs a non-trivial share of baseline blind spots", c12_text)
        self.assertNotIn("repair has", c12_text)
        self.assertNotIn("feeds repair", c12_text)
        self.assertNotIn("repair baseline", c12_text)
        self.assertNotIn("repair candidate", c12_text)
        self.assertIn("outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json", paths)
        self.assertIn("outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json", paths)
        self.assertIn("outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_by_category.csv", paths)
        self.assertIn("outputs/matrix_real/base/bsc_join_report.json", paths)
        self.assertIn("outputs/matrix_real/sft_only/bsc_join_report.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_join_report.json", paths)
        self.assertIn("recovered_dimension_rate", metrics)
        self.assertNotIn("repair_rate", metrics)
        self.assertIn("loss_rate", metrics)
        self.assertIn("transition_balance", metrics)
        self.assertIn("n_matched_records", metrics)
        self.assertIn("n_unmatched_baseline_records", metrics)
        self.assertIn("n_unmatched_candidate_records", metrics)
        self.assertIn("baseline_duplicate_record_count", metrics)
        self.assertIn("candidate_duplicate_record_count", metrics)
        self.assertIn("coverage_tau", metrics)
        self.assertIn("baseline", values)
        self.assertIn("candidate", values)
        self.assertIn("embedding_model", values)
        self.assertIn("query_alignment_exact", values)
        self.assertIn("gold_rows_match_total_gold", values)
        self.assertIn("per_item_rows_match_n_matched_records", values)
        self.assertIn("net_positive_transition", values)
        self.assertIn(("baseline_sha256", "output_sha256"), value_comparison_keys)
        self.assertIn(("candidate_sha256", "output_sha256"), value_comparison_keys)
        self.assertIn("input SHA binding to the BSC join outputs", claims["C12"]["notes"])
        self.assertIn("no unmatched or duplicate records", claims["C12"]["notes"])
        for method in ["base", "sft_only", "sft_rl"]:
            with self.subTest(method=method):
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "gold",
                        "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "output",
                        f"data/processed/matrix_real/{method}/bsc_eval.jsonl",
                    ),
                    value_specs,
                )
                for key, value in [
                    ("query_alignment_exact", "True"),
                    ("n_missing_predictions", "0"),
                    ("n_unmatched_predictions", "0"),
                    ("gold_duplicate_record_count", "0"),
                    ("prediction_duplicate_record_count", "0"),
                    ("output_rows_match_n_joined", "True"),
                    ("output_truncated_by_limit", "False"),
                ]:
                    self.assertIn(
                        (
                            f"outputs/matrix_real/{method}/bsc_join_report.json",
                            key,
                            value,
                        ),
                        value_specs,
                    )
        self.assertIn("join_key", values)
        self.assertIn("respect_valid_flags", values)

    def test_evidence_matrix_real_requires_semantic_space_visualization(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C13"]))
        metrics = collect_metric_names(claims["C13"])
        values = collect_value_keys(claims["C13"])
        value_specs = collect_value_specs(claims["C13"])
        value_comparison_keys = collect_value_comparison_keys(claims["C13"])
        csv_checks = claims["C13"].get("csv_checks", [])
        file_sha256_checks = claims["C13"].get("file_sha256_checks", [])

        self.assertIn("outputs/matrix_real/semantic_space/semantic_space.svg", paths)
        self.assertIn("outputs/matrix_real/semantic_space/semantic_space.pdf", paths)
        self.assertIn("outputs/matrix_real/semantic_space/semantic_space_points.csv", paths)
        self.assertIn("outputs/matrix_real/semantic_space/semantic_space_summary.json", paths)
        self.assertIn("outputs/matrix_real/base/bsc_join_report.json", paths)
        self.assertIn("outputs/matrix_real/sft_only/bsc_join_report.json", paths)
        self.assertIn("outputs/matrix_real/sft_rl/bsc_join_report.json", paths)
        self.assertIn("n_gold", metrics)
        self.assertIn("n_generated", metrics)
        self.assertIn("n_gold_clusters", metrics)
        self.assertIn("sft_rl_vs_sft_only_generated_gold_category_coverage_delta", metrics)
        self.assertIn("sft_rl_vs_sft_only_nearest_gold_category_coverage_delta", metrics)
        self.assertIn("sft_rl_vs_sft_only_gold_cluster_coverage_delta", metrics)
        self.assertIn("sft_rl_vs_sft_only_nearest_gold_cluster_entropy_delta", metrics)
        self.assertIn("sft_rl_vs_sft_only_generated_dispersion_delta", metrics)
        self.assertIn("sft_rl_vs_sft_only_nearest_gold_similarity_delta", metrics)
        self.assertIn("requested_projection", values)
        self.assertIn("projection", values)
        self.assertIn("embedding_model", values)
        self.assertIn("gold_cluster_tau", values)
        self.assertIn("point_csv_schema_version", values)
        self.assertIn("point_csv_columns", values)
        self.assertIn("output_artifacts_schema_version", values)
        self.assertIn("point_csv", values)
        self.assertIn("point_csv_rows_match_n_points", values)
        self.assertIn("svg", values)
        self.assertIn("pdf", values)
        self.assertIn("methods", values)
        self.assertIn("generated_gold_category_coverage_by_method", values)
        self.assertIn("nearest_gold_category_coverage_by_method", values)
        self.assertIn("gold_cluster_counts", values)
        self.assertIn("nearest_gold_cluster_coverage_by_method", values)
        self.assertIn("nearest_gold_cluster_distribution_by_method", values)
        self.assertIn("nearest_gold_cluster_entropy_by_method", values)
        self.assertIn("generated_dispersion_by_method", values)
        self.assertIn(("inputs.0.sha256", "output_sha256"), value_comparison_keys)
        self.assertIn(("inputs.1.sha256", "output_sha256"), value_comparison_keys)
        self.assertIn(("inputs.2.sha256", "output_sha256"), value_comparison_keys)
        self.assertIn(("inputs.0.join_report.output_sha256", "inputs.0.sha256"), value_comparison_keys)
        self.assertIn(("inputs.1.join_report.output_sha256", "inputs.1.sha256"), value_comparison_keys)
        self.assertIn(("inputs.2.join_report.output_sha256", "inputs.2.sha256"), value_comparison_keys)
        self.assertIn(
            (
                "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                "point_csv",
                "outputs/matrix_real/semantic_space/semantic_space_points.csv",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                "svg",
                "outputs/matrix_real/semantic_space/semantic_space.svg",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                "pdf",
                "outputs/matrix_real/semantic_space/semantic_space.pdf",
            ),
            value_specs,
        )
        projection_gate = next(item for item in claims["C13"]["values"] if item.get("key") == "projection")
        self.assertEqual(projection_gate["op"], "in")
        self.assertEqual(projection_gate["value"], ["umap", "umap_fallback_pca"])
        for index, method in enumerate(["base", "sft_only", "sft_rl"]):
            with self.subTest(method=method):
                self.assertIn(
                    (
                        "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                        f"inputs.{index}.path",
                        f"data/processed/matrix_real/{method}/bsc_eval.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                        "gold",
                        "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                        f"inputs.{index}.join_report.path",
                        f"outputs/matrix_real/{method}/bsc_join_report.json",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                        f"inputs.{index}.join_report.gold",
                        "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    ),
                    value_specs,
                )
                self.assertIn(
                    (
                        "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                        f"inputs.{index}.join_report.join_key",
                        "query",
                    ),
                    value_specs,
                )
                for key in [
                    "unmatched_gold",
                    "unmatched_prediction",
                    "duplicate_gold_keys",
                    "duplicate_prediction_keys",
                ]:
                    self.assertIn(
                        (
                            "outputs/matrix_real/semantic_space/semantic_space_summary.json",
                            f"inputs.{index}.join_report.{key}",
                            "0",
                        ),
                        value_specs,
                    )
                self.assertIn(
                    (f"inputs.{index}.join_report.output", f"inputs.{index}.path"),
                    value_comparison_keys,
                )
                self.assertIn(
                    (f"inputs.{index}.n_records", f"inputs.{index}.join_report.n_joined"),
                    value_comparison_keys,
                )
        self.assertGreaterEqual(len(csv_checks), 4)
        schema_checks = [item for item in csv_checks if item.get("column_mode") == "exact"]
        self.assertTrue(schema_checks)
        self.assertEqual(schema_checks[0]["columns"], POINT_CSV_COLUMNS)
        generated_checks = [
            item
            for item in csv_checks
            if item.get("where") == {"source_type": "generated"}
        ]
        self.assertTrue(generated_checks)
        self.assertIn("nearest_gold_point_id", generated_checks[0]["non_empty"])
        self.assertIn("nearest_gold_category", generated_checks[0]["non_empty"])
        self.assertIn("nearest_gold_cluster_id", generated_checks[0]["non_empty"])
        self.assertIn("nearest_gold_similarity", generated_checks[0]["numeric"])
        sft_rl_checks = [
            item
            for item in csv_checks
            if item.get("where") == {"source_type": "generated", "method": "sft_rl"}
        ]
        self.assertTrue(sft_rl_checks)
        self.assertGreaterEqual(sft_rl_checks[0]["min_rows"], 1)
        sha_keys = {item["json_key"] for item in file_sha256_checks}
        self.assertIn("point_csv_sha256", sha_keys)
        self.assertIn("svg_sha256", sha_keys)
        self.assertIn("pdf_sha256", sha_keys)

    def test_methods_matrix_real_requests_umap_semantic_space(self) -> None:
        config = read_json("configs/methods_matrix_real.template.json")
        semantic_space = config["common"]["semantic_space"]

        self.assertEqual(semantic_space["projection"], "umap")
        self.assertEqual(semantic_space["gold_cluster_tau"], 0.75)
        self.assertEqual(semantic_space["methods"], ["base", "sft_only", "sft_rl"])

    def test_methods_matrix_real_export_binds_full_evidence_matrix(self) -> None:
        config = read_json("configs/methods_matrix_real.template.json")
        common = config["common"]

        self.assertEqual(common["evidence_json"], "outputs/evidence_real/evidence_matrix.json")
        self.assertEqual(common["evidence_csv"], "outputs/evidence_real/evidence_matrix.csv")
        self.assertEqual(common["evidence_md"], "outputs/evidence_real/evidence_matrix.md")

        generated = read_json("configs/pipeline_matrix_real.generated.json")
        export = next(stage for stage in generated["stages"] if stage["name"] == "export")
        self.assertEqual(export["args"]["evidence_json"], "outputs/evidence_real/evidence_matrix.json")
        self.assertEqual(export["args"]["evidence_csv"], "outputs/evidence_real/evidence_matrix.csv")
        self.assertEqual(export["args"]["evidence_md"], "outputs/evidence_real/evidence_matrix.md")

    def test_generated_real_pipelines_request_umap_semantic_space(self) -> None:
        for path in ["configs/pipeline_matrix_real.generated.json", "configs/pipeline_real_run.generated.json"]:
            with self.subTest(path=path):
                config = read_json(path)
                stage = next(item for item in config["stages"] if item["name"] == "semantic_space")
                args = stage["args"]

                self.assertEqual(stage["type"], "semantic_space")
                self.assertEqual(args["projection"], "umap")
                self.assertEqual(args["gold_cluster_tau"], 0.75)
                self.assertEqual(args["embedding_model"], "BAAI/bge-large-en-v1.5")
                self.assertEqual(
                    args["input"],
                    [
                        "base=data/processed/matrix_real/base/bsc_eval.jsonl",
                        "sft_only=data/processed/matrix_real/sft_only/bsc_eval.jsonl",
                        "sft_rl=data/processed/matrix_real/sft_rl/bsc_eval.jsonl",
                    ],
                )

    def test_generated_real_pipelines_include_sft_only_and_sft_rl_dimension_transitions(self) -> None:
        expected = {
            "dimension_transition_base_to_sft_only": {
                "candidate": "data/processed/matrix_real/sft_only/bsc_eval.jsonl",
                "output_dir": "outputs/matrix_real/dimension_transition/base_to_sft_only",
                "candidate_label": "sft_only",
            },
            "dimension_transition_base_to_sft_rl": {
                "candidate": "data/processed/matrix_real/sft_rl/bsc_eval.jsonl",
                "output_dir": "outputs/matrix_real/dimension_transition/base_to_sft_rl",
                "candidate_label": "sft_rl",
            },
        }
        for path in ["configs/pipeline_matrix_real.generated.json", "configs/pipeline_real_run.generated.json"]:
            with self.subTest(path=path):
                config = read_json(path)
                stages = {item["name"]: item for item in config["stages"]}
                for name, fields in expected.items():
                    stage = stages[name]
                    args = stage["args"]

                    self.assertEqual(stage["type"], "dimension_transition")
                    self.assertEqual(args["baseline"], "data/processed/matrix_real/base/bsc_eval.jsonl")
                    self.assertEqual(args["baseline_label"], "base")
                    self.assertEqual(args["embedding_model"], "BAAI/bge-large-en-v1.5")
                    self.assertEqual(args["coverage_tau"], 0.75)
                    self.assertEqual(args["join_key"], "query")
                    for key, value in fields.items():
                        self.assertEqual(args[key], value)

                export_stages = [
                    item
                    for item in config["stages"]
                    if item["type"] == "export" and "transition_summary_json" in item.get("args", {})
                ]
                self.assertTrue(export_stages)
                for export_stage in export_stages:
                    self.assertEqual(
                        export_stage["args"]["transition_summary_json"],
                        [
                            "outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json",
                            "outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json",
                        ],
                    )

    def test_methods_matrix_real_requests_endpoint_bsc_human_audit_packs(self) -> None:
        config = read_json("configs/methods_matrix_real.template.json")
        audit_pack = config["common"]["bsc_human_audit_pack"]

        self.assertTrue(audit_pack["enabled"])
        self.assertEqual(audit_pack["methods"], ["base", "sft_rl"])
        self.assertEqual(audit_pack["matched"], 25)
        self.assertEqual(audit_pack["unmatched"], 25)
        self.assertEqual(audit_pack["seed"], 13)

    def test_paper_facing_bsc_matrices_use_test_main_holdout(self) -> None:
        for relative_path in [
            "configs/methods_matrix_real.template.json",
            "configs/methods_matrix_judgebench.template.json",
            "configs/methods_matrix_rewardbench2.template.json",
        ]:
            with self.subTest(config=relative_path):
                config = read_json(relative_path)
                self.assertEqual(
                    config["common"]["bsc_gold"],
                    "data/processed/splits/rubricbench_gold_test_main.jsonl",
                )

        real = read_json("configs/methods_matrix_real.template.json")
        self.assertEqual(
            real["common"]["teacher_union_ablation"]["gold"],
            "data/processed/splits/rubricbench_gold_train_seed.jsonl",
        )
        self.assertEqual(
            real["common"]["verifier_filter_ablation"]["gold"],
            "data/processed/splits/rubricbench_gold_train_seed.jsonl",
        )
        for relative_path in [
            "configs/pipeline_matrix_real.generated.json",
            "configs/pipeline_matrix_judgebench.generated.json",
            "configs/pipeline_matrix_rewardbench2.generated.json",
        ]:
            with self.subTest(generated_pipeline=relative_path):
                pipeline = read_json(relative_path)
                prepare_bsc = [stage for stage in pipeline["stages"] if stage["type"] == "prepare_bsc"]
                self.assertGreater(len(prepare_bsc), 0)
                self.assertEqual(
                    {stage["args"]["gold"] for stage in prepare_bsc},
                    {"data/processed/splits/rubricbench_gold_test_main.jsonl"},
                )
        generated_real = read_json("configs/pipeline_matrix_real.generated.json")
        teacher_union = next(stage for stage in generated_real["stages"] if stage["type"] == "teacher_union_ablation")
        verifier_filter = next(stage for stage in generated_real["stages"] if stage["type"] == "verifier_filter_ablation")
        self.assertEqual(teacher_union["args"]["gold"], "data/processed/splits/rubricbench_gold_train_seed.jsonl")
        self.assertEqual(verifier_filter["args"]["gold"], "data/processed/splits/rubricbench_gold_train_seed.jsonl")

    def test_evidence_matrix_real_requires_cross_domain_generalization(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C8"]))
        metrics = collect_metric_names(claims["C8"])

        self.assertIn("outputs/generalization_matrix/main_table.csv", paths)
        self.assertIn("outputs/generalization_matrix/healthbench_base/bsc/summary.json", paths)
        self.assertIn("outputs/generalization_matrix/healthbench_sft_rl/bsc/summary.json", paths)
        self.assertIn("outputs/generalization_matrix/writingbench_base/bsc/summary.json", paths)
        self.assertIn("outputs/generalization_matrix/writingbench_sft_rl/bsc/summary.json", paths)
        self.assertIn("mean_coverage", metrics)

    def test_evidence_matrix_real_requires_judgebench_downstream(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C9"]))
        metrics = collect_metric_names(claims["C9"])
        metric_specs = collect_metric_specs(claims["C9"])
        values = collect_value_keys(claims["C9"])
        value_specs = collect_value_specs(claims["C9"])
        value_comparison_keys = collect_value_comparison_keys(claims["C9"])
        table_value_specs = collect_table_value_specs(claims["C9"])

        self.assertIn("outputs/matrix_judgebench/main_table.csv", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_base/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_sft_rl/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_base/downstream_api_budget/budget.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_sft_rl/downstream_api_budget/budget.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_base/downstream_join_report.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_sft_rl/downstream_join_report.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_base/downstream/summary.json", paths)
        self.assertIn("outputs/matrix_judgebench/judgebench_sft_rl/downstream/summary.json", paths)
        self.assertIn("outputs/contamination_audit/judgebench_downstream_holdout_contamination.json", paths)
        self.assertIn("metrics[metric=correct].ci_lower", metrics)
        self.assertIn("metrics[metric=correct].ci_upper", metrics)
        self.assertIn("n", metrics)
        self.assertIn("overlap_query_count", metrics)
        self.assertIn("holdout_unique_queries", metrics)
        for path in [
            "outputs/matrix_judgebench/judgebench_base/downstream_join_report.json",
            "outputs/matrix_judgebench/judgebench_sft_rl/downstream_join_report.json",
        ]:
            with self.subTest(path=path):
                self.assertIn((path, "n_missing_rubrics", "0"), metric_specs)
                self.assertIn((path, "n_unmatched_rubrics", "0"), metric_specs)
                self.assertIn((path, "source_duplicate_record_count", "0"), metric_specs)
                self.assertIn((path, "rubric_duplicate_record_count", "0"), metric_specs)
        self.assertIn("scorer", values)
        self.assertIn("paper_claim_eligible", values)
        self.assertIn("paper_claim_eligibility_blockers", values)
        self.assertIn("scorer_provider", values)
        self.assertIn("budget_report", values)
        self.assertIn("input", values)
        self.assertIn("input_sha256", values)
        self.assertIn("per_item_output", values)
        self.assertIn("per_item_sha256", values)
        self.assertIn("preferences", values)
        self.assertIn("rubrics", values)
        self.assertIn("output", values)
        self.assertIn("output_sha256", values)
        self.assertIn("data_source", values)
        self.assertIn("model", values)
        self.assertIn("query_alignment_exact", values)
        self.assertIn("output_rows_match_n_joined", values)
        self.assertIn("output_truncated_by_limit", values)
        self.assertIn("scorer_provider_sha256", values)
        self.assertIn("benchmark_format", values)
        self.assertIn("scorer_contract.calls_per_record_per_provider", values)
        self.assertIn("scorer_contract.unit_field", values)
        self.assertIn("ok", values)
        self.assertIn("contract.input", values)
        self.assertIn("contract.providers", values)
        self.assertIn("contract.calls_per_record_per_provider", values)
        self.assertIn("contract.unit_field", values)
        self.assertIn(
            ("outputs/matrix_judgebench/main_table.csv", "judgebench_base", "downstream_status", "pass"),
            table_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_judgebench/main_table.csv",
                "judgebench_base",
                "downstream_paper_claim_eligible",
                "true",
            ),
            table_value_specs,
        )
        self.assertIn(
            ("outputs/matrix_judgebench/main_table.csv", "judgebench_sft_rl", "downstream_status", "pass"),
            table_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_judgebench/main_table.csv",
                "judgebench_sft_rl",
                "downstream_paper_claim_eligible",
                "true",
            ),
            table_value_specs,
        )
        self.assertIn(("input_sha256", "contract.input_sha256"), value_comparison_keys)
        self.assertIn(("input_sha256", "output_sha256"), value_comparison_keys)
        self.assertIn(("scorer_provider_sha256", "contract.providers_sha256"), value_comparison_keys)
        self.assertIn(("per_item_sha256", "input_sha256"), value_comparison_keys)
        self.assertIn(("per_item_rows", "n"), value_comparison_keys)
        self.assertIn(
            (
                "outputs/matrix_judgebench/judgebench_base/downstream_join_report.json",
                "preferences",
                "data/processed/judgebench_pref.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_judgebench/judgebench_sft_rl/downstream_join_report.json",
                "output",
                "data/processed/matrix_judgebench/judgebench_sft_rl/downstream_eval.jsonl",
            ),
            value_specs,
        )
        for path in [
            "outputs/matrix_judgebench/judgebench_base/downstream_join_report.json",
            "outputs/matrix_judgebench/judgebench_sft_rl/downstream_join_report.json",
        ]:
            with self.subTest(path=path):
                self.assertIn((path, "query_alignment_exact", "True"), value_specs)
                self.assertIn((path, "output_rows_match_n_joined", "True"), value_specs)
                self.assertIn((path, "output_truncated_by_limit", "False"), value_specs)

    def test_evidence_matrix_real_requires_rewardbench2_multicandidate_downstream(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C10"]))
        metrics = collect_metric_names(claims["C10"])
        metric_specs = collect_metric_specs(claims["C10"])
        values = collect_value_keys(claims["C10"])
        value_specs = collect_value_specs(claims["C10"])
        value_comparison_keys = collect_value_comparison_keys(claims["C10"])
        table_value_specs = collect_table_value_specs(claims["C10"])

        self.assertIn("outputs/matrix_rewardbench2/main_table.csv", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_base/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_ci/bootstrap_ci.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_base/downstream_api_budget/budget.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_api_budget/budget.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_base/downstream_join_report.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_join_report.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_base/downstream/summary.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream/summary.json", paths)
        self.assertIn("outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json", paths)
        self.assertIn("metrics[metric=correct].ci_lower", metrics)
        self.assertIn("metrics[metric=correct].ci_upper", metrics)
        self.assertIn("n", metrics)
        self.assertIn("overlap_query_count", metrics)
        self.assertIn("holdout_unique_queries", metrics)
        self.assertIn("holdout_raw_unique_queries", metrics)
        for path in [
            "outputs/matrix_rewardbench2/rewardbench2_base/downstream_join_report.json",
            "outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_join_report.json",
        ]:
            with self.subTest(path=path):
                self.assertIn((path, "n_missing_rubrics", "0"), metric_specs)
                self.assertIn((path, "n_unmatched_rubrics", "0"), metric_specs)
                self.assertIn((path, "source_duplicate_record_count", "0"), metric_specs)
                self.assertIn((path, "rubric_duplicate_record_count", "0"), metric_specs)
        self.assertIn("scorer", values)
        self.assertIn("paper_claim_eligible", values)
        self.assertIn("paper_claim_eligibility_blockers", values)
        self.assertIn("scorer_provider", values)
        self.assertIn("budget_report", values)
        self.assertIn("input", values)
        self.assertIn("input_sha256", values)
        self.assertIn("per_item_output", values)
        self.assertIn("per_item_sha256", values)
        self.assertIn("benchmark", values)
        self.assertIn("rubrics", values)
        self.assertIn("output", values)
        self.assertIn("output_sha256", values)
        self.assertIn("data_source", values)
        self.assertIn("model", values)
        self.assertIn("query_alignment_exact", values)
        self.assertIn("output_rows_match_n_joined", values)
        self.assertIn("output_truncated_by_limit", values)
        self.assertIn("scorer_provider_sha256", values)
        self.assertIn("benchmark_format", values)
        self.assertIn("scorer_contract.calls_per_record_per_provider", values)
        self.assertIn("scorer_contract.unit_field", values)
        self.assertIn("scorer_contract.unit_multiplier_field", values)
        self.assertIn("ok", values)
        self.assertIn(
            ("outputs/matrix_rewardbench2/main_table.csv", "rewardbench2_base", "downstream_status", "pass"),
            table_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_rewardbench2/main_table.csv",
                "rewardbench2_base",
                "downstream_paper_claim_eligible",
                "true",
            ),
            table_value_specs,
        )
        self.assertIn(
            ("outputs/matrix_rewardbench2/main_table.csv", "rewardbench2_sft_rl", "downstream_status", "pass"),
            table_value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_rewardbench2/main_table.csv",
                "rewardbench2_sft_rl",
                "downstream_paper_claim_eligible",
                "true",
            ),
            table_value_specs,
        )
        self.assertIn("contract.input", values)
        self.assertIn("contract.providers", values)
        self.assertIn("contract.calls_per_record_per_provider", values)
        self.assertIn("contract.unit_field", values)
        self.assertIn("contract.unit_multiplier_field", values)
        self.assertIn(("input_sha256", "contract.input_sha256"), value_comparison_keys)
        self.assertIn(("input_sha256", "output_sha256"), value_comparison_keys)
        self.assertIn(("scorer_provider_sha256", "contract.providers_sha256"), value_comparison_keys)
        self.assertIn(("per_item_sha256", "input_sha256"), value_comparison_keys)
        self.assertIn(("per_item_rows", "n"), value_comparison_keys)
        self.assertIn(
            (
                "outputs/matrix_rewardbench2/rewardbench2_base/downstream_join_report.json",
                "benchmark",
                "data/processed/rewardbench2_multicandidate.clean.jsonl",
            ),
            value_specs,
        )
        self.assertIn(
            (
                "outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_join_report.json",
                "output",
                "data/processed/matrix_rewardbench2/rewardbench2_sft_rl/downstream_eval.jsonl",
            ),
            value_specs,
        )
        for path in [
            "outputs/matrix_rewardbench2/rewardbench2_base/downstream_join_report.json",
            "outputs/matrix_rewardbench2/rewardbench2_sft_rl/downstream_join_report.json",
        ]:
            with self.subTest(path=path):
                self.assertIn((path, "query_alignment_exact", "True"), value_specs)
                self.assertIn((path, "output_rows_match_n_joined", "True"), value_specs)
                self.assertIn((path, "output_truncated_by_limit", "False"), value_specs)

    def test_evidence_matrix_real_requires_downstream_policy_rlvr_gate(self) -> None:
        config = read_json("configs/evidence_matrix_real.template.json")
        claims = {item["id"]: item for item in config["claims"]}
        paths = list(collect_paths(claims["C11"]))
        metrics = collect_metric_names(claims["C11"])

        self.assertIn("outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json", paths)
        self.assertIn("outputs/policy_rlvr/downstream_rlvr_completion_gate.json", paths)
        self.assertIn("outputs/policy_rlvr/healthbench_hard_policy", paths)
        self.assertIn("outputs/policy_rlvr/healthbench_hard_eval.json", paths)
        self.assertIn("outputs/policy_rlvr/arenahard_policy", paths)
        self.assertIn("outputs/policy_rlvr/arenahard_eval.json", paths)
        self.assertIn("ok", metrics)

    def test_data_sources_real_contains_cross_domain_query_pools(self) -> None:
        config = read_json("configs/data_sources_real.template.json")
        datasets = {item["name"]: item for item in config["datasets"]}

        self.assertIn("healthbench", datasets)
        self.assertIn("beir_nq", datasets)
        self.assertIn("ifbench", datasets)
        self.assertIn("writingbench", datasets)
        self.assertIn("healthbench_hard", datasets)
        self.assertIn("arenahard", datasets)
        self.assertIn("rewardbench2", datasets)
        self.assertIn("rmbench", datasets)
        self.assertEqual(datasets["healthbench"]["normalizations"][0]["target"], "query_pool")
        self.assertEqual(datasets["healthbench"]["source"]["preset"], "healthbench")
        self.assertEqual(datasets["healthbench"]["normalizations"][0]["query_key"], "prompt")
        self.assertEqual(datasets["beir_nq"]["source"]["preset"], "beir_nq")
        self.assertEqual(datasets["beir_nq"]["normalizations"][0]["target"], "query_pool")
        self.assertEqual(datasets["beir_nq"]["normalizations"][0]["query_key"], "text")
        self.assertEqual(datasets["ifbench"]["source"]["preset"], "ifbench")
        self.assertEqual(datasets["ifbench"]["normalizations"][0]["target"], "query_pool")
        self.assertEqual(datasets["ifbench"]["normalizations"][0]["query_key"], "prompt")
        self.assertEqual(datasets["writingbench"]["normalizations"][0]["target"], "query_pool")
        self.assertEqual(datasets["writingbench"]["source"]["preset"], "writingbench")
        self.assertEqual(datasets["healthbench_hard"]["normalizations"][0]["output"], "data/processed/healthbench_hard_queries.jsonl")
        self.assertEqual(datasets["arenahard"]["normalizations"][0]["output"], "data/processed/arenahard_queries.jsonl")
        self.assertEqual(datasets["rewardbench2"]["source"]["preset"], "rewardbench2")
        self.assertEqual(datasets["rewardbench2"]["normalizations"][0]["target"], "query_pool")
        self.assertEqual(datasets["rewardbench2"]["normalizations"][1]["target"], "multicandidate")
        self.assertEqual(datasets["rmbench"]["source"]["official_url"], "https://github.com/THU-KEG/RM-Bench")
        self.assertEqual(datasets["rmbench"]["normalizations"][0]["target"], "query_pool")
        self.assertIn("proxy gold", datasets["healthbench"]["source"]["note"].lower())
        self.assertIn("search-intent", datasets["beir_nq"]["source"]["note"].lower())
        self.assertIn("proxy gold", datasets["writingbench"]["source"]["note"].lower())

    def test_hard_gold_sources_require_official_url_and_provenance(self) -> None:
        data_sources = read_json("configs/data_sources_real.template.json")
        datasets = {item["name"]: item for item in data_sources["datasets"]}
        generated_data_pipeline = read_json("configs/pipeline_data_real.generated.json")
        generated_data_manifest = read_json("configs/manifest_data_real.generated.json")
        stages = {stage["name"]: stage for stage in generated_data_pipeline["stages"]}

        self.assertEqual(generated_data_pipeline["stages"][0]["type"], "init_data_source_config")
        self.assertEqual(
            generated_data_pipeline["stages"][0]["args"]["report_json"],
            "outputs/data_sources/local_config_init.json",
        )
        self.assertIn("outputs/data_sources/local_config_init.json", generated_data_manifest["required_files"])
        self.assertIn("outputs/data_sources/local_config_post_download.json", generated_data_manifest["required_files"])
        self.assertIn("outputs/data_sources/source_report_pre_normalization.json", generated_data_manifest["required_files"])
        self.assertIn("outputs/data_sources/source_report.json", generated_data_manifest["required_files"])
        self.assertEqual(data_sources["source_report"]["config"], "configs/data_sources_real.local.json")
        self.assertEqual(
            stages["data_source_report"]["args"]["config"],
            "configs/data_sources_real.local.json",
        )
        stage_names = [stage["name"] for stage in generated_data_pipeline["stages"]]
        post_download_idx = stage_names.index("init_data_source_local_config_post_download")
        pre_source_report_idx = stage_names.index("data_source_report_pre_normalization")
        source_report_idx = stage_names.index("data_source_report")
        self.assertLess(stage_names.index("download_researchrubrics"), post_download_idx)
        self.assertLess(post_download_idx, pre_source_report_idx)
        self.assertLess(pre_source_report_idx, stage_names.index("profile_researchrubrics"))
        self.assertLess(stage_names.index("validate_researchrubrics_query_pool_2"), source_report_idx)

        for dataset_name, paper_url in {
            "rubricbench": "https://arxiv.org/abs/2603.01562",
            "researchrubrics": "https://arxiv.org/abs/2511.07685",
        }.items():
            dataset = datasets[dataset_name]
            gold_norm = next(item for item in dataset["normalizations"] if item["target"] == "gold")
            validation = gold_norm["validation"]

            self.assertTrue(dataset["source"]["require_official_url"])
            self.assertTrue(dataset["source"]["require_raw_sha256"])
            self.assertEqual(dataset["source"]["raw_sha256"], "")
            self.assertEqual(gold_norm["paper_url"], paper_url)
            self.assertTrue(validation["require_provenance"])
            self.assertEqual(validation["required_data_source"], [dataset_name])
            self.assertTrue(validation["strict"])

            normalize_stage = stages[f"normalize_{dataset_name}_gold_1"]["args"]
            validate_stage = stages[f"validate_{dataset_name}_gold_1"]["args"]
            self.assertEqual(normalize_stage["paper_url"], paper_url)
            self.assertTrue(validate_stage["require_provenance"])
            self.assertEqual(validate_stage["required_data_source"], [dataset_name])
            self.assertIn(f"paper_url={paper_url}", validate_stage["required_provenance"])
            self.assertTrue(validate_stage["strict"])

        self.assertEqual(
            datasets["researchrubrics"]["source"]["official_url"],
            "https://huggingface.co/datasets/ScaleAI/researchrubrics/resolve/main/processed_data.jsonl",
        )
        self.assertEqual(
            datasets["rubricbench"]["source"]["official_url"],
            "https://huggingface.co/datasets/DonJoey/rubricbench/resolve/main/data/train-00000-of-00001.parquet",
        )
        self.assertTrue(datasets["researchrubrics"]["source"]["download_enabled"])
        self.assertTrue(datasets["rubricbench"]["source"]["download_enabled"])
        self.assertIn("download_rubricbench", stage_names)
        self.assertEqual(datasets["researchrubrics"]["source"]["raw_sha256"], "")

    def test_researchrubrics_scoped_data_pipeline_excludes_rubricbench_blocker(self) -> None:
        pipeline = read_json("configs/pipeline_data_researchrubrics.generated.json")
        manifest = read_json("configs/manifest_data_researchrubrics.generated.json")
        names = [stage["name"] for stage in pipeline["stages"]]

        self.assertIn("download_researchrubrics", names)
        self.assertIn("validate_researchrubrics_gold_1", names)
        self.assertNotIn("profile_rubricbench", names)
        self.assertNotIn("validate_rubricbench_gold_1", names)
        source_report = next(stage for stage in pipeline["stages"] if stage["name"] == "data_source_report")
        pre_source_report = next(
            stage for stage in pipeline["stages"] if stage["name"] == "data_source_report_pre_normalization"
        )
        self.assertEqual(pre_source_report["args"]["required_dataset"], ["researchrubrics"])
        self.assertEqual(source_report["args"]["required_dataset"], ["researchrubrics"])
        self.assertEqual(
            pre_source_report["args"]["output_json"],
            "outputs/data_sources/source_report_pre_normalization_researchrubrics.json",
        )
        self.assertEqual(source_report["args"]["output_json"], "outputs/data_sources/source_report_researchrubrics.json")
        self.assertIn("data/raw/researchrubrics_raw.jsonl", manifest["required_files"])
        self.assertIn("outputs/data_sources/source_report_researchrubrics.json", manifest["required_files"])
        self.assertNotIn("data/raw/rubricbench_raw.jsonl", manifest["required_files"])
        self.assertNotIn("outputs/data_sources/source_report.json", manifest["required_files"])

    def test_sprint_plan_initializes_local_data_source_config_without_overwrite(self) -> None:
        plan = read_json("configs/sprint_plan_20day.template.json")
        commands = [
            command
            for phase in plan["phases"]
            for task in phase.get("tasks", [])
            for command in task.get("commands", [])
        ]

        init_commands = [
            command
            for command in commands
            if "configs/data_sources_real.template.json" in command
            and "configs/data_sources_real.local.json" in command
        ]
        self.assertTrue(any(command.startswith("python3 scripts/init_data_source_local_config.py ") for command in init_commands))
        self.assertTrue(any("outputs/data_sources/local_config_init.json" in command for command in init_commands))
        self.assertNotIn("cp configs/data_sources_real.template.json configs/data_sources_real.local.json", commands)

    def test_sprint_plan_checks_paper_asset_index_after_sync(self) -> None:
        plan = read_json("configs/sprint_plan_20day.template.json")
        paper_phase = next(phase for phase in plan["phases"] if phase["name"] == "Paper And Submission")
        sync_task = next(task for task in paper_phase["tasks"] if task["goal"] == "Sync artifacts into the manuscript scaffold.")

        self.assertIn(
            "python3 scripts/check_paper_asset_index.py --asset-index paper/asset_index.md --output outputs/paper_artifacts/paper_asset_index_check.json --output-md outputs/paper_artifacts/paper_asset_index_check.md --strict",
            sync_task["commands"],
        )
        self.assertIn("outputs/paper_artifacts/paper_asset_index_check.json", sync_task["artifacts"])

    def test_sprint_plan_exposes_minimal_blocked_report_refresh(self) -> None:
        plan = read_json("configs/sprint_plan_20day.template.json")
        minimal_phase = next(phase for phase in plan["phases"] if phase["name"] == "Minimal Motivation")
        bsc_task = next(
            task
            for task in minimal_phase["tasks"]
            if task["goal"] == "Run base evaluation-criteria elicitation and BSC diagnosis."
        )

        self.assertIn(
            "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_minimal_claim.generated.json --from-stage audit --to-stage paper_asset_index_check_post_sync",
            bsc_task["commands"],
        )

    def test_preflight_real_checks_all_generation_query_pools(self) -> None:
        config = read_json("configs/preflight_real.template.json")
        args = config["stages"][0]["args"]
        inputs = set(args["input"])
        providers = set(args["providers"])
        required_providers = set(args["required_provider"])
        required_provider_in = set(args["required_provider_in"])
        required_env = set(args["required_env"])
        example_env = {
            item["api_key_env"]
            for path in [
                "configs/generators.example.jsonl",
                "configs/providers.example.jsonl",
                "configs/verifier.example.jsonl",
                "configs/judge_scorer.example.jsonl",
            ]
            for item in read_jsonl(path)
        }

        self.assertTrue(args["strict"])
        self.assertLessEqual(example_env, required_env)
        self.assertIn("data/processed/rubricbench_gold.jsonl", inputs)
        self.assertIn("data/processed/rubricbench_queries.jsonl", inputs)
        self.assertIn("data/processed/splits/rubricbench_gold_train_seed.jsonl", inputs)
        self.assertIn("data/processed/rewardbench_pref.jsonl", inputs)
        self.assertIn("data/processed/rewardbench_queries.jsonl", inputs)
        self.assertIn("data/processed/judgebench_pref.jsonl", inputs)
        self.assertIn("data/processed/judgebench_queries.jsonl", inputs)
        self.assertIn("data/processed/rewardbench2_queries.clean.jsonl", inputs)
        self.assertIn("data/processed/rewardbench2_multicandidate.clean.jsonl", inputs)
        self.assertNotIn("data/processed/rewardbench2_queries.jsonl", inputs)
        self.assertNotIn("data/processed/rewardbench2_multicandidate.jsonl", inputs)
        self.assertIn("data/processed/rmbench_queries.jsonl", inputs)
        self.assertIn("data/processed/healthbench_queries.jsonl", inputs)
        self.assertIn("data/processed/writingbench_queries.jsonl", inputs)
        self.assertIn("data/processed/healthbench_hard_queries.jsonl", inputs)
        self.assertIn("data/processed/arenahard_queries.jsonl", inputs)
        self.assertEqual(
            providers,
            {
                "configs/generators.local.jsonl",
                "configs/providers.local.jsonl",
                "configs/verifier.local.jsonl",
                "configs/judge_scorer.local.jsonl",
            },
        )
        self.assertEqual(
            required_providers,
            {
                "base",
                "gpt4o",
                "claude",
                "sft_only",
                "sft_rl",
                "gpt-4o",
                "deepseek",
                "qwen",
                "meta-verifier",
                "judge-scorer",
            },
        )
        self.assertIn("configs/providers.local.jsonl:gpt-4o,deepseek,qwen", required_provider_in)
        self.assertIn("configs/verifier.local.jsonl:meta-verifier", required_provider_in)
        self.assertIn("configs/judge_scorer.local.jsonl:judge-scorer", required_provider_in)

    def test_real_run_training_config_paths_are_committed_and_aligned(self) -> None:
        assembly = read_json("configs/real_run_assembly.template.json")
        training_commands = read_json("configs/training_commands.example.json")
        local_training_commands = read_json("configs/training_commands.local.json")
        reward_ablation_generators = read_jsonl("configs/generators_reward_ablation.local.jsonl")
        reward_ablation_generator_example = read_jsonl("configs/generators_reward_ablation.example.jsonl")
        sft_manifest = read_json("configs/manifest_sft_data_real.template.json")
        verl_template = (ROOT / "configs/verl_grpo_bsc.example.yaml").read_text(encoding="utf-8")
        local_verl_template = (ROOT / "configs/verl_grpo_bsc.local.yaml").read_text(encoding="utf-8")

        training_config = ROOT / assembly["training_commands"]["config"]
        self.assertTrue(training_config.exists(), assembly["training_commands"]["config"])
        self.assertIn("data/processed/proxy_gold_verl.parquet", sft_manifest["required_files"])
        self.assertIn("outputs/sft_data/proxy_gold_build_report.json", sft_manifest["required_files"])
        self.assertIn("outputs/sft_data/proxy_gold_verl_report.json", sft_manifest["required_files"])
        self.assertIn("data/processed/proxy_gold_verl.parquet", verl_template)
        self.assertIn("data/processed/proxy_gold_verl.parquet", local_verl_template)
        self.assertIn("BSC_VERIFIER=rule", verl_template)
        self.assertIn("BSC_VERIFIER=rule", local_verl_template)
        self.assertIn("BSC_W_VALID=0.0 BSC_VERIFIER=none", verl_template)
        self.assertIn("BSC_W_VALID=0.0 BSC_VERIFIER=none", local_verl_template)
        self.assertEqual(training_commands["sft"]["sft_data"], "data/processed/blindspot_sft.jsonl")
        self.assertEqual(training_commands["sft"]["output_dir"], "outputs/checkpoints/evaluation_criteria_policy_sft")
        self.assertEqual(training_commands["grpo"]["rl_data"], "data/processed/proxy_gold_verl.parquet")
        self.assertEqual(training_commands["grpo"]["output_dir"], "outputs/checkpoints/evaluation_criteria_policy_rl")
        self.assertEqual(training_commands["grpo"]["rl_data_report"], "outputs/sft_data/proxy_gold_verl_report.json")
        self.assertEqual(local_training_commands["sft"]["sft_data"], "data/processed/blindspot_sft.jsonl")
        self.assertEqual(
            local_training_commands["sft"]["output_dir"],
            "outputs/checkpoints/evaluation_criteria_policy_sft",
        )
        self.assertEqual(local_training_commands["grpo"]["rl_data"], "data/processed/proxy_gold_verl.parquet")
        self.assertEqual(
            local_training_commands["grpo"]["output_dir"],
            "outputs/checkpoints/evaluation_criteria_policy_rl",
        )
        self.assertEqual(
            local_training_commands["grpo"]["rl_data_report"],
            "outputs/sft_data/proxy_gold_verl_report.json",
        )
        self.assertEqual(training_commands["grpo"]["reward_function"], "src/blindspot_rl/verl_reward.py:compute_score")
        self.assertEqual(
            local_training_commands["grpo"]["reward_function"],
            "src/blindspot_rl/verl_reward.py:compute_score",
        )
        for config_name, config in [
            ("example", training_commands),
            ("local", local_training_commands),
        ]:
            with self.subTest(training_config=config_name):
                ablation = config["reward_component_ablation"]
                self.assertTrue(ablation["enabled"])
                self.assertEqual(ablation["output_dir"], "outputs/reward_component_training_ablation")
                variants = ablation["variants"]
                self.assertEqual(set(variants), {"no_red", "no_valid", "no_verifier", "cov_only"})
                self.assertEqual(config["grpo"]["env"]["BSC_VERIFIER"], "rule")
                self.assertEqual(variants["no_red"]["env"]["BSC_W_RED"], "0.0")
                self.assertEqual(variants["no_valid"]["env"]["BSC_W_VALID"], "0.0")
                self.assertEqual(variants["no_valid"]["env"]["BSC_VERIFIER"], "none")
                self.assertEqual(variants["no_verifier"]["env"]["BSC_VERIFIER"], "none")
                self.assertNotIn("BSC_W_VALID", variants["no_verifier"]["env"])
                self.assertNotIn("BSC_W_RED", variants["no_verifier"]["env"])
                self.assertEqual(variants["cov_only"]["env"]["BSC_W_VALID"], "0.0")
                self.assertEqual(variants["cov_only"]["env"]["BSC_W_RED"], "0.0")
                self.assertEqual(variants["cov_only"]["env"]["BSC_VERIFIER"], "none")
        for config_name, providers in [
            ("example", reward_ablation_generator_example),
            ("local", reward_ablation_generators),
        ]:
            with self.subTest(reward_ablation_generators=config_name):
                by_name = {item["name"]: item for item in providers}
                self.assertEqual(set(by_name), {"no_red", "no_valid", "no_verifier", "cov_only"})
                self.assertEqual(by_name["no_red"]["model"], "outputs/checkpoints/evaluation_criteria_policy_rl_no_red")
                self.assertEqual(by_name["no_valid"]["model"], "outputs/checkpoints/evaluation_criteria_policy_rl_no_valid")
                self.assertEqual(by_name["no_verifier"]["model"], "outputs/checkpoints/evaluation_criteria_policy_rl_no_verifier")
                self.assertEqual(by_name["cov_only"]["model"], "outputs/checkpoints/evaluation_criteria_policy_rl_cov_only")

        training_manifest = read_json("outputs/training_commands/training_manifest.json")
        ablation_done_template = read_json("outputs/reward_component_training_ablation/training_done.template.json")
        self.assertEqual(
            training_manifest["expected_training_done"],
            "outputs/training_commands/training_done.json",
        )
        self.assertIn(
            "scripts/fill_training_done_sha256.py --input outputs/training_commands/training_done.json",
            training_manifest["training_done_sha256_command"],
        )
        self.assertEqual(
            training_manifest["reward_component_ablation"]["reward_variants"],
            ["full", "no_red", "no_valid", "no_verifier", "cov_only"],
        )
        self.assertEqual(
            training_manifest["reward_component_ablation"]["training_done_template"],
            "outputs/reward_component_training_ablation/training_done.template.json",
        )
        self.assertEqual(
            training_manifest["reward_component_ablation"]["expected_training_done"],
            "outputs/reward_component_training_ablation/training_done.json",
        )
        self.assertIn(
            "scripts/fill_training_done_sha256.py --input outputs/reward_component_training_ablation/training_done.json",
            training_manifest["reward_component_ablation"]["sha256_fill_command"],
        )
        self.assertEqual(
            ablation_done_template["reward_variants"],
            ["full", "no_red", "no_valid", "no_verifier", "cov_only"],
        )
        self.assertEqual(
            set(ablation_done_template["variants"]),
            {"full", "no_red", "no_valid", "no_verifier", "cov_only"},
        )
        self.assertEqual(ablation_done_template["variants"]["no_verifier"]["env"]["BSC_VERIFIER"], "none")
        self.assertEqual(ablation_done_template["variants"]["no_verifier"]["env"]["BSC_W_VALID"], "0.5")
        self.assertEqual(ablation_done_template["variants"]["no_verifier"]["env"]["BSC_W_RED"], "0.5")
        self.assertEqual(ablation_done_template["variants"]["no_valid"]["env"]["BSC_W_VALID"], "0.0")
        self.assertEqual(ablation_done_template["variants"]["cov_only"]["env"]["BSC_W_RED"], "0.0")
        self.assertNotIn("rubricbench_verl.parquet", verl_template)
        self.assertNotIn("data/processed/verl/rubricbench_rl_train.parquet", local_verl_template)
        self.assertNotIn("data/processed/verl/rubricbench_rl_val.parquet", local_verl_template)

    def test_sprint_plan_keeps_grpo_on_proxy_gold_not_hard_gold(self) -> None:
        sprint = read_json("configs/sprint_plan_20day.template.json")
        text = json.dumps(sprint, ensure_ascii=False)

        self.assertIn("--input data/processed/splits/rubricbench_gold_train_seed.jsonl", text)
        self.assertIn("--from-stage filter_teacher_rubrics --to-stage filter_proxy_gold_rewardbench2_downstream_overlap", text)
        self.assertIn("data/processed/blindspot_sft.unfiltered.jsonl", text)
        self.assertIn("data/processed/proxy_gold.unfiltered.jsonl", text)
        self.assertIn("outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json", text)
        self.assertIn("outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json", text)
        self.assertIn("--file-name blindspot_sft.jsonl", text)
        self.assertIn("data/processed/proxy_gold_verl.parquet", text)
        self.assertIn("--input data/processed/proxy_gold.jsonl", text)
        self.assertIn("--data-source multi_teacher_proxy", text)
        self.assertNotIn("--sft-output data/processed/blindspot_sft.jsonl", text)
        self.assertNotIn("--proxy-gold-output data/processed/proxy_gold.jsonl", text)
        self.assertIn("data/processed/rewardbench2_queries.clean.jsonl", text)
        self.assertNotIn("generate_model_rubrics.py --input data/processed/rewardbench2_queries.jsonl", text)
        self.assertNotIn("estimate_api_budget.py --input data/processed/rewardbench2_queries.jsonl", text)
        self.assertNotIn("generate_teacher_rubrics.py --input data/processed/rubricbench_queries.jsonl", text)
        self.assertNotIn("estimate_api_budget.py --input data/processed/rubricbench_queries.jsonl --providers configs/providers.local.jsonl", text)
        self.assertNotIn("--sft-output data/processed/sft.jsonl", text)
        self.assertNotIn("--file-name sft.jsonl", text)
        self.assertNotIn("data/processed/verl_train.parquet", text)
        self.assertNotIn(
            "--input data/processed/rubricbench_gold.jsonl --output data/processed/verl_train.parquet",
            text,
        )

    def test_real_run_uses_domain_scoped_model_rubric_files(self) -> None:
        assembly = read_json("configs/real_run_assembly.template.json")
        main_matrix = read_json("configs/methods_matrix_real.template.json")
        generalization = read_json("configs/methods_matrix_generalization.template.json")
        validation = read_json("configs/validate_rubrics_real.template.json")

        domain_outputs = {domain["name"]: domain["output"] for domain in assembly["model_generation"]["domains"]}
        self.assertEqual(
            domain_outputs["generate_model_rubrics_rubricbench"],
            "data/processed/rubricbench_model_rubrics.jsonl",
        )
        self.assertEqual(
            domain_outputs["generate_model_rubrics_rewardbench"],
            "data/processed/rewardbench_model_rubrics.jsonl",
        )
        self.assertEqual(
            domain_outputs["generate_model_rubrics_judgebench"],
            "data/processed/judgebench_model_rubrics.jsonl",
        )
        self.assertEqual(
            domain_outputs["generate_model_rubrics_rewardbench2"],
            "data/processed/rewardbench2_model_rubrics.jsonl",
        )
        self.assertEqual(
            domain_outputs["generate_model_rubrics_rmbench"],
            "data/processed/rmbench_model_rubrics.jsonl",
        )
        self.assertEqual(
            domain_outputs["generate_model_rubrics_healthbench"],
            "data/processed/healthbench_model_rubrics.jsonl",
        )
        self.assertEqual(
            domain_outputs["generate_model_rubrics_writingbench"],
            "data/processed/writingbench_model_rubrics.jsonl",
        )

        self.assertEqual(main_matrix["common"]["bsc_rubrics"], "data/processed/rubricbench_model_rubrics.jsonl")
        self.assertEqual(
            main_matrix["common"]["downstream_preferences"],
            "data/processed/splits/rewardbench_pref_downstream_holdout.jsonl",
        )
        self.assertEqual(main_matrix["common"]["downstream_rubrics"], "data/processed/rewardbench_model_rubrics.jsonl")
        self.assertNotIn("model_rubrics", main_matrix["common"])
        verifier_ablation = main_matrix["common"]["verifier_filter_ablation"]
        self.assertTrue(verifier_ablation["enabled"])
        self.assertEqual(verifier_ablation["raw_teachers"], "data/processed/teacher_rubrics_raw.jsonl")
        self.assertEqual(verifier_ablation["filtered_teachers"], "data/processed/teacher_rubrics_filtered.jsonl")

        method_paths = {method["name"]: method["bsc_rubrics"] for method in generalization["methods"]}
        self.assertEqual(method_paths["healthbench_base"], "data/processed/healthbench_model_rubrics.jsonl")
        self.assertEqual(method_paths["healthbench_sft_rl"], "data/processed/healthbench_model_rubrics.jsonl")
        self.assertEqual(method_paths["writingbench_base"], "data/processed/writingbench_model_rubrics.jsonl")
        self.assertEqual(method_paths["writingbench_sft_rl"], "data/processed/writingbench_model_rubrics.jsonl")

        validation_inputs = {stage["args"]["input"] for stage in validation["stages"]}
        self.assertIn("data/processed/rubricbench_model_rubrics.jsonl", validation_inputs)
        self.assertIn("data/processed/rewardbench_model_rubrics.jsonl", validation_inputs)
        self.assertIn("data/processed/judgebench_model_rubrics.jsonl", validation_inputs)
        self.assertIn("data/processed/rewardbench2_model_rubrics.jsonl", validation_inputs)
        self.assertIn("data/processed/rmbench_model_rubrics.jsonl", validation_inputs)
        self.assertIn("data/processed/healthbench_model_rubrics.jsonl", validation_inputs)
        self.assertIn("data/processed/writingbench_model_rubrics.jsonl", validation_inputs)
        for stage in validation["stages"]:
            if stage["name"].startswith("validate_") and "model_rubrics" in stage["name"]:
                self.assertTrue(stage["args"]["require_valid_flags"], stage["name"])

    def test_trained_method_gates_require_full_training_provenance(self) -> None:
        configs = [
            read_json("configs/methods_matrix_real.template.json"),
            read_json("configs/methods_matrix_judgebench.template.json"),
            read_json("configs/methods_matrix_rewardbench2.template.json"),
            read_json("configs/methods_matrix_generalization.template.json"),
        ]
        generated = [
            read_json("configs/pipeline_matrix_real.generated.json")["stages"][0]["args"],
            read_json("configs/pipeline_matrix_judgebench.generated.json")["stages"][0]["args"],
            read_json("configs/pipeline_matrix_rewardbench2.generated.json")["stages"][0]["args"],
            read_json("configs/pipeline_matrix_generalization.generated.json")["stages"][0]["args"],
        ]

        for item in configs:
            gate = item["common"]["trained_method_gate"]
            with self.subTest(gate=gate["name"]):
                self.assertIn("sft_checkpoint,rl_checkpoint", gate["required_json"][0])
                self.assertIn("served_methods,served_generators", gate["required_json"][0])
                self.assertIn("serving.base,serving.sft_only,serving.sft_rl", gate["required_json"][0])
                self.assertIn("operator,date,sft_config,grpo_config,sft_data,rl_data,rl_data_report,reward_function", gate["required_json"][0])
                self.assertIn("sft_config_sha256,grpo_config_sha256,sft_data_sha256", gate["required_json"][0])
                self.assertIn("rl_data_sha256,rl_data_report_sha256,reward_function_sha256", gate["required_json"][0])
                self.assertIn(
                    "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
                    gate["required_json_contains"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl",
                    gate["required_json_contains"],
                )
                self.assertIn("outputs/verifier/teacher_rubrics_filtered_report.json", gate["required_paths"])
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:input,input_sha256,output,output_sha256,mode,provider,provider_sha256,budget_report,budget_report_sha256,preflight_report,preflight_report_sha256,n_input_records,n_output_records,n_input_rubrics,n_valid_rubrics",
                    gate["required_json"],
                )
                self.assertIn("outputs/sft_data/proxy_gold_build_report.json", gate["required_paths"])
                self.assertTrue(
                    any(
                        item.startswith("outputs/sft_data/proxy_gold_build_report.json:")
                        and "input_sha256" in item
                        and "proxy_gold_output_sha256" in item
                        for item in gate["required_json"]
                    )
                )
                for marker in ["test_main", "holdout", "downstream"]:
                    self.assertIn(
                        f"outputs/sft_data/proxy_gold_build_report.json:forbidden_data_source_markers={marker}",
                        gate["required_json_contains"],
                    )
                    self.assertIn(
                        f"outputs/sft_data/proxy_gold_verl_report.json:forbidden_source_markers={marker}",
                        gate["required_json_contains"],
                    )
                for split in ["test_main", "holdout", "downstream", "test"]:
                    self.assertIn(
                        f"outputs/sft_data/proxy_gold_build_report.json:forbidden_splits={split}",
                        gate["required_json_contains"],
                    )
                    self.assertIn(
                        f"outputs/sft_data/proxy_gold_verl_report.json:forbidden_splits={split}",
                        gate["required_json_contains"],
                    )
                self.assertIn(
                    "outputs/training_commands/training_done.json:sft_config=configs/llamafactory_sft.local.yaml",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:grpo_config=configs/verl_grpo_bsc.local.yaml",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:sft_data=data/processed/blindspot_sft.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:rl_data=data/processed/proxy_gold_verl.parquet",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:rl_data_report=outputs/sft_data/proxy_gold_verl_report.json",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:input=data/processed/teacher_rubrics_raw.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:output=data/processed/teacher_rubrics_filtered.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:mode=api",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:provider=configs/verifier.local.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:budget_report=outputs/api_budget/meta_verifier_budget.json",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:preflight_report=outputs/preflight/sft_data_preflight.json",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:input=data/processed/teacher_rubrics_training_clean.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:sft_output=data/processed/blindspot_sft.unfiltered.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:proxy_gold_output=data/processed/proxy_gold.unfiltered.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:data_source=multi_teacher_proxy",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_verl_report.json:input=data/processed/proxy_gold.jsonl",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_verl_report.json:output=data/processed/proxy_gold_verl.parquet",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_verl_report.json:input_sha256=data/processed/proxy_gold.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:sft_config_sha256=configs/llamafactory_sft.local.yaml",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:grpo_config_sha256=configs/verl_grpo_bsc.local.yaml",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:sft_data_sha256=data/processed/blindspot_sft.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:rl_data_sha256=data/processed/proxy_gold_verl.parquet",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:rl_data_report_sha256=outputs/sft_data/proxy_gold_verl_report.json",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:reward_function_sha256=src/blindspot_rl/verl_reward.py",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:input_sha256=data/processed/teacher_rubrics_raw.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:output_sha256=data/processed/teacher_rubrics_filtered.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:provider_sha256=configs/verifier.local.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:budget_report_sha256=outputs/api_budget/meta_verifier_budget.json",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:preflight_report_sha256=outputs/preflight/sft_data_preflight.json",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:input_sha256=data/processed/teacher_rubrics_training_clean.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:sft_output_sha256=data/processed/blindspot_sft.unfiltered.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:proxy_gold_output_sha256=data/processed/proxy_gold.unfiltered.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:reward_function=src/blindspot_rl/verl_reward.py:compute_score",
                    gate["required_json_equals"],
                )

        for gate in generated:
            with self.subTest(generated_gate=gate["name"]):
                self.assertIn("served_methods,served_generators", gate["required_json"][0])
                self.assertIn("serving.sft_rl", gate["required_json"][0])
                self.assertIn("sft_config_sha256,grpo_config_sha256,sft_data_sha256", gate["required_json"][0])
                self.assertIn("rl_data_sha256,rl_data_report_sha256,reward_function_sha256", gate["required_json"][0])
                self.assertIn("outputs/verifier/teacher_rubrics_filtered_report.json", gate["required_path"])
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:input,input_sha256,output,output_sha256,mode,provider,provider_sha256,budget_report,budget_report_sha256,preflight_report,preflight_report_sha256,n_input_records,n_output_records,n_input_rubrics,n_valid_rubrics",
                    gate["required_json"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
                    gate["required_json_contains"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl",
                    gate["required_json_contains"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:mode=api",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:budget_report=outputs/api_budget/meta_verifier_budget.json",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:preflight_report=outputs/preflight/sft_data_preflight.json",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:reward_function=src/blindspot_rl/verl_reward.py:compute_score",
                    gate["required_json_equals"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:sft_config_sha256=configs/llamafactory_sft.local.yaml",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:grpo_config_sha256=configs/verl_grpo_bsc.local.yaml",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:sft_data_sha256=data/processed/blindspot_sft.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:rl_data_sha256=data/processed/proxy_gold_verl.parquet",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:rl_data_report_sha256=outputs/sft_data/proxy_gold_verl_report.json",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/training_commands/training_done.json:reward_function_sha256=src/blindspot_rl/verl_reward.py",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:output_sha256=data/processed/teacher_rubrics_filtered.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/verifier/teacher_rubrics_filtered_report.json:budget_report_sha256=outputs/api_budget/meta_verifier_budget.json",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_build_report.json:proxy_gold_output_sha256=data/processed/proxy_gold.unfiltered.jsonl",
                    gate["required_json_sha256"],
                )
                self.assertIn(
                    "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
                    gate["required_json_sha256"],
                )

    def test_result_card_real_tracks_real_run_startup_gates(self) -> None:
        config = read_json("configs/result_card_real.template.json")
        paths = list(collect_paths(config))
        raw_gate_types = {item["name"]: item["type"] for item in config["raw_audit_gates"]}

        self.assertIn("outputs/data_sources/local_config_init.json", paths)
        self.assertEqual(raw_gate_types["Data Source Local Config"], "data_source_local_config")
        self.assertIn("outputs/preflight/real_run_preflight.json", paths)
        self.assertEqual(raw_gate_types["Real Run Preflight"], "preflight")
        self.assertIn("outputs/preflight/sft_data_preflight.json", paths)
        self.assertEqual(raw_gate_types["SFT Data Preflight"], "preflight")
        self.assertIn("outputs/data_readiness_audit.json", paths)
        self.assertIn("outputs/submission_readiness/latex_compile_report.json", paths)
        self.assertEqual(raw_gate_types["AAAI LaTeX Compile"], "latex_compile")
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench Proxy-Train Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench Proxy-Train vs RewardBench Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_judgebench_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench Proxy-Train vs JudgeBench Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench Proxy-Train vs RewardBench-2 Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench2_queries_rubricbench_train_seed_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench-2 Query Pool vs RubricBench Train-Seed Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench2_queries_researchrubrics_train_seed_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench-2 Query Pool vs ResearchRubrics Train-Seed Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench2_multicandidate_rubricbench_train_seed_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench-2 Multicandidate vs RubricBench Train-Seed Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json", paths)
        self.assertEqual(raw_gate_types["RewardBench-2 Multicandidate vs ResearchRubrics Train-Seed Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json", paths)
        self.assertEqual(raw_gate_types["Pre-SFT Clean Proxy-Train vs Hard-Gold Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_rewardbench_downstream_audit.json", paths)
        self.assertEqual(raw_gate_types["Pre-SFT Clean Proxy-Train vs RewardBench Holdout Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_judgebench_downstream_audit.json", paths)
        self.assertEqual(raw_gate_types["Pre-SFT Clean Proxy-Train vs JudgeBench Holdout Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/clean_proxy_train_vs_rewardbench2_downstream_audit.json", paths)
        self.assertEqual(raw_gate_types["Pre-SFT Clean Proxy-Train vs RewardBench-2 Holdout Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/hard_gold_holdout_contamination.json", paths)
        self.assertEqual(raw_gate_types["Hard-Gold Holdout Contamination Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json", paths)
        self.assertEqual(raw_gate_types["RewardBench Downstream Holdout Contamination Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/judgebench_downstream_holdout_contamination.json", paths)
        self.assertEqual(raw_gate_types["JudgeBench Downstream Holdout Contamination Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json", paths)
        self.assertEqual(raw_gate_types["RewardBench-2 Downstream Holdout Contamination Audit"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/blindspot_sft_hard_gold_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["BlindSpot SFT vs Hard-Gold Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/blindspot_sft_rewardbench_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["BlindSpot SFT vs RewardBench Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/blindspot_sft_judgebench_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["BlindSpot SFT vs JudgeBench Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["BlindSpot SFT vs RewardBench-2 Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/proxy_gold_hard_gold_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["Proxy-Gold vs Hard-Gold Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/proxy_gold_rewardbench_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["Proxy-Gold vs RewardBench Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/proxy_gold_judgebench_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["Proxy-Gold vs JudgeBench Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json", paths)
        self.assertEqual(raw_gate_types["Proxy-Gold vs RewardBench-2 Holdout Filter"], "contamination_audit")
        self.assertIn("outputs/api_budget/model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/judgebench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench2_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/rmbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/teacher_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/healthbench_model_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/writingbench_teacher_rubrics_budget.json", paths)
        self.assertIn("outputs/api_budget/meta_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rubricbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/judgebench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rewardbench2_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/rmbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/healthbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/api_budget/writingbench_model_rubrics_verifier_budget.json", paths)
        self.assertIn("outputs/verifier/rubricbench_model_rubrics_stats.jsonl", paths)
        self.assertIn("outputs/validation/rubricbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/rewardbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/judgebench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/rewardbench2_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/rmbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/healthbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/writingbench_model_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/teacher_rubrics/validation_report.json", paths)
        self.assertIn("outputs/validation/teacher_rubrics_filtered/validation_report.json", paths)
        self.assertIn("configs/manifest_sft_data_real.template.json", paths)
        self.assertIn("configs/manifest_matrix_judgebench.generated.json", paths)
        self.assertIn("configs/manifest_matrix_rewardbench2.generated.json", paths)
        self.assertIn("configs/manifest_matrix_generalization.generated.json", paths)
        self.assertIn("outputs/training_commands/training_completion_gate.json", paths)
        self.assertEqual(raw_gate_types["Training Completion Gate"], "manual_gate")
        self.assertIn("outputs/matrix_real/trained_method_gate.json", paths)
        self.assertIn("outputs/matrix_judgebench/trained_method_gate.json", paths)
        self.assertIn("outputs/matrix_rewardbench2/trained_method_gate.json", paths)
        self.assertIn("outputs/generalization_matrix/trained_method_gate.json", paths)
        self.assertIn("outputs/policy_rlvr/downstream_rlvr_completion_gate.json", paths)
        self.assertIn("outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json", paths)
        self.assertNotIn("outputs/minimal_claim/preflight.json", paths)

    def test_submission_readiness_tracks_raw_gates(self) -> None:
        template = read_json("configs/submission_readiness_real.template.json")
        generated = read_json("configs/pipeline_real_run.generated.json")
        template_stage_names = [stage["name"] for stage in template["stages"]]
        generated_stage_names = [stage["name"] for stage in generated["stages"]]
        template_gate_specs = set(template["stages"][1]["args"]["raw_gate"])
        generated_stage = next(stage for stage in generated["stages"] if stage["name"] == "submission_readiness")
        generated_compile = next(stage for stage in generated["stages"] if stage["name"] == "latex_compile_check")
        generated_gate_specs = set(generated_stage["args"]["raw_gate"])

        self.assertLess(template_stage_names.index("latex_compile_check"), template_stage_names.index("submission_readiness"))
        self.assertLess(generated_stage_names.index("latex_compile_check"), generated_stage_names.index("submission_readiness"))
        self.assertEqual(generated_compile["type"], "latex_compile_check")
        self.assertTrue(generated_compile["args"]["compile"])
        self.assertTrue(generated_compile["args"]["require_official_style"])
        self.assertTrue(generated_compile["args"]["require_anonymous"])
        self.assertEqual(generated_compile["args"]["max_pages"], 8)
        self.assertEqual(
            generated_compile["args"]["output_json"],
            "outputs/submission_readiness/latex_compile_report.json",
        )
        self.assertTrue(template["stages"][1]["args"]["strict"])
        self.assertTrue(generated_stage["args"]["strict"])
        expected = {
            "Data Source Local Config|data_source_local_config|outputs/data_sources/local_config_init.json",
            "Data Source Report|data_source_report|outputs/data_sources/source_report.json",
            "Data Readiness Audit|audit|outputs/data_readiness_audit.json",
            "Real Run Preflight|preflight|outputs/preflight/real_run_preflight.json",
            "SFT Data Preflight|preflight|outputs/preflight/sft_data_preflight.json",
            "RubricBench Gold Validation|gold_validation|outputs/data_validation/rubricbench_gold.json",
            "ResearchRubrics Gold Validation|gold_validation|outputs/data_validation/researchrubrics_gold.json",
            "RewardBench Proxy-Train Holdout Filter|contamination_audit|outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json",
            "RewardBench Proxy-Train vs RewardBench Holdout Filter|contamination_audit|outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench_holdout_filter.json",
            "RewardBench Proxy-Train vs JudgeBench Holdout Filter|contamination_audit|outputs/contamination_audit/rewardbench_pref_sft_proxy_train_judgebench_holdout_filter.json",
            "RewardBench Proxy-Train vs RewardBench-2 Holdout Filter|contamination_audit|outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json",
            "RewardBench-2 Query Pool vs RubricBench Train-Seed Filter|contamination_audit|outputs/contamination_audit/rewardbench2_queries_rubricbench_train_seed_filter.json",
            "RewardBench-2 Query Pool vs ResearchRubrics Train-Seed Filter|contamination_audit|outputs/contamination_audit/rewardbench2_queries_researchrubrics_train_seed_filter.json",
            "RewardBench-2 Multicandidate vs RubricBench Train-Seed Filter|contamination_audit|outputs/contamination_audit/rewardbench2_multicandidate_rubricbench_train_seed_filter.json",
            "RewardBench-2 Multicandidate vs ResearchRubrics Train-Seed Filter|contamination_audit|outputs/contamination_audit/rewardbench2_multicandidate_researchrubrics_train_seed_filter.json",
            "Pre-SFT Clean Proxy-Train vs Hard-Gold Audit|contamination_audit|outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json",
            "Pre-SFT Clean Proxy-Train vs RewardBench Holdout Audit|contamination_audit|outputs/contamination_audit/clean_proxy_train_vs_rewardbench_downstream_audit.json",
            "Pre-SFT Clean Proxy-Train vs JudgeBench Holdout Audit|contamination_audit|outputs/contamination_audit/clean_proxy_train_vs_judgebench_downstream_audit.json",
            "Pre-SFT Clean Proxy-Train vs RewardBench-2 Holdout Audit|contamination_audit|outputs/contamination_audit/clean_proxy_train_vs_rewardbench2_downstream_audit.json",
            "Hard-Gold Holdout Contamination Audit|contamination_audit|outputs/contamination_audit/hard_gold_holdout_contamination.json",
            "RewardBench Downstream Holdout Contamination Audit|contamination_audit|outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json",
            "JudgeBench Downstream Holdout Contamination Audit|contamination_audit|outputs/contamination_audit/judgebench_downstream_holdout_contamination.json",
            "RewardBench-2 Downstream Holdout Contamination Audit|contamination_audit|outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json",
            "BlindSpot SFT vs Hard-Gold Holdout Filter|contamination_audit|outputs/contamination_audit/blindspot_sft_hard_gold_holdout_filter.json",
            "BlindSpot SFT vs RewardBench Holdout Filter|contamination_audit|outputs/contamination_audit/blindspot_sft_rewardbench_holdout_filter.json",
            "BlindSpot SFT vs JudgeBench Holdout Filter|contamination_audit|outputs/contamination_audit/blindspot_sft_judgebench_holdout_filter.json",
            "BlindSpot SFT vs RewardBench-2 Holdout Filter|contamination_audit|outputs/contamination_audit/blindspot_sft_rewardbench2_holdout_filter.json",
            "Proxy-Gold vs Hard-Gold Holdout Filter|contamination_audit|outputs/contamination_audit/proxy_gold_hard_gold_holdout_filter.json",
            "Proxy-Gold vs RewardBench Holdout Filter|contamination_audit|outputs/contamination_audit/proxy_gold_rewardbench_holdout_filter.json",
            "Proxy-Gold vs JudgeBench Holdout Filter|contamination_audit|outputs/contamination_audit/proxy_gold_judgebench_holdout_filter.json",
            "Proxy-Gold vs RewardBench-2 Holdout Filter|contamination_audit|outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json",
            "RubricBench Model Evaluation-Criteria API Budget|api_budget|outputs/api_budget/model_rubrics_budget.json",
            "Teacher Evaluation-Criteria API Budget|api_budget|outputs/api_budget/teacher_rubrics_budget.json",
            "Meta-Verifier API Budget|api_budget|outputs/api_budget/meta_verifier_budget.json",
            "RubricBench Model Criteria Verifier API Budget|api_budget|outputs/api_budget/rubricbench_model_rubrics_verifier_budget.json",
            "RubricBench Model Criteria Verifier Stats|generic|outputs/verifier/rubricbench_model_rubrics_stats.jsonl",
            "Filtered Teacher Evaluation-Criteria Validation|validation|outputs/validation/teacher_rubrics_filtered/validation_report.json",
            "Training Completion Gate|manual_gate|outputs/training_commands/training_completion_gate.json",
            "Matrix Trained Method Gate|manual_gate|outputs/matrix_real/trained_method_gate.json",
            "JudgeBench Trained Method Gate|manual_gate|outputs/matrix_judgebench/trained_method_gate.json",
            "RewardBench-2 Trained Method Gate|manual_gate|outputs/matrix_rewardbench2/trained_method_gate.json",
            "Generalization Trained Method Gate|manual_gate|outputs/generalization_matrix/trained_method_gate.json",
            "Downstream Policy RLVR Completion Gate|manual_gate|outputs/policy_rlvr/downstream_rlvr_completion_gate.json",
            "AAAI LaTeX Compile|latex_compile|outputs/submission_readiness/latex_compile_report.json",
        }
        self.assertTrue(expected.issubset(template_gate_specs))
        self.assertTrue(expected.issubset(generated_gate_specs))

    def test_real_run_assembly_enforces_downstream_policy_rlvr_manual_gate(self) -> None:
        assembly = read_json("configs/real_run_assembly.template.json")
        generated = read_json("configs/pipeline_real_run.generated.json")
        gate = assembly["downstream_rlvr_completion_gate"]
        training_gate = assembly["training_completion_gate"]
        generated_gate = next(
            stage["args"]
            for stage in generated["stages"]
            if stage["name"] == "downstream_rlvr_completion_gate"
        )

        self.assertTrue(assembly["downstream_rlvr_commands"]["enabled"])
        self.assertEqual(
            assembly["downstream_rlvr_commands"]["config"],
            "configs/downstream_rlvr_commands.example.json",
        )
        self.assertEqual(gate["output"], "outputs/policy_rlvr/downstream_rlvr_completion_gate.json")
        self.assertTrue(gate["strict"])
        self.assertIn("outputs/training_commands/training_done.json:sft_checkpoint", training_gate["required_json"][0])
        self.assertIn("served_methods,served_generators", training_gate["required_json"][0])
        self.assertIn("serving.base,serving.sft_only,serving.sft_rl", training_gate["required_json"][0])
        self.assertIn("sft_config_sha256,grpo_config_sha256,sft_data_sha256", training_gate["required_json"][0])
        self.assertIn("rl_data_sha256,rl_data_report_sha256,reward_function_sha256", training_gate["required_json"][0])
        self.assertIn(
            "outputs/training_commands/training_done.json:served_methods=base,sft_only,sft_rl",
            training_gate["required_json_contains"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl",
            training_gate["required_json_contains"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:rl_data=data/processed/proxy_gold_verl.parquet",
            training_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:rl_data_report=outputs/sft_data/proxy_gold_verl_report.json",
            training_gate["required_json_equals"],
        )
        self.assertIn("outputs/sft_data/proxy_gold_build_report.json", training_gate["required_paths"])
        for marker in ["test_main", "holdout", "downstream"]:
            self.assertIn(
                f"outputs/sft_data/proxy_gold_build_report.json:forbidden_data_source_markers={marker}",
                training_gate["required_json_contains"],
            )
            self.assertIn(
                f"outputs/sft_data/proxy_gold_verl_report.json:forbidden_source_markers={marker}",
                training_gate["required_json_contains"],
            )
        for split in ["test_main", "holdout", "downstream", "test"]:
            self.assertIn(
                f"outputs/sft_data/proxy_gold_build_report.json:forbidden_splits={split}",
                training_gate["required_json_contains"],
            )
            self.assertIn(
                f"outputs/sft_data/proxy_gold_verl_report.json:forbidden_splits={split}",
                training_gate["required_json_contains"],
            )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json:input=data/processed/teacher_rubrics_training_clean.jsonl",
            training_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json:sft_output=data/processed/blindspot_sft.unfiltered.jsonl",
            training_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json:proxy_gold_output=data/processed/proxy_gold.unfiltered.jsonl",
            training_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_verl_report.json:input=data/processed/proxy_gold.jsonl",
            training_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_verl_report.json:output=data/processed/proxy_gold_verl.parquet",
            training_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_verl_report.json:input_sha256=data/processed/proxy_gold.jsonl",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:sft_config_sha256=configs/llamafactory_sft.local.yaml",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:grpo_config_sha256=configs/verl_grpo_bsc.local.yaml",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:sft_data_sha256=data/processed/blindspot_sft.jsonl",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:rl_data_sha256=data/processed/proxy_gold_verl.parquet",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:rl_data_report_sha256=outputs/sft_data/proxy_gold_verl_report.json",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:reward_function_sha256=src/blindspot_rl/verl_reward.py",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json:input_sha256=data/processed/teacher_rubrics_training_clean.jsonl",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json:sft_output_sha256=data/processed/blindspot_sft.unfiltered.jsonl",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json:proxy_gold_output_sha256=data/processed/proxy_gold.unfiltered.jsonl",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet",
            training_gate["required_json_sha256"],
        )
        self.assertIn(
            "outputs/training_commands/training_done.json:reward_function=src/blindspot_rl/verl_reward.py:compute_score",
            training_gate["required_json_equals"],
        )
        self.assertIn("outputs/policy_rlvr/downstream_rlvr_done.json:healthbench_hard_policy", gate["required_json"][0])
        self.assertIn("benchmarks.healthbench_hard.criteria_policy_checkpoint", gate["required_json"][0])
        self.assertIn("benchmarks.arenahard.criteria_policy_checkpoint", gate["required_json"][0])
        self.assertNotIn("benchmarks.healthbench_hard.rubric_generator", gate["required_json"][0])
        self.assertNotIn("benchmarks.arenahard.rubric_generator", gate["required_json"][0])
        self.assertIn(
            "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.train_data=data/processed/healthbench_hard_policy_rlvr.parquet",
            gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
            gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.arenahard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
            gate["required_json_equals"],
        )
        self.assertFalse(any("benchmarks.healthbench_hard.rubric_generator" in item for item in gate["required_json_equals"]))
        self.assertFalse(any("benchmarks.arenahard.rubric_generator" in item for item in gate["required_json_equals"]))
        self.assertIn(
            "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.arenahard.reward_function=src/blindspot_rl/policy_reward.py:compute_score",
            gate["required_json_equals"],
        )
        self.assertIn("benchmarks.healthbench_hard.criteria_policy_checkpoint", generated_gate["required_json"][0])
        self.assertIn("benchmarks.arenahard.criteria_policy_checkpoint", generated_gate["required_json"][0])
        self.assertNotIn("benchmarks.healthbench_hard.rubric_generator", generated_gate["required_json"][0])
        self.assertNotIn("benchmarks.arenahard.rubric_generator", generated_gate["required_json"][0])
        self.assertIn(
            "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.healthbench_hard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
            generated_gate["required_json_equals"],
        )
        self.assertIn(
            "outputs/policy_rlvr/downstream_rlvr_done.json:benchmarks.arenahard.criteria_policy_checkpoint=outputs/checkpoints/evaluation_criteria_policy_rl",
            generated_gate["required_json_equals"],
        )
        self.assertFalse(
            any("benchmarks.healthbench_hard.rubric_generator" in item for item in generated_gate["required_json_equals"])
        )
        self.assertFalse(
            any("benchmarks.arenahard.rubric_generator" in item for item in generated_gate["required_json_equals"])
        )
        self.assertIn("outputs/downstream_rlvr_commands/downstream_rlvr_manifest.json", gate["required_paths"])
        self.assertIn("outputs/policy_rlvr/downstream_rlvr_done.json", gate["required_paths"])
        self.assertIn("outputs/policy_rlvr/healthbench_hard_eval.json", gate["required_paths"])
        self.assertEqual(
            assembly["downstream_rlvr_data_pipeline"],
            "configs/policy_rlvr_data_real.template.json",
        )

    def test_downstream_policy_rlvr_eval_commands_reference_existing_entrypoint(self) -> None:
        config = read_json("configs/downstream_rlvr_commands.example.json")

        for benchmark in config["benchmarks"]:
            command = benchmark["eval_command"]
            self.assertIn("scripts/evaluate_policy_outputs.py", command)
            self.assertTrue((ROOT / "scripts/evaluate_policy_outputs.py").exists())

    def test_downstream_policy_rlvr_configs_have_example_reward_hooks(self) -> None:
        command_config = read_json("configs/downstream_rlvr_commands.example.json")
        expected_examples = {
            "healthbench_hard": ROOT / "configs/verl_policy_grpo_healthbench_hard.example.yaml",
            "arenahard": ROOT / "configs/verl_policy_grpo_arenahard.example.yaml",
        }

        for benchmark in command_config["benchmarks"]:
            name = benchmark["name"]
            example_path = expected_examples[name]
            text = example_path.read_text(encoding="utf-8")
            self.assertTrue(example_path.exists())
            self.assertIn("src/blindspot_rl/policy_reward.py", text)
            self.assertIn("name: compute_score", text)
            self.assertIn(f"BSC_POLICY_RUBRIC_FILE={benchmark['rubric_file']}", text)
            self.assertIn(".local.yaml", benchmark["config"])


def read_json(relative_path: str) -> Any:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def read_jsonl(relative_path: str) -> list[dict[str, Any]]:
    rows = []
    for line in (ROOT / relative_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def expected_real_preflight_report_path() -> str:
    assembly = read_json("configs/real_run_assembly.template.json")
    preflight = read_json(assembly["preflight_pipeline"])
    for stage in preflight["stages"]:
        if stage.get("type") == "preflight":
            output = stage.get("args", {}).get("output")
            if isinstance(output, str) and output:
                return output
    raise AssertionError("real-run preflight pipeline has no preflight output path")


def collect_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str):
                paths.append(item)
            else:
                paths.extend(collect_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(collect_paths(item))
    elif isinstance(value, str) and value.startswith("outputs/"):
        paths.append(value)
    return paths


def collect_metric_names(value: Any) -> set[str]:
    metrics: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"metric", "left_metric", "right_metric"} and isinstance(item, str):
                metrics.add(item)
            else:
                metrics.update(collect_metric_names(item))
    elif isinstance(value, list):
        for item in value:
            metrics.update(collect_metric_names(item))
    return metrics


def collect_metric_specs(value: Any) -> set[tuple[str, str, str]]:
    specs: set[tuple[str, str, str]] = set()
    if isinstance(value, dict):
        metrics = value.get("metrics")
        if isinstance(metrics, list):
            for item in metrics:
                if not isinstance(item, dict):
                    continue
                specs.add(
                    (
                        str(item.get("path", "")),
                        str(item.get("metric", "")),
                        str(item.get("value", "")),
                    )
                )
        for key, item in value.items():
            if key != "metrics":
                specs.update(collect_metric_specs(item))
    elif isinstance(value, list):
        for item in value:
            specs.update(collect_metric_specs(item))
    return specs


def collect_value_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "key" and isinstance(item, str):
                keys.add(item)
            else:
                keys.update(collect_value_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(collect_value_keys(item))
    return keys


def collect_value_specs(value: Any) -> set[tuple[str, str, str]]:
    specs: set[tuple[str, str, str]] = set()
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, dict):
                    continue
                specs.add(
                    (
                        str(item.get("path", "")),
                        str(item.get("key", "")),
                        str(item.get("value", "")),
                    )
                )
        for key, item in value.items():
            if key != "values":
                specs.update(collect_value_specs(item))
    elif isinstance(value, list):
        for item in value:
            specs.update(collect_value_specs(item))
    return specs


def collect_file_sha256_specs(value: Any) -> set[tuple[str, str, str]]:
    specs: set[tuple[str, str, str]] = set()
    if isinstance(value, dict):
        checks = value.get("file_sha256_checks")
        if isinstance(checks, list):
            for item in checks:
                if not isinstance(item, dict):
                    continue
                specs.add(
                    (
                        str(item.get("json_path", "")),
                        str(item.get("json_key", "")),
                        str(item.get("file_path", "")),
                    )
                )
        for key, item in value.items():
            if key != "file_sha256_checks":
                specs.update(collect_file_sha256_specs(item))
    elif isinstance(value, list):
        for item in value:
            specs.update(collect_file_sha256_specs(item))
    return specs


def collect_value_comparison_keys(value: Any) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        if isinstance(value.get("left_key"), str) and isinstance(value.get("right_key"), str):
            keys.add((value["left_key"], value["right_key"]))
        for item in value.values():
            keys.update(collect_value_comparison_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(collect_value_comparison_keys(item))
    return keys


def collect_table_value_specs(value: Any) -> set[tuple[str, str, str, str]]:
    specs: set[tuple[str, str, str, str]] = set()
    if isinstance(value, dict):
        table_values = value.get("table_values")
        if isinstance(table_values, list):
            for item in table_values:
                if not isinstance(item, dict):
                    continue
                specs.add(
                    (
                        str(item.get("path", "")),
                        str(item.get("row_value", "")),
                        str(item.get("key", "")),
                        str(item.get("value", "")),
                    )
                )
        for key, item in value.items():
            if key != "table_values":
                specs.update(collect_table_value_specs(item))
    elif isinstance(value, list):
        for item in value:
            specs.update(collect_table_value_specs(item))
    return specs


def assert_pipeline_budget_contracts(testcase: unittest.TestCase, pipeline: dict[str, Any]) -> None:
    budget_by_output = {
        stage.get("args", {}).get("output"): stage
        for stage in pipeline.get("stages", [])
        if stage.get("type") == "api_budget"
    }
    api_stage_types = {
        "generate_model_rubrics",
        "generate_teachers",
        "filter_verifier",
        "downstream",
        "multicandidate_downstream",
    }
    checked = 0
    for stage in pipeline.get("stages", []):
        stage_type = stage.get("type")
        if stage_type not in api_stage_types:
            continue
        args = stage.get("args", {})
        report_path = args.get("require_budget_report")
        testcase.assertTrue(report_path, f"{stage['name']} missing require_budget_report")
        budget = budget_by_output.get(report_path)
        testcase.assertIsNotNone(budget, f"{stage['name']} has no matching api_budget stage")
        if budget is None:
            continue
        budget_args = budget.get("args", {})
        testcase.assertEqual(budget_args.get("input"), args.get("input"), stage["name"])
        testcase.assertEqual(budget_args.get("providers"), args.get("providers") or args.get("provider"), stage["name"])

        if stage_type in {"generate_model_rubrics", "generate_teachers"}:
            testcase.assertNotIn("unit_field", budget_args, stage["name"])
            testcase.assertEqual(budget_args.get("calls_per_record_per_provider", 1), 1, stage["name"])
        elif stage_type == "filter_verifier":
            testcase.assertEqual(budget_args.get("unit_field"), "rubrics", stage["name"])
            testcase.assertEqual(budget_args.get("calls_per_record_per_provider"), 1, stage["name"])
        elif stage_type == "downstream":
            testcase.assertEqual(budget_args.get("unit_field"), "rubrics", stage["name"])
            testcase.assertNotIn("unit_multiplier_field", budget_args, stage["name"])
            testcase.assertEqual(budget_args.get("calls_per_record_per_provider"), 2, stage["name"])
        elif stage_type == "multicandidate_downstream":
            testcase.assertEqual(budget_args.get("unit_field"), "rubrics", stage["name"])
            testcase.assertEqual(budget_args.get("unit_multiplier_field"), "candidates", stage["name"])
            testcase.assertEqual(budget_args.get("calls_per_record_per_provider"), 1, stage["name"])
        checked += 1
    testcase.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
