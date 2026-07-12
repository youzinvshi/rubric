from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_submission_gap_report import build_gap_report, load_json, to_markdown


class BuildSubmissionGapReportTest(unittest.TestCase):
    def test_load_json_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_json(Path("/tmp/missing_submission_gap_report_input.json"))

        self.assertIn("Required JSON file is missing", str(context.exception))

    def test_groups_required_evidence_by_aaai_evidence_phase(self) -> None:
        readiness = {
            "ok": False,
            "hard_blockers": [
                "required evidence claim C0 is missing_evidence",
                "required evidence claim C1 is missing_evidence",
                "required evidence claim C13 is missing_evidence",
                "required evidence claim C4 is missing_evidence",
                "required evidence claim C9 is missing_evidence",
                "required evidence claim C10 is missing_evidence",
                "required paper figures not synced: figures/semantic_space.pdf, figures/semantic_space.svg",
                "required paper tables not synced: tables/downstream_utility_table.tex",
            ],
            "warnings": ["some claims still have missing evidence"],
        }
        evidence = [
            {
                "claim_id": "C0",
                "status": "missing_evidence",
                "claim": "Hard-gold and downstream holdouts are clean.",
                "evidence": "[missing] Hard-gold contamination audit",
            },
            {
                "claim_id": "C1",
                "status": "missing_evidence",
                "claim": "Base hard-gold BSC evidence is available.",
                "evidence": "[missing] Base BSC summary",
            },
            {
                "claim_id": "C13",
                "status": "missing_evidence",
                "claim": "Semantic-space visualization assets are available for auditing coverage regions.",
                "evidence": "[missing] Semantic-space SVG (outputs/matrix_real/semantic_space/semantic_space.svg)",
            },
            {
                "claim_id": "C4",
                "status": "missing_evidence",
                "claim": "BSC-optimized evaluation criteria preserve downstream utility.",
                "evidence": "[missing] RewardBench downstream summary",
            },
            {
                "claim_id": "C9",
                "status": "missing_evidence",
                "claim": "JudgeBench downstream utility is reportable.",
                "evidence": "[missing] JudgeBench downstream summary",
            },
            {
                "claim_id": "C10",
                "status": "missing_evidence",
                "claim": "RewardBench-2 downstream utility is reportable.",
                "evidence": "[missing] RewardBench-2 downstream summary",
            },
        ]

        report = build_gap_report(readiness, evidence)
        phases = {phase["id"]: phase for phase in report["phases"]}

        self.assertFalse(report["ok"])
        self.assertEqual(phases["semantic_space"]["claim_ids"], ["C13"])
        self.assertEqual(phases["downstream_utility"]["claim_ids"], ["C10", "C4", "C9"])
        self.assertTrue(any("C13" in item for item in phases["semantic_space"]["readiness_blockers"]))
        self.assertTrue(any("C4" in item for item in phases["downstream_utility"]["readiness_blockers"]))
        self.assertTrue(any("semantic_space.pdf" in item for item in phases["semantic_space"]["readiness_blockers"]))
        self.assertTrue(any("downstream_utility_table" in item for item in phases["downstream_utility"]["readiness_blockers"]))
        self.assertEqual(phases["data_isolation_hard_gold"]["blocked_by_prior_phases"], [])
        self.assertIn("data_isolation_hard_gold", phases["downstream_utility"]["blocked_by_prior_phases"])
        self.assertIn("main_hard_gold_bsc", phases["semantic_space"]["blocked_by_prior_phases"])
        self.assertIn("claim_gaps=1", phases["semantic_space"]["summary"])
        self.assertIn("readiness_blockers=2", phases["semantic_space"]["summary"])
        self.assertIn("blocked_by_prior_phases=2", phases["semantic_space"]["summary"])

    def test_gap_report_exposes_claim_ladder_status(self) -> None:
        report = build_gap_report(
            {"ok": False, "hard_blockers": [], "warnings": []},
            [
                {"claim_id": "C1", "status": "safe_to_claim"},
                {"claim_id": "C6", "status": "safe_to_claim"},
                {"claim_id": "C0", "status": "missing_evidence"},
                {"claim_id": "C2", "status": "safe_to_claim"},
                {"claim_id": "C3", "status": "safe_to_claim"},
                {"claim_id": "C5", "status": "contradicted"},
                {"claim_id": "C7", "status": "safe_to_claim"},
                {"claim_id": "C14", "status": "safe_to_claim"},
            ],
        )
        by_level = {row["level"]: row for row in report["claim_ladder"]}
        text = to_markdown(report)

        self.assertEqual(by_level["motivation"]["status"], "safe_to_claim")
        self.assertEqual(by_level["metric-support"]["status"], "missing_evidence")
        self.assertIn("C0: missing_evidence", by_level["metric-support"]["missing_or_non_safe_claims"])
        self.assertEqual(by_level["method-support"]["status"], "blocked")
        self.assertIn("C5: contradicted", by_level["method-support"]["missing_or_non_safe_claims"])
        self.assertIn("## Claim Ladder Status", text)
        self.assertIn("| motivation | `safe_to_claim` | C1, C6 | none |", text)
        self.assertIn("| method-support | `blocked` | C5, C7, C14 | C5: contradicted |", text)

    def test_c0_stale_provenance_gets_operator_action_chain(self) -> None:
        report = build_gap_report(
            {"ok": False, "hard_blockers": ["required evidence claim C0 is missing_evidence"], "warnings": []},
            [
                {
                    "claim_id": "C0",
                    "status": "missing_evidence",
                    "claim": "Training artifacts are query-disjoint from held-out evaluations.",
                    "evidence": (
                        "[missing] Teacher verifier report binds raw teacher input "
                        "(outputs/verifier/teacher_rubrics_filtered_report.json); "
                        "[missing] Proxy-gold build report writes unfiltered SFT data "
                        "(sft_output=data/processed/blindspot_sft.jsonl == "
                        "data/processed/blindspot_sft.unfiltered.jsonl); "
                        "[missing] Proxy-gold build report writes unfiltered proxy-gold data "
                        "(proxy_gold_output=data/processed/proxy_gold.jsonl == "
                        "data/processed/proxy_gold.unfiltered.jsonl)"
                    ),
                }
            ],
        )

        handoff = report["operator_handoff"]
        actions = handoff["c0_provenance_action_chain"]
        text = to_markdown(report)

        self.assertEqual(handoff["training_artifacts"], [])
        self.assertEqual(handoff["training_data_chain"], [])
        self.assertEqual([item["stage"] for item in actions], [
            "sft_data_preflight_and_teacher_budgets",
            "generate_teacher_rubrics",
            "api_budget_meta_verifier",
            "filter_teacher_rubrics",
            "build_sft_all_domains",
            "convert_proxy_gold_to_verl_and_audit_holdouts",
            "refresh_evidence_and_reports",
        ])
        self.assertIn("outputs/preflight/sft_data_preflight.json", actions[0]["outputs"])
        self.assertIn("data/processed/teacher_rubrics_raw.jsonl", actions[1]["outputs"])
        self.assertIn("outputs/api_budget/meta_verifier_budget.json", actions[2]["outputs"])
        self.assertIn("outputs/verifier/teacher_rubrics_filtered_report.json", actions[3]["outputs"])
        self.assertIn("data/processed/blindspot_sft.unfiltered.jsonl", actions[4]["outputs"])
        self.assertIn("data/processed/proxy_gold.jsonl", actions[4]["outputs"])
        self.assertIn("outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json", actions[5]["outputs"])
        self.assertIn("### C0 Provenance Action Chain", text)
        self.assertIn("--from-stage api_budget_teacher_rubrics --to-stage sft_data_preflight", text)
        self.assertIn("--from-stage generate_teacher_rubrics_rubricbench --to-stage generate_teacher_rubrics_writingbench", text)
        self.assertIn("--from-stage api_budget_meta_verifier --to-stage api_budget_meta_verifier", text)
        self.assertIn("--from-stage build_sft_all_domains --to-stage filter_proxy_gold_rewardbench2_downstream_overlap", text)

    def test_maps_asset_index_blocker_summary_to_evidence_phases(self) -> None:
        readiness = {
            "ok": False,
            "hard_blockers": [],
            "warnings": [],
            "paper": {
                "asset_index": [
                    {
                        "blocker_summary": [
                            {
                                "category": "main_bsc_table",
                                "count": 1,
                                "label": "main hard-gold BSC paper table",
                            },
                            {
                                "category": "downstream_utility",
                                "count": 1,
                                "label": "RewardBench/JudgeBench/RewardBench-2 utility table",
                            },
                            {
                                "category": "reward_ablation",
                                "count": 1,
                                "label": "reward-component ablation table",
                            },
                            {
                                "category": "dimension_transition",
                                "count": 1,
                                "label": "dimension-transition audit paper table",
                            },
                            {
                                "category": "semantic_space",
                                "count": 4,
                                "label": "semantic-space SVG/PDF/CSV/JSON assets",
                            },
                            {
                                "category": "experiment_summary",
                                "count": 1,
                                "label": "paper-facing experiment summary",
                            },
                        ]
                    },
                    {
                        "warning_summary": [
                            {
                                "category": "api_handoff",
                                "count": 2,
                                "label": "API handoff reviewer-facing docs",
                                "warnings": [
                                    "asset index declares warning: artifact is missing or empty: outputs/paper_artifacts/api_handoff.json",
                                    "asset index declares warning: artifact is missing or empty: outputs/paper_artifacts/api_handoff.md",
                                ],
                            },
                            {
                                "category": "audit_report",
                                "count": 1,
                                "label": "matrix audit report",
                                "warnings": [
                                    "asset index declares warning: artifact is missing or empty: outputs/paper_artifacts/audit_report.json",
                                ],
                            },
                        ]
                    }
                ]
            },
        }

        report = build_gap_report(readiness, [])
        phases = {phase["id"]: phase for phase in report["phases"]}
        text = to_markdown(report)

        self.assertEqual(phases["downstream_utility"]["status"], "blocked")
        self.assertEqual(phases["main_hard_gold_bsc"]["paper_asset_blockers"][0]["evidence_gate"], "C1/C2/C3/C14")
        self.assertEqual(phases["main_hard_gold_bsc"]["paper_asset_blockers"][1]["evidence_gate"], "C12")
        self.assertEqual(phases["downstream_utility"]["paper_asset_blockers"][0]["evidence_gate"], "C4/C9/C10")
        self.assertEqual(
            phases["downstream_utility"]["paper_asset_blockers"][0]["source_artifacts"],
            [
                "outputs/matrix_real/main_table.csv",
                "outputs/matrix_judgebench/main_table.csv",
                "outputs/matrix_rewardbench2/main_table.csv",
            ],
        )
        self.assertIn(
            "scripts/export_paper_artifacts.py",
            phases["downstream_utility"]["paper_asset_blockers"][0]["producer"],
        )
        self.assertEqual(phases["ablations"]["paper_asset_blockers"][0]["evidence_gate"], "C7")
        self.assertEqual(phases["semantic_space"]["paper_asset_blockers"][0]["evidence_gate"], "C13")
        self.assertIn(
            "scripts/build_semantic_space_visualization.py",
            phases["semantic_space"]["paper_asset_blockers"][0]["producer"],
        )
        self.assertEqual(phases["paper_readiness"]["paper_asset_blockers"][0]["evidence_gate"], "C0-C14")
        self.assertEqual(phases["paper_readiness"]["paper_asset_warnings"][0]["category"], "api_handoff")
        self.assertEqual(phases["paper_readiness"]["paper_asset_warnings"][1]["category"], "audit_report")
        self.assertIn("paper_asset_warnings=2", phases["paper_readiness"]["summary"])
        self.assertIn("### Paper Asset Blockers", text)
        self.assertIn("### Paper Asset Warnings", text)
        self.assertIn("- Summary: blocked:", text)
        self.assertIn("`main_bsc_table`: 1", text)
        self.assertIn("`dimension_transition`: 1", text)
        self.assertIn("`semantic_space`: 4", text)
        self.assertIn("`experiment_summary`: 1", text)
        self.assertIn("`api_handoff`: 2", text)
        self.assertIn("`audit_report`: 1", text)
        self.assertIn("Source artifacts", text)
        self.assertIn("outputs/matrix_real/semantic_space/semantic_space_summary.json", text)

    def test_to_markdown_lists_next_actions_without_generator_framing(self) -> None:
        report = build_gap_report(
            {
                "ok": False,
                "hard_blockers": [
                    "required evidence claim C0 is missing_evidence",
                    "required evidence claim C7 is missing_evidence",
                ],
                "warnings": [],
            },
            [
                {
                    "claim_id": "C0",
                    "status": "missing_evidence",
                    "claim": "Hard-gold holdouts are clean.",
                    "evidence": "[missing] Hard-gold contamination audit",
                },
                {
                    "claim_id": "C7",
                    "status": "missing_evidence",
                    "claim": "Ablations report reward-component variants.",
                    "evidence": "[missing] Reward ablation table",
                }
            ],
        )

        text = to_markdown(report)

        self.assertIn("Ablations", text)
        self.assertIn("no redundancy", text)
        self.assertIn("SFT-only versus SFT+GRPO", text)
        self.assertIn("Depends on: data_isolation_hard_gold, main_hard_gold_bsc, training_and_serving", text)
        self.assertIn("Blocked by prior phases: data_isolation_hard_gold", text)
        self.assertIn("### Claim Discipline", text)
        self.assertIn("C0 is safe_to_claim with query-disjoint holdouts", text)
        self.assertIn("A BSC coverage change is only metric evidence", text)
        self.assertIn("C12/C14 audit dimension-level recovery over SFT-only", text)
        self.assertIn("dimension-level recovery remains a permitted conclusion only after those gates pass", text)
        self.assertIn("Do not describe BSC coverage changes as supporting judge utility until C4/C9/C10 pass", text)
        self.assertIn("requires separately trained variants", text)
        self.assertIn("Treat the plot as illustrative until C13 verifies point-level provenance", text)
        self.assertIn("write empirical claims only when the relevant evidence rows are safe_to_claim", text)
        self.assertNotIn("rubric generator", text.lower())
        self.assertNotIn("rubric generation", text.lower())

    def test_phase_claim_discipline_is_json_serialized(self) -> None:
        report = build_gap_report(
            {
                "ok": False,
                "hard_blockers": ["required evidence claim C4 is missing_evidence"],
                "warnings": [],
            },
            [
                {
                    "claim_id": "C4",
                    "status": "missing_evidence",
                    "claim": "Downstream utility is reportable.",
                    "evidence": "[missing] RewardBench downstream summary",
                }
            ],
        )
        phases = {phase["id"]: phase for phase in report["phases"]}

        self.assertIn("claim_discipline", phases["downstream_utility"])
        self.assertIn("API/model scorer outputs", phases["downstream_utility"]["claim_discipline"])
        self.assertIn("paper_claim_eligible summaries", phases["downstream_utility"]["claim_discipline"])

    def test_reviewer_response_phase_tracks_rebuttal_manifest_gaps(self) -> None:
        report = build_gap_report(
            {"ok": False, "hard_blockers": [], "warnings": []},
            [],
            rebuttal_manifest={
                "schema_version": 1,
                "entry_count": 2,
                "defense_status_counts": {"needs_evidence": 1, "needs_readiness": 1},
                "readiness_ok": False,
                "concern_templates": {
                    "source": "DEFAULT_CONCERNS",
                    "count": 2,
                    "sha256": "abc123",
                },
            },
        )
        phases = {phase["id"]: phase for phase in report["phases"]}
        text = to_markdown(report)

        self.assertEqual(phases["reviewer_response_readiness"]["status"], "blocked")
        self.assertIn(
            "rebuttal pack was built while submission readiness was false",
            phases["reviewer_response_readiness"]["manifest_gaps"],
        )
        self.assertIn(
            "rebuttal pack has no answer_ready entries",
            phases["reviewer_response_readiness"]["manifest_gaps"],
        )
        self.assertIn(
            "rebuttal pack has entries waiting for submission readiness",
            phases["reviewer_response_readiness"]["manifest_gaps"],
        )
        self.assertIn("Reviewer-Facing Rebuttal Readiness", text)
        self.assertIn("Use only answer_ready rebuttal entries whose matched Evidence Matrix rows are safe_to_claim and whose readiness_ok flag is true", text)
        self.assertIn("readiness_ok is true, and relevant entries are answer_ready", text)
        self.assertIn("### Manifest Gaps", text)

    def test_cli_input_shape_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = root / "readiness.json"
            evidence = root / "evidence.json"
            readiness.write_text(json.dumps({"ok": True, "hard_blockers": [], "warnings": []}), encoding="utf-8")
            evidence.write_text(json.dumps([]), encoding="utf-8")

            report = build_gap_report(load_json(readiness), load_json(evidence))

        self.assertTrue(report["ok"])
        self.assertEqual(report["hard_blocker_count"], 0)

    def test_execution_sequence_uses_dedicated_matrix_pipelines(self) -> None:
        report = build_gap_report({"ok": False, "hard_blockers": [], "warnings": []}, [])
        sequence = {item["id"]: item for item in report["execution_sequence"]}
        text = to_markdown(report)

        self.assertIn(
            "configs/pipeline_matrix_real.generated.json",
            sequence["main_matrix_and_ablations"]["commands"][0],
        )
        self.assertEqual(
            sequence["downstream_matrices"]["commands"],
            [
                "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_judgebench.generated.json",
                "python3 scripts/run_experiment_pipeline.py --config configs/pipeline_matrix_rewardbench2.generated.json",
            ],
        )
        self.assertNotIn("--from-stage judgebench_base_prepare_downstream --to-stage export", text)
        self.assertIn("## Execution Sequence", text)

    def test_execution_sequence_summarizes_missing_preflight_prerequisites(self) -> None:
        report = build_gap_report(
            {
                "ok": False,
                "hard_blockers": [
                    "required paper tables not synced: tables/rl_stage_ablation_table.tex",
                    "blindspot_sft: missing or empty file: data/processed/blindspot_sft.jsonl",
                ],
                "warnings": [],
            },
            [],
            preflight_reports=[
                {
                    "hard_blockers": [
                        "missing input file: data/processed/rmbench_queries.jsonl",
                        "missing API env GPT_AK_1 for provider gpt-5.4",
                        "missing API env GPT_AK_1 for provider gpt-5.4",
                        "missing provider config: configs/judge_scorer.local.jsonl",
                        "missing required provider: judge-scorer",
                        "missing required provider in configs/generators.local.jsonl: sft_rl",
                        "missing required env var: OPENAI_API_KEY",
                    ],
                    "blockers": [],
                    "providers": [
                        {
                            "path": "configs/generators.local.jsonl",
                            "providers": [
                                {"name": "gpt-5.4", "api_key_env": "GPT_AK_1", "api_key_present": False},
                                {"name": "sft_rl", "api_key_env": "LOCAL_OPENAI_API_KEY", "api_key_present": True},
                            ],
                        },
                        {
                            "path": "configs/verifier.local.jsonl",
                            "providers": [
                                {"name": "meta-verifier", "api_key_env": "GPT_AK_3", "api_key_present": False},
                            ],
                        },
                    ],
                }
            ],
            gate_reports=[
                {
                    "blockers": [
                        "blindspot_sft: missing or empty file: data/processed/blindspot_sft.jsonl",
                        "proxy_gold: missing or empty file: data/processed/proxy_gold.jsonl",
                        "proxy_gold_verl: missing or empty file: data/processed/proxy_gold_verl.parquet",
                    ]
                }
            ],
        )
        sequence = {item["id"]: item for item in report["execution_sequence"]}
        handoff = report["operator_handoff"]
        text = to_markdown(report)

        self.assertEqual(report["missing_prerequisites"]["input_files"], ["data/processed/rmbench_queries.jsonl"])
        self.assertEqual(report["missing_prerequisites"]["api_env"], ["GPT_AK_1 for provider gpt-5.4"])
        self.assertIn("data/processed/blindspot_sft.jsonl", sequence["data_isolation"]["missing_prerequisites"]["training_artifacts"])
        self.assertIn("data/processed/proxy_gold.jsonl", report["missing_prerequisites"]["training_artifacts"])
        self.assertIn("data/processed/proxy_gold_verl.parquet", report["missing_prerequisites"]["training_artifacts"])
        self.assertIn("evidence_gates=C1,C2,C3,C5,C6,C7,C12,C13,C14", sequence["main_matrix_and_ablations"]["summary"])
        self.assertIn("blocked_by_prior_phases=1", sequence["main_matrix_and_ablations"]["summary"])
        self.assertIn("missing_prerequisites=3", sequence["main_matrix_and_ablations"]["summary"])
        self.assertIn("missing_prerequisites=1", sequence["paper_export_and_readiness"]["summary"])
        self.assertIn("OPENAI_API_KEY", sequence["training_and_serving"]["missing_prerequisites"]["required_env"])
        self.assertIn(
            "configs/judge_scorer.local.jsonl",
            sequence["model_evaluation_criteria_generation"]["missing_prerequisites"]["provider_configs"],
        )
        self.assertNotIn("model_rubric_generation", sequence)
        self.assertIn("tables/rl_stage_ablation_table.tex", sequence["paper_export_and_readiness"]["missing_prerequisites"]["paper_artifacts"])
        self.assertEqual(handoff["status"], "ready_for_operator_input")
        self.assertIn("export GPT_AK_1=<REDACTED>", handoff["redacted_env_exports"])
        self.assertIn("export OPENAI_API_KEY=<REDACTED>", handoff["redacted_env_exports"])
        self.assertIn("configs/judge_scorer.local.jsonl", handoff["provider_configs"])
        self.assertIn("data/processed/proxy_gold_verl.parquet", handoff["training_artifacts"])
        self.assertEqual(len(handoff["training_data_chain"]), 8)
        self.assertEqual(handoff["training_data_chain"][0]["stage"], "sft_data_preflight")
        self.assertEqual(handoff["training_data_chain"][-1]["stage"], "audit_*_holdout_contamination")
        self.assertIn("overlap_query_count == 0", handoff["training_data_chain"][-1]["checks"])
        self.assertEqual(
            handoff["training_artifact_producers"]["data/processed/blindspot_sft.unfiltered.jsonl"]["stage"],
            "build_sft_all_domains",
        )
        self.assertEqual(
            handoff["training_artifact_producers"]["data/processed/blindspot_sft.jsonl"]["stage"],
            "filter_blindspot_sft_rewardbench2_downstream_overlap",
        )
        self.assertEqual(
            handoff["training_artifact_producers"]["data/processed/proxy_gold.jsonl"]["output_report"],
            "outputs/contamination_audit/proxy_gold_rewardbench2_holdout_filter.json",
        )
        self.assertEqual(
            handoff["training_artifact_producers"]["data/processed/proxy_gold_verl.parquet"]["output_report"],
            "outputs/sft_data/proxy_gold_verl_report.json",
        )
        self.assertIn(
            "outputs/sft_data/proxy_gold_build_report.json",
            handoff["training_artifact_producers"]["data/processed/proxy_gold_verl.parquet"]["upstream_inputs"],
        )
        self.assertEqual(handoff["provider_entries_by_file"]["configs/generators.local.jsonl"], ["sft_rl"])
        self.assertEqual(handoff["api_env_by_provider"]["gpt-5.4"], "GPT_AK_1")
        self.assertEqual(handoff["api_env_by_provider"]["meta-verifier"], "GPT_AK_3")
        self.assertEqual(handoff["api_env_by_file"]["configs/verifier.local.jsonl"], ["meta-verifier: GPT_AK_3"])
        self.assertIn(
            "test -f configs/judge_scorer.local.jsonl || cp configs/judge_scorer.example.jsonl configs/judge_scorer.local.jsonl",
            handoff["provider_template_commands"],
        )
        self.assertIn(
            "test -f configs/generators.local.jsonl || cp configs/generators.example.jsonl configs/generators.local.jsonl",
            handoff["provider_template_commands"],
        )
        self.assertIn(
            "test -f configs/verifier.local.jsonl || cp configs/verifier.example.jsonl configs/verifier.local.jsonl",
            handoff["provider_template_commands"],
        )
        self.assertIn("Do not commit local provider files or API key values.", handoff["safety_notes"])
        self.assertIn("- Summary: phase_status=blocked;", text)
        self.assertNotIn("sk-", text)
        self.assertIn("Missing prerequisites", text)
        self.assertIn("## Operator Handoff", text)
        self.assertIn("### Provider Template Commands", text)
        self.assertIn("### Provider Entries By File", text)
        self.assertIn("### API Env By Provider", text)
        self.assertIn("### Training Data Chain", text)
        self.assertIn("sft_data_preflight", text)
        self.assertIn("filter_teacher_rubrics", text)
        self.assertIn("overlap_query_count == 0", text)
        self.assertIn("### Training Artifact Producers", text)
        self.assertIn("build_sft_all_domains", text)
        self.assertIn("convert_proxy_gold_to_verl", text)
        self.assertIn("outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet", text)
        self.assertIn("meta-verifier", text)
        self.assertIn("export GPT_AK_1=<REDACTED>", text)
        self.assertIn("configs/generators.local.jsonl: sft_rl", text)


if __name__ == "__main__":
    unittest.main()
