from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

from scripts.run_experiment_pipeline import build_command, format_command, load_pipeline_config, require_ready_handoff, select_stages


class RunExperimentPipelineTest(unittest.TestCase):
    def test_load_pipeline_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_pipeline_config(Path("/tmp/missing_blindspot_pipeline.json"))

        self.assertIn("Pipeline config is missing", str(context.exception))

    def test_load_pipeline_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_pipeline_config(path)

        self.assertIn("Pipeline config is not valid JSON", str(context.exception))

    def test_load_pipeline_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_pipeline_config(path)

        self.assertIn("Pipeline config must be a JSON object", str(context.exception))

    def test_build_command_converts_args(self) -> None:
        cmd = build_command(
            {
                "name": "bsc",
                "type": "bsc",
                "args": {
                    "input": "x.jsonl",
                    "output_dir": "out",
                    "dedupe_query": True,
                    "skip": False,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("bsc_diagnose.py", joined)
        self.assertIn("--output-dir out", joined)
        self.assertIn("--dedupe-query", joined)
        self.assertNotIn("--skip", joined)

    def test_build_command_supports_non_strict_audit_stage(self) -> None:
        cmd = build_command(
            {
                "name": "audit",
                "type": "audit",
                "args": {
                    "manifest": "configs/manifest.json",
                    "output": "outputs/audit.json",
                    "non_strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("audit_experiment.py", joined)
        self.assertIn("--non-strict", joined)

    def test_build_command_supports_split_dataset_stage(self) -> None:
        cmd = build_command(
            {
                "name": "split_rubricbench",
                "type": "split_dataset",
                "args": {
                    "input": "data/processed/rubricbench_gold.jsonl",
                    "split": ["train_seed:50", "dev:20", "test_main:rest"],
                    "output_dir": "data/processed/splits",
                    "output_prefix": "rubricbench_gold_",
                    "main_eval_split": ["test_main"],
                    "manifest": "outputs/data_splits/rubricbench_gold_split.json",
                },
            }
        )
        joined = " ".join(cmd)

        self.assertIn("split_dataset.py", joined)
        self.assertIn("--split train_seed:50", joined)
        self.assertIn("--split test_main:rest", joined)
        self.assertIn("--main-eval-split test_main", joined)

    def test_select_stages_can_slice_pipeline_by_stage_names(self) -> None:
        stages = [
            {"name": "preflight", "type": "preflight", "args": {}},
            {"name": "api_budget", "type": "api_budget", "args": {}},
            {"name": "generate", "type": "generate_model_rubrics", "args": {}},
            {"name": "bsc", "type": "bsc", "args": {}},
            {"name": "result_card", "type": "result_card", "args": {}},
        ]

        selected = select_stages(stages, from_stage="generate", to_stage="result_card")

        self.assertEqual([stage["name"] for stage in selected], ["generate", "bsc", "result_card"])

    def test_select_stages_filters_only_within_requested_range(self) -> None:
        stages = [
            {"name": "preflight", "type": "preflight", "args": {}},
            {"name": "api_budget", "type": "api_budget", "args": {}},
            {"name": "generate", "type": "generate_model_rubrics", "args": {}},
            {"name": "bsc", "type": "bsc", "args": {}},
            {"name": "result_card", "type": "result_card", "args": {}},
        ]

        selected = select_stages(
            stages,
            only=["preflight", "generate", "result_card"],
            from_stage="api_budget",
            to_stage="result_card",
        )

        self.assertEqual([stage["name"] for stage in selected], ["generate", "result_card"])

    def test_select_stages_reports_missing_or_reversed_stage_ranges(self) -> None:
        stages = [
            {"name": "preflight", "type": "preflight", "args": {}},
            {"name": "api_budget", "type": "api_budget", "args": {}},
            {"name": "generate", "type": "generate_model_rubrics", "args": {}},
        ]

        with self.assertRaises(SystemExit) as missing_from:
            select_stages(stages, from_stage="missing")
        self.assertIn("--from-stage was not found", str(missing_from.exception))

        with self.assertRaises(SystemExit) as missing_only:
            select_stages(stages, only=["missing"])
        self.assertIn("--only stage was not found", str(missing_only.exception))

        with self.assertRaises(SystemExit) as reversed_range:
            select_stages(stages, from_stage="generate", to_stage="preflight")
        self.assertIn("--from-stage must not come after --to-stage", str(reversed_range.exception))

    def test_require_ready_handoff_passes_only_ready_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "api_handoff.json"
            handoff.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "status": "ready_for_paid_run",
                        "blockers": [],
                        "resume_requirements": {"ready": True},
                    }
                ),
                encoding="utf-8",
            )

            require_ready_handoff(handoff)

    def test_require_ready_handoff_blocks_missing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "api_handoff.json"
            handoff.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "status": "blocked",
                        "blockers": ["preflight: ok=false"],
                        "resume_requirements": {
                            "ready": False,
                            "missing_env": ["LOCAL_OPENAI_API_KEY"],
                            "next_command": "python3 scripts/run_experiment_pipeline.py --only preflight",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as context:
                require_ready_handoff(handoff)

        message = str(context.exception)
        self.assertIn("Required API handoff is not ready", message)
        self.assertIn("missing_env=LOCAL_OPENAI_API_KEY", message)
        self.assertIn("next_command=python3 scripts/run_experiment_pipeline.py --only preflight", message)

    def test_build_command_repeats_list_args(self) -> None:
        cmd = build_command(
            {
                "name": "summarize",
                "type": "summarize",
                "args": {"bsc": ["a=x", "b=y"], "output_csv": "out.csv"},
            }
        )
        self.assertEqual(cmd.count("--bsc"), 2)

    def test_build_command_expands_sweep_grid_args(self) -> None:
        cmd = build_command(
            {
                "name": "bsc_sweep",
                "type": "bsc_sweep",
                "args": {
                    "coverage_tau": [0.5, 0.7],
                    "redundancy_tau": [0.8, 0.9],
                    "output_dir": "out",
                },
            }
        )
        self.assertEqual(cmd.count("--coverage-tau"), 1)
        self.assertEqual(cmd.count("--redundancy-tau"), 1)
        self.assertIn("0.5", cmd)
        self.assertIn("0.7", cmd)

    def test_build_command_supports_bootstrap_ci_stage(self) -> None:
        cmd = build_command(
            {
                "name": "ci",
                "type": "bootstrap_ci",
                "args": {
                    "input": "outputs/per_item.csv",
                    "metric": ["coverage", "blind"],
                    "n_boot": 100,
                    "output_json": "outputs/ci.json",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("bootstrap_metric_ci.py", joined)
        self.assertEqual(cmd.count("--metric"), 2)
        self.assertIn("--n-boot 100", joined)

    def test_build_command_supports_bsc_human_audit_pack_stage(self) -> None:
        cmd = build_command(
            {
                "name": "bsc_human_audit_pack",
                "type": "bsc_human_audit_pack",
                "args": {
                    "input": "data/bsc_eval.jsonl",
                    "output_dir": "outputs/audit_pack",
                    "embedding_model": "token-overlap",
                    "matched": 2,
                    "unmatched": 3,
                },
            }
        )
        joined = " ".join(cmd)

        self.assertIn("build_bsc_human_audit_pack.py", joined)
        self.assertIn("--output-dir", cmd)
        self.assertIn("outputs/audit_pack", cmd)

    def test_build_command_supports_bsc_human_audit_summary_stage(self) -> None:
        cmd = build_command(
            {
                "name": "bsc_human_audit_summary",
                "type": "bsc_human_audit_summary",
                "args": {
                    "input": "outputs/audit_pack/audit_items.csv",
                    "output_json": "outputs/audit_pack/human_label_summary.json",
                    "output_md": "outputs/audit_pack/human_label_summary.md",
                    "min_labeled": 50,
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)

        self.assertIn("summarize_bsc_human_audit_labels.py", joined)
        self.assertIn("--output-json", cmd)
        self.assertIn("--strict", cmd)

    def test_build_command_supports_blindspot_map_stage(self) -> None:
        cmd = build_command(
            {
                "name": "blindspot_map",
                "type": "blindspot_map",
                "args": {
                    "input": "data/bsc_eval.jsonl",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "coverage_tau": 0.75,
                    "model": "base",
                    "output_dir": "outputs/blindspot_map",
                },
            }
        )
        joined = " ".join(cmd)

        self.assertIn("blindspot_attribution.py", joined)
        self.assertIn("--input data/bsc_eval.jsonl", joined)
        self.assertIn("--coverage-tau 0.75", joined)
        self.assertIn("--output-dir outputs/blindspot_map", joined)

    def test_build_command_supports_budget_curve_stage(self) -> None:
        cmd = build_command(
            {
                "name": "budget_curve",
                "type": "budget_curve",
                "args": {
                    "input": "data/bsc_eval.jsonl",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "coverage_tau": 0.75,
                    "redundancy_tau": 0.85,
                    "k": [3, 5, 10],
                    "output_dir": "outputs/budget_curve",
                },
            }
        )
        joined = " ".join(cmd)

        self.assertIn("evaluate_budget_curve.py", joined)
        self.assertIn("--input data/bsc_eval.jsonl", joined)
        self.assertIn("--k 3 5 10", joined)
        self.assertIn("--output-dir outputs/budget_curve", joined)

    def test_build_command_supports_dimension_transition_stage(self) -> None:
        cmd = build_command(
            {
                "name": "transition",
                "type": "dimension_transition",
                "args": {
                    "baseline": "data/base.jsonl",
                    "candidate": "data/sft.jsonl",
                    "output_dir": "outputs/repair",
                    "embedding_model": "token-overlap",
                    "coverage_tau": 0.75,
                    "baseline_label": "base",
                    "candidate_label": "sft",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("evaluate_blindspot_repair.py", joined)
        self.assertIn("--baseline data/base.jsonl", joined)
        self.assertIn("--candidate data/sft.jsonl", joined)
        self.assertIn("--candidate-label sft", joined)

    def test_build_command_supports_semantic_space_stage(self) -> None:
        cmd = build_command(
            {
                "name": "semantic_space",
                "type": "semantic_space",
                "args": {
                    "input": ["base=data/base.jsonl", "sft=data/sft.jsonl"],
                    "join_report": ["base=outputs/base/bsc_join_report.json"],
                    "output_dir": "outputs/semantic",
                    "embedding_model": "token-overlap",
                    "projection": "umap",
                    "gold_cluster_tau": 0.75,
                    "max_points": 100,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_semantic_space_visualization.py", joined)
        self.assertEqual(cmd.count("--input"), 2)
        self.assertEqual(cmd.count("--join-report"), 1)
        self.assertIn("--output-dir outputs/semantic", joined)
        self.assertIn("--projection umap", joined)
        self.assertIn("--gold-cluster-tau 0.75", joined)
        self.assertIn("--max-points 100", joined)

    def test_build_command_supports_sync_paper_stage(self) -> None:
        cmd = build_command(
            {
                "name": "sync",
                "type": "sync_paper",
                "args": {
                    "artifacts_dir": "outputs/paper_artifacts",
                    "paper_dir": "paper",
                    "extra_doc": ["outputs/result_card/result_card.json", "outputs/result_card/result_card.md"],
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("sync_paper_artifacts.py", joined)
        self.assertIn("--artifacts-dir outputs/paper_artifacts", joined)
        self.assertEqual(cmd.count("--extra-doc"), 2)
        self.assertIn("--extra-doc outputs/result_card/result_card.json", joined)

    def test_build_command_supports_pipeline_specific_required_paper_files(self) -> None:
        cmd = build_command(
            {
                "name": "sync",
                "type": "sync_paper",
                "args": {
                    "artifacts_dir": "outputs/paper_artifacts",
                    "paper_dir": "paper",
                    "required_file": ["main_table.tex", "experiment_summary.md"],
                },
            }
        )
        joined = " ".join(cmd)
        self.assertEqual(cmd.count("--required-file"), 2)
        self.assertIn("--required-file main_table.tex", joined)
        self.assertIn("--required-file experiment_summary.md", joined)

    def test_build_command_supports_submission_readiness_stage(self) -> None:
        cmd = build_command(
            {
                "name": "readiness",
                "type": "submission_readiness",
                "args": {
                    "audit_report": "outputs/audit.json",
                    "evidence_matrix": "outputs/evidence.json",
                    "output_json": "outputs/readiness.json",
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("check_submission_readiness.py", joined)
        self.assertIn("--output-json outputs/readiness.json", joined)
        self.assertIn("--strict", joined)

    def test_build_command_supports_submission_gap_report_stage(self) -> None:
        cmd = build_command(
            {
                "name": "gap_report",
                "type": "submission_gap_report",
                "args": {
                    "readiness_report": "outputs/readiness.json",
                    "evidence_matrix": "outputs/evidence.json",
                    "rebuttal_manifest": "outputs/rebuttal/rebuttal_pack_manifest.json",
                    "output_dir": "outputs/gap_report",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_submission_gap_report.py", joined)
        self.assertIn("--rebuttal-manifest outputs/rebuttal/rebuttal_pack_manifest.json", joined)
        self.assertIn("--output-dir outputs/gap_report", joined)

    def test_build_command_supports_bsc_gold_sanity_stage(self) -> None:
        cmd = build_command(
            {
                "name": "bsc_gold_sanity",
                "type": "bsc_gold_sanity",
                "args": {
                    "gold": "data/processed/rubricbench_gold.jsonl",
                    "output_dir": "outputs/bsc_gold_sanity",
                    "min_joined": 1147,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("run_bsc_gold_sanity.py", joined)
        self.assertIn("--min-joined 1147", joined)

    def test_build_command_supports_minimal_bsc_chain_smoke_stage(self) -> None:
        cmd = build_command(
            {
                "name": "minimal_bsc_chain_smoke",
                "type": "minimal_bsc_chain_smoke",
                "args": {
                    "gold": "data/processed/rubricbench_gold.jsonl",
                    "output_dir": "outputs/minimal_bsc_chain_smoke",
                    "limit": 100,
                    "min_joined": 100,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("run_minimal_bsc_chain_smoke.py", joined)
        self.assertIn("--limit 100", joined)
        self.assertIn("--min-joined 100", joined)

    def test_build_command_supports_minimal_api_handoff_stage(self) -> None:
        cmd = build_command(
            {
                "name": "minimal_api_handoff",
                "type": "minimal_api_handoff",
                "args": {
                    "pipeline": "configs/pipeline_minimal_claim.generated.json",
                    "preflight": "outputs/minimal_claim/base/preflight/preflight.json",
                    "api_budget": "outputs/minimal_claim/base/api_budget/model_rubrics_budget.json",
                    "bsc_gold_sanity": "outputs/minimal_claim/base/bsc_gold_sanity/summary.json",
                    "output_json": "outputs/minimal_claim/base/handoff/api_handoff.json",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_minimal_api_handoff.py", joined)
        self.assertIn("--api-budget outputs/minimal_claim/base/api_budget/model_rubrics_budget.json", joined)
        self.assertIn("--bsc-gold-sanity outputs/minimal_claim/base/bsc_gold_sanity/summary.json", joined)

    def test_build_command_supports_paper_asset_index_check_stage(self) -> None:
        cmd = build_command(
            {
                "name": "paper_asset_index_check",
                "type": "paper_asset_index_check",
                "args": {
                    "asset_index": "paper/asset_index.md",
                    "output": "outputs/minimal_claim/base/paper_artifacts/paper_asset_index_check.json",
                    "output_md": "outputs/minimal_claim/base/paper_artifacts/paper_asset_index_check.md",
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("check_paper_asset_index.py", joined)
        self.assertIn("--asset-index paper/asset_index.md", joined)
        self.assertIn("--output-md outputs/minimal_claim/base/paper_artifacts/paper_asset_index_check.md", joined)
        self.assertIn("--strict", joined)

    def test_build_command_supports_latex_compile_check_stage(self) -> None:
        cmd = build_command(
            {
                "name": "latex_compile_check",
                "type": "latex_compile_check",
                "args": {
                    "paper_dir": "paper",
                    "output_json": "outputs/submission_readiness/latex_compile_report.json",
                    "output_md": "outputs/submission_readiness/latex_compile_report.md",
                    "compile": True,
                    "require_official_style": True,
                    "require_anonymous": True,
                    "max_pages": 8,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("check_latex_compile.py", joined)
        self.assertIn("--paper-dir paper", joined)
        self.assertIn("--output-json outputs/submission_readiness/latex_compile_report.json", joined)
        self.assertIn("--compile", joined)
        self.assertIn("--require-official-style", joined)
        self.assertIn("--require-anonymous", joined)
        self.assertIn("--max-pages 8", joined)

    def test_format_command_quotes_raw_gate_values_for_copyable_dry_run(self) -> None:
        cmd = build_command(
            {
                "name": "readiness",
                "type": "submission_readiness",
                "args": {
                    "raw_gate": [
                        "Data Source Report|data_source_report|outputs/data_sources/source_report.json",
                    ],
                    "output_json": "outputs/readiness.json",
                },
            }
        )
        formatted = format_command(cmd)

        self.assertIn(
            "'Data Source Report|data_source_report|outputs/data_sources/source_report.json'",
            formatted,
        )
        self.assertNotIn("--raw-gate Data Source Report|data_source_report", formatted)

    def test_build_command_supports_sprint_plan_stage(self) -> None:
        cmd = build_command(
            {
                "name": "sprint",
                "type": "sprint_plan",
                "args": {"config": "configs/sprint.json", "output_dir": "outputs/sprint"},
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_sprint_plan.py", joined)
        self.assertIn("--output-dir outputs/sprint", joined)

    def test_build_command_supports_dashboard_stage(self) -> None:
        cmd = build_command(
            {
                "name": "dashboard",
                "type": "dashboard",
                "args": {
                    "config": "configs/dashboard.json",
                    "output_json": "outputs/dashboard.json",
                    "output_md": "outputs/dashboard.md",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_run_dashboard.py", joined)
        self.assertIn("--output-json outputs/dashboard.json", joined)

    def test_build_command_supports_result_card_stage(self) -> None:
        cmd = build_command(
            {
                "name": "result_card",
                "type": "result_card",
                "args": {
                    "config": "configs/result_card.json",
                    "output_json": "outputs/result_card.json",
                    "output_md": "outputs/result_card.md",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_result_card.py", joined)
        self.assertIn("--output-json outputs/result_card.json", joined)

    def test_build_command_supports_data_source_report_stage(self) -> None:
        cmd = build_command(
            {
                "name": "data_sources",
                "type": "data_source_report",
                "args": {
                    "config": "configs/data_sources.json",
                    "output_json": "outputs/data_sources.json",
                    "output_md": "outputs/data_sources.md",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_data_source_report.py", joined)
        self.assertIn("--output-json outputs/data_sources.json", joined)

    def test_build_command_supports_init_data_source_config_stage(self) -> None:
        cmd = build_command(
            {
                "name": "init_data_sources",
                "type": "init_data_source_config",
                "args": {
                    "template": "configs/data_sources_real.template.json",
                    "output": "configs/data_sources_real.local.json",
                    "report_json": "outputs/data_sources/local_config_init.json",
                    "required_dataset": ["rubricbench"],
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("init_data_source_local_config.py", joined)
        self.assertIn("--template configs/data_sources_real.template.json", joined)
        self.assertIn("--report-json outputs/data_sources/local_config_init.json", joined)
        self.assertIn("--required-dataset rubricbench", joined)

    def test_build_command_supports_rebuttal_pack_stage(self) -> None:
        cmd = build_command(
            {
                "name": "rebuttal",
                "type": "rebuttal_pack",
                "args": {
                    "evidence_matrix": "outputs/evidence.json",
                    "readiness_report": "outputs/readiness.json",
                    "output_dir": "outputs/rebuttal",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("build_rebuttal_pack.py", joined)
        self.assertIn("--evidence-matrix outputs/evidence.json", joined)

    def test_build_command_supports_preflight_stage(self) -> None:
        cmd = build_command(
            {
                "name": "preflight",
                "type": "preflight",
                "args": {
                    "input": ["data/a.jsonl", "data/b.jsonl"],
                    "providers": ["configs/providers.jsonl"],
                    "output": "outputs/preflight.json",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("preflight_real_run.py", joined)
        self.assertIn("--input data/a.jsonl --input data/b.jsonl", joined)

    def test_build_command_supports_api_budget_stage(self) -> None:
        cmd = build_command(
            {
                "name": "budget",
                "type": "api_budget",
                "args": {
                    "input": "data/queries.jsonl",
                    "providers": "configs/providers.jsonl",
                    "output": "outputs/budget.json",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("estimate_api_budget.py", joined)
        self.assertIn("--providers configs/providers.jsonl", joined)

    def test_build_command_supports_generation_budget_gate(self) -> None:
        cmd = build_command(
            {
                "name": "generate",
                "type": "generate_model_rubrics",
                "args": {
                    "input": "data/queries.jsonl",
                    "providers": "configs/providers.jsonl",
                    "output": "data/rubrics.jsonl",
                    "require_budget_report": "outputs/api_budget/budget.json",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("generate_model_rubrics.py", joined)
        self.assertIn("--require-budget-report outputs/api_budget/budget.json", joined)

    def test_build_command_supports_validate_rubrics_stage(self) -> None:
        cmd = build_command(
            {
                "name": "validate",
                "type": "validate_rubrics",
                "args": {"input": "data/rubrics.jsonl", "output_dir": "outputs/validation"},
            }
        )
        joined = " ".join(cmd)
        self.assertIn("validate_rubric_outputs.py", joined)
        self.assertIn("--output-dir outputs/validation", joined)

    def test_build_command_supports_multicandidate_stages(self) -> None:
        prepare = build_command(
            {
                "name": "prepare_rb2",
                "type": "prepare_multicandidate",
                "args": {"benchmark": "data/rb2.jsonl", "rubrics": "data/rubrics.jsonl", "output": "out.jsonl"},
            }
        )
        evaluate = build_command(
            {
                "name": "eval_rb2",
                "type": "multicandidate_downstream",
                "args": {"input": "out.jsonl", "output_dir": "outputs/rb2"},
            }
        )

        self.assertIn("prepare_multicandidate_eval.py", " ".join(prepare))
        self.assertIn("--benchmark data/rb2.jsonl", " ".join(prepare))
        self.assertIn("evaluate_multicandidate_downstream.py", " ".join(evaluate))

    def test_build_command_supports_validate_gold_stage(self) -> None:
        cmd = build_command(
            {
                "name": "validate_gold",
                "type": "validate_gold",
                "args": {
                    "input": "data/processed/rubricbench_gold.jsonl",
                    "target": "query_pool",
                    "min_records": 100,
                    "forbidden_data_source": ["toy", "proxy"],
                    "output_json": "outputs/validation/gold.json",
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("validate_gold_data.py", joined)
        self.assertIn("--target query_pool", joined)
        self.assertIn("--min-records 100", joined)
        self.assertEqual(cmd.count("--forbidden-data-source"), 2)
        self.assertIn("--strict", cmd)

    def test_build_command_supports_holdout_contamination_audit_stage(self) -> None:
        cmd = build_command(
            {
                "name": "holdout_audit",
                "type": "holdout_contamination_audit",
                "args": {
                    "holdout": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    "training": ["sft=data/processed/blindspot_sft.jsonl", "proxy=data/processed/proxy_gold.jsonl"],
                    "query_key": ["query", "prompt"],
                    "output": "outputs/contamination/audit.json",
                    "output_csv": "outputs/contamination/overlap.csv",
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("audit_holdout_contamination.py", joined)
        self.assertEqual(cmd.count("--training"), 2)
        self.assertEqual(cmd.count("--query-key"), 2)
        self.assertIn("--strict", cmd)

    def test_build_command_supports_holdout_contamination_filter_stage(self) -> None:
        cmd = build_command(
            {
                "name": "filter_rewardbench",
                "type": "filter_holdout_contamination",
                "args": {
                    "holdout": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                    "input": "data/processed/splits/rewardbench_pref_sft_proxy_train.jsonl",
                    "output": "data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl",
                    "report": "outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json",
                    "query_key": ["query", "input"],
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("filter_holdout_contamination.py", joined)
        self.assertIn("--output data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl", joined)
        self.assertEqual(cmd.count("--query-key"), 2)
        self.assertIn("--strict", cmd)

    def test_build_command_supports_sample_records_stage(self) -> None:
        cmd = build_command(
            {
                "name": "sample",
                "type": "sample_records",
                "args": {"input": "data/in.jsonl", "output": "data/out.jsonl", "n": 100, "seed": 13},
            }
        )
        joined = " ".join(cmd)
        self.assertIn("sample_records.py", joined)
        self.assertIn("--n 100", joined)

    def test_build_command_supports_verifier_filter_ablation_stage(self) -> None:
        cmd = build_command(
            {
                "name": "verifier_filter_ablation",
                "type": "verifier_filter_ablation",
                "args": {
                    "raw_teachers": "data/processed/teacher_rubrics_raw.jsonl",
                    "filtered_teachers": "data/processed/teacher_rubrics_filtered.jsonl",
                    "gold": "data/processed/rubricbench_gold.jsonl",
                    "output_dir": "outputs/verifier_filter_ablation",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("run_verifier_filter_ablation.py", joined)
        self.assertIn("--raw-teachers data/processed/teacher_rubrics_raw.jsonl", joined)
        self.assertIn("--filtered-teachers data/processed/teacher_rubrics_filtered.jsonl", joined)

    def test_build_command_supports_convert_verl_stage(self) -> None:
        cmd = build_command(
            {
                "name": "convert_verl",
                "type": "convert_verl",
                "args": {
                    "input": "data/processed/proxy_gold.jsonl",
                    "output": "data/processed/proxy_gold_verl.parquet",
                    "data_source": "proxy_gold",
                    "min_records": 1000,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("convert_to_verl_parquet.py", joined)
        self.assertIn("--input data/processed/proxy_gold.jsonl", joined)
        self.assertIn("--min-records 1000", joined)

    def test_build_command_supports_training_commands_stage(self) -> None:
        cmd = build_command(
            {
                "name": "training_commands",
                "type": "training_commands",
                "args": {
                    "config": "configs/training_commands.example.json",
                    "output_dir": "outputs/training_commands",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("make_training_commands.py", joined)
        self.assertIn("--config configs/training_commands.example.json", joined)
        self.assertIn("--output-dir outputs/training_commands", joined)

    def test_build_command_supports_convert_policy_rlvr_stage(self) -> None:
        cmd = build_command(
            {
                "name": "convert_policy",
                "type": "convert_policy_rlvr",
                "args": {
                    "input": "data/processed/healthbench_hard_queries.jsonl",
                    "output": "data/processed/healthbench_hard_policy_rlvr.parquet",
                    "data_source": "healthbench_hard",
                    "metadata_key": ["difficulty", "category"],
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("convert_policy_rlvr_data.py", joined)
        self.assertIn("--data-source healthbench_hard", joined)
        self.assertEqual(cmd.count("--metadata-key"), 2)

    def test_build_command_supports_manual_gate_stage(self) -> None:
        cmd = build_command(
            {
                "name": "training_completion_gate",
                "type": "manual_gate",
                "args": {
                    "name": "training readiness",
                    "required_path": ["outputs/checkpoints/sft", "outputs/checkpoints/rl"],
                    "required_json": ["outputs/training_commands/training_done.json:sft_checkpoint,rl_checkpoint"],
                    "required_json_contains": [
                        "outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl"
                    ],
                    "required_json_equals": [
                        "outputs/training_commands/training_done.json:rl_data=data/processed/proxy_gold_verl.parquet"
                    ],
                    "required_json_sha256": [
                        "outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet"
                    ],
                    "output": "outputs/training/gate.json",
                    "strict": True,
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("check_manual_gate.py", joined)
        self.assertIn("--required-path outputs/checkpoints/sft", joined)
        self.assertIn("--required-path outputs/checkpoints/rl", joined)
        self.assertIn("--required-json outputs/training_commands/training_done.json:sft_checkpoint,rl_checkpoint", joined)
        self.assertIn("--required-json-contains outputs/training_commands/training_done.json:served_generators=base,sft_only,sft_rl", joined)
        self.assertIn("--required-json-equals outputs/training_commands/training_done.json:rl_data=data/processed/proxy_gold_verl.parquet", joined)
        self.assertIn("--required-json-sha256 outputs/sft_data/proxy_gold_verl_report.json:output_sha256=data/processed/proxy_gold_verl.parquet", joined)
        self.assertIn("--strict", cmd)

    def test_build_command_supports_downstream_rlvr_commands_stage(self) -> None:
        cmd = build_command(
            {
                "name": "downstream_rlvr_commands",
                "type": "downstream_rlvr_commands",
                "args": {
                    "config": "configs/downstream_rlvr_commands.example.json",
                    "output_dir": "outputs/downstream_rlvr_commands",
                },
            }
        )
        joined = " ".join(cmd)
        self.assertIn("make_downstream_rlvr_commands.py", joined)
        self.assertIn("--config configs/downstream_rlvr_commands.example.json", joined)
        self.assertIn("--output-dir outputs/downstream_rlvr_commands", joined)


if __name__ == "__main__":
    unittest.main()
