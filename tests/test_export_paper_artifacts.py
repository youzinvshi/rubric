from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_paper_asset_index import SEMANTIC_POINT_CSV_COLUMNS
from scripts.export_paper_artifacts import (
    add_display_columns,
    build_rl_stage_ablation_rows,
    build_summary_markdown,
    format_cell,
    latex_table,
    main,
    parse_labeled_path_specs,
    read_csv,
    read_csv_checked,
    read_downstream_tables,
    read_json,
    read_transition_summaries,
    collect_semantic_space_artifacts,
    repair_summary_to_row,
    transition_summary_to_row,
    sanitize_main_downstream_metrics,
)
from scripts.sync_paper_artifacts import file_sha256


class ExportPaperArtifactsTest(unittest.TestCase):
    def test_format_cell_formats_float_and_escapes_method(self) -> None:
        self.assertEqual(format_cell("0.123456"), "0.1235")
        self.assertEqual(format_cell("sft_only"), r"sft\_only")

    def test_latex_table_contains_caption_and_rows(self) -> None:
        table = latex_table(
            rows=[{"method": "toy", "cov": "0.5"}],
            columns=[("method", "Method"), ("cov", "Cov")],
            caption="Cap.",
            label="tab:x",
        )
        self.assertIn(r"\caption{Cap.}", table)
        self.assertIn("toy & 0.5000", table)

    def test_read_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("method,cov\ntoy,0.5\n", encoding="utf-8")
            self.assertEqual(read_csv(path), [{"method": "toy", "cov": "0.5"}])

    def test_add_display_columns_prefers_ci_text(self) -> None:
        rows = add_display_columns(
            [{"method": "base", "cov": "0.5", "cov_ci": "0.5 [0.4, 0.6]"}],
            {"cov": "cov_ci", "blind": "blind_ci"},
        )

        self.assertEqual(rows[0]["cov_display"], "0.5 [0.4, 0.6]")
        self.assertEqual(rows[0]["blind_display"], "")

    def test_summary_mentions_teacher_union_rows(self) -> None:
        text = build_summary_markdown(
            main_rows=[],
            rl_stage_rows=[{"comparison": "SFT-only -> SFT+GRPO"}],
            downstream_rows=[{"benchmark": "RewardBench"}],
            ablation_rows=[],
            teacher_union_rows=[{"variant": "multi_teacher_union"}],
            verifier_filter_rows=[{"variant": "verifier_filtered"}],
            repair_rows=[{"comparison": "base -> sft"}],
            audit={},
            copied=[],
        )
        self.assertIn("RL-stage ablation rows: 1", text)
        self.assertIn("Downstream utility rows: 1", text)
        self.assertIn("Teacher-union ablation rows: 1", text)
        self.assertIn("Verifier-filter ablation rows: 1", text)
        self.assertIn("Dimension-transition rows: 1", text)
        self.assertIn("Semantic-space artifacts: 0", text)
        self.assertIn("teacher_union_ablation_table.tex", text)
        self.assertIn("rl_stage_ablation_table.tex", text)
        self.assertIn("verifier_filter_ablation_table.tex", text)
        self.assertIn("dimension_transition_table.tex", text)
        self.assertNotIn("repair_table.tex", text)
        self.assertIn("result_card.md", text)

    def test_parse_labeled_path_specs_accepts_benchmark_paths(self) -> None:
        specs = parse_labeled_path_specs(["RewardBench=outputs/matrix_real/main_table.csv"])

        self.assertEqual(specs[0][0], "RewardBench")
        self.assertEqual(str(specs[0][1]), "outputs/matrix_real/main_table.csv")

    def test_read_downstream_tables_adds_benchmark_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "downstream.csv"
            path.write_text(
                (
                    "method,downstream_status,downstream_paper_claim_eligible,"
                    "downstream_n,accuracy,tie_rate,mean_margin\n"
                    "base,pass,true,10,0.6,0.1,0.2\n"
                ),
                encoding="utf-8",
            )

            rows, blockers, warnings = read_downstream_tables([("RewardBench", path)])

        self.assertEqual(blockers, [])
        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]["benchmark"], "RewardBench")
        self.assertEqual(rows[0]["accuracy"], "0.6")

    def test_read_downstream_tables_blocks_non_paper_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "downstream.csv"
            path.write_text(
                (
                    "method,downstream_status,downstream_paper_claim_eligible,"
                    "downstream_n,accuracy,tie_rate,mean_margin\n"
                    "base,not_paper_eligible,false,,0.9,0.0,0.5\n"
                ),
                encoding="utf-8",
            )

            rows, blockers, warnings = read_downstream_tables([("RewardBench", path)])

        self.assertEqual(rows, [])
        self.assertEqual(warnings, [])
        self.assertIn("RewardBench downstream row for base is not paper-eligible", blockers[0])

    def test_sanitize_main_downstream_metrics_clears_non_eligible_accuracy(self) -> None:
        rows = [
            {
                "method": "base",
                "downstream_status": "not_paper_eligible",
                "downstream_paper_claim_eligible": "false",
                "accuracy": "0.9",
                "accuracy_ci": "0.9000 [0.8000, 1.0000]",
                "tie_rate": "0.0",
                "mean_margin": "0.5",
                "downstream_n": "10",
            }
        ]

        blockers = sanitize_main_downstream_metrics(rows)

        self.assertIn("main table downstream metrics for base are not paper-eligible", blockers[0])
        self.assertEqual(rows[0]["accuracy"], "")
        self.assertEqual(rows[0]["accuracy_ci"], "")
        self.assertEqual(rows[0]["downstream_n"], "")

    def test_build_rl_stage_ablation_rows_computes_deltas(self) -> None:
        rows, warnings = build_rl_stage_ablation_rows(
            [
                {
                    "method": "sft_only",
                    "cov": "0.50",
                    "blind": "0.50",
                    "red": "0.10",
                    "hall": "0.20",
                    "accuracy": "0.60",
                    "coverage_per_generated_criterion": "0.10",
                },
                {
                    "method": "sft_rl",
                    "cov": "0.62",
                    "blind": "0.38",
                    "red": "0.12",
                    "hall": "0.18",
                    "accuracy": "0.65",
                    "coverage_per_generated_criterion": "0.12",
                },
            ]
        )

        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]["comparison"], "SFT-only -> SFT+GRPO")
        self.assertEqual(rows[0]["delta_cov"], "0.1200")
        self.assertEqual(rows[0]["delta_cov_per_gen"], "0.0200")
        self.assertEqual(rows[0]["delta_red"], "0.0200")
        self.assertEqual(rows[0]["delta_hall"], "-0.0200")
        self.assertEqual(rows[0]["delta_accuracy"], "0.0500")

    def test_read_csv_checked_reports_missing_required_csv(self) -> None:
        rows, blockers, warnings = read_csv_checked(Path("/tmp/missing_main_table.csv"), "main table", required=True)

        self.assertEqual(rows, [])
        self.assertIn("main table CSV is missing", blockers[0])
        self.assertEqual(warnings, [])

    def test_read_json_returns_load_error_for_invalid_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            path.write_text("{bad", encoding="utf-8")

            data = read_json(path)

        self.assertIn("_load_error", data)
        self.assertIn("not valid JSON", data["_load_error"])

    def test_summary_includes_blockers_and_warnings(self) -> None:
        text = build_summary_markdown(
            main_rows=[],
            ablation_rows=[],
            teacher_union_rows=[],
            audit={},
            copied=[],
            repair_rows=[],
            blockers=["main table CSV is missing"],
            warnings=["optional ablation table is missing"],
        )

        self.assertIn("Blockers: 1", text)
        self.assertIn("main table CSV is missing", text)
        self.assertIn("optional ablation table is missing", text)

    def test_main_writes_summary_when_required_main_table_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            missing_main = root / "missing_main.csv"
            audit = root / "audit.json"
            audit.write_text("{bad", encoding="utf-8")

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(missing_main),
                "--audit-report",
                str(audit),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")

        self.assertIn("main table CSV is missing", summary)
        self.assertIn("audit report is not readable", summary)
        self.assertNotIn("`main_table.tex`", summary)
        self.assertIn("`experiment_summary.md`", summary)
        self.assertFalse((output_dir / "main_table.tex").exists())

    def test_main_copies_machine_readable_evidence_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            main_table = root / "main.csv"
            handoff_json = root / "api_handoff.json"
            handoff_md = root / "api_handoff.md"
            evidence_json = root / "evidence_matrix.json"
            evidence_csv = root / "evidence_matrix.csv"
            evidence_md = root / "evidence_matrix.md"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")
            handoff_json.write_text('{"status": "blocked"}', encoding="utf-8")
            handoff_md.write_text("# Handoff\n", encoding="utf-8")
            evidence_json.write_text('{"claims": []}', encoding="utf-8")
            evidence_csv.write_text("claim_id,status\nC1,missing_evidence\n", encoding="utf-8")
            evidence_md.write_text("| Claim ID | Status |\n| --- | --- |\n", encoding="utf-8")

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--handoff-json",
                str(handoff_json),
                "--handoff-md",
                str(handoff_md),
                "--evidence-json",
                str(evidence_json),
                "--evidence-csv",
                str(evidence_csv),
                "--evidence-md",
                str(evidence_md),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            self.assertEqual((output_dir / "api_handoff.json").read_text(encoding="utf-8"), '{"status": "blocked"}')
            self.assertEqual((output_dir / "api_handoff.md").read_text(encoding="utf-8"), "# Handoff\n")
            self.assertEqual((output_dir / "evidence_matrix.json").read_text(encoding="utf-8"), '{"claims": []}')
            self.assertEqual(
                (output_dir / "evidence_matrix.csv").read_text(encoding="utf-8"),
                "claim_id,status\nC1,missing_evidence\n",
            )
            main_table_tex = (output_dir / "main_table.tex").read_text(encoding="utf-8")
            self.assertIn("Evidence-gated BlindSpot-RL main matrix", main_table_tex)
            self.assertIn("Trained and downstream claims are reportable only after their evidence gates pass", main_table_tex)
            self.assertNotIn("BlindSpot-RL main results", main_table_tex)
            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")
            self.assertIn("api_handoff.json", summary)
            self.assertIn("api_handoff.md", summary)
            self.assertIn("evidence_matrix.json", summary)
            self.assertIn("evidence_matrix.csv", summary)

    def test_transition_summary_to_row_builds_paper_table_row(self) -> None:
        row = transition_summary_to_row(
            {
                "baseline_label": "base",
                "candidate_label": "sft_grpo",
                "total_gold": 10,
                "baseline_blind_gold": 6,
                "recovered_gold": 4,
                "lost_gold": 1,
                "baseline_coverage": 0.4,
                "candidate_coverage": 0.7,
                "recovered_dimension_rate": 2 / 3,
                "loss_rate": 0.25,
                "net_transition_rate": 0.3,
            },
            Path("outputs/repair/transition_summary.json"),
        )

        self.assertEqual(row["comparison"], "base -> sft_grpo")
        self.assertEqual(row["recovered_gold"], "4")
        self.assertNotIn("repaired_gold", row)
        self.assertEqual(row["recovered_dimension_rate"], str(2 / 3))
        self.assertEqual(row["net_transition_rate"], "0.3")

    def test_transition_summary_to_row_keeps_legacy_metric_aliases_readable(self) -> None:
        row = repair_summary_to_row(
            {
                "baseline_label": "base",
                "candidate_label": "sft_grpo",
                "repaired_gold": 3,
                "repair_rate": 0.5,
                "net_repair_rate": 0.25,
            },
            Path("outputs/repair/transition_summary.json"),
        )

        self.assertEqual(row["recovered_gold"], "3")
        self.assertNotIn("repaired_gold", row)
        self.assertEqual(row["recovered_dimension_rate"], "0.5")
        self.assertEqual(row["net_transition_rate"], "0.25")

    def test_read_transition_summaries_reads_multiple_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "sft_transition.json"
            second = root / "grpo_transition.json"
            first.write_text(
                '{"baseline_label":"base","candidate_label":"sft","recovered_dimension_rate":0.5}',
                encoding="utf-8",
            )
            second.write_text(
                '{"baseline_label":"base","candidate_label":"grpo","recovered_dimension_rate":0.75}',
                encoding="utf-8",
            )

            rows, blockers, warnings = read_transition_summaries([first, second])

        self.assertEqual([row["comparison"] for row in rows], ["base -> sft", "base -> grpo"])
        self.assertEqual(blockers, [])
        self.assertEqual(warnings, [])

    def test_main_writes_dimension_transition_table_from_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            main_table = root / "main.csv"
            repair = root / "transition_summary.json"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")
            repair.write_text(
                (
                    '{"baseline_label":"base","candidate_label":"sft_grpo",'
                    '"total_gold":10,"baseline_blind_gold":6,"recovered_gold":4,'
                    '"lost_gold":1,"baseline_coverage":0.4,"candidate_coverage":0.7,'
                    '"recovered_dimension_rate":0.6667,"loss_rate":0.25,"net_transition_rate":0.3}'
                ),
                encoding="utf-8",
            )

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--transition-summary-json",
                str(repair),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            transition_table = (output_dir / "dimension_transition_table.tex").read_text(encoding="utf-8")
            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")
            self.assertIn("base -> sft\\_grpo", transition_table)
            self.assertIn("0.6667", transition_table)
            self.assertIn("Recovered", transition_table)
            self.assertIn("Net Transition", transition_table)
            self.assertIn("Evidence-gated gold-dimension transition audit over recovered", transition_table)
            self.assertIn("dimension-level recovery wording requires C12 support", transition_table)
            self.assertNotIn("Gold-dimension blind-spot reduction audit", transition_table)
            self.assertNotIn("Net Recovery", transition_table)
            self.assertTrue((output_dir / "dimension_transition_table.csv").exists())
            transition_csv = (output_dir / "dimension_transition_table.csv").read_text(encoding="utf-8")
            self.assertIn("recovered_gold", transition_csv)
            self.assertIn("recovered_dimension_rate", transition_csv)
            self.assertIn("net_transition_rate", transition_csv)
            self.assertNotIn("repaired_gold", transition_csv)
            self.assertNotIn("repair_rate", transition_csv)
            self.assertTrue((output_dir / "dimension_transition_table.md").exists())
            self.assertFalse((output_dir / "repair_table.tex").exists())
            self.assertFalse((output_dir / "repair_table.csv").exists())
            self.assertFalse((output_dir / "repair_table.md").exists())
            self.assertIn("Dimension-transition rows: 1", summary)

    def test_main_writes_verifier_filter_ablation_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            main_table = root / "main.csv"
            verifier = root / "verifier_filter.csv"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")
            verifier.write_text(
                (
                    "variant,mean_coverage,mean_blind,mean_redundancy,mean_hallucination,"
                    "mean_reward,mean_n_gen,coverage_delta_vs_no_verifier,hallucination_delta_vs_no_verifier\n"
                    "no_verifier_filter,0.5,0.5,0.2,0.3,0.6,8,0,0\n"
                    "verifier_filtered,0.55,0.45,0.1,0.1,0.8,6,0.05,-0.2\n"
                ),
                encoding="utf-8",
            )

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--verifier-filter-csv",
                str(verifier),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            table = (output_dir / "verifier_filter_ablation_table.tex").read_text(encoding="utf-8")
            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")

        self.assertIn("verifier\\_filtered", table)
        self.assertIn(r"\Delta$Hall", table)
        self.assertIn("Evidence-gated verifier-filtering ablation", table)
        self.assertIn("proxy-quality claims require C7 support", table)
        self.assertIn("Verifier-filter ablation rows: 2", summary)

    def test_main_writes_gate_aware_reward_and_teacher_union_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            main_table = root / "main.csv"
            ablation = root / "ablation.csv"
            teacher = root / "teacher_union.csv"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")
            ablation.write_text(
                "variant,mean_coverage,mean_blind,mean_redundancy,mean_hallucination,mean_reward\n"
                "full,0.6,0.4,0.1,0.0,0.8\n",
                encoding="utf-8",
            )
            teacher.write_text(
                "variant,mean_coverage,mean_blind,mean_redundancy,mean_reward,mean_n_gen,coverage_gain_vs_best_single\n"
                "multi_teacher_union,0.6,0.4,0.1,0.8,8,0.05\n",
                encoding="utf-8",
            )

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--ablation-csv",
                str(ablation),
                "--teacher-union-csv",
                str(teacher),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            ablation_table = (output_dir / "ablation_table.tex").read_text(encoding="utf-8")
            teacher_table = (output_dir / "teacher_union_ablation_table.tex").read_text(encoding="utf-8")

        self.assertIn("Evidence-gated reward-component ablations", ablation_table)
        self.assertIn("Attribution to reward terms is reportable only after C7 passes", ablation_table)
        self.assertIn("Evidence-gated single-teacher versus multi-teacher union", teacher_table)
        self.assertIn("proxy-gold construction", teacher_table)

    def test_main_writes_combined_downstream_utility_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            main_table = root / "main.csv"
            judgebench = root / "judgebench.csv"
            main_table.write_text(
                (
                    "method,cov,blind,red,hall,downstream_status,downstream_paper_claim_eligible,"
                    "accuracy,accuracy_ci,tie_rate,tie_rate_ci,mean_margin,mean_margin_ci,downstream_n\n"
                    "base,0.6,0.4,0.1,0.0,pass,true,"
                    "0.7,\"0.7000 [0.6000, 0.8000]\",0.1,"
                    "\"0.1000 [0.0000, 0.2000]\",0.2,\"0.2000 [0.1000, 0.3000]\",10\n"
                ),
                encoding="utf-8",
            )
            judgebench.write_text(
                (
                    "method,downstream_status,downstream_paper_claim_eligible,"
                    "accuracy,accuracy_ci,tie_rate,tie_rate_ci,mean_margin,mean_margin_ci,downstream_n\n"
                    "sft_rl,pass,true,0.75,\"0.7500 [0.6500, 0.8500]\",0.05,"
                    "\"0.0500 [0.0000, 0.1000]\",0.3,\"0.3000 [0.2000, 0.4000]\",20\n"
                ),
                encoding="utf-8",
            )

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--downstream-table-csv",
                f"RewardBench={main_table}",
                "--downstream-table-csv",
                f"JudgeBench={judgebench}",
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            table = (output_dir / "downstream_utility_table.tex").read_text(encoding="utf-8")
            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")

        self.assertIn("RewardBench", table)
        self.assertIn("JudgeBench", table)
        self.assertIn("sft\\_rl", table)
        self.assertIn("0.7500 [0.6500, 0.8500]", table)
        self.assertIn("Evidence-gated downstream judge-utility rows", table)
        self.assertIn("paper-eligibility metadata passes", table)
        self.assertIn("Downstream utility rows: 2", summary)

    def test_main_writes_rl_stage_ablation_table_from_main_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            main_table = root / "main.csv"
            main_table.write_text(
                (
                    "method,cov,blind,red,hall,coverage_per_generated_criterion,accuracy\n"
                    "sft_only,0.50,0.50,0.10,0.20,0.10,0.60\n"
                    "sft_rl,0.62,0.38,0.12,0.18,0.12,0.65\n"
                ),
                encoding="utf-8",
            )

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            table = (output_dir / "rl_stage_ablation_table.tex").read_text(encoding="utf-8")
            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")

        self.assertIn("SFT-only -> SFT+GRPO", table)
        self.assertIn("0.1200", table)
        self.assertIn(r"\Delta$Cov/Gen", table)
        self.assertIn("0.0200", table)
        self.assertIn("-0.0200", table)
        self.assertIn("Evidence-gated SFT-only versus SFT+GRPO stage comparison", table)
        self.assertIn("RL-stage support is reportable only after C14", table)
        self.assertIn("RL-stage ablation rows: 1", summary)

    def test_main_removes_stale_transition_tables_when_repair_input_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            output_dir.mkdir()
            (output_dir / "dimension_transition_table.tex").write_text("stale transition table", encoding="utf-8")
            (output_dir / "dimension_transition_table.csv").write_text("stale transition csv", encoding="utf-8")
            (output_dir / "repair_table.tex").write_text("stale repair table", encoding="utf-8")
            (output_dir / "repair_table.csv").write_text("stale repair csv", encoding="utf-8")
            main_table = root / "main.csv"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            self.assertFalse((output_dir / "dimension_transition_table.tex").exists())
            self.assertFalse((output_dir / "dimension_transition_table.csv").exists())
            self.assertFalse((output_dir / "repair_table.tex").exists())
            self.assertFalse((output_dir / "repair_table.csv").exists())
            self.assertIn(
                "Dimension-transition rows: 0",
                (output_dir / "experiment_summary.md").read_text(encoding="utf-8"),
            )

    def test_main_removes_stale_managed_docs_when_sources_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            output_dir.mkdir()
            for name in [
                "readiness_report.json",
                "submission_gap_report.json",
                "rebuttal_pack_manifest.json",
                "result_card.json",
                "api_handoff.json",
            ]:
                (output_dir / name).write_text('{"stale": true}', encoding="utf-8")
            main_table = root / "main.csv"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            self.assertFalse((output_dir / "readiness_report.json").exists())
            self.assertFalse((output_dir / "submission_gap_report.json").exists())
            self.assertFalse((output_dir / "rebuttal_pack_manifest.json").exists())
            self.assertFalse((output_dir / "result_card.json").exists())
            self.assertFalse((output_dir / "api_handoff.json").exists())

    def test_collect_semantic_space_artifacts_requires_four_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            semantic_dir = Path(tmp) / "semantic"
            write_valid_semantic_space_artifacts(semantic_dir)

            files, blockers, warnings = collect_semantic_space_artifacts(semantic_dir)

        self.assertEqual(
            [path.name for path in files],
            ["semantic_space.svg", "semantic_space.pdf", "semantic_space_points.csv", "semantic_space_summary.json"],
        )
        self.assertEqual(blockers, [])
        self.assertEqual(warnings, [])

    def test_collect_semantic_space_artifacts_blocks_unproven_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            semantic_dir = Path(tmp) / "semantic"
            semantic_dir.mkdir()
            (semantic_dir / "semantic_space.svg").write_text("<svg></svg>", encoding="utf-8")
            (semantic_dir / "semantic_space.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (semantic_dir / "semantic_space_points.csv").write_text("x,y\n0,0\n", encoding="utf-8")
            (semantic_dir / "semantic_space_summary.json").write_text('{"n_points": 1}', encoding="utf-8")

            files, blockers, warnings = collect_semantic_space_artifacts(semantic_dir)

        self.assertEqual(files, [])
        self.assertTrue(any("embedding_model must be BAAI/bge-large-en-v1.5" in item for item in blockers))
        self.assertEqual(warnings, [])

    def test_main_copies_semantic_space_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "artifacts"
            semantic_dir = root / "semantic"
            main_table = root / "main.csv"
            main_table.write_text("method,cov,blind,red,hall,accuracy\nbase,0.6,0.4,0.1,0.0,0.7\n", encoding="utf-8")
            write_valid_semantic_space_artifacts(semantic_dir)

            argv = [
                "export_paper_artifacts.py",
                "--main-table-csv",
                str(main_table),
                "--semantic-space-dir",
                str(semantic_dir),
                "--output-dir",
                str(output_dir),
            ]
            with patch("sys.argv", argv):
                main()

            self.assertEqual((output_dir / "semantic_space.svg").read_text(encoding="utf-8"), "<svg></svg>")
            self.assertEqual((output_dir / "semantic_space.pdf").read_bytes(), b"%PDF-1.4\n%%EOF\n")
            self.assertTrue((output_dir / "semantic_space_points.csv").exists())
            self.assertIn(
                "Semantic-space artifacts: 4",
                (output_dir / "experiment_summary.md").read_text(encoding="utf-8"),
            )

def write_valid_semantic_space_artifacts(semantic_dir: Path) -> None:
    semantic_dir.mkdir()
    svg = semantic_dir / "semantic_space.svg"
    pdf = semantic_dir / "semantic_space.pdf"
    points = semantic_dir / "semantic_space_points.csv"
    summary = semantic_dir / "semantic_space_summary.json"
    svg.write_text("<svg></svg>", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    rows = [
        [
            "0",
            "0",
            "human_gold",
            "gold",
            "evidence_grounding",
            "g000",
            "0",
            "0.0",
            "0.0",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "q",
            "Uses evidence",
            "",
        ]
    ]
    for idx, method in enumerate(["base", "sft_only", "sft_rl"], start=1):
        rows.append(
            [
                str(idx),
                "0",
                method,
                "generated",
                "evidence_grounding",
                "",
                "0",
                f"0.{idx}",
                f"0.{idx}",
                "0",
                "0",
                "0",
                "evidence_grounding",
                "g000",
                "0.9000",
                "true",
                "q",
                f"{method} uses evidence",
                "Uses evidence",
            ]
        )
    points.write_text(
        ",".join(SEMANTIC_POINT_CSV_COLUMNS)
        + "\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    input_provenance = [
        {
            "label": label,
            "sha256": "abc",
            "join_report": {
                "join_key": "query",
                "gold": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                "output_sha256": "abc",
            },
        }
        for label in ["base", "sft_only", "sft_rl"]
    ]
    summary.write_text(
        json.dumps(
            {
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "requested_projection": "umap",
                "projection": "umap_fallback_pca",
                "gold_cluster_tau": 0.75,
                "point_csv_schema_version": 3,
                "point_csv_columns": SEMANTIC_POINT_CSV_COLUMNS,
                "output_artifacts_schema_version": 1,
                "point_csv_rows_match_n_points": True,
                "n_points": 4,
                "n_gold": 1,
                "n_generated": 3,
                "n_gold_clusters": 1,
                "methods": ["base", "sft_only", "sft_rl"],
                "generated_gold_category_coverage_by_method": {"base": 1.0, "sft_only": 1.0, "sft_rl": 1.0},
                "nearest_gold_category_coverage_by_method": {"base": 1.0, "sft_only": 1.0, "sft_rl": 1.0},
                "nearest_gold_cluster_coverage_by_method": {"base": 1.0, "sft_only": 1.0, "sft_rl": 1.0},
                "nearest_gold_cluster_distribution_by_method": {
                    "base": {"g000": 1},
                    "sft_only": {"g000": 1},
                    "sft_rl": {"g000": 1},
                },
                "nearest_gold_cluster_entropy_by_method": {"base": 0.0, "sft_only": 0.0, "sft_rl": 0.0},
                "generated_dispersion_by_method": {"base": 0.0, "sft_only": 0.0, "sft_rl": 0.0},
                "mean_nearest_gold_similarity_by_method": {"base": 0.9, "sft_only": 0.9, "sft_rl": 0.9},
                "inputs": input_provenance,
                "point_csv_sha256": file_sha256(points),
                "svg_sha256": file_sha256(svg),
                "pdf_sha256": file_sha256(pdf),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
