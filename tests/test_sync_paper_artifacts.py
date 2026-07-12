from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sync_paper_artifacts import (
    REQUIRED_FILES,
    file_sha256,
    inspect_artifacts,
    inspect_extra_docs,
    inspect_narrative_text,
    sync_artifacts,
    write_asset_index,
)


class SyncPaperArtifactsTest(unittest.TestCase):
    def test_required_files_include_all_core_paper_tables(self) -> None:
        for name in [
            "main_table.tex",
            "rl_stage_ablation_table.tex",
            "downstream_utility_table.tex",
            "ablation_table.tex",
            "teacher_union_ablation_table.tex",
            "verifier_filter_ablation_table.tex",
            "dimension_transition_table.tex",
            "semantic_space.svg",
            "semantic_space.pdf",
            "semantic_space_points.csv",
            "semantic_space_summary.json",
        ]:
            self.assertIn(name, REQUIRED_FILES)

    def test_sync_artifacts_copies_tables_and_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            paper = root / "paper"
            artifacts.mkdir()
            (artifacts / "main_table.tex").write_text("table", encoding="utf-8")
            (artifacts / "rl_stage_ablation_table.tex").write_text("rl table", encoding="utf-8")
            (artifacts / "downstream_utility_table.tex").write_text("downstream table", encoding="utf-8")
            (artifacts / "dimension_transition_table.tex").write_text("transition table", encoding="utf-8")
            (artifacts / "verifier_filter_ablation_table.tex").write_text("verifier table", encoding="utf-8")
            (artifacts / "semantic_space.svg").write_text("<svg></svg>", encoding="utf-8")
            (artifacts / "semantic_space.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (artifacts / "semantic_space_points.csv").write_text("x,y\n0,0\n", encoding="utf-8")
            (artifacts / "semantic_space_summary.json").write_text('{"n_points": 1}', encoding="utf-8")
            (artifacts / "api_handoff.json").write_text('{"status": "blocked"}', encoding="utf-8")
            (artifacts / "api_handoff.md").write_text("# Handoff\n", encoding="utf-8")
            (artifacts / "audit_report.json").write_text('{"ok": false}', encoding="utf-8")
            (artifacts / "evidence_matrix.csv").write_text("claim_id,status\n", encoding="utf-8")
            (artifacts / "evidence_matrix.json").write_text('{"claims": []}', encoding="utf-8")
            (artifacts / "evidence_matrix.md").write_text("evidence", encoding="utf-8")
            (artifacts / "result_card.md").write_text("card", encoding="utf-8")
            (artifacts / "result_card.json").write_text("{}", encoding="utf-8")
            (artifacts / "sprint_plan.md").write_text("Generate method evaluation criteria", encoding="utf-8")
            (artifacts / "sprint_plan.csv").write_text("day,goal\n", encoding="utf-8")
            (artifacts / "sprint_plan.json").write_text("[]", encoding="utf-8")
            (artifacts / "readiness_report.md").write_text("blocked", encoding="utf-8")
            (artifacts / "readiness_report.json").write_text('{"ok": false}', encoding="utf-8")
            (artifacts / "rebuttal_pack.md").write_text("rebuttal", encoding="utf-8")
            (artifacts / "rebuttal_pack.json").write_text("[]", encoding="utf-8")
            (artifacts / "rebuttal_pack_manifest.json").write_text("{}", encoding="utf-8")
            (artifacts / "submission_gap_report.md").write_text("gap", encoding="utf-8")
            (artifacts / "submission_gap_report.json").write_text("{}", encoding="utf-8")
            (artifacts / "real_run_dashboard.md").write_text("dashboard", encoding="utf-8")
            (artifacts / "real_run_dashboard.json").write_text("{}", encoding="utf-8")

            copied = sync_artifacts(artifacts, paper)
            write_asset_index(paper, artifacts, copied)

            self.assertEqual((paper / "tables" / "main_table.tex").read_text(encoding="utf-8"), "table")
            self.assertEqual(
                (paper / "tables" / "rl_stage_ablation_table.tex").read_text(encoding="utf-8"),
                "rl table",
            )
            self.assertEqual(
                (paper / "tables" / "downstream_utility_table.tex").read_text(encoding="utf-8"),
                "downstream table",
            )
            self.assertEqual(
                (paper / "tables" / "dimension_transition_table.tex").read_text(encoding="utf-8"),
                "transition table",
            )
            self.assertFalse((paper / "tables" / "repair_table.tex").exists())
            self.assertEqual(
                (paper / "tables" / "verifier_filter_ablation_table.tex").read_text(encoding="utf-8"),
                "verifier table",
            )
            self.assertEqual((paper / "figures" / "semantic_space.svg").read_text(encoding="utf-8"), "<svg></svg>")
            self.assertEqual((paper / "figures" / "semantic_space.pdf").read_bytes(), b"%PDF-1.4\n%%EOF\n")
            self.assertEqual((paper / "asset_index" / "semantic_space_points.csv").read_text(encoding="utf-8"), "x,y\n0,0\n")
            self.assertEqual((paper / "asset_index" / "semantic_space_summary.json").read_text(encoding="utf-8"), '{"n_points": 1}')
            self.assertEqual((paper / "asset_index" / "api_handoff.json").read_text(encoding="utf-8"), '{"status": "blocked"}')
            self.assertEqual((paper / "asset_index" / "api_handoff.md").read_text(encoding="utf-8"), "# Handoff\n")
            self.assertEqual((paper / "asset_index" / "audit_report.json").read_text(encoding="utf-8"), '{"ok": false}')
            self.assertEqual((paper / "asset_index" / "evidence_matrix.csv").read_text(encoding="utf-8"), "claim_id,status\n")
            self.assertEqual((paper / "asset_index" / "evidence_matrix.json").read_text(encoding="utf-8"), '{"claims": []}')
            self.assertEqual((paper / "asset_index" / "evidence_matrix.md").read_text(encoding="utf-8"), "evidence")
            self.assertEqual((paper / "asset_index" / "result_card.md").read_text(encoding="utf-8"), "card")
            self.assertEqual((paper / "asset_index" / "result_card.json").read_text(encoding="utf-8"), "{}")
            self.assertEqual(
                (paper / "asset_index" / "sprint_plan.md").read_text(encoding="utf-8"),
                "Generate method evaluation criteria",
            )
            self.assertEqual((paper / "asset_index" / "readiness_report.md").read_text(encoding="utf-8"), "blocked")
            self.assertEqual((paper / "asset_index" / "rebuttal_pack.md").read_text(encoding="utf-8"), "rebuttal")
            self.assertEqual((paper / "asset_index" / "submission_gap_report.md").read_text(encoding="utf-8"), "gap")
            self.assertEqual((paper / "asset_index" / "real_run_dashboard.md").read_text(encoding="utf-8"), "dashboard")
            index = (paper / "asset_index.md").read_text(encoding="utf-8")
            self.assertIn("main_table.tex", index)
            self.assertIn("rl_stage_ablation_table.tex", index)
            self.assertIn("downstream_utility_table.tex", index)
            self.assertIn("dimension_transition_table.tex", index)
            self.assertNotIn("repair_table.tex", index)
            self.assertIn("verifier_filter_ablation_table.tex", index)
            self.assertIn("semantic_space.svg", index)
            self.assertIn("semantic_space.pdf", index)
            self.assertIn("api_handoff.json", index)
            self.assertIn("api_handoff.md", index)
            self.assertIn("audit_report.json", index)
            self.assertIn("evidence_matrix.csv", index)
            self.assertIn("evidence_matrix.json", index)
            self.assertIn("evidence_matrix.md", index)
            self.assertIn("result_card.md", index)
            self.assertIn("sprint_plan.md", index)
            self.assertIn("readiness_report.md", index)
            self.assertIn("rebuttal_pack.md", index)
            self.assertIn("submission_gap_report.md", index)
            self.assertIn("real_run_dashboard.md", index)
            self.assertIn("SHA256", index)
            self.assertIn(file_sha256(paper / "asset_index" / "audit_report.json"), index)

    def test_sync_artifacts_copies_extra_docs_by_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            paper = root / "paper"
            result_dir = root / "result_card"
            artifacts.mkdir()
            result_dir.mkdir()
            extra = result_dir / "result_card.json"
            extra.write_text('{"status": "blocked"}', encoding="utf-8")

            copied = sync_artifacts(artifacts, paper, extra_docs=[extra])
            write_asset_index(paper, artifacts, copied)

            target = paper / "asset_index" / "result_card.json"
            self.assertEqual(target.read_text(encoding="utf-8"), '{"status": "blocked"}')
            self.assertEqual(copied[-1]["source"], str(extra))
            self.assertEqual(copied[-1]["target"], str(target))
            self.assertIn(file_sha256(target), (paper / "asset_index.md").read_text(encoding="utf-8"))

    def test_extra_docs_replace_existing_doc_index_entries_for_same_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            paper = root / "paper"
            result_dir = root / "result_card"
            artifacts.mkdir()
            result_dir.mkdir()
            (artifacts / "result_card.json").write_text('{"status": "minimal"}', encoding="utf-8")
            extra = result_dir / "result_card.json"
            extra.write_text('{"status": "real"}', encoding="utf-8")

            copied = sync_artifacts(artifacts, paper, extra_docs=[extra])
            write_asset_index(paper, artifacts, copied)

            target = paper / "asset_index" / "result_card.json"
            index = (paper / "asset_index.md").read_text(encoding="utf-8")
            result_rows = [line for line in index.splitlines() if f"`{target}`" in line]
            self.assertEqual(target.read_text(encoding="utf-8"), '{"status": "real"}')
            self.assertEqual(len(result_rows), 1)
            self.assertIn(str(extra), result_rows[0])
            self.assertNotIn("minimal", target.read_text(encoding="utf-8"))

    def test_extra_docs_refresh_artifacts_dir_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            paper = root / "paper"
            dashboard_dir = root / "dashboard"
            artifacts.mkdir()
            dashboard_dir.mkdir()
            (artifacts / "real_run_dashboard.json").write_text('{"status": "stale"}', encoding="utf-8")
            extra = dashboard_dir / "real_run_dashboard.json"
            extra.write_text('{"status": "blocked"}', encoding="utf-8")

            sync_artifacts(artifacts, paper, extra_docs=[extra])

            self.assertEqual(
                (artifacts / "real_run_dashboard.json").read_text(encoding="utf-8"),
                '{"status": "blocked"}',
            )
            self.assertEqual(
                (paper / "asset_index" / "real_run_dashboard.json").read_text(encoding="utf-8"),
                '{"status": "blocked"}',
            )

    def test_inspect_extra_docs_reports_missing_outputs(self) -> None:
        blockers = inspect_extra_docs([Path("/tmp/missing_result_card.json")])

        self.assertEqual(blockers, ["extra doc is missing or empty: /tmp/missing_result_card.json"])

    def test_inspect_artifacts_suppresses_missing_doc_warning_when_supplied_by_extra_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            artifacts.mkdir()

            blockers, warnings = inspect_artifacts(
                artifacts,
                required_files=set(),
                supplied_doc_names={"evidence_matrix.json", "result_card.md"},
            )

        self.assertFalse(any("evidence_matrix.json" in item for item in warnings))
        self.assertFalse(any("result_card.md" in item for item in warnings))
        self.assertTrue(any("evidence_matrix.md" in item for item in warnings))
        self.assertEqual(blockers, [])

    def test_inspect_artifacts_blocks_unsupported_narrative_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            artifacts.mkdir()
            (artifacts / "experiment_summary.md").write_text(
                "We prove state-of-the-art results with a better rubric generator.\n",
                encoding="utf-8",
            )

            blockers, _ = inspect_artifacts(
                artifacts,
                required_files={"experiment_summary.md"},
            )

        self.assertTrue(any("narrative blocker" in item for item in blockers))
        self.assertTrue(any("state-of-the-art" in item for item in blockers))
        self.assertTrue(any("better rubric generator" in item for item in blockers))

    def test_inspect_extra_docs_blocks_unsupported_narrative_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp) / "result_card.md"
            extra.write_text("SFT+GRPO significantly improves downstream utility.\n", encoding="utf-8")

            blockers = inspect_extra_docs([extra])

        self.assertEqual(
            blockers,
            [
                f"narrative blocker in {extra}: unsupported phrase `significantly improves` at line 1",
            ],
        )

    def test_inspect_extra_docs_blocks_sota_and_method_improves_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp) / "result_card.md"
            extra.write_text(
                "SOTA claim.\nSFT+GRPO improves over SFT-only.\nDownstream utility improves.\n",
                encoding="utf-8",
            )

            blockers = inspect_extra_docs([extra])

        self.assertTrue(any("unsupported phrase `sota` at line 1" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `sft+grpo improves` at line 2" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `downstream utility improves` at line 3" in item for item in blockers))

    def test_inspect_extra_docs_blocks_bsc_only_utility_or_reduction_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp) / "result_card.md"
            extra.write_text(
                "Higher BSC alone supports judge utility.\n"
                "Coverage gain proves blind-spot reduction.\n",
                encoding="utf-8",
            )

            blockers = inspect_extra_docs([extra])

        self.assertTrue(any("unsupported phrase `higher bsc alone supports judge utility` at line 1" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `coverage gain proves blind-spot reduction` at line 2" in item for item in blockers))

    def test_inspect_extra_docs_blocks_sft_rl_reduction_framing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp) / "result_card.md"
            extra.write_text(
                "Keep SFT/RL reduction claims deferred until real evidence exists.\n",
                encoding="utf-8",
            )

            blockers = inspect_extra_docs([extra])

        self.assertTrue(any("unsupported phrase `sft/rl reduction` at line 1" in item for item in blockers))

    def test_inspect_extra_docs_blocks_repair_rate_and_criteria_text_improvement_framing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp) / "rl_execution_plan.md"
            extra.write_text(
                "Blind-Spot Repair Rate is the main metric.\n"
                "Do not sell this as criteria-text improvement.\n",
                encoding="utf-8",
            )

            blockers = inspect_extra_docs([extra])

        self.assertTrue(any("unsupported phrase `blind-spot repair` at line 1" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `repair rate` at line 1" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `criteria-text improvement` at line 2" in item for item in blockers))

    def test_inspect_extra_docs_blocks_legacy_claim_promotion_and_engineering_framing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp) / "result_card.md"
            extra.write_text(
                "Do not promote empirical claims.\n"
                "This result is not promotable to dimension recovery.\n"
                "This is an engineering choice.\n",
                encoding="utf-8",
            )

            blockers = inspect_extra_docs([extra])

        self.assertTrue(any("unsupported phrase `do not promote` at line 1" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `promotable` at line 2" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `engineering choice` at line 3" in item for item in blockers))

    def test_inspect_artifacts_blocks_latex_sync_placeholder_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            artifacts.mkdir()
            (artifacts / "main_table.tex").write_text(
                "Main table not synced yet. Run\n"
                "\\texttt{scripts/sync\\_paper\\_artifacts.py} after exporting artifacts.\n",
                encoding="utf-8",
            )

            blockers, _ = inspect_artifacts(
                artifacts,
                required_files={"main_table.tex"},
            )

        self.assertTrue(any("unsupported phrase `main table not synced` at line 1" in item for item in blockers))
        self.assertTrue(any("unsupported phrase `run scripts/sync_paper_artifacts.py`" in item for item in blockers))

    def test_inspect_artifacts_blocks_repairrate_metric_label_in_paper_facing_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            artifacts.mkdir()
            (artifacts / "experiment_summary.md").write_text(
                "Main table label: RepairRate(RL) > RepairRate(SFT).\n",
                encoding="utf-8",
            )

            blockers, _ = inspect_artifacts(
                artifacts,
                required_files={"experiment_summary.md"},
            )

        self.assertTrue(any("unsupported phrase `repairrate` at line 1" in item for item in blockers))

    def test_inspect_artifacts_blocks_unproven_semantic_space_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            artifacts.mkdir()
            (artifacts / "semantic_space.svg").write_text("<svg></svg>", encoding="utf-8")
            (artifacts / "semantic_space.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (artifacts / "semantic_space_points.csv").write_text("x,y\n0,0\n", encoding="utf-8")
            (artifacts / "semantic_space_summary.json").write_text('{"n_points": 1}', encoding="utf-8")

            blockers, _ = inspect_artifacts(
                artifacts,
                required_files={
                    "semantic_space.svg",
                    "semantic_space.pdf",
                    "semantic_space_points.csv",
                    "semantic_space_summary.json",
                },
            )

        self.assertTrue(any("embedding_model must be BAAI/bge-large-en-v1.5" in item for item in blockers))
        self.assertTrue(any("point CSV columns do not match schema" in item for item in blockers))

    def test_inspect_narrative_text_allows_schema_compatibility_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence_matrix.json"
            path.write_text(
                '{"path":"outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json"}',
                encoding="utf-8",
            )

            blockers = inspect_narrative_text(path)

        self.assertEqual(blockers, [])

    def test_inspect_artifacts_reports_missing_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "missing_artifacts"

            blockers, warnings = inspect_artifacts(artifacts)

        self.assertIn("artifacts directory is missing", blockers[0])
        self.assertEqual(warnings, [])

    def test_inspect_artifacts_can_use_pipeline_specific_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            artifacts.mkdir()
            (artifacts / "main_table.tex").write_text("table", encoding="utf-8")
            (artifacts / "experiment_summary.md").write_text("summary", encoding="utf-8")

            blockers, warnings = inspect_artifacts(
                artifacts,
                required_files={"main_table.tex", "experiment_summary.md"},
            )

        self.assertEqual(blockers, [])
        self.assertTrue(any("rl_stage_ablation_table.tex" in item for item in warnings))
        self.assertTrue(any("semantic_space.pdf" in item for item in warnings))

    def test_sync_artifacts_removes_stale_managed_outputs_when_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            paper = root / "paper"
            artifacts.mkdir()
            (paper / "tables").mkdir(parents=True)
            (paper / "asset_index").mkdir(parents=True)
            (paper / "tables" / "main_table.tex").write_text("stale table", encoding="utf-8")
            (paper / "tables" / "rl_stage_ablation_table.tex").write_text("stale rl table", encoding="utf-8")
            (paper / "tables" / "downstream_utility_table.tex").write_text("stale downstream table", encoding="utf-8")
            (paper / "tables" / "ablation_table.tex").write_text("stale ablation table", encoding="utf-8")
            (paper / "tables" / "teacher_union_ablation_table.tex").write_text("stale teacher table", encoding="utf-8")
            (paper / "tables" / "dimension_transition_table.tex").write_text("stale transition table", encoding="utf-8")
            (paper / "tables" / "repair_table.tex").write_text("stale repair table", encoding="utf-8")
            (paper / "tables" / "verifier_filter_ablation_table.tex").write_text("stale verifier table", encoding="utf-8")
            (paper / "figures").mkdir(parents=True)
            (paper / "figures" / "semantic_space.svg").write_text("stale semantic figure", encoding="utf-8")
            (paper / "figures" / "semantic_space.pdf").write_text("stale semantic pdf", encoding="utf-8")
            (paper / "asset_index" / "api_handoff.json").write_text("stale handoff json", encoding="utf-8")
            (paper / "asset_index" / "api_handoff.md").write_text("stale handoff md", encoding="utf-8")
            (paper / "asset_index" / "audit_report.json").write_text("stale audit", encoding="utf-8")
            (paper / "asset_index" / "evidence_matrix.csv").write_text("stale evidence csv", encoding="utf-8")
            (paper / "asset_index" / "evidence_matrix.json").write_text("stale evidence json", encoding="utf-8")
            (paper / "asset_index" / "evidence_matrix.md").write_text("stale evidence", encoding="utf-8")
            (paper / "asset_index" / "paper_asset_index_check.json").write_text("deprecated check json", encoding="utf-8")
            (paper / "asset_index" / "paper_asset_index_check.md").write_text("deprecated check md", encoding="utf-8")
            (paper / "asset_index" / "result_card.md").write_text("stale card", encoding="utf-8")
            (artifacts / "result_card.md").write_text("fresh card", encoding="utf-8")

            copied = sync_artifacts(artifacts, paper)

            self.assertFalse((paper / "tables" / "main_table.tex").exists())
            self.assertFalse((paper / "tables" / "rl_stage_ablation_table.tex").exists())
            self.assertFalse((paper / "tables" / "downstream_utility_table.tex").exists())
            self.assertFalse((paper / "tables" / "ablation_table.tex").exists())
            self.assertFalse((paper / "tables" / "teacher_union_ablation_table.tex").exists())
            self.assertFalse((paper / "tables" / "dimension_transition_table.tex").exists())
            self.assertFalse((paper / "tables" / "repair_table.tex").exists())
            self.assertFalse((paper / "tables" / "verifier_filter_ablation_table.tex").exists())
            self.assertFalse((paper / "figures" / "semantic_space.svg").exists())
            self.assertFalse((paper / "figures" / "semantic_space.pdf").exists())
            self.assertFalse((paper / "asset_index" / "api_handoff.json").exists())
            self.assertFalse((paper / "asset_index" / "api_handoff.md").exists())
            self.assertFalse((paper / "asset_index" / "audit_report.json").exists())
            self.assertFalse((paper / "asset_index" / "evidence_matrix.csv").exists())
            self.assertFalse((paper / "asset_index" / "evidence_matrix.json").exists())
            self.assertFalse((paper / "asset_index" / "evidence_matrix.md").exists())
            self.assertFalse((paper / "asset_index" / "paper_asset_index_check.json").exists())
            self.assertFalse((paper / "asset_index" / "paper_asset_index_check.md").exists())
            self.assertEqual((paper / "asset_index" / "result_card.md").read_text(encoding="utf-8"), "fresh card")
            self.assertEqual(len(copied), 1)
            self.assertEqual(copied[0]["kind"], "doc")
            self.assertEqual(copied[0]["source"], str(artifacts / "result_card.md"))
            self.assertEqual(copied[0]["target"], str(paper / "asset_index" / "result_card.md"))
            self.assertEqual(copied[0]["sha256"], file_sha256(paper / "asset_index" / "result_card.md"))

    def test_asset_index_records_blockers_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            paper = root / "paper"
            artifacts.mkdir()
            (artifacts / "experiment_summary.md").write_text("summary", encoding="utf-8")
            (artifacts / "result_card.md").write_text("", encoding="utf-8")

            blockers, warnings = inspect_artifacts(artifacts)
            copied = sync_artifacts(artifacts, paper)
            write_asset_index(paper, artifacts, copied, blockers=blockers, warnings=warnings)

            index = (paper / "asset_index.md").read_text(encoding="utf-8")

        self.assertIn("main_table.tex", blockers[0])
        self.assertTrue(any("dimension_transition_table.tex" in item for item in blockers))
        self.assertFalse(any("repair_table.tex" in item for item in blockers))
        self.assertTrue(any("result_card.md" in item for item in warnings))
        self.assertIn("Blockers: 11", index)
        self.assertIn("## Blocker Summary", index)
        self.assertIn("`rl_stage_ablation`: 1", index)
        self.assertIn("`downstream_utility`: 1", index)
        self.assertIn("`semantic_space`: 4", index)
        self.assertIn("Warnings:", index)
        self.assertIn("## Warning Summary", index)
        self.assertIn("`api_handoff`: 2", index)
        self.assertIn("`audit_report`: 1", index)
        self.assertNotIn("result_card.md` |", index)

    def test_main_writes_asset_index_when_artifacts_dir_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "missing_artifacts"
            paper = root / "paper"

            copied = sync_artifacts(artifacts, paper)
            blockers, warnings = inspect_artifacts(artifacts)
            write_asset_index(paper, artifacts, copied, blockers=blockers, warnings=warnings)

            index = (paper / "asset_index.md").read_text(encoding="utf-8")

        self.assertEqual(copied, [])
        self.assertIn("artifacts directory is missing", index)


if __name__ == "__main__":
    unittest.main()
