from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.check_paper_asset_index import SEMANTIC_POINT_CSV_COLUMNS
from scripts.check_submission_readiness import (
    ALLOWED_EVIDENCE_STATUSES,
    REQUIRED_EVIDENCE_CLAIMS,
    REQUIRED_PAPER_FILES,
    REQUIRED_SYNCED_DOCS,
    REQUIRED_SYNCED_FIGURES,
    REQUIRED_SYNCED_TABLES,
    asset_index_check,
    build_claim_discipline,
    build_report,
    parse_raw_gate_spec,
    summarize_evidence,
    summarize_raw_gate,
    to_markdown,
)


class CheckSubmissionReadinessTest(unittest.TestCase):
    def test_required_synced_tables_cover_paper_facing_generated_tables(self) -> None:
        self.assertIn("tables/rl_stage_ablation_table.tex", REQUIRED_SYNCED_TABLES)
        self.assertIn("tables/downstream_utility_table.tex", REQUIRED_SYNCED_TABLES)
        self.assertIn("tables/verifier_filter_ablation_table.tex", REQUIRED_SYNCED_TABLES)
        self.assertIn("tables/dimension_transition_table.tex", REQUIRED_SYNCED_TABLES)
        self.assertNotIn("tables/repair_table.tex", REQUIRED_SYNCED_TABLES)

    def test_required_synced_figures_cover_semantic_space_assets(self) -> None:
        self.assertIn("figures/semantic_space.pdf", REQUIRED_SYNCED_FIGURES)
        self.assertIn("figures/semantic_space.svg", REQUIRED_SYNCED_FIGURES)

    def test_required_synced_docs_cover_reviewer_facing_rebuttal_pack(self) -> None:
        self.assertIn("asset_index/real_run_dashboard.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/real_run_dashboard.md", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/evidence_matrix.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/evidence_matrix.csv", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/evidence_matrix.md", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/semantic_space_points.csv", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/semantic_space_summary.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/result_card.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/result_card.md", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/submission_gap_report.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/submission_gap_report.md", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/readiness_report.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/readiness_report.md", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/rebuttal_pack.json", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/rebuttal_pack.md", REQUIRED_SYNCED_DOCS)
        self.assertIn("asset_index/rebuttal_pack_manifest.json", REQUIRED_SYNCED_DOCS)

    def test_required_paper_files_cover_full_manuscript_sections(self) -> None:
        for path in [
            "main.tex",
            "sections/abstract.tex",
            "sections/introduction.tex",
            "sections/blindspot_phenomenon.tex",
            "sections/related_work.tex",
            "sections/method.tex",
            "sections/experiments.tex",
            "sections/limitations.tex",
            "sections/conclusion.tex",
        ]:
            self.assertIn(path, REQUIRED_PAPER_FILES)

    def test_required_paper_files_match_main_tex_section_inputs(self) -> None:
        main = Path("paper/main.tex").read_text(encoding="utf-8")
        section_inputs = {
            f"{match}.tex"
            for match in re.findall(r"\\input\{(sections/[^}]+)\}", main)
        }

        self.assertTrue(section_inputs)
        self.assertTrue(section_inputs.issubset(set(REQUIRED_PAPER_FILES)))

    def test_required_evidence_claims_cover_core_aaai_story(self) -> None:
        for claim_id in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10", "C12", "C13", "C14"]:
            self.assertIn(claim_id, REQUIRED_EVIDENCE_CLAIMS)

    def test_allowed_evidence_statuses_cover_claim_ladder_lifecycle(self) -> None:
        for status in ["safe_to_claim", "missing_evidence", "contradicted", "not_yet_supported"]:
            self.assertIn(status, ALLOWED_EVIDENCE_STATUSES)

    def test_required_evidence_claims_exist_in_real_template(self) -> None:
        config = json.loads(Path("configs/evidence_matrix_real.template.json").read_text(encoding="utf-8"))
        claim_ids = {claim["id"] for claim in config["claims"]}

        self.assertTrue(set(REQUIRED_EVIDENCE_CLAIMS).issubset(claim_ids))

    def test_summarize_evidence_counts_statuses(self) -> None:
        summary = summarize_evidence(
            [
                {"status": "safe_to_claim"},
                {"status": "missing_evidence"},
                {"status": "contradicted"},
            ]
        )
        self.assertEqual(summary["safe_to_claim"], 1)
        self.assertEqual(summary["missing_evidence"], 1)
        self.assertEqual(summary["contradicted"], 1)
        self.assertEqual(summary["total"], 3)

    def test_build_report_passes_with_audit_evidence_and_paper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["summary"]["status"], "ready")
        self.assertEqual(report["summary"]["hard_blocker_count"], 0)
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["hard_blockers"], [])

    def test_build_report_blocks_missing_required_synced_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper, include_synced_tables=False)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["blockers"], report["hard_blockers"])
        self.assertEqual(report["summary"]["hard_blocker_count"], len(report["hard_blockers"]))
        self.assertGreaterEqual(report["summary"]["hard_blocker_category_count"], 1)
        self.assertTrue(any(row["category"] == "paper_tables" for row in report["hard_blocker_summary"]))
        self.assertTrue(any("required paper tables not synced" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_missing_required_synced_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper, include_synced_figures=False)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper figures not synced" in item for item in report["hard_blockers"]))
        self.assertIn("figures/semantic_space.pdf", report["hard_blockers"][0])

    def test_build_report_blocks_missing_required_synced_rebuttal_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper, include_synced_docs=False)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing docs not synced" in item for item in report["hard_blockers"]))
        self.assertIn("asset_index/rebuttal_pack_manifest.json", report["hard_blockers"][0])

    def test_build_report_blocks_synced_doc_missing_claim_ladder_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index" / "result_card.json").write_text('{"ok": false}', encoding="utf-8")
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("result_card.json missing required JSON field `claim_ladder`" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_synced_gap_report_execution_summary_without_ready_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            gap_report = submission_gap_report_placeholder()
            gap_report["execution_sequence"][0]["summary"] = "phase_status=blocked"
            (paper / "asset_index" / "submission_gap_report.json").write_text(
                json.dumps(gap_report),
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("submission_gap_report.json execution step 0 summary missing ready_to_run" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_synced_gap_report_missing_paper_asset_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            gap_report = submission_gap_report_placeholder()
            gap_report["phases"][0].pop("paper_asset_warnings")
            (paper / "asset_index" / "submission_gap_report.json").write_text(
                json.dumps(gap_report),
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(
            any(
                "submission_gap_report.json phase 0 missing required field `paper_asset_warnings` list" in item
                for item in report["hard_blockers"]
            )
        )

    def test_build_report_blocks_synced_gap_report_missing_execution_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            gap_report = submission_gap_report_placeholder()
            gap_report["execution_sequence"][0].pop("commands")
            (paper / "asset_index" / "submission_gap_report.json").write_text(
                json.dumps(gap_report),
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(
            any(
                "submission_gap_report.json execution step 0 missing required non-empty field `commands` list" in item
                for item in report["hard_blockers"]
            )
        )

    def test_build_report_blocks_synced_gap_report_markdown_missing_execution_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index" / "submission_gap_report.md").write_text(
                "## Claim Ladder Status\n",
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("submission_gap_report.md missing required text `## Execution Sequence`" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_synced_evidence_matrix_csv_missing_required_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index" / "evidence_matrix.csv").write_text(
                evidence_matrix_csv(exclude={"C12"}),
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("evidence_matrix.csv missing required claim ids: C12" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_synced_evidence_matrix_markdown_unsupported_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index" / "evidence_matrix.md").write_text(
                evidence_matrix_markdown().replace(
                    "| C6 | Main Results | missing_evidence | C6 claim |",
                    "| C6 | Main Results | ready | C6 claim |",
                ),
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("evidence_matrix.md row" in item and "unsupported status `ready`" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_synced_rebuttal_pack_missing_matched_claims_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index" / "rebuttal_pack.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "R1",
                            "topic": "Data Contamination",
                            "question": "Could training contaminate evaluation?",
                            "defense_status": "needs_evidence",
                            "recommended_position": "Wait for C0.",
                            "matched_claims": [],
                            "readiness_ok": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertTrue(any("required paper reviewer-facing doc is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("rebuttal_pack.json row 0 missing required field `matched_claims`" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_missing_conclusion_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "sections" / "conclusion.tex").unlink()
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("missing required paper files: sections/conclusion.tex", report["hard_blockers"])

    def test_build_report_blocks_unsafe_required_evidence_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
            rows[2]["status"] = "missing_evidence"
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("required evidence claim C2 is missing_evidence", report["hard_blockers"])

    def test_build_report_blocks_raw_evidence_matrix_unsupported_status_even_for_nonrequired_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
            rows.append(evidence_claim_row("C8", "ready"))
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("evidence matrix claim C8 has unsupported status `ready`", report["hard_blockers"])

    def test_build_report_blocks_raw_evidence_matrix_non_object_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows: list[object] = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
            rows.append("bad row")
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("evidence matrix row 13 must be a JSON object", report["hard_blockers"])

    def test_build_report_blocks_raw_evidence_matrix_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
            rows[0].pop("evidence")
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("evidence matrix claim C0 missing required field `evidence`", report["hard_blockers"])

    def test_build_report_exposes_claim_ladder_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
            for row in rows:
                if row["claim_id"] == "C5":
                    row["status"] = "contradicted"
                if row["claim_id"] == "C12":
                    row["status"] = "missing_evidence"
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        by_level = {row["level"]: row for row in report["claim_ladder"]}
        self.assertEqual(by_level["motivation"]["status"], "safe_to_claim")
        self.assertEqual(by_level["metric-support"]["status"], "safe_to_claim")
        self.assertEqual(by_level["method-support"]["status"], "blocked")
        self.assertIn("C5: contradicted", by_level["method-support"]["missing_or_non_safe_claims"])
        self.assertEqual(by_level["judge-utility support"]["status"], "missing_evidence")
        self.assertIn("C12: missing_evidence", by_level["judge-utility support"]["missing_or_non_safe_claims"])

    def test_blocked_report_includes_submission_claim_discipline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper, include_synced_tables=False, include_synced_figures=False)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
            for row in rows:
                if row["claim_id"] in {"C0", "C2", "C3", "C4", "C5", "C7", "C9", "C10", "C12", "C13", "C14"}:
                    row["status"] = "missing_evidence"
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        discipline = "\n".join(report["claim_discipline"])
        self.assertFalse(report["ok"])
        self.assertIn("Do not treat this package as AAAI-ready", discipline)
        self.assertIn(
            "SFT+GRPO coverage, dimension-level recovery, downstream utility, ablation, and semantic-space claims are permitted only after",
            discipline,
        )
        self.assertIn("Do not claim clean hard-gold/proxy/downstream isolation until C0 passes", discipline)
        self.assertIn("do not describe them as dimension-level recovery until C3, C12, and C14 pass", discipline)
        self.assertIn("Do not claim downstream judge-utility support until C4/C9/C10 pass", discipline)
        self.assertIn("Do not claim ablation support until C5/C7 pass", discipline)
        self.assertIn("Treat semantic-space plots as illustrative until C13 verifies", discipline)
        self.assertIn("Do not submit the paper package until required paper-facing tables, figures, and reviewer-facing docs are synced", discipline)

    def test_safe_report_claim_discipline_limits_to_safe_claims(self) -> None:
        discipline = build_claim_discipline(
            ok=True,
            evidence_summary={"safe_to_claim": len(REQUIRED_EVIDENCE_CLAIMS), "total": len(REQUIRED_EVIDENCE_CLAIMS)},
            hard_blockers=[],
            warnings=[],
        )

        self.assertEqual(len(discipline), 1)
        self.assertIn("report only claims whose evidence rows are safe_to_claim", discipline[0])

    def test_build_report_blocks_missing_required_evidence_claim_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS if claim_id != "C10"]
            evidence.write_text(json.dumps(rows), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("required evidence claim C10 is missing from evidence matrix", report["hard_blockers"])

    def test_build_report_blocks_required_evidence_without_claim_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            evidence.write_text('[{"status": "safe_to_claim"}]', encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn(
            "evidence matrix has no claim_id fields for required evidence checks",
            report["hard_blockers"],
        )

    def test_build_report_blocks_failed_raw_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            budget = root / "budget.json"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)
            write_json(budget, {"ok": False, "blockers": ["cost too high"], "total": {"calls": 10}})

            report = build_report(
                audit,
                evidence,
                paper,
                raw_gate_specs=[f"API Budget|api_budget|{budget}"],
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["raw_gates"][0]["status"], "blocked")
        self.assertIn("raw gate API Budget is blocked", report["hard_blockers"][0])

    def test_build_report_blocks_failed_latex_compile_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            compile_report = root / "latex_compile_report.json"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)
            write_json(
                compile_report,
                {
                    "ok": False,
                    "pdf_bytes": 0,
                    "page_count": 0,
                    "max_pages": 8,
                    "official_style_active": False,
                    "submission_mode_declared": False,
                    "bibliography_style_active": False,
                    "anonymous_author_declared": False,
                    "blockers": ["missing official AAAI style file: paper/aaai2026.sty"],
                    "warnings": [],
                },
            )

            report = build_report(
                audit,
                evidence,
                paper,
                raw_gate_specs=[f"AAAI LaTeX Compile|latex_compile|{compile_report}"],
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["raw_gates"][0]["status"], "blocked")
        self.assertIn("max_pages=8", report["raw_gates"][0]["summary"])
        self.assertIn("official_style_active=False", report["raw_gates"][0]["summary"])
        self.assertIn("bibliography_style_active=False", report["raw_gates"][0]["summary"])
        self.assertIn("anonymous_author_declared=False", report["raw_gates"][0]["summary"])
        self.assertIn("raw gate AAAI LaTeX Compile is blocked", report["hard_blockers"][0])

    def test_build_report_blocks_failed_data_source_local_config_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            gate = root / "local_config_init.json"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)
            write_json(
                gate,
                {
                    "ok": False,
                    "source_overall_status": "blocked",
                    "blockers": ["rubricbench official_url missing"],
                    "warnings": [],
                },
            )

            report = build_report(
                audit,
                evidence,
                paper,
                raw_gate_specs=[f"Data Source Local Config|data_source_local_config|{gate}"],
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["raw_gates"][0]["status"], "blocked")
        self.assertIn("source_status=blocked", report["raw_gates"][0]["summary"])
        self.assertIn("raw gate Data Source Local Config is blocked", report["hard_blockers"][0])

    def test_scoped_data_source_report_ignores_unrelated_dataset_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            source = root / "source.json"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)
            write_json(
                source,
                {
                    "overall_status": "blocked",
                    "datasets": [
                        {"name": "rubricbench", "blockers": [], "warnings": []},
                        {"name": "researchrubrics", "blockers": ["researchrubrics missing"], "warnings": []},
                    ],
                    "blockers": ["researchrubrics missing"],
                    "warnings": [],
                },
            )

            report = build_report(
                audit,
                evidence,
                paper,
                raw_gate_specs=[f"Data Source Report|data_source_report[rubricbench]|{source}"],
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["raw_gates"][0]["status"], "pass")
        self.assertEqual(report["raw_gates"][0]["required_datasets"], ["rubricbench"])

    def test_scoped_data_source_report_blocks_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            write_json(source, {"overall_status": "pass", "datasets": [], "blockers": [], "warnings": []})

            gate = summarize_raw_gate(
                {"name": "Sources", "type": "data_source_report[rubricbench]", "path": str(source)}
            )

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("missing required dataset", gate["blockers"][0])

    def test_query_validation_raw_gate_blocks_when_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query = Path(tmp) / "queries.json"
            write_json(
                query,
                {"ok": False, "target": "query_pool", "n_records": 5, "blockers": ["missing data_source"], "warnings": []},
            )

            gate = summarize_raw_gate(
                {"name": "RubricBench Queries", "type": "query_validation", "path": str(query)}
            )

        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["type"], "query_validation")
        self.assertIn("missing data_source", gate["blockers"][0])

    def test_bsc_gold_sanity_raw_gate_blocks_when_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            write_json(
                summary,
                {
                    "ok": False,
                    "n_joined": 1140,
                    "mean_coverage": 1.0,
                    "blockers": ["joined records below required minimum: 1140 < 1147"],
                },
            )

            gate = summarize_raw_gate({"name": "BSC Sanity", "type": "bsc_gold_sanity", "path": str(summary)})

        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["type"], "bsc_gold_sanity")
        self.assertIn("1140 joined", gate["summary"])
        self.assertIn("joined records below required minimum", gate["blockers"][0])

    def test_minimal_api_handoff_raw_gate_blocks_when_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "api_handoff.json"
            write_json(
                handoff,
                {
                    "ok": False,
                    "status": "blocked",
                    "blockers": ["preflight: ok=false"],
                    "commands": {"paid_range_run": "python scripts/run_experiment_pipeline.py"},
                    "resume_requirements": {
                        "missing_env": ["LOCAL_OPENAI_API_KEY", "OPENAI_API_KEY"],
                        "next_command": "python3 scripts/run_experiment_pipeline.py --only preflight",
                        "offline_rerun_command": "python3 scripts/run_experiment_pipeline.py --only preflight",
                        "paid_run_command": None,
                    },
                },
            )

            gate = summarize_raw_gate(
                {"name": "API Handoff", "type": "minimal_api_handoff", "path": str(handoff)}
            )

        self.assertEqual(gate["status"], "blocked")
        self.assertEqual(gate["type"], "minimal_api_handoff")
        self.assertIn("status=blocked", gate["summary"])
        self.assertIn("missing_env=LOCAL_OPENAI_API_KEY,OPENAI_API_KEY", gate["summary"])
        self.assertIn("next=offline_rerun", gate["summary"])
        self.assertIn("preflight: ok=false", gate["blockers"])

    def test_build_report_blocks_asset_index_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index.md").write_text(
                "\n".join(
                    [
                        "# BlindSpot-RL Paper Asset Index",
                        "",
                        "- Blockers: 1",
                        "- Warnings: 0",
                        "",
                        "## Blockers",
                        "",
                        "- artifact is missing or empty: outputs/matrix_real/paper_artifacts/main_table.tex",
                        "",
                        "## Blocker Summary",
                        "",
                        "- `rl_stage_ablation`: 1 (SFT-only vs SFT+GRPO paper table)",
                        "",
                        "## Warnings",
                        "",
                        "- none",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertEqual(report["paper"]["asset_index"][0]["status"], "blocked")
        self.assertEqual(report["paper"]["asset_index"][0]["blocker_summary"][0]["category"], "rl_stage_ablation")
        self.assertIn("1 blocker categories", report["paper"]["asset_index"][0]["summary"])
        self.assertTrue(any("paper asset index is blocked" in item for item in report["hard_blockers"]))
        self.assertTrue(any("main_table.tex" in item for item in report["hard_blockers"]))

    def test_build_report_blocks_unproven_semantic_space_even_if_asset_index_has_no_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index" / "semantic_space_points.csv").write_text("x,y\n0,0\n", encoding="utf-8")
            (paper / "asset_index" / "semantic_space_summary.json").write_text('{"n_points": 1}', encoding="utf-8")
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        semantic_doc = next(
            item for item in report["paper"]["synced_docs"] if item["path"] == "asset_index/semantic_space_summary.json"
        )
        self.assertEqual(semantic_doc["status"], "blocked")
        self.assertTrue(
            any("embedding_model must be BAAI/bge-large-en-v1.5" in blocker for blocker in semantic_doc["blockers"])
        )
        self.assertTrue(any("point_csv_columns do not match schema" in blocker for blocker in semantic_doc["blockers"]))
        self.assertTrue(any("semantic_space_summary.json" in item for item in report["hard_blockers"]))

    def test_build_report_warns_on_asset_index_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            (paper / "asset_index.md").write_text(
                "\n".join(
                    [
                        "# BlindSpot-RL Paper Asset Index",
                        "",
                        "- Blockers: 0",
                        "- Warnings: 1",
                        "",
                        "## Blockers",
                        "",
                        "- none",
                        "",
                        "## Warnings",
                        "",
                        "- artifact is missing or empty: outputs/matrix_real/paper_artifacts/ablation_table.tex",
                        "",
                        "## Warning Summary",
                        "",
                        "- `api_handoff`: 2 (API handoff reviewer-facing docs)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            audit.write_text('{"ok": true}', encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertTrue(report["ok"])
        self.assertEqual(report["paper"]["asset_index"][0]["status"], "warn")
        self.assertEqual(report["paper"]["asset_index"][0]["warning_summary"][0]["category"], "api_handoff")
        self.assertIn("1 warning categories", report["paper"]["asset_index"][0]["summary"])
        self.assertTrue(any("paper asset index has warnings" in item for item in report["warnings"]))

    def test_asset_index_check_blocks_missing_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = asset_index_check(Path(tmp), "asset_index.md")

        self.assertEqual(item["status"], "missing")
        self.assertIn("missing paper asset index", item["blockers"][0])

    def test_build_report_blocks_invalid_audit_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text("{bad", encoding="utf-8")
            write_safe_required_evidence(evidence)

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("audit report is not readable", report["hard_blockers"][0])

    def test_build_report_blocks_invalid_evidence_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            evidence.write_text("{bad", encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("evidence matrix is not readable", report["hard_blockers"][0])

    def test_summarize_raw_gate_blocks_invalid_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.json"
            path.write_text("{bad", encoding="utf-8")

            gate = summarize_raw_gate({"name": "API Budget", "type": "api_budget", "path": str(path)})

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("report is not readable", gate["summary"])

    def test_generic_raw_gate_accepts_jsonl_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.jsonl"
            path.write_text('{"ok": true}\n{"ok": false}\n', encoding="utf-8")

            gate = summarize_raw_gate({"name": "Stats", "type": "generic", "path": str(path)})

        self.assertEqual(gate["status"], "pass")
        self.assertIn("artifact present", gate["summary"])

    def test_contamination_audit_raw_gate_blocks_overlap_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contamination.json"
            write_json(
                path,
                {
                    "ok": True,
                    "overlap_query_count": 2,
                    "blockers": [],
                    "warnings": ["sample overlap examples truncated"],
                },
            )

            gate = summarize_raw_gate({"name": "Hard-Gold Audit", "type": "contamination_audit", "path": str(path)})

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("overlaps=2", gate["summary"])
        self.assertIn("2 overlapping holdout query(s) found", gate["blockers"])
        self.assertEqual(gate["warnings"], ["sample overlap examples truncated"])

    def test_contamination_audit_raw_gate_blocks_ok_false_filter_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filter.json"
            write_json(
                path,
                {
                    "ok": False,
                    "removed_records": 1,
                    "blockers": ["output file is missing"],
                    "warnings": [],
                },
            )

            gate = summarize_raw_gate({"name": "Proxy Filter", "type": "contamination_audit", "path": str(path)})

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("removed=1", gate["summary"])
        self.assertIn("output file is missing", gate["blockers"])

    def test_contamination_audit_raw_gate_separates_missing_artifacts_from_clean_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contamination.json"
            write_json(
                path,
                {
                    "ok": False,
                    "artifact_status": "blocked",
                    "overlap_status": "not_auditable",
                    "overlap_query_count": 0,
                    "blockers": ["sft: missing or empty file"],
                    "warnings": [],
                },
            )

            gate = summarize_raw_gate({"name": "Hard-Gold Audit", "type": "contamination_audit", "path": str(path)})

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("artifact_status=blocked", gate["summary"])
        self.assertIn("overlap_status=not_auditable", gate["summary"])
        self.assertIn("overlaps=0", gate["summary"])
        self.assertIn("sft: missing or empty file", gate["blockers"])

    def test_contamination_audit_raw_gate_accepts_clean_filter_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filter.json"
            write_json(
                path,
                {
                    "ok": True,
                    "removed_records": 1,
                    "blockers": [],
                    "warnings": [],
                },
            )

            gate = summarize_raw_gate({"name": "Proxy Filter", "type": "contamination_audit", "path": str(path)})

        self.assertEqual(gate["status"], "pass")
        self.assertIn("removed=1", gate["summary"])

    def test_parse_raw_gate_spec_requires_name_type_path(self) -> None:
        self.assertEqual(
            parse_raw_gate_spec("Data Source|data_source_report|outputs/source.json"),
            {"name": "Data Source", "type": "data_source_report", "path": "outputs/source.json"},
        )
        with self.assertRaises(ValueError):
            parse_raw_gate_spec("bad-spec")

    def test_build_report_blocks_contradicted_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            paper = root / "paper"
            write_required_paper_files(paper)
            audit.write_text('{"ok": true}', encoding="utf-8")
            evidence.write_text(json.dumps([evidence_claim_row("C0", "contradicted")]), encoding="utf-8")

            report = build_report(audit, evidence, paper)

        self.assertFalse(report["ok"])
        self.assertIn("one or more claims are contradicted", report["hard_blockers"])

    def test_to_markdown_reports_blockers(self) -> None:
        text = to_markdown(
            {
                "ok": False,
                "audit_ok": False,
                "evidence": {"safe_to_claim": 0, "total": 1, "missing_evidence": 1, "contradicted": 0},
                "raw_gates": [
                    {"name": "API Budget", "type": "api_budget", "status": "blocked", "summary": "0 calls"}
                ],
                "hard_blockers": ["audit report is not ok"],
                "warnings": [],
                "claim_ladder": [
                    {
                        "level": "motivation",
                        "status": "missing_evidence",
                        "required_claim_ids": ["C1", "C6"],
                        "missing_or_non_safe_claims": ["C1: missing_evidence", "C6: missing_evidence"],
                    }
                ],
                "claim_discipline": ["Do not treat this package as AAAI-ready while submission readiness is false."],
                "paper": {
                    "required": [],
                    "synced_tables": [],
                    "synced_figures": [],
                    "synced_docs": [],
                    "asset_index": [
                        {
                            "path": "asset_index.md",
                            "present": True,
                            "status": "blocked",
                            "bytes": 100,
                            "blocker_summary": [
                                {
                                    "category": "semantic_space",
                                    "count": 4,
                                    "label": "semantic-space SVG/PDF/CSV/JSON assets",
                                }
                            ],
                        }
                    ],
                },
            }
        )
        self.assertIn("- Status: `blocked`", text)
        self.assertIn("- Hard blockers: `1`", text)
        self.assertIn("- Hard blocker categories: `", text)
        self.assertIn("- Warnings: `0`", text)
        self.assertIn("audit report is not ok", text)
        self.assertIn("## Claim Ladder Status", text)
        self.assertIn("| motivation | `missing_evidence` | C1, C6 | C1: missing_evidence; C6: missing_evidence |", text)
        self.assertIn("## Claim Discipline", text)
        self.assertIn("Do not treat this package as AAAI-ready", text)
        self.assertIn("Raw Gates", text)
        self.assertIn("API Budget", text)
        self.assertIn("## Paper Asset Blocker Summary", text)
        self.assertIn("`semantic_space`: 4", text)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def evidence_claim_row(claim_id: str, status: str) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "claim": f"{claim_id} claim",
        "paper_section": "Main Results",
        "status": status,
        "evidence": f"{claim_id} evidence",
        "notes": "",
    }


def write_safe_required_evidence(path: Path) -> None:
    rows = [evidence_claim_row(claim_id, "safe_to_claim") for claim_id in REQUIRED_EVIDENCE_CLAIMS]
    path.write_text(json.dumps(rows), encoding="utf-8")


def write_required_paper_files(
    paper: Path,
    include_synced_tables: bool = True,
    include_synced_figures: bool = True,
    include_synced_docs: bool = True,
) -> None:
    for path in REQUIRED_PAPER_FILES:
        full_path = paper / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("x", encoding="utf-8")
    if include_synced_tables:
        for path in REQUIRED_SYNCED_TABLES:
            full_path = paper / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("table", encoding="utf-8")
    if include_synced_figures:
        for path in REQUIRED_SYNCED_FIGURES:
            full_path = paper / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("figure", encoding="utf-8")
        write_valid_semantic_space_bundle(paper)
    if include_synced_docs:
        for path in REQUIRED_SYNCED_DOCS:
            if path in {"asset_index/semantic_space_points.csv", "asset_index/semantic_space_summary.json"}:
                if not include_synced_figures:
                    continue
                if not (paper / path).exists():
                    write_valid_semantic_space_bundle(paper)
                continue
            full_path = paper / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(synced_doc_placeholder(path), encoding="utf-8")
    (paper / "asset_index.md").write_text(
        "\n".join(
            [
                "# BlindSpot-RL Paper Asset Index",
                "",
                "- Blockers: 0",
                "- Warnings: 0",
                "",
                "## Blockers",
                "",
                "- none",
                "",
                "## Warnings",
                "",
                "- none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def synced_doc_placeholder(path: str) -> str:
    if path == "asset_index/real_run_dashboard.json":
        return json.dumps({"sections": [{"type": "rebuttal_manifest", "metrics": {"claim_ladder_safe": 0}}]})
    if path == "asset_index/real_run_dashboard.md":
        return "claim_ladder_safe=0/4\n"
    if path == "asset_index/evidence_matrix.json":
        return json.dumps(evidence_matrix_rows())
    if path == "asset_index/evidence_matrix.csv":
        return evidence_matrix_csv()
    if path == "asset_index/evidence_matrix.md":
        return evidence_matrix_markdown()
    if path == "asset_index/result_card.json":
        return json.dumps({"claim_ladder": [{"level": "motivation"}], "dashboard_diagnostics": [{"name": "dash"}]})
    if path == "asset_index/result_card.md":
        return "## Dashboard Diagnostics\n\n## Claim Ladder\n"
    if path == "asset_index/submission_gap_report.json":
        return json.dumps(submission_gap_report_placeholder())
    if path == "asset_index/submission_gap_report.md":
        return (
            "## Claim Ladder Status\n\n"
            "## Execution Sequence\n\n"
            "- Summary: phase_status=blocked; ready_to_run=false\n\n"
            "- Commands:\n\n"
            "- Unlocks: `outputs/contamination_audit/*`\n\n"
            "- Missing prerequisites: none\n\n"
            "### Paper Asset Warnings\n"
        )
    if path == "asset_index/readiness_report.json":
        return json.dumps({"claim_ladder": [{"level": "motivation"}], "claim_discipline": ["gate claims"]})
    if path == "asset_index/readiness_report.md":
        return "## Claim Ladder Status\n\n## Claim Discipline\n"
    if path == "asset_index/rebuttal_pack.json":
        return json.dumps(
            [
                {
                    "id": "R8",
                    "topic": "Data Contamination",
                    "question": "Could training contaminate evaluation?",
                    "defense_status": "needs_evidence",
                    "recommended_position": "Require C0 before answering.",
                    "matched_claims": [{"claim_id": "C0", "status": "missing_evidence"}],
                    "readiness_ok": False,
                }
            ]
        )
    if path == "asset_index/rebuttal_pack_manifest.json":
        return json.dumps({"claim_ladder": [{"level": "motivation"}], "defense_status_counts": {"needs_evidence": 1}})
    if path == "asset_index/rebuttal_pack.md":
        return "## Claim Ladder Status\n"
    return "doc"


def write_valid_semantic_space_bundle(paper: Path) -> None:
    figures = paper / "figures"
    asset_index = paper / "asset_index"
    figures.mkdir(parents=True, exist_ok=True)
    asset_index.mkdir(parents=True, exist_ok=True)
    svg = figures / "semantic_space.svg"
    pdf = figures / "semantic_space.pdf"
    points = asset_index / "semantic_space_points.csv"
    summary = asset_index / "semantic_space_summary.json"
    svg.write_text("<svg></svg>", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    row = {
        "point_id": "g0",
        "record_idx": "0",
        "method": "gold",
        "source_type": "gold",
        "category": "helpfulness",
        "gold_cluster_id": "0",
        "rubric_idx": "0",
        "x": "0.0",
        "y": "0.0",
        "nearest_gold_point_id": "g0",
        "nearest_gold_record_idx": "0",
        "nearest_gold_rubric_idx": "0",
        "nearest_gold_category": "helpfulness",
        "nearest_gold_cluster_id": "0",
        "nearest_gold_similarity": "1.0",
        "nearest_gold_same_record": "true",
        "query": "q",
        "text": "gold criterion",
        "nearest_gold_text": "gold criterion",
    }
    with points.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEMANTIC_POINT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        for method, point_id in [("base", "b0"), ("sft_only", "s0"), ("sft_rl", "r0")]:
            generated = dict(row)
            generated.update(
                {
                    "point_id": point_id,
                    "method": method,
                    "source_type": "generated",
                    "text": f"{method} criterion",
                }
            )
            writer.writerow(generated)
    summary.write_text(
        json.dumps(
            {
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "requested_projection": "umap",
                "projection": "umap",
                "gold_cluster_tau": 0.75,
                "point_csv_schema_version": 3,
                "point_csv_columns": SEMANTIC_POINT_CSV_COLUMNS,
                "output_artifacts_schema_version": 1,
                "point_csv_rows_match_n_points": True,
                "methods": ["base", "sft_only", "sft_rl"],
                "n_points": 4,
                "n_gold": 1,
                "n_generated": 3,
                "n_gold_clusters": 1,
                "generated_gold_category_coverage_by_method": {"base": 1, "sft_only": 1, "sft_rl": 1},
                "nearest_gold_category_coverage_by_method": {"base": 1, "sft_only": 1, "sft_rl": 1},
                "nearest_gold_cluster_coverage_by_method": {"base": 1, "sft_only": 1, "sft_rl": 1},
                "nearest_gold_cluster_distribution_by_method": {
                    "base": {"0": 1},
                    "sft_only": {"0": 1},
                    "sft_rl": {"0": 1},
                },
                "nearest_gold_cluster_entropy_by_method": {"base": 0, "sft_only": 0, "sft_rl": 0},
                "generated_dispersion_by_method": {"base": 0, "sft_only": 0, "sft_rl": 0},
                "mean_nearest_gold_similarity_by_method": {"base": 1, "sft_only": 1, "sft_rl": 1},
                "inputs": [
                    {
                        "label": method,
                        "path": f"data/processed/matrix_real/{method}/bsc_eval.jsonl",
                        "sha256": "abc",
                        "n_records": 1,
                        "join_report": {
                            "join_key": "query",
                            "gold": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                            "output": f"data/processed/matrix_real/{method}/bsc_eval.jsonl",
                            "output_sha256": "abc",
                            "n_joined": 1,
                        },
                    }
                    for method in ["base", "sft_only", "sft_rl"]
                ],
                "point_csv_sha256": sha256(points),
                "svg_sha256": sha256(svg),
                "pdf_sha256": sha256(pdf),
            }
        ),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_matrix_rows(exclude: set[str] | None = None) -> list[dict[str, str]]:
    exclude = exclude or set()
    return [
        {
            "claim_id": claim_id,
            "claim": f"{claim_id} claim",
            "paper_section": "Main Results",
            "status": "missing_evidence",
            "evidence": "[missing] artifact",
            "notes": "",
        }
        for claim_id in REQUIRED_EVIDENCE_CLAIMS
        if claim_id not in exclude
    ]


def evidence_matrix_csv(exclude: set[str] | None = None) -> str:
    lines = ["claim_id,claim,paper_section,status,evidence,notes"]
    for row in evidence_matrix_rows(exclude=exclude):
        lines.append(
            ",".join(
                [
                    row["claim_id"],
                    row["claim"],
                    row["paper_section"],
                    row["status"],
                    row["evidence"],
                    row["notes"],
                ]
            )
        )
    return "\n".join(lines) + "\n"


def evidence_matrix_markdown(exclude: set[str] | None = None) -> str:
    body = "\n".join(
        f"| {row['claim_id']} | {row['paper_section']} | {row['status']} | {row['claim']} | {row['evidence']} |"
        for row in evidence_matrix_rows(exclude=exclude)
    )
    return f"""| Claim ID | Section | Status | Claim | Evidence |
| --- | --- | --- | --- | --- |
{body}
"""


def submission_gap_report_placeholder() -> dict[str, object]:
    return {
        "claim_ladder": [{"level": "motivation"}],
        "operator_handoff": {"status": "ready"},
        "phases": [
            {
                "id": "data_isolation_hard_gold",
                "summary": "blocked: claim_gaps=1",
                "paper_asset_warnings": [],
            }
        ],
        "execution_sequence": [
            {
                "id": "data_isolation",
                "phase_id": "data_isolation_hard_gold",
                "commands": ["python3 scripts/run_experiment_pipeline.py --config configs/pipeline_real_run.generated.json"],
                "unlocks": ["outputs/contamination_audit/*"],
                "evidence_gates": ["C0"],
                "missing_prerequisites": {},
                "summary": "phase_status=blocked; evidence_gates=C0; ready_to_run=false",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
