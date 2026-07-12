from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_paper_asset_index import (
    ALLOWED_EVIDENCE_STATUSES,
    REQUIRED_EVIDENCE_CLAIM_IDS,
    SEMANTIC_POINT_CSV_COLUMNS,
    build_report,
    file_sha256,
    parse_section_items,
    parse_synced_artifact_rows,
    semantic_space_contract_blockers,
    to_markdown,
)


class CheckPaperAssetIndexTest(unittest.TestCase):
    def test_required_evidence_claim_ids_cover_core_aaai_story(self) -> None:
        for claim_id in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10", "C12", "C13", "C14"]:
            self.assertIn(claim_id, REQUIRED_EVIDENCE_CLAIM_IDS)

    def test_allowed_evidence_statuses_cover_claim_ladder_lifecycle(self) -> None:
        for status in ["safe_to_claim", "missing_evidence", "contradicted", "not_yet_supported"]:
            self.assertIn(status, ALLOWED_EVIDENCE_STATUSES)

    def test_build_report_passes_when_paths_and_sha_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text('{"ok": true}', encoding="utf-8")
            paper.write_text('{"ok": true}', encoding="utf-8")
            sha = file_sha256(paper)
            index.write_text(asset_index_text(sha=sha), encoding="utf-8")

            report = build_report(index, root=root)

        self.assertTrue(report["ok"])
        self.assertEqual(report["declared_count"], 1)
        self.assertEqual(report["actual_count"], 1)
        self.assertEqual(report["blockers"], [])
        self.assertTrue(report["checks"][0]["sha256_matches"])

    def test_build_report_blocks_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text('{"ok": true}', encoding="utf-8")
            paper.write_text('{"ok": false}', encoding="utf-8")
            index.write_text(asset_index_text(sha="0" * 64), encoding="utf-8")

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertIn("SHA256 mismatch", report["blockers"][0])

    def test_build_report_blocks_missing_paper_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            index.parent.mkdir(parents=True)
            source.write_text('{"ok": true}', encoding="utf-8")
            index.write_text(asset_index_text(sha="0" * 64), encoding="utf-8")

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(any("paper artifact is missing" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_result_card_missing_claim_ladder_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "result_card.json"
            paper = root / "paper" / "asset_index" / "result_card.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text('{"ok": false}', encoding="utf-8")
            paper.write_text('{"ok": false}', encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/result_card.json",
                    paper_path="paper/asset_index/result_card.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("missing JSON field `claim_ladder`" in item for item in report["blockers"]))

    def test_build_report_accepts_indexed_readiness_report_claim_ladder_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "readiness_report.json"
            paper = root / "paper" / "asset_index" / "readiness_report.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = '{"claim_ladder":[{"level":"motivation"}],"claim_discipline":["gate claims"]}'
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/readiness_report.json",
                    paper_path="paper/asset_index/readiness_report.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertTrue(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])

    def test_build_report_blocks_indexed_gap_report_missing_phase_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "submission_gap_report.json"
            paper = root / "paper" / "asset_index" / "submission_gap_report.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = submission_gap_report_json()
            content["phases"][0].pop("summary")
            serialized = json.dumps(content)
            source.write_text(serialized, encoding="utf-8")
            paper.write_text(serialized, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/submission_gap_report.json",
                    paper_path="paper/asset_index/submission_gap_report.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("phase 0 missing `summary`" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_gap_report_missing_paper_asset_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "submission_gap_report.json"
            paper = root / "paper" / "asset_index" / "submission_gap_report.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = submission_gap_report_json()
            content["phases"][0].pop("paper_asset_warnings")
            serialized = json.dumps(content)
            source.write_text(serialized, encoding="utf-8")
            paper.write_text(serialized, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/submission_gap_report.json",
                    paper_path="paper/asset_index/submission_gap_report.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("phase 0 missing `paper_asset_warnings` list" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_gap_report_missing_execution_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "submission_gap_report.json"
            paper = root / "paper" / "asset_index" / "submission_gap_report.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = submission_gap_report_json()
            content["execution_sequence"][0].pop("commands")
            serialized = json.dumps(content)
            source.write_text(serialized, encoding="utf-8")
            paper.write_text(serialized, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/submission_gap_report.json",
                    paper_path="paper/asset_index/submission_gap_report.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("execution step 0 missing non-empty `commands` list" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_gap_report_markdown_missing_ready_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "submission_gap_report.md"
            paper = root / "paper" / "asset_index" / "submission_gap_report.md"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = (
                "## Claim Ladder Status\n\n"
                "## Execution Sequence\n\n"
                "- Summary: phase_status=blocked\n\n"
                "### Paper Asset Warnings\n"
            )
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/submission_gap_report.md",
                    paper_path="paper/asset_index/submission_gap_report.md",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("missing text `ready_to_run=`" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_evidence_matrix_missing_required_claim_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "evidence_matrix.json"
            paper = root / "paper" / "asset_index" / "evidence_matrix.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = json.dumps(evidence_matrix_rows(exclude={"C14"}))
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/evidence_matrix.json",
                    paper_path="paper/asset_index/evidence_matrix.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("missing required claim ids C14" in item for item in report["blockers"]))

    def test_build_report_accepts_indexed_evidence_matrix_json_and_markdown_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_json = root / "outputs" / "evidence_matrix.json"
            paper_json = root / "paper" / "asset_index" / "evidence_matrix.json"
            source_md = root / "outputs" / "evidence_matrix.md"
            paper_md = root / "paper" / "asset_index" / "evidence_matrix.md"
            index = root / "paper" / "asset_index.md"
            source_json.parent.mkdir(parents=True)
            paper_json.parent.mkdir(parents=True)
            rows = evidence_matrix_rows()
            json_content = json.dumps(rows)
            md_content = evidence_matrix_markdown(rows)
            source_json.write_text(json_content, encoding="utf-8")
            paper_json.write_text(json_content, encoding="utf-8")
            source_md.write_text(md_content, encoding="utf-8")
            paper_md.write_text(md_content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_rows(
                    [
                        ("outputs/evidence_matrix.json", "paper/asset_index/evidence_matrix.json", file_sha256(paper_json)),
                        ("outputs/evidence_matrix.md", "paper/asset_index/evidence_matrix.md", file_sha256(paper_md)),
                    ]
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertTrue(report["ok"])
        self.assertEqual(report["actual_count"], 2)
        self.assertTrue(all(item["contract_checked"] for item in report["checks"]))

    def test_build_report_accepts_indexed_evidence_matrix_csv_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "evidence_matrix.csv"
            paper = root / "paper" / "asset_index" / "evidence_matrix.csv"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = evidence_matrix_csv(evidence_matrix_rows())
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/evidence_matrix.csv",
                    paper_path="paper/asset_index/evidence_matrix.csv",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertTrue(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])

    def test_build_report_blocks_indexed_evidence_matrix_csv_missing_required_claim_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "evidence_matrix.csv"
            paper = root / "paper" / "asset_index" / "evidence_matrix.csv"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = evidence_matrix_csv(evidence_matrix_rows(exclude={"C12"}))
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/evidence_matrix.csv",
                    paper_path="paper/asset_index/evidence_matrix.csv",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("missing required claim ids C12" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_evidence_matrix_csv_unsupported_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "evidence_matrix.csv"
            paper = root / "paper" / "asset_index" / "evidence_matrix.csv"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = evidence_matrix_csv(evidence_matrix_rows()).replace(
                "C6,C6 claim,Main Results,missing_evidence,",
                "C6,C6 claim,Main Results,ready,",
            )
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/evidence_matrix.csv",
                    paper_path="paper/asset_index/evidence_matrix.csv",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("unsupported status `ready`" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_rebuttal_pack_missing_matched_claims_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "rebuttal_pack.json"
            paper = root / "paper" / "asset_index" / "rebuttal_pack.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            content = json.dumps(
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
            )
            source.write_text(content, encoding="utf-8")
            paper.write_text(content, encoding="utf-8")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/rebuttal_pack.json",
                    paper_path="paper/asset_index/rebuttal_pack.json",
                    sha=file_sha256(paper),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("row 0 missing `matched_claims`" in item for item in report["blockers"]))

    def test_build_report_blocks_indexed_semantic_space_without_provenance_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "paper_artifacts" / "semantic_space_summary.json"
            paper_summary = root / "paper" / "asset_index" / "semantic_space_summary.json"
            paper_points = root / "paper" / "asset_index" / "semantic_space_points.csv"
            paper_svg = root / "paper" / "figures" / "semantic_space.svg"
            paper_pdf = root / "paper" / "figures" / "semantic_space.pdf"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper_summary.parent.mkdir(parents=True)
            paper_svg.parent.mkdir(parents=True)
            source.write_text('{"n_points": 1}', encoding="utf-8")
            paper_summary.write_text('{"n_points": 1}', encoding="utf-8")
            paper_points.write_text("x,y\n0,0\n", encoding="utf-8")
            paper_svg.write_text("<svg></svg>", encoding="utf-8")
            paper_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            index.write_text(
                asset_index_text_for_row(
                    source="outputs/paper_artifacts/semantic_space_summary.json",
                    paper_path="paper/asset_index/semantic_space_summary.json",
                    sha=file_sha256(paper_summary),
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"][0]["contract_checked"])
        self.assertTrue(any("embedding_model must be BAAI/bge-large-en-v1.5" in item for item in report["blockers"]))
        self.assertTrue(any("point CSV columns do not match schema" in item for item in report["blockers"]))

    def test_semantic_space_contract_requires_all_method_level_points_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "semantic_space_summary.json"
            points = root / "semantic_space_points.csv"
            svg = root / "semantic_space.svg"
            pdf = root / "semantic_space.pdf"
            svg.write_text("<svg></svg>", encoding="utf-8")
            pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            points.write_text(
                ",".join(SEMANTIC_POINT_CSV_COLUMNS)
                + "\n"
                + semantic_point_row(method="human_gold", source_type="gold", point_id="0")
                + "\n"
                + semantic_point_row(method="sft_rl", source_type="generated", point_id="1")
                + "\n",
                encoding="utf-8",
            )
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
                        "n_points": 2,
                        "n_gold": 1,
                        "n_generated": 1,
                        "n_gold_clusters": 1,
                        "generated_gold_category_coverage_by_method": {"sft_rl": 1.0},
                        "nearest_gold_category_coverage_by_method": {"sft_rl": 1.0},
                        "nearest_gold_cluster_coverage_by_method": {"sft_rl": 1.0},
                        "nearest_gold_cluster_distribution_by_method": {"sft_rl": {"g000": 1}},
                        "nearest_gold_cluster_entropy_by_method": {"sft_rl": 0.0},
                        "generated_dispersion_by_method": {"sft_rl": 0.0},
                        "mean_nearest_gold_similarity_by_method": {"sft_rl": 0.9},
                        "inputs": [
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
                        ],
                        "point_csv_sha256": file_sha256(points),
                        "svg_sha256": file_sha256(svg),
                        "pdf_sha256": file_sha256(pdf),
                    }
                ),
                encoding="utf-8",
            )

            blockers = semantic_space_contract_blockers(
                summary_path=summary,
                points_path=points,
                svg_path=svg,
                pdf_path=pdf,
                paper_path="paper/asset_index/semantic_space_summary.json",
            )

        self.assertTrue(any("point CSV missing generated points for base, sft_only" in item for item in blockers))
        self.assertTrue(
            any("`nearest_gold_cluster_coverage_by_method` missing methods base, sft_only" in item for item in blockers)
        )

    def test_build_report_blocks_declared_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text("x", encoding="utf-8")
            paper.write_text("x", encoding="utf-8")
            index.write_text(asset_index_text(sha=file_sha256(paper), declared_count=2), encoding="utf-8")

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertIn("synced artifact count mismatch", report["blockers"][0])

    def test_build_report_blocks_asset_index_declared_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text("x", encoding="utf-8")
            paper.write_text("x", encoding="utf-8")
            index.write_text(
                asset_index_text(
                    sha=file_sha256(paper),
                    blockers=["artifact is missing or empty: outputs/main_table.tex"],
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertFalse(report["ok"])
        self.assertIn("asset index declares blocker", report["blockers"][0])

    def test_build_report_summarizes_paper_facing_asset_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text("x", encoding="utf-8")
            paper.write_text("x", encoding="utf-8")
            index.write_text(
                asset_index_text(
                    sha=file_sha256(paper),
                    blockers=[
                        "artifact is missing or empty: outputs/main_table.tex",
                        "artifact is missing or empty: outputs/rl_stage_ablation_table.tex",
                        "artifact is missing or empty: outputs/downstream_utility_table.tex",
                        "artifact is missing or empty: outputs/ablation_table.tex",
                        "artifact is missing or empty: outputs/teacher_union_ablation_table.tex",
                        "artifact is missing or empty: outputs/verifier_filter_ablation_table.tex",
                        "artifact is missing or empty: outputs/dimension_transition_table.tex",
                        "artifact is missing or empty: outputs/semantic_space.svg",
                        "artifact is missing or empty: outputs/semantic_space.pdf",
                        "artifact is missing or empty: outputs/semantic_space_points.csv",
                        "artifact is missing or empty: outputs/semantic_space_summary.json",
                        "artifact is missing or empty: outputs/experiment_summary.md",
                    ],
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)
            markdown = to_markdown(report)

        categories = {item["category"]: item for item in report["blocker_summary"]}
        self.assertEqual(categories["main_bsc_table"]["count"], 1)
        self.assertEqual(categories["rl_stage_ablation"]["count"], 1)
        self.assertEqual(categories["downstream_utility"]["count"], 1)
        self.assertEqual(categories["reward_ablation"]["count"], 1)
        self.assertEqual(categories["teacher_union_ablation"]["count"], 1)
        self.assertEqual(categories["verifier_filter_ablation"]["count"], 1)
        self.assertEqual(categories["dimension_transition"]["count"], 1)
        self.assertEqual(categories["semantic_space"]["count"], 4)
        self.assertEqual(categories["experiment_summary"]["count"], 1)
        self.assertNotIn("uncategorized", categories)
        self.assertIn("## Blocker Summary", markdown)
        self.assertIn("semantic_space", markdown)

    def test_build_report_preserves_asset_index_warnings_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text("x", encoding="utf-8")
            paper.write_text("x", encoding="utf-8")
            index.write_text(
                asset_index_text(
                    sha=file_sha256(paper),
                    warnings=["artifact is missing or empty: outputs/ablation_table.tex"],
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)

        self.assertTrue(report["ok"])
        self.assertIn("asset index declares warning", report["warnings"][0])

    def test_build_report_summarizes_asset_index_warnings_by_artifact_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "outputs" / "artifact.json"
            paper = root / "paper" / "asset_index" / "artifact.json"
            index = root / "paper" / "asset_index.md"
            source.parent.mkdir(parents=True)
            paper.parent.mkdir(parents=True)
            source.write_text("x", encoding="utf-8")
            paper.write_text("x", encoding="utf-8")
            index.write_text(
                asset_index_text(
                    sha=file_sha256(paper),
                    warnings=[
                        "artifact is missing or empty: outputs/paper_artifacts/api_handoff.json",
                        "artifact is missing or empty: outputs/paper_artifacts/api_handoff.md",
                        "artifact is missing or empty: outputs/paper_artifacts/audit_report.json",
                    ],
                ),
                encoding="utf-8",
            )

            report = build_report(index, root=root)
            markdown = to_markdown(report)

        categories = {item["category"]: item for item in report["warning_summary"]}
        self.assertTrue(report["ok"])
        self.assertEqual(categories["api_handoff"]["count"], 2)
        self.assertEqual(categories["audit_report"]["count"], 1)
        self.assertNotIn("uncategorized", categories)
        self.assertIn("## Warning Summary", markdown)
        self.assertIn("`api_handoff`: 2", markdown)

    def test_parse_rows_and_markdown_summary(self) -> None:
        rows = parse_synced_artifact_rows(asset_index_text(sha="a" * 64))
        report = {"ok": False, "asset_index": "paper/asset_index.md", "declared_count": 1, "actual_count": 1, "checks": [], "blockers": ["bad"], "blocker_summary": [], "warnings": [], "warning_summary": []}

        self.assertEqual(rows[0]["paper_path"], "paper/asset_index/artifact.json")
        self.assertIn("Paper Asset Index Check", to_markdown(report))
        self.assertEqual(parse_section_items(asset_index_text(sha="a" * 64, blockers=["bad"]), "Blockers"), ["bad"])


def asset_index_text(
    sha: str,
    declared_count: int = 1,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
) -> str:
    blockers = blockers or []
    warnings = warnings or []
    blocker_lines = "\n".join(f"- {item}" for item in blockers) if blockers else "- none"
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- none"
    return f"""# BlindSpot-RL Paper Asset Index

- Synced artifacts: {declared_count}
- Blockers: {len(blockers)}
- Warnings: {len(warnings)}

## Blockers

{blocker_lines}

## Warnings

{warning_lines}

## Synced Artifacts

| Kind | Source | Paper Path | SHA256 |
| --- | --- | --- | --- |
| doc | `outputs/artifact.json` | `paper/asset_index/artifact.json` | `{sha}` |
"""


def asset_index_text_for_row(source: str, paper_path: str, sha: str) -> str:
    return f"""# BlindSpot-RL Paper Asset Index

- Synced artifacts: 1
- Blockers: 0
- Warnings: 0

## Blockers

- none

## Warnings

- none

## Synced Artifacts

| Kind | Source | Paper Path | SHA256 |
| --- | --- | --- | --- |
| doc | `{source}` | `{paper_path}` | `{sha}` |
"""


def asset_index_text_for_rows(rows: list[tuple[str, str, str]]) -> str:
    table_rows = "\n".join(f"| doc | `{source}` | `{paper_path}` | `{sha}` |" for source, paper_path, sha in rows)
    return f"""# BlindSpot-RL Paper Asset Index

- Synced artifacts: {len(rows)}
- Blockers: 0
- Warnings: 0

## Blockers

- none

## Warnings

- none

## Synced Artifacts

| Kind | Source | Paper Path | SHA256 |
| --- | --- | --- | --- |
{table_rows}
"""


def evidence_matrix_rows(exclude: set[str] | None = None) -> list[dict[str, str]]:
    exclude = exclude or set()
    claim_ids = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10", "C12", "C13", "C14"]
    return [
        {
            "claim_id": claim_id,
            "claim": f"{claim_id} claim",
            "paper_section": "Main Results",
            "status": "missing_evidence",
            "evidence": "[missing] artifact",
        }
        for claim_id in claim_ids
        if claim_id not in exclude
    ]


def evidence_matrix_markdown(rows: list[dict[str, str]]) -> str:
    body = "\n".join(
        f"| {row['claim_id']} | {row['paper_section']} | {row['status']} | {row['claim']} | {row['evidence']} |"
        for row in rows
    )
    return f"""| Claim ID | Section | Status | Claim | Evidence |
| --- | --- | --- | --- | --- |
{body}
"""


def evidence_matrix_csv(rows: list[dict[str, str]]) -> str:
    lines = ["claim_id,claim,paper_section,status,evidence,notes"]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["claim_id"],
                    row["claim"],
                    row["paper_section"],
                    row["status"],
                    row["evidence"],
                    row.get("notes", ""),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def semantic_point_row(method: str, source_type: str, point_id: str) -> str:
    values = {
        "point_id": point_id,
        "record_idx": "0",
        "method": method,
        "source_type": source_type,
        "category": "evidence_grounding",
        "gold_cluster_id": "g000" if source_type == "gold" else "",
        "rubric_idx": "0",
        "x": "0.0",
        "y": "0.0",
        "nearest_gold_point_id": "0" if source_type == "generated" else "",
        "nearest_gold_record_idx": "0" if source_type == "generated" else "",
        "nearest_gold_rubric_idx": "0" if source_type == "generated" else "",
        "nearest_gold_category": "evidence_grounding" if source_type == "generated" else "",
        "nearest_gold_cluster_id": "g000" if source_type == "generated" else "",
        "nearest_gold_similarity": "0.9000" if source_type == "generated" else "",
        "nearest_gold_same_record": "true" if source_type == "generated" else "",
        "query": "q",
        "text": "Uses evidence",
        "nearest_gold_text": "Uses evidence" if source_type == "generated" else "",
    }
    return ",".join(values[column] for column in SEMANTIC_POINT_CSV_COLUMNS)


def submission_gap_report_json() -> dict[str, object]:
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
