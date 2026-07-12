from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_result_card import build_result_card, default_claim_ladder, load_config, to_markdown


class BuildResultCardTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_result_card_config.json"))

        self.assertIn("Result Card config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Result Card config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Result Card config must be a JSON object", str(context.exception))

    def test_safe_card_from_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = write_json(root / "gold.json", {"ok": True, "n_records": 100, "blockers": [], "warnings": []})
            bsc = write_json(root / "bsc.json", {"n": 100, "mean_coverage": 0.7, "mean_blind": 0.3})
            downstream = write_json(
                root / "downstream.json",
                {"n": 50, "accuracy": 0.8, "scorer": "api", "paper_claim_eligible": True},
            )
            evidence = write_json(root / "evidence.json", [{"status": "safe_to_claim"}])
            dashboard = write_json(root / "dashboard.json", {"overall_status": "pass"})

            card = build_result_card(
                {
                    "gold_validations": [{"name": "Gold", "path": str(gold)}],
                    "bsc_summaries": [{"method": "base", "path": str(bsc)}],
                    "downstream_summaries": [{"method": "base", "path": str(downstream)}],
                    "evidence_matrix": str(evidence),
                    "dashboard": str(dashboard),
                }
            )

            self.assertEqual(card["claim_decision"]["status"], "safe_to_claim")
            self.assertEqual(card["status"], "safe_to_claim")
            self.assertTrue(card["ok"])
            self.assertIn("configured safe_claim", card["claim_decision"]["claim_discipline"][0])
            self.assertEqual(card["metrics"]["bsc"][0]["coverage"], 0.7)
            self.assertTrue(card["metrics"]["downstream"][0]["paper_claim_eligible"])
            self.assertIn("| Method | Status | N | Accuracy | Tie | Margin | Scorer | Eligible |", to_markdown(card))
            self.assertIn("| base | `pass` | 50 | 0.8000", to_markdown(card))
            self.assertIn("| api | True |", to_markdown(card))

    def test_required_downstream_blocks_when_not_paper_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bsc = write_json(root / "bsc.json", {"n": 100, "mean_coverage": 0.7})
            downstream = write_json(
                root / "downstream.json",
                {"n": 50, "accuracy": 0.9, "scorer": "keyword", "paper_claim_eligible": False},
            )
            evidence = write_json(root / "evidence.json", [{"status": "safe_to_claim"}])

            card = build_result_card(
                {
                    "require_downstream": True,
                    "bsc_summaries": [{"method": "base", "path": str(bsc)}],
                    "downstream_summaries": [{"method": "base", "path": str(downstream)}],
                    "evidence_matrix": str(evidence),
                }
            )
            md = to_markdown(card)

        self.assertEqual(card["metrics"]["downstream"][0]["status"], "not_paper_eligible")
        self.assertEqual(card["metrics"]["downstream"][0]["accuracy"], None)
        self.assertEqual(card["claim_decision"]["status"], "blocked")
        self.assertIn("not paper-eligible", card["claim_decision"]["blockers"][0])
        self.assertIn("| base | `not_paper_eligible` |  |  |", md)
        self.assertNotIn("0.9000", md)

    def test_dashboard_diagnostics_expose_submission_gap_training_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dashboard = write_json(
                root / "dashboard.json",
                {
                    "overall_status": "blocked",
                    "sections": [
                        {
                            "name": "Submission Gap Report",
                            "type": "submission_gap_report",
                            "status": "blocked",
                            "summary": "8 phases, execution_steps=6, training_chain_steps=7, prereq_items=60",
                            "metrics": {
                                "execution_steps": 6,
                                "blocked_execution_steps": 6,
                                "training_chain_steps": 7,
                                "missing_prerequisite_items": 60,
                                "hard_blockers": 26,
                            },
                        },
                        {
                            "name": "Rebuttal Pack Manifest",
                            "type": "rebuttal_manifest",
                            "status": "warn",
                            "summary": "11 entries, claim_ladder_safe=0/4",
                            "metrics": {
                                "claim_ladder_levels": 4,
                                "claim_ladder_safe": 0,
                                "claim_ladder_non_safe_levels": [
                                    "motivation",
                                    "metric-support",
                                    "method-support",
                                    "judge-utility support",
                                ],
                            },
                        },
                    ],
                },
            )

            card = build_result_card({"dashboard": str(dashboard)})
            md = to_markdown(card)

        self.assertEqual(card["dashboard_diagnostics"][0]["training_chain_steps"], 7)
        self.assertEqual(card["dashboard_diagnostics"][0]["missing_prerequisite_items"], 60)
        self.assertEqual(card["dashboard_diagnostics"][1]["claim_ladder_levels"], 4)
        self.assertEqual(card["dashboard_diagnostics"][1]["claim_ladder_safe"], 0)
        self.assertIn("judge-utility support", card["dashboard_diagnostics"][1]["claim_ladder_non_safe_levels"])
        self.assertIn("dashboard report is blocked", card["claim_decision"]["blockers"])
        self.assertIn("## Dashboard Diagnostics", md)
        self.assertIn("| Submission Gap Report | `blocked` | 6/6 | 7 | 60 | 26 | - |", md)
        self.assertIn("| Rebuttal Pack Manifest | `warn` | 0/0 | 0 | 0 | 0 | 0/4 safe; blocked:", md)

    def test_result_card_exposes_claim_ladder_and_downgrade_rules(self) -> None:
        card = build_result_card({})
        md = to_markdown(card)

        levels = [row["level"] for row in card["claim_ladder"]]
        self.assertEqual(levels, ["motivation", "metric-support", "method-support", "judge-utility support"])
        self.assertEqual(card["claim_ladder"][0]["required_claim_ids"], default_claim_ladder()[0]["required_claim_ids"])
        self.assertTrue(all(row["status"] == "missing_evidence" for row in card["claim_ladder"]))
        self.assertIn("Frozen 100-example hard-gold diagnostic", card["claim_ladder"][0]["evidence_required"])
        self.assertIn("Without downstream support, report metric-only BSC evidence", card["claim_ladder"][1]["downgrade_rule"])
        self.assertIn("Without C14, report proxy-gold supervision evidence rather than RLVR evidence", card["claim_ladder"][2]["downgrade_rule"])
        self.assertIn("without C0, no trained-method row is paper-facing", card["claim_ladder"][3]["downgrade_rule"])
        self.assertIn("C1: missing", card["claim_ladder"][0]["missing_or_non_safe_claims"])
        self.assertIn("## Claim Ladder", md)
        self.assertIn("| Level | Status | Required Claims | Evidence Required | Paper Sentence | Downgrade Rule |", md)
        self.assertIn("| metric-support | `missing_evidence` | C0, C2, C3 | Hard-gold BSC coverage change", md)
        self.assertIn("If C0/C2/C3 pass, report a hard-gold BSC coverage change as metric evidence", md)
        self.assertIn("If C0/C4/C9/C10/C12 pass, report held-out downstream judge-utility support", md)
        self.assertNotIn("The method has metric evidence for a reportable coverage change over human-gold dimensions", md)
        self.assertNotIn("The criteria have held-out downstream judge-utility support", md)
        self.assertNotIn("Hard-gold BSC increase", md)
        self.assertIn("| motivation blockers |", md)
        self.assertIn("RewardBench, RewardBench-2, and JudgeBench API/model scorer rows", md)

    def test_claim_ladder_status_uses_evidence_matrix_claim_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = write_json(
                root / "evidence.json",
                [
                    {"claim_id": "C1", "status": "safe_to_claim"},
                    {"claim_id": "C6", "status": "safe_to_claim"},
                    {"claim_id": "C0", "status": "safe_to_claim"},
                    {"claim_id": "C2", "status": "missing_evidence"},
                    {"claim_id": "C3", "status": "safe_to_claim"},
                    {"claim_id": "C5", "status": "safe_to_claim"},
                    {"claim_id": "C7", "status": "contradicted"},
                    {"claim_id": "C14", "status": "safe_to_claim"},
                ],
            )

            card = build_result_card({"evidence_matrix": str(evidence)})
            by_level = {row["level"]: row for row in card["claim_ladder"]}

        self.assertEqual(by_level["motivation"]["status"], "safe_to_claim")
        self.assertEqual(by_level["metric-support"]["status"], "missing_evidence")
        self.assertIn("C2: missing_evidence", by_level["metric-support"]["missing_or_non_safe_claims"])
        self.assertEqual(by_level["method-support"]["status"], "blocked")
        self.assertIn("C7: contradicted", by_level["method-support"]["missing_or_non_safe_claims"])
        self.assertIn("C4: missing", by_level["judge-utility support"]["missing_or_non_safe_claims"])

    def test_blocked_card_when_raw_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = write_json(root / "gold.json", {"ok": False, "n_records": 1, "blockers": ["too few"], "warnings": []})
            bsc = write_json(root / "bsc.json", {"n": 1})

            card = build_result_card(
                {
                    "gold_validations": [{"name": "Gold", "path": str(gold)}],
                    "bsc_summaries": [{"method": "base", "path": str(bsc)}],
                }
            )

            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertEqual(card["status"], "blocked")
            self.assertFalse(card["ok"])
            self.assertIn("Raw audit gate `Gold`", card["claim_decision"]["blockers"][0])

    def test_generic_raw_gate_accepts_non_empty_jsonl_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "verified.jsonl"
            artifact.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            evidence = write_json(root / "evidence.json", [{"status": "safe_to_claim"}])

            card = build_result_card(
                {
                    "raw_audit_gates": [{"name": "Verified Rubrics", "type": "generic", "path": str(artifact)}],
                    "evidence_matrix": str(evidence),
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "pass")
            self.assertEqual(card["claim_decision"]["status"], "safe_to_claim")

    def test_contamination_audit_gate_blocks_overlap_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = write_json(
                root / "contamination.json",
                {
                    "ok": True,
                    "overlap_query_count": 3,
                    "blockers": [],
                    "warnings": [],
                },
            )
            evidence = write_json(root / "evidence.json", [{"status": "safe_to_claim"}])

            card = build_result_card(
                {
                    "raw_audit_gates": [{"name": "Hard-Gold Audit", "type": "contamination_audit", "path": str(audit)}],
                    "evidence_matrix": str(evidence),
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertIn("overlaps=3", card["raw_audit_gates"][0]["summary"])
            self.assertIn("3 overlapping holdout query(s) found", card["claim_decision"]["blockers"][0])
            self.assertEqual(card["claim_decision"]["status"], "blocked")

    def test_contamination_audit_gate_accepts_clean_filter_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = write_json(
                root / "filter.json",
                {
                    "ok": True,
                    "removed_records": 1,
                    "blockers": [],
                    "warnings": [],
                },
            )
            evidence = write_json(root / "evidence.json", [{"status": "safe_to_claim"}])

            card = build_result_card(
                {
                    "raw_audit_gates": [{"name": "Proxy Filter", "type": "contamination_audit", "path": str(audit)}],
                    "evidence_matrix": str(evidence),
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "pass")
            self.assertIn("removed=1", card["raw_audit_gates"][0]["summary"])
            self.assertEqual(card["claim_decision"]["status"], "safe_to_claim")

    def test_contamination_audit_gate_separates_missing_artifacts_from_clean_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = write_json(
                root / "contamination.json",
                {
                    "ok": False,
                    "artifact_status": "blocked",
                    "overlap_status": "not_auditable",
                    "overlap_query_count": 0,
                    "blockers": ["proxy_gold: missing or empty file"],
                    "warnings": [],
                },
            )
            evidence = write_json(root / "evidence.json", [{"status": "safe_to_claim"}])

            card = build_result_card(
                {
                    "raw_audit_gates": [{"name": "Hard-Gold Audit", "type": "contamination_audit", "path": str(audit)}],
                    "evidence_matrix": str(evidence),
                }
            )

        self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
        self.assertIn("artifact_status=blocked", card["raw_audit_gates"][0]["summary"])
        self.assertIn("overlap_status=not_auditable", card["raw_audit_gates"][0]["summary"])
        self.assertIn("overlaps=0", card["raw_audit_gates"][0]["summary"])
        self.assertEqual(card["claim_decision"]["status"], "blocked")

    def test_card_blocks_on_failed_query_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = write_json(
                root / "queries.json",
                {"ok": False, "target": "query_pool", "n_records": 10, "blockers": ["toy source"], "warnings": []},
            )

            card = build_result_card(
                {"query_validations": [{"name": "RubricBench Queries", "path": str(query)}]}
            )

            self.assertEqual(card["raw_audit_gates"][0]["type"], "query_validation")
            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("toy source", card["raw_audit_gates"][0]["blockers"][0])

    def test_card_deduplicates_same_raw_and_query_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = write_json(
                root / "queries.json",
                {"ok": True, "target": "query_pool", "n_records": 100, "blockers": [], "warnings": []},
            )

            card = build_result_card(
                {
                    "raw_audit_gates": [
                        {"name": "Raw Queries", "type": "query_validation", "path": str(query)}
                    ],
                    "query_validations": [{"name": "RubricBench Queries", "path": str(query)}],
                }
            )

            query_gates = [gate for gate in card["raw_audit_gates"] if gate["type"] == "query_validation"]
            self.assertEqual(len(query_gates), 1)

    def test_deferred_card_when_evidence_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = write_json(root / "gold.json", {"ok": True, "n_records": 10, "blockers": [], "warnings": []})
            bsc = write_json(root / "bsc.json", {"n": 10})

            card = build_result_card(
                {
                    "gold_validations": [{"name": "Gold", "path": str(gold)}],
                    "bsc_summaries": [{"method": "base", "path": str(bsc)}],
                    "evidence_matrix": str(root / "missing_evidence.json"),
                }
            )

            self.assertEqual(card["claim_decision"]["status"], "deferred")
            self.assertTrue(card["claim_decision"]["warnings"])

    def test_card_collects_confidence_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ci = write_json(
                root / "ci.json",
                {
                    "n": 10,
                    "confidence": 0.95,
                    "metrics": [
                        {"metric": "coverage", "n": 10, "mean": 0.7, "ci_lower": 0.6, "ci_upper": 0.8, "status": "pass"}
                    ],
                },
            )

            card = build_result_card({"confidence_intervals": [{"name": "BSC CI", "path": str(ci)}]})

            self.assertEqual(card["confidence_intervals"][0]["status"], "pass")
            self.assertEqual(card["confidence_intervals"][0]["metrics"][0]["metric"], "coverage")

    def test_card_blocks_on_failed_rubric_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = write_json(root / "validation.json", {"ok": False, "n_records": 5, "ok_records": 4})

            card = build_result_card(
                {
                    "raw_audit_gates": [
                        {"name": "Rubric Validation", "type": "validation", "path": str(validation)}
                    ]
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("failed validation", card["raw_audit_gates"][0]["blockers"][0])
            self.assertIn("criteria output record(s)", card["raw_audit_gates"][0]["blockers"][0])
            self.assertNotIn("rubric output record", card["raw_audit_gates"][0]["blockers"][0])

    def test_card_blocks_on_failed_schema_contract_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema = write_json(
                root / "schema.json",
                {
                    "ok": False,
                    "selected_target": None,
                    "targets": [{"target": "multicandidate", "ok": False}],
                    "blockers": ["schema unmapped"],
                    "warnings": [],
                },
            )

            card = build_result_card(
                {
                    "raw_audit_gates": [
                        {"name": "RM-Bench Schema", "type": "schema_contract", "path": str(schema)}
                    ]
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("schema unmapped", card["raw_audit_gates"][0]["blockers"][0])

    def test_card_blocks_on_failed_api_budget_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = write_json(
                root / "budget.json",
                {
                    "ok": False,
                    "blockers": ["estimated cost USD exceeds max_cost_usd: 20 > 10"],
                    "total": {"calls": 100, "total_tokens": 50000, "estimated_cost_usd": 20.0},
                },
            )

            card = build_result_card(
                {"raw_audit_gates": [{"name": "API Budget", "type": "api_budget", "path": str(gate)}]}
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("estimated cost USD exceeds", card["raw_audit_gates"][0]["blockers"][0])

    def test_card_blocks_on_failed_latex_compile_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = write_json(
                root / "latex_compile.json",
                {
                    "ok": False,
                    "pdf_bytes": 0,
                    "page_count": 0,
                    "max_pages": 8,
                    "official_style_active": False,
                    "bibliography_style_active": False,
                    "anonymous_author_declared": False,
                    "blockers": ["official AAAI style was not observed in paper/main.log"],
                    "warnings": [],
                },
            )

            card = build_result_card(
                {"raw_audit_gates": [{"name": "AAAI LaTeX Compile", "type": "latex_compile", "path": str(gate)}]}
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("max_pages=8", card["raw_audit_gates"][0]["summary"])
            self.assertIn("official_style_active=False", card["raw_audit_gates"][0]["summary"])
            self.assertIn("bibliography_style_active=False", card["raw_audit_gates"][0]["summary"])
            self.assertIn("anonymous_author_declared=False", card["raw_audit_gates"][0]["summary"])

    def test_card_blocks_on_minimal_api_handoff_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = write_json(
                root / "api_handoff.json",
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

            card = build_result_card(
                {
                    "raw_audit_gates": [
                        {"name": "API Handoff", "type": "minimal_api_handoff", "path": str(gate)}
                    ]
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("missing_env=LOCAL_OPENAI_API_KEY,OPENAI_API_KEY", card["raw_audit_gates"][0]["summary"])
            self.assertIn("next=offline_rerun", card["raw_audit_gates"][0]["summary"])
            self.assertIn("preflight: ok=false", card["raw_audit_gates"][0]["blockers"])

    def test_card_blocks_on_failed_data_source_local_config_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = write_json(
                root / "local_config_init.json",
                {
                    "ok": False,
                    "source_overall_status": "blocked",
                    "blockers": ["rubricbench raw_sha256 missing"],
                    "warnings": [],
                },
            )

            card = build_result_card(
                {
                    "raw_audit_gates": [
                        {"name": "Data Source Local Config", "type": "data_source_local_config", "path": str(gate)}
                    ]
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertIn("source_status=blocked", card["raw_audit_gates"][0]["summary"])
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("raw_sha256 missing", card["raw_audit_gates"][0]["blockers"][0])

    def test_card_scopes_data_source_report_to_required_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_json(
                root / "source.json",
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

            card = build_result_card(
                {
                    "data_source_reports": [
                        {
                            "name": "Minimal Sources",
                            "path": str(source),
                            "required_datasets": ["rubricbench"],
                        }
                    ]
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "pass")
            self.assertEqual(card["raw_audit_gates"][0]["required_datasets"], ["rubricbench"])
            self.assertNotEqual(card["claim_decision"]["status"], "blocked")

    def test_card_blocks_on_invalid_raw_gate_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "budget.json"
            gate.write_text("{bad", encoding="utf-8")

            card = build_result_card(
                {"raw_audit_gates": [{"name": "API Budget", "type": "api_budget", "path": str(gate)}]}
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertIn("report is not readable", card["raw_audit_gates"][0]["summary"])
            self.assertEqual(card["claim_decision"]["status"], "blocked")

    def test_card_blocks_on_invalid_bsc_summary_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bsc = root / "bsc.json"
            bsc.write_text("{bad", encoding="utf-8")

            card = build_result_card({"bsc_summaries": [{"method": "base", "path": str(bsc)}]})

            self.assertEqual(card["metrics"]["bsc"][0]["status"], "blocked")
            self.assertIn("metric summary is not readable", card["metrics"]["bsc"][0]["blockers"][0])
            self.assertEqual(card["claim_decision"]["status"], "blocked")

    def test_card_blocks_on_invalid_evidence_matrix_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            evidence.write_text("{bad", encoding="utf-8")

            card = build_result_card({"evidence_matrix": str(evidence)})

            self.assertEqual(card["claim_evidence"]["status"], "blocked")
            self.assertIn("evidence matrix is not readable", card["claim_evidence"]["blockers"][0])
            self.assertEqual(card["claim_decision"]["status"], "blocked")

    def test_card_blocks_on_failed_preflight_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = write_json(
                root / "preflight.json",
                {
                    "ok": False,
                    "hard_blockers": ["missing required provider: sft_rl"],
                    "warnings": ["LOCAL_OPENAI_API_KEY is not set"],
                },
            )

            card = build_result_card(
                {"raw_audit_gates": [{"name": "Preflight", "type": "preflight", "path": str(gate)}]}
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("missing required provider", card["raw_audit_gates"][0]["blockers"][0])

    def test_card_blocks_on_failed_manual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = write_json(
                root / "manual_gate.json",
                {
                    "ok": False,
                    "checks": [
                        {"path": "outputs/policy_rlvr/healthbench_hard_policy", "present": True},
                        {"path": "outputs/policy_rlvr/healthbench_hard_eval.json", "present": False},
                    ],
                    "json_checks": [
                        {
                            "path": "outputs/policy_rlvr/downstream_rlvr_done.json",
                            "present": True,
                            "valid_json": True,
                            "missing_keys": ["operator"],
                        }
                    ],
                    "json_contains_checks": [
                        {
                            "path": "outputs/training_commands/training_done.json",
                            "key": "served_generators",
                            "present": True,
                            "valid_json": True,
                            "missing_values": ["sft_rl"],
                        }
                    ],
                    "json_sha256_checks": [
                        {
                            "path": "outputs/sft_data/proxy_gold_verl_report.json",
                            "key": "output_sha256",
                            "target_path": "data/processed/proxy_gold_verl.parquet",
                            "present": True,
                            "target_present": True,
                            "valid_json": True,
                            "matches": False,
                        }
                    ],
                    "blockers": ["Missing required path: outputs/policy_rlvr/healthbench_hard_eval.json"],
                },
            )

            card = build_result_card(
                {
                    "raw_audit_gates": [
                        {"name": "Downstream Policy RLVR", "type": "manual_gate", "path": str(gate)}
                    ]
                }
            )

            self.assertEqual(card["raw_audit_gates"][0]["status"], "blocked")
            self.assertEqual(
                card["raw_audit_gates"][0]["summary"],
                "1/2 required paths present; 0/1 JSON contracts valid; "
                "0/1 JSON contains contracts valid; 0/1 JSON SHA256 contracts valid",
            )
            self.assertEqual(card["claim_decision"]["status"], "blocked")

    def test_card_blocks_on_failed_manifest_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_json(
                root / "downstream_rlvr_manifest.json",
                {
                    "ok": False,
                    "scripts": ["outputs/downstream_rlvr_commands/run_healthbench_hard_rlvr.sh"],
                    "benchmarks": [],
                    "blockers": ["healthbench_hard: missing required train_data at data/train.parquet"],
                },
            )

            card = build_result_card(
                {
                    "manifests": [
                        {"name": "Downstream RLVR Manifest", "path": str(manifest)}
                    ]
                }
            )

            self.assertEqual(card["pack_builder_manifest"][0]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("Downstream RLVR Manifest", card["claim_decision"]["blockers"][0])

    def test_card_warns_on_rebuttal_pack_manifest_without_ready_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_json(
                root / "rebuttal_pack_manifest.json",
                {
                    "schema_version": 1,
                    "entry_count": 2,
                    "defense_status_counts": {"needs_evidence": 2},
                    "readiness_ok": False,
                    "matched_claim_ids": ["C1"],
                    "concern_templates": {
                        "source": "DEFAULT_CONCERNS",
                        "count": 2,
                        "sha256": "abc123",
                    },
                },
            )

            card = build_result_card(
                {
                    "manifests": [
                        {"name": "Rebuttal Pack Manifest", "path": str(manifest)}
                    ]
                }
            )
            md = to_markdown(card)

        row = card["pack_builder_manifest"][0]
        self.assertEqual(row["status"], "warn")
        self.assertIn("2 rebuttal entries", row["summary"])
        self.assertIn("needs_evidence=2", row["summary"])
        self.assertIn("needs_readiness=0", row["summary"])
        self.assertIn("cannot_claim=0", row["summary"])
        self.assertIn("missing_claim_mapping=0", row["summary"])
        self.assertIn("templates=2", row["summary"])
        self.assertEqual(card["claim_decision"]["status"], "deferred")
        self.assertIn("Manifest `Rebuttal Pack Manifest` is warn", card["claim_decision"]["warnings"])
        self.assertIn("### Manifest Diagnostics", md)
        self.assertIn("Rebuttal Pack Manifest: warning: rebuttal pack was built while submission readiness was false", md)
        self.assertIn("Rebuttal Pack Manifest: warning: rebuttal pack has no answer_ready entries", md)

    def test_card_summarizes_rebuttal_pack_entries_waiting_for_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_json(
                root / "rebuttal_pack_manifest.json",
                {
                    "schema_version": 1,
                    "entry_count": 2,
                    "defense_status_counts": {"needs_readiness": 2},
                    "readiness_ok": False,
                    "matched_claim_ids": ["C1"],
                    "concern_templates": {
                        "source": "DEFAULT_CONCERNS",
                        "count": 2,
                        "sha256": "abc123",
                    },
                },
            )

            card = build_result_card(
                {
                    "manifests": [
                        {"name": "Rebuttal Pack Manifest", "path": str(manifest)}
                    ]
                }
            )

        row = card["pack_builder_manifest"][0]
        self.assertEqual(row["status"], "warn")
        self.assertIn("needs_readiness=2", row["summary"])
        self.assertIn("needs_evidence=0", row["summary"])

    def test_card_blocks_on_readiness_ok_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = write_json(root / "readiness.json", {"ok": False, "raw_gates": []})

            card = build_result_card({"readiness_report": str(readiness)})

            self.assertEqual(card["readiness"]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertEqual(card["status"], "blocked")
            self.assertFalse(card["ok"])
            self.assertIn("readiness report is blocked", card["claim_decision"]["blockers"])

    def test_claim_discipline_explains_blocked_real_paper_claim_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = write_json(
                root / "evidence.json",
                [
                    {"claim_id": "C0", "status": "missing_evidence"},
                    {"claim_id": "C2", "status": "missing_evidence"},
                    {"claim_id": "C3", "status": "missing_evidence"},
                    {"claim_id": "C4", "status": "missing_evidence"},
                    {"claim_id": "C5", "status": "missing_evidence"},
                    {"claim_id": "C7", "status": "missing_evidence"},
                    {"claim_id": "C9", "status": "missing_evidence"},
                    {"claim_id": "C10", "status": "missing_evidence"},
                    {"claim_id": "C12", "status": "missing_evidence"},
                    {"claim_id": "C13", "status": "missing_evidence"},
                    {"claim_id": "C14", "status": "missing_evidence"},
                ],
            )
            readiness = write_json(
                root / "readiness.json",
                {
                    "ok": False,
                    "hard_blockers": ["required evidence claim C0 is missing_evidence"],
                    "claim_discipline": [
                        "Do not treat this package as AAAI-ready while submission readiness is false.",
                        "With zero safe_to_claim rows, restrict paper-facing empirical content to planned protocol and clearly marked diagnostic evidence.",
                    ],
                },
            )

            card = build_result_card(
                {
                    "evidence_matrix": str(evidence),
                    "readiness_report": str(readiness),
                }
            )
            md = to_markdown(card)

        discipline = "\n".join(card["claim_decision"]["claim_discipline"])
        self.assertEqual(card["claim_decision"]["status"], "blocked")
        self.assertIn("Do not write empirical claims", discipline)
        self.assertIn("every required paper-facing row is safe_to_claim", discipline)
        self.assertNotIn("free of missing or contradicted claims", discipline)
        self.assertIn("Do not treat this package as AAAI-ready while submission readiness is false.", discipline)
        self.assertIn("With zero safe_to_claim rows, restrict paper-facing empirical content", discipline)
        self.assertIn("Do not claim clean hard-gold/proxy/downstream isolation until C0 is safe_to_claim", discipline)
        self.assertIn("Treat BSC coverage changes as metric evidence only", discipline)
        self.assertIn("do not call them dimension-level recovery until C3, C12, and C14 pass", discipline)
        self.assertIn("Do not claim downstream judge-utility support until C4/C9/C10 pass", discipline)
        self.assertIn("Do not claim ablation support until C5/C7 pass", discipline)
        self.assertIn("Treat semantic-space plots as illustrative until C13 verifies", discipline)
        self.assertIn("### Claim Discipline", md)
        self.assertIn("Do not treat this package as AAAI-ready", md)
        self.assertIn("paper_claim_eligible summaries", md)

    def test_card_blocks_when_readiness_reports_missing_c5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = write_json(
                root / "readiness.json",
                {
                    "ok": False,
                    "hard_blockers": ["required evidence claim C5 is missing from evidence matrix"],
                },
            )

            card = build_result_card({"readiness_report": str(readiness)})

            self.assertEqual(card["readiness"]["status"], "blocked")
            self.assertEqual(card["claim_decision"]["status"], "blocked")
            self.assertIn("required evidence claim C5", card["readiness"]["data"]["hard_blockers"][0])
            self.assertIn("readiness report is blocked", card["claim_decision"]["blockers"])

    def test_markdown_contains_result_card_sections(self) -> None:
        card = {
            "title": "Card",
            "experiment_id": "exp",
            "scope": "scope",
            "raw_audit_gates": [],
            "pack_builder_manifest": [],
            "metrics": {"bsc": [], "downstream": []},
            "confidence_intervals": [],
            "claim_decision": {
                "status": "deferred",
                "safe_claim": "",
                "deferred_claim": "",
                "claim_discipline": ["Do not write empirical claims without evidence."],
                "blockers": [],
                "warnings": [],
            },
            "notes": [],
        }
        md = to_markdown(card)
        self.assertIn("Raw Audit Gates", md)
        self.assertIn("Confidence Intervals", md)
        self.assertIn("Claim Decision", md)
        self.assertIn("Claim Discipline", md)


def write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
