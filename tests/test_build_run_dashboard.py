from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_run_dashboard import build_dashboard, file_sha256, load_config, to_markdown


class BuildRunDashboardTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_dashboard_config.json"))

        self.assertIn("Run Dashboard config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Run Dashboard config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Run Dashboard config must be a JSON object", str(context.exception))

    def test_dashboard_summarizes_pass_warn_and_budget_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            evidence = root / "evidence.json"
            budget = root / "budget.json"
            audit.write_text(json.dumps({"ok": True, "present_files": ["a"], "missing_files": [], "summary_checks": []}), encoding="utf-8")
            evidence.write_text(
                json.dumps(
                    [
                        {"status": "safe_to_claim"},
                        {"status": "missing_evidence"},
                    ]
                ),
                encoding="utf-8",
            )
            budget.write_text(
                json.dumps(
                    {
                        "n_queries": 2,
                        "n_providers": 1,
                        "ok": True,
                        "total": {"calls": 2, "total_tokens": 1000, "estimated_cost_usd": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {
                    "sections": [
                        {"name": "Audit", "type": "audit", "path": str(audit)},
                        {"name": "Evidence", "type": "evidence", "path": str(evidence)},
                        {"name": "Budget", "type": "api_budget", "path": str(budget)},
                    ]
                }
            )

            self.assertEqual(dashboard["overall_status"], "warn")
            self.assertEqual(len(dashboard["blockers"]), 0)

    def test_api_budget_section_blocks_when_budget_exceeds_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budget = root / "budget.json"
            budget.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "blockers": ["API calls exceeds max_calls: 10 > 5"],
                        "n_queries": 2,
                        "n_providers": 1,
                        "total": {"calls": 10, "total_tokens": 1000, "estimated_cost_usd": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Budget", "type": "api_budget", "path": str(budget)}]}
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["metrics"]["blockers"], 1)
            self.assertEqual(dashboard["sections"][0]["metrics"]["calls"], 10)
            self.assertIn("API calls exceeds", dashboard["blockers"][0])

    def test_contamination_audit_section_blocks_not_auditable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "contamination.json"
            audit.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "artifact_status": "blocked",
                        "overlap_status": "not_auditable",
                        "overlap_query_count": 0,
                        "holdout_unique_queries": 491,
                        "training_unique_queries": 0,
                        "blockers": ["proxy_gold_verl: missing or empty file"],
                        "warnings": [],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Hard-Gold Audit", "type": "contamination_audit", "path": str(audit)}]}
            )

            section = dashboard["sections"][0]
            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(section["status"], "blocked")
            self.assertEqual(section["metrics"]["overlap_query_count"], 0)
            self.assertIn("artifact_status=blocked", section["summary"])
            self.assertIn("overlap_status=not_auditable", section["summary"])
            self.assertTrue(any("proxy_gold_verl" in blocker for blocker in dashboard["blockers"]))

    def test_latex_compile_section_blocks_when_official_style_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "latex_compile.json"
            report.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "pdf_bytes": 100,
                        "page_count": 8,
                        "max_pages": 8,
                        "official_style_active": False,
                        "official_style_files_present": True,
                        "submission_mode_declared": False,
                        "bibliography_style_active": False,
                        "anonymous_author_declared": False,
                        "blockers": ["official AAAI style was not observed in paper/main.log"],
                        "warnings": [],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "AAAI LaTeX Compile", "type": "latex_compile", "path": str(report)}]}
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["metrics"]["max_pages"], 8)
            self.assertEqual(dashboard["sections"][0]["metrics"]["official_style_active"], False)
            self.assertEqual(dashboard["sections"][0]["metrics"]["bibliography_style_active"], False)
            self.assertEqual(dashboard["sections"][0]["metrics"]["anonymous_author_declared"], False)
            self.assertIn("official AAAI style", dashboard["blockers"][0])

    def test_dashboard_blocks_on_invalid_section_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budget = root / "budget.json"
            budget.write_text("{bad", encoding="utf-8")

            dashboard = build_dashboard(
                {"sections": [{"name": "Budget", "type": "api_budget", "path": str(budget)}]}
            )

        self.assertEqual(dashboard["overall_status"], "blocked")
        self.assertEqual(dashboard["sections"][0]["status"], "blocked")
        self.assertIn("report is not readable", dashboard["sections"][0]["summary"])
        self.assertIn("report is not readable", dashboard["blockers"][0])
        self.assertIn("Resolve blockers in Budget", dashboard["next_actions"][0])

    def test_dashboard_blocks_on_preflight_and_contradicted_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight = root / "preflight.json"
            evidence = root / "evidence.json"
            preflight.write_text(
                json.dumps({"ok": False, "hard_blockers": ["missing API env"], "warnings": []}),
                encoding="utf-8",
            )
            evidence.write_text(json.dumps([{"status": "contradicted"}]), encoding="utf-8")

            dashboard = build_dashboard(
                {
                    "sections": [
                        {"name": "Preflight", "type": "preflight", "path": str(preflight)},
                        {"name": "Evidence", "type": "evidence", "path": str(evidence)},
                    ]
                }
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertGreaterEqual(len(dashboard["blockers"]), 2)
            self.assertIn("Resolve blockers", dashboard["next_actions"][0])

    def test_optional_missing_report_is_warning(self) -> None:
        dashboard = build_dashboard(
            {
                "sections": [
                    {"name": "Optional Readiness", "type": "readiness", "path": "/tmp/no_such_report.json", "required": False}
                ]
            }
        )

        self.assertEqual(dashboard["overall_status"], "warn")
        self.assertEqual(dashboard["sections"][0]["status"], "warn")
        self.assertIn("optional report is missing", dashboard["warnings"][0])

    def test_optional_teacher_union_artifact_is_visible_when_missing(self) -> None:
        dashboard = build_dashboard(
            {
                "sections": [
                    {
                        "name": "C5 Teacher-Union Ablation JSON",
                        "type": "generic",
                        "path": "/tmp/no_such_teacher_union_ablation.json",
                        "required": False,
                    }
                ]
            }
        )

        self.assertEqual(dashboard["overall_status"], "warn")
        self.assertEqual(dashboard["sections"][0]["name"], "C5 Teacher-Union Ablation JSON")
        self.assertEqual(dashboard["sections"][0]["status"], "warn")
        self.assertIn("C5 Teacher-Union Ablation JSON", dashboard["warnings"][0])

    def test_gold_validation_section_passes_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good_gold.json"
            bad = root / "bad_gold.json"
            good.write_text(
                json.dumps({"ok": True, "n_records": 100, "min_records": 100, "min_rubrics_per_query": 1, "blockers": [], "warnings": []}),
                encoding="utf-8",
            )
            bad.write_text(
                json.dumps({"ok": False, "n_records": 1, "min_records": 100, "min_rubrics_per_query": 1, "blockers": ["record count too low"], "warnings": []}),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {
                    "sections": [
                        {"name": "Good Gold", "type": "gold_validation", "path": str(good)},
                        {"name": "Bad Gold", "type": "gold_validation", "path": str(bad)},
                    ]
                }
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["status"], "pass")
            self.assertEqual(dashboard["sections"][1]["status"], "blocked")
            self.assertIn("record count too low", dashboard["blockers"][0])

    def test_data_source_report_section_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "source_report.json"
            report.write_text(
                json.dumps(
                    {
                        "overall_status": "blocked",
                        "datasets": [{"name": "rubricbench"}],
                        "blockers": ["rubricbench raw missing"],
                        "warnings": ["rewardbench not downloaded"],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Sources", "type": "data_source_report", "path": str(report)}]}
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["metrics"]["datasets"], 1)
            self.assertIn("rubricbench raw missing", dashboard["blockers"][0])

    def test_schema_contract_section_blocks_when_unmapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "schema.json"
            report.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "n_records": 10,
                        "selected_target": None,
                        "targets": [{"target": "preference", "ok": False}],
                        "blockers": ["No downstream schema target met the minimum normalized-record threshold."],
                        "warnings": [],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "RM-Bench Schema", "type": "schema_contract", "path": str(report)}]}
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["metrics"]["compatible_targets"], 0)
            self.assertIn("No downstream schema target", dashboard["blockers"][0])

    def test_manual_gate_section_blocks_when_required_paths_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "manual_gate.json"
            report.write_text(
                json.dumps(
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
                        "blockers": ["Missing required path: outputs/policy_rlvr/healthbench_hard_eval.json"],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Downstream Policy RLVR", "type": "manual_gate", "path": str(report)}]}
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["metrics"]["missing"], 1)
            self.assertEqual(dashboard["sections"][0]["metrics"]["json_missing_keys"], 1)
            self.assertEqual(dashboard["sections"][0]["metrics"]["json_missing_values"], 1)
            self.assertIn("0/1 JSON contracts valid", dashboard["sections"][0]["summary"])
            self.assertIn("0/1 JSON contains contracts valid", dashboard["sections"][0]["summary"])
            self.assertIn("healthbench_hard_eval", dashboard["blockers"][0])

    def test_confidence_interval_section_passes_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good_ci.json"
            bad = root / "bad_ci.json"
            good.write_text(
                json.dumps(
                    {
                        "n": 100,
                        "confidence": 0.95,
                        "n_boot": 1000,
                        "metrics": [{"metric": "blind", "status": "pass", "ci_lower": 0.2}],
                    }
                ),
                encoding="utf-8",
            )
            bad.write_text(
                json.dumps(
                    {
                        "n": 100,
                        "confidence": 0.95,
                        "n_boot": 1000,
                        "metrics": [{"metric": "coverage", "status": "missing"}],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {
                    "sections": [
                        {"name": "Good CI", "type": "confidence_interval", "path": str(good)},
                        {"name": "Bad CI", "type": "confidence_interval", "path": str(bad)},
                    ]
                }
            )

            self.assertEqual(dashboard["overall_status"], "warn")
            self.assertEqual(dashboard["sections"][0]["status"], "pass")
            self.assertEqual(dashboard["sections"][0]["metrics"]["metric_count"], 1)
            self.assertEqual(dashboard["sections"][1]["status"], "warn")
            self.assertIn("coverage", dashboard["warnings"][0])

    def test_validation_section_blocks_on_failed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "validation.json"
            validation.write_text(
                json.dumps({"ok": False, "n_records": 4, "ok_records": 3, "ok_rate": 0.75}),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Rubric Validation", "type": "validation", "path": str(validation)}]}
            )

            self.assertEqual(dashboard["overall_status"], "blocked")
            self.assertEqual(dashboard["sections"][0]["metrics"]["failed_records"], 1)
            self.assertIn("failed validation", dashboard["blockers"][0])
            self.assertIn("criteria output record(s)", dashboard["blockers"][0])
            self.assertNotIn("rubric output record", dashboard["blockers"][0])

    def test_rebuttal_manifest_warns_when_readiness_is_false_but_sha_contracts_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            readiness = root / "readiness.json"
            pack_json = root / "rebuttal_pack.json"
            pack_md = root / "rebuttal_pack.md"
            manifest = root / "rebuttal_pack_manifest.json"
            for path, text in [
                (evidence, "[]\n"),
                (readiness, '{"ok": false}\n'),
                (pack_json, "[]\n"),
                (pack_md, "# Pack\n"),
            ]:
                path.write_text(text, encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entry_count": 2,
                        "defense_status_counts": {"needs_evidence": 2},
                        "readiness_ok": False,
                        "matched_claim_ids": ["C1"],
                        "claim_ladder": [
                            {"level": "motivation", "status": "safe_to_claim"},
                            {"level": "metric-support", "status": "missing_evidence"},
                            {"level": "method-support", "status": "missing_evidence"},
                            {"level": "judge-utility support", "status": "blocked"},
                        ],
                        "concern_templates": {"source": "DEFAULT_CONCERNS", "count": 2, "sha256": "abc123"},
                        "inputs": {
                            "evidence_matrix": file_record(evidence),
                            "readiness_report": file_record(readiness),
                        },
                        "outputs": {
                            "rebuttal_pack_json": file_record(pack_json),
                            "rebuttal_pack_md": file_record(pack_md),
                        },
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Rebuttal Pack Manifest", "type": "rebuttal_manifest", "path": str(manifest)}]}
            )

        section = dashboard["sections"][0]
        self.assertEqual(dashboard["overall_status"], "warn")
        self.assertEqual(section["type"], "rebuttal_manifest")
        self.assertEqual(section["status"], "warn")
        self.assertEqual(section["metrics"]["entry_count"], 2)
        self.assertEqual(section["metrics"]["needs_evidence"], 2)
        self.assertEqual(section["metrics"]["needs_readiness"], 0)
        self.assertEqual(section["metrics"]["cannot_claim"], 0)
        self.assertEqual(section["metrics"]["missing_claim_mapping"], 0)
        self.assertEqual(section["metrics"]["concern_templates"], 2)
        self.assertEqual(section["metrics"]["input_records"], 2)
        self.assertEqual(section["metrics"]["output_records"], 2)
        self.assertEqual(section["metrics"]["claim_ladder_levels"], 4)
        self.assertEqual(section["metrics"]["claim_ladder_safe"], 1)
        self.assertEqual(section["metrics"]["claim_ladder_missing"], 2)
        self.assertEqual(section["metrics"]["claim_ladder_blocked"], 1)
        self.assertEqual(
            section["metrics"]["claim_ladder_non_safe_levels"],
            ["metric-support", "method-support", "judge-utility support"],
        )
        self.assertIn("readiness_ok=False", section["summary"])
        self.assertIn("cannot_claim=0", section["summary"])
        self.assertIn("missing_claim_mapping=0", section["summary"])
        self.assertIn("claim_ladder_safe=1/4", section["summary"])
        self.assertIn("submission readiness was false", dashboard["warnings"][0])

    def test_rebuttal_manifest_warns_when_entries_wait_for_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            readiness = root / "readiness.json"
            pack_json = root / "rebuttal_pack.json"
            pack_md = root / "rebuttal_pack.md"
            manifest = root / "rebuttal_pack_manifest.json"
            for path, text in [
                (evidence, "[]\n"),
                (readiness, '{"ok": false}\n'),
                (pack_json, "[]\n"),
                (pack_md, "# Pack\n"),
            ]:
                path.write_text(text, encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entry_count": 2,
                        "defense_status_counts": {"needs_readiness": 2},
                        "readiness_ok": False,
                        "matched_claim_ids": ["C1"],
                        "claim_ladder": [],
                        "concern_templates": {"source": "DEFAULT_CONCERNS", "count": 2, "sha256": "abc123"},
                        "inputs": {
                            "evidence_matrix": file_record(evidence),
                            "readiness_report": file_record(readiness),
                        },
                        "outputs": {
                            "rebuttal_pack_json": file_record(pack_json),
                            "rebuttal_pack_md": file_record(pack_md),
                        },
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Rebuttal Pack Manifest", "type": "rebuttal_manifest", "path": str(manifest)}]}
            )

        section = dashboard["sections"][0]
        self.assertEqual(section["status"], "warn")
        self.assertEqual(section["metrics"]["needs_readiness"], 2)
        self.assertIn("needs_readiness=2", section["summary"])
        self.assertTrue(any("need submission readiness" in item for item in dashboard["warnings"]))

    def test_rebuttal_manifest_blocks_when_sha_contract_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            pack_json = root / "rebuttal_pack.json"
            manifest = root / "rebuttal_pack_manifest.json"
            evidence.write_text("[]\n", encoding="utf-8")
            pack_json.write_text("[]\n", encoding="utf-8")
            stale_record = file_record(pack_json)
            pack_json.write_text('[{"changed": true}]\n', encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entry_count": 1,
                        "defense_status_counts": {"answer_ready": 1},
                        "readiness_ok": True,
                        "matched_claim_ids": ["C1"],
                        "concern_templates": {"source": "DEFAULT_CONCERNS", "count": 1, "sha256": "abc123"},
                        "inputs": {"evidence_matrix": file_record(evidence)},
                        "outputs": {"rebuttal_pack_json": stale_record},
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Rebuttal Pack Manifest", "type": "rebuttal_manifest", "path": str(manifest)}]}
            )

        self.assertEqual(dashboard["overall_status"], "blocked")
        self.assertEqual(dashboard["sections"][0]["metrics"]["sha_mismatches"], 1)
        self.assertIn("sha256 is stale", dashboard["blockers"][0])

    def test_rebuttal_manifest_blocks_when_concern_template_record_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            pack_json = root / "rebuttal_pack.json"
            manifest = root / "rebuttal_pack_manifest.json"
            evidence.write_text("[]\n", encoding="utf-8")
            pack_json.write_text("[]\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entry_count": 1,
                        "defense_status_counts": {"answer_ready": 1},
                        "readiness_ok": True,
                        "matched_claim_ids": ["C1"],
                        "inputs": {"evidence_matrix": file_record(evidence)},
                        "outputs": {"rebuttal_pack_json": file_record(pack_json)},
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {"sections": [{"name": "Rebuttal Pack Manifest", "type": "rebuttal_manifest", "path": str(manifest)}]}
            )

        self.assertEqual(dashboard["overall_status"], "blocked")
        self.assertIn("concern template count does not match entry_count", dashboard["blockers"][0])

    def test_submission_gap_report_summarizes_blocked_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gap_report = root / "submission_gap_report.json"
            gap_report.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "readiness_ok": False,
                        "hard_blocker_count": 3,
                        "warning_count": 1,
                        "execution_sequence": [
                            {"id": "data", "phase_status": "blocked", "blocked_by_prior_phases": []},
                            {"id": "paper", "phase_status": "pass", "blocked_by_prior_phases": ["data"]},
                            {"id": "rebuttal", "phase_status": "pass", "blocked_by_prior_phases": []},
                        ],
                        "missing_prerequisites": {
                            "input_files": ["data/processed/rmbench_queries.jsonl"],
                            "required_env": ["OPENAI_API_KEY", "GPT_AK_1"],
                            "api_env_by_file": {
                                "configs/generators.local.jsonl": ["base: LOCAL_OPENAI_API_KEY", "gpt4o: OPENAI_API_KEY"],
                            },
                            "api_env_by_provider": {
                                "meta-verifier": "GPT_AK_3",
                            },
                        },
                        "operator_handoff": {
                            "training_data_chain": [
                                {"stage": "sft_data_preflight"},
                                {"stage": "audit_*_holdout_contamination"},
                            ],
                        },
                        "phases": [
                            {"id": "data", "name": "Data Isolation", "status": "blocked"},
                            {"id": "paper", "name": "Paper Readiness", "status": "pass"},
                            {
                                "id": "reviewer",
                                "name": "Reviewer-Facing Rebuttal Readiness",
                                "status": "blocked",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dashboard = build_dashboard(
                {
                    "sections": [
                        {
                            "name": "Submission Gap Report",
                            "type": "submission_gap_report",
                            "path": str(gap_report),
                        }
                    ]
                }
            )

        section = dashboard["sections"][0]
        self.assertEqual(dashboard["overall_status"], "blocked")
        self.assertEqual(section["type"], "submission_gap_report")
        self.assertEqual(section["metrics"]["phase_count"], 3)
        self.assertEqual(section["metrics"]["blocked_phases"], 2)
        self.assertEqual(section["metrics"]["hard_blockers"], 3)
        self.assertEqual(section["metrics"]["execution_steps"], 3)
        self.assertEqual(section["metrics"]["blocked_execution_steps"], 2)
        self.assertEqual(section["metrics"]["training_chain_steps"], 2)
        self.assertEqual(section["metrics"]["missing_prerequisite_categories"], 4)
        self.assertEqual(section["metrics"]["missing_prerequisite_items"], 6)
        self.assertIn("3 phases, blocked=2", section["summary"])
        self.assertIn("execution_steps=3", section["summary"])
        self.assertIn("training_chain_steps=2", section["summary"])
        self.assertIn("prereq_items=6", section["summary"])
        self.assertIn("Reviewer-Facing Rebuttal Readiness", "\n".join(dashboard["blockers"]))

    def test_to_markdown_includes_next_actions(self) -> None:
        dashboard = {
            "title": "Dash",
            "objective": "Obj",
            "overall_status": "pass",
            "sections": [
                {
                    "name": "Audit",
                    "type": "audit",
                    "status": "pass",
                    "summary": "ok",
                    "path": "audit.json",
                }
            ],
            "blockers": [],
            "warnings": [],
            "next_actions": ["Run real pipeline"],
        }
        md = to_markdown(dashboard)
        self.assertIn("Run real pipeline", md)
        self.assertIn("Overall status", md)

def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    unittest.main()
