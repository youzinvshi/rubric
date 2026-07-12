from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_evidence_matrix import build_matrix, load_config, nested_get, parse_selector, to_markdown


class BuildEvidenceMatrixTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_evidence_matrix_config.json"))

        self.assertIn("Evidence Matrix config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Evidence Matrix config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Evidence Matrix config must be a JSON object", str(context.exception))

    def test_nested_get_supports_dicts_and_lists(self) -> None:
        data = {"rows": [{"n": 2}]}
        self.assertEqual(nested_get(data, "rows.0.n"), 2)
        self.assertIsNone(nested_get(data, "rows.1.n"))

    def test_nested_get_supports_list_selector(self) -> None:
        data = {"metrics": [{"metric": "coverage", "ci_lower": 0.5}, {"metric": "blind", "ci_lower": 0.25}]}
        self.assertEqual(nested_get(data, "metrics[metric=blind].ci_lower"), 0.25)
        self.assertIsNone(nested_get(data, "metrics[metric=missing].ci_lower"))

    def test_parse_selector(self) -> None:
        self.assertEqual(parse_selector("metrics[metric=blind]"), ("metrics", "metric", "blind"))
        self.assertIsNone(parse_selector("metrics.0"))

    def test_build_matrix_marks_safe_and_missing_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"n": 2, "mean_blind": 0.25}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "section": "Motivation",
                        "claim": "Blind spots are measurable.",
                        "artifacts": ["summary.json"],
                        "metrics": [
                            {"path": "summary.json", "metric": "n", "op": ">", "value": 0},
                            {"path": "summary.json", "metric": "mean_blind", "op": ">=", "value": 0.0},
                        ],
                    },
                    {
                        "id": "C2",
                        "claim": "Missing claim.",
                        "artifacts": ["missing.csv"],
                    },
                ]
            }
            rows = build_matrix(config, root=root)
        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertEqual(rows[1]["status"], "missing_evidence")

    def test_build_matrix_supports_metric_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.json").write_text('{"mean_coverage": 0.5}', encoding="utf-8")
            (root / "rl.json").write_text('{"mean_coverage": 0.7}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "RL improves coverage.",
                        "comparisons": [
                            {
                                "label": "RL Cov gain",
                                "left_path": "rl.json",
                                "left_metric": "mean_coverage",
                                "right_path": "base.json",
                                "right_metric": "mean_coverage",
                                "op": ">=",
                                "value": 0.1,
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)
        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("RL Cov gain", rows[0]["evidence"])

    def test_build_matrix_can_gate_on_ci_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ci.json").write_text(
                '{"metrics": [{"metric": "blind", "ci_lower": 0.21}]}',
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Blind spot is robust.",
                        "metrics": [
                            {
                                "path": "ci.json",
                                "metric": "metrics[metric=blind].ci_lower",
                                "op": ">=",
                                "value": 0.2,
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)
        self.assertEqual(rows[0]["status"], "safe_to_claim")

    def test_build_matrix_supports_exact_json_value_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(
                '{"embedding_model": "BAAI/bge-large-en-v1.5", "weights": {"coverage": 1.0}}',
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "BSC protocol is locked.",
                        "values": [
                            {
                                "label": "Embedding model",
                                "path": "summary.json",
                                "key": "embedding_model",
                                "op": "==",
                                "value": "BAAI/bge-large-en-v1.5",
                            },
                            {
                                "label": "Coverage weight",
                                "path": "summary.json",
                                "key": "weights.coverage",
                                "op": "==",
                                "value": 1.0,
                            },
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("Embedding model", rows[0]["evidence"])

    def test_build_matrix_supports_json_value_membership_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"projection": "umap_fallback_pca"}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic visualization uses an allowed projection.",
                        "values": [
                            {
                                "path": "summary.json",
                                "key": "projection",
                                "op": "in",
                                "value": ["umap", "umap_fallback_pca"],
                            }
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("projection=umap_fallback_pca in", rows[0]["evidence"])

    def test_build_matrix_marks_json_value_outside_membership_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"projection": "pca"}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic visualization uses an allowed projection.",
                        "values": [
                            {
                                "path": "summary.json",
                                "key": "projection",
                                "op": "in",
                                "value": ["umap", "umap_fallback_pca"],
                            }
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("projection=pca in", rows[0]["evidence"])

    def test_build_matrix_supports_list_root_metric_and_value_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ablation.json").write_text(
                '[{"variant":"no_verifier_filter","n":3},{"variant":"verifier_filtered","n":3}]',
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Verifier-filter ablation has both variants.",
                        "metrics": [
                            {"path": "ablation.json", "metric": "0.n", "op": ">", "value": 0},
                            {"path": "ablation.json", "metric": "1.n", "op": ">", "value": 0},
                        ],
                        "values": [
                            {
                                "path": "ablation.json",
                                "key": "0.variant",
                                "op": "==",
                                "value": "no_verifier_filter",
                            },
                            {
                                "path": "ablation.json",
                                "key": "1.variant",
                                "op": "==",
                                "value": "verifier_filtered",
                            },
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("0.n=3.0000", rows[0]["evidence"])
        self.assertIn("0.variant=no_verifier_filter", rows[0]["evidence"])

    def test_build_matrix_marks_failed_value_check_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"embedding_model": "token-overlap"}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "BSC uses production embeddings.",
                        "values": [
                            {
                                "path": "summary.json",
                                "key": "embedding_model",
                                "op": "==",
                                "value": "BAAI/bge-large-en-v1.5",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("token-overlap", rows[0]["evidence"])
    def test_build_matrix_can_treat_configured_failures_as_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audit.json").write_text('{"ok": false, "overlap_query_count": 0}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C0",
                        "claim": "Holdout audits are complete.",
                        "values": [
                            {
                                "label": "Final contamination audit completed",
                                "path": "audit.json",
                                "key": "ok",
                                "op": "==",
                                "value": True,
                                "fail_status": "missing",
                            }
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "missing_evidence")
        self.assertIn("[missing] Final contamination audit completed", rows[0]["evidence"])

    def test_build_matrix_supports_value_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"input_sha256": "abc"}', encoding="utf-8")
            (root / "budget.json").write_text('{"contract": {"input_sha256": "abc"}}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Budget report is bound to evaluated input.",
                        "value_comparisons": [
                            {
                                "label": "Input hash matches budget",
                                "left_path": "summary.json",
                                "left_key": "input_sha256",
                                "right_path": "budget.json",
                                "right_key": "contract.input_sha256",
                                "op": "==",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("Input hash matches budget", rows[0]["evidence"])

    def test_build_matrix_supports_file_sha256_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "semantic_space_points.csv"
            artifact.write_text("point_id\n0\n", encoding="utf-8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            (root / "summary.json").write_text(f'{{"point_csv_sha256": "{digest}"}}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic-space summary is bound to point CSV.",
                        "file_sha256_checks": [
                            {
                                "label": "Point CSV SHA matches summary",
                                "json_path": "summary.json",
                                "json_key": "point_csv_sha256",
                                "file_path": "semantic_space_points.csv",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("Point CSV SHA matches summary", rows[0]["evidence"])

    def test_build_matrix_marks_file_sha256_mismatch_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "semantic_space_points.csv"
            artifact.write_text("point_id\n0\n", encoding="utf-8")
            stale_digest = "0" * 64
            (root / "summary.json").write_text(f'{{"point_csv_sha256": "{stale_digest}"}}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic-space summary is bound to point CSV.",
                        "file_sha256_checks": [
                            {
                                "json_path": "summary.json",
                                "json_key": "point_csv_sha256",
                                "file_path": "semantic_space_points.csv",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("000000", rows[0]["evidence"])

    def test_build_matrix_marks_empty_file_sha256_contract_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "semantic_space_points.csv"
            artifact.write_text("point_id\n0\n", encoding="utf-8")
            (root / "summary.json").write_text('{"point_csv_sha256": ""}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic-space summary is bound to point CSV.",
                        "file_sha256_checks": [
                            {
                                "json_path": "summary.json",
                                "json_key": "point_csv_sha256",
                                "file_path": "semantic_space_points.csv",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "missing_evidence")
        self.assertIn("[missing] file SHA256 check", rows[0]["evidence"])
        self.assertIn("point_csv_sha256(summary.json) is empty", rows[0]["evidence"])

    def test_build_matrix_supports_table_value_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main_table.csv").write_text(
                "method,downstream_status,downstream_paper_claim_eligible,accuracy\n"
                "base,pass,true,0.7\n",
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Downstream table is paper eligible.",
                        "table_values": [
                            {
                                "label": "Base table downstream status",
                                "path": "main_table.csv",
                                "row_key": "method",
                                "row_value": "base",
                                "key": "downstream_status",
                                "op": "==",
                                "value": "pass",
                            },
                            {
                                "label": "Base table downstream eligibility",
                                "path": "main_table.csv",
                                "row_key": "method",
                                "row_value": "base",
                                "key": "downstream_paper_claim_eligible",
                                "op": "==",
                                "value": "true",
                            },
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("Base table downstream status", rows[0]["evidence"])

    def test_build_matrix_supports_csv_content_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "points.csv").write_text(
                "point_id,method,source_type,x,y,nearest_gold_point_id,nearest_gold_category,nearest_gold_similarity,nearest_gold_text\n"
                "0,human_gold,gold,0.0,0.0,,,,\n"
                "1,sft_rl,generated,0.1,0.2,0,evidence_grounding,0.9,Uses evidence\n",
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic-space points are auditable.",
                        "csv_checks": [
                            {
                                "label": "Semantic point CSV schema",
                                "path": "points.csv",
                                "columns": [
                                    "point_id",
                                    "method",
                                    "source_type",
                                    "x",
                                    "y",
                                    "nearest_gold_point_id",
                                    "nearest_gold_category",
                                    "nearest_gold_similarity",
                                    "nearest_gold_text",
                                ],
                                "column_mode": "exact",
                            },
                            {
                                "label": "Generated points have nearest-gold audit fields",
                                "path": "points.csv",
                                "where": {"source_type": "generated"},
                                "min_rows": 1,
                                "non_empty": [
                                    "nearest_gold_point_id",
                                    "nearest_gold_category",
                                    "nearest_gold_similarity",
                                    "nearest_gold_text",
                                ],
                                "numeric": ["x", "y", "nearest_gold_similarity"],
                            },
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "safe_to_claim")
        self.assertIn("Generated points have nearest-gold audit fields", rows[0]["evidence"])

    def test_build_matrix_marks_failed_csv_content_check_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "points.csv").write_text(
                "point_id,method,source_type,nearest_gold_point_id,nearest_gold_category\n"
                "1,sft_rl,generated,,\n",
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Semantic-space points are auditable.",
                        "csv_checks": [
                            {
                                "label": "Generated points have nearest-gold audit fields",
                                "path": "points.csv",
                                "where": {"source_type": "generated"},
                                "min_rows": 1,
                                "non_empty": ["nearest_gold_point_id", "nearest_gold_category"],
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("empty nearest_gold_point_id", rows[0]["evidence"])

    def test_build_matrix_marks_failed_table_value_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main_table.csv").write_text(
                "method,downstream_status,downstream_paper_claim_eligible,accuracy\n"
                "base,not_paper_eligible,false,\n",
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Downstream table is paper eligible.",
                        "table_values": [
                            {
                                "path": "main_table.csv",
                                "row_value": "base",
                                "key": "downstream_status",
                                "op": "==",
                                "value": "pass",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("not_paper_eligible", rows[0]["evidence"])

    def test_build_matrix_marks_failed_value_comparison_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"input_sha256": "abc"}', encoding="utf-8")
            (root / "budget.json").write_text('{"contract": {"input_sha256": "def"}}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Budget report is bound to evaluated input.",
                        "value_comparisons": [
                            {
                                "left_path": "summary.json",
                                "left_key": "input_sha256",
                                "right_path": "budget.json",
                                "right_key": "contract.input_sha256",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("abc", rows[0]["evidence"])
        self.assertIn("def", rows[0]["evidence"])

    def test_build_matrix_marks_empty_value_comparison_contract_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"input_sha256": "abc"}', encoding="utf-8")
            (root / "budget.json").write_text('{"contract": {"input_sha256": ""}}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Budget report is bound to evaluated input.",
                        "value_comparisons": [
                            {
                                "left_path": "summary.json",
                                "left_key": "input_sha256",
                                "right_path": "budget.json",
                                "right_key": "contract.input_sha256",
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "missing_evidence")
        self.assertIn("[missing] value comparison", rows[0]["evidence"])
        self.assertIn("contract.input_sha256(budget.json)= is empty", rows[0]["evidence"])

    def test_downstream_claim_remains_contradicted_when_provenance_hashes_disagree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "scorer": "api",
                        "paper_claim_eligible": True,
                        "paper_claim_eligibility_blockers": [],
                        "scorer_provider_sha256": "provider-summary-sha",
                        "input_sha256": "eval-input-sha",
                        "per_item_sha256": "per-item-summary-sha",
                        "per_item_rows": 2,
                    }
                ),
                encoding="utf-8",
            )
            (root / "budget.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "contract": {
                            "input_sha256": "eval-input-sha",
                            "providers_sha256": "provider-budget-sha",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "ci.json").write_text(
                json.dumps({"input_sha256": "per-item-ci-sha", "n": 2}),
                encoding="utf-8",
            )
            (root / "main_table.csv").write_text(
                "method,downstream_status,downstream_paper_claim_eligible,accuracy\n"
                "sft_rl,pass,true,0.7\n",
                encoding="utf-8",
            )
            config = {
                "claims": [
                    {
                        "id": "C4",
                        "claim": "Downstream utility is paper eligible only with bound provenance.",
                        "values": [
                            {"path": "summary.json", "key": "scorer", "op": "==", "value": "api"},
                            {"path": "summary.json", "key": "paper_claim_eligible", "op": "==", "value": True},
                            {
                                "path": "summary.json",
                                "key": "paper_claim_eligibility_blockers",
                                "op": "==",
                                "value": [],
                            },
                            {"path": "budget.json", "key": "ok", "op": "==", "value": True},
                        ],
                        "table_values": [
                            {
                                "path": "main_table.csv",
                                "row_value": "sft_rl",
                                "key": "downstream_status",
                                "op": "==",
                                "value": "pass",
                            },
                            {
                                "path": "main_table.csv",
                                "row_value": "sft_rl",
                                "key": "downstream_paper_claim_eligible",
                                "op": "==",
                                "value": "true",
                            },
                        ],
                        "value_comparisons": [
                            {
                                "label": "Summary provider hash matches budget report",
                                "left_path": "summary.json",
                                "left_key": "scorer_provider_sha256",
                                "right_path": "budget.json",
                                "right_key": "contract.providers_sha256",
                            },
                            {
                                "label": "Summary per-item hash matches CI input",
                                "left_path": "summary.json",
                                "left_key": "per_item_sha256",
                                "right_path": "ci.json",
                                "right_key": "input_sha256",
                            },
                            {
                                "label": "Summary per-item row count matches CI rows",
                                "left_path": "summary.json",
                                "left_key": "per_item_rows",
                                "right_path": "ci.json",
                                "right_key": "n",
                            },
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertIn("[pass] summary.json::paper_claim_eligible", rows[0]["evidence"])
        self.assertIn("[pass] main_table.csv::method=sft_rl::downstream_status", rows[0]["evidence"])
        self.assertIn("[fail] Summary provider hash matches budget report", rows[0]["evidence"])
        self.assertIn("[fail] Summary per-item hash matches CI input", rows[0]["evidence"])

    def test_build_matrix_marks_invalid_metric_json_as_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("{bad", encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "Blind spots are measurable.",
                        "metrics": [
                            {"path": "summary.json", "metric": "mean_blind", "op": ">=", "value": 0.1}
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "missing_evidence")
        self.assertIn("not valid JSON", rows[0]["evidence"])

    def test_build_matrix_marks_invalid_comparison_json_as_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.json").write_text("{bad", encoding="utf-8")
            (root / "rl.json").write_text('{"mean_coverage": 0.7}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "RL improves coverage.",
                        "comparisons": [
                            {
                                "left_path": "rl.json",
                                "left_metric": "mean_coverage",
                                "right_path": "base.json",
                                "right_metric": "mean_coverage",
                                "op": ">=",
                                "value": 0.1,
                            }
                        ],
                    }
                ]
            }

            rows = build_matrix(config, root=root)

        self.assertEqual(rows[0]["status"], "missing_evidence")
        self.assertIn("not valid JSON", rows[0]["evidence"])

    def test_build_matrix_marks_failed_comparison_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.json").write_text('{"accuracy": 0.8}', encoding="utf-8")
            (root / "rl.json").write_text('{"accuracy": 0.7}', encoding="utf-8")
            config = {
                "claims": [
                    {
                        "id": "C1",
                        "claim": "RL improves downstream accuracy.",
                        "comparisons": [
                            {
                                "left_path": "rl.json",
                                "left_metric": "accuracy",
                                "right_path": "base.json",
                                "right_metric": "accuracy",
                                "op": ">=",
                                "value": 0.0,
                            }
                        ],
                    }
                ]
            }
            rows = build_matrix(config, root=root)
        self.assertEqual(rows[0]["status"], "contradicted")

    def test_to_markdown_escapes_pipes(self) -> None:
        text = to_markdown(
            [
                {
                    "claim_id": "C1",
                    "paper_section": "A|B",
                    "status": "safe_to_claim",
                    "claim": "x",
                    "evidence": "y",
                }
            ]
        )
        self.assertIn("A\\|B", text)


if __name__ == "__main__":
    unittest.main()
