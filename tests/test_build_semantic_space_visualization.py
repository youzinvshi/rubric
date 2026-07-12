from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blindspot_rl.reward_bsc import TokenOverlapEmbedder
from scripts.budget_gate import file_sha256
from scripts.build_semantic_space_visualization import (
    POINT_CSV_COLUMNS,
    build_semantic_space,
    main,
    parse_input_spec,
    project_2d,
)


class BuildSemanticSpaceVisualizationTest(unittest.TestCase):
    def test_parse_input_spec_accepts_optional_label(self) -> None:
        label, path = parse_input_spec("sft_rl=/tmp/x.jsonl")

        self.assertEqual(label, "sft_rl")
        self.assertEqual(path, Path("/tmp/x.jsonl"))

    def test_build_semantic_space_outputs_gold_and_generated_points(self) -> None:
        points, summary = build_semantic_space(
            [
                {
                    "query": "q",
                    "gold_rubrics": ["Uses citations and evidence", "Follows the requested JSON format"],
                    "response": ["Uses evidence from the source", "Follows JSON format exactly"],
                    "_input_label": "base",
                }
            ],
            embedder=TokenOverlapEmbedder(),
        )

        self.assertEqual(summary["n_gold"], 2)
        self.assertEqual(summary["n_generated"], 2)
        self.assertEqual(summary["methods"], ["base"])
        self.assertEqual(len(points), 4)
        self.assertEqual(summary["n_points"], 4)
        self.assertEqual([point["point_id"] for point in points], [0, 1, 2, 3])
        self.assertTrue(all("x" in point and "y" in point for point in points))
        self.assertEqual(points[0]["source_type"], "gold")
        self.assertEqual(points[0]["category"], "evidence_grounding")
        self.assertEqual(summary["generated_gold_category_count_by_method"]["base"], 2)
        self.assertEqual(summary["generated_gold_category_coverage_by_method"]["base"], 1.0)
        self.assertIn("base", summary["mean_nearest_gold_similarity_by_method"])
        self.assertIn("base", summary["generated_dispersion_by_method"])
        self.assertEqual(summary["nearest_gold_category_count_by_method"]["base"], 2)
        self.assertEqual(summary["nearest_gold_category_coverage_by_method"]["base"], 1.0)
        self.assertEqual(summary["n_gold_clusters"], 2)
        self.assertEqual(summary["nearest_gold_cluster_count_by_method"]["base"], 2)
        self.assertEqual(summary["nearest_gold_cluster_coverage_by_method"]["base"], 1.0)
        self.assertEqual(summary["nearest_gold_cluster_distribution_by_method"]["base"], {"g000": 1, "g001": 1})
        self.assertEqual(summary["nearest_gold_cluster_entropy_by_method"]["base"], 1.0)
        self.assertEqual(summary["gold_cluster_tau"], 0.75)
        self.assertEqual(summary["embedding_model"], "TokenOverlapEmbedder")
        self.assertEqual(summary["point_csv_schema_version"], 3)
        self.assertEqual(summary["point_csv_columns"], POINT_CSV_COLUMNS)
        gold_points = [point for point in points if point["source_type"] == "gold"]
        self.assertTrue(all(point["gold_cluster_id"] for point in gold_points))
        generated_points = [point for point in points if point["source_type"] == "generated"]
        self.assertTrue(all(point["nearest_gold_point_id"] != "" for point in generated_points))
        self.assertTrue(all(point["nearest_gold_cluster_id"] != "" for point in generated_points))
        self.assertTrue(all(point["nearest_gold_category"] for point in generated_points))
        self.assertTrue(all(point["nearest_gold_similarity"] >= 0.0 for point in generated_points))
        self.assertTrue(all(point["nearest_gold_same_record"] for point in generated_points))

    def test_summary_records_sft_rl_vs_sft_only_visualization_deltas(self) -> None:
        _, summary = build_semantic_space(
            [
                {
                    "query": "q",
                    "gold_rubrics": ["Uses citations and evidence", "Follows the requested JSON format"],
                    "response": ["Uses citations"],
                    "_input_label": "sft_only",
                },
                {
                    "query": "q",
                    "gold_rubrics": ["Uses citations and evidence", "Follows the requested JSON format"],
                    "response": ["Uses citations", "Follows JSON format exactly"],
                    "_input_label": "sft_rl",
                },
            ],
            embedder=TokenOverlapEmbedder(),
        )

        self.assertEqual(summary["methods"], ["sft_only", "sft_rl"])
        self.assertGreaterEqual(summary["sft_rl_vs_sft_only_generated_gold_category_coverage_delta"], 0.0)
        self.assertGreaterEqual(summary["sft_rl_vs_sft_only_nearest_gold_category_coverage_delta"], 0.0)
        self.assertGreaterEqual(summary["sft_rl_vs_sft_only_gold_cluster_coverage_delta"], 0.0)
        self.assertGreaterEqual(summary["sft_rl_vs_sft_only_nearest_gold_cluster_entropy_delta"], 0.0)
        self.assertIsNotNone(summary["sft_rl_vs_sft_only_nearest_gold_similarity_delta"])
        self.assertIsNone(summary["sft_rl_vs_sft_only_generated_dispersion_delta"])

    def test_requested_umap_falls_back_without_optional_dependency(self) -> None:
        points, summary = build_semantic_space(
            [
                {
                    "query": "q",
                    "gold_rubrics": ["Uses citations and evidence"],
                    "response": ["Follows JSON format exactly"],
                    "_input_label": "sft_rl",
                }
            ],
            embedder=TokenOverlapEmbedder(),
            projection="umap",
        )

        self.assertEqual(len(points), 2)
        self.assertEqual(summary["requested_projection"], "umap")
        self.assertIn(summary["projection"], {"umap", "umap_fallback_pca"})

    def test_tsne_projection_records_requested_method_or_fallback(self) -> None:
        coords, projection = project_2d(
            TokenOverlapEmbedder().encode(["alpha evidence", "beta format", "gamma safety"]),
            method="tsne",
        )

        self.assertEqual(coords.shape, (3, 2))
        self.assertIn(projection, {"tsne", "tsne_fallback_pca"})

    def test_main_writes_visualization_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "records.jsonl"
            output_dir = root / "semantic_space"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "q",
                        "split": "test_main",
                        "data_source": "rubricbench",
                        "gold_rubrics": ["Uses citations and evidence"],
                        "response": ["Uses citations"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            expected_input_sha256 = file_sha256(input_path)
            join_report_path = root / "bsc_join_report.json"
            join_report_path.write_text(
                json.dumps(
                    {
                        "output": str(input_path),
                        "output_sha256": expected_input_sha256,
                        "gold": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                        "prediction": "data/processed/base/predictions.jsonl",
                        "join_key": "query",
                        "n_joined": 1,
                        "unmatched_gold": 0,
                        "unmatched_prediction": 0,
                        "duplicate_gold_keys": 0,
                        "duplicate_prediction_keys": 0,
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "build_semantic_space_visualization.py",
                "--input",
                f"base={input_path}",
                "--join-report",
                f"base={join_report_path}",
                "--output-dir",
                str(output_dir),
                "--embedding-model",
                "token-overlap",
                "--projection",
                "umap",
            ]
            with patch("sys.argv", argv):
                main()

            summary = json.loads((output_dir / "semantic_space_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_points"], 2)
            self.assertEqual(summary["requested_projection"], "umap")
            self.assertEqual(summary["gold_cluster_tau"], 0.75)
            self.assertEqual(summary["embedding_model"], "token-overlap")
            self.assertEqual(summary["point_csv_columns"], POINT_CSV_COLUMNS)
            self.assertEqual(
                summary["inputs"],
                [
                    {
                        "schema_version": 2,
                        "label": "base",
                        "path": str(input_path),
                        "sha256": expected_input_sha256,
                        "n_records": 1,
                        "n_unique_queries": 1,
                        "split_counts": {"test_main": 1},
                        "data_source_counts": {"rubricbench": 1},
                        "join_report": {
                            "path": str(join_report_path),
                            "sha256": file_sha256(join_report_path),
                            "output_sha256": expected_input_sha256,
                            "output": str(input_path),
                            "gold": "data/processed/splits/rubricbench_gold_test_main.jsonl",
                            "prediction": "data/processed/base/predictions.jsonl",
                            "join_key": "query",
                            "n_joined": 1,
                            "unmatched_gold": 0,
                            "unmatched_prediction": 0,
                            "duplicate_gold_keys": 0,
                            "duplicate_prediction_keys": 0,
                        },
                    }
                ],
            )
            self.assertTrue((output_dir / "semantic_space_points.csv").exists())
            self.assertEqual(summary["output_artifacts_schema_version"], 1)
            self.assertEqual(summary["point_csv"], str(output_dir / "semantic_space_points.csv"))
            self.assertEqual(summary["point_csv_sha256"], file_sha256(output_dir / "semantic_space_points.csv"))
            self.assertEqual(summary["point_csv_rows_count"], summary["n_points"])
            self.assertTrue(summary["point_csv_rows_match_n_points"])
            self.assertEqual(summary["svg"], str(output_dir / "semantic_space.svg"))
            self.assertEqual(summary["svg_sha256"], file_sha256(output_dir / "semantic_space.svg"))
            self.assertEqual(summary["pdf"], str(output_dir / "semantic_space.pdf"))
            self.assertEqual(summary["pdf_sha256"], file_sha256(output_dir / "semantic_space.pdf"))
            with (output_dir / "semantic_space_points.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), POINT_CSV_COLUMNS)
            generated = [row for row in rows if row["source_type"] == "generated"]
            self.assertEqual(len(generated), 1)
            self.assertNotEqual(generated[0]["nearest_gold_point_id"], "")
            self.assertNotEqual(generated[0]["nearest_gold_cluster_id"], "")
            self.assertEqual(generated[0]["nearest_gold_category"], "evidence_grounding")
            svg = (output_dir / "semantic_space.svg").read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Evaluation-Criteria Semantic Space", svg)
            pdf = (output_dir / "semantic_space.pdf").read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF-1.4"))
            self.assertIn(b"%%EOF", pdf)


if __name__ == "__main__":
    unittest.main()
