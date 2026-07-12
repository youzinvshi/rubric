from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.budget_gate import file_sha256
from scripts.build_sft_data import filter_records_by_data_source, group_teacher_outputs, holdout_source_blockers, main


class BuildSFTDataTest(unittest.TestCase):
    def test_filter_records_by_data_source_keeps_selected_domains(self) -> None:
        records = [
            {"query": "q1", "rubrics": ["a"], "data_source": "rubricbench_teacher_generation"},
            {"query": "q2", "rubrics": ["b"], "data_source": "healthbench_teacher_generation"},
            {"query": "q3", "rubrics": ["c"], "data_source": "writingbench_teacher_generation"},
        ]

        filtered = filter_records_by_data_source(records, ["healthbench_teacher_generation"])

        self.assertEqual([record["query"] for record in filtered], ["q2"])

    def test_filter_records_by_data_source_no_filter_keeps_all(self) -> None:
        records = [{"query": "q1", "rubrics": ["a"], "data_source": "rubricbench"}]

        self.assertEqual(filter_records_by_data_source(records, []), records)

    def test_group_teacher_outputs_accepts_filtered_records(self) -> None:
        records = filter_records_by_data_source(
            [
                {"query": "q1", "teacher": "a", "rubrics": ["criterion a"], "data_source": "healthbench"},
                {"query": "q2", "teacher": "b", "rubrics": ["criterion b"], "data_source": "writingbench"},
            ],
            ["healthbench"],
        )

        grouped = group_teacher_outputs(records)

        self.assertEqual(list(grouped), ["q1"])
        self.assertEqual(grouped["q1"][0]["rubrics"], ["criterion a"])

    def test_holdout_source_blockers_rejects_test_main_records(self) -> None:
        blockers = holdout_source_blockers(
            [
                {"query": "q1", "data_source": "rubricbench_gold_test_main", "rubrics": ["criterion a"]},
                {"query": "q2", "metadata": {"split": "test_main"}, "rubrics": ["criterion b"]},
            ],
            input_path=Path("data/teacher_outputs/rubricbench_test_main.jsonl"),
        )

        self.assertEqual(len(blockers), 3)
        self.assertIn("input path", blockers[0])
        self.assertIn("data_source=rubricbench_gold_test_main", blockers[1])
        self.assertIn("split=test_main", blockers[2])

    def test_holdout_source_blockers_rejects_downstream_holdout_sources(self) -> None:
        blockers = holdout_source_blockers(
            [
                {"query": "q1", "data_source": "rewardbench_downstream_holdout", "rubrics": ["criterion a"]},
                {"query": "q2", "metadata": {"split": "holdout"}, "rubrics": ["criterion b"]},
            ],
            input_path=Path("data/teacher_outputs/rewardbench_downstream_holdout.jsonl"),
        )

        self.assertEqual(len(blockers), 3)
        self.assertIn("input path", blockers[0])
        self.assertIn("data_source=rewardbench_downstream_holdout", blockers[1])
        self.assertIn("split=holdout", blockers[2])

    def test_main_blocks_test_main_holdout_records_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "teachers.jsonl"
            sft_path = root / "sft.jsonl"
            proxy_path = root / "proxy.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "query": "hard-gold query",
                        "teacher": "gpt-4o",
                        "rubrics": ["criterion a"],
                        "split": "test_main",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "build_sft_data.py",
                    "--input",
                    str(input_path),
                    "--sft-output",
                    str(sft_path),
                    "--proxy-gold-output",
                    str(proxy_path),
                    "--embedding-model",
                    "token-overlap",
                ],
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()

            self.assertIn("split=test_main is forbidden", str(ctx.exception))
            self.assertFalse(sft_path.exists())
            self.assertFalse(proxy_path.exists())

    def test_main_enforces_min_teachers_before_building_proxy_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "teachers.jsonl"
            sft_path = root / "sft.jsonl"
            proxy_path = root / "proxy.jsonl"
            stats_path = root / "stats.jsonl"
            report_path = root / "report.json"
            records = [
                {"query": "q1", "teacher": "gpt-4o", "rubrics": ["criterion a"]},
                {"query": "q1", "teacher": "claude", "rubrics": ["criterion b"]},
                {"query": "q2", "teacher": "gpt-4o", "rubrics": ["criterion c"]},
            ]
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "build_sft_data.py",
                    "--input",
                    str(input_path),
                    "--sft-output",
                    str(sft_path),
                    "--proxy-gold-output",
                    str(proxy_path),
                    "--stats-output",
                    str(stats_path),
                    "--report-output",
                    str(report_path),
                    "--embedding-model",
                    "token-overlap",
                    "--min-teachers",
                    "2",
                ],
            ):
                main()

            proxy_rows = [json.loads(line) for line in proxy_path.read_text(encoding="utf-8").splitlines()]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual([row["query"] for row in proxy_rows], ["q1"])
            self.assertEqual(proxy_rows[0]["teachers"], ["claude", "gpt-4o"])
            self.assertEqual(report["input"], str(input_path))
            self.assertEqual(report["input_sha256"], file_sha256(input_path))
            self.assertEqual(report["sft_output"], str(sft_path))
            self.assertEqual(report["sft_output_sha256"], file_sha256(sft_path))
            self.assertEqual(report["proxy_gold_output"], str(proxy_path))
            self.assertEqual(report["proxy_gold_output_sha256"], file_sha256(proxy_path))
            self.assertEqual(report["stats_output"], str(stats_path))
            self.assertEqual(report["stats_output_sha256"], file_sha256(stats_path))
            self.assertEqual(report["n_input_records"], 3)
            self.assertEqual(report["n_filtered_records"], 3)
            self.assertEqual(report["n_grouped_queries"], 2)
            self.assertEqual(report["n_sft_records"], 1)
            self.assertEqual(report["n_proxy_gold_records"], 1)
            self.assertEqual(report["embedding_model"], "token-overlap")
            self.assertEqual(report["min_teachers"], 2)
            self.assertIn("test_main", report["forbidden_data_source_markers"])
            self.assertIn("holdout", report["forbidden_data_source_markers"])
            self.assertIn("downstream", report["forbidden_data_source_markers"])
            self.assertIn("test_main", report["forbidden_splits"])
            self.assertIn("holdout", report["forbidden_splits"])
            self.assertIn("downstream", report["forbidden_splits"])
            self.assertIn("test", report["forbidden_splits"])


if __name__ == "__main__":
    unittest.main()
