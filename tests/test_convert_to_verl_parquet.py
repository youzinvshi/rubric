from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import blindspot_rl.verl_reward as verl_reward
from scripts.budget_gate import file_sha256
from scripts.convert_to_verl_parquet import convert_records, holdout_source_blockers, load_json_records


ROOT = Path(__file__).resolve().parents[1]


class ConvertToVerlParquetTest(unittest.TestCase):
    def test_convert_records_preserves_verl_reward_columns(self) -> None:
        rows = [
            {"query": "q1", "gold_rubrics": ["g1", "g1", "g2"], "data_source": "rubricbench"},
            {"prompt": "q2\u2028continued", "gold": "- g3\u2028line\n- g4"},
            {"query": "missing gold"},
        ]

        converted = convert_records(rows, data_source="fallback", prompt_template="Generate:\n{query}")

        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["prompt"], "Generate:\nq1")
        self.assertEqual(converted[0]["gold_rubrics"], ["g1", "g2"])
        self.assertEqual(converted[0]["data_source"], "rubricbench")
        self.assertEqual(converted[0]["ground_truth"], {"gold_rubrics": ["g1", "g2"]})
        self.assertEqual(
            converted[0]["extra_info"],
            {"gold_rubrics": ["g1", "g2"], "query": "q1", "data_source": "rubricbench"},
        )
        self.assertEqual(converted[1]["prompt"], "Generate:\nq2 continued")
        self.assertEqual(converted[1]["gold_rubrics"], ["g3", "g4"])
        self.assertEqual(converted[1]["data_source"], "fallback")
        self.assertEqual(converted[1]["ground_truth"], {"gold_rubrics": ["g3", "g4"]})
        self.assertEqual(
            converted[1]["extra_info"],
            {"gold_rubrics": ["g3", "g4"], "query": "q2 continued", "data_source": "fallback"},
        )

    def test_converted_record_is_accepted_by_verl_reward_hook(self) -> None:
        converted = convert_records(
            [{"query": "photosynthesis", "gold_rubrics": ["mentions sunlight"]}],
            prompt_template="{query}",
        )[0]

        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["mentions sunlight"]',
                ground_truth=converted["ground_truth"],
                extra_info=converted["extra_info"],
            )

        self.assertGreater(reward, 0.0)

    def test_load_json_records_accepts_wrapped_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text(json.dumps({"records": [{"query": "q", "gold_rubrics": ["g"]}]}), encoding="utf-8")

            rows = list(load_json_records(path))

        self.assertEqual(rows, [{"query": "q", "gold_rubrics": ["g"]}])

    def test_holdout_source_blockers_rejects_test_main_sources(self) -> None:
        blockers = holdout_source_blockers(
            [
                {"query": "q1", "gold_rubrics": ["g1"], "data_source": "rubricbench_gold_test_main"},
                {"query": "q2", "gold_rubrics": ["g2"], "metadata": {"split": "test_main"}},
            ],
            input_path=Path("data/processed/splits/rubricbench_gold_test_main.jsonl"),
            data_source="rubricbench_test_main",
        )

        self.assertEqual(len(blockers), 4)
        self.assertIn("input path", blockers[0])
        self.assertIn("data_source=rubricbench_test_main", blockers[1])
        self.assertIn("record 1 data_source=rubricbench_gold_test_main", blockers[2])
        self.assertIn("record 2 split=test_main", blockers[3])

    def test_holdout_source_blockers_rejects_downstream_holdout_sources(self) -> None:
        blockers = holdout_source_blockers(
            [
                {"query": "q1", "gold_rubrics": ["g1"], "data_source": "rewardbench_downstream_holdout"},
                {"query": "q2", "gold_rubrics": ["g2"], "metadata": {"split": "holdout"}},
            ],
            input_path=Path("data/processed/rewardbench_downstream_holdout.jsonl"),
            data_source="rewardbench_downstream",
        )

        self.assertEqual(len(blockers), 4)
        self.assertIn("input path", blockers[0])
        self.assertIn("data_source=rewardbench_downstream", blockers[1])
        self.assertIn("record 1 data_source=rewardbench_downstream_holdout", blockers[2])
        self.assertIn("record 2 split=holdout", blockers[3])

    def test_cli_blocks_test_main_holdout_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "rubricbench_gold_test_main.jsonl"
            output = root / "proxy_gold_verl.jsonl"
            source.write_text(
                json.dumps({"query": "hard-gold query", "gold_rubrics": ["g1"]}) + "\n",
                encoding="utf-8",
            )

            failed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/convert_to_verl_parquet.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--data-source",
                    "rubricbench_test_main",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("forbidden for GRPO/RLVR training conversion", failed.stderr + failed.stdout)
        self.assertFalse(output.exists())

    def test_cli_writes_jsonl_and_enforces_min_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "gold.jsonl"
            output = root / "verl.jsonl"
            report_path = root / "verl_report.json"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"query": "q1", "gold_rubrics": ["g1"]}),
                        json.dumps({"query": "q2", "gold_rubrics": ["g2"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/convert_to_verl_parquet.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--data-source",
                    "rubricbench",
                    "--min-records",
                    "2",
                    "--report-output",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 2)
            self.assertEqual(set(rows[0]), {"prompt", "gold_rubrics", "data_source", "ground_truth", "extra_info"})
            self.assertEqual(report["input"], str(source))
            self.assertEqual(report["input_sha256"], file_sha256(source))
            self.assertEqual(report["output"], str(output))
            self.assertEqual(report["output_sha256"], file_sha256(output))
            self.assertEqual(report["n_records"], 2)
            self.assertEqual(report["data_source"], "rubricbench")
            self.assertIn("test_main", report["forbidden_source_markers"])
            self.assertIn("holdout", report["forbidden_source_markers"])
            self.assertIn("downstream", report["forbidden_source_markers"])
            self.assertIn("test_main", report["forbidden_splits"])
            self.assertIn("holdout", report["forbidden_splits"])
            self.assertIn("downstream", report["forbidden_splits"])
            self.assertIn("test", report["forbidden_splits"])

            failed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/convert_to_verl_parquet.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(root / "too_small.jsonl"),
                    "--min-records",
                    "3",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Too few valid verl records", failed.stderr + failed.stdout)

    def test_cli_writes_parquet_that_round_trips_into_reward_hook(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas/pyarrow are required for parquet round-trip")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "gold.jsonl"
            output = root / "verl.parquet"
            source.write_text(
                json.dumps({"query": "photosynthesis", "gold_rubrics": ["mentions sunlight"]}) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/convert_to_verl_parquet.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--data-source",
                    "rubricbench",
                    "--min-records",
                    "1",
                ],
                cwd=ROOT,
                check=True,
            )
            record = pd.read_parquet(output).to_dict(orient="records")[0]

        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["mentions sunlight"]',
                ground_truth=record["ground_truth"],
                extra_info=record["extra_info"],
            )

        self.assertGreater(reward, 0.0)


@contextmanager
def reward_env(**values: str) -> Generator[None, None, None]:
    old = {key: os.environ.get(key) for key in values}
    old_embedder = verl_reward._EMBEDDER
    old_verifier = verl_reward._VERIFIER
    try:
        for key, value in values.items():
            os.environ[key] = value
        verl_reward._EMBEDDER = None
        verl_reward._VERIFIER = None
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        verl_reward._EMBEDDER = old_embedder
        verl_reward._VERIFIER = old_verifier


if __name__ == "__main__":
    unittest.main()
