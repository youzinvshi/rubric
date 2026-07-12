from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.convert_policy_rlvr_data import convert_records, load_records


ROOT = Path(__file__).resolve().parents[1]


class ConvertPolicyRLVRDataTest(unittest.TestCase):
    def test_convert_records_dedupes_prompts_and_preserves_metadata(self) -> None:
        rows = [
            {"query": "Solve task", "difficulty": "hard"},
            {"prompt": "Solve task", "difficulty": "duplicate"},
            {"instruction": "Second task", "difficulty": "easy"},
            {"no_query": "skip"},
        ]

        converted = convert_records(
            rows,
            data_source="healthbench_hard",
            prompt_template="Answer carefully:\n{query}",
            metadata_keys=("difficulty",),
        )

        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["prompt"], "Answer carefully:\nSolve task")
        self.assertEqual(converted[0]["data_source"], "healthbench_hard")
        self.assertEqual(converted[0]["difficulty"], "hard")

    def test_load_records_accepts_json_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.json"
            path.write_text(json.dumps({"queries": [{"query": "q1"}]}), encoding="utf-8")

            rows = list(load_records(path))

        self.assertEqual(rows, [{"query": "q1"}])

    def test_cli_writes_jsonl_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "queries.jsonl"
            output = root / "policy.jsonl"
            source.write_text('{"query":"q1"}\n{"query":"q2"}\n', encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/convert_policy_rlvr_data.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--data-source",
                    "arenahard",
                    "--prompt-template",
                    "Policy prompt: {query}",
                    "--min-records",
                    "2",
                ],
                cwd=ROOT,
                check=True,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt"], "Policy prompt: q1")
        self.assertEqual(rows[0]["data_source"], "arenahard")


if __name__ == "__main__":
    unittest.main()
