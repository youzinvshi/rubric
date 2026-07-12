from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.llm_api import LLMConfig
from scripts.generate_model_rubrics import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE, generate_rows, load_done, write_jsonl
from scripts.generate_teacher_rubrics import SYSTEM_PROMPT as TEACHER_SYSTEM_PROMPT, construct_prompt
from scripts.prepare_bsc_eval import build_prediction_map


class FakeClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]]) -> str:
        assert "q1" in messages[-1]["content"]
        return '["mentions evidence", "checks format"]'


class GenerateModelRubricsTest(unittest.TestCase):
    def test_default_prompts_use_evaluation_criteria_framing(self) -> None:
        combined = "\n".join([DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE, TEACHER_SYSTEM_PROMPT])

        self.assertIn("elicit evaluation criteria", combined.lower())
        self.assertIn("Elicit 6-10 atomic evaluation criteria", DEFAULT_USER_TEMPLATE)
        self.assertNotIn("You generate evaluation rubrics", combined)
        self.assertNotIn("Generate 6-10 rubrics", combined)

    def test_teacher_domain_prompts_use_criteria_framing(self) -> None:
        examples = [
            (
                "rewardbench",
                {
                    "chosen": "A",
                    "rejected": "B",
                },
            ),
            ("healthbench", {}),
            ("writingbench", {}),
            ("ifbench", {}),
            ("beir", {}),
            ("generic", {}),
        ]

        for data_source, record in examples:
            with self.subTest(data_source=data_source):
                system, user = construct_prompt(data_source, record, "q1")
                combined = f"{system}\n{user}"
                self.assertIn("criteria", combined.lower())
                self.assertNotIn("Generate 6-10 rubrics", combined)
                self.assertNotIn("Generate 5-8 atomic evaluation rubrics", combined)
                self.assertNotIn("You are generating", system)

    def test_generate_rows_outputs_joinable_model_records(self) -> None:
        rows = generate_rows(
            queries=["q1"],
            configs=[
                LLMConfig(
                    name="base",
                    model="local-model",
                    base_url="http://localhost:8000/v1",
                    api_key_env="LOCAL_KEY",
                )
            ],
            client_factory=FakeClient,
        )
        self.assertEqual(rows[0]["method"], "base")
        self.assertEqual(rows[0]["model"], "base")
        self.assertEqual(rows[0]["model_name_or_path"], "local-model")
        self.assertEqual(rows[0]["rubrics"], ["mentions evidence", "checks format"])

        pred_map = build_prediction_map(rows, model="base")
        self.assertEqual(pred_map["q1"]["rubrics"], ["mentions evidence", "checks format"])

    def test_load_done_uses_method_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "done.jsonl"
            write_jsonl(path, [{"query": "q1", "method": "base"}, {"query": "q2", "model": "gpt4o"}])
            self.assertEqual(load_done(path), {("q1", "base"), ("q2", "gpt4o")})

    def test_generate_rows_streams_to_sink_instead_of_returning(self) -> None:
        sunk: list[dict] = []
        rows = generate_rows(
            queries=["q1"],
            configs=[
                LLMConfig(
                    name="base",
                    model="local-model",
                    base_url="http://localhost:8000/v1",
                    api_key_env="LOCAL_KEY",
                )
            ],
            client_factory=FakeClient,
            row_sink=sunk.append,
        )
        self.assertEqual(rows, [])
        self.assertEqual(len(sunk), 1)
        self.assertEqual(sunk[0]["method"], "base")
        self.assertEqual(sunk[0]["rubrics"], ["mentions evidence", "checks format"])


if __name__ == "__main__":
    unittest.main()
