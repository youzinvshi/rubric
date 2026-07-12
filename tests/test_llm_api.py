from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from blindspot_rl.llm_api import (
    APIMetaVerifier,
    LLMConfig,
    OpenAICompatibleClient,
    parse_score,
    parse_valid_flag,
)


def _capture_responses_payload(config: LLMConfig) -> dict:
    """Invoke the /responses branch and capture the JSON payload it posts."""

    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        del timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        body = json.dumps(
            {"output": [{"content": [{"type": "output_text", "text": "OK"}]}]}
        ).encode("utf-8")
        return io.BytesIO(body)

    client = OpenAICompatibleClient(config)
    with mock.patch.dict("os.environ", {config.api_key_env: "test-key"}), mock.patch(
        "blindspot_rl.llm_api.urllib.request.urlopen", fake_urlopen
    ):
        out = client.chat([{"role": "user", "content": "hi"}])
    assert out == "OK"
    return captured["payload"]


class LLMAPITest(unittest.TestCase):
    def test_api_meta_verifier_uses_criteria_prompt_framing(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.messages = None

            def chat(self, messages):  # noqa: ANN001
                self.messages = messages
                return '{"valid": true, "reason": "ok"}'

        client = FakeClient()
        verifier = APIMetaVerifier(client)  # type: ignore[arg-type]

        self.assertEqual(verifier.judge("Checks whether the answer cites evidence.", prompt="q"), 1)
        combined = "\n".join(message["content"] for message in client.messages)
        self.assertIn("evaluation-criteria verifier", combined)
        self.assertIn("A valid criterion", combined)
        self.assertIn("Candidate criterion", combined)
        self.assertNotIn("strict rubric verifier", combined)
        self.assertNotIn("A valid rubric", combined)
        self.assertNotIn("\n\nRubric:\n", combined)

    def test_parse_valid_flag_json_true(self) -> None:
        self.assertTrue(parse_valid_flag('{"valid": true, "reason": "atomic"}'))

    def test_parse_valid_flag_json_false_inside_text(self) -> None:
        self.assertFalse(parse_valid_flag('Result:\n{"valid": false, "reason": "generic"}'))

    def test_parse_valid_flag_json_string_false_is_false(self) -> None:
        self.assertFalse(parse_valid_flag('{"valid": "false", "reason": "generic"}'))

    def test_parse_valid_flag_json_numeric_flags(self) -> None:
        self.assertTrue(parse_valid_flag('{"valid": 1}'))
        self.assertFalse(parse_valid_flag('{"valid": 0}'))

    def test_parse_valid_flag_unknown_string_fails_closed(self) -> None:
        self.assertFalse(parse_valid_flag('{"valid": "maybe"}'))

    def test_parse_valid_flag_plain_yes(self) -> None:
        self.assertTrue(parse_valid_flag("yes"))

    def test_parse_score_json(self) -> None:
        self.assertEqual(parse_score('{"score": 0.5, "reason": "partial"}'), 0.5)

    def test_parse_score_clamps_plain_number(self) -> None:
        self.assertEqual(parse_score("2"), 1.0)

    def test_responses_omits_reasoning_by_default(self) -> None:
        config = LLMConfig(
            name="gpt-4o",
            model="gpt-4o-2024-11-20",
            base_url="https://modelhub.example.com/v1/responses",
            api_key_env="GPT_AK_TEST",
            max_tokens=300,
        )
        payload = _capture_responses_payload(config)
        self.assertNotIn("reasoning", payload)
        self.assertEqual(payload["max_output_tokens"], 300)

    def test_responses_includes_reasoning_when_configured(self) -> None:
        config = LLMConfig(
            name="gpt-5.4",
            model="gpt-5.4-2026-03-05",
            base_url="https://modelhub.example.com/v1/responses",
            api_key_env="GPT_AK_TEST",
            max_tokens=600,
            reasoning_effort="high",
        )
        payload = _capture_responses_payload(config)
        self.assertEqual(payload["reasoning"], {"effort": "high", "summary": "detailed"})

    def test_responses_raises_on_truncated_incomplete_response(self) -> None:
        config = LLMConfig(
            name="gpt-5.4",
            model="gpt-5.4-2026-03-05",
            base_url="https://modelhub.example.com/v1/responses",
            api_key_env="GPT_AK_TEST",
            max_tokens=1200,
            reasoning_effort="high",
        )

        def fake_urlopen(req, timeout=None):  # noqa: ANN001
            del req, timeout
            body = json.dumps(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "max_output_tokens": 1200,
                    "output": [{"type": "reasoning", "content": []}],
                }
            ).encode("utf-8")
            return io.BytesIO(body)

        client = OpenAICompatibleClient(config)
        with mock.patch.dict("os.environ", {config.api_key_env: "test-key"}), mock.patch(
            "blindspot_rl.llm_api.urllib.request.urlopen", fake_urlopen
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("max_output_tokens", str(ctx.exception))
        self.assertIn("no output_text", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
