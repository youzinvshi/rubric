from __future__ import annotations

import unittest

from scripts.run_relaxed_verifier_record_audit import audit_record


class RelaxedVerifierRecordAuditTest(unittest.TestCase):
    def test_audit_prompt_uses_criteria_framing(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.messages = None

            def chat(self, messages):  # noqa: ANN001
                self.messages = messages
                return (
                    '{"rubrics":[{"idx":0,"valid":true,"atomic":true,'
                    '"decidable":true,"relevant":true,"non_hallucinated":true,'
                    '"reason":"ok"}]}'
                )

        client = FakeClient()
        decisions = audit_record(
            client=client,  # type: ignore[arg-type]
            query="q",
            rubrics=["Checks whether the answer cites evidence."],
            data_source="rewardbench",
            max_rubrics=12,
        )

        self.assertEqual(len(decisions), 1)
        combined = "\n".join(message["content"] for message in client.messages)
        self.assertIn("candidate criterion", combined.lower())
        self.assertIn("A valid criterion", combined)
        self.assertIn("Candidate criteria", combined)
        self.assertNotIn("candidate rubric", combined.lower())
        self.assertNotIn("A valid rubric", combined)
        self.assertNotIn("Candidate rubrics", combined)


if __name__ == "__main__":
    unittest.main()
