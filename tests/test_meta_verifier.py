from __future__ import annotations

import unittest

from blindspot_rl.meta_verifier import (
    APIMetaVerifier,
    RuleMetaVerifier,
    build_api_verifier_prompt,
    filter_proxy_rubrics,
    parse_api_decision,
)


class MetaVerifierTest(unittest.TestCase):
    def test_rule_verifier_fails_closed_on_generic_short_rubrics(self) -> None:
        verifier = RuleMetaVerifier()

        self.assertEqual(verifier.judge("good", prompt="q"), 0)
        self.assertEqual(verifier.judge("must cite evidence", prompt="q"), 1)

    def test_parse_api_decision_requires_all_quality_flags(self) -> None:
        valid = parse_api_decision(
            "Checks whether the answer cites evidence.",
            '{"valid": true, "atomic": true, "decidable": true, "relevant": true, '
            '"non_hallucinated": true, "reason": "ok"}',
        )
        invalid = parse_api_decision(
            "Checks whether the answer cites evidence.",
            '{"valid": true, "atomic": true, "decidable": true, "relevant": false, '
            '"non_hallucinated": true, "reason": "not grounded"}',
        )

        self.assertTrue(valid.valid)
        self.assertFalse(invalid.valid)

    def test_filter_marks_exact_duplicates_and_preserves_flag_alignment(self) -> None:
        result = filter_proxy_rubrics(
            query="answer the question",
            candidates=[
                "Checks whether the answer cites evidence.",
                "Checks whether the answer cites evidence.",
                "good",
            ],
        )

        self.assertEqual(result.rubrics_before_filter, [
            "Checks whether the answer cites evidence.",
            "Checks whether the answer cites evidence.",
            "good",
        ])
        self.assertEqual(result.verified_rubrics, ["Checks whether the answer cites evidence."])
        self.assertEqual(result.valid_flags, [1, 0, 0])
        self.assertEqual(result.verifier_decisions[1].reason, "exact_duplicate")

    def test_filter_enforces_max_rubrics_after_validation(self) -> None:
        result = filter_proxy_rubrics(
            query="answer the question",
            candidates=[
                "Checks whether the answer cites evidence.",
                "Checks whether the answer follows the requested format.",
                "Checks whether the answer includes the final result.",
            ],
            max_rubrics=2,
        )

        self.assertEqual(len(result.verified_rubrics), 2)
        self.assertEqual(result.valid_flags, [1, 1, 0])
        self.assertEqual(result.verifier_decisions[2].reason, "max_rubrics_exceeded")

    def test_domain_prompt_relaxes_health_and_search_dimensions(self) -> None:
        health_system, health_user = build_api_verifier_prompt(
            query="Can this medicine interact with my blood pressure drug?",
            rubric="Checks whether the answer recommends professional care for risky interactions.",
            data_source="healthbench",
        )
        search_system, _ = build_api_verifier_prompt(
            query="who sings love will keep us alive",
            rubric="Checks whether the result is supported by authoritative evidence.",
            data_source="beir_nq",
        )

        self.assertIn("meta-verifier for evaluation-criteria training data", health_system)
        self.assertNotIn("rubric-generation training data", health_system)
        self.assertNotIn("Candidate rubric", health_user)
        self.assertIn("Candidate criterion", health_user)
        self.assertNotIn("A valid rubric", health_system)
        self.assertIn("A valid criterion", health_system)
        self.assertIn("red-flag escalation", health_system)
        self.assertIn("professional-care recommendation", health_system)
        self.assertIn("source authority", search_system)
        self.assertIn("evidence support", search_system)

    def test_api_verifier_receives_data_source_policy(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.messages = None

            def chat(self, messages):  # noqa: ANN001
                self.messages = messages
                return (
                    '{"valid": true, "atomic": true, "decidable": true, '
                    '"relevant": true, "non_hallucinated": true, "reason": "ok"}'
                )

        client = FakeClient()
        verifier = APIMetaVerifier(client)  # type: ignore[arg-type]
        decision = verifier.verify(
            "Checks whether the answer includes urgent-care guidance for red flags.",
            query="I have mild ear pain; should I wait?",
            data_source="healthbench",
        )

        self.assertTrue(decision.valid)
        self.assertIn("medical/health evaluation task", client.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
