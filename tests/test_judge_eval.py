from __future__ import annotations

import unittest

from blindspot_rl.judge_eval import (
    APIRubricScorer,
    KeywordRubricScorer,
    aggregate_multicandidate_results,
    aggregate_preference_results,
    evaluate_multicandidate,
    evaluate_preference,
    score_answer,
)


class JudgeEvalTest(unittest.TestCase):
    def test_score_answer_averages_rubric_scores(self) -> None:
        scorer = KeywordRubricScorer()
        score = score_answer(
            query="q",
            answer="The answer mentions sunlight and oxygen.",
            rubrics=["mentions sunlight", "mentions oxygen"],
            scorer=scorer,
        )
        self.assertGreater(score, 0.0)

    def test_evaluate_preference_marks_chosen_correct(self) -> None:
        scorer = KeywordRubricScorer()
        result = evaluate_preference(
            query="q",
            chosen="Uses sunlight and releases oxygen.",
            rejected="Talks about soil only.",
            rubrics=["uses sunlight", "releases oxygen"],
            scorer=scorer,
        )
        self.assertTrue(result.correct)
        self.assertFalse(result.tie)
        self.assertEqual(result.n_rubrics, 2)

    def test_aggregate_preference_results(self) -> None:
        scorer = KeywordRubricScorer()
        good = evaluate_preference("q", "sunlight", "soil", ["sunlight"], scorer)
        tied = evaluate_preference("q", "soil", "soil", ["sunlight"], scorer)
        summary = aggregate_preference_results([good, tied])
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["tie_rate"], 0.5)

    def test_evaluate_multicandidate_selects_gold_candidate(self) -> None:
        scorer = KeywordRubricScorer()
        result = evaluate_multicandidate(
            query="q",
            candidates=["talks about soil", "uses sunlight and oxygen", "mentions rocks"],
            label=1,
            rubrics=["uses sunlight", "mentions oxygen"],
            scorer=scorer,
        )

        self.assertTrue(result.correct)
        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.n_candidates, 3)

    def test_aggregate_multicandidate_results(self) -> None:
        scorer = KeywordRubricScorer()
        good = evaluate_multicandidate("q", ["soil", "sunlight"], 1, ["sunlight"], scorer)
        tied = evaluate_multicandidate("q", ["soil", "soil"], 1, ["sunlight"], scorer)
        summary = aggregate_multicandidate_results([good, tied])

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["tie_rate"], 0.5)
        self.assertEqual(summary["mean_candidates"], 2)

    def test_api_rubric_scorer_parses_client_score(self) -> None:
        class FakeClient:
            def chat(self, messages):  # type: ignore[no-untyped-def]
                self.messages = messages
                return '{"score": 1.0, "reason": "satisfied"}'

        scorer = APIRubricScorer(FakeClient())  # type: ignore[arg-type]
        self.assertEqual(scorer.score("q", "answer", "rubric"), 1.0)


if __name__ == "__main__":
    unittest.main()
