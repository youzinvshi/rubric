"""Downstream chosen-vs-rejected evaluation with generated rubrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from blindspot_rl.llm_api import OpenAICompatibleClient, parse_score
from blindspot_rl.reward_bsc import parse_rubrics


class RubricScorer(Protocol):
    """Scores one answer against one rubric criterion."""

    def score(self, query: str, answer: str, rubric: str) -> float:
        """Return a numeric score. Larger means the answer satisfies the rubric better."""


@dataclass(frozen=True)
class PreferenceResult:
    chosen_score: float
    rejected_score: float
    margin: float
    correct: bool
    tie: bool
    n_rubrics: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "margin": self.margin,
            "correct": self.correct,
            "tie": self.tie,
            "n_rubrics": self.n_rubrics,
        }


@dataclass(frozen=True)
class MultiCandidateResult:
    label: int
    prediction: int
    label_score: float
    top_score: float
    margin: float
    correct: bool
    tie: bool
    n_candidates: int
    n_rubrics: int
    scores: tuple[float, ...]

    def as_dict(self) -> dict[str, float | int | bool | list[float]]:
        return {
            "label": self.label,
            "prediction": self.prediction,
            "label_score": self.label_score,
            "top_score": self.top_score,
            "margin": self.margin,
            "correct": self.correct,
            "tie": self.tie,
            "n_candidates": self.n_candidates,
            "n_rubrics": self.n_rubrics,
            "scores": list(self.scores),
        }


class KeywordRubricScorer:
    """Deterministic scorer for smoke tests.

    Real paper experiments should replace this with an LLM/API scorer that
    judges answer-rubric satisfaction. This scorer simply measures weighted
    token overlap between a rubric and an answer.
    """

    def score(self, query: str, answer: str, rubric: str) -> float:
        del query
        rubric_tokens = set(_tokens(rubric))
        if not rubric_tokens:
            return 0.0
        answer_tokens = set(_tokens(answer))
        return len(rubric_tokens & answer_tokens) / len(rubric_tokens)


class CallableRubricScorer:
    """Adapter for API functions with signature fn(query, answer, rubric)."""

    def __init__(self, fn: Callable[[str, str, str], float]):
        self.fn = fn

    def score(self, query: str, answer: str, rubric: str) -> float:
        return float(self.fn(query, answer, rubric))


class APIRubricScorer:
    """LLM/API scorer for paper downstream validation."""

    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def score(self, query: str, answer: str, rubric: str) -> float:
        content = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You judge whether an answer satisfies one rubric item. "
                        "Return JSON only: {\"score\": number, \"reason\": \"...\"}. "
                        "Use score=1 if fully satisfied, 0.5 if partially satisfied, "
                        "and 0 if not satisfied or unverifiable."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query:\n{query}\n\n"
                        f"Answer:\n{answer}\n\n"
                        f"Rubric:\n{rubric}"
                    ),
                },
            ]
        )
        return parse_score(content)


def score_answer(
    query: str,
    answer: str,
    rubrics: Sequence[str] | str | Any,
    scorer: RubricScorer,
) -> float:
    """Average rubric satisfaction score for one answer."""

    parsed = parse_rubrics(rubrics, dedupe=True)
    if not parsed:
        return 0.0
    scores = [float(scorer.score(query, answer, rubric)) for rubric in parsed]
    return sum(scores) / len(scores)


def evaluate_preference(
    query: str,
    chosen: str,
    rejected: str,
    rubrics: Sequence[str] | str | Any,
    scorer: RubricScorer,
    tie_epsilon: float = 1e-8,
) -> PreferenceResult:
    """Return whether rubric-based scores prefer chosen over rejected."""

    parsed = parse_rubrics(rubrics, dedupe=True)
    chosen_score = score_answer(query, chosen, parsed, scorer)
    rejected_score = score_answer(query, rejected, parsed, scorer)
    margin = chosen_score - rejected_score
    tie = abs(margin) <= tie_epsilon
    return PreferenceResult(
        chosen_score=chosen_score,
        rejected_score=rejected_score,
        margin=margin,
        correct=margin > tie_epsilon,
        tie=tie,
        n_rubrics=len(parsed),
    )


def evaluate_multicandidate(
    query: str,
    candidates: Sequence[str],
    label: int,
    rubrics: Sequence[str] | str | Any,
    scorer: RubricScorer,
    tie_epsilon: float = 1e-8,
) -> MultiCandidateResult:
    """Return whether rubric-based scores select the gold candidate."""

    parsed = parse_rubrics(rubrics, dedupe=True)
    if not candidates:
        raise ValueError("multicandidate evaluation requires at least one candidate")
    if label < 0 or label >= len(candidates):
        raise ValueError(f"label index {label} is outside {len(candidates)} candidates")

    scores = tuple(score_answer(query, str(answer), parsed, scorer) for answer in candidates)
    top_score = max(scores)
    top_indices = [idx for idx, score in enumerate(scores) if abs(score - top_score) <= tie_epsilon]
    prediction = top_indices[0]
    label_score = scores[label]
    best_non_label = max((score for idx, score in enumerate(scores) if idx != label), default=label_score)
    margin = label_score - best_non_label
    tie = len(top_indices) > 1
    return MultiCandidateResult(
        label=label,
        prediction=prediction,
        label_score=label_score,
        top_score=top_score,
        margin=margin,
        correct=prediction == label and not tie,
        tie=tie,
        n_candidates=len(candidates),
        n_rubrics=len(parsed),
        scores=scores,
    )


def aggregate_preference_results(results: Sequence[PreferenceResult]) -> dict[str, float | int]:
    """Aggregate downstream preference metrics."""

    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "tie_rate": 0.0, "mean_margin": 0.0}
    correct = sum(1 for item in results if item.correct)
    ties = sum(1 for item in results if item.tie)
    mean_margin = sum(item.margin for item in results) / n
    return {
        "n": n,
        "accuracy": correct / n,
        "tie_rate": ties / n,
        "mean_margin": mean_margin,
    }


def aggregate_multicandidate_results(results: Sequence[MultiCandidateResult]) -> dict[str, float | int]:
    """Aggregate multi-candidate downstream metrics."""

    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "tie_rate": 0.0, "mean_margin": 0.0, "mean_candidates": 0.0}
    correct = sum(1 for item in results if item.correct)
    ties = sum(1 for item in results if item.tie)
    mean_margin = sum(item.margin for item in results) / n
    mean_candidates = sum(item.n_candidates for item in results) / n
    return {
        "n": n,
        "accuracy": correct / n,
        "tie_rate": ties / n,
        "mean_margin": mean_margin,
        "mean_candidates": mean_candidates,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower())
