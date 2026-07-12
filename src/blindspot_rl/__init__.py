"""BlindSpot-RL toolkit."""

from blindspot_rl.reward_bsc import (
    BSCMetrics,
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    category_balanced_coverage_reward,
    compute_metrics,
    compute_category_balanced_reward,
    compute_reward,
    coverage_reward,
    hallucination_rate,
    parse_rubrics,
    redundancy_penalty,
    semantic_dedupe,
    validity_reward,
    verl_reward_fn,
)
from blindspot_rl.judge_eval import (
    APIRubricScorer,
    CallableRubricScorer,
    KeywordRubricScorer,
    PreferenceResult,
    aggregate_preference_results,
    evaluate_preference,
    score_answer,
)
from blindspot_rl.verl_reward import compute_score as verl_compute_score

__all__ = [
    "CallableRubricScorer",
    "APIRubricScorer",
    "BSCMetrics",
    "KeywordRubricScorer",
    "PreferenceResult",
    "SentenceTransformerEmbedder",
    "TokenOverlapEmbedder",
    "aggregate_preference_results",
    "category_balanced_coverage_reward",
    "compute_category_balanced_reward",
    "compute_metrics",
    "compute_reward",
    "coverage_reward",
    "evaluate_preference",
    "hallucination_rate",
    "parse_rubrics",
    "redundancy_penalty",
    "score_answer",
    "semantic_dedupe",
    "validity_reward",
    "verl_reward_fn",
    "verl_compute_score",
]
