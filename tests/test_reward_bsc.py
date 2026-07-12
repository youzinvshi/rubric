from __future__ import annotations

import math
import os
import unittest
from contextlib import contextmanager
from typing import Generator

import numpy as np

import blindspot_rl.verl_reward as canonical_verl_reward
from blindspot_rl import reward_bsc
from blindspot_rl.reward_bsc import (
    BAD_FORMAT_REWARD,
    TokenOverlapEmbedder,
    category_balanced_coverage_reward,
    compute_metrics,
    compute_category_balanced_reward,
    compute_reward,
    coverage_reward,
    parse_rubrics,
    redundancy_penalty,
    semantic_dedupe,
    validity_reward,
    verl_reward_fn,
)


class RewardBSCTest(unittest.TestCase):
    def test_module_docstring_uses_policy_framing(self) -> None:
        self.assertIn("evaluation-criteria policy RLVR training", reward_bsc.__doc__ or "")
        self.assertNotIn("rubric-generation policy RLVR training", reward_bsc.__doc__ or "")
        self.assertNotIn("Rubric-Generator RLVR training", reward_bsc.__doc__ or "")

    def test_parse_json_list_and_dict_items(self) -> None:
        text = """
        ```json
        [
          {"criterion": "Mentions the safety tradeoff"},
          {"description": "Checks the final answer"},
          "Uses evidence from the prompt"
        ]
        ```
        """
        self.assertEqual(
            parse_rubrics(text),
            [
                "Mentions the safety tradeoff",
                "Checks the final answer",
                "Uses evidence from the prompt",
            ],
        )

    def test_parse_gold_rubric_mapping_contract_fields(self) -> None:
        self.assertEqual(
            parse_rubrics({"gold_rubrics": ["mentions evidence", "mentions evidence", "checks answer"]}, dedupe=True),
            ["mentions evidence", "checks answer"],
        )
        self.assertEqual(
            parse_rubrics('{"gold": ["mentions evidence", {"criterion": "checks answer"}]}'),
            ["mentions evidence", "checks answer"],
        )

    def test_parse_numpy_array_from_parquet_gold_column(self) -> None:
        self.assertEqual(
            parse_rubrics(np.array(["mentions evidence", "checks answer"], dtype=object)),
            ["mentions evidence", "checks answer"],
        )

    def test_parse_markdown_bullets_preserves_duplicates_for_red(self) -> None:
        text = """
        - Shows the equation
        - Shows the equation
        1. Computes the answer
        """
        self.assertEqual(
            parse_rubrics(text),
            ["Shows the equation", "Shows the equation", "Computes the answer"],
        )

    def test_parse_plain_multiline_rubrics_with_braces_falls_back_to_lines(self) -> None:
        text = """
        Does the response define a function named `empty_list`?
        Is every element in the returned list an empty dictionary (`{})?
        Does the response include comments?
        """

        self.assertEqual(
            parse_rubrics(text),
            [
                "Does the response define a function named `empty_list`?",
                "Is every element in the returned list an empty dictionary (`{})?",
                "Does the response include comments?",
            ],
        )


    def test_coverage_reward_counts_gold_hits(self) -> None:
        embedder = TokenOverlapEmbedder()
        gold = ["plants use sunlight", "water becomes glucose", "oxygen released"]
        gen = ["plants use sunlight", "water becomes sugar"]
        cov = coverage_reward(gen, gold, tau=0.5, embedder=embedder)
        self.assertTrue(math.isclose(cov, 2 / 3, rel_tol=1e-6))

    def test_category_balanced_coverage_macro_averages_present_categories(self) -> None:
        embedder = TokenOverlapEmbedder()
        gold = ["fact one", "fact two", "safety policy"]
        categories = ["factuality", "factuality", "safety"]
        gen = ["fact one"]

        micro = coverage_reward(gen, gold, tau=0.99, embedder=embedder)
        balanced = category_balanced_coverage_reward(gen, gold, categories, tau=0.99, embedder=embedder)

        self.assertTrue(math.isclose(micro, 1 / 3, rel_tol=1e-6))
        self.assertTrue(math.isclose(balanced, 0.25, rel_tol=1e-6))

    def test_category_balanced_coverage_requires_aligned_categories(self) -> None:
        with self.assertRaises(ValueError):
            category_balanced_coverage_reward(
                ["fact one"],
                ["fact one", "safety policy"],
                ["factuality"],
                embedder=TokenOverlapEmbedder(),
            )


    def test_redundancy_penalty_counts_duplicate_pairs(self) -> None:
        embedder = TokenOverlapEmbedder()
        gen = ["shows equation", "shows equation", "final answer"]
        red = redundancy_penalty(gen, tau=0.99, embedder=embedder)
        self.assertTrue(math.isclose(red, 1 / 3, rel_tol=1e-6))

    def test_semantic_dedupe_preserves_first_unique_dimensions(self) -> None:
        embedder = TokenOverlapEmbedder()
        rubrics = ["shows equation", "shows equation", "final answer"]
        deduped = semantic_dedupe(rubrics, tau=0.99, embedder=embedder)
        self.assertEqual(deduped, ["shows equation", "final answer"])


    def test_validity_reward_accepts_callable_verifier(self) -> None:
        def verifier(rubric: str, _prompt: str | None = None) -> bool:
            return "bad" not in rubric

        self.assertEqual(validity_reward(["good criterion", "bad criterion"], verifier=verifier), 0.5)


    def test_compute_reward_bad_format_is_strong_negative(self) -> None:
        reward = compute_reward(
            prompt="q",
            response="",
            gold_rubrics=["must mention evidence"],
            embedder=TokenOverlapEmbedder(),
        )
        self.assertEqual(reward, BAD_FORMAT_REWARD)


    def test_compute_metrics_reports_blind_and_hallucination(self) -> None:
        def verifier(rubric: str, _prompt: str | None = None) -> bool:
            return rubric != "invalid"

        metrics = compute_metrics(
            response=["match gold", "invalid"],
            gold_rubrics=["match gold"],
            verifier=verifier,
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )
        self.assertEqual(metrics.coverage, 1.0)
        self.assertEqual(metrics.blind, 0.0)
        self.assertEqual(metrics.validity, 0.5)
        self.assertEqual(metrics.hallucination, 0.5)

    def test_compute_category_balanced_reward_uses_four_term_formula(self) -> None:
        reward = compute_category_balanced_reward(
            prompt="q",
            response=["fact one"],
            gold_rubrics=["fact one", "safety policy"],
            gold_categories=["factuality", "safety"],
            weights=(0.7, 0.3, 0.5, 0.5),
            embedder=TokenOverlapEmbedder(),
            coverage_tau=0.99,
        )

        self.assertTrue(math.isclose(reward, 1.0, rel_tol=1e-6))

    def test_legacy_verl_reward_fn_uses_canonical_verifier_default(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward_fn(
                {"prompt": "q", "gold_rubrics": ["bad"]},
                '["bad"]',
            )

        self.assertEqual(reward, 1.0)


@contextmanager
def reward_env(**values: str) -> Generator[None, None, None]:
    old = {key: os.environ.get(key) for key in values}
    old_embedder = canonical_verl_reward._EMBEDDER
    old_verifier = canonical_verl_reward._VERIFIER
    try:
        for key, value in values.items():
            os.environ[key] = value
        canonical_verl_reward._EMBEDDER = None
        canonical_verl_reward._VERIFIER = None
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        canonical_verl_reward._EMBEDDER = old_embedder
        canonical_verl_reward._VERIFIER = old_verifier


if __name__ == "__main__":
    unittest.main()
