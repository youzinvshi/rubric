from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from typing import Generator

import blindspot_rl.verl_reward as verl_reward
from blindspot_rl.reward_bsc import BAD_FORMAT_REWARD


class VerlRewardTest(unittest.TestCase):
    def test_compute_score_accepts_verl_style_inputs(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["mentions sunlight"]',
                ground_truth=["mentions sunlight"],
                extra_info={"prompt": "photosynthesis"},
            )
        self.assertGreater(reward, 0.0)

    def test_compute_score_accepts_dict_ground_truth(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["mentions sunlight"]',
                ground_truth={"gold_rubrics": ["mentions sunlight"]},
                extra_info={"prompt": "photosynthesis"},
            )

        self.assertGreater(reward, 0.0)

    def test_compute_score_accepts_json_string_ground_truth_contract(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["mentions sunlight"]',
                ground_truth='{"gold_rubrics": ["mentions sunlight"]}',
                extra_info={"prompt": "photosynthesis"},
            )

        self.assertGreater(reward, 0.0)

    def test_reward_hook_blocks_missing_gold(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["well formed rubric"]',
                ground_truth=[],
                extra_info={"prompt": "q"},
            )

        self.assertEqual(reward, BAD_FORMAT_REWARD)

    def test_reward_hook_blocks_unparseable_response(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str="",
                ground_truth=["must mention evidence"],
                extra_info={"prompt": "q"},
            )

        self.assertEqual(reward, BAD_FORMAT_REWARD)

    def test_reward_hook_honors_weight_env(self) -> None:
        with reward_env(
            BSC_EMBEDDING_MODEL="token-overlap",
            BSC_W_COV="2.0",
            BSC_W_VALID="0.0",
            BSC_W_RED="0.0",
            BSC_COVERAGE_TAU="0.99",
        ):
            reward = verl_reward.compute_score(
                solution_str='["match gold"]',
                ground_truth=["match gold"],
                extra_info={"prompt": "q"},
            )

        self.assertEqual(reward, 2.0)

    def test_reward_hook_uses_rule_verifier_by_default(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["bad"]',
                ground_truth=["bad"],
                extra_info={"prompt": "q"},
            )

        self.assertEqual(reward, 1.0)

    def test_reward_hook_can_disable_verifier_for_ablation(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap", BSC_VERIFIER="none"):
            reward = verl_reward.compute_score(
                solution_str='["bad"]',
                ground_truth=["bad"],
                extra_info={"prompt": "q"},
            )

        self.assertEqual(reward, 1.5)

    def test_reward_hook_valid_flags_drive_validity_term(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            reward = verl_reward.compute_score(
                solution_str='["match gold", "bad"]',
                ground_truth=["match gold"],
                extra_info={"prompt": "q", "valid_flags": [1, 0]},
            )

        self.assertEqual(reward, 1.25)

    def test_reward_hook_rejects_misaligned_valid_flags(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap"):
            with self.assertRaises(ValueError):
                verl_reward.compute_score(
                    solution_str='["match gold", "bad"]',
                    ground_truth=["match gold"],
                    extra_info={"prompt": "q", "valid_flags": [1]},
                )

    def test_reward_hook_rejects_unsupported_verifier_mode(self) -> None:
        with reward_env(BSC_EMBEDDING_MODEL="token-overlap", BSC_VERIFIER="api"):
            with self.assertRaises(ValueError):
                verl_reward.compute_score(
                    solution_str='["match gold"]',
                    ground_truth=["match gold"],
                    extra_info={"prompt": "q"},
                )


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
