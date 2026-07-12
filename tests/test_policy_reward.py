from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from blindspot_rl import policy_reward


class PolicyRewardTest(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("BSC_POLICY_RUBRIC_FILE", None)
        os.environ.pop("BSC_POLICY_MISSING_REWARD", None)
        policy_reward._RUBRIC_CACHE = None
        policy_reward._RUBRIC_CACHE_PATH = None

    def test_compute_score_uses_extra_info_rubrics(self) -> None:
        reward = policy_reward.compute_score(
            solution_str="The answer mentions sunlight and oxygen.",
            extra_info={
                "prompt": "Explain photosynthesis.",
                "rubrics": ["mentions sunlight", "mentions oxygen"],
            },
        )

        self.assertGreater(reward, 0.0)

    def test_compute_score_loads_query_rubrics_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rubrics.jsonl"
            path.write_text(
                '{"query":"Explain photosynthesis.","rubrics":["mentions sunlight","mentions oxygen"]}\n',
                encoding="utf-8",
            )
            os.environ["BSC_POLICY_RUBRIC_FILE"] = str(path)

            reward = policy_reward.compute_score(
                solution_str="Uses sunlight to produce oxygen.",
                extra_info={"prompt": "Explain photosynthesis."},
            )

        self.assertGreater(reward, 0.0)

    def test_compute_score_returns_missing_reward_without_rubrics(self) -> None:
        os.environ["BSC_POLICY_MISSING_REWARD"] = "-0.25"

        reward = policy_reward.compute_score(
            solution_str="answer",
            extra_info={"prompt": "query"},
        )

        self.assertEqual(reward, -0.25)


if __name__ == "__main__":
    unittest.main()
