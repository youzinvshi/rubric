from __future__ import annotations

import unittest

from scripts.profile_dataset_schema import profile_records, suggest_key, walk_paths


class ProfileDatasetSchemaTest(unittest.TestCase):
    def test_walk_paths_includes_nested_list_indices(self) -> None:
        paths = dict(walk_paths({"messages": [{"content": "q"}]}))
        self.assertEqual(paths["messages.0.content"], "q")

    def test_profile_records_suggests_nested_mapping(self) -> None:
        profile = profile_records(
            [
                {
                    "sample": {"query": "q1"},
                    "answers": {
                        "chosen": "good",
                        "rejected": "bad",
                    },
                }
            ]
        )
        self.assertEqual(profile["suggested_mappings"]["query_key"], "sample.query")
        self.assertEqual(profile["suggested_mappings"]["chosen_key"], "answers.chosen")
        self.assertEqual(profile["suggested_mappings"]["rejected_key"], "answers.rejected")

    def test_suggest_key_prefers_exact_match(self) -> None:
        self.assertEqual(suggest_key(["prompt", "meta.prompt"], ("prompt",)), "prompt")


if __name__ == "__main__":
    unittest.main()
