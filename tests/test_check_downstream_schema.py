from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.check_downstream_schema import build_schema_report


class CheckDownstreamSchemaTest(unittest.TestCase):
    def test_report_selects_multicandidate_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(
                '{"prompt":"q1","responses":["bad","good"],"correct_index":1}\n',
                encoding="utf-8",
            )
            report = build_schema_report(args_for(path))

            self.assertTrue(report["ok"])
            self.assertEqual(report["selected_target"], "multicandidate")
            self.assertEqual(report["targets"][1]["normalized_records"], 1)

    def test_report_blocks_when_no_target_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text('{"unmapped":"x"}\n', encoding="utf-8")
            report = build_schema_report(args_for(path))

            self.assertFalse(report["ok"])
            self.assertTrue(report["blockers"])


def args_for(path: Path) -> Namespace:
    return Namespace(
        input=path,
        target=["preference", "multicandidate"],
        data_source="toy",
        query_key=None,
        chosen_key=None,
        rejected_key=None,
        candidates_key=None,
        label_key=None,
        min_records=1,
        limit=None,
        output_json=Path("out.json"),
        output_md=None,
        strict=False,
    )


if __name__ == "__main__":
    unittest.main()
