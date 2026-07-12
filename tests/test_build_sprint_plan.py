from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_sprint_plan import build_plan, load_config, to_markdown


class BuildSprintPlanTest(unittest.TestCase):
    def test_load_config_reports_missing_file(self) -> None:
        with self.assertRaises(SystemExit) as context:
            load_config(Path("/tmp/missing_sprint_plan_config.json"))

        self.assertIn("Sprint plan config is missing", str(context.exception))

    def test_load_config_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Sprint plan config is not valid JSON", str(context.exception))

    def test_load_config_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                load_config(path)

        self.assertIn("Sprint plan config must be a JSON object", str(context.exception))

    def test_build_plan_expands_phase_tasks_to_days(self) -> None:
        config = {
            "total_days": 3,
            "claim_discipline": ["global discipline"],
            "phases": [
                {
                    "name": "A",
                    "days": 2,
                    "goal": "phase goal",
                    "evidence_gates": ["gate"],
                    "claim_ladder_levels": ["metric-support"],
                    "paper_claims": ["phase claim"],
                    "claim_discipline": ["phase discipline"],
                    "tasks": [{"goal": "day 1", "commands": ["cmd"], "artifacts": ["out"]}],
                },
                {"name": "B", "days": 1, "goal": "day 3"},
            ],
        }
        rows = build_plan(config)
        self.assertEqual([row["day"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["goal"], "day 1")
        self.assertEqual(rows[1]["goal"], "phase goal")
        self.assertEqual(rows[1]["evidence_gates"], ["gate"])
        self.assertEqual(rows[1]["claim_ladder_levels"], ["metric-support"])
        self.assertEqual(rows[0]["paper_claims"], ["phase claim"])
        self.assertEqual(rows[0]["claim_discipline"], ["phase discipline"])
        self.assertEqual(rows[2]["claim_discipline"], ["global discipline"])

    def test_to_markdown_includes_commands_and_exit_criteria(self) -> None:
        rows = [
            {
                "day": 1,
                "phase": "Data",
                "goal": "Normalize",
                "commands": ["python x.py"],
                "artifacts": ["out.json"],
                "evidence_gates": ["audit ok"],
                "claim_ladder_levels": ["motivation"],
                "paper_claims": ["only claim after audit"],
                "claim_discipline": ["do not promote before audit"],
                "exit_criteria": "done",
            }
        ]
        text = to_markdown(rows, {"title": "Sprint", "total_days": 1, "claim_discipline": ["global gate"]})
        self.assertIn("`python x.py`", text)
        self.assertIn("Paper Claims", text)
        self.assertIn("Claim Ladder", text)
        self.assertIn("motivation", text)
        self.assertIn("only claim after audit", text)
        self.assertIn("Global Claim Discipline", text)
        self.assertIn("global gate", text)
        self.assertIn("Claim Ladder Milestones", text)
        self.assertIn("judge-utility support", text)
        self.assertIn("Without downstream support, report metric-only BSC evidence", text)
        self.assertIn("Claim Discipline By Day", text)
        self.assertIn("Day 1: do not promote before audit", text)
        self.assertIn("Day 1: done", text)

    def test_real_sprint_plan_template_keeps_claims_evidence_gated(self) -> None:
        config = load_config(Path("configs/sprint_plan_20day.template.json"))
        rows = build_plan(config)
        text = to_markdown(rows, config)

        self.assertIn("Use minimal C1/C2 only for Section 2 motivation", text)
        self.assertIn("Claim Ladder Milestones", text)
        self.assertIn("metric-support", text)
        self.assertIn("method-support", text)
        self.assertIn("judge-utility support", text)
        self.assertTrue(any("motivation" in row["claim_ladder_levels"] for row in rows))
        self.assertTrue(any("method-support" in row["claim_ladder_levels"] for row in rows))
        self.assertIn("do not treat them as real-run method-result gates", text)
        self.assertIn("Do not write SFT+GRPO coverage, dimension-level recovery, downstream utility", text)
        self.assertIn("corresponding real C0-C14 evidence rows are safe_to_claim", text)
        self.assertIn("A BSC coverage change alone is metric evidence", text)
        self.assertIn("C2/C3/C14 rows are present in the Evidence Matrix", text)
        self.assertIn("C4/C9/C10 downstream rows are present in the Evidence Matrix", text)
        self.assertIn("claims remain deferred unless the relevant rows are safe_to_claim", text)
        self.assertIn("Generate method evaluation criteria for base/API/SFT/RL methods.", text)
        self.assertNotIn("Generate method rubrics for base/API/SFT/RL methods.", text)
        self.assertNotIn("sft_rl coverage gain over base passes Evidence Matrix", text)
        self.assertNotIn("downstream accuracy does not regress", text)
        self.assertNotIn("guarantee acceptance", text.lower())
        self.assertNotIn("rubric generator", text.lower())

    def test_real_sprint_plan_exports_full_paper_evidence_chain(self) -> None:
        config = load_config(Path("configs/sprint_plan_20day.template.json"))
        rows = build_plan(config)
        export_commands = [
            command
            for row in rows
            for command in row["commands"]
            if "scripts/export_paper_artifacts.py" in command
        ]

        self.assertEqual(len(export_commands), 1)
        export_command = export_commands[0]
        self.assertIn("--main-table-csv outputs/matrix_real/main_table.csv", export_command)
        self.assertIn("--main-table-md outputs/matrix_real/main_table.md", export_command)
        self.assertIn("--downstream-table-csv RewardBench=outputs/matrix_real/main_table.csv", export_command)
        self.assertIn("--downstream-table-csv JudgeBench=outputs/matrix_judgebench/main_table.csv", export_command)
        self.assertIn(
            "--downstream-table-csv RewardBench-2=outputs/matrix_rewardbench2/main_table.csv",
            export_command,
        )
        self.assertIn("--ablation-csv outputs/bsc_ablation/ablation_summary.csv", export_command)
        self.assertIn(
            "--teacher-union-csv outputs/teacher_union_ablation/teacher_union_ablation.csv",
            export_command,
        )
        self.assertIn(
            "--verifier-filter-csv outputs/verifier_filter_ablation/verifier_filter_ablation.csv",
            export_command,
        )
        self.assertIn(
            "--transition-summary-json outputs/matrix_real/dimension_transition/base_to_sft_only/transition_summary.json",
            export_command,
        )
        self.assertIn(
            "--transition-summary-json outputs/matrix_real/dimension_transition/base_to_sft_rl/transition_summary.json",
            export_command,
        )
        self.assertIn("--semantic-space-dir outputs/matrix_real/semantic_space", export_command)
        self.assertIn("--evidence-json outputs/evidence_real/evidence_matrix.json", export_command)
        self.assertIn("--evidence-csv outputs/evidence_real/evidence_matrix.csv", export_command)
        self.assertIn("--evidence-md outputs/evidence_real/evidence_matrix.md", export_command)

        export_rows = [row for row in rows if any("scripts/export_paper_artifacts.py" in command for command in row["commands"])]
        self.assertEqual(len(export_rows), 1)
        artifacts = export_rows[0]["artifacts"]
        self.assertIn("outputs/paper_artifacts/downstream_utility_table.tex", artifacts)
        self.assertIn("outputs/paper_artifacts/verifier_filter_ablation_table.tex", artifacts)
        self.assertIn("outputs/paper_artifacts/dimension_transition_table.tex", artifacts)
        self.assertIn("outputs/paper_artifacts/semantic_space_summary.json", artifacts)
        self.assertIn("outputs/paper_artifacts/semantic_space_points.csv", artifacts)


if __name__ == "__main__":
    unittest.main()
