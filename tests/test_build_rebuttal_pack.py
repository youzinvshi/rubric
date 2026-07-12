from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile

from scripts.build_rebuttal_pack import (
    DEFAULT_CONCERNS,
    build_rebuttal_manifest,
    build_rebuttal_pack,
    claim_ladder_from_evidence_rows,
    file_sha256,
    json_sha256,
    match_claims,
    to_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


class BuildRebuttalPackTest(unittest.TestCase):
    def test_match_claims_uses_section_and_keywords(self) -> None:
        rows = [
            {"paper_section": "Main Results", "claim": "Coverage improves without redundancy", "status": "safe_to_claim"},
            {"paper_section": "Downstream", "claim": "Accuracy improves", "status": "safe_to_claim"},
        ]
        matches = match_claims(
            rows,
            {"claim_sections": ["Main Results"], "keywords": ["redundancy"]},
        )
        self.assertEqual(len(matches), 1)
        self.assertIn("redundancy", matches[0]["claim"])

    def test_build_rebuttal_pack_marks_answer_ready(self) -> None:
        rows = [
            {
                "claim_id": "C1",
                "paper_section": "Motivation",
                "claim": "Single-model blind spots are measurable.",
                "status": "safe_to_claim",
                "evidence": "pass",
            }
        ]
        pack = build_rebuttal_pack(
            rows,
            [
                {
                    "id": "R1",
                    "topic": "Motivation",
                    "question": "Why?",
                    "claim_sections": ["Motivation"],
                }
            ],
            {"ok": True},
        )
        self.assertEqual(pack[0]["defense_status"], "answer_ready")
        self.assertTrue(pack[0]["readiness_ok"])

    def test_build_rebuttal_pack_downgrades_safe_claim_when_readiness_is_false(self) -> None:
        rows = [
            {
                "claim_id": "C1",
                "paper_section": "Motivation",
                "claim": "Single-model blind spots are measurable.",
                "status": "safe_to_claim",
                "evidence": "pass",
            }
        ]
        pack = build_rebuttal_pack(
            rows,
            [
                {
                    "id": "R1",
                    "topic": "Motivation",
                    "question": "Why?",
                    "claim_sections": ["Motivation"],
                }
            ],
            {"ok": False},
        )

        self.assertEqual(pack[0]["defense_status"], "needs_readiness")
        self.assertFalse(pack[0]["readiness_ok"])
        self.assertIn("submission readiness is not ok", pack[0]["recommended_position"])
        self.assertIn("keep this as a draft response", pack[0]["recommended_position"])

    def test_build_rebuttal_pack_marks_contradicted_as_cannot_claim(self) -> None:
        rows = [
            {
                "claim_id": "C4",
                "paper_section": "Downstream",
                "claim": "Downstream improves.",
                "status": "contradicted",
                "evidence": "fail",
            }
        ]
        pack = build_rebuttal_pack(
            rows,
            [{"id": "R4", "topic": "Downstream", "question": "Useful?", "claim_sections": ["Downstream"]}],
        )
        self.assertEqual(pack[0]["defense_status"], "cannot_claim")

    def test_needs_evidence_uses_concern_specific_fallback(self) -> None:
        rows = [
            {
                "claim_id": "C0",
                "paper_section": "Data Hygiene",
                "claim": "RubricBench test_main is query-disjoint from training.",
                "status": "missing_evidence",
                "evidence": "missing SHA-bound report",
            }
        ]
        pack = build_rebuttal_pack(
            rows,
            [
                {
                    "id": "R8",
                    "topic": "Data Contamination",
                    "question": "Contaminated?",
                    "claim_sections": ["Data Hygiene"],
                    "fallback": "Require C0 safe_to_claim with SHA-bound training/evaluation provenance.",
                }
            ],
        )

        self.assertEqual(pack[0]["defense_status"], "needs_evidence")
        self.assertEqual(
            pack[0]["recommended_position"],
            "Require C0 safe_to_claim with SHA-bound training/evaluation provenance.",
        )

    def test_to_markdown_lists_unmapped_claims(self) -> None:
        text = to_markdown(
            [
                {
                    "id": "R9",
                    "topic": "Unknown",
                    "question": "Missing?",
                    "defense_status": "missing_claim_mapping",
                    "recommended_position": "Add mapping.",
                    "matched_claims": [],
                }
            ]
        )
        self.assertIn("No mapped claim yet", text)

    def test_default_concerns_include_ablation_failure_positions(self) -> None:
        by_id = {concern["id"]: concern for concern in DEFAULT_CONCERNS}

        self.assertIn("R6", by_id)
        self.assertIn("SFT-only", by_id["R6"]["fallback"])
        self.assertIn("RL stage is not supported", by_id["R6"]["fallback"])
        self.assertIn("C14", by_id["R6"]["fallback"])
        self.assertIn("BGE, verifier, threshold, and bootstrap-CI protocol", by_id["R6"]["fallback"])
        self.assertIn("BSC-only coverage changes are metric results", by_id["R3"]["fallback"])
        self.assertIn("supports it over", by_id["R5"]["fallback"])
        self.assertIn("same hard-gold BSC protocol", by_id["R5"]["fallback"])
        self.assertIn("coverage change still reportable under redundancy and hallucination controls", by_id["R3"]["question"])
        self.assertNotIn("hallucinated rubrics", by_id["R3"]["question"])

    def test_default_concerns_cover_reviewer_attack_points_from_playbook(self) -> None:
        by_id = {concern["id"]: concern for concern in DEFAULT_CONCERNS}

        self.assertIn("single-model evaluation-criteria policy", by_id["R1"]["question"])
        self.assertNotIn("single-model rubrics", by_id["R1"]["question"])

        self.assertIn("R7", by_id)
        self.assertIn("self-defined metric", by_id["R7"]["question"])
        self.assertIn("human-gold dimensions", by_id["R7"]["fallback"])
        self.assertIn("fixed BGE/threshold protocols", by_id["R7"]["fallback"])
        self.assertIn("C6-passing matched/unmatched human-audit summaries", by_id["R7"]["fallback"])
        self.assertIn("RewardBench/JudgeBench/RewardBench-2", by_id["R7"]["fallback"])

        self.assertIn("R8", by_id)
        self.assertIn("contaminate evaluation", by_id["R8"]["question"])
        self.assertIn("C0 safe_to_claim", by_id["R8"]["fallback"])
        self.assertIn("RubricBench test_main hard-gold holdout", by_id["R8"]["fallback"])
        self.assertIn("query-disjoint proxy/SFT/RL artifacts", by_id["R8"]["fallback"])
        self.assertIn("SHA-bound training/evaluation provenance", by_id["R8"]["fallback"])

        self.assertIn("R9", by_id)
        self.assertIn("Verifier Bias", by_id["R9"]["topic"])
        self.assertIn("no-verifier ablation", by_id["R9"]["fallback"])
        self.assertIn("C6-gated human-audit checks", by_id["R9"]["fallback"])

        self.assertIn("R10", by_id)
        self.assertIn("Human Audit", by_id["R10"]["topic"])
        self.assertIn("do not call BSC alignment validated", by_id["R10"]["fallback"])
        self.assertIn("human-audit summaries pass C6 strict gates", by_id["R10"]["fallback"])
        self.assertIn("auto-matched agreement", by_id["R10"]["fallback"])
        self.assertIn("auto-unmatched confirmation", by_id["R10"]["fallback"])

        self.assertIn("coverage_tau=0.75", by_id["R2"]["fallback"])
        self.assertIn("redundancy_tau=0.85", by_id["R2"]["fallback"])
        self.assertIn("Do not tune thresholds on RubricBench test_main", by_id["R2"]["fallback"])
        self.assertIn("C6 strict-gate human-audit summaries", by_id["R2"]["fallback"])

        self.assertIn("R11", by_id)
        self.assertEqual(by_id["R11"]["topic"], "Open-Ended RLVR Transfer")
        self.assertIn("open-ended semantic criteria elicitation", by_id["R11"]["question"])
        self.assertIn("prompt tuning or SFT imitation", by_id["R11"]["question"])
        self.assertIn("C14 supports SFT+GRPO over SFT-only", by_id["R11"]["fallback"])
        self.assertIn("C7 confirms the relevant trained reward-component ablations", by_id["R11"]["fallback"])
        self.assertNotIn("rubric coverage", by_id["R4"]["question"].lower())
        self.assertIn("evaluation-dimension coverage", by_id["R4"]["question"])
        self.assertIn("held-out preference-judging utility", by_id["R4"]["question"])
        self.assertIn("RewardBench, JudgeBench, and RewardBench-2", by_id["R4"]["fallback"])
        self.assertIn("paper_claim_eligible summaries", by_id["R4"]["fallback"])

    def test_default_concerns_avoid_unsupported_positive_overclaims(self) -> None:
        serialized = json.dumps(DEFAULT_CONCERNS, ensure_ascii=False).lower()
        banned = [
            "state-of-the-art",
            "guarantee acceptance",
            "guaranteed acceptance",
            "significantly improves",
            "significantly outperforms",
            "sft+grpo improves over",
            "better rubric generator",
        ]
        for phrase in banned:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, serialized)

    def test_rebuttal_manifest_binds_inputs_outputs_and_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence_matrix.json"
            readiness = root / "readiness_report.json"
            output_json = root / "rebuttal_pack.json"
            output_md = root / "rebuttal_pack.md"
            evidence.write_text('[{"claim_id": "C1"}]\n', encoding="utf-8")
            readiness.write_text('{"ok": false}\n', encoding="utf-8")
            output_json.write_text("[]\n", encoding="utf-8")
            output_md.write_text("# Pack\n", encoding="utf-8")
            concerns = [{"id": "R1", "topic": "Motivation", "question": "Why?"}]

            manifest = build_rebuttal_manifest(
                rows=[
                    {
                        "defense_status": "needs_evidence",
                        "readiness_ok": False,
                        "matched_claims": [{"claim_id": "C1"}],
                    },
                    {
                        "defense_status": "answer_ready",
                        "readiness_ok": True,
                        "matched_claims": [{"claim_id": "C4"}],
                    },
                ],
                concerns=concerns,
                claim_ladder=claim_ladder_from_evidence_rows(
                    [
                        {"claim_id": "C1", "status": "safe_to_claim"},
                        {"claim_id": "C6", "status": "safe_to_claim"},
                        {"claim_id": "C0", "status": "missing_evidence"},
                    ]
                ),
                concerns_path=None,
                evidence_matrix=evidence,
                readiness_report=readiness,
                output_json=output_json,
                output_md=output_md,
            )

            self.assertEqual(manifest["entry_count"], 2)
            self.assertEqual(manifest["defense_status_counts"]["cannot_claim"], 0)
            self.assertEqual(manifest["defense_status_counts"]["missing_claim_mapping"], 0)
            self.assertEqual(manifest["defense_status_counts"]["needs_readiness"], 0)
            self.assertEqual(manifest["defense_status_counts"]["needs_evidence"], 1)
            self.assertEqual(manifest["defense_status_counts"]["answer_ready"], 1)
            self.assertFalse(manifest["readiness_ok"])
            self.assertEqual(manifest["matched_claim_ids"], ["C1", "C4"])
            by_level = {row["level"]: row for row in manifest["claim_ladder"]}
            self.assertEqual(by_level["motivation"]["status"], "safe_to_claim")
            self.assertEqual(by_level["metric-support"]["status"], "missing_evidence")
            self.assertIn("C0: missing_evidence", by_level["metric-support"]["missing_or_non_safe_claims"])
            self.assertEqual(manifest["concern_templates"]["source"], "DEFAULT_CONCERNS")
            self.assertEqual(manifest["concern_templates"]["count"], 1)
            self.assertEqual(manifest["concern_templates"]["sha256"], json_sha256(concerns))
            self.assertEqual(manifest["inputs"]["evidence_matrix"]["sha256"], file_sha256(evidence))
            self.assertEqual(manifest["inputs"]["readiness_report"]["sha256"], file_sha256(readiness))
            self.assertEqual(manifest["outputs"]["rebuttal_pack_json"]["sha256"], file_sha256(output_json))
            self.assertIn("reviewer-facing readiness", manifest["claim_discipline"][0])

    def test_rebuttal_markdown_includes_claim_ladder_status(self) -> None:
        ladder = claim_ladder_from_evidence_rows(
            [
                {"claim_id": "C1", "status": "safe_to_claim"},
                {"claim_id": "C6", "status": "safe_to_claim"},
                {"claim_id": "C4", "status": "missing_evidence"},
            ]
        )
        text = to_markdown(
            [
                {
                    "id": "R4",
                    "topic": "Downstream Utility",
                    "question": "Useful?",
                    "defense_status": "needs_evidence",
                    "recommended_position": "Keep unclaimed.",
                    "matched_claims": [],
                }
            ],
            {"ok": False},
            ladder,
        )

        self.assertIn("## Claim Ladder Status", text)
        self.assertIn("| motivation | `safe_to_claim` | C1, C6 | none |", text)
        self.assertIn("| judge-utility support | `missing_evidence` | C0, C4, C9, C10, C12 |", text)
        self.assertIn("C4: missing_evidence", text)

    def test_new_default_concerns_map_to_evidence_sections(self) -> None:
        rows = [
            {
                "claim_id": "C0",
                "paper_section": "Data Hygiene",
                "claim": "RubricBench test_main is query-disjoint from training.",
                "status": "safe_to_claim",
                "evidence": "pass",
            },
            {
                "claim_id": "C6",
                "paper_section": "Robustness",
                "claim": "BSC findings are auditable across semantic threshold settings.",
                "status": "missing_evidence",
                "evidence": "missing human audit",
            },
            {
                "claim_id": "C7",
                "paper_section": "Ablation",
                "claim": "Ablations include the no-verifier-filtering proxy-gold variant.",
                "status": "missing_evidence",
                "evidence": "missing no-verifier run",
            },
            {
                "claim_id": "C4",
                "paper_section": "Downstream",
                "claim": "Downstream utility improves.",
                "status": "safe_to_claim",
                "evidence": "pass",
            },
            {
                "claim_id": "C14",
                "paper_section": "Main Results",
                "claim": "The GRPO/RLVR stage improves over SFT-only without relying on materially higher redundancy.",
                "status": "missing_evidence",
                "evidence": "missing SFT-only comparison",
            },
        ]
        by_id = {entry["id"]: entry for entry in build_rebuttal_pack(rows, DEFAULT_CONCERNS)}

        self.assertEqual(by_id["R8"]["defense_status"], "answer_ready")
        self.assertEqual(by_id["R8"]["matched_claims"][0]["claim_id"], "C0")

        self.assertEqual(by_id["R9"]["defense_status"], "needs_evidence")
        self.assertEqual(by_id["R9"]["matched_claims"][0]["claim_id"], "C7")

        self.assertEqual(by_id["R10"]["defense_status"], "needs_evidence")
        self.assertEqual(by_id["R10"]["matched_claims"][0]["claim_id"], "C6")

        self.assertEqual(by_id["R7"]["defense_status"], "needs_evidence")
        self.assertEqual(
            {claim["claim_id"] for claim in by_id["R7"]["matched_claims"]},
            {"C4", "C6"},
        )

        self.assertEqual(by_id["R11"]["defense_status"], "needs_evidence")
        self.assertEqual(by_id["R11"]["matched_claims"][0]["claim_id"], "C14")

    def test_generated_rebuttal_pack_uses_hypothesis_framed_real_claims(self) -> None:
        md_path = ROOT / "outputs" / "rebuttal_pack" / "rebuttal_pack.md"
        json_path = ROOT / "outputs" / "rebuttal_pack" / "rebuttal_pack.json"
        text = md_path.read_text(encoding="utf-8")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        serialized = text + "\n" + json.dumps(payload, ensure_ascii=False)

        self.assertIn("are tested for downstream chosen-vs-rejected utility", serialized)
        self.assertIn(
            "is evaluated against the best single teacher for human-gold evaluation-dimension coverage",
            serialized,
        )
        self.assertIn("tested against SFT-only", serialized)
        self.assertIn("coverage change still reportable under redundancy and hallucination controls", serialized)

        for stale_claim in [
            "preserve or improve downstream",
            "preserve or improve chosen-vs-rejected",
            "preserve or improve multi-candidate",
            "criteria union covers more",
            "is tested for higher human-gold evaluation-dimension coverage",
            "coverage gain explained by redundant or hallucinated criteria",
            "The GRPO/RLVR stage improves over SFT-only",
        ]:
            with self.subTest(stale_claim=stale_claim):
                self.assertNotIn(stale_claim, serialized)


if __name__ == "__main__":
    unittest.main()
