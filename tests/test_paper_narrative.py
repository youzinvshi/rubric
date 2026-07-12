from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


class PaperNarrativeTest(unittest.TestCase):
    def test_abstract_keeps_blindspot_not_better_generator_framing(self) -> None:
        text = read("sections/abstract.tex")
        normalized = " ".join(text.split())

        self.assertIn("evaluation blind spots", text)
        self.assertIn("36.9\\%", text)
        self.assertIn("63.1\\%", text)
        self.assertIn("95\\% CI: 57.9--68.2", text)
        self.assertIn("single-model evaluation-criteria policy covers only 36.9\\%", normalized)
        self.assertIn("human-gold evaluation dimensions", normalized)
        self.assertNotIn("human-authored evaluation dimensions", normalized)
        self.assertIn("coverage credit requires semantic match to hidden evaluation dimensions", normalized)
        self.assertIn("fail-closed validity checks penalize duplicate, invalid, undecidable, irrelevant, or hallucinated criteria", normalized)
        self.assertNotIn("a candidate criterion is useful only if", normalized)
        self.assertIn("open-ended criteria elicitation as verifiable semantic reward optimization", normalized)
        self.assertIn("rather than as surface-form optimization", normalized)
        self.assertIn("evidence-gated experiments then test when", normalized)
        self.assertIn("artifacts are sufficient to write a dimension-level coverage-change statement", normalized)
        self.assertIn("paper-facing dimension-recovery wording is permitted only after", normalized)
        self.assertNotIn("reward supports a dimension-recovery statement", normalized)
        self.assertNotIn("paper-facing dimension-recovery claim is promoted only after", normalized)
        self.assertIn("dimension-transition", normalized)
        self.assertIn("tests GRPO/RLVR optimization", normalized)
        self.assertNotIn("strong single-model rubric generator", normalized)
        self.assertNotIn("The central claim is not that we build a better rubric generator", normalized)
        self.assertNotIn("enabling LLM judges to repair", normalized)
        self.assertNotIn("then optimizes a rubric generator", normalized)
        self.assertNotIn("test whether this reward repairs systematic gaps", normalized)
        self.assertNotIn("can reduce systematic dimension gaps", normalized)
        self.assertNotIn("can be trained with verifiable semantic rewards", normalized)

    def test_title_avoids_unverified_repair_claim(self) -> None:
        main = read("main.tex")
        bib = read("references.bib")

        expected = "Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation"
        self.assertIn(expected, main)
        self.assertIn(expected, bib)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Covering Evaluation Blind Spots", main)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Covering Evaluation Blind Spots", bib)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Evaluation Blind Spots", main)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Evaluation Blind Spots", bib)
        self.assertNotIn("BlindSpot-RL: Repairing Evaluation Blind Spots", main)
        self.assertNotIn("BlindSpot-RL: Repairing Evaluation Blind Spots", bib)

    def test_readme_uses_current_aaai_positioning_and_claim_discipline(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())

        self.assertIn("upstream failure mode in LLM-as-a-Judge", normalized)
        self.assertIn("its evaluation-criteria basis may already omit dimensions", normalized)
        self.assertNotIn("the rubric it relies on may already omit", normalized)
        self.assertIn("evaluation blind spots", text)
        self.assertIn("not as a generic criteria-text polishing problem", normalized)
        self.assertIn("Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation", text)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Covering Evaluation Blind Spots", text)
        self.assertIn("0.3692", text)
        self.assertIn("0.6308", text)
        self.assertIn("against human-gold evaluation dimensions", normalized)
        self.assertNotIn("against human rubrics", normalized)
        self.assertIn("Evidence-gated", text)
        self.assertIn("BSC-only results are metric-support, not downstream judge-utility evidence", normalized)
        self.assertIn("RewardBench, RewardBench-2, and JudgeBench rows require the API/model scorer", normalized)
        self.assertIn("paper-eligibility gates before utility claims are safe", normalized)
        self.assertIn("1,785-example clean split", normalized)
        self.assertIn("RewardBench/JudgeBench/ RewardBench-2 downstream holdouts", normalized)
        self.assertIn("Proxy-gold criteria are used for training scale", normalized)
        self.assertIn("must not be described as equivalent to human-gold evaluation dimensions", normalized)
        self.assertIn("--input data/processed/rewardbench2_queries.clean.jsonl", text)
        self.assertNotIn("--input data/processed/rewardbench2_queries.jsonl", text)
        self.assertIn("After a method produces evaluation criteria for RewardBench/JudgeBench prompts", text)
        self.assertIn("criterion-based scoring ranks `chosen` above `rejected`", text)
        self.assertIn("test when a training-based dimension-level recovery statement is warranted", normalized)
        self.assertNotIn("test whether training reduces dimension-level blind spots", normalized)
        self.assertIn("Any dimension-recovery claim should remain deferred", normalized)
        self.assertIn("semantic_space.pdf", text)
        self.assertIn("supports UMAP/t-SNE projections", normalized)
        self.assertIn("requested and actual projection", normalized)
        self.assertIn("audited deterministic PCA fallback", normalized)
        self.assertIn("SVG/PDF pair is the audit figure", normalized)
        self.assertIn("BSC_VERIFIER=rule", text)
        self.assertIn("BSC_W_VALID=0.0 BSC_VERIFIER=none", text)
        self.assertIn("paper_claim_eligible=false", text)
        self.assertIn("paper_claim_eligible=true", text)
        self.assertIn("only exposes downstream", text)
        self.assertIn("downstream_paper_claim_eligible", text)
        self.assertIn("When `require_downstream=true`, a downstream summary", text)
        self.assertIn("treated as `not_paper_eligible`", text)
        self.assertIn("input_sha256", text)
        self.assertIn("scorer_provider_sha256", text)
        self.assertIn("contract.providers_sha256", text)
        self.assertIn("The converter fails closed if the input path, `--data-source`, or record-level", text)
        self.assertIn("metadata indicates `test_main`", text)
        self.assertIn("GRPO data must come\nfrom proxy-gold criteria", text)
        self.assertIn("--data-source multi_teacher_proxy", text)
        self.assertIn("--report-output outputs/sft_data/proxy_gold_verl_report.json", text)
        self.assertIn("The report records input/output SHA256 hashes and the converted record count", text)
        self.assertIn("validates `outputs/sft_data/proxy_gold_verl_report.json` against", text)
        self.assertIn("stale GRPO data or\na stale conversion report blocks trained-method claims", text)
        self.assertIn("--output-dir outputs/submission_readiness/gap_report", text)
        self.assertNotIn("--output-dir outputs/submission_gap_report", text)
        self.assertIn("--output-json outputs/dashboard/real_run_dashboard.json", text)
        self.assertIn("--output-md outputs/dashboard/real_run_dashboard.md", text)
        self.assertNotIn("--output-json outputs/dashboard/run_dashboard.json", text)
        self.assertIn("`outputs/dashboard/real_run_dashboard.{json,md}`", text)
        self.assertIn("`outputs/submission_readiness/gap_report/submission_gap_report.{json,md}`", text)
        self.assertIn("Submission readiness also checks that the dashboard", text)
        self.assertIn("It also uses a `submission_gap_report` section", text)
        self.assertIn("top-level `Claim Ladder Status` table", text)
        self.assertIn("claim-ladder layer", text)
        self.assertIn("claim-ladder milestones", text)
        self.assertIn("Each day records the manuscript conclusion layer it can unlock", text)
        self.assertIn("same four-level claim ladder and downgrade rules", text)
        self.assertIn("Its Markdown and\nmanifest also expose `Claim Ladder Status`", text)
        self.assertIn("`claim_ladder_safe=x/4`", text)
        self.assertIn("Dashboard Diagnostics table", text)
        self.assertIn("Use the same claim ladder as the paper", text)
        self.assertIn("the frozen 100-example hard-gold\ndiagnostic is only a motivation claim", text)
        self.assertIn("hard-gold BSC coverage change with stable\nredundancy and hallucination is metric-support", text)
        self.assertIn("artifact_status=blocked", text)
        self.assertIn("overlap_status=not_auditable", text)
        self.assertIn("SFT-only vs SFT+GRPO row\nand reward-component ablations are required for method-support", text)
        self.assertIn("RewardBench/RewardBench-2/JudgeBench API-scorer rows are required for\njudge-utility support", text)
        self.assertIn("without downstream support, write metric-only BSC evidence", text)
        self.assertIn("without C12, write aggregate coverage change rather than dimension-level recovery", text)
        self.assertIn("without C14, write proxy-gold supervision evidence rather than RLVR evidence", text)
        self.assertIn("without clean C0 provenance, no trained-method row is paper-facing", text)
        self.assertIn("requires `outputs/verifier/teacher_rubrics_filtered_report.json`", text)
        self.assertIn("binds the raw teacher criteria file, filtered teacher criteria file", text)
        self.assertIn("SFT-data preflight report, and meta-verifier budget\nreport by SHA256", text)
        self.assertIn("compares that report's `output_sha256`\nagainst `proxy_gold_build_report.json:input_sha256`", text)
        self.assertIn("--input data/processed/teacher_rubrics_raw.jsonl", text)
        self.assertIn("--report-output outputs/verifier/teacher_rubrics_filtered_report.json", text)
        self.assertIn("--require-preflight-report outputs/preflight/sft_data_preflight.json", text)
        self.assertIn("filtered teacher output SHA256", text)
        self.assertIn("table_values", text)
        self.assertIn("downstream_status=pass", text)
        self.assertIn("cannot satisfy C4/C9/C10", text)
        self.assertIn("C13 validates the exact point CSV", text)
        self.assertIn("including SFT+GRPO rows", text)
        self.assertIn("nearest-gold id, category, similarity, and text fields", normalized)
        self.assertNotIn("first engineering milestone", text)
        self.assertNotIn("After a Rubric-Generator produces rubrics", text)
        self.assertNotIn("support the claim that training repairs baseline blind spots", normalized)

    def test_section_2_keeps_frozen_diagnostic_numbers(self) -> None:
        text = read("sections/blindspot_phenomenon.tex")
        normalized = " ".join(text.split())

        for value in ["0.3692", "0.6308", "0.5785--0.6823", "0.6667", "74 of 100", "21", "0.0768", "0.1227"]:
            self.assertIn(value, text)
        self.assertIn("100 hard-gold RubricBench examples", text)
        self.assertIn("not driven by a small number of outliers", text)
        self.assertIn("not merely a formatting artifact", text)
        self.assertIn("base single-model evaluation-criteria policy achieves", text)
        self.assertIn("minimal motivation} gate", text)
        self.assertIn("separate from the full real-run C0--C14\nreadiness matrix", text)
        self.assertIn("cannot make a\ntrained-method row paper-facing", text)
        self.assertIn("not that an\nevaluation-criteria policy occasionally omits a niche item", text)
        self.assertIn("most hard-gold\nqueries lose a large fraction of the human evaluative basis", text)
        self.assertIn("final preference accuracy can hide an\nupstream coverage gap", text)
        self.assertIn("rather than as another surface metric\nfor judging criteria verbosity", text)
        self.assertIn("Not Just a Criteria-Budget Effect", text)
        self.assertIn("denote human-gold evaluation dimensions", normalized)
        self.assertIn("human-gold evaluation dimensions even after producing plausible criteria", normalized)
        self.assertNotIn("denote human-authored evaluation dimensions", normalized)
        self.assertNotIn("human-authored evaluation dimensions even after producing plausible criteria", normalized)
        self.assertIn("This attribution is descriptive motivation\nevidence only", text)
        self.assertIn("category-level reduction claims require the later C12\ndimension-transition audit", text)
        self.assertNotIn("category-level repair claims", text)

    def test_section_2_distribution_numbers_match_bsc_summary(self) -> None:
        text = read("sections/blindspot_phenomenon.tex")
        summary = json.loads((ROOT / "outputs/minimal_claim/base/bsc/summary.json").read_text(encoding="utf-8"))

        self.assertIn(f"{summary['median_blind']:.4f}", text)
        self.assertIn(f"{summary['queries_coverage_le_0_5']} of {summary['n']}", text)
        self.assertIn(f"{summary['queries_zero_coverage']} queries have zero covered", text)

    def test_section_2_numbers_are_backed_by_frozen_evidence_gate(self) -> None:
        text = read("sections/blindspot_phenomenon.tex")
        evidence = json.loads((ROOT / "configs/evidence_minimal_claim.generated.json").read_text(encoding="utf-8"))
        frozen_metrics = {
            item["metric"]: item
            for item in evidence["claims"][0]["metrics"]
            if item["label"].startswith("Frozen diagnostic")
        }

        self.assertEqual(frozen_metrics["mean_coverage"]["value"], 0.36919517010450364)
        self.assertEqual(frozen_metrics["mean_blind"]["value"], 0.6308048298954964)
        self.assertEqual(frozen_metrics["median_blind"]["value"], 0.6666666567325592)
        self.assertEqual(frozen_metrics["queries_coverage_le_0_5"]["value"], 74)
        self.assertEqual(frozen_metrics["queries_zero_coverage"]["value"], 21)
        self.assertEqual(frozen_metrics["mean_redundancy"]["value"], 0.07675396825396825)
        self.assertEqual(frozen_metrics["mean_hallucination"]["value"], 0.12273412698412695)
        self.assertIn("0.3692", text)
        self.assertIn("0.6308", text)
        self.assertIn("74 of 100", text)
        self.assertIn("21 queries", text)

    def test_section_2_threshold_robustness_numbers_match_sweep(self) -> None:
        text = read("sections/blindspot_phenomenon.tex")
        sweep = json.loads(
            (ROOT / "outputs/minimal_claim/base/bsc_sweep/threshold_sweep.json").read_text(encoding="utf-8")
        )
        loose = next(row for row in sweep if row["coverage_tau"] == 0.7 and row["redundancy_tau"] == 0.85)
        strict = next(row for row in sweep if row["coverage_tau"] == 0.8 and row["redundancy_tau"] == 0.85)

        self.assertIn(f"{loose['mean_blind']:.4f}", text)
        self.assertIn(f"{strict['mean_blind']:.4f}", text)
        self.assertIn("not an artifact of a single semantic threshold", text)

    def test_section_2_blindspot_attribution_numbers_match_outputs_and_gate(self) -> None:
        text = read("sections/blindspot_phenomenon.tex")
        with (ROOT / "outputs/minimal_claim/base/blindspot_map/category_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as f:
            rows = {row["category"]: row for row in csv.DictReader(f)}
        evidence = json.loads((ROOT / "configs/evidence_minimal_claim.generated.json").read_text(encoding="utf-8"))
        table_values = {
            (str(item["row_value"]), item["key"]): item
            for item in evidence["claims"][0].get("table_values", [])
            if item["label"].startswith("Frozen attribution")
        }

        category_names = {
            "constraint_following": "Constraint following",
            "intent_reasoning": "Intent/reasoning",
            "completeness": "Completeness",
        }
        for category, label in category_names.items():
            row = rows[category]
            self.assertIn(f"{float(row['blind_rate']):.4f}", text)
            self.assertIn(f"{float(row['coverage']):.4f}", text)
            self.assertIn(label, text)
            self.assertEqual(table_values[(category, "blind_rate")]["value"], float(row["blind_rate"]))
            self.assertEqual(table_values[(category, "coverage")]["value"], float(row["coverage"]))

    def test_section_2_budget_curve_numbers_match_outputs_and_gate(self) -> None:
        text = read("sections/blindspot_phenomenon.tex")
        normalized = " ".join(text.split())
        with (ROOT / "outputs/minimal_claim/base/budget_curve/coverage_by_k.csv").open(
            "r", encoding="utf-8", newline=""
        ) as f:
            rows = {row["k"]: row for row in csv.DictReader(f)}
        evidence = json.loads((ROOT / "configs/evidence_minimal_claim.generated.json").read_text(encoding="utf-8"))
        table_values = {
            (str(item["row_value"]), item["key"]): item
            for item in evidence["claims"][0].get("table_values", [])
            if item["label"].startswith("Frozen budget curve")
        }

        for k in ["3", "10", "15"]:
            row = rows[k]
            self.assertEqual(table_values[(k, "mean_coverage")]["value"], float(row["mean_coverage"]))
            self.assertEqual(table_values[(k, "total_gen")]["value"], int(row["total_gen"]))
        self.assertIn(f"{float(rows['3']['mean_coverage']):.4f} at $K=3$", text)
        self.assertIn(f"{float(rows['10']['mean_coverage']):.4f} at $K=10$", text)
        self.assertIn(f"produced {int(rows['10']['total_gen'])} total criteria", text)
        self.assertIn("Under a larger\ncriteria budget, coverage moves", text)
        self.assertIn("simply making criteria lists longer is insufficient", normalized)
        self.assertNotIn("Coverage increases", text)
        self.assertNotIn("coverage increase", text)

    def test_experiments_keep_required_evidence_closure(self) -> None:
        text = read("sections/experiments.tex")
        normalized = " ".join(text.split())

        self.assertIn("RubricBench \\texttt{test\\_main}", text)
        self.assertIn("not used for SFT, proxy criteria elicitation, or reward tuning", text)
        self.assertIn("not used for prompt selection, hyperparameter tuning, checkpoint\nselection, or verifier calibration", text)
        self.assertIn("used only for small-scale initialization and protocol calibration", normalized)
        self.assertIn("Proxy-gold training data is used only for scaling supervision", normalized)
        self.assertIn("RewardBench, RewardBench-2, and JudgeBench", text)
        self.assertIn("1,785", text)
        self.assertIn("597 RewardBench records", normalized)
        self.assertIn("551 normalized unique queries", normalized)
        self.assertIn("1,815 RewardBench-2 records", normalized)
        self.assertIn("1,814 raw stripped unique queries", normalized)
        self.assertIn("1,813 normalized unique queries", normalized)
        self.assertIn("620 JudgeBench records", normalized)
        self.assertIn("528 normalized unique queries", normalized)
        self.assertIn("downstream RewardBench, JudgeBench, and RewardBench-2 holdouts", normalized)
        self.assertIn("non-overlap held-out RewardBench, RewardBench-2, and JudgeBench", normalized)
        self.assertIn(
            "bind the exact training inputs, proxy-train filtering reports, zero-overlap holdout audits, and evaluation outputs",
            normalized,
        )
        self.assertIn("not considered evidence without the corresponding non-overlap audit", normalized)
        self.assertIn("overlap\\_query\\_count == 0} but \\texttt{artifact\\_status=blocked}", normalized)
        self.assertIn("treated as not auditable rather than as clean isolation", normalized)
        self.assertIn("these metrics are conjunctive rather than independent", normalized)
        self.assertIn("reportable hard-gold semantic coverage change under the fixed protocol", normalized)
        self.assertIn("no material degradation in redundancy or hallucination", normalized)
        self.assertNotIn("positive hard-gold semantic coverage change", normalized)
        self.assertIn("LLM-generated evaluation criteria exhibit systematic blind spots", normalized)
        self.assertNotIn("LLM rubric generators exhibit systematic evaluation blind spots", normalized)
        self.assertNotIn("higher BSC", text)
        self.assertIn("base evaluation-criteria policy", normalized)
        self.assertIn("SFT-only policy", text)
        self.assertIn("full SFT+RL policy", text)
        self.assertIn("trained policy", normalized)
        self.assertNotIn("base generator", normalized)
        self.assertNotIn("trained generator", normalized)
        self.assertIn("held-out downstream judge-utility support", normalized)
        self.assertIn("numerically stronger BSC value alone is treatable only as metric evidence", normalized)

        self.assertIn("This minimal motivation gate is safe for Section 2 only", normalized)
        self.assertIn("not the full real-run readiness matrix and does not unlock any trained-method row", normalized)
        self.assertIn("Claim Ladder and Downgrade Rules", text)
        self.assertIn("uses a claim ladder rather than a single all-or-nothing success criterion", normalized)
        self.assertIn("frozen 100-example hard-gold diagnostic is a \\emph{motivation} claim", normalized)
        self.assertIn("hard-gold BSC coverage change with stable redundancy and hallucination is a \\emph{metric-support} claim", normalized)
        self.assertIn("becomes \\emph{method-support} only when the SFT-only versus SFT+GRPO comparison", normalized)
        self.assertIn("becomes \\emph{judge-utility support} only when the held-out RewardBench, RewardBench-2, and JudgeBench rows", normalized)
        self.assertIn("Without downstream support, the result is reported as metric-only BSC evidence", normalized)
        self.assertIn("Without C12 dimension-transition support, the result is reported as an aggregate coverage change", normalized)
        self.assertIn("Without C14 SFT-only versus SFT+GRPO support", normalized)
        self.assertIn("Without clean C0 contamination and provenance gates, no trained method row is paper-facing", normalized)
        self.assertIn("\\label{tab:evidence_gates}", text)
        self.assertIn("\\IfFileExists{tables/main_table.tex}{\\input{tables/main_table}}", text)
        self.assertIn("\\typeout{Main results table not included because tables/main_table.tex is missing.}", text)
        self.assertNotIn("\\fbox", text)
        self.assertNotIn("Main table not synced yet", text)
        self.assertNotIn("Run\n\\texttt{scripts/sync\\_paper\\_artifacts.py}", text)
        self.assertIn("C4/C9/C10", text)
        self.assertIn("RewardBench, JudgeBench, and RewardBench-2 utility", normalized)
        self.assertIn("C7 & Reward-component and verifier ablations", text)
        self.assertIn("C12 & Query-aligned per-gold dimension transition audit", text)
        self.assertIn("C13 & Point-level semantic-space CSV, JSON summary, and rendered figure", text)
        self.assertIn("C14 & SFT-only versus SFT+GRPO under the same protocol", text)
        self.assertIn("The gates are conjunctive for the central method claim", normalized)
        for phrase in [
            "No redundancy penalty",
            "No hallucination/validity term",
            "No verifier filtering",
            "SFT-only vs. SFT+GRPO",
            "Single teacher vs. multi-teacher union",
        ]:
            self.assertIn(phrase, text)
        self.assertIn("figures/semantic_space.pdf", text)
        self.assertIn("The relevant RL-stage claim test is not whether RL produces longer criteria lists", normalized)
        self.assertIn("The RL-stage claim is safe only if", normalized)
        self.assertNotIn("The target conclusion is not that RL produces longer criteria lists", normalized)
        self.assertIn("same BGE embedding model", text)
        self.assertIn("coverage and redundancy thresholds", text)
        self.assertIn("verifier-backed hallucination source", text)
        self.assertIn("mean generated-criteria count, generated-to-gold ratio", normalized)
        self.assertIn("coverage per generated criterion remain bounded under the length-control audit", normalized)
        self.assertIn("bootstrap-CI protocol", text)
        self.assertIn("removing the redundancy penalty\nraises coverage only by increasing duplicate criteria", text)
        self.assertIn("unsupported for dimension-level recovery", normalized)
        self.assertIn("Evidence supporting attribution\nto reward components rather than verbosity", text)
        self.assertNotIn("components, not verbosity, explain the effect", text)
        self.assertNotIn("reward hacking rather than faithful dimension-level recovery", normalized)
        self.assertIn("removing the validity term or\nverifier filtering raises coverage while hallucination rises", text)
        self.assertIn("SFT-only matches SFT+GRPO under the same protocol", normalized)
        self.assertIn("RLVR stage should be\nreported as unsupported for the observed coverage change", text)
        self.assertIn("multi-teacher union does\nnot beat the strongest single teacher", text)
        self.assertIn("uncontrolled data-construction choice rather than an empirically supported data-construction\nadvantage", text)
        self.assertIn("recovered human-gold dimensions exceeding lost dimensions", normalized)
        self.assertNotIn("net positive recovery", normalized)
        self.assertIn("baseline and candidate rows align by query with no unmatched records", normalized)
        self.assertIn("respect verifier \\texttt{valid\\_flags}", text)
        self.assertIn("trained-ablation completion artifact must enumerate", normalized)
        self.assertIn(
            "\\texttt{full}, \\texttt{no\\_red}, \\texttt{no\\_valid}, \\texttt{no\\_verifier}, and \\texttt{cov\\_only}",
            normalized,
        )
        self.assertIn("keeps the verifier enabled for the full and no-redundancy variants", normalized)
        self.assertIn("disables both the validity weight and verifier reward for the no-validity variant", normalized)
        self.assertIn("disables the validity weight, redundancy weight, and verifier reward", normalized)
        self.assertIn("verifier filtering does not increase hallucination", normalized)
        self.assertIn("does not collapse hard-gold semantic coverage", normalized)
        self.assertIn("To audit the proposed semantic-coverage mechanism", normalized)
        self.assertNotIn("To make the orthogonality mechanism visible", normalized)
        self.assertIn("The visualization audit tests whether SFT+GRPO criteria", normalized)
        self.assertIn("reportable semantic coverage change", normalized)
        self.assertNotIn("positive semantic coverage change", normalized)
        self.assertNotIn("higher semantic coverage", normalized)
        self.assertIn("C13 audit tests whether SFT+GRPO has a reportable nearest-gold region-coverage change", normalized)
        self.assertIn("subject to point-level CSV/JSON evidence", normalized)
        self.assertNotIn("the figure tests whether SFT+GRPO covers", normalized)
        self.assertNotIn("C13 audit tests whether SFT+GRPO covers", normalized)
        self.assertNotIn("broader gold regions than SFT-only", normalized)
        self.assertIn("Downstream utility is therefore the external validity check for BSC", normalized)
        self.assertIn("metric-only evidence and not as judge-utility evidence", normalized)
        self.assertNotIn("BSC is higher", text)
        self.assertIn("records the exact joined evaluation input", normalized)
        self.assertIn("current input/provider SHA256 hashes", text)
        self.assertIn("compares these summary hashes against the budget-report contract hashes", normalized)
        self.assertIn("matching paths alone cannot make stale API-budget reports", normalized)
        self.assertIn("verifier-filtering report", normalized)
        self.assertIn("raw teacher criteria, filtered teacher criteria, Meta-Verifier provider", normalized)
        self.assertIn("SFT-data preflight report, and meta-verifier budget report by SHA256", normalized)
        self.assertIn("filtered-output digest must match the proxy-gold construction input digest", normalized)
        self.assertIn("scorer-call contract", normalized)
        self.assertIn("paper\\_claim\\_eligible", text)
        self.assertIn("keyword smoke runs paper-facing utility evidence", normalized)
        self.assertIn("nearest-gold category coverage, nearest-gold gold-cluster coverage", normalized)
        self.assertIn("point-level CSV and JSON summary do not show nearest-gold cluster coverage change", normalized)
        self.assertIn("auditable only under C13: a mechanism claim is reportable only if the point-level evidence supports nearest-gold coverage change", normalized)
        self.assertIn("redundancy ablation is consistent with local-collapse evidence", normalized)
        self.assertNotIn("broader nearest-gold cluster coverage", normalized)
        self.assertNotIn("broader nearest-gold coverage", normalized)
        self.assertNotIn("cover more semantic regions", normalized)
        self.assertIn("illustrative only, not as mechanism evidence", normalized)

    def test_method_states_rewardbench_split_and_clean_proxy_train_counts(self) -> None:
        text = read("sections/method.tex")
        normalized = " ".join(text.split())

        self.assertIn("RewardBench is first split by query", normalized)
        self.assertIn("\\texttt{sft\\_proxy\\_train}, \\texttt{dev}, and \\texttt{downstream\\_holdout}", normalized)
        self.assertIn("1,791/597/597 records", normalized)
        self.assertIn("clean proxy-training partition contains 1,785 records", normalized)
        self.assertIn("RewardBench-2 and JudgeBench are full downstream holdouts", normalized)
        self.assertIn("597 RewardBench records with 551 normalized unique queries", normalized)
        self.assertIn("620 JudgeBench records with 528 normalized unique queries", normalized)
        self.assertIn("1,815 RewardBench-2 records with 1,814 raw stripped unique queries", normalized)
        self.assertIn("1,813 normalized unique queries", normalized)

    def test_paper_body_and_real_evidence_matrix_avoid_generator_framing(self) -> None:
        section_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((PAPER / "sections").glob("*.tex")))
        evidence_text = (ROOT / "configs/evidence_matrix_real.template.json").read_text(encoding="utf-8")
        asset_text = "\n".join(
            [
                (PAPER / "asset_index" / "evidence_matrix.json").read_text(encoding="utf-8"),
                (PAPER / "asset_index" / "result_card.json").read_text(encoding="utf-8"),
                (PAPER / "asset_index" / "result_card.md").read_text(encoding="utf-8"),
            ]
        )

        self.assertNotIn("generator", section_text.lower())
        self.assertNotIn("generator", evidence_text.lower())
        self.assertIn("evaluation-criteria policy leaves measurable blind spots", asset_text)
        self.assertIn("human-gold evaluation dimensions", asset_text)
        self.assertIn("RubricBench test_main hard-gold", asset_text)
        self.assertIn("hard-gold/proxy/downstream isolation", asset_text)
        self.assertNotIn("Single-model rubric generation", asset_text)
        self.assertNotIn("hard-gold rubrics", asset_text)

    def test_synced_public_assets_use_c6_gated_human_audit_language(self) -> None:
        docs = synced_public_asset_texts()
        self.assertTrue(docs)
        combined = "\n".join(docs.values())

        self.assertIn("c6-gated human-audit summary", combined)
        self.assertIn("c6 strict-gate human-audit summaries", combined)
        for phrase in [
            "completed human audit",
            "human audit is complete",
            "human audit has enough completed labels",
            "human audit has no invalid labels",
            "human audit uncertain rate is bounded",
            "rubric generator",
            "does not improve coverage",
            "coverage improvement",
            "utility improvements",
            "expected hypotheses",
            "semantic-space visualization assets are available",
            "cover broader human-gold evaluation-dimension regions",
        ]:
            with self.subTest(phrase=phrase):
                offenders = [path for path, text in docs.items() if phrase in text]
                self.assertEqual(offenders, [])

    def test_main_sections_keep_unfinished_training_results_evidence_gated(self) -> None:
        combined = "\n".join(
            read(path)
            for path in [
                "sections/abstract.tex",
                "sections/introduction.tex",
                "sections/experiments.tex",
            ]
        )
        normalized = " ".join(combined.split())

        self.assertNotIn("We show how RLVR/GRPO can be applied", combined)
        self.assertNotIn("uses this reward to extend RLVR/GRPO", combined)
        self.assertNotIn("Show that RLVR/GRPO can optimize", combined)
        self.assertNotIn("The expected conclusion is not that RL produces longer rubrics", combined)
        self.assertNotIn("broader coverage of gold regions after SFT+GRPO supports", normalized)
        self.assertNotIn("SFT+GRPO criteria cover more nearest-gold clusters", combined)
        self.assertNotIn("SFT+GRPO covers more nearest-gold regions", combined)
        self.assertIn("show a reportable nearest-gold cluster-coverage change relative to SFT-only", normalized)
        self.assertIn("reportable nearest-gold region-coverage change relative to SFT-only", normalized)
        self.assertIn("formulates a test of whether this reward can extend RLVR/GRPO", normalized)
        self.assertIn("empirical dimension-recovery wording is allowed only after hard-gold, downstream, ablation, and dimension-transition evidence gates pass", normalized)
        self.assertIn("claim is safe only if the hard-gold evidence", normalized)
        self.assertIn("When do the evidence gates warrant a dimension-level coverage-change or recovery statement", normalized)
        self.assertIn("dimension-transition audits, and criteria-budget curves", normalized)
        self.assertNotIn("When the evidence gates pass, does BlindSpot-RL support dimension-level recovery", normalized)
        self.assertNotIn("Does BlindSpot-RL reduce dimension-level blind spots without merely increasing length", normalized)
        self.assertNotIn("Does BlindSpot-RL repair blind spots without merely increasing length", normalized)

    def test_introduction_keeps_three_part_research_contribution_framing(self) -> None:
        text = read("sections/introduction.tex")
        normalized = " ".join(text.split())
        contribution_block = text.split("Our contributions are:", 1)[1].split("\\end{itemize}", 1)[0]

        self.assertEqual(contribution_block.count("\\item"), 3)
        self.assertIn("problem--metric--method structure", normalized)
        self.assertIn("rather than a claim that a larger policy is simply better", normalized)
        self.assertIn("The experiments are deliberately claim-gated", normalized)
        self.assertIn("determine which empirical statements are safe to include", normalized)
        self.assertIn("evaluation blind spots as an upstream failure mode", normalized)
        self.assertIn("different object of study from judge accuracy, preference alignment, or reward-model ranking", normalized)
        self.assertIn("whether the evaluative basis available to the judge already contains the human-prioritized dimensions", normalized)
        self.assertIn("correct-looking decision can still hide systematic missing dimensions", normalized)
        self.assertIn("from the human-gold criteria while still covering the same semantic dimension", normalized)
        self.assertIn("Blind-Spot Coverage, a semantic metric and verifiable reward", normalized)
        self.assertIn("evidence-gated RLVR/GRPO training route", normalized)
        self.assertNotIn("from the human-authored criteria", normalized)
        self.assertNotIn("contamination-aware evidence pipeline", contribution_block)
        self.assertNotIn("larger policy is simply better", contribution_block)

    def test_method_documents_downstream_holdout_filtering_protocol(self) -> None:
        text = read("sections/method.tex")
        normalized = " ".join(text.split())

        self.assertIn("query-level holdout filtering", normalized)
        self.assertIn("RubricBench \\texttt{test\\_main}", text)
        self.assertIn("not used to choose prompts, tune reward weights, calibrate the verifier, select checkpoints", normalized)
        self.assertIn("construct proxy-gold targets", normalized)
        self.assertIn("undecidable criteria therefore reduce $R_{\\mathrm{valid}}$ just like hallucinated or irrelevant criteria", normalized)
        self.assertIn("downstream RewardBench, JudgeBench, and RewardBench-2 holdouts", normalized)
        self.assertIn("\\texttt{overlap\\_query\\_count == 0}", text)
        self.assertIn("\\texttt{artifact\\_status=blocked}", text)
        self.assertIn("\\texttt{overlap\\_status=not\\_auditable}", text)
        self.assertIn("cannot support C0", normalized)
        self.assertIn("nested reward metadata", normalized)
        self.assertIn("\\texttt{extra\\_info.query}", text)
        self.assertIn("\\texttt{proxy\\_gold\\_verl.parquet}", text)
        self.assertIn("verifier-filtering report binds the raw teacher criteria", normalized)
        self.assertIn("filtered teacher criteria, the Meta-Verifier provider", normalized)
        self.assertIn("SFT-data preflight report, and the meta-verifier budget report by SHA256", normalized)
        self.assertIn("filtered teacher digest is the same input digest consumed by the proxy-gold construction report", normalized)
        self.assertIn("main and downstream utility claims", normalized)

    def test_related_work_positions_against_judge_accuracy_and_rubric_conditioning(self) -> None:
        text = read("sections/related_work.tex")
        normalized = " ".join(text.split())

        self.assertIn("G-Eval", text)
        self.assertIn("\\cite{liu2023geval}", text)
        self.assertIn("RewardBench", text)
        self.assertIn("\\cite{lambert2024rewardbench}", text)
        self.assertIn("Prometheus", text)
        self.assertIn("\\cite{kim2023prometheus}", text)
        self.assertIn("final decision layer", normalized)
        self.assertIn("dimension-level coverage target against human-gold dimensions", normalized)
        self.assertIn("assume that the supplied criteria already contain the right dimensions", normalized)
        self.assertIn("complementary to judge accuracy, preference alignment, and reward-model ranking metrics", normalized)
        self.assertIn("evaluative basis is incomplete for harder neighboring cases", normalized)
        self.assertIn("distinction is between evaluating a judge's output and auditing the dimension set", normalized)
        self.assertIn("A preference label can certify that a decision was correct on one instance", normalized)
        self.assertIn("does not certify that the judge would attend to safety, constraint following, evidence grounding", normalized)
        self.assertIn("observable judgment can be acceptable while the latent evaluation basis is missing dimensions", normalized)
        self.assertIn("A high preference-agreement score is therefore evidence about observed decisions", normalized)
        self.assertIn("not evidence that the judge represented every human-gold evaluation dimension", normalized)
        self.assertIn("unit of analysis is not the rubric text as an artifact", normalized)
        self.assertIn("semantic set of evaluation dimensions it covers", normalized)
        self.assertIn("measure which human evaluation dimensions are omitted", normalized)
        self.assertIn("test when dimension-level coverage-change or recovery wording is supported under the evidence gates", normalized)
        self.assertIn("separate a supported dimension-level coverage-change statement from verbosity, paraphrase diversity", normalized)
        self.assertNotIn("test whether a dimension-recovery statement is supported under the evidence gates", normalized)
        self.assertNotIn("separate a supported dimension-recovery statement from verbosity", normalized)
        self.assertNotIn("measure and test reductions in the subset of human evaluation dimensions omitted", normalized)
        self.assertNotIn("separate genuine blind-spot reduction from verbosity, paraphrase diversity", normalized)
        self.assertIn("not to imitate a preferred wording style", normalized)
        self.assertIn("different from prompt tuning an evaluation-criteria policy for surface-form optimization", normalized)
        self.assertIn("human evaluation dimensions are covered, non-redundant, and decidable", normalized)
        self.assertIn("whether verifiable rewards can supervise semantic dimension elicitation when exact textual answers are unavailable", normalized)
        self.assertIn("not whether reinforcement learning can improve a downstream prompt in an uncontrolled way", normalized)
        self.assertNotIn("measure and repair the subset", normalized)

    def test_method_cites_fixed_embedding_protocol(self) -> None:
        text = read("sections/method.tex")
        normalized = " ".join(text.split())

        self.assertIn("\\texttt{BAAI/bge-large-en-v1.5}", text)
        self.assertIn("\\cite{xiao2023cpack}", text)
        self.assertIn("toy smoke tests use a separate token-overlap embedder", normalized)
        self.assertIn("never used for paper-facing BSC claims", normalized)
        self.assertIn("fixed by protocol before final hard-gold evaluation", normalized)
        self.assertIn("rather than tuned on RubricBench \\texttt{test\\_main}", normalized)
        self.assertIn("threshold sweeps, matched/unmatched human-audit packs, and bootstrap confidence intervals", normalized)
        self.assertIn("paired bootstrap protocol", normalized)
        self.assertIn("invalid, undecidable, irrelevant, hallucinated, or otherwise unverifiable criteria", normalized.lower())
        self.assertIn("receive no validity credit", normalized)
        self.assertIn("verifiability comes from dimension-level semantic coverage", normalized)
        self.assertIn("not from matching a single reference string", normalized)
        self.assertIn("In paper-facing hard-gold evaluation, $G$ denotes human-gold evaluation dimensions", normalized)
        self.assertIn("in training-time reward computation, the hidden criteria are proxy-gold targets from disjoint query pools", normalized)
        self.assertIn("These roles are not interchangeable", normalized)
        self.assertIn("apply an RLVR-style optimization test to an open-ended task", normalized)
        self.assertNotIn("apply RLVR-style optimization to an open-ended task", normalized)
        self.assertIn("without reducing the task to exact-answer prediction", normalized)
        self.assertIn("not prompt tuning with a downstream preference signal", normalized)
        self.assertIn("computed from hidden evaluation dimensions under the relevant evidence role", normalized)
        self.assertIn("human-gold for hard-gold evaluation and proxy-gold during training", normalized)
        self.assertNotIn("computed from hidden human-gold dimensions, intra-output redundancy", normalized)
        self.assertIn("fail-closed validity checks over atomicity, decidability, relevance, and non-hallucination", normalized)
        self.assertIn("cannot receive paper-facing credit merely by changing surface wording", normalized)
        self.assertIn("hidden criteria used by this reward during training are proxy-gold training targets", normalized)
        self.assertIn("not RubricBench \\texttt{test\\_main} human-gold dimensions", normalized)
        self.assertIn("hard-gold split remains an external test", normalized)
        self.assertIn("semantic reward optimization under holdout evidence", normalized)

    def test_all_paper_citations_have_bib_entries(self) -> None:
        section_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((PAPER / "sections").glob("*.tex")))
        section_text += "\n" + (PAPER / "main.tex").read_text(encoding="utf-8")
        bib = (PAPER / "references.bib").read_text(encoding="utf-8")
        cited_keys = {
            key.strip()
            for cite in re.findall(r"\\cite\{([^}]+)\}", section_text)
            for key in cite.split(",")
            if key.strip()
        }
        bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib))

        self.assertTrue(cited_keys)
        self.assertEqual(cited_keys - bib_keys, set())

    def test_main_tex_prefers_official_aaai_submission_style(self) -> None:
        text = read("main.tex")
        normalized = " ".join(text.split())

        self.assertIn("\\documentclass[letterpaper]{article}", text)
        self.assertIn("\\IfFileExists{aaai2026.sty}", text)
        self.assertIn("\\usepackage[submission]{aaai2026}", text)
        self.assertIn("\\IfFileExists{aaai2026.bst}", text)
        self.assertIn("\\bibliographystyle{aaai2026}", text)
        self.assertIn("\\usepackage[margin=1in]{geometry}", text)
        self.assertIn("Anonymous Submission", text)
        self.assertIn("aaaisubmission", normalized)
        self.assertIn("\\input{sections/conclusion}", text)

    def test_conclusion_closes_problem_metric_evidence_loop_without_overclaiming(self) -> None:
        text = read("sections/conclusion.tex")
        normalized = " ".join(text.split())

        self.assertIn("\\section{Conclusion}", text)
        self.assertIn("evaluation dimensions considered before scoring", normalized)
        self.assertIn("evaluation blind spots as a first-class failure mode", normalized)
        self.assertIn("Blind-Spot Coverage makes this failure mode measurable", normalized)
        self.assertIn("test open-ended criteria elicitation under verifiable-reward optimization", normalized)
        self.assertNotIn("can be optimized with verifiable rewards", normalized)
        self.assertIn("minimal motivation gate supports the hard-gold blind-spot diagnostic", normalized)
        self.assertIn("separate from the full real-run C0--C14 readiness matrix", normalized)
        self.assertNotIn("current safe claim is the hard-gold blind-spot diagnostic", normalized)
        self.assertIn("Statements about whether RLVR/GRPO warrants a dimension-level coverage-change or recovery statement", normalized)
        self.assertIn("whether a result is supported by downstream utility", normalized)
        self.assertIn("whether semantic-space coverage evidence is paper-facing", normalized)
        self.assertIn("permitted only when the hard-gold, contamination, ablation, downstream, dimension-transition, and visualization gates pass", normalized)
        self.assertIn("a BSC coverage change alone remains a metric result", normalized)
        self.assertIn("until redundancy, hallucination, and held-out downstream utility evidence support", normalized)
        self.assertIn("not a claim of a better policy by construction", normalized)
        self.assertIn("testing when dimension-level coverage-change or recovery wording about evaluation blind spots is supported", normalized)
        self.assertNotIn("testing whether dimension-recovery statements about evaluation blind spots are supported", normalized)
        self.assertNotIn("Claims that RLVR/GRPO reduces blind spots", text)
        self.assertNotIn("Statements about whether RLVR/GRPO reduces blind spots", normalized)
        self.assertNotIn("testing reductions in evaluation blind spots", normalized)
        self.assertNotIn("testing repairs for evaluation blind spots", normalized)
        self.assertNotIn("we prove", text.lower())
        self.assertNotIn("significantly improves", text.lower())
        self.assertNotIn("state-of-the-art", text.lower())

    def test_limitations_state_metric_and_contamination_boundaries(self) -> None:
        text = read("sections/limitations.tex")
        normalized = " ".join(text.split())

        self.assertIn("recall-style diagnostic over evaluation dimensions", normalized)
        self.assertIn("not a full measure of final judge correctness", normalized)
        self.assertIn("expose missing evaluative dimensions and test attempted reductions", normalized)
        self.assertIn("apparent coverage change is only metric-support unless the RewardBench, RewardBench-2, and JudgeBench rows pass", normalized)
        self.assertNotIn("coverage gain", normalized)
        self.assertNotIn("coverage increase", text)
        self.assertIn("metric-only BSC evidence rather than downstream utility", normalized)
        self.assertNotIn("test repair attempts", normalized)
        self.assertIn("not to replace held-out judge-utility evaluation", normalized)
        self.assertIn("coverage changes rather than over-interpreting one threshold", normalized)
        self.assertIn("coverage change should be described as dimension-level recovery only after", normalized)
        self.assertIn("per-gold dimension audit shows recovered and lost dimensions", normalized)
        self.assertIn("query-aligned inputs, fixed BGE embeddings, verifier valid-flag filtering", normalized)
        self.assertIn("downstream utility gates", normalized)
        self.assertIn("Aggregate coverage alone cannot establish dimension-level recovery", normalized)
        self.assertIn("lost dimensions, duplicate paraphrases, or newly invalid criteria", normalized)
        self.assertIn("normalized query-level exact match", normalized)
        self.assertIn("do not rule out every paraphrased, templated, or semantically near-duplicate query", normalized)
        self.assertIn("zero exact-overlap count is not a clean-isolation certificate", normalized)
        self.assertIn("artifact\\_status=blocked", text)
        self.assertIn("overlap\\_status=not\\_auditable", text)
        self.assertIn("missing C0 evidence, not a clean holdout claim", normalized)
        self.assertIn("annotation-pack builder for matched and unmatched dimension pairs", normalized)
        self.assertIn("only after human labels are filled and audited", normalized)
        self.assertIn("reduce threshold and embedding-model arbitrariness", normalized)
        self.assertIn("do not eliminate the need for downstream utility and human-audit evidence", normalized)
        self.assertIn("avoiding stronger claims", normalized)

    def test_paper_sections_avoid_unsupported_acceptance_or_sota_claims(self) -> None:
        section_dir = PAPER / "sections"
        combined = "\n".join(path.read_text(encoding="utf-8") for path in sorted(section_dir.glob("*.tex")))
        lowered = combined.lower()

        banned_phrases = [
            "state-of-the-art",
            "state of the art",
            "sota",
            "guarantee acceptance",
            "guaranteed acceptance",
            "drastically",
            "significantly improves",
            "significantly outperforms",
        ]
        for phrase in banned_phrases:
            self.assertNotIn(phrase, lowered)

    def test_public_facing_docs_avoid_unsupported_positive_overclaims(self) -> None:
        docs = {
            "README.md": (ROOT / "README.md").read_text(encoding="utf-8").lower(),
            "STORYLINE.md": read("STORYLINE.md").lower(),
            "AAAI_ACCEPTANCE_PLAYBOOK.md": read("AAAI_ACCEPTANCE_PLAYBOOK.md").lower(),
            "RL_EXECUTION_PLAN.md": read("RL_EXECUTION_PLAN.md").lower(),
        }
        banned_positive_claims = [
            "we prove",
            "we show that sft+grpo",
            "guarantee acceptance",
            "guaranteed acceptance",
            "achieving sota",
            "achieves sota",
            "state-of-the-art results",
            "significantly improves",
            "significantly outperforms",
            "promote empirical claims",
            "promoting empirical claims",
            "promoting",
            "do not promote",
            "engineering choice",
            "engineering scaling choice",
        ]
        for name, text in docs.items():
            with self.subTest(doc=name):
                for phrase in banned_positive_claims:
                    self.assertNotIn(phrase, text)

    def test_public_facing_docs_avoid_unverified_repair_metric_framing(self) -> None:
        docs = {
            "README.md": (ROOT / "README.md").read_text(encoding="utf-8").lower(),
            "STORYLINE.md": read("STORYLINE.md").lower(),
            "AAAI_ACCEPTANCE_PLAYBOOK.md": read("AAAI_ACCEPTANCE_PLAYBOOK.md").lower(),
            "RL_EXECUTION_PLAN.md": read("RL_EXECUTION_PLAN.md").lower(),
        }
        banned_repair_framing = [
            "evaluate_blindspot_repair.py",
            "scripts/evaluate_blindspot_repair.py",
            "blind-spot repair rate",
            "repair rate",
            "repairrate",
            "repair attempts",
            "not repair",
            "criteria-text improvement",
            "criteria-text improvement problem",
        ]
        allowed_implementation_terms = [
            "outputs/matrix_real/dimension_transition",
            "<dimension_transition_dir>",
        ]
        for name, text in docs.items():
            stripped = text
            for allowed in allowed_implementation_terms:
                stripped = stripped.replace(allowed, "")
            with self.subTest(doc=name):
                for phrase in banned_repair_framing:
                    self.assertNotIn(phrase, stripped)

    def test_synced_paper_assets_avoid_stale_strong_reduction_framing(self) -> None:
        docs = synced_public_asset_texts()
        banned_stale_phrases = [
            "can be optimized with verifiable rewards",
            "it tests whether verifiable blind-spot coverage can be optimized",
            "test whether training reduces dimension-level blind spots",
            "test whether rlvr/grpo can reduce",
            "test whether rlvr/grpo can optimize",
            "coverage change 是否支持 evaluation blind-spot reduction",
            "判定是否可以称为 blind-spot reduction",
            "claims that rlvr/grpo reduces",
            "testing reductions in evaluation blind spots",
            "measure and test reductions",
            "separate genuine blind-spot reduction",
            "apply rlvr-style optimization to an open-ended task",
            "can reduce systematic dimension gaps",
            "coverage improvement",
            "utility improvements",
            "expected hypotheses",
            "hard-gold domain rubrics",
            "generate method rubrics",
            "promote empirical claims",
            "does not by itself promote",
            "does not promote any paper claim",
            "promoting visualization claims",
            "promoting",
            "do not promote",
            "engineering choice",
            "engineering scaling choice",
        ]
        self.assertGreater(len(docs), 0)
        for name, text in docs.items():
            with self.subTest(doc=name):
                for phrase in banned_stale_phrases:
                    self.assertNotIn(phrase, text)

    def test_generated_reviewer_artifacts_avoid_legacy_repair_artifact_names(self) -> None:
        roots = [
            ROOT / "outputs" / "paper_artifacts",
            ROOT / "outputs" / "result_card",
            ROOT / "outputs" / "submission_readiness",
            ROOT / "outputs" / "rebuttal_pack",
            ROOT / "outputs" / "dashboard",
            ROOT / "outputs" / "matrix_smoke",
            ROOT / "outputs" / "minimal_claim" / "base",
            PAPER / "asset_index",
        ]
        banned_path_fragments = [
            "blindspot_repair",
            "repair_smoke",
            "repair_summary.json",
            "repair_per_item.csv",
            "repair_by_category.csv",
            "repair_gold_items.jsonl",
            "repair_table.csv",
            "repair_table.md",
            "repair_table.tex",
            "result_card.verify",
            "outputs/submission_gap_report",
        ]
        banned_text_fragments = [
            "repaired_gold",
            '"repaired"',
            "repair_summary.json",
            "repair_per_item.csv",
            "repair_by_category.csv",
            "repair_gold_items.jsonl",
            "outputs/matrix_real/blindspot_repair",
            "outputs/matrix_smoke/blindspot_repair",
            "outputs/submission_gap_report",
        ]
        text_suffixes = {".csv", ".json", ".jsonl", ".md", ".tex", ".txt"}
        checked_files = 0

        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(ROOT).as_posix().lower()
                for fragment in banned_path_fragments:
                    self.assertNotIn(fragment, relative)
                if not path.is_file() or path.suffix.lower() not in text_suffixes:
                    continue
                checked_files += 1
                text = path.read_text(encoding="utf-8").lower()
                for fragment in banned_text_fragments:
                    self.assertNotIn(fragment, text)

        self.assertGreater(checked_files, 0)

    def test_public_facing_docs_avoid_generator_framing(self) -> None:
        docs = {
            "README.md": (ROOT / "README.md").read_text(encoding="utf-8").lower(),
            "STORYLINE.md": read("STORYLINE.md").lower(),
            "AAAI_ACCEPTANCE_PLAYBOOK.md": read("AAAI_ACCEPTANCE_PLAYBOOK.md").lower(),
            "RL_EXECUTION_PLAN.md": read("RL_EXECUTION_PLAN.md").lower(),
        }
        banned_framing = [
            "rubric-generator",
            "rubric generator",
            "better rubric",
            "rubric generation",
            "evaluation-criteria generation",
            "generator quality",
            "proxy generation",
            "rubric items",
            "criteria text generation",
            "rubric_generator_sft",
            "rubric_generator_rl",
        ]
        allowed_interface_terms = ["served_generators", "configs/generators", "evaluation_criteria_policy_sft", "evaluation_criteria_policy_rl"]
        for name, text in docs.items():
            stripped = text
            for allowed in allowed_interface_terms:
                stripped = stripped.replace(allowed, "")
            with self.subTest(doc=name):
                for phrase in banned_framing:
                    self.assertNotIn(phrase, stripped)

    def test_storyline_uses_current_aaai_claim_discipline(self) -> None:
        text = read("STORYLINE.md")
        normalized = " ".join(text.split())
        unquoted = normalized.replace("> ", "")

        for value in ["0.3692", "0.6308", "0.5785--0.6823", "0.0768", "0.1227"]:
            self.assertIn(value, text)
        self.assertIn("Do not frame the paper as", text)
        self.assertIn("Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation", text)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Covering Evaluation Blind Spots", text)
        self.assertIn("test when", text)
        self.assertIn("use evidence-gated experiments to test when RLVR/GRPO warrants a dimension-level recovery statement", unquoted)
        self.assertIn("Formulate an evidence-gated RLVR/GRPO optimization test", text)
        self.assertNotIn("use evidence-gated experiments to test whether RLVR/GRPO can reduce these dimension-level blind spots", unquoted)
        self.assertNotIn("Test whether RLVR/GRPO can optimize", text)
        self.assertIn("minimal motivation gate supports the 100-example diagnostic blind-spot", normalized)
        self.assertIn("for Section 2 only; trained-method claims wait for the real C0-C14 gates", normalized)
        self.assertNotIn("The current safe claim is the 100-example diagnostic blind-spot finding", text)
        self.assertIn("only when C12/C14 show dimension-level recovery", normalized)
        self.assertIn("rather than a completed reduction result", normalized)
        self.assertIn("aligned hard-gold inputs", normalized)
        self.assertIn("Unsafe Claims Until Real Gates Pass", text)
        self.assertIn("SFT+GRPO is already paper-facing on RubricBench `test_main`", text)
        self.assertIn("BlindSpot-RL already has paper-facing RewardBench/JudgeBench/RewardBench-2 utility evidence", text)
        self.assertNotIn("coverage improvement", normalized)
        self.assertNotIn("utility improvements", normalized)
        self.assertIn("C2/C3/C4/C5/C6/C7/C9/C10/C12/C13/C14", text)
        self.assertIn("Gate-to-Claim Map", text)
        self.assertIn("C0 | Zero-overlap hard-gold, proxy-train, and downstream audits", text)
        self.assertIn("C4/C9/C10 | RewardBench, JudgeBench, and RewardBench-2 utility", text)
        self.assertIn("C12 | Query-aligned per-gold dimension transition audit", text)
        self.assertIn("C14 | SFT-only versus SFT+GRPO under the same protocol", text)
        self.assertIn("The central method sentence requires the gates jointly", normalized)
        self.assertIn("Use the semantic-space figure to audit, under C13", normalized)
        self.assertIn(
            "show a reportable nearest-gold region-coverage change relative to SFT-only without local collapse",
            normalized,
        )
        self.assertIn("nearest-gold similarity, and dispersion must support the visual impression", normalized)
        self.assertNotIn("nearest-gold regions after SFT+GRPO", normalized)
        self.assertIn("overlap_query_count == 0", text)
        self.assertNotIn("Show that RLVR/GRPO can optimize", text)
        self.assertIn("API scorer audit", text)
        self.assertNotIn("Achieving SOTA", text)

    def test_playbook_c0_includes_teacher_verifier_provenance_chain(self) -> None:
        text = read("AAAI_ACCEPTANCE_PLAYBOOK.md")
        normalized = " ".join(text.split())

        self.assertIn("C0 is content-bound, not path-bound", text)
        self.assertIn("teacher_rubrics_filtered_report.json", text)
        self.assertIn("raw teacher criteria", normalized)
        self.assertIn("filtered teacher criteria", normalized)
        self.assertIn("Meta-Verifier provider", text)
        self.assertIn("SFT preflight report", text)
        self.assertIn("meta-verifier budget report", text)
        self.assertIn("verifier-filtered output hash", normalized)
        self.assertIn("proxy_gold_build_report.json:input_sha256", text)
        self.assertIn("proxy_gold_verl_report.json", text)
        self.assertNotIn("drastically reduces", text)
        self.assertNotIn("significantly improve pairwise", text)
        self.assertNotIn("Cov \u2248 60%", text)
        self.assertNotIn("Blind \u2248 40%", text)
        self.assertIn("Treat the contamination-aware, evidence-gated pipeline as the experiment protocol", normalized)
        self.assertIn("not as the central research contribution", normalized)
        self.assertIn("Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation", text)
        self.assertNotIn("BlindSpot-RL: Verifiable Semantic Rewards for Covering Evaluation Blind Spots", text)
        self.assertIn("hidden gold criteria", normalized)
        self.assertNotIn("hidden gold rubrics", normalized)
        self.assertIn("generated criteria dimensions", normalized)
        self.assertNotIn("generated rubric dimensions", normalized)
        self.assertIn("configured to produce SVG/PDF plus point-level CSV/JSON artifacts for C13 audit", normalized)
        self.assertNotIn("Produces a paper-facing SVG plus point-level CSV/JSON for audit", text)
        self.assertIn("nearest-gold category coverage, nearest-gold cluster coverage, and dispersion", normalized)
        self.assertNotIn("Provide a contamination-aware, evidence-gated experiment pipeline", text)

    def test_playbook_lists_latex_ready_semantic_space_assets(self) -> None:
        text = read("AAAI_ACCEPTANCE_PLAYBOOK.md")
        normalized = " ".join(text.split())

        for name in [
            "semantic_space.svg",
            "semantic_space.pdf",
            "semantic_space_points.csv",
            "semantic_space_summary.json",
        ]:
            self.assertIn(name, text)
        self.assertIn("requests UMAP", text)
        self.assertIn("deterministic PCA fallback", text)
        self.assertIn("method-level gold-category coverage", text)
        self.assertIn("nearest-gold category coverage", text)
        self.assertIn("nearest-gold gold-cluster coverage", text)
        self.assertIn("generated-criteria dispersion", text)
        self.assertIn("nearest-gold similarity", text)
        self.assertIn("AAAI Formatting Gate", text)
        self.assertIn("aaai2026.sty", text)
        self.assertIn("aaai2026.bst", text)

    def test_playbook_distinguishes_minimal_and_real_evidence_claim_ids(self) -> None:
        text = read("AAAI_ACCEPTANCE_PLAYBOOK.md")
        normalized = " ".join(text.split())

        self.assertIn("minimal diagnostic gates for Section 2 only", normalized)
        self.assertIn("not the real-run method-result gates", normalized)
        self.assertIn("Minimal C1 is `safe_to_claim`", text)
        self.assertIn("Minimal C2 is `safe_to_claim`", text)
        self.assertIn("In the real paper matrix, C2 is reserved for the SFT+GRPO hard-gold coverage", normalized)
        self.assertIn("The threshold robustness role is C6 in the real matrix", normalized)
        self.assertNotIn("- C2 is `safe_to_claim`", text)
        self.assertIn("use evidence-gated experiments to test when RLVR/GRPO warrants a dimension-level recovery statement", normalized)
        self.assertIn("Formulate an evidence-gated RLVR/GRPO optimization test", text)
        self.assertNotIn("use evidence-gated experiments to test whether RLVR/GRPO can reduce dimension-level blind spots", normalized)
        self.assertIn("scripts/check_latex_compile.py", text)
        self.assertIn("outputs/submission_readiness/latex_compile_report.json", text)
        self.assertIn("AAAI LaTeX Compile", text)
        self.assertIn("anonymous author line", normalized)
        self.assertIn("`aaai2026` bibliography style", text)
        self.assertIn("`max_pages=8` limit", text)
        self.assertIn("submission", text)
        self.assertIn("official template", text)
        self.assertNotIn("use RLVR/GRPO to repair", text)
        self.assertIn("Every paper-facing downstream row must use the API/model scorer", text)
        self.assertIn("Gate-to-Claim Map", text)
        self.assertIn("planned experiments or planned hypotheses", text)
        self.assertIn("Reviewer concern | Evidence that answers it | If evidence is missing", text)
        self.assertIn("Write criteria-elicitation and aggregate coverage-change language only", text)
        self.assertIn("Keep BSC as metric-support only, not downstream judge-utility support", text)
        self.assertIn("No trained-method row is paper-facing, even if metric values look favorable", text)
        self.assertIn("Report proxy-gold supervision evidence rather than RLVR-stage support", text)
        self.assertIn("Treat the semantic-space figure as illustrative only", text)
        self.assertIn("C13 validates `semantic_space_points.csv`, `semantic_space_summary.json`, SVG/PDF outputs", text)
        self.assertNotIn("expected hypotheses", text)
        self.assertIn("SFT+GRPO is already paper-facing on RubricBench test_main", text)
        self.assertIn("BlindSpot-RL already has paper-facing RewardBench / RewardBench-2 / JudgeBench utility evidence", text)
        self.assertNotIn("coverage improvement", normalized)
        self.assertNotIn("utility improvements", normalized)
        self.assertIn("C0 | Zero-overlap RubricBench `test_main`, proxy-train, RewardBench, JudgeBench, and RewardBench-2 audits", text)
        self.assertIn("C4/C9/C10 | RewardBench, JudgeBench, and RewardBench-2 utility", text)
        self.assertIn("C7 | Reward-component variants plus no-verifier-filtering ablation", text)
        self.assertIn("C13 | Semantic-space point CSV, JSON summary, and rendered SVG/PDF", text)
        self.assertIn("A row can look\nbetter numerically and still remain non-paper-facing", text)
        self.assertIn("three evidence families pass\ntogether", text)
        self.assertIn("hard-gold BSC evidence shows a reportable coverage change, redundancy\nand hallucination do not materially worsen", text)
        self.assertIn("A BSC coverage change by itself is only a metric result", normalized)
        self.assertIn("budget report whose `ok` field is `true`", text)
        self.assertIn("budget report must bind the exact `downstream_eval.jsonl` input", text)
        self.assertIn("their SHA256 hashes", text)
        self.assertIn("pairwise or multi-candidate call contract", text)
        self.assertIn("stale `ok=true` report", normalized)
        self.assertIn("downstream `summary.json` must repeat the exact joined input", text)
        self.assertIn("current input/provider SHA256 hashes", text)
        self.assertIn("benchmark format, scorer provider,\n     budget report, and scorer contract", text)
        self.assertIn("compare the summary hashes against the budget-report\n     contract hashes", text)
        self.assertIn("matching paths alone are not enough", text)
        self.assertIn("experiment summarizer and artifact exporter must also enforce", text)
        self.assertIn("downstream accuracy/tie/margin cells are paper-facing only", text)
        self.assertIn("downstream_paper_claim_eligible=true", text)
        self.assertIn("final `main_table.csv`", text)
        self.assertIn("not_paper_eligible", text)
        self.assertIn("calls_per_record_per_provider", text)
        self.assertIn("candidate multiplier", normalized)
        self.assertIn("keyword scorer results are smoke-test evidence only", normalized)
        self.assertIn("summary.json:n >= 100", text)
        self.assertIn("not just the holdout file size", text)
        self.assertIn("reward-component variant identities", text)
        self.assertIn("`full`, `no_red`,", text)
        self.assertIn("`no_valid`, `no_verifier`, `cov_only`", text)
        normalized = " ".join(text.split())
        self.assertIn("component weights for coverage, validity", normalized)
        self.assertIn("and redundancy", text)
        self.assertIn("mean generated-criteria ratio", text)
        self.assertIn("length expansion alone", text)
        self.assertIn("`no_red` gains that come with duplicate criteria", text)
        self.assertIn("`no_valid` or\n     `no_verifier_filter` gains that raise hallucination are not faithful\n     coverage", text)
        self.assertIn("SFT-only parity means the RL stage is not supported", normalized)
        self.assertIn("weak `multi_teacher_union` row means teacher union is only an uncontrolled data-construction choice", normalized)
        self.assertIn("C5 covers single-teacher vs multi-teacher union", text)
        self.assertIn("`multi_teacher_union` row", text)
        self.assertIn("data/processed/teacher_rubrics_raw.jsonl", text)
        self.assertIn("data/processed/rubricbench_gold.jsonl", text)
        self.assertIn("at least two single-teacher", text)
        self.assertIn("rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json", text)
        self.assertIn("clean_proxy_train_vs_hard_gold_audit.json", text)
        self.assertIn("clean_proxy_train_vs_rewardbench_downstream_audit.json", text)
        self.assertIn("clean_proxy_train_vs_judgebench_downstream_audit.json", text)
        self.assertIn("clean_proxy_train_vs_rewardbench2_downstream_audit.json", text)
        self.assertIn("rewardbench2_downstream_holdout_contamination.json", text)
        self.assertIn("pre-SFT audits", text)
        self.assertIn("do not replace the final C0 holdout audits", text)
        self.assertIn("overlap_query_count == 0", text)
        self.assertIn("artifact_status=blocked", text)
        self.assertIn("overlap_status=not_auditable", text)
        self.assertIn("missing C0 evidence, not a clean-isolation result", normalized)
        self.assertIn("not used for\n    SFT, proxy criteria elicitation, reward tuning, prompt selection, verifier\n    calibration, hyperparameter tuning, or checkpoint selection", text)
        self.assertIn("Paper-facing\n    method-result claims must bind the exact training inputs, holdout audit\n    reports, and evaluation outputs", text)
        self.assertIn("keep the embedding\n    model/thresholds fixed before final evaluation", text)
        self.assertIn("validated", text)
        self.assertIn("do not tune thresholds on RubricBench `test_main`", text)
        self.assertIn("fixed `coverage_tau=0.75` / `redundancy_tau=0.85` protocol", text)
        self.assertIn("paired bootstrap confidence intervals under the same BGE model", text)
        self.assertIn("scripts/build_bsc_human_audit_pack.py", text)
        self.assertIn("scripts/summarize_bsc_human_audit_labels.py", text)
        self.assertIn("min-auto-matched-human-match-rate", text)
        self.assertIn("min-auto-unmatched-confirmation-rate", text)
        self.assertIn("annotation_pack_ready", text)
        self.assertIn("human_labels_completed", text)
        self.assertIn("human_audit_complete", text)
        self.assertIn("human_label_summary.json", text)
        self.assertIn("ok == true", text)
        self.assertIn("at least 50 completed labels", text)
        self.assertIn("uncertain rate at most 0.2", text)
        self.assertIn("auto-matched human-match rate", text)
        self.assertIn("auto-unmatched confirmation rate", text)
        self.assertIn("C6-passing human audit", text)
        self.assertIn("C6-passing matched/unmatched human-audit summaries", text)
        self.assertIn("C6-gated human-audit checks", text)
        self.assertIn("C6 strict-gate human-audit summaries", text)
        self.assertNotIn("a completed human audit of matched/unmatched pairs", text)
        self.assertNotIn("a completed matched/unmatched human audit", text)
        self.assertIn("nearest-gold categories, and nearest-gold semantic clusters as SFT-only", text)
        self.assertIn("A visually broader plot is only", text)
        self.assertIn("C13 validates the exact point CSV schema", text)
        self.assertIn("including SFT+GRPO rows", text)
        self.assertIn("non-empty\n     nearest-gold audit fields", text)
        self.assertIn("semantic_space_points.csv", text)
        self.assertIn("semantic_space_summary.json", text)

    def test_rl_execution_plan_tracks_current_real_run_protocol(self) -> None:
        text = read("RL_EXECUTION_PLAN.md")
        normalized = " ".join(text.split())

        self.assertIn("1,785 clean proxy-train", normalized)
        self.assertIn("data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl", text)
        self.assertIn("data/processed/blindspot_sft.jsonl", text)
        self.assertIn("data/processed/proxy_gold_verl.parquet", text)
        self.assertIn('"rl_data_report": "outputs/sft_data/proxy_gold_verl_report.json"', text)
        self.assertIn("outputs/training_commands/training_done.json", text)
        self.assertIn("served_generators", text)
        self.assertIn("base", text)
        self.assertIn("sft_only", text)
        self.assertIn("sft_rl", text)
        self.assertIn("R = 1.0 * R_cov", text)
        self.assertIn("+ 0.5 * R_valid", text)
        self.assertIn("- 0.5 * R_red", text)
        self.assertIn("SFT-only vs SFT+GRPO", text)
        self.assertIn("RewardBench、JudgeBench、RewardBench-2 downstream utility", text)
        self.assertIn("verifiable semantic rewards for evaluation blind spots", text)
        self.assertIn("能稳定生成 evaluation criteria 的 policy", text)
        self.assertIn("不是把论文卖点写成“更顺滑的 criteria policy”", text)
        self.assertIn("检验一个研究假设", text)
        self.assertIn("verifiable semantic reward 产生可报告的 human-gold coverage change", text)
        self.assertIn("RL 阶段测试 hard blind-spot coverage 是否出现可报告 coverage change", text)
        self.assertIn("`dimension-level recovery` 只能在 C12/C14、downstream utility 和 human-audit gates 通过后作为结论写入", text)
        self.assertIn("coverage 是否可靠对应 human-gold evaluation dimensions", text)
        self.assertIn("在 dev 集上具备可审计的格式、有效性和覆盖变化", text)
        self.assertIn("候选排序 sanity check", text)
        self.assertIn("reward 是否出现预期方向的 sanity change", text)
        self.assertIn("redundancy 是否 materially 偏移", text)
        self.assertIn("是否能描述为 dimension-level recovery，必须由", text)
        self.assertIn("C12/C14 的 per-gold dimension-transition audit", text)
        self.assertIn("SFT+GRPO 相对 SFT-only 出现预注册的 BSC coverage change", text)
        self.assertIn("C12 显示 recovered human-gold dimensions 多于 lost dimensions", text)
        self.assertIn("都通过 API/model scorer、budget report、join audit 和", text)
        self.assertIn("`paper_claim_eligible=true` 后，才能作为 judge-utility 支撑", text)
        self.assertIn("未通过 C12/C14 时，只能写 coverage change", text)
        self.assertIn("不能写 dimension-level recovery", text)
        self.assertIn("It formulates an evidence-gated optimization test for verifiable", text)
        self.assertIn("blind-spot coverage in open-ended evaluation-criteria elicitation", text)
        self.assertNotIn("It tests whether verifiable blind-spot coverage can be optimized", text)
        self.assertIn("dimension-recovery claims are permitted only after hard-gold", text)
        self.assertIn("human audit 和 semantic-space gates 同时支持", normalized)
        self.assertIn("coverage change 是否具备写成 evaluation dimension-recovery statement", text)
        self.assertIn("判定是否具备写成 dimension-level recovery\nstatement 的证据", text)
        self.assertNotIn("coverage change 是否支持 evaluation blind-spot reduction", text)
        self.assertNotIn("判定是否可以称为 blind-spot reduction", text)
        self.assertIn("是否写成 paper-facing coverage change 或\ndimension-level recovery 取决于 evidence gate", text)
        self.assertIn("用于判定 coverage change 是否仍成立", text)
        self.assertIn("先验证 reward 与 human-gold dimensions 的对齐", text)
        self.assertIn("verifier filtering 是否让 proxy-gold 更可审计、更少无效项", text)
        self.assertIn("产生可报告 semantic coverage change", text)
        self.assertIn("bootstrap-CI 支撑的\n  hard-gold BSC coverage change", text)
        self.assertIn("`coverage_per_generated_criterion`、redundancy 和 hallucination", text)
        self.assertNotIn("verifiable semantic reward 覆盖更多 human-gold 维度", text)
        self.assertNotIn("在 dev 集上明显优于 base model", text)
        self.assertNotIn("先验证训练链路稳定，不追求最终 paper 效果", text)
        self.assertNotIn("reward 是否上升", text)
        self.assertNotIn("redundancy 是否上升", text)
        self.assertNotIn("coverage 上升但 validity 明显下降", text)
        self.assertNotIn("SFT+GRPO 的 BSC coverage 高于 SFT-only", text)
        self.assertNotIn("`SFT+GRPO BSC > SFT-only BSC`", text)
        self.assertNotIn("downstream utility 不下降", text)
        self.assertNotIn("RL 阶段测试 hard blind-spot coverage 是否能在冗余、幻觉和无效项受控时提升", text)
        self.assertNotIn("verifier filtering 是否提升 proxy-gold 质量", text)
        self.assertNotIn("在控制冗余和幻觉时提高 semantic coverage", text)
        self.assertNotIn("证明提升不是靠“生成更多 criteria”", text)
        self.assertNotIn("先证明 reward 与 human-gold dimensions 对齐", text)
        self.assertNotIn("再证明 RL 训练稳定", text)
        self.assertNotIn("显著升高", text)
        self.assertNotIn("会生成合理 rubric", text)
        self.assertNotIn("优先修复", text)
        self.assertNotIn("RL 解决 hard blind spots", text)
        self.assertNotIn("coverage 是否真的代表“补盲”", text)
        self.assertNotIn("验证 RL 是否真的修复 blind spots", text)
        self.assertNotIn("修复更多 blind spots", text)
        self.assertNotIn("最后再证明它真的在修 blind spots", text)
        self.assertNotIn("R_hard", text)
        self.assertNotIn("dynamic sampling", text.lower())
        self.assertNotIn("hard replay", text.lower())
        self.assertNotIn("reward_sanity_report.md", text)
        self.assertNotIn("repair evaluation blind spots", text)


def read(relative_path: str) -> str:
    return (PAPER / relative_path).read_text(encoding="utf-8")


def synced_public_asset_texts() -> dict[str, str]:
    docs: dict[str, str] = {}
    asset_roots = [PAPER / "asset_index", ROOT / "outputs" / "paper_artifacts"]
    text_suffixes = {".md", ".tex", ".txt", ".json", ".csv"}
    for asset_root in asset_roots:
        if not asset_root.exists():
            continue
        for path in sorted(asset_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            docs[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8").lower()
    return docs


if __name__ == "__main__":
    unittest.main()
