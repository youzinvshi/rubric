# AAAI Submission Playbook for BlindSpot-RL

## Core Positioning

Do not frame the paper as an engineering effort to polish criteria text. Frame it as:

> LLM evaluators suffer from systematic blind spots in the evaluation dimensions they consider. We introduce Blind-Spot Coverage (BSC), a verifiable semantic reward for open-ended criteria elicitation, and use evidence-gated experiments to test when RLVR/GRPO warrants a dimension-level recovery statement while controlling redundancy and hallucination.

Current paper title:

> Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation

The paper should make three research contributions:

1. Move LLM-as-a-Judge upstream from final score accuracy to missing evaluation dimensions.
2. Define BSC as a computable and verifiable semantic reward for open-ended criteria elicitation.
3. Formulate an evidence-gated RLVR/GRPO optimization test for a subjective, semantic evaluation-criteria task, not only deterministic math/code/QA tasks.

## Evidence Already Safe to Claim

Source: `paper/asset_index/evidence_matrix.md`, `outputs/minimal_claim/base/result_card/result_card.md`.

- These are minimal diagnostic gates for Section 2 only, not the real-run
  method-result gates in `outputs/evidence_real/evidence_matrix.json`.
- Minimal C1 is `safe_to_claim`: a single-model evaluation-criteria policy
  leaves measurable blind spots against human-gold evaluation dimensions.
- Minimal C2 is `safe_to_claim`: the blind-spot diagnostic is auditable across
  semantic threshold settings.
- In the real paper matrix, C2 is reserved for the SFT+GRPO hard-gold coverage
  change and is not safe until the real matrix artifacts pass. The threshold
  robustness role is C6 in the real matrix.
- On 100 RubricBench hard-gold examples:
  - Mean coverage: `0.3692`
  - Mean blind: `0.6308`
  - Blind 95% CI: `[0.5785, 0.6823]`
  - Mean redundancy: `0.0768`
  - Mean hallucination: `0.1227`
- These are frozen Section 2 diagnostic numbers. The minimal-claim evidence
  matrix must check the exact snapshot, not only that blind spot exceeds a
  loose non-triviality threshold.
- Blind-spot attribution shows strongest gaps in:
  - constraint following: blind `0.8226`
  - intent/reasoning: blind `0.7692`
  - completeness: blind `0.6219`
- Budget curve shows longer outputs help only partially and saturates at coverage `0.3692`; this supports the Section 2 motivation that the issue is not just too few criteria items.
- Semantic-space visualization infrastructure is implemented by `scripts/build_semantic_space_visualization.py`; it emits `semantic_space.svg`, `semantic_space.pdf`, `semantic_space_points.csv`, and `semantic_space_summary.json`. C13 checks method-level gold-category coverage, nearest-gold category coverage, nearest-gold gold-cluster coverage, generated-criteria dispersion, and nearest-gold similarity before visualization claims are written.

## Code Evidence to Cite in Writing

- BSC reward: `src/blindspot_rl/reward_bsc.py`
  - `coverage_reward`: semantic coverage of gold dimensions.
  - `redundancy_penalty`: duplicate-pair penalty.
  - `validity_reward` / `hallucination_rate`: verifier-backed validity.
  - `compute_metrics`: scalar BSC reward.
  - `compute_category_balanced_reward`: long-tail category balancing.
- GRPO/verl hook: `src/blindspot_rl/verl_reward.py`
  - Exposes `compute_score`, resolves hidden gold criteria, applies BSC reward with rule-verifier or response-aligned validity flags.
- Meta-verifier: `src/blindspot_rl/meta_verifier.py`
  - Fail-closed rule/API verification.
  - Domain-aware verifier policies for HealthBench, BEIR/NQ, WritingBench, IFBench, RewardBench.
- Multi-teacher generation: `scripts/generate_teacher_rubrics.py`
  - Domain-specific prompts and auditable `generation_failed=true` rows.
- Proxy-gold construction: `scripts/build_sft_data.py`
  - Teacher union, semantic dedupe, min-teacher/min-rubric gates.
- Semantic-space figure: `scripts/build_semantic_space_visualization.py`
  - Embeds gold and generated criteria dimensions.
  - Is configured to produce SVG/PDF plus point-level CSV/JSON artifacts for C13 audit.
- Dimension-transition audit:
  - Compares baseline and candidate outputs at the human-gold dimension level.
  - Emits `transition_summary.json`, `transition_per_item.csv`, `transition_by_category.csv`, and `transition_gold_items.jsonl`.
  - Reports recovered/lost dimension rates, category-level transition summaries, and per-gold audit rows.
  - Counts invalid generated criteria as non-covering when `valid_flags` are present.
- Verifier-filter ablation: `scripts/run_verifier_filter_ablation.py`
  - Compares proxy-gold teacher unions built from raw teacher criteria vs verifier-filtered teacher criteria.
  - Emits `verifier_filter_ablation.csv/json/md` plus per-item audit rows.
- Data isolation:
  - `data/processed/dataset_manifest.json`
  - `data_architecture_summary.md`
  - `scripts/filter_holdout_contamination.py`
  - `scripts/audit_holdout_contamination.py`
  - `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_filter.json`
  - `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench_holdout_filter.json`
  - `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_judgebench_holdout_filter.json`
  - `outputs/contamination_audit/rewardbench_pref_sft_proxy_train_rewardbench2_holdout_filter.json`
  - `outputs/contamination_audit/clean_proxy_train_vs_hard_gold_audit.json`
  - `outputs/contamination_audit/clean_proxy_train_vs_rewardbench_downstream_audit.json`
  - `outputs/contamination_audit/clean_proxy_train_vs_judgebench_downstream_audit.json`
  - `outputs/contamination_audit/clean_proxy_train_vs_rewardbench2_downstream_audit.json`
  - `outputs/contamination_audit/hard_gold_holdout_contamination.json`
  - `outputs/contamination_audit/rewardbench_downstream_holdout_contamination.json`
  - `outputs/contamination_audit/judgebench_downstream_holdout_contamination.json`
  - `outputs/contamination_audit/rewardbench2_downstream_holdout_contamination.json`

The `clean_proxy_train_vs_*_audit.json` files are pre-SFT audits for the
current 1,785-example clean RewardBench proxy-train split only. They verify
zero normalized-query overlap against RubricBench `test_main`, RewardBench
holdout, JudgeBench, and RewardBench-2. They are useful early warning evidence,
but they do not replace the final C0 holdout audits, which must run after
`blindspot_sft`, `proxy_gold`, and `proxy_gold_verl` exist.

C0 is content-bound, not path-bound. Filter reports must record
`holdout_sha256`, `input_sha256`, and `output_sha256`; final contamination
audits must record the holdout SHA256 and every training artifact SHA256. The
real evidence matrix checks that each filter output hash feeds the next filter
input hash, and that the final clean proxy-train hash is the exact artifact
inspected by both pre-SFT and final hard-gold/downstream audits. It also checks
that `teacher_rubrics_filtered_report.json` binds the raw teacher criteria,
filtered teacher criteria, Meta-Verifier provider, SFT preflight report, and
meta-verifier budget report by SHA256. The verifier-filtered output hash must
feed `proxy_gold_build_report.json:input_sha256`, and
`proxy_gold_build_report.json` plus `proxy_gold_verl_report.json` must bind the
current `blindspot_sft.jsonl`, `proxy_gold.jsonl`, and
`proxy_gold_verl.parquet` SHA256 values to the same training artifacts inspected
by every final holdout audit.
If a final audit reports `overlap_query_count == 0` while any configured
training-side artifact is missing or stale, it remains
`artifact_status=blocked` and `overlap_status=not_auditable`. That state is
missing C0 evidence, not a clean-isolation result.

Treat the contamination-aware, evidence-gated pipeline as the experiment
protocol, not as the central research contribution. Its role is to decide which
problem--metric--method claims are safe to write: hard-gold isolation protects
the blind-spot diagnosis, API/budget/provenance gates protect downstream
utility claims, and semantic-space evidence must be interpreted through
nearest-gold category coverage, nearest-gold cluster coverage, and dispersion
rather than through the visual plot alone.

## Claims Not Yet Safe to Write as Results

Do not state these as completed unless new evidence gates pass:

- "SFT+GRPO is already paper-facing on RubricBench test_main."
- "BlindSpot-RL already has paper-facing RewardBench / RewardBench-2 / JudgeBench utility evidence."
- "A semantic-space plot alone establishes the redundancy penalty's mechanism."
- "The method has the strongest overall result."

These can be written as planned experiments or planned hypotheses until tables are populated.

## Minimum AAAI Experiment Package

### Gate-to-Claim Map

Use this map before writing any result sentence into the manuscript:

| Gate | Evidence needed | Claim allowed | If missing |
| --- | --- | --- | --- |
| C0 | Zero-overlap RubricBench `test_main`, proxy-train, RewardBench, JudgeBench, and RewardBench-2 audits with SHA-bound provenance | Any trained-method row can be paper-facing | Report only the frozen diagnostic and other non-training evidence |
| C2/C3 | Hard-gold BSC, redundancy, hallucination, confidence intervals, threshold robustness, and bounded criteria count under one fixed BGE/verifier protocol | Metric-support for coverage changes | Do not write aggregate method conclusions |
| C4/C9/C10 | RewardBench, JudgeBench, and RewardBench-2 utility under the API/model scorer, current budget report, and table-level eligibility flags | Judge-utility support | Keep BSC evidence metric-only |
| C7 | Reward-component variants plus no-verifier-filtering ablation with trained variant identities and verifier settings recorded | Support attribution to reward components rather than verbosity or unverifiable criteria | Treat as an uncontrolled training comparison |
| C12 | Query-aligned per-gold dimension transition audit with recovered and lost dimensions under the same embedding and threshold protocol | Dimension-level recovery evidence | Write aggregate coverage change only |
| C13 | Semantic-space point CSV, JSON summary, and rendered SVG/PDF with nearest-gold category, cluster, similarity, and dispersion checks | Mechanism visualization | Treat the plot as illustrative only |
| C14 | SFT-only versus SFT+GRPO rows under the same hard-gold protocol, verifier source, and bootstrap-CI settings | RLVR-stage support | Report proxy-gold supervision only |

The central method claim requires all relevant gates together. A row can look
better numerically and still remain non-paper-facing if its C0 provenance,
downstream scorer contract, ablation identity, or dimension-transition audit is
missing.

Required main tables:

The central method claim is allowed only when three evidence families pass
together: hard-gold BSC evidence shows a reportable coverage change, redundancy
and hallucination do not materially worsen, and downstream judge utility is
supported on held-out RewardBench, RewardBench-2, and JudgeBench under the
API/model scorer. A BSC coverage change by itself is only a metric result; it
is not enough to claim better judging utility.

1. Hard-gold BSC table on RubricBench `test_main`.
   - Methods: base, GPT-4o, GPT-5, GPT-5.4, SFT-only, SFT+GRPO.
   - Metrics: Cov, Blind, Red, Hall, reward, average rubric count.
2. Downstream utility table.
   - Datasets: RewardBench holdout, RewardBench-2, JudgeBench.
   - Metrics: accuracy, tie rate, mean margin.
   - Every paper-facing downstream row must use the API/model scorer with a
     configured provider and a budget report whose `ok` field is `true`; keyword
     scorer results are smoke-test evidence only.
   - The budget report must bind the exact `downstream_eval.jsonl` input,
     `configs/judge_scorer.local.jsonl` provider, their SHA256 hashes, rubric
     unit field, and the pairwise or multi-candidate call contract; a stale
     `ok=true` report from a different scorer run is not valid evidence.
   - The downstream `summary.json` must repeat the exact joined input,
     current input/provider SHA256 hashes, benchmark format, scorer provider,
     budget report, and scorer contract (`calls_per_record_per_provider`,
     `unit_field`, and any candidate multiplier) so the result remains
     auditable after tables are exported.
   - The evidence gate must compare the summary hashes against the budget-report
     contract hashes; matching paths alone are not enough for paper-facing
     downstream utility claims.
   - The experiment summarizer and artifact exporter must also enforce this
     eligibility: downstream accuracy/tie/margin cells are paper-facing only
     when `paper_claim_eligible=true`, `downstream_status=pass`, and
     `downstream_paper_claim_eligible=true`.
   - The evidence matrix must additionally check the final `main_table.csv`
     rows with `table_values`, so C4/C9/C10 cannot pass if exported downstream
     rows are `not_paper_eligible` or missing the table-level eligibility flag.
   - The evidence gate checks actual evaluated examples (`summary.json:n >= 100`)
     for both base and SFT+RL, not just the holdout file size.
3. Dimension-transition recovery audit table.
   - Compare base -> SFT-only and base -> SFT+GRPO on identical hard-gold examples.
   - Metrics: recovered gold dimensions, recovered-dimension rate, lost dimensions, loss rate, net recovered-minus-lost rate.
   - C12 requires exact base/candidate `bsc_eval.jsonl` input identities,
     SHA binding to the corresponding BSC join-report `output_sha256`,
     `join_key=query`, no unmatched or duplicate records, BGE embeddings,
     `coverage_tau=0.75`, verifier `valid_flags` filtering, gold-row
     consistency, and recovered dimensions exceeding lost dimensions before a
     dimension-level recovery statement is reportable.
   - Required outputs:
     `<dimension_transition_dir>/transition_summary.json`,
     `<dimension_transition_dir>/transition_per_item.csv`,
     `<dimension_transition_dir>/transition_by_category.csv`, and
     `<dimension_transition_dir>/transition_gold_items.jsonl`.
4. Ablation table.
   - no redundancy penalty
   - no validity/hallucination term
   - no verifier filtering
   - SFT-only vs SFT+GRPO
   - single teacher vs multi-teacher union
   - C5 covers single-teacher vs multi-teacher union and requires the
     `multi_teacher_union` row to use `data/processed/teacher_rubrics_raw.jsonl`
     against `data/processed/rubricbench_gold.jsonl`, BGE embeddings, the
     configured BSC thresholds, `dedupe_tau=0.85`, at least two single-teacher
     variants, and RubricBench-scale matched queries.
   - Evidence gates: C7 covers reward/no-verifier-filtering ablations and
     checks the reward-component variant identities (`full`, `no_red`,
     `no_valid`, `no_verifier`, `cov_only`) plus the component weights for
     coverage, validity, and redundancy. It also requires the trained ablation
     completion file to record verifier envs (`rule` for full/no-red, `none`
     for no-valid/no-verifier/cov-only)
     and requires verifier filtering not to increase hallucination or collapse
     coverage; C14 covers SFT-only vs SFT+GRPO and requires
     both rows to use the same BGE embedding model, coverage/redundancy
     thresholds, verifier-backed hallucination source, bootstrap-CI, bounded
     generated-criteria count, and non-degraded coverage per generated criterion
     protocol before any RL-stage gain is reportable. C3 also bounds the
     SFT+GRPO/base mean generated-criteria ratio so length expansion alone
     cannot satisfy the "not just more criteria" claim.
   - Interpretation rules: `no_red` gains that come with duplicate criteria
     are reward hacking, not faithful dimension coverage; `no_valid` or
     `no_verifier_filter` gains that raise hallucination are not faithful
     coverage; SFT-only parity means the RL stage is not supported; and a weak
     `multi_teacher_union` row means teacher union is only an uncontrolled
     data-construction choice, not a supported data construction advantage.
5. Robustness table.
   - threshold sweep over coverage tau and redundancy tau
   - budget curve over K criteria items
   - human audit pack for matched/unmatched BSC pairs:
     `python3 scripts/build_bsc_human_audit_pack.py --input <bsc_eval.jsonl> --output-dir <audit_dir> --embedding-model BAAI/bge-large-en-v1.5 --matched 25 --unmatched 25`
   - after human annotation, summarize labels with:
     `python3 scripts/summarize_bsc_human_audit_labels.py --input <audit_dir>/audit_items.csv --output-json <audit_dir>/human_label_summary.json --output-md <audit_dir>/human_label_summary.md --min-labeled 50 --min-auto-matched-human-match-rate 0.8 --min-auto-unmatched-confirmation-rate 0.8 --strict`
   - The generated audit pack is only `annotation_pack_ready`; do not claim a
     C6-passing human audit until human labels are filled and
     `status == human_audit_complete` in the audited summary. The summary also
     records `human_labels_completed`, invalid labels, uncertain-label rate,
     auto-matched human-match rate, and auto-unmatched confirmation rate.
   - C6 requires the completed `human_label_summary.json`/`.md` reports for both
     base and SFT+GRPO, `ok == true`, at least 50 completed labels, no invalid
     labels, uncertain rate at most 0.2, and both auto-matched and
     auto-unmatched confirmation rates at least 0.8.
6. Visualization.
   - UMAP/t-SNE of gold, SFT-only, and SFT+GRPO criteria.
   - Color by blind-spot category or gold cluster.
   - The real matrix requests UMAP via `scripts/build_semantic_space_visualization.py`; if optional UMAP dependencies are unavailable, the script records a deterministic PCA fallback in `semantic_space_summary.json`.
   - C13 requires SFT+GRPO to cover at least as many gold categories,
     nearest-gold categories, and nearest-gold semantic clusters as SFT-only,
     preserve generated-criteria dispersion, avoid lower nearest-gold
     cluster-distribution entropy, and avoid lower nearest-gold similarity before the
     visualization claim is reportable. A visually broader plot is only
     illustrative unless `semantic_space_points.csv` and
     `semantic_space_summary.json` support the same mechanism; the point CSV
     must include nearest-gold id, category, similarity, and text columns for
     every generated criterion. C13 validates the exact point CSV schema and
     requires generated rows, including SFT+GRPO rows, to carry non-empty
     nearest-gold audit fields.

## AAAI Formatting Gate

- `paper/main.tex` is wired for the official AAAI kit: if `aaai2026.sty` is
  present in `paper/`, the paper loads it with the `submission` option.
- If the official style file is absent, `main.tex` falls back to a local
  `article` draft format only so narrative tests and asset synchronization can
  run on machines without the AAAI kit.
- Before final submission, place the official `aaai2026.sty` and
  `aaai2026.bst` from the AAAI author kit in `paper/`, compile with the AAAI
  style active, and verify page limits, anonymity, font size, and bibliography
  formatting under the official template.
- The real-run pipeline includes `scripts/check_latex_compile.py`, which writes
  `outputs/submission_readiness/latex_compile_report.json`/`.md`; submission
  readiness consumes this report through the `AAAI LaTeX Compile` raw gate and
  blocks if the compiled PDF is missing, invalid, or not using the official
  AAAI style. The gate also checks that `main.tex` declares the AAAI
  `submission` option, the `aaai2026` bibliography style, the anonymous author
  line, and the configured `max_pages=8` limit before final submission.

## Reviewer Attack Points and Defenses

Use this as a rebuttal-readiness triage table before writing or defending any
trained-method result:

| Reviewer concern | Evidence that answers it | If evidence is missing |
| --- | --- | --- |
| "This only optimizes wording." | Section 2 defines dimension-level blind spots; BSC measures semantic coverage of human-gold dimensions under a fixed BGE protocol; C12 audits recovered and lost human-gold dimensions query-by-query. | Write criteria-elicitation and aggregate coverage-change language only; do not write dimension-level recovery. |
| "BSC is self-defined and may not matter." | C6 matched/unmatched human audit validates BSC alignment; C4/C9/C10 show held-out RewardBench, JudgeBench, and RewardBench-2 judge utility under the API/model scorer. | Keep BSC as metric-support only, not downstream judge-utility support. |
| "The model may just output more criteria." | C3/C14 bound generated-criteria count and generated-to-gold ratio, audit coverage per generated criterion, and control redundancy; C7 no-redundancy/no-validity ablations; criteria-budget curves show whether coverage changes survive fixed budgets. | Attribute only aggregate metric movement; do not claim faithful coverage of new dimensions. |
| "Training may contaminate evaluation." | C0 binds RubricBench `test_main`, proxy-train, RewardBench, JudgeBench, and RewardBench-2 audits by SHA256 and requires auditable `overlap_query_count == 0` on the exact training artifacts. | No trained-method row is paper-facing, even if metric values look favorable. |
| "RLVR/GRPO is not the source of the effect." | C14 compares SFT-only and SFT+GRPO under the same hard-gold BSC/verifier/bootstrap protocol; C7 records reward-component variants and verifier settings. | Report proxy-gold supervision evidence rather than RLVR-stage support. |
| "The visualization is only a pretty plot." | C13 validates `semantic_space_points.csv`, `semantic_space_summary.json`, SVG/PDF outputs, nearest-gold category/cluster coverage, nearest-gold similarity, and dispersion. | Treat the semantic-space figure as illustrative only. |

- "This only optimizes wording."
  - Defense: emphasize upstream evaluation dimension coverage, not criteria-text quality.
- "BSC is self-defined."
  - Defense: define BSC against human-gold dimensions, keep the embedding
    model/thresholds fixed before final evaluation, add downstream judge
    utility on held-out RewardBench / RewardBench-2 / JudgeBench, and require
    C6-passing matched/unmatched human-audit summaries before treating BSC alignment as
    validated.
- "The model may just output more criteria."
  - Defense: budget curve and redundancy penalty ablation.
- "Training may contaminate evaluation."
  - Defense: RubricBench `test_main` hard-gold holdout; it is not used for
    SFT, proxy criteria elicitation, reward tuning, prompt selection, verifier
    calibration, hyperparameter tuning, or checkpoint selection. Proxy-gold
    training is separated by query-disjoint splits; RewardBench proxy-train is
    filtered against `test_main` and the RewardBench/JudgeBench/RewardBench-2
    downstream holdouts before teacher generation, and every holdout
    contamination audit must report `overlap_query_count == 0`. The audits
    inspect visible prompt fields and nested reward metadata such as
    `extra_info.query`, so verl parquet records are checked against their
    original queries, not only formatted training prompts. If a configured
    training artifact is absent, the audit remains `artifact_status=blocked`
    and `overlap_status=not_auditable` even when the current overlap count is
    zero. Paper-facing
    method-result claims must bind the exact training inputs, holdout audit
    reports, and evaluation outputs.
- "Verifier bias controls the result."
  - Defense: report rule verifier, relaxed API audit, verifier ablation, and C6-gated human-audit checks for matched/unmatched pairs.
- "Embedding threshold is arbitrary."
  - Defense: do not tune thresholds on RubricBench `test_main`; report the
    fixed `coverage_tau=0.75` / `redundancy_tau=0.85` protocol, threshold
    sweep, paired bootstrap confidence intervals under the same BGE model, and
    C6 strict-gate human-audit summaries.

## Next High-Leverage Work

1. Finish full proxy criteria elicitation and verifier filtering.
2. Build SFT data and train SFT-only baseline.
3. Run GRPO with BSC reward through `src/blindspot_rl/verl_reward.py`.
4. Evaluate on RubricBench `test_main`.
5. Run the dimension-transition audit for base -> SFT-only and base -> SFT+GRPO.
6. Run downstream utility with an API/model rubric scorer, not keyword scorer.
7. Generate semantic visualization:
   - Minimal diagnostic example:
     `python3 scripts/build_semantic_space_visualization.py --input base=data/processed/minimal_claim/base/bsc_eval.jsonl --output-dir outputs/minimal_claim/base/semantic_space --embedding-model BAAI/bge-large-en-v1.5`
   - Full paper comparison should repeat `--input label=path` for base, SFT-only, and SFT+GRPO outputs.
8. Run readiness/evidence gates before writing any result claim into the paper.
