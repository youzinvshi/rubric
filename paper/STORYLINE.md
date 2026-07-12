# BlindSpot-RL Storyline

## Core Positioning

Do not frame the paper as criteria-text polishing. The research problem is
upstream of final judge accuracy: LLM evaluators can make plausible decisions
while omitting evaluation dimensions that human annotators consider essential.
We call these omissions **evaluation blind spots**.

The strongest AAAI-facing claim is:

> LLM evaluators suffer from systematic blind spots in the evaluation
> dimensions they consider. We introduce Blind-Spot Coverage (BSC), a
> verifiable semantic reward for open-ended criteria elicitation, and use
> evidence-gated experiments to test when RLVR/GRPO warrants a
> dimension-level recovery statement while controlling redundancy
> and hallucination.

Current paper title:

> Evaluation Blind Spots: Verifiable Semantic Rewards for Open-Ended Criteria Elicitation

## Frozen Motivation Evidence

Use the 100-example hard-gold diagnostic as the main Section 2 motivation:

- Coverage: `0.3692`
- Mean blind-spot rate: `0.6308`
- Blind-rate 95% CI: `0.5785--0.6823`
- Redundancy: `0.0768`
- Hallucination/invalidity: `0.1227`

Treat these as a frozen diagnostic snapshot, not as loose illustrative
numbers. The minimal-claim evidence matrix must check the exact snapshot before
Section 2 can cite it as paper-facing motivation evidence.

This supports the Section 2 motivation that blind spots are not occasional missed details.
They are a systematic coverage gap in the evaluative basis itself.

## Paper Arc

1. **Problem discovery:** LLM-as-a-Judge work usually evaluates final scores,
   preferences, or reward-model rankings. We move the question earlier: what
   dimensions does the judge consider before scoring?
2. **Metric:** BSC measures whether generated criteria semantically cover
   human-gold evaluation dimensions while penalizing redundancy and invalid or
   hallucinated criteria.
3. **Method:** BSC makes open-ended criteria elicitation testable with
   verifiable rewards. BlindSpot-RL formulates proxy-gold SFT followed by
   GRPO/RLVR with BSC reward.
4. **Evidence closure:** Claims are allowed only when hard-gold isolation,
   BSC coverage changes, redundancy/hallucination controls, downstream utility, ablations,
   and semantic-space visualization all pass evidence gates.

## Section Plan

### Section 1: Introduction

- Start from the limitation of final-score evaluation.
- Define evaluation blind spots as missing human-prioritized dimensions.
- Emphasize that this is a reliability issue in the evaluative basis, not a
  generic generation-quality issue.
- Contributions:
  1. Identify and quantify evaluation blind spots.
  2. Define BSC as a metric and verifiable reward.
  3. Formulate an evidence-gated RLVR/GRPO optimization test for open-ended
     semantic criteria elicitation.
- Treat the contamination-aware, evidence-gated pipeline as the experiment
  protocol that protects the claims, not as the central research contribution.

### Section 2: Evaluation Blind Spots

- Present the dimension-level blind-spot definition:
  `max_j cos(e(g_i), e(r_j)) < tau`.
- Report the frozen 100-example diagnostic numbers above.
- Include category-level blind-spot attribution to show structure rather than
  random misses.
- Include criteria-budget analysis to show that blind spots are not solved by
  simply making criteria lists longer.

### Section 3: Method

- Define BSC:
  semantic gold coverage minus redundancy and invalidity/hallucination.
- Explain fail-closed verification and why unverifiable criteria should not
  receive reward.
- Explain the two-stage training protocol:
  proxy-gold SFT, then GRPO/RLVR with BSC reward.
- State the data isolation protocol:
  RubricBench `test_main` remains hard-gold holdout; proxy-gold data is used
  only for scaled training; downstream benchmarks remain non-overlap holdouts.
- RewardBench proxy-train is filtered against RubricBench `test_main` and the
  RewardBench, JudgeBench, and RewardBench-2 downstream holdouts before teacher
  generation. The downstream utility claims require both API scorer audit
  metadata and query-overlap audits with `overlap_query_count == 0`.
- `overlap_query_count == 0` is not enough when the configured training-side
  artifacts are incomplete. Such reports remain `artifact_status=blocked` and
  `overlap_status=not_auditable`, so C0 is still missing rather than clean.

### Section 4: Experiments

The final paper tables must test all three before the claims can be permitted:

- Whether hard-gold BSC evidence shows a reportable coverage change.
- Whether redundancy and hallucination do not worsen materially.
- Whether downstream utility is supported on RewardBench, JudgeBench, and
  RewardBench-2.

Minimum ablations:

- No redundancy penalty.
- No hallucination/validity term.
- No verifier filtering.
- SFT-only vs. SFT+GRPO.
- Single-teacher vs. multi-teacher union.

The main conclusion should be evidence-gated: the RL stage should be described
as reducing semantic blind spots only when C12/C14 show dimension-level
recovery over SFT-only on aligned hard-gold inputs while keeping redundancy and
hallucination within the configured thresholds. Until then, the manuscript
should describe the RLVR stage as a tested hypothesis rather than a completed
reduction result.

### Gate-to-Claim Map

Use the same gate map as the main manuscript table:

| Gate | Evidence needed | Claim allowed | If missing |
| --- | --- | --- | --- |
| C0 | Zero-overlap hard-gold, proxy-train, and downstream audits with SHA-bound provenance | Trained-method rows can be paper-facing | Report only non-training diagnostics |
| C2/C3 | Hard-gold BSC plus redundancy, hallucination, confidence intervals, and threshold checks under one fixed protocol | Metric-support for coverage changes | Do not write aggregate method conclusions |
| C4/C9/C10 | RewardBench, JudgeBench, and RewardBench-2 utility under the fixed scorer and budget contract | Judge-utility support | Keep BSC evidence metric-only |
| C7 | Reward-component and verifier ablations | Support attribution to reward components rather than verbosity | Treat as an uncontrolled training comparison |
| C12 | Query-aligned per-gold dimension transition audit | Dimension-level recovery evidence | Write aggregate coverage change only |
| C13 | Point-level semantic-space CSV, JSON summary, and rendered figure with cluster-coverage checks | Mechanism visualization | Treat the figure as illustrative only |
| C14 | SFT-only versus SFT+GRPO under the same protocol | RLVR-stage support | Report proxy-gold supervision only |

The central method sentence requires the gates jointly: C0 for data isolation,
C2/C3 for metric support, C4/C9/C10 for downstream utility, C7 for reward
components, C12 for dimension-level transitions, C13 for mechanism
visualization, and C14 for the RLVR-stage comparison.

### Section 5: Visualization and Analysis

- Use the semantic-space figure to audit, under C13, whether SFT+GRPO criteria
  show a reportable nearest-gold region-coverage change relative to SFT-only
  without local collapse.
- Treat the figure as mechanism evidence, not as a replacement for BSC.
- Keep point-level CSV and JSON summary synchronized so the figure is
  auditable; nearest-gold category coverage, nearest-gold cluster coverage,
  nearest-gold cluster-distribution entropy, nearest-gold similarity, and
  dispersion must support the visual impression before claiming broader
  semantic coverage.

## Unsafe Claims Until Real Gates Pass

Do not write these as conclusions until the corresponding evidence gates pass:

- "SFT+GRPO is already paper-facing on RubricBench `test_main`."
- "BlindSpot-RL already has paper-facing RewardBench/JudgeBench/RewardBench-2 utility evidence."
- "The method has the strongest overall result."
- "RL always reduces hallucination."
- "The method generalizes broadly across all domains."

Use evidence-gated wording instead:

- "The pipeline is designed to test whether..."
- "The claim is safe only after C2/C3/C4/C5/C6/C7/C9/C10/C12/C13/C14 pass."
- "The minimal motivation gate supports the 100-example diagnostic blind-spot
  finding for Section 2 only; trained-method claims wait for the real C0-C14
  gates."
