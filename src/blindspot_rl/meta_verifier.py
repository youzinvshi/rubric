"""Meta-verifier utilities for proxy evaluation-criteria filtering.

The verifier is intentionally fail-closed: items that cannot be parsed,
are too generic, or cannot be judged by the configured verifier are marked
invalid rather than silently entering proxy-gold training data.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from blindspot_rl.llm_api import OpenAICompatibleClient, extract_json, parse_bool_flag
from blindspot_rl.reward_bsc import Embedder, TokenOverlapEmbedder, pairwise_cosine, parse_rubrics


GENERIC_STANDALONE = {
    "accurate",
    "bad",
    "clear",
    "clarity",
    "complete",
    "concise",
    "correct",
    "detailed",
    "factual",
    "good",
    "helpful",
    "quality",
    "relevant",
    "safe",
    "useful",
}


@dataclass(frozen=True)
class VerificationDecision:
    rubric: str
    valid: bool
    reason: str
    source: str
    atomic: bool | None = None
    decidable: bool | None = None
    relevant: bool | None = None
    non_hallucinated: bool | None = None
    duplicate_of: int | None = None
    raw_response: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class FilterResult:
    rubrics_before_filter: list[str]
    verified_rubrics: list[str]
    valid_flags: list[int]
    verifier_decisions: list[VerificationDecision] = field(default_factory=list)

    @property
    def validity(self) -> float:
        return sum(self.valid_flags) / max(len(self.valid_flags), 1)

    def decisions_as_dicts(self) -> list[dict[str, Any]]:
        return [decision.as_dict() for decision in self.verifier_decisions]


class RubricVerifier(Protocol):
    def verify(self, rubric: str, *, query: str, data_source: str = "") -> VerificationDecision:
        """Return a detailed verification decision for one criterion item."""


class RuleMetaVerifier:
    """Deterministic verifier used for smoke tests and cheap pre-filtering.

    It catches obvious failures but does not replace API verification for paper
    claims. The rules are deliberately conservative so they do not discard
    domain-specific criteria merely because they are long or technical.
    """

    def __init__(self, *, reject_generic_terms: bool = False):
        self.reject_generic_terms = reject_generic_terms

    def verify(self, rubric: str, *, query: str = "", data_source: str = "") -> VerificationDecision:
        del data_source
        text = normalize_rubric_text(rubric)
        if not text:
            return VerificationDecision(text, False, "empty", "rule")

        lowered = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", lowered)
        del query
        generic_hits = sorted(set(tokens) & GENERIC_STANDALONE)
        if len(tokens) < 3:
            return VerificationDecision(text, False, "too_short", "rule")
        if lowered in GENERIC_STANDALONE:
            return VerificationDecision(text, False, "generic_standalone", "rule")
        if len(tokens) <= 4 and any(token in GENERIC_STANDALONE for token in tokens):
            return VerificationDecision(text, False, "too_generic", "rule")
        if self.reject_generic_terms and generic_hits:
            return VerificationDecision(text, False, "generic_term", "rule")
        if len(text) > 800:
            return VerificationDecision(text, False, "too_long", "rule")
        if looks_like_meta_instruction(text):
            return VerificationDecision(text, False, "meta_instruction", "rule")
        return VerificationDecision(
            text,
            True,
            "rule_pass",
            "rule",
            atomic=None,
            decidable=None,
            relevant=None,
            non_hallucinated=None,
        )

    def judge(self, rubric: str, **kwargs: Any) -> int:
        return int(self.verify(rubric, query=str(kwargs.get("prompt") or kwargs.get("query") or "")).valid)


class APIMetaVerifier:
    """LLM-backed evaluation-criteria verifier with structured parsing."""

    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def verify(self, rubric: str, *, query: str = "", data_source: str = "") -> VerificationDecision:
        text = normalize_rubric_text(rubric)
        system_prompt, user_prompt = build_api_verifier_prompt(query=query, rubric=text, data_source=data_source)
        content = self.client.chat(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )
        return parse_api_decision(text, content)

    def judge(self, rubric: str, **kwargs: Any) -> int:
        return int(
            self.verify(
                rubric,
                query=str(kwargs.get("prompt") or kwargs.get("query") or ""),
                data_source=str(kwargs.get("data_source") or ""),
            ).valid
        )


def filter_proxy_rubrics(
    query: str,
    candidates: Sequence[Any],
    verifier: RubricVerifier | None = None,
    *,
    data_source: str = "",
    embedder: Embedder | None = None,
    dedup_tau: float = 0.90,
    max_rubrics: int | None = None,
    rule_prefilter: bool = True,
    reject_generic_terms: bool = False,
) -> FilterResult:
    """Filter and deduplicate candidate proxy criteria.

    `valid_flags` is aligned with `rubrics_before_filter`; flags are set after
    rule/API validation, exact duplicate removal, semantic duplicate removal,
    and optional max-rubric truncation.
    """

    rubrics = parse_rubrics(list(candidates), dedupe=False)
    decisions: list[VerificationDecision] = []
    rule_verifier = RuleMetaVerifier(reject_generic_terms=reject_generic_terms)
    active_verifier = verifier or rule_verifier

    for rubric in rubrics:
        rule_decision = rule_verifier.verify(rubric, query=query)
        if rule_prefilter and not rule_decision.valid:
            decisions.append(rule_decision)
            continue
        if verifier is None:
            decisions.append(rule_decision)
            continue
        try:
            api_decision = active_verifier.verify(rule_decision.rubric, query=query, data_source=data_source)
        except Exception as exc:  # pragma: no cover - exercised in integration
            api_decision = VerificationDecision(
                rule_decision.rubric,
                False,
                f"verifier_error:{type(exc).__name__}",
                "api",
                raw_response=str(exc),
            )
        decisions.append(api_decision)

    decisions = mark_exact_duplicates(decisions)
    decisions = mark_semantic_duplicates(decisions, embedder=embedder, dedup_tau=dedup_tau)
    decisions = enforce_max_rubrics(decisions, max_rubrics=max_rubrics)
    verified = [decision.rubric for decision in decisions if decision.valid]
    return FilterResult(
        rubrics_before_filter=rubrics,
        verified_rubrics=verified,
        valid_flags=[int(decision.valid) for decision in decisions],
        verifier_decisions=decisions,
    )


def parse_api_decision(rubric: str, response: str) -> VerificationDecision:
    candidate = extract_json(response) or response.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return VerificationDecision(
            rubric,
            False,
            "invalid_json_response",
            "api",
            raw_response=response,
        )
    if not isinstance(parsed, dict):
        return VerificationDecision(rubric, False, "non_object_response", "api", raw_response=response)

    atomic = parse_bool_flag(parsed.get("atomic"))
    decidable = parse_bool_flag(parsed.get("decidable"))
    relevant = parse_bool_flag(parsed.get("relevant"))
    non_hallucinated = parse_bool_flag(parsed.get("non_hallucinated"))
    valid = parse_bool_flag(parsed.get("valid")) and atomic and decidable and relevant and non_hallucinated
    reason = str(parsed.get("reason") or ("api_pass" if valid else "api_reject"))
    return VerificationDecision(
        normalize_rubric_text(rubric),
        valid,
        reason,
        "api",
        atomic=atomic,
        decidable=decidable,
        relevant=relevant,
        non_hallucinated=non_hallucinated,
        raw_response=response,
    )


def build_api_verifier_prompt(query: str, rubric: str, data_source: str = "") -> tuple[str, str]:
    """Build a domain-aware verifier prompt.

    The default verifier is still strict, but relevance is interpreted against
    the source task type. This prevents valid medical/search rubrics from being
    rejected merely because they encode standard evaluation dimensions rather
    than words that literally appear in the query.
    """

    policy = domain_policy(data_source)
    system = (
        "You are a strict but domain-aware meta-verifier for evaluation-criteria training data. "
        "Return JSON only with keys: valid, atomic, decidable, relevant, non_hallucinated, reason. "
        "A valid criterion must satisfy all four checks.\n\n"
        "Definitions:\n"
        "- atomic: the rubric tests one evaluation dimension. It may include examples or a short list "
        "of acceptable signals if they all serve the same dimension.\n"
        "- decidable: a judge can answer yes/no or mostly yes/no by inspecting a candidate answer. "
        "Do not reject merely because expert judgment is required.\n"
        "- relevant: the rubric is grounded in the user query OR in standard evaluation needs for the "
        "declared data source/task type.\n"
        "- non_hallucinated: the rubric does not require facts, entities, constraints, or external "
        "documents unrelated to the query/task type.\n\n"
        "Important calibration:\n"
        "- Do not require every good criterion to cover the entire query. A criterion can be valid if it "
        "checks one legitimate aspect of the answer.\n"
        "- Do not reject a criterion simply because it is about safety, evidence, source quality, "
        "freshness, or intent satisfaction when those are standard dimensions for the task type.\n"
        "- Reject broad style-only criteria such as generic helpfulness/clarity unless the task type "
        "or user query specifically calls for writing quality, presentation, or readability.\n\n"
        f"Task-specific policy:\n{policy}"
    )
    user = (
        f"Data source/task type: {data_source or 'unknown'}\n\n"
        f"User query:\n{query}\n\n"
        f"Candidate criterion:\n{rubric}\n\n"
        "Return JSON only. Example schema:\n"
        '{"valid": true, "atomic": true, "decidable": true, "relevant": true, '
        '"non_hallucinated": true, "reason": "short reason"}'
    )
    return system, user


def domain_policy(data_source: str) -> str:
    ds = data_source.lower()
    if "healthbench" in ds or "medical" in ds or "health" in ds:
        return (
            "- This is a medical/health evaluation task.\n"
            "- Treat medical factuality, uncertainty calibration, risk disclosure, red-flag escalation, "
            "professional-care recommendation, contraindication/interaction caution, and avoiding "
            "overconfident diagnosis/treatment as relevant even when not explicitly requested.\n"
            "- A safety rubric can be atomic if it checks one safety dimension, such as whether the "
            "answer gives urgent-care guidance for red flags; examples of red flags do not make it "
            "non-atomic.\n"
            "- Do not reject clinician/source/guideline caution merely because it goes beyond the "
            "surface wording of the query; reject it only if it is medically unrelated or asks for "
            "specific facts not grounded in the task."
        )
    if any(key in ds for key in ("beir", "serp", "search", "nq", "retrieval")):
        return (
            "- This is a search-intent / retrieval evaluation task.\n"
            "- Treat topical relevance, user intent satisfaction, evidence support, answerability from "
            "the document/snippet, source authority, freshness/recency when time-sensitive, and avoiding "
            "misleading information as relevant search-evaluation dimensions.\n"
            "- A rubric about citations, evidence, or authoritative sourcing can be valid if it checks "
            "whether a result/answer is supportable or trustworthy for the query.\n"
            "- Do not reject source-quality or freshness rubrics solely because the query is short. "
            "Reject only if the source/freshness requirement is plainly unrelated to the query intent."
        )
    if "writingbench" in ds or "writing" in ds:
        return (
            "- This is a writing-quality evaluation task.\n"
            "- Treat task fulfillment, structure, style/tone, audience fit, formatting constraints, "
            "genre compliance, and factual consistency as relevant.\n"
            "- Rubrics about clarity or organization can be valid when they test a concrete writing "
            "requirement, but reject vague all-purpose praise criteria."
        )
    if "ifbench" in ds or "advancedif" in ds or "instruction" in ds:
        return (
            "- This is an instruction-following evaluation task.\n"
            "- Treat explicit constraints, implicit constraints, safety constraints, keyword counts, "
            "format requirements, ordering, exclusions, and exact wording constraints as relevant.\n"
            "- Exact keyword-count checks are valid when the query contains keyword-count requirements."
        )
    if "rewardbench" in ds or "pairwise" in ds:
        return (
            "- This is a pairwise response-quality judgment task.\n"
            "- Treat instruction following, factuality, completeness, refusal/safety behavior, reasoning "
            "quality, concision, and user-intent satisfaction as relevant.\n"
            "- For unsafe user requests, rubrics about safe refusal, avoiding actionable harm, and "
            "redirecting to benign alternatives are relevant and non-hallucinated."
        )
    return "- Apply the default strict rubric checks against the user query."


def mark_exact_duplicates(decisions: Sequence[VerificationDecision]) -> list[VerificationDecision]:
    seen: dict[str, int] = {}
    output: list[VerificationDecision] = []
    for idx, decision in enumerate(decisions):
        if not decision.valid:
            output.append(decision)
            continue
        key = normalized_key(decision.rubric)
        if key in seen:
            output.append(
                replace_decision(decision, valid=False, reason="exact_duplicate", duplicate_of=seen[key])
            )
        else:
            seen[key] = idx
            output.append(decision)
    return output


def mark_semantic_duplicates(
    decisions: Sequence[VerificationDecision],
    *,
    embedder: Embedder | None = None,
    dedup_tau: float = 0.90,
) -> list[VerificationDecision]:
    valid_indices = [idx for idx, decision in enumerate(decisions) if decision.valid]
    if len(valid_indices) < 2:
        return list(decisions)

    texts = [decisions[idx].rubric for idx in valid_indices]
    sim = pairwise_cosine(texts, texts, embedder or TokenOverlapEmbedder())
    keep_valid_positions: list[int] = []
    duplicate_of: dict[int, int] = {}
    for pos, original_idx in enumerate(valid_indices):
        match = None
        for kept_pos in keep_valid_positions:
            if float(sim[pos, kept_pos]) >= dedup_tau:
                match = valid_indices[kept_pos]
                break
        if match is None:
            keep_valid_positions.append(pos)
        else:
            duplicate_of[original_idx] = match

    output = list(decisions)
    for idx, original_idx in duplicate_of.items():
        output[idx] = replace_decision(
            output[idx],
            valid=False,
            reason="semantic_duplicate",
            duplicate_of=original_idx,
        )
    return output


def enforce_max_rubrics(
    decisions: Sequence[VerificationDecision],
    *,
    max_rubrics: int | None,
) -> list[VerificationDecision]:
    if max_rubrics is None or max_rubrics <= 0:
        return list(decisions)
    output: list[VerificationDecision] = []
    kept = 0
    for decision in decisions:
        if not decision.valid:
            output.append(decision)
            continue
        kept += 1
        if kept <= max_rubrics:
            output.append(decision)
        else:
            output.append(replace_decision(decision, valid=False, reason="max_rubrics_exceeded"))
    return output


def replace_decision(decision: VerificationDecision, **updates: Any) -> VerificationDecision:
    data = decision.as_dict()
    data.update(updates)
    return VerificationDecision(**data)


def normalize_rubric_text(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    normalized = re.sub(r"^\s*(?:[-*•]|\d+[\.)]|[a-zA-Z][\.)])\s+", "", normalized)
    return normalized.strip(" \t\r\n\"'")


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_rubric_text(text).lower())


def looks_like_meta_instruction(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith("return ")
        or lowered.startswith("generate ")
        or "json list" in lowered
        or "candidate answer a" in lowered
        or "candidate answer b" in lowered
    )


def decisions_to_jsonable(decisions: Iterable[VerificationDecision]) -> list[dict[str, Any]]:
    return [decision.as_dict() for decision in decisions]


def validity_summary(results: Sequence[FilterResult]) -> dict[str, Any]:
    n_input = sum(len(result.rubrics_before_filter) for result in results)
    n_valid = sum(sum(result.valid_flags) for result in results)
    reason_counts: dict[str, int] = {}
    for result in results:
        for decision in result.verifier_decisions:
            if not decision.valid:
                reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
    return {
        "n_records": len(results),
        "n_input_rubrics": n_input,
        "n_valid_rubrics": n_valid,
        "validity": n_valid / max(n_input, 1),
        "invalid_reason_counts": reason_counts,
    }
