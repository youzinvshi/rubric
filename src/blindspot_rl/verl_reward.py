"""verl-compatible reward entry points for BlindSpot-RL.

Copy or reference this module from verl custom reward configs. It exposes
multiple common names because reward hook signatures differ across verl
versions and examples.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from blindspot_rl.reward_bsc import (
    BAD_FORMAT_REWARD,
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    compute_metrics,
    parse_rubrics,
)


_EMBEDDER = None
_VERIFIER = None


def compute_score(
    data_source: str | None = None,
    solution_str: Any | None = None,
    ground_truth: Any | None = None,
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Primary verl reward hook.

    Expected fields can come from either `ground_truth` or `extra_info`:
    - gold_rubrics: list[str]
    - prompt/query: optional prompt for verifier
    - valid_flags/verifier_flags/validity_flags: optional flags aligned to the
      generated response. If absent, the configured fail-closed verifier is used.
    """

    del data_source
    extra = dict(extra_info or {})
    gold = resolve_gold_rubrics(extra=extra, ground_truth=ground_truth, kwargs=kwargs)
    prompt = extra.get("prompt") or extra.get("query") or kwargs.get("prompt") or ""
    response = solution_str if solution_str is not None else kwargs.get("response", "")
    valid_flags = resolve_valid_flags(extra=extra, ground_truth=ground_truth, kwargs=kwargs)
    verifier = build_flag_verifier(response=response, flags=valid_flags) if valid_flags is not None else get_verifier()
    return reward_func(prompt=prompt, response=response, gold_rubrics=gold, verifier=verifier)


def reward_func(
    prompt: str,
    response: Any,
    gold_rubrics: Any,
    **kwargs: Any,
) -> float:
    """Framework-agnostic reward wrapper used by verl/OpenRLHF adapters."""

    gold = parse_rubrics(gold_rubrics, dedupe=True)
    if not gold:
        return BAD_FORMAT_REWARD
    verifier = kwargs["verifier"] if "verifier" in kwargs else get_verifier()
    weights = (
        float(os.environ.get("BSC_W_COV", "1.0")),
        float(os.environ.get("BSC_W_VALID", "0.5")),
        float(os.environ.get("BSC_W_RED", "0.5")),
    )
    metrics = compute_metrics(
        response=response,
        gold_rubrics=gold,
        prompt=prompt,
        verifier=verifier,
        embedder=get_embedder(),
        coverage_tau=float(os.environ.get("BSC_COVERAGE_TAU", "0.75")),
        redundancy_tau=float(os.environ.get("BSC_REDUNDANCY_TAU", "0.85")),
        weights=weights,
    )
    return metrics.reward if metrics.n_gen > 0 else BAD_FORMAT_REWARD


def resolve_gold_rubrics(
    extra: Mapping[str, Any],
    ground_truth: Any | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> list[str]:
    """Resolve gold rubrics across common verl/OpenRLHF payload shapes."""

    kwargs = kwargs or {}
    candidates = [
        extra.get("gold_rubrics"),
        extra.get("gold"),
        extra.get("rubrics_gold"),
        ground_truth,
        kwargs.get("gold_rubrics"),
        kwargs.get("gold"),
        kwargs.get("rubrics_gold"),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = resolve_gold_rubrics(extra=candidate)
            if nested:
                return nested
        parsed = parse_rubrics(candidate, dedupe=True)
        if parsed:
            return parsed
    return []


def resolve_valid_flags(
    extra: Mapping[str, Any],
    ground_truth: Any | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> list[Any] | None:
    """Resolve optional verifier flags across common reward payload shapes."""

    kwargs = kwargs or {}
    candidates = [
        extra.get("valid_flags"),
        extra.get("verifier_flags"),
        extra.get("validity_flags"),
        kwargs.get("valid_flags"),
        kwargs.get("verifier_flags"),
        kwargs.get("validity_flags"),
    ]
    if isinstance(ground_truth, Mapping):
        candidates.extend(
            [
                ground_truth.get("valid_flags"),
                ground_truth.get("verifier_flags"),
                ground_truth.get("validity_flags"),
            ]
        )

    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, list):
            raise ValueError("valid_flags must be a list when supplied to the BSC reward hook")
        return candidate
    return None


def build_flag_verifier(response: Any, flags: list[Any]) -> Any:
    """Build a verifier from response-aligned valid flags.

    This mirrors the offline BSC evaluator contract and fails closed on
    malformed flags instead of silently treating the validity term as perfect.
    """

    gen = parse_rubrics(response, dedupe=False)
    if len(flags) != len(gen):
        raise ValueError(f"valid_flags length mismatch: {len(flags)} flags for {len(gen)} generated rubrics")
    bool_flags = [parse_valid_flag(flag) for flag in flags]
    cursor = {"idx": 0}

    def verifier(_rubric: str, _prompt: str | None = None, **_: Any) -> bool:
        del _rubric, _prompt
        flag = bool_flags[cursor["idx"]]
        cursor["idx"] += 1
        return flag

    return verifier


def parse_valid_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "valid"}:
            return True
        if text in {"0", "false", "no", "invalid"}:
            return False
    raise ValueError(f"valid_flags contains non-binary value: {value!r}")


def get_embedder() -> SentenceTransformerEmbedder | TokenOverlapEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        model = os.environ.get("BSC_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        if model == "token-overlap":
            _EMBEDDER = TokenOverlapEmbedder()
        else:
            _EMBEDDER = SentenceTransformerEmbedder(model)
    return _EMBEDDER


def get_verifier() -> Any | None:
    """Return the configured training-time verifier for R_valid."""

    global _VERIFIER
    mode = os.environ.get("BSC_VERIFIER", "rule").strip().lower()
    if mode in {"", "none", "off", "disabled"}:
        return None
    if mode != "rule":
        raise ValueError(f"Unsupported BSC_VERIFIER={mode!r}; supported values are 'rule' and 'none'")
    if _VERIFIER is None:
        from blindspot_rl.meta_verifier import RuleMetaVerifier

        _VERIFIER = RuleMetaVerifier(
            reject_generic_terms=os.environ.get("BSC_REJECT_GENERIC_TERMS", "0").strip().lower()
            in {"1", "true", "yes"}
        )
    return _VERIFIER


def compute_reward(*args: Any, **kwargs: Any) -> float:
    """Alias for reward registries that expect `compute_reward`."""

    return compute_score(*args, **kwargs)
