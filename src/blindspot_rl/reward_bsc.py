"""BlindSpot-RL reward utilities.

This module implements Blind-Spot Coverage (BSC) and the reward used by
evaluation-criteria policy RLVR training. The functions are intentionally framework-light:
they can be imported by verl/OpenRLHF reward hooks, and reused by offline
diagnostics.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_COVERAGE_TAU = 0.75
DEFAULT_REDUNDANCY_TAU = 0.85
DEFAULT_WEIGHTS = (1.0, 0.5, 0.5)
DEFAULT_BALANCED_WEIGHTS = (0.7, 0.3, 0.5, 0.5)
BAD_FORMAT_REWARD = -1.0


class Embedder(Protocol):
    """Minimal embedding interface used by BSC metrics."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 2D float array with one embedding per text."""


class Verifier(Protocol):
    """Minimal verifier interface.

    Implementations may expose ``judge(rubric)`` or
    ``judge(rubric, prompt=...)`` and should return 1/0 or True/False.
    """

    def judge(self, rubric: str, **kwargs: Any) -> int | bool:
        """Return whether one rubric item is valid and verifiable."""


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers.

    The heavy dependency and model weights are loaded only when encode() is
    first called. This keeps unit tests and format checks lightweight.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for real embeddings. "
                    "Install requirements.txt or pass a custom embedder."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class TokenOverlapEmbedder:
    """Small deterministic embedder for smoke tests and CI.

    It is not a replacement for BGE in paper experiments. It only makes local
    tests runnable without network/model downloads.
    """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vocab: dict[str, int] = {}
        tokenized: list[list[str]] = []
        for text in texts:
            tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower())
            tokenized.append(tokens)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)

        vectors = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for row, tokens in enumerate(tokenized):
            for token in tokens:
                vectors[row, vocab[token]] += 1.0
        return _l2_normalize(vectors)


@dataclass(frozen=True)
class BSCMetrics:
    """Metrics reported by BlindSpot-RL diagnostics."""

    coverage: float
    blind: float
    redundancy: float
    validity: float
    hallucination: float
    reward: float
    n_gold: int
    n_gen: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "coverage": self.coverage,
            "blind": self.blind,
            "redundancy": self.redundancy,
            "validity": self.validity,
            "hallucination": self.hallucination,
            "reward": self.reward,
            "n_gold": self.n_gold,
            "n_gen": self.n_gen,
        }


_DEFAULT_EMBEDDER: Embedder | None = None


def get_default_embedder() -> Embedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder()
    return _DEFAULT_EMBEDDER


def parse_rubrics(text: Any, dedupe: bool = False) -> list[str]:
    """Extract atomic rubric strings from common model-output formats.

    Supported formats:
    - JSON list: ["criterion", {"criterion": "..."}]
    - JSON object containing rubrics/criteria/items
    - Markdown bullets or numbered lists
    - Plain multiline text as a last resort
    """

    if text is None:
        return []
    if isinstance(text, list):
        return _normalize_rubrics(_extract_from_list(text), dedupe=dedupe)
    if isinstance(text, Mapping):
        return _normalize_rubrics(_extract_from_mapping(text), dedupe=dedupe)
    if not isinstance(text, (str, bytes)) and isinstance(text, Iterable):
        return _normalize_rubrics(_extract_from_list(text), dedupe=dedupe)

    raw = str(text).strip()
    if not raw:
        return []

    json_candidate = _extract_json_candidate(raw)
    if json_candidate:
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, list):
                extracted = _normalize_rubrics(_extract_from_list(parsed), dedupe=dedupe)
                if extracted:
                    return extracted
            if isinstance(parsed, Mapping):
                extracted = _normalize_rubrics(_extract_from_mapping(parsed), dedupe=dedupe)
                if extracted:
                    return extracted
        except json.JSONDecodeError:
            pass

    bullet_items = _extract_bullets(raw)
    if bullet_items:
        return _normalize_rubrics(bullet_items, dedupe=dedupe)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return _normalize_rubrics(lines, dedupe=dedupe)


def coverage_reward(
    gen_rubrics: Sequence[str],
    gold_rubrics: Sequence[str],
    tau: float = DEFAULT_COVERAGE_TAU,
    embedder: Embedder | None = None,
) -> float:
    """Return R_cov: semantic coverage of gold dimensions by generated rubrics."""

    gen = _normalize_rubrics(gen_rubrics, dedupe=True)
    gold = _normalize_rubrics(gold_rubrics, dedupe=True)
    if not gold:
        return 0.0
    if not gen:
        return 0.0

    sim = pairwise_cosine(gold, gen, embedder)
    hits = (sim.max(axis=1) >= tau).astype(np.float32)
    return float(hits.mean())


def category_balanced_coverage_reward(
    gen_rubrics: Sequence[str],
    gold_rubrics: Sequence[str],
    gold_categories: Sequence[str],
    tau: float = DEFAULT_COVERAGE_TAU,
    embedder: Embedder | None = None,
) -> float:
    """Return macro-average coverage over gold rubric categories.

    Only categories present in the query's gold rubrics are averaged. This makes
    the reward query-local and prevents models from gaining reward by covering
    only the high-frequency/easy rubric dimensions.
    """

    gen = _normalize_rubrics(gen_rubrics, dedupe=True)
    gold = _normalize_rubrics(gold_rubrics, dedupe=False)
    categories = [str(category).strip() or "uncategorized" for category in gold_categories]
    if len(categories) != len(gold):
        raise ValueError(f"gold_categories length mismatch: {len(categories)} categories for {len(gold)} gold rubrics")
    if not gold:
        return 0.0
    if not gen:
        return 0.0

    sim = pairwise_cosine(gold, gen, embedder)
    hits = (sim.max(axis=1) >= tau).astype(np.float32)
    by_category: dict[str, list[float]] = {}
    for category, hit in zip(categories, hits):
        by_category.setdefault(category, []).append(float(hit))
    category_coverages = [sum(values) / len(values) for values in by_category.values() if values]
    return float(sum(category_coverages) / len(category_coverages)) if category_coverages else 0.0


def redundancy_penalty(
    gen_rubrics: Sequence[str],
    tau: float = DEFAULT_REDUNDANCY_TAU,
    embedder: Embedder | None = None,
) -> float:
    """Return semantic duplicate ratio among generated rubrics.

    The denominator follows the paper sketch: duplicate unordered pairs divided
    by number of generated items, so the penalty grows with repeated dimensions.
    """

    gen = _normalize_rubrics(gen_rubrics, dedupe=False)
    n = len(gen)
    if n < 2:
        return 0.0

    sim = pairwise_cosine(gen, gen, embedder)
    upper = np.triu(sim >= tau, k=1)
    duplicate_pairs = int(upper.sum())
    return float(duplicate_pairs / n)


def semantic_dedupe(
    rubrics: Sequence[str],
    tau: float = DEFAULT_REDUNDANCY_TAU,
    embedder: Embedder | None = None,
) -> list[str]:
    """Greedily remove semantically duplicate rubric dimensions.

    This is used for multi-teacher union construction. It preserves the first
    occurrence, so upstream teacher order can encode preference if desired.
    """

    items = _normalize_rubrics(rubrics, dedupe=True)
    if len(items) < 2:
        return items

    sim = pairwise_cosine(items, items, embedder)
    keep: list[int] = []
    for idx in range(len(items)):
        if all(sim[idx, kept_idx] < tau for kept_idx in keep):
            keep.append(idx)
    return [items[idx] for idx in keep]


def validity_reward(
    gen_rubrics: Sequence[str],
    verifier: Verifier | Callable[..., int | bool] | None = None,
    prompt: str | None = None,
) -> float:
    """Return R_valid: share of generated rubrics judged valid/verifiable."""

    gen = _normalize_rubrics(gen_rubrics, dedupe=False)
    if not gen:
        return 0.0
    if verifier is None:
        return 1.0

    flags = [_call_verifier(verifier, rubric, prompt=prompt) for rubric in gen]
    return float(sum(flags) / len(flags))


def hallucination_rate(
    gen_rubrics: Sequence[str],
    verifier: Verifier | Callable[..., int | bool] | None = None,
    prompt: str | None = None,
) -> float:
    """Return Hall = 1 - R_valid."""

    return 1.0 - validity_reward(gen_rubrics, verifier=verifier, prompt=prompt)


def compute_metrics(
    response: Any,
    gold_rubrics: Sequence[str],
    prompt: str | None = None,
    verifier: Verifier | Callable[..., int | bool] | None = None,
    embedder: Embedder | None = None,
    coverage_tau: float = DEFAULT_COVERAGE_TAU,
    redundancy_tau: float = DEFAULT_REDUNDANCY_TAU,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> BSCMetrics:
    """Compute BSC metrics plus the weighted RL reward."""

    gen = parse_rubrics(response)
    if not gen:
        return BSCMetrics(
            coverage=0.0,
            blind=1.0 if gold_rubrics else 0.0,
            redundancy=0.0,
            validity=0.0,
            hallucination=1.0,
            reward=BAD_FORMAT_REWARD,
            n_gold=len(_normalize_rubrics(gold_rubrics, dedupe=True)),
            n_gen=0,
        )

    r_cov = coverage_reward(gen, gold_rubrics, tau=coverage_tau, embedder=embedder)
    r_valid = validity_reward(gen, verifier=verifier, prompt=prompt)
    r_red = redundancy_penalty(gen, tau=redundancy_tau, embedder=embedder)
    reward = weights[0] * r_cov + weights[1] * r_valid - weights[2] * r_red

    return BSCMetrics(
        coverage=r_cov,
        blind=1.0 - r_cov,
        redundancy=r_red,
        validity=r_valid,
        hallucination=1.0 - r_valid,
        reward=float(reward),
        n_gold=len(_normalize_rubrics(gold_rubrics, dedupe=True)),
        n_gen=len(gen),
    )


def compute_reward(
    prompt: str,
    response: Any,
    gold_rubrics: Sequence[str],
    verifier: Verifier | Callable[..., int | bool] | None = None,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    embedder: Embedder | None = None,
    coverage_tau: float = DEFAULT_COVERAGE_TAU,
    redundancy_tau: float = DEFAULT_REDUNDANCY_TAU,
) -> float:
    """Compute the scalar reward used by GRPO/RLVR."""

    metrics = compute_metrics(
        response=response,
        gold_rubrics=gold_rubrics,
        prompt=prompt,
        verifier=verifier,
        embedder=embedder,
        coverage_tau=coverage_tau,
        redundancy_tau=redundancy_tau,
        weights=weights,
    )
    return metrics.reward


def compute_category_balanced_reward(
    prompt: str,
    response: Any,
    gold_rubrics: Sequence[str],
    gold_categories: Sequence[str],
    verifier: Verifier | Callable[..., int | bool] | None = None,
    weights: tuple[float, float, float, float] = DEFAULT_BALANCED_WEIGHTS,
    embedder: Embedder | None = None,
    coverage_tau: float = DEFAULT_COVERAGE_TAU,
    redundancy_tau: float = DEFAULT_REDUNDANCY_TAU,
) -> float:
    """Compute R = w1*R_cov + w2*R_bal_cov + w3*R_valid - w4*R_red."""

    gen = parse_rubrics(response)
    if not gen:
        return BAD_FORMAT_REWARD

    r_cov = coverage_reward(gen, gold_rubrics, tau=coverage_tau, embedder=embedder)
    r_bal_cov = category_balanced_coverage_reward(
        gen,
        gold_rubrics,
        gold_categories,
        tau=coverage_tau,
        embedder=embedder,
    )
    r_valid = validity_reward(gen, verifier=verifier, prompt=prompt)
    r_red = redundancy_penalty(gen, tau=redundancy_tau, embedder=embedder)
    return float(weights[0] * r_cov + weights[1] * r_bal_cov + weights[2] * r_valid - weights[3] * r_red)


def verl_reward_fn(data_item: Mapping[str, Any], response: Any, **kwargs: Any) -> float:
    """Small adapter for verl-style custom reward hooks.

    Expected data fields:
    - prompt: model prompt
    - gold_rubrics: list[str], hidden from model but available to reward
    """

    from blindspot_rl.verl_reward import reward_func

    prompt = str(data_item.get("prompt", ""))
    gold = data_item.get("gold_rubrics") or data_item.get("gold") or []
    return reward_func(prompt=prompt, response=response, gold_rubrics=gold, **kwargs)


def pairwise_cosine(
    left: Sequence[str],
    right: Sequence[str],
    embedder: Embedder | None = None,
) -> np.ndarray:
    """Compute cosine similarity between two text lists."""

    encoder = embedder or get_default_embedder()
    left_texts = list(left)
    right_texts = list(right)
    combined = left_texts + right_texts
    combined_emb = _as_2d_float(encoder.encode(combined))
    left_emb = combined_emb[: len(left_texts)]
    right_emb = combined_emb[len(left_texts) :]

    left_emb = _l2_normalize(left_emb)
    right_emb = _l2_normalize(right_emb)
    left_emb = left_emb.astype(np.float64, copy=False)
    right_emb = right_emb.astype(np.float64, copy=False)
    return np.einsum("ik,jk->ij", left_emb, right_emb)


def _extract_json_candidate(text: str) -> str | None:
    if text.startswith("[") or text.startswith("{"):
        return text
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fenced:
        return fenced.group(1).strip()
    list_start = text.find("[")
    list_end = text.rfind("]")
    if 0 <= list_start < list_end:
        return text[list_start : list_end + 1]
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if 0 <= obj_start < obj_end:
        return text[obj_start : obj_end + 1]
    return None


def _extract_from_mapping(obj: Mapping[str, Any]) -> list[str]:
    for key in ("gold_rubrics", "gold", "rubrics", "criteria", "items", "rubric_list", "dimensions"):
        value = obj.get(key)
        if isinstance(value, list):
            return _extract_from_list(value)
    for key in ("criterion", "criteria", "description", "text", "content"):
        value = obj.get(key)
        if isinstance(value, str):
            return [value]
    return []


def _extract_from_list(items: Iterable[Any]) -> list[str]:
    rubrics: list[str] = []
    for item in items:
        if isinstance(item, str):
            rubrics.append(item)
        elif isinstance(item, Mapping):
            extracted = _extract_from_mapping(item)
            if extracted:
                rubrics.extend(extracted)
            else:
                rubrics.append(json.dumps(item, ensure_ascii=False))
        elif item is not None:
            rubrics.append(str(item))
    return rubrics


def _extract_bullets(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:[-*•]|\d+[\.)]|[a-zA-Z][\.)])\s+(.*)$", stripped)
        if match:
            items.append(match.group(1).strip())
    return items


def _normalize_rubrics(items: Iterable[Any], dedupe: bool = True) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _clean_rubric_text(str(item))
        key = re.sub(r"\s+", " ", text.lower())
        if text and (not dedupe or key not in seen):
            normalized.append(text)
            seen.add(key)
    return normalized


def _clean_rubric_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(?:[-*•]|\d+[\.)]|[a-zA-Z][\.)])\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'")


def _call_verifier(
    verifier: Verifier | Callable[..., int | bool],
    rubric: str,
    prompt: str | None = None,
) -> int:
    try:
        if callable(verifier) and not hasattr(verifier, "judge"):
            value = verifier(rubric, prompt=prompt)
        else:
            value = verifier.judge(rubric, prompt=prompt)  # type: ignore[union-attr]
    except TypeError:
        if callable(verifier) and not hasattr(verifier, "judge"):
            value = verifier(rubric)
        else:
            value = verifier.judge(rubric)  # type: ignore[union-attr]
    return int(bool(value))


def _as_2d_float(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={arr.shape}")
    return arr


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = array / norms
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def safe_float(value: float) -> float:
    """Convert NaN/inf to 0 for robust JSON reports."""

    return value if math.isfinite(value) else 0.0
