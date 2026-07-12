"""Policy-level reward hook using generated rubrics as the reward source."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from blindspot_rl.judge_eval import KeywordRubricScorer, score_answer
from blindspot_rl.reward_bsc import parse_rubrics


_RUBRIC_CACHE: dict[str, Any] | None = None
_RUBRIC_CACHE_PATH: str | None = None
_SCORER = KeywordRubricScorer()


def compute_score(
    data_source: str | None = None,
    solution_str: Any | None = None,
    ground_truth: Any | None = None,
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """verl-style policy reward entry point.

    The policy rollout answer is scored against rubrics for the same prompt.
    Rubrics can come from, in priority order:
    - `extra_info["rubrics"]`, `extra_info["generated_rubrics"]`, or `ground_truth`
    - a query-to-rubrics file pointed to by `BSC_POLICY_RUBRIC_FILE`

    The evaluation-criteria policy checkpoint path can be passed through
    `BSC_POLICY_CRITERIA_POLICY_CHECKPOINT`; this hook records the contract but does not
    serve models itself. Real runs should generate/cache rubrics before policy
    GRPO or provide a reward server that wraps this logic.
    """

    del data_source
    extra = dict(extra_info or {})
    prompt = str(extra.get("prompt") or extra.get("query") or kwargs.get("prompt") or "")
    answer = solution_str if solution_str is not None else kwargs.get("response", "")
    rubrics = (
        extra.get("rubrics")
        or extra.get("generated_rubrics")
        or extra.get("model_rubrics")
        or ground_truth
        or kwargs.get("rubrics")
        or lookup_rubrics(prompt)
    )
    parsed = parse_rubrics(rubrics, dedupe=True)
    if not prompt or not str(answer).strip() or not parsed:
        return float(os.environ.get("BSC_POLICY_MISSING_REWARD", "-1.0"))
    return score_answer(prompt, str(answer), parsed, _SCORER)


def lookup_rubrics(prompt: str) -> Any:
    if not prompt:
        return None
    cache = load_rubric_cache()
    return cache.get(make_key(prompt))


def load_rubric_cache() -> dict[str, Any]:
    global _RUBRIC_CACHE, _RUBRIC_CACHE_PATH
    path = os.environ.get("BSC_POLICY_RUBRIC_FILE", "")
    if not path:
        return {}
    if _RUBRIC_CACHE is not None and _RUBRIC_CACHE_PATH == path:
        return _RUBRIC_CACHE
    _RUBRIC_CACHE_PATH = path
    _RUBRIC_CACHE = {}
    for record in load_records(Path(path)):
        query = pick_first(record, "query", "prompt", "instruction")
        rubrics = pick_first(record, "rubrics", "generated_rubrics", "model_rubrics", "gold_rubrics", "response")
        if query and rubrics is not None:
            _RUBRIC_CACHE[make_key(query)] = rubrics
    return _RUBRIC_CACHE


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ensure_record(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return records
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [ensure_record(item) for item in unwrap_records(data)]
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        return [ensure_record(item) for item in pd.read_parquet(path).to_dict(orient="records")]
    raise ValueError(f"Unsupported rubric file format: {path}")


def unwrap_records(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["records", "data", "items", "examples", "rows"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return [data]


def ensure_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def make_key(value: Any) -> str:
    return " ".join(str(value).strip().split())


def reward_func(*args: Any, **kwargs: Any) -> float:
    """Alias for reward registries that expect `reward_func`."""

    return compute_score(*args, **kwargs)


def compute_reward(*args: Any, **kwargs: Any) -> float:
    """Alias for reward registries that expect `compute_reward`."""

    return compute_score(*args, **kwargs)
