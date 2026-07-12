"""Small OpenAI-compatible API utilities for BlindSpot-RL data generation."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import fields
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LLMConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: int = 120
    reasoning_effort: str | None = None


class OpenAICompatibleClient:
    """Minimal chat-completions client.

    This avoids binding the project to one vendor SDK. Any endpoint compatible
    with `/chat/completions` can be used by changing config JSONL.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var {self.config.api_key_env} for provider {self.config.name}."
            )

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        
        # Responses API bypass: if base_url ends with "responses", use the responses-style payload.
        if self.config.base_url.rstrip("/").endswith("/responses"):
            url = self.config.base_url
            content_text = ""
            for m in messages:
                role = m.get("role", "")
                text = m.get("content", "")
                if role == "system":
                    content_text += f"[System]\n{text}\n\n"
                elif role == "user":
                    content_text += f"[User]\n{text}\n\n"
                elif role == "assistant":
                    content_text += f"[Assistant]\n{text}\n\n"
            
            payload = {
                "model": self.config.model,
                "input": [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": content_text.strip()}]
                }],
                "max_output_tokens": self.config.max_tokens,
            }
            if self.config.reasoning_effort:
                payload["reasoning"] = {"effort": self.config.reasoning_effort, "summary": "detailed"}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "X-Request-ID": f"blindspot_{int(time.time()*1000)}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    output_list = data.get("output") or []
                    for out_item in output_list:
                        content_list = out_item.get("content") or []
                        for item in content_list:
                            if item.get("type") == "output_text":
                                return str(item.get("text"))
                    incomplete = data.get("incomplete_details") or {}
                    reason = incomplete.get("reason")
                    status = data.get("status")
                    raise RuntimeError(
                        "Responses API returned no output_text "
                        f"(status={status!r}, incomplete_reason={reason!r}, "
                        f"max_output_tokens={data.get('max_output_tokens')!r}). "
                        "Raise max_tokens for reasoning models so reasoning does not "
                        "exhaust the output budget before an answer is emitted."
                    )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Responses API HTTP {exc.code}: {detail}") from exc

        # Standard OpenAI payload
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        return str(data["choices"][0]["message"]["content"])


class APIMetaVerifier:
    """Verifier used by R_valid and SFT hallucination filtering."""

    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def judge(self, rubric: str, **kwargs: Any) -> int:
        query = str(kwargs.get("prompt") or kwargs.get("query") or "")
        content = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation-criteria verifier. Return JSON only: "
                        "{\"valid\": true/false, \"reason\": \"...\"}. "
                        "A valid criterion is atomic, directly relevant to the query, "
                        "and can be judged yes/no from an answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query:\n{query}\n\nCandidate criterion:\n{rubric}",
                },
            ]
        )
        return int(parse_valid_flag(content))


def load_llm_configs(path: str | os.PathLike[str]) -> list[LLMConfig]:
    configs = []
    allowed_keys = {field.name for field in fields(LLMConfig)}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            config_obj = {key: value for key, value in obj.items() if key in allowed_keys}
            try:
                configs.append(LLMConfig(**config_obj))
            except TypeError as exc:
                raise ValueError(f"Invalid LLM config at line {line_no}: {obj}") from exc
    return configs


def parse_valid_flag(text: str) -> bool:
    """Parse verifier output robustly."""

    candidate = extract_json(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "valid" in obj:
                return parse_bool_flag(obj["valid"])
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    if "\"valid\": true" in lowered or "'valid': true" in lowered:
        return True
    if "\"valid\": false" in lowered or "'valid': false" in lowered:
        return False
    return lowered.strip() in {"true", "valid", "1", "yes"}


def parse_bool_flag(value: Any) -> bool:
    """Parse verifier booleans without treating arbitrary strings as truthy."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "valid", "1", "yes"}:
            return True
        if normalized in {"false", "invalid", "0", "no"}:
            return False
    return False


def parse_score(text: str) -> float:
    """Parse a 0-1 score from JSON or plain text."""

    candidate = extract_json(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "score" in obj:
                return clamp01(float(obj["score"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    stripped = text.strip()
    try:
        return clamp01(float(stripped))
    except ValueError:
        pass
    lowered = stripped.lower()
    if "yes" in lowered or "satisfied" in lowered:
        return 1.0
    if "no" in lowered or "not satisfied" in lowered:
        return 0.0
    return 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def extract_json(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        return stripped[start : end + 1]
    return None


def sleep_between_calls(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
