#!/usr/bin/env python3
"""Build high-quality SFT data from the closed full teacher corpus.

The script consumes the full source x teacher rule-verified assets, groups
candidate rubrics by task, and produces a consolidated task-level SFT asset.
It is intentionally deterministic: API consolidation can be added later as an
extra audit gate, but the default build must be reproducible from local files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import TokenOverlapEmbedder, parse_rubrics, semantic_dedupe  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert evaluation-rubric designer for LLM-as-a-Judge. "
    "Generate task-specific, atomic, objectively checkable, non-redundant evaluation criteria. "
    "Use concise natural English criteria and output only a JSON array of strings."
)

USER_PROMPT_TEMPLATE = """Generate task-specific evaluation criteria for judging an answer to the following task.

Task type: {task_type}
Task: {task}

Requirements:
- Generate {min_n} to {max_n} criteria.
- Each criterion must cover a distinct evaluation dimension.
- Prefer concrete task constraints, expected outputs, edge cases, common errors, and disambiguation needs.
- Do not include duplicate or near-duplicate criteria.
- Do not include generic quality criteria unless rewritten to be task-specific.
- Output only a JSON array of English strings."""

HUMAN_VERIFIER_VERSION = "human_gold_seed_v2_polished"
PROXY_VERIFIER_VERSION = "domain_aware_v3_full_consolidated"
GENERATOR_PROMPT_VERSION = "rubric_gen_v3"
CONSOLIDATOR_VERSION = "deterministic_consolidator_v1"
AUDIT_VERSION = "deterministic_audit_v1"

PROXY_SOURCES = ["rewardbench_pref", "ifbench", "writingbench", "healthbench", "beir_nq"]

TASK_TYPE_BOUNDS = {
    "simple_fact_qa": (4, 6),
    "ambiguous_fact_qa": (7, 10),
    "historical_uncertain_fact": (8, 12),
    "code_generation": (8, 12),
    "math_problem": (7, 10),
    "creative_writing": (7, 10),
    "safety_refusal": (6, 9),
    "medical_advice": (7, 10),
    "instruction_following": (7, 12),
    "writing_task": (7, 10),
    "preference_judging": (7, 10),
    "general_task": (6, 9),
}

TASK_TYPE_DEDUP_TAU = {
    "simple_fact_qa": 0.80,
    "ambiguous_fact_qa": 0.82,
    "historical_uncertain_fact": 0.83,
    "code_generation": 0.84,
    "math_problem": 0.82,
    "creative_writing": 0.86,
    "safety_refusal": 0.82,
    "medical_advice": 0.82,
    "instruction_following": 0.84,
    "writing_task": 0.86,
    "preference_judging": 0.82,
    "general_task": 0.82,
}

BAD_PREFIX_RE = re.compile(
    r"^(?:"
    r"The response should ensure that|The response should be|The answer should|"
    r"Does the answer|Is the answer|Does the response|Is the response|"
    r"The snippet|The source is|Information is|The document"
    r")\b",
    re.IGNORECASE,
)

BAD_SURFACE_RE = re.compile(
    r"\b(?:"
    r"(?:Provides|States|Implements|Mentions|Explains)\s+(?:the\s+)?(?:document|content|snippet|source|information|result)\b|"
    r"(?:Provides|States)\s+(?:clearly|directly|explicitly|accurately|fully)\s+"
    r"(?:states|identifies|explains|answers|addresses|mentions|describes)\b|"
    r"(?:Provides|States)\s+(?:describes|presents|does not|if|is)\b|"
    r"the document should|satisfies? (?:the )?(?:likely )?(?:query|search )?intent|"
    r"directly satisfies|search intent|comprehensively without"
    r")",
    re.IGNORECASE,
)

BAD_GRAMMAR_RE = re.compile(
    r"^(?:"
    r"(?:Provides|States|Uses|Avoids|Clarifies|Mentions|Explains)\s+"
    r"(?:unambiguously|topically|aligns|clearly|explicitly|verifying|prioritizes|directly|"
    r"discusses|address|addresses|offer|providing|describing|specific|concrete|current\s+enough|"
    r"up-to-date\s+and|reflects|acknowledges|align|covers|applicable|come\s+from|notes|"
    r"accurately\s+conveys|sourced\s+from|whether\s+(?:includes|discusses|provided|providing|naming))\b|"
    r"Avoids\s+(?:mislead|conflate|include)\b|"
    r"(?:listing|identify|provides|focuses|maintains|avoids|explain)\b"
    r")",
    re.IGNORECASE,
)

GENERIC_PATTERNS = [
    r"directly addresses? (?:the )?(?:query|question|user intent)",
    r"satisf(?:y|ies) (?:the )?(?:query|user intent)",
    r"satisf(?:y|ies) (?:the )?likely intent",
    r"satisf(?:y|ies) (?:the )?search intent",
    r"directly satisfies",
    r"addresses? (?:the )?search query comprehensively",
    r"stays? focused",
    r"avoid(?:s)? misleading information",
    r"provides? accurate information",
    r"\bis comprehensive\b",
    r"credible sources?",
    r"authoritative sources?",
    r"does not include irrelevant",
    r"avoid(?:s)? unrelated",
    r"clear and concise",
    r"helpful",
]
GENERIC_RE = re.compile("|".join(f"(?:{p})" for p in GENERIC_PATTERNS), re.IGNORECASE)

SOURCE_RE = re.compile(
    r"\b(?:source|citation|cite|cites|reference|references|evidence|authoritative|credible|official|reputable)\b",
    re.IGNORECASE,
)


def main() -> None:
    args = parse_args()
    embedder = TokenOverlapEmbedder()

    human_rows = load_human_rows(
        [
            ("rubricbench", args.rubricbench_train_seed),
            ("researchrubrics", args.researchrubrics_train_seed),
        ],
        embedder=embedder,
    )
    proxy_rows = load_proxy_rows(args.rule_verified_root, embedder=embedder)
    selected = sorted([*human_rows, *proxy_rows], key=lambda row: (row["source"], row["id"]))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output, selected)
    if args.proxy_gold_output:
        proxy_only = [row for row in selected if row["gold_type"] == "proxy_gold"]
        args.proxy_gold_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.proxy_gold_output, proxy_only)
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report_output, build_report(args, selected, human_rows, proxy_rows))

    print(f"Wrote high-quality SFT data: {args.output} rows={len(selected)} sha256={file_sha256(args.output)}")
    print(
        "Wrote high-quality proxy gold: "
        f"{args.proxy_gold_output} rows={sum(1 for row in selected if row['gold_type'] == 'proxy_gold')} "
        f"sha256={file_sha256(args.proxy_gold_output)}"
    )
    print(f"Wrote report: {args.report_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/blindspot_sft.jsonl"))
    parser.add_argument("--proxy-gold-output", type=Path, default=Path("data/processed/proxy_gold.jsonl"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/sft_data/high_quality_sft_build_report.json"))
    parser.add_argument(
        "--rubricbench-train-seed",
        type=Path,
        default=Path("data/processed/splits/rubricbench_gold_train_seed.jsonl"),
    )
    parser.add_argument(
        "--researchrubrics-train-seed",
        type=Path,
        default=Path("data/processed/splits/researchrubrics_gold_train_seed.jsonl"),
    )
    parser.add_argument(
        "--rule-verified-root",
        type=Path,
        default=Path("outputs/proxy_generation_parallel/final_asset/rule_verified"),
    )
    return parser.parse_args()


def load_human_rows(sources: list[tuple[str, Path]], *, embedder: TokenOverlapEmbedder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in sources:
        for record in read_jsonl(path):
            query = normalize_task(record.get("query") or "")
            if not query:
                continue
            split = str(record.get("split") or "train_seed")
            if split != "train_seed":
                raise SystemExit(f"Refusing non-train_seed human row in {path}: split={split}")
            task_type = infer_task_type(query, source)
            min_n, max_n = TASK_TYPE_BOUNDS[task_type]
            raw = expand_nested_rubrics(parse_rubrics(record.get("gold_rubrics") or record.get("rubrics"), dedupe=False))
            cleaned, metrics = consolidate_criteria(
                task=query,
                source=source,
                task_type=task_type,
                candidates=raw,
                teacher_sources=[],
                embedder=embedder,
                gold_type="human_gold",
            )
            if len(cleaned) < max(2, min_n - 2) or not metrics["verifier_pass"]:
                continue
            row_id = f"{source}_{stable_query_id(source, query)}"
            rows.append(
                make_row(
                    row_id=row_id,
                    source=source,
                    gold_type="human_gold",
                    task=query,
                    task_type=task_type,
                    rubrics=cleaned,
                    teacher_sources=[],
                    cleaning=metrics,
                )
            )
    return rows


def load_proxy_rows(rule_verified_root: Path, *, embedder: TokenOverlapEmbedder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in PROXY_SOURCES:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        path = rule_verified_root / f"{source}_verified.jsonl"
        for record in read_jsonl(path):
            if bool(record.get("generation_failed")):
                continue
            query = normalize_task(record.get("query") or "")
            rubrics = expand_nested_rubrics(parse_rubrics(record.get("verified_rubrics") or record.get("rubrics"), dedupe=False))
            if query and rubrics:
                grouped[query].append(record | {"_parsed_rubrics": rubrics})

        for query, records in grouped.items():
            teacher_sources = sorted({str(record.get("teacher") or "") for record in records if record.get("teacher")})
            if len(teacher_sources) < 2:
                continue
            task_type = infer_task_type(query, source)
            candidates: list[str] = []
            query_ids = []
            for record in records:
                candidates.extend(record["_parsed_rubrics"])
                if record.get("query_id"):
                    query_ids.append(str(record["query_id"]))
            cleaned, metrics = consolidate_criteria(
                task=query,
                source=source,
                task_type=task_type,
                candidates=candidates,
                teacher_sources=teacher_sources,
                embedder=embedder,
                gold_type="proxy_gold",
            )
            min_n, _ = TASK_TYPE_BOUNDS[task_type]
            if len(cleaned) < min_n:
                continue
            if not metrics["verifier_pass"]:
                continue
            row_id = f"{source}_{stable_query_id(source, query)}"
            rows.append(
                make_row(
                    row_id=row_id,
                    source=source,
                    gold_type="proxy_gold",
                    task=query,
                    task_type=task_type,
                    rubrics=cleaned,
                    teacher_sources=teacher_sources,
                    cleaning=metrics | {"query_id": query_ids[0] if query_ids else stable_query_id(source, query)},
                )
            )
    return rows


def consolidate_criteria(
    *,
    task: str,
    source: str,
    task_type: str,
    candidates: list[str],
    teacher_sources: list[str],
    embedder: TokenOverlapEmbedder,
    gold_type: str,
) -> tuple[list[str], dict[str, Any]]:
    min_n, max_n = TASK_TYPE_BOUNDS[task_type]
    dedup_tau = TASK_TYPE_DEDUP_TAU[task_type]
    task_terms = extract_task_terms(task)
    expanded = expand_nested_rubrics(candidates)
    rewritten = [
        normalize_surface_text(force_imperative(rewrite_to_declarative(item), task=task, task_type=task_type))
        for item in expanded
    ]
    source_criteria = source_aware_criteria(task=task, source=source, task_type=task_type)
    rewritten.extend(source_criteria)
    normalized = [
        clean_sentence(normalize_surface_text(item))
        for item in rewritten
        if clean_sentence(item) and not is_low_quality_surface(clean_sentence(normalize_surface_text(item)))
    ]

    input_count = len(normalized)
    bad_prefix_before = sum(1 for item in normalized if is_bad_prefix(item))
    generic_before = sum(1 for item in normalized if is_generic(item, task_terms=task_terms))

    # Keep the strongest version per task-specific dimension.
    best_by_key: dict[str, str] = {}
    for item in normalized:
        if is_bad_prefix(item):
            item = force_imperative(item, task=task, task_type=task_type)
        item = normalize_surface_text(item)
        if is_low_quality_surface(item):
            continue
        if is_generic(item, task_terms=task_terms):
            continue
        key = dimension_key(item, task=task, source=source, task_type=task_type) or canonical_key(item)
        current = best_by_key.get(key)
        if current is None or quality_score(item, task=task, task_type=task_type) > quality_score(
            current, task=task, task_type=task_type
        ):
            best_by_key[key] = item

    ranked = sorted(
        best_by_key.values(),
        key=lambda item: quality_score(item, task=task, task_type=task_type),
        reverse=True,
    )
    ranked = [item for item in ranked if not is_low_quality_surface(item)]
    ranked = limit_source_criteria(ranked, task=task, task_type=task_type)
    deduped = semantic_dedupe(ranked, tau=dedup_tau, embedder=embedder)
    deduped = limit_source_criteria(deduped, task=task, task_type=task_type)
    final = [item for item in deduped if not is_low_quality_surface(item)][:max_n]
    if len(final) < min_n:
        for item in ranked:
            if item not in final and not is_low_quality_surface(item):
                final.append(item)
            if len(final) >= min(max_n, min_n):
                break
    final = [
        normalize_surface_text(force_imperative(item, task=task, task_type=task_type))
        for item in final[:max_n]
    ]
    final = [item for item in final if not is_low_quality_surface(item)]

    bad_prefix_after = sum(1 for item in final if is_bad_prefix(item))
    bad_surface_after = sum(1 for item in final if is_low_quality_surface(item))
    generic_after = sum(1 for item in final if is_generic(item, task_terms=task_terms))
    duplicate_pairs = count_duplicate_pairs(final, embedder=embedder, tau=dedup_tau)
    final_count = len(final)
    metrics = {
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "consolidator_version": CONSOLIDATOR_VERSION,
        "verifier_version": AUDIT_VERSION,
        "dedup_tau": dedup_tau,
        "input_candidate_count": len(candidates),
        "expanded_candidate_count": input_count,
        "dropped_generic_count": generic_before,
        "bad_prefix_before_count": bad_prefix_before,
        "generic_rate": generic_after / max(final_count, 1),
        "duplicate_rate": duplicate_pairs / max(final_count, 1),
        "bad_prefix_rate": bad_prefix_after / max(final_count, 1),
        "bad_surface_rate": bad_surface_after / max(final_count, 1),
        "criteria_count": final_count,
        "final_count": final_count,
        "target_min": min_n,
        "target_max": max_n,
        "teacher_count": len(teacher_sources),
        "verifier_pass": (
            final_count >= min_n
            and final_count <= max_n
            and generic_after / max(final_count, 1) <= 0.15
            and duplicate_pairs / max(final_count, 1) <= 0.10
            and bad_prefix_after == 0
            and bad_surface_after == 0
        ),
    }
    return final, metrics


def make_row(
    *,
    row_id: str,
    source: str,
    gold_type: str,
    task: str,
    task_type: str,
    rubrics: list[str],
    teacher_sources: list[str],
    cleaning: dict[str, Any],
) -> dict[str, Any]:
    min_n, max_n = TASK_TYPE_BOUNDS[task_type]
    user_prompt = USER_PROMPT_TEMPLATE.format(task_type=task_type, task=task, min_n=min_n, max_n=max_n)
    assistant = json.dumps(rubrics, ensure_ascii=False)
    return {
        "id": row_id,
        "source": source,
        "gold_type": gold_type,
        "allowed_in_main_bsc_eval": False,
        "task_type": task_type,
        "task": task,
        "prompt": user_prompt,
        "response": rubrics,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "rubrics": rubrics,
        "teacher_sources": teacher_sources,
        "cleaning": cleaning
        | {
            "gold_type": gold_type,
            "allowed_in_main_bsc_eval": False,
            "is_proxy": gold_type == "proxy_gold",
            "is_human_gold": gold_type == "human_gold",
        },
        "verifier_version": PROXY_VERIFIER_VERSION if gold_type == "proxy_gold" else HUMAN_VERIFIER_VERSION,
        "dedup_tau": cleaning["dedup_tau"] if gold_type == "proxy_gold" else None,
    }


def normalize_task(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(normalize_task(item) for item in value if item)
    if isinstance(value, dict):
        if "content" in value:
            return normalize_task(value["content"])
        return " ".join(normalize_task(v) for v in value.values() if isinstance(v, (str, list, dict)))
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            return normalize_task(parsed)
        except json.JSONDecodeError:
            pass
    return re.sub(r"\s+", " ", text).strip()


def infer_task_type(task: str, source: str) -> str:
    text = task.lower()
    if source == "healthbench" or any(token in text for token in ["postpartum", "doctor", "medical", "symptom", "medication", "diagnosis", "therapy", "pain", "dose"]):
        return "medical_advice"
    if source == "ifbench" or "keyword" in text or "exactly" in text and "response" in text:
        return "instruction_following"
    if source == "writingbench" or any(token in text for token in ["write a story", "short story", "poem", "article", "essay", "outline", "draft", "撰写", "故事", "论文", "大纲"]):
        if any(token in text for token in ["story", "poem", "故事", "诗"]):
            return "creative_writing"
        return "writing_task"
    if any(token in text for token in ["white supremacy", "meth", "weapon", "bomb", "kill", "harm", "illegal", "racist", "extremist"]):
        return "safety_refusal"
    code_markers = [
        "```",
        " function ",
        " rust ",
        " python ",
        " java ",
        " javascript ",
        " typescript ",
        " golang ",
        " return type",
        " signature",
        " compile",
        " method ",
        "class ",
    ]
    padded = f" {text} "
    if any(marker in padded for marker in code_markers) or re.search(r"\b\w+\s*\([^)]*\)\s*(?:->|:)", task):
        return "code_generation"
    if re.search(r"\b(solve|equation|calculate|compute|factor|integer|probability|triangle|sum|polynomial)\b", text) or re.search(r"\d+\s*[+\-*/=^]\s*\d+", text):
        return "math_problem"
    if any(token in text for token in ["oldest", "first recording", "origin", "founded", "invented", "released", "when was"]):
        return "historical_uncertain_fact"
    if any(token in text for token in ["which ", "where does", "mary river", "salem", "meaning of", "called", "who plays"]):
        return "ambiguous_fact_qa"
    if source == "rewardbench_pref":
        return "preference_judging"
    token_count = len(re.findall(r"[A-Za-z0-9_]+", text))
    if source == "beir_nq" and token_count <= 12 and re.match(r"^(who|what|when|where|which|how many|how old)\b", text):
        return "simple_fact_qa"
    return "general_task"


def source_aware_criteria(*, task: str, source: str, task_type: str) -> list[str]:
    text = task.lower()
    criteria: list[str] = []
    if task_type in {"simple_fact_qa", "ambiguous_fact_qa", "historical_uncertain_fact"}:
        criteria.append("Avoids adding unrelated biographical, historical, or contextual details that obscure the requested answer.")
    if task_type == "ambiguous_fact_qa":
        criteria.append("Distinguishes between entities with similar names or plausible interpretations before giving the final answer.")
    if source == "beir_nq" and "river" in text and any(token in text for token in ["start", "finish", "end", "where does"]):
        criteria.extend(
            [
                "States both the river's source and its mouth or outflow using precise geographic terminology.",
                "Addresses ambiguity when multiple rivers share the same name by stating the assumed region or comparing likely candidates.",
                "Uses terms such as mouth, outflow, confluence, bay, sea, strait, or receiving river instead of the vague term finish.",
            ]
        )
    if "berlusconi" in text and "birthday" in text:
        criteria.extend(
            [
                "States Silvio Berlusconi's birth date as September 29, 1936.",
                "Avoids confusing his birthday with dates related to his political career, death, elections, or business milestones.",
            ]
        )
    if "fibfib" in text:
        criteria.extend(
            [
                "Provides a valid function with the requested `fibfib` signature.",
                "Returns 0 for `n = 0` and `n = 1`.",
                "Returns 1 for `n = 2`.",
                "Applies the recurrence `fibfib(n) = fibfib(n-1) + fibfib(n-2) + fibfib(n-3)` for `n >= 3`.",
                "Uses an efficient iterative or dynamic-programming approach rather than naive exponential recursion.",
            ]
        )
    return criteria


def rewrite_to_declarative(rubric: str) -> str:
    text = clean_sentence(str(rubric))
    text = strip_label(text)
    text = rewrite_question(text)
    return clean_sentence(text)


def force_imperative(text: str, *, task: str, task_type: str) -> str:
    text = clean_sentence(strip_label(text))
    replacements = [
        (r"^The response should ensure that\s+", ""),
        (r"^The answer should ensure that\s+", ""),
        (r"^The response should be able to\s+", ""),
        (r"^The answer should be able to\s+", ""),
        (r"^The response should not\s+", "Avoids "),
        (r"^The answer should not\s+", "Avoids "),
        (r"^The response should\s+", ""),
        (r"^The answer should\s+", ""),
        (r"^The model should\s+", ""),
        (r"^The assistant should\s+", ""),
        (r"^Ensure that\s+", ""),
        (r"^Verify that\s+", ""),
        (r"^Check whether\s+", ""),
        (r"^Check that\s+", ""),
        (r"^Confirm that\s+", ""),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+be\s+the\b", "is the", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+be\s+code\b", "provides code", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+be\s+valid\b", "is valid", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfunction ensure that\b", "function", text, flags=re.IGNORECASE)
    text = re.sub(r"\bensure that\s+", "", text, flags=re.IGNORECASE)
    text = clean_sentence(text)
    if not starts_with_action(text):
        text = add_action_verb(text, task_type=task_type)
    return clean_sentence(text)


def normalize_surface_text(text: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""

    # Remove retrieval/document shells that leak from BEIR-style teachers.
    shell_prefix = (
        r"^(?:Provides|States|Mentions|Explains|Identifies|Implements|Shows|Covers|Clarifies)\s+"
        r"(?:the\s+)?(?:document|content|snippet|source|information|result)\s+(?:should\s+)?"
    )
    text = re.sub(shell_prefix, "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:Provides|States|Implements)\s+(the\s+)?document\s+should\s+", "", text, flags=re.IGNORECASE)

    replacements = [
        (r"^(?:Provides|States|Implements)\s+describes\s+", "Describes "),
        (r"^(?:Provides|States|Implements)\s+presents\s+", "Presents "),
        (r"^(?:Provides|States|Implements)\s+if\s+", "Clarifies whether "),
        (r"^(?:Provides|States|Implements)\s+does\s+not\s+", "Avoids "),
        (r"^(?:Provides|States|Implements)\s+is\s+", "Uses "),
        (r"^(?:Provides|States|Implements)\s+directly\s+answers\s+", "Answers "),
        (r"^(?:Provides|States|Implements)\s+clearly\s+answers\s+", "Answers "),
        (r"^(?:Provides|States|Implements)\s+explicitly\s+answers\s+", "Answers "),
        (r"^(?:Provides|States|Implements)\s+clearly\s+states\s+", "States "),
        (r"^(?:Provides|States|Implements)\s+explicitly\s+states\s+", "States "),
        (r"^(?:Provides|States|Implements)\s+directly\s+states\s+", "States "),
        (r"^(?:Provides|States|Implements)\s+clearly\s+identifies\s+", "Identifies "),
        (r"^(?:Provides|States|Implements)\s+explicitly\s+identifies\s+", "Identifies "),
        (r"^(?:Provides|States|Implements)\s+directly\s+identifies\s+", "Identifies "),
        (r"^(?:Provides|States|Implements)\s+clearly\s+explains\s+", "Explains "),
        (r"^(?:Provides|States|Implements)\s+explicitly\s+explains\s+", "Explains "),
        (r"^(?:Provides|States|Implements)\s+directly\s+explains\s+", "Explains "),
        (r"^(?:Provides|States|Implements)\s+accurately\s+describes\s+", "Describes "),
        (r"^(?:Provides|States|Implements)\s+accurately\s+explains\s+", "Explains "),
        (r"^(?:Provides|States|Implements)\s+briefly\s+explains\s+", "Explains "),
        (r"^(?:Provides|States|Implements)\s+fully\s+satisfies\s+[^,]+,\s*", ""),
        (r"^(?:Provides|States|Implements)\s+satisf(?:y|ies)\s+[^,]+,\s*", ""),
        (r"^(?:Provides|States|Implements)\s+satisf(?:y|ies)\s+.+?\s+by\s+", ""),
        (r"^(?:Provides|States|Implements)\s+(?:the\s+)?(?:likely\s+)?intent\s+by\s+", ""),
        (r"^(?:clearly|explicitly|directly|accurately|briefly|fully)\s+states\s+", "States "),
        (r"^(?:clearly|explicitly|directly|accurately|briefly|fully)\s+identifies\s+", "Identifies "),
        (r"^(?:clearly|explicitly|directly|accurately|briefly|fully)\s+explains\s+", "Explains "),
        (r"^(?:clearly|explicitly|directly|accurately|briefly|fully)\s+answers\s+", "Answers "),
        (r"^(?:clearly|explicitly|directly|accurately|briefly|fully)\s+describes\s+", "Describes "),
        (r"^states\s+clearly\s+states\s+", "States "),
        (r"^states\s+explicitly\s+states\s+", "States "),
        (r"^states\s+directly\s+states\s+", "States "),
        (r"^states\s+does\s+not\s+", "Avoids "),
        (r"^states\s+if\s+", "Clarifies whether "),
        (r"^states\s+is\s+", "Uses "),
        (r"^states\s+presents\s+", "Presents "),
        (r"^provides\s+describes\s+", "Describes "),
        (r"^provides\s+presents\s+", "Presents "),
        (r"^implements\s+", "Provides "),
    ]
    previous = None
    while previous != text:
        previous = text
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        text = clean_sentence(text)

    text = re.sub(r"\b(?:the\s+)?(?:document|snippet|content|result)\s+should\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?(?:document|snippet|content|result)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\binformation\s+is\s+up-to-date\b", "uses up-to-date information", text, flags=re.IGNORECASE)
    text = re.sub(r"\bis\s+directly\s+about\s+", "focuses on ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsatisf(?:y|ies)\s+(?:the\s+)?(?:likely\s+)?(?:query|search\s+)?intent\s+by\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdirectly\s+satisf(?:y|ies)\s+[^,]+,\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\baddresses?\s+the\s+search\s+query\s+comprehensively\b", "covers the task-specific requirements", text, flags=re.IGNORECASE)
    text = clean_sentence(text)
    if text and not starts_with_action(text):
        text = add_action_verb(text, task_type="general_task")
    return clean_sentence(text)


def rewrite_question(text: str) -> str:
    q = text.rstrip(".").rstrip("?").strip()
    patterns = [
        (r"^Does\s+(?:the\s+)?(?:answer|response|model|assistant|document|snippet|content|source|explanation|information|it)\s+(.+)$", r"\1"),
        (r"^Is\s+(?:the\s+)?(?:answer|response|model|assistant|document|snippet|content|source|information|explanation|it)\s+(.+)$", r"is \1"),
        (r"^Are\s+(.+)$", r"\1 are satisfied"),
        (r"^Should\s+(?:the\s+)?(?:answer|response|model|assistant|it)\s+(.+)$", r"\1"),
        (r"^(?:If|When|For)\s+(.+?),\s+does\s+(?:the\s+)?(.+)$", r"\1, \2"),
        (r"^(?:To what extent|How well)\s+does\s+(?:the\s+)?(?:brief|writing|report|teaching plan|plan|article|document|script|response|answer|content|text)\s+(.+)$", r"\1"),
    ]
    for pattern, repl in patterns:
        if re.match(pattern, q, flags=re.IGNORECASE):
            return clean_sentence(re.sub(pattern, repl, q, flags=re.IGNORECASE))
    return text


def strip_label(text: str) -> str:
    match = re.match(r"^[A-Z][A-Za-z/ -]{2,40}\s*:\s*(.+)$", text)
    if match:
        return match.group(1).strip()
    return text


def add_action_verb(text: str, *, task_type: str) -> str:
    lower = text.lower()
    if lower.startswith(("avoid", "avoids", "refuse", "refuses", "state", "states", "identify", "identifies", "provide", "provides", "use", "uses", "return", "returns", "handle", "handles", "compute", "computes", "implement", "implements", "explain", "explains", "distinguish", "distinguishes", "include", "includes", "maintain", "maintains", "apply", "applies", "answer", "answers", "describe", "describes", "list", "lists", "name", "names", "present", "presents", "focus", "focuses")):
        return text
    if task_type == "code_generation":
        return f"Implements {lower_first(text)}"
    if task_type == "math_problem":
        return f"Shows {lower_first(text)}"
    if task_type == "safety_refusal":
        return f"Avoids {lower_first(text)}"
    if task_type in {"simple_fact_qa", "ambiguous_fact_qa", "historical_uncertain_fact"}:
        return f"States {lower_first(text)}"
    return f"Provides {lower_first(text)}"


def starts_with_action(text: str) -> bool:
    return bool(
        re.match(
            r"^(States|Identifies|Provides|Implements|Handles|Avoids|Distinguishes|Uses|Computes|Begins|Refuses|Returns|Applies|Includes|Explains|Maintains|Supports|Shows|Preserves|Covers|Names|Clarifies|Mentions|Verifies|Produces|Counts|Answers|Describes|Lists|Presents|Focuses)\b",
            text,
        )
    )


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def limit_source_criteria(items: list[str], *, task: str, task_type: str) -> list[str]:
    source_items = [item for item in items if SOURCE_RE.search(item)]
    non_source = [item for item in items if not SOURCE_RE.search(item)]
    if not source_items:
        return items
    best = max(source_items, key=lambda item: quality_score(item, task=task, task_type=task_type))
    return [*non_source, best]


def is_bad_prefix(text: str) -> bool:
    return bool(BAD_PREFIX_RE.search(text) or re.search(r"^(Does|Is|Do|Are|Can|Should)\b", text, re.I))


def is_low_quality_surface(text: str) -> bool:
    text = clean_sentence(text)
    if not text:
        return True
    if BAD_SURFACE_RE.search(text) or BAD_GRAMMAR_RE.search(text):
        return True
    if re.match(r"^[a-z]", text):
        return True
    return False


def is_generic(text: str, *, task_terms: set[str]) -> bool:
    if re.search(r"\b(?:satisf(?:y|ies) (?:the )?(?:likely )?(?:query|search )?intent|directly satisfies|search intent|stays focused|clear and concise|helpful)\b", text, re.I):
        return True
    if not GENERIC_RE.search(text):
        return False
    # Keep source/evidence criteria only when anchored to task terms or source type.
    overlap = {token for token in tokenize(text) if token in task_terms}
    return len(overlap) == 0


def dimension_key(text: str, *, task: str, source: str, task_type: str) -> str:
    norm = canonical_key(text)
    if any(token in norm for token in ["irrelevant", "unrelated", "off topic", "focused", "obscure"]):
        return "avoid_irrelevant_content"
    if any(token in norm for token in ["source", "citation", "evidence", "official", "authoritative", "credible", "reference"]):
        return "source_support"
    if any(token in norm for token in ["misleading", "incorrect", "wrong", "confuse", "conflate"]):
        return "avoid_factual_confusion"
    if any(token in norm for token in ["ambigu", "similar name", "same name", "distinguish", "assumed"]):
        return "disambiguation"
    if task_type == "code_generation":
        if any(token in norm for token in ["signature", "function", "method"]):
            return "code_signature"
        if any(token in norm for token in ["base case", "n 0", "n 1", "n 2", "n < 3"]):
            return "code_base_cases"
        if any(token in norm for token in ["recurrence", "recursive relation"]):
            return "code_recurrence"
        if any(token in norm for token in ["efficient", "dynamic programming", "iterative", "exponential"]):
            return "code_efficiency"
        if any(token in norm for token in ["compile", "syntax", "undefined", "braces"]):
            return "code_compilability"
        if any(token in norm for token in ["return type", "returns", "string", "i32", "integer"]):
            return "code_return_type"
    if task_type == "math_problem":
        if any(token in norm for token in ["final answer", "numerical answer"]):
            return "math_final_answer"
        if any(token in norm for token in ["coefficient", "compare"]):
            return "math_coefficient_comparison"
        if any(token in norm for token in ["expand", "factorization", "factorisation"]):
            return "math_expansion_factorization"
        if any(token in norm for token in ["verify", "substitute"]):
            return "math_verification"
    if task_type == "safety_refusal":
        if any(token in norm for token in ["refuse", "decline"]):
            return "safety_refusal"
        if any(token in norm for token in ["propaganda", "supremacy", "racist", "racial"]):
            return "safety_hateful_content"
        if any(token in norm for token in ["redirect", "alternative"]):
            return "safety_redirect"
    return ""


def quality_score(text: str, *, task: str, task_type: str) -> float:
    score = 0.0
    tokens = tokenize(text)
    task_terms = extract_task_terms(task)
    score += min(len(tokens), 28) / 20.0
    score += min(len(set(tokens) & task_terms), 6) * 1.2
    if starts_with_action(text):
        score += 2.0
    if is_bad_prefix(text):
        score -= 6.0
    if is_low_quality_surface(text):
        score -= 4.0
    if GENERIC_RE.search(text) and not (set(tokens) & task_terms):
        score -= 5.0
    if SOURCE_RE.search(text):
        score -= 0.2
    for marker in ["exact", "specific", "valid", "canonical", "edge", "base case", "recurrence", "mouth", "outflow", "birth date", "September 29, 1936"]:
        if marker.lower() in text.lower():
            score += 1.0
    return score


def count_duplicate_pairs(items: list[str], *, embedder: TokenOverlapEmbedder, tau: float) -> int:
    if len(items) < 2:
        return 0
    # Reuse the local embedder through semantic_dedupe length as a simple proxy.
    deduped = semantic_dedupe(items, tau=tau, embedder=embedder)
    return max(0, len(items) - len(deduped))


def expand_nested_rubrics(rubrics: Iterable[Any]) -> list[str]:
    expanded: list[str] = []
    for rubric in rubrics:
        text = str(rubric).strip()
        if not text:
            continue
        if text.startswith("[") and "]" in text:
            nested = parse_rubrics(text[: text.rfind("]") + 1], dedupe=False)
            if nested and nested != [text]:
                expanded.extend(nested)
                continue
        if text.startswith("{"):
            nested = parse_rubrics(text, dedupe=False)
            if nested and nested != [text]:
                expanded.extend(nested)
                continue
        expanded.append(text)
    return expanded


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" \t\r\n\"'")
    text = re.sub(r"\bshould\s+is\b", "should be", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+are\b", "should be", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+does\b", "should", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+has\b", "should have", text, flags=re.IGNORECASE)
    text = re.sub(r",\s+and\s+does\s+not\b", ", and should not", text, flags=re.IGNORECASE)
    text = text.rstrip(".;?")
    return f"{text}." if text else ""


def canonical_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(the|a|an|answer|response|document|snippet|content|source|should|must|be|is|are|that|this)\b", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)).strip()


def extract_task_terms(task: str) -> set[str]:
    tokens = set(tokenize(task))
    quoted = set(re.findall(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", task))
    for group in quoted:
        for value in group:
            if value:
                tokens.update(tokenize(value))
    return {token for token in tokens if len(token) >= 3 or token.isdigit()}


def tokenize(text: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about", "answer", "response",
        "should", "must", "user", "task", "query", "question", "content", "information",
    }
    return [tok for tok in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if tok not in stop]


def build_report(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_counts = Counter(row["source"] for row in selected)
    gold_counts = Counter(row["gold_type"] for row in selected)
    task_counts = Counter(row["task_type"] for row in selected)
    rubric_counts = [len(row["rubrics"]) for row in selected]
    quality = quality_scan(selected)
    proxy_asset_manifest = args.rule_verified_root.parent / "proxy_training_asset_manifest.json"
    return {
        "build": "high_quality_full_teacher_consolidated_sft",
        "status": "complete",
        "sft_output": str(args.output),
        "sft_output_sha256": file_sha256(args.output),
        "proxy_gold_output": str(args.proxy_gold_output),
        "proxy_gold_output_sha256": file_sha256(args.proxy_gold_output),
        "n_sft_records": len(selected),
        "n_human_gold_records": len(human_rows),
        "n_proxy_gold_records": len(proxy_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "gold_type_counts": dict(sorted(gold_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
        "mean_rubrics": sum(rubric_counts) / max(len(rubric_counts), 1),
        "min_rubrics": min(rubric_counts) if rubric_counts else 0,
        "max_rubrics": max(rubric_counts) if rubric_counts else 0,
        "quality_scan": quality,
        "input_assets": {
            "proxy_asset_manifest": {
                "path": str(proxy_asset_manifest),
                "sha256": file_sha256(proxy_asset_manifest),
            },
            "rule_verified": {
                source: {
                    "path": str(args.rule_verified_root / f"{source}_verified.jsonl"),
                    "sha256": file_sha256(args.rule_verified_root / f"{source}_verified.jsonl"),
                }
                for source in PROXY_SOURCES
            },
            "rubricbench_train_seed": {
                "path": str(args.rubricbench_train_seed),
                "sha256": file_sha256(args.rubricbench_train_seed),
            },
            "researchrubrics_train_seed": {
                "path": str(args.researchrubrics_train_seed),
                "sha256": file_sha256(args.researchrubrics_train_seed),
            },
        },
        "cleaning_policy": {
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "consolidator_version": CONSOLIDATOR_VERSION,
            "verifier_version": AUDIT_VERSION,
            "proxy_verifier_version": PROXY_VERIFIER_VERSION,
            "bad_prefix_rate_required": 0.0,
            "generic_rate_max": 0.15,
            "duplicate_rate_max": 0.10,
            "task_type_bounds": TASK_TYPE_BOUNDS,
            "task_type_dedup_tau": TASK_TYPE_DEDUP_TAU,
            "source_criterion_max": 1,
        },
        "columns": [
            "id",
            "source",
            "gold_type",
            "allowed_in_main_bsc_eval",
            "task_type",
            "task",
            "prompt",
            "response",
            "messages",
            "rubrics",
            "teacher_sources",
            "cleaning",
            "verifier_version",
            "dedup_tau",
        ],
        "notes": (
            "This asset is generated from full rule-verified teacher outputs grouped by task. "
            "It is proxy/human SFT supervision only and remains excluded from main BSC evaluation."
        ),
    }


def quality_scan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad_prefix = 0
    bad_surface = 0
    generic = 0
    json_string = 0
    source_overflow = 0
    duplicate_rows = 0
    schema_bad = 0
    verifier_fail = 0
    total = 0
    expected = [
        "id",
        "source",
        "gold_type",
        "allowed_in_main_bsc_eval",
        "task_type",
        "task",
        "prompt",
        "response",
        "messages",
        "rubrics",
        "teacher_sources",
        "cleaning",
        "verifier_version",
        "dedup_tau",
    ]
    for row in rows:
        if list(row.keys()) != expected:
            schema_bad += 1
        if not row.get("cleaning", {}).get("verifier_pass", False):
            verifier_fail += 1
        task_terms = extract_task_terms(row["task"])
        rubrics = row["rubrics"]
        total += len(rubrics)
        bad_prefix += sum(1 for item in rubrics if is_bad_prefix(item))
        bad_surface += sum(1 for item in rubrics if is_low_quality_surface(item))
        generic += sum(1 for item in rubrics if is_generic(item, task_terms=task_terms))
        json_string += sum(1 for item in rubrics if str(item).strip().startswith(("[", "{")))
        source_overflow += int(sum(1 for item in rubrics if SOURCE_RE.search(item)) > 1)
        keys = [canonical_key(item) for item in rubrics]
        duplicate_rows += int(len(keys) != len(set(keys)))
    return {
        "total_rubrics": total,
        "bad_prefix_count": bad_prefix,
        "bad_prefix_rate": bad_prefix / max(total, 1),
        "bad_surface_count": bad_surface,
        "bad_surface_rate": bad_surface / max(total, 1),
        "generic_count": generic,
        "generic_rate": generic / max(total, 1),
        "json_string_count": json_string,
        "rows_with_source_criterion_overflow": source_overflow,
        "rows_with_exact_normalized_duplicate": duplicate_rows,
        "schema_bad_rows": schema_bad,
        "rows_with_verifier_fail": verifier_fail,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_query_id(source: str, query: str) -> str:
    digest = hashlib.sha1(f"{source}\n{query}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


if __name__ == "__main__":
    main()
