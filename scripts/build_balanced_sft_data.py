#!/usr/bin/env python3
"""Build source-balanced SFT data from human seed and closed proxy assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import TokenOverlapEmbedder, parse_rubrics, semantic_dedupe  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402


PROMPT_TEMPLATE = (
    "请为以下任务生成一组高质量评估标准。要求：全面、原子化、可判定、低冗余、与任务相关。"
    "任务：{query}"
)

HUMAN_VERIFIER_VERSION = "human_gold_seed"
PROXY_VERIFIER_VERSION = "domain_aware_v2_polished"

DEFAULT_TARGET_COUNTS = {
    "human_seed": 560,
    "rewardbench_pref": 560,
    "ifbench": 280,
    "writingbench": 560,
    "healthbench": 420,
    "beir_nq": 420,
}

PROXY_SOURCES = ["rewardbench_pref", "ifbench", "writingbench", "healthbench", "beir_nq"]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    embedder = TokenOverlapEmbedder()

    human_rows = load_human_rows(
        [
            ("rubricbench", args.rubricbench_train_seed),
            ("researchrubrics", args.researchrubrics_train_seed),
        ],
        max_rubrics=args.max_rubrics,
    )
    proxy_by_source = {
        source: load_proxy_rows(
            source=source,
            path=args.rule_verified_root / f"{source}_verified.jsonl",
            dedup_tau=args.dedup_tau,
            max_rubrics=args.max_rubrics,
            min_rubrics=args.min_proxy_rubrics,
            embedder=embedder,
        )
        for source in PROXY_SOURCES
    }

    target_counts = dict(DEFAULT_TARGET_COUNTS)
    selected: list[dict[str, Any]] = []
    selected.extend(sample_rows(human_rows, target_counts["human_seed"], rng, allow_oversample=True))
    for source in PROXY_SOURCES:
        selected.extend(sample_rows(proxy_by_source[source], target_counts[source], rng, allow_oversample=False))

    selected = sorted(selected, key=lambda row: (row["source"], row["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)

    if args.proxy_gold_output:
        proxy_rows = [to_proxy_gold(row) for row in selected if row["gold_type"] == "proxy_gold"]
        args.proxy_gold_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.proxy_gold_output, proxy_rows)

    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report_output, build_report(args, selected, human_rows, proxy_by_source, target_counts))

    print(f"Wrote SFT data: {args.output} rows={len(selected)} sha256={file_sha256(args.output)}")
    if args.proxy_gold_output:
        print(
            "Wrote proxy gold: "
            f"{args.proxy_gold_output} rows={sum(1 for row in selected if row['gold_type'] == 'proxy_gold')} "
            f"sha256={file_sha256(args.proxy_gold_output)}"
        )
    if args.report_output:
        print(f"Wrote report: {args.report_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/blindspot_sft.jsonl"))
    parser.add_argument("--proxy-gold-output", type=Path, default=Path("data/processed/proxy_gold.jsonl"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/sft_data/proxy_gold_build_report.json"))
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
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dedup-tau", type=float, default=0.85)
    parser.add_argument("--max-rubrics", type=int, default=12)
    parser.add_argument("--min-proxy-rubrics", type=int, default=6)
    return parser.parse_args()


def load_human_rows(sources: list[tuple[str, Path]], *, max_rubrics: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in sources:
        for idx, record in enumerate(read_jsonl(path)):
            query = str(record.get("query") or "")
            rubrics = polish_human_rubrics(
                parse_rubrics(record.get("gold_rubrics") or record.get("rubrics"), dedupe=False),
                max_rubrics=max_rubrics,
            )
            if not query or not rubrics:
                continue
            split = str(record.get("split") or "train_seed")
            if split != "train_seed":
                raise SystemExit(f"Refusing non-train_seed human row in {path}: split={split}")
            rows.append(
                make_sft_row(
                    row_id=f"{source}_{stable_query_id(source, query)}",
                    source=source,
                    query=query,
                    response=rubrics,
                    gold_type="human_gold",
                    teacher_sources=[],
                    query_id=stable_query_id(source, query),
                    source_split="train_seed",
                    input_paths=[str(path)],
                )
            )
    return rows


def load_proxy_rows(
    *,
    source: str,
    path: Path,
    dedup_tau: float,
    max_rubrics: int,
    min_rubrics: int,
    embedder: TokenOverlapEmbedder,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(path):
        if bool(record.get("generation_failed")):
            continue
        rubrics = parse_rubrics(record.get("verified_rubrics") or record.get("rubrics"), dedupe=False)
        if not rubrics:
            continue
        query = str(record.get("query") or "")
        if not query:
            continue
        grouped[query].append(record | {"_parsed_rubrics": rubrics})

    rows: list[dict[str, Any]] = []
    for query, records in grouped.items():
        teacher_sources = sorted({str(record.get("teacher") or "") for record in records if record.get("teacher")})
        if len(teacher_sources) < 2:
            continue
        union: list[str] = []
        input_paths = set()
        query_ids = []
        for record in records:
            union.extend(record["_parsed_rubrics"])
            input_paths.add(str(path))
            if record.get("query_id"):
                query_ids.append(str(record["query_id"]))
        rubrics = polish_proxy_rubrics(
            query=query,
            source=source,
            rubrics=union,
            dedup_tau=dedup_tau,
            max_rubrics=max_rubrics,
            embedder=embedder,
        )
        if len(rubrics) < min_rubrics:
            continue
        query_id = query_ids[0] if query_ids else stable_query_id(source, query)
        rows.append(
            make_sft_row(
                row_id=f"{source}_{stable_query_id(source, query)}",
                source=source,
                query=query,
                response=rubrics,
                gold_type="proxy_gold",
                teacher_sources=teacher_sources,
                query_id=query_id,
                source_split="proxy_train",
                input_paths=sorted(input_paths),
            )
        )
    return rows


def polish_proxy_rubrics(
    *,
    query: str,
    source: str,
    rubrics: list[str],
    dedup_tau: float,
    max_rubrics: int,
    embedder: TokenOverlapEmbedder,
) -> list[str]:
    """Canonicalize multi-teacher rubrics into SFT-quality target criteria."""

    candidates = [
        force_declarative(rewrite_to_declarative(rubric))
        for rubric in expand_nested_rubrics(parse_rubrics(rubrics, dedupe=False))
    ]
    candidates.extend(source_aware_criteria(query=query, source=source))
    candidates = [criterion for criterion in candidates if criterion]

    best_by_dimension: dict[str, str] = {}
    for criterion in candidates:
        key = rubric_dimension_key(criterion, source=source)
        if not key:
            key = canonical_text_key(criterion)
        current = best_by_dimension.get(key)
        if current is None or rubric_quality_score(criterion, source=source, query=query) > rubric_quality_score(
            current, source=source, query=query
        ):
            best_by_dimension[key] = criterion

    ranked = sorted(
        best_by_dimension.values(),
        key=lambda item: rubric_quality_score(item, source=source, query=query),
        reverse=True,
    )
    deduped = semantic_dedupe(ranked, tau=dedup_tau, embedder=embedder)
    return deduped[:max_rubrics]


def polish_human_rubrics(rubrics: list[str], *, max_rubrics: int) -> list[str]:
    rewritten = [force_declarative(rewrite_to_declarative(rubric)) for rubric in expand_nested_rubrics(rubrics)]
    best_by_key: dict[str, str] = {}
    for rubric in rewritten:
        if not rubric:
            continue
        key = canonical_text_key(rubric)
        current = best_by_key.get(key)
        if current is None or rubric_quality_score(rubric, source="human_seed", query="") > rubric_quality_score(
            current, source="human_seed", query=""
        ):
            best_by_key[key] = rubric
    return list(best_by_key.values())[:max_rubrics]


def expand_nested_rubrics(rubrics: list[str]) -> list[str]:
    expanded: list[str] = []
    for rubric in rubrics:
        text = str(rubric).strip()
        if text.startswith(("[", "{")):
            if text.startswith("[") and "]" in text:
                nested = parse_rubrics(text[: text.rfind("]") + 1], dedupe=False)
                if nested and nested != [text]:
                    expanded.extend(nested)
                    continue
            nested = parse_rubrics(text, dedupe=False)
            if nested and nested != [text]:
                expanded.extend(nested)
                continue
        expanded.append(text)
    return expanded


def source_aware_criteria(*, query: str, source: str) -> list[str]:
    query_l = query.lower()
    criteria: list[str] = []
    if source == "beir_nq":
        if "river" in query_l and any(token in query_l for token in ["finish", "end", "start", "where does"]):
            criteria.extend(
                [
                    (
                        "The response should explicitly state both the river's source and its mouth or "
                        "outflow using precise geographic terminology."
                    ),
                    (
                        "The response should address ambiguity when multiple rivers share the same name by "
                        "stating the assumed region or comparing the likely candidates."
                    ),
                    (
                        "The response should use terms such as mouth, outflow, confluence, bay, sea, strait, "
                        "or receiving river instead of the vague term finish when describing where the river ends."
                    ),
                ]
            )
        criteria.append(
            "The response should cite or name authoritative evidence sources when the task depends on factual retrieval."
        )
    return criteria


def rewrite_to_declarative(rubric: str) -> str:
    text = re.sub(r"\s+", " ", str(rubric)).strip(" \t\r\n\"'")
    if not text:
        return ""
    text = strip_quality_label(text)
    if not text:
        return ""

    text = rewrite_canonical_subject(text)
    question = text.rstrip(".").rstrip("?").strip()
    question_l = question.lower()
    match = re.match(r"^(If|For|When)\s+(.+?),\s+does\s+(?:the\s+)?(.+)$", question, flags=re.IGNORECASE)
    if match:
        prefix = match.group(1).capitalize()
        condition = match.group(2).strip()
        rest = match.group(3).strip()
        return clean_sentence(f"{prefix} {condition}, the response should ensure that {rest}")
    match = re.match(
        r"^(.+?),\s+does\s+(?:the\s+)?(?:response|solution|answer|model|assistant|function|implementation)?\s*(.+)$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        condition = match.group(1).strip()
        rest = match.group(2).strip()
        return clean_sentence(f"{condition}, the response should {rest}")
    if question_l.startswith("does not "):
        return clean_sentence(f"The response should not {question[len('does not '):].strip()}")
    match = re.match(
        r"^does\s+(?:the\s+)?(?:answer|response|model|assistant|document|snippet|document/snippet|content|source|explanation|information|it)\s+(.+)$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_sentence(f"The response should {match.group(1).strip()}")
    match = re.match(
        r"^is\s+(?:the\s+)?(?:answer|response|model|assistant|document|snippet|document/snippet|content|source|information|explanation|it)\s+(.+)$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        rest = match.group(1).strip()
        if rest.lower().startswith(("clearly ", "directly ", "accurately ", "fully ", "explicitly ")):
            return clean_sentence(f"The response should {rest}")
        return clean_sentence(f"The response should be {rest}")
    match = re.match(r"^is\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should be {match.group(1).strip()}")
    match = re.match(r"^should\s+(?:the\s+)?(?:answer|response|model|assistant|it)\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should {match.group(1).strip()}")
    match = re.match(r"^does\s+each\s+(.+?)\s+(include|contain|mention|use|have|satisfy|follow|meet)\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"Each {match.group(1).strip()} should {match.group(2).strip()} {match.group(3).strip()}")
    match = re.match(
        r"^does\s+(?:the\s+)?(?:result|reply|text|search result|assistant['’]s response|assistant response|candidate answer)\s+(.+)$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_sentence(f"The response should {match.group(1).strip()}")
    match = re.match(r"^does\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should ensure that {match.group(1).strip()}")
    match = re.match(
        r"^(?:To what extent|How well)\s+does\s+(?:the\s+)?(?:brief|writing|report|teaching plan|plan|article|document|script|response|answer|content|text)\s+(.+)$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_sentence(f"The response should {match.group(1).strip()}")
    match = re.match(r"^do\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should ensure that {match.group(1).strip()}")
    match = re.match(r"^are\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should ensure that {match.group(1).strip()} are satisfied")
    match = re.match(r"^can\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should ensure that {match.group(1).strip()}")
    match = re.match(r"^should\s+(.+)$", question, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should ensure that {match.group(1).strip()}")
    text = clean_sentence(text)
    if text.lower().startswith("does not "):
        return clean_sentence(f"The response should not {text[len('does not '):].strip()}")
    return text


def force_declarative(text: str) -> str:
    text = clean_sentence(text)
    match = re.match(r"^Does not\s+(.+)$", text, flags=re.IGNORECASE)
    if match:
        return clean_sentence(f"The response should not {match.group(1).strip()}")
    match = re.match(
        r"^(?:To what extent|How well)\s+does\s+(?:the\s+)?(?:brief|writing|report|teaching plan|plan|article|document|script|response|answer|content|text)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_sentence(f"The response should {match.group(1).strip()}")
    return text


def rewrite_canonical_subject(text: str) -> str:
    text = re.sub(
        r"^If applicable,\s+does\s+(?:the\s+)?(?:document|snippet|document/snippet|content|answer|response|result)\s+(.+)$",
        lambda m: f"If applicable, the response should {m.group(1).strip()}",
        text,
        flags=re.IGNORECASE,
    )
    replacements = {
        "directly answers": "directly answer",
        "clearly identifies": "clearly identify",
        "explicitly identifies": "explicitly identify",
        "provides": "provide",
        "includes": "include",
        "avoids": "avoid",
        "mentions": "mention",
        "reflects": "reflect",
        "addresses": "address",
        "matches": "match",
        "maintains": "maintain",
        "offers": "offer",
        "names": "name",
        "clarifies": "clarify",
        "stays": "stay",
        "uses": "use",
    }
    subject_match = re.match(
        r"^(?:The\s+)?(?:document|document/snippet|snippet|content|information|result|search result)(?:'s)?\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if subject_match:
        rest = subject_match.group(1).strip()
        return clean_sentence(f"The response should {verb_to_base(rest, replacements)}")
    verb_match = re.match(
        r"^(Offers|Maintains|Provides|Includes|Avoids|Names|Clarifies|Stays|Uses|Mentions|Reflects|Addresses|Matches)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if verb_match:
        verb = replacements.get(verb_match.group(1).lower(), verb_match.group(1).lower())
        return clean_sentence(f"The response should {verb} {verb_match.group(2).strip()}")
    return text


def verb_to_base(rest: str, replacements: dict[str, str]) -> str:
    rest_l = rest.lower()
    for inflected, base in replacements.items():
        if rest_l.startswith(inflected):
            return f"{base}{rest[len(inflected):]}"
    return rest


def strip_quality_label(text: str) -> str:
    generic_label = re.match(r"^[A-Z][A-Za-z/ -]{2,32}\s*:\s*(.+)$", text)
    if generic_label:
        body = generic_label.group(1).strip()
        if re.match(r"^(Does|Is|Do|Are|Can|Should|To what extent|How well)\b", body, flags=re.IGNORECASE):
            return body
    match = re.match(r"^(?:Completeness|Freshness|Factuality|Accuracy|Evidence|Relevance|Clarity)\s*:\s*(.+)$", text)
    if not match:
        return text
    body = match.group(1).strip()
    if body.lower().startswith(("covers ", "reflects ", "avoids ", "provides ", "includes ", "stays ")):
        return f"The response should {body}"
    return body


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" \t\r\n\"'")
    if not text:
        return ""
    text = re.sub(r"\bshould\s+is\b", "should be", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+are\b", "should be", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+was\b", "should be", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+were\b", "should be", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+does\b", "should", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+has\b", "should have", text, flags=re.IGNORECASE)
    text = re.sub(r",\s+and\s+does\s+not\b", ", and should not", text, flags=re.IGNORECASE)
    text = text.rstrip(".;?")
    return f"{text}."


def canonical_text_key(text: str) -> str:
    text = re.sub(r"\b(the|a|an|answer|response|document|snippet|content|source|should|must|be)\b", " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)).strip()


def rubric_dimension_key(text: str, *, source: str) -> str:
    normalized = canonical_text_key(text)
    if source == "beir_nq":
        if any(token in normalized for token in ["ambigu", "disambiguat", "same name", "assumed region", "candidate"]):
            return "beir_disambiguation"
        if any(token in normalized for token in ["mouth", "outflow", "confluence", "finish", "start", "source"]):
            return "beir_start_finish_endpoint"
        if any(token in normalized for token in ["coordinate", "place name", "geographic", "location", "precise"]):
            return "beir_precise_geography"
        if any(token in normalized for token in ["authoritative", "credible", "citation", "link", "evidence", "source"]):
            return "beir_authoritative_evidence"
        if any(token in normalized for token in ["misleading", "incorrect", "mixing", "wrong", "confusing"]):
            return "beir_factual_consistency"
        if any(token in normalized for token in ["fresh", "current", "outdated", "naming convention"]):
            return "beir_freshness"
        if any(token in normalized for token in ["intent", "focused", "directly addressing", "topical"]):
            return "beir_intent_relevance"
        if any(token in normalized for token in ["complete", "comprehensive", "both ends", "covers both"]):
            return "beir_completeness"
    if any(token in normalized for token in ["authoritative", "credible", "citation", "evidence", "source"]):
        return "evidence_support"
    if any(token in normalized for token in ["misleading", "incorrect", "factual", "accurate", "false"]):
        return "factuality"
    if any(token in normalized for token in ["complete", "comprehensive", "omitting", "missing"]):
        return "completeness"
    if any(token in normalized for token in ["safety", "risk", "harm", "medical", "professional care"]):
        return "safety"
    if any(token in normalized for token in ["constraint", "instruction", "requirement"]):
        return "constraint_following"
    return ""


def rubric_quality_score(text: str, *, source: str, query: str) -> float:
    score = 0.0
    text_l = text.lower()
    if text.startswith(("The response should", "The answer should", "The model should")):
        score += 3.0
    if text.rstrip().endswith("?"):
        score -= 4.0
    if any(token in text_l for token in ["specific", "precise", "verifiable", "cite", "citation", "evidence"]):
        score += 1.0
    if any(token in text_l for token in ["clear", "good", "quality", "helpful"]):
        score -= 0.5
    if source == "beir_nq":
        for token in ["ambigu", "assumed region", "mouth", "outflow", "confluence", "authoritative", "misleading"]:
            if token in text_l:
                score += 1.5
    if "river" in query.lower() and any(token in text_l for token in ["mouth", "outflow", "confluence", "same name"]):
        score += 2.0
    score += min(len(text.split()), 28) / 100.0
    return score


def make_sft_row(
    *,
    row_id: str,
    source: str,
    query: str,
    response: list[str],
    gold_type: str,
    teacher_sources: list[str],
    query_id: str,
    source_split: str,
    input_paths: list[str],
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(query=query)
    return {
        "id": row_id,
        "source": source,
        "gold_type": gold_type,
        "allowed_in_main_bsc_eval": False,
        "prompt": prompt,
        "response": response,
        "teacher_sources": teacher_sources,
        "verifier_version": PROXY_VERIFIER_VERSION if gold_type == "proxy_gold" else HUMAN_VERIFIER_VERSION,
        "dedup_tau": 0.85 if gold_type == "proxy_gold" else None,
    }


def sample_rows(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    allow_oversample: bool,
) -> list[dict[str, Any]]:
    if len(rows) >= n:
        return rng.sample(rows, n)
    if not allow_oversample:
        raise SystemExit(f"Need {n} rows, only {len(rows)} available.")
    if not rows:
        raise SystemExit("Cannot oversample from empty rows.")
    sampled = list(rows)
    while len(sampled) < n:
        base = rng.choice(rows)
        duplicate = dict(base)
        duplicate["id"] = f"{base['id']}__oversample_{len(sampled)}"
        duplicate["oversampled"] = True
        sampled.append(duplicate)
    return sampled


def to_proxy_gold(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source": row["source"],
        "gold_type": row["gold_type"],
        "allowed_in_main_bsc_eval": False,
        "prompt": row["prompt"],
        "response": row["response"],
        "teacher_sources": row["teacher_sources"],
        "verifier_version": row["verifier_version"],
        "dedup_tau": row["dedup_tau"],
    }


def build_report(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    proxy_by_source: dict[str, list[dict[str, Any]]],
    target_counts: dict[str, int],
) -> dict[str, Any]:
    source_counts = Counter(row["source"] for row in selected)
    gold_counts = Counter(row["gold_type"] for row in selected)
    rubric_counts = [len(row["response"]) for row in selected]
    format_valid_count = sum(1 for row in selected if is_list_of_strings(row.get("response")))
    question_like_count = sum(count_question_like_rubrics(row.get("response")) for row in selected)
    total_rubric_count = sum(rubric_counts)
    exact_duplicate_rows = sum(1 for row in selected if has_exact_normalized_duplicate(row.get("response")))
    proxy_asset_manifest = args.rule_verified_root.parent / "proxy_training_asset_manifest.json"
    input_assets = {
        "rubricbench_train_seed": {
            "path": str(args.rubricbench_train_seed),
            "sha256": file_sha256(args.rubricbench_train_seed),
        },
        "researchrubrics_train_seed": {
            "path": str(args.researchrubrics_train_seed),
            "sha256": file_sha256(args.researchrubrics_train_seed),
        },
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
    }
    return {
        "input": "closed_human_seed_and_proxy_assets",
        "input_sha256": input_assets["proxy_asset_manifest"]["sha256"],
        "input_assets": input_assets,
        "sft_output": str(args.output),
        "sft_output_sha256": file_sha256(args.output),
        "proxy_gold_output": str(args.proxy_gold_output) if args.proxy_gold_output else "",
        "proxy_gold_output_sha256": file_sha256(args.proxy_gold_output) if args.proxy_gold_output else "",
        "n_input_records": len(human_rows) + sum(len(rows) for rows in proxy_by_source.values()),
        "n_filtered_records": len(selected),
        "n_sft_records": len(selected),
        "n_proxy_gold_records": gold_counts.get("proxy_gold", 0),
        "target_counts": target_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "gold_type_counts": dict(sorted(gold_counts.items())),
        "source_ratios": {key: value / max(len(selected), 1) for key, value in sorted(source_counts.items())},
        "mean_rubrics": sum(rubric_counts) / max(len(rubric_counts), 1),
        "min_rubrics": min(rubric_counts) if rubric_counts else 0,
        "max_rubrics": max(rubric_counts) if rubric_counts else 0,
        "format_valid_count": format_valid_count,
        "format_valid_rate": format_valid_count / max(len(selected), 1),
        "question_like_rubric_count": question_like_count,
        "question_like_rubric_rate": question_like_count / max(total_rubric_count, 1),
        "rows_with_exact_normalized_duplicate": exact_duplicate_rows,
        "polishing": {
            "enabled": True,
            "version": PROXY_VERIFIER_VERSION,
            "steps": [
                "rewrite_question_style_to_declarative_criteria",
                "source_aware_dimension_key_dedup_for_proxy_rubrics",
                "semantic_dedup_after_dimension_dedup",
                "beir_search_geography_ambiguity_criteria_injection",
            ],
        },
        "sft_stage_metrics": {
            "format_valid_rate": format_valid_count / max(len(selected), 1),
            "mean_rubrics": sum(rubric_counts) / max(len(rubric_counts), 1),
            "verifier_pass_rate": "evaluate_after_sft_generation",
            "bsc_on_rubricbench_test_main": "evaluate_after_sft_generation",
        },
        "dedupe_tau": args.dedup_tau,
        "verifier_version": PROXY_VERIFIER_VERSION,
        "allowed_in_main_bsc_eval": False,
        "forbidden_data_source_markers": ["test_main", "holdout", "downstream"],
        "forbidden_splits": ["test_main", "holdout", "downstream", "test"],
        "columns": {
            "sft": [
                "id",
                "source",
                "gold_type",
                "allowed_in_main_bsc_eval",
                "prompt",
                "response",
                "teacher_sources",
                "verifier_version",
                "dedup_tau",
            ],
            "proxy_gold": [
                "id",
                "source",
                "gold_type",
                "allowed_in_main_bsc_eval",
                "prompt",
                "response",
                "teacher_sources",
                "verifier_version",
                "dedup_tau",
            ],
        },
        "notes": (
            "SFT is a supervised initialization asset for stable JSON/list formatting and high-quality "
            "evaluation criteria generation. It is not used for main BSC evaluation."
        ),
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


def is_list_of_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def count_question_like_rubrics(value: Any) -> int:
    if not is_list_of_strings(value):
        return 0
    return sum(
        1
        for item in value
        if item.strip().endswith("?")
        or re.match(r"^(Does|Is|Do|Are|Can|Should)\b", item.strip())
        or re.search(r"\bdoes\s+(?:the\s+)?(?:answer|response|solution|document|snippet|content|function|implementation)\b", item, re.I)
    )


def has_exact_normalized_duplicate(value: Any) -> bool:
    if not is_list_of_strings(value):
        return False
    keys = [canonical_text_key(item) for item in value]
    return len(keys) != len(set(keys))


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
