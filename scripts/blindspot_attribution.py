#!/usr/bin/env python3
"""Build a Blind-Spot Map from BSC eval records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import (  # noqa: E402
    SentenceTransformerEmbedder,
    TokenOverlapEmbedder,
    pairwise_cosine,
    parse_rubrics,
)


CATEGORIES = [
    "factuality",
    "completeness",
    "constraint_following",
    "safety",
    "domain_knowledge",
    "evidence_grounding",
    "intent_reasoning",
]

KEYWORDS = {
    "safety": [
        "safe",
        "safety",
        "harm",
        "illegal",
        "policy",
        "sexual",
        "adult",
        "minor",
        "unsafe",
        "violence",
        "安全",
        "风险",
        "违法",
        "未成年",
        "伤害",
    ],
    "evidence_grounding": [
        "evidence",
        "citation",
        "cite",
        "source",
        "reference",
        "quote",
        "grounded",
        "based on",
        "支持",
        "引用",
        "证据",
        "来源",
        "依据",
    ],
    "constraint_following": [
        "must",
        "should not",
        "format",
        "approximately",
        "word",
        "json",
        "bullet",
        "limit",
        "only",
        "avoid",
        "要求",
        "格式",
        "字数",
        "限制",
        "不能",
        "不得",
        "必须",
    ],
    "domain_knowledge": [
        "code",
        "python",
        "java",
        "api",
        "protobuf",
        "excel",
        "model",
        "equation",
        "algorithm",
        "医学",
        "法律",
        "代码",
        "函数",
        "经纬度",
        "领域",
        "专业",
    ],
    "intent_reasoning": [
        "intent",
        "infer",
        "reason",
        "because",
        "why",
        "clarify",
        "ambiguous",
        "causal",
        "step",
        "推理",
        "原因",
        "意图",
        "需求",
        "澄清",
        "歧义",
        "为什么",
    ],
    "factuality": [
        "correct",
        "accurate",
        "true",
        "states that",
        "identify",
        "explains that",
        "指出",
        "准确",
        "正确",
        "事实",
        "说明",
        "明确",
    ],
    "completeness": [
        "include",
        "cover",
        "address",
        "all",
        "multiple",
        "comprehensive",
        "完整",
        "覆盖",
        "包括",
        "全面",
        "多个",
    ],
}


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    embedder = TokenOverlapEmbedder() if args.embedding_model == "token-overlap" else SentenceTransformerEmbedder(args.embedding_model)
    rows, category_rows, summary = build_blindspot_map(
        records,
        coverage_tau=args.coverage_tau,
        embedder=embedder,
        model=args.model,
    )
    write_outputs(args.output_dir, rows, category_rows, summary)
    print(
        f"Blind-Spot Map wrote {summary['n']} records, "
        f"uncovered_gold={summary['uncovered_gold']} to {args.output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute unmatched gold evaluation dimensions to blind-spot categories."
    )
    parser.add_argument("--input", required=True, type=Path, help="BSC eval JSONL/JSON/parquet.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--coverage-tau", default=0.75, type=float)
    parser.add_argument("--model", default="")
    return parser.parse_args()


def build_blindspot_map(
    records: list[dict[str, Any]],
    coverage_tau: float,
    embedder: Any,
    model: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals = {category: {"total_gold": 0, "uncovered_gold": 0} for category in CATEGORIES}

    for idx, record in enumerate(records):
        query = str(pick_first(record, "query", "prompt", "instruction") or "")
        gold = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold"), dedupe=False)
        gen = parse_rubrics(
            pick_first(record, "response", "model_rubrics", "generated_rubrics", "prediction", "output"),
            dedupe=True,
        )
        categories = [classify_rubric(rubric) for rubric in gold]
        max_sims = [0.0 for _ in gold]
        matched_indices = [-1 for _ in gold]
        if gold and gen:
            sim = pairwise_cosine(gold, gen, embedder)
            max_sims = [float(value) for value in sim.max(axis=1)]
            matched_indices = [int(value) for value in sim.argmax(axis=1)]

        uncovered = []
        covered_by_category: dict[str, int] = {}
        total_by_category: dict[str, int] = {}
        for gold_idx, (rubric, category, max_sim, match_idx) in enumerate(zip(gold, categories, max_sims, matched_indices)):
            matched = max_sim >= coverage_tau
            totals[category]["total_gold"] += 1
            total_by_category[category] = total_by_category.get(category, 0) + 1
            if matched:
                covered_by_category[category] = covered_by_category.get(category, 0) + 1
            else:
                totals[category]["uncovered_gold"] += 1
                uncovered.append(
                    {
                        "gold_idx": gold_idx,
                        "rubric": rubric,
                        "category": category,
                        "matched": False,
                        "max_similarity": round(max_sim, 6),
                        "best_generated_idx": match_idx,
                        "best_generated_rubric": gen[match_idx] if 0 <= match_idx < len(gen) else "",
                    }
                )

        rows.append(
            {
                "idx": idx,
                "model": model or str(record.get("model") or ""),
                "query": query,
                "n_gold": len(gold),
                "n_gen": len(gen),
                "n_uncovered_gold": len(uncovered),
                "blind_rate": (len(uncovered) / len(gold)) if gold else 0.0,
                "category_coverage": {
                    category: (covered_by_category.get(category, 0) / total)
                    for category, total in sorted(total_by_category.items())
                    if total
                },
                "uncovered_gold_rubrics": uncovered,
            }
        )

    category_rows = []
    for category in CATEGORIES:
        total = totals[category]["total_gold"]
        uncovered = totals[category]["uncovered_gold"]
        covered = total - uncovered
        category_rows.append(
            {
                "category": category,
                "total_gold": total,
                "covered_gold": covered,
                "uncovered_gold": uncovered,
                "coverage": (covered / total) if total else 0.0,
                "blind_rate": (uncovered / total) if total else 0.0,
            }
        )

    total_gold = sum(row["total_gold"] for row in category_rows)
    uncovered_gold = sum(row["uncovered_gold"] for row in category_rows)
    summary = {
        "n": len(rows),
        "model": model,
        "coverage_tau": coverage_tau,
        "categories": CATEGORIES,
        "total_gold": total_gold,
        "uncovered_gold": uncovered_gold,
        "mean_blind_over_gold": (uncovered_gold / total_gold) if total_gold else 0.0,
    }
    return rows, category_rows, summary


def classify_rubric(rubric: str) -> str:
    text = rubric.lower()
    scores = {category: 0 for category in CATEGORIES}
    for category, keywords in KEYWORDS.items():
        scores[category] = sum(1 for keyword in keywords if keyword.lower() in text)
    best_category, best_score = max(scores.items(), key=lambda item: (item[1], -CATEGORIES.index(item[0])))
    if best_score > 0:
        return best_category
    return "completeness"


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("records") or data.get("data") or [data]
        yield from records
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    raise ValueError(f"Unsupported input format: {path}")


def pick_first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "blindspots.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "category_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
