#!/usr/bin/env python3
"""Build a paper-facing semantic-space visualization for evaluation criteria.

The script supports UMAP/t-SNE for paper-facing runs and falls back to
deterministic PCA when optional visualization dependencies are unavailable.
Real paper runs should use the same embedding model as BSC (default:
BAAI/bge-large-en-v1.5); tests can use `--embedding-model token-overlap`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blindspot_rl.reward_bsc import SentenceTransformerEmbedder, TokenOverlapEmbedder, parse_rubrics  # noqa: E402
from scripts.blindspot_attribution import CATEGORIES, classify_rubric  # noqa: E402
from scripts.budget_gate import file_sha256  # noqa: E402


SOURCE_COLORS = {
    "gold": "#222222",
    "generated": "#1f77b4",
}

CATEGORY_COLORS = {
    "factuality": "#1f77b4",
    "completeness": "#ff7f0e",
    "constraint_following": "#2ca02c",
    "safety": "#d62728",
    "domain_knowledge": "#9467bd",
    "evidence_grounding": "#8c564b",
    "intent_reasoning": "#e377c2",
}

POINT_CSV_COLUMNS = [
    "point_id",
    "record_idx",
    "method",
    "source_type",
    "category",
    "gold_cluster_id",
    "rubric_idx",
    "x",
    "y",
    "nearest_gold_point_id",
    "nearest_gold_record_idx",
    "nearest_gold_rubric_idx",
    "nearest_gold_category",
    "nearest_gold_cluster_id",
    "nearest_gold_similarity",
    "nearest_gold_same_record",
    "query",
    "text",
    "nearest_gold_text",
]


def main() -> None:
    args = parse_args()
    records = []
    input_metadata = []
    join_reports = parse_labeled_path_specs(args.join_report or [])
    for spec in args.input:
        label, path = parse_input_spec(spec)
        labeled_records = load_labeled_records(path, label=label)
        input_metadata.append(input_metadata_for(label=label, path=path, records=labeled_records, join_report=join_reports.get(label)))
        records.extend(labeled_records)
    if not records:
        raise SystemExit("No records found.")

    embedder = TokenOverlapEmbedder() if args.embedding_model == "token-overlap" else SentenceTransformerEmbedder(args.embedding_model)
    points, summary = build_semantic_space(
        records,
        embedder=embedder,
        embedding_model=args.embedding_model,
        projection=args.projection,
        gold_cluster_tau=args.gold_cluster_tau,
        max_points=args.max_points,
    )
    if not points:
        raise SystemExit("No rubric points found.")
    summary["inputs"] = input_metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.output_dir / "semantic_space_points.csv"
    svg_path = args.output_dir / "semantic_space.svg"
    pdf_path = args.output_dir / "semantic_space.pdf"
    summary_path = args.output_dir / "semantic_space_summary.json"

    write_points_csv(points_path, points)
    svg_path.write_text(render_svg(points, summary), encoding="utf-8")
    pdf_path.write_bytes(render_pdf(points, summary))
    summary.update(output_artifact_metadata(points_path=points_path, svg_path=svg_path, pdf_path=pdf_path, n_points=len(points)))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Semantic-space visualization wrote {summary['n_points']} points "
        f"({summary['n_gold']} gold, {summary['n_generated']} generated) to {args.output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluation-criteria semantic-space visualization assets.")
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="Input JSONL/JSON/parquet. Use label=path to compare methods; repeatable.",
    )
    parser.add_argument(
        "--join-report",
        action="append",
        help="Optional label=path BSC join report for input provenance; repeatable.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--projection", choices=("pca", "tsne", "umap"), default="pca")
    parser.add_argument(
        "--gold-cluster-tau",
        type=float,
        default=0.75,
        help="Gold-gold cosine threshold for deterministic semantic cluster components.",
    )
    parser.add_argument("--max-points", type=int, help="Optional cap for fast draft visualization.")
    return parser.parse_args()


def build_semantic_space(
    records: Sequence[dict[str, Any]],
    *,
    embedder: Any,
    embedding_model: str | None = None,
    projection: str = "pca",
    gold_cluster_tau: float = 0.75,
    max_points: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_points = collect_rubric_points(records)
    if max_points is not None:
        raw_points = raw_points[:max_points]
    if not raw_points:
        return [], summary_for(
            [],
            projection=projection,
            requested_projection=projection,
            embedding_model=resolve_embedding_model_name(embedder, embedding_model),
            gold_cluster_tau=gold_cluster_tau,
        )

    texts = [point["text"] for point in raw_points]
    vectors = np.asarray(embedder.encode(texts), dtype=np.float64)
    coords, actual_projection = project_2d(vectors, method=projection)
    points = []
    for idx, (point, coord) in enumerate(zip(raw_points, coords)):
        row = {
            "point_id": idx,
            **point,
            "x": float(coord[0]),
            "y": float(coord[1]),
        }
        points.append(row)
    attach_gold_cluster_fields(points, vectors, gold_cluster_tau=gold_cluster_tau)
    attach_nearest_gold_fields(points, vectors)
    return points, summary_for(
        points,
        projection=actual_projection,
        requested_projection=projection,
        embedding_model=resolve_embedding_model_name(embedder, embedding_model),
        gold_cluster_tau=gold_cluster_tau,
        vectors=vectors,
    )


def collect_rubric_points(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        query = str(pick_first(record, "query", "prompt", "instruction") or "")
        method = str(record.get("_input_label") or record.get("method") or record.get("model") or record.get("teacher") or "generated")
        gold = parse_rubrics(pick_first(record, "gold_rubrics", "gold", "rubrics_gold"), dedupe=True)
        generated = parse_rubrics(
            pick_first(record, "response", "model_rubrics", "generated_rubrics", "prediction", "output", "rubrics"),
            dedupe=True,
        )

        gold_categories = [classify_rubric(item) for item in gold]
        for rubric_idx, rubric in enumerate(gold):
            points.append(
                {
                    "record_idx": record_idx,
                    "query": query,
                    "method": "human_gold",
                    "source_type": "gold",
                    "rubric_idx": rubric_idx,
                    "category": gold_categories[rubric_idx],
                    "text": rubric,
                }
            )
        for rubric_idx, rubric in enumerate(generated):
            points.append(
                {
                    "record_idx": record_idx,
                    "query": query,
                    "method": method,
                    "source_type": "generated",
                    "rubric_idx": rubric_idx,
                    "category": infer_generated_category(rubric, gold, gold_categories),
                    "text": rubric,
                }
            )
    return points


def infer_generated_category(rubric: str, gold: Sequence[str], gold_categories: Sequence[str]) -> str:
    del gold, gold_categories
    return classify_rubric(rubric)


def project_2d(vectors: np.ndarray, method: str = "pca") -> tuple[np.ndarray, str]:
    matrix = ensure_2d(vectors)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64), method
    if method == "tsne":
        projected = project_tsne(matrix)
        if projected is not None:
            return projected, "tsne"
        return project_pca(matrix), "tsne_fallback_pca"
    if method == "umap":
        projected = project_umap(matrix)
        if projected is not None:
            return projected, "umap"
        return project_pca(matrix), "umap_fallback_pca"
    if method == "pca":
        return project_pca(matrix), "pca"
    raise ValueError(f"Unsupported projection method: {method}")


def project_pca(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            vt = np.nan_to_num(vt, nan=0.0, posinf=0.0, neginf=0.0)
            coords = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        coords = centered[:, :2]
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    return ensure_2d_coords(coords)


def project_tsne(matrix: np.ndarray) -> np.ndarray | None:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    n = matrix.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    perplexity = min(30.0, max(1.0, float(n - 1) / 3.0))
    if perplexity >= n:
        perplexity = max(1.0, float(n - 1))
    try:
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=13,
        ).fit_transform(matrix)
    except Exception:
        return None
    return ensure_2d_coords(np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0))


def project_umap(matrix: np.ndarray) -> np.ndarray | None:
    try:
        import umap
    except Exception:
        return None
    n = matrix.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    try:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, n - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=13,
        )
        coords = reducer.fit_transform(matrix)
    except Exception:
        return None
    return ensure_2d_coords(np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0))


def ensure_2d(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] == 0:
        matrix = np.zeros((matrix.shape[0], 1), dtype=np.float64)
    return matrix


def ensure_2d_coords(coords: np.ndarray) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[1] == 0:
        array = np.zeros((array.shape[0], 2), dtype=np.float64)
    if array.shape[1] == 1:
        array = np.column_stack([array[:, 0], np.zeros(array.shape[0], dtype=np.float64)])
    return array[:, :2]


def summary_for(
    points: Sequence[dict[str, Any]],
    projection: str,
    requested_projection: str | None = None,
    embedding_model: str | None = None,
    gold_cluster_tau: float = 0.75,
    vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    methods = sorted({str(point["method"]) for point in points if point.get("source_type") == "generated"})
    category_counts = {category: 0 for category in CATEGORIES}
    gold_category_counts = {category: 0 for category in CATEGORIES}
    generated_category_counts_by_method = {method: {category: 0 for category in CATEGORIES} for method in methods}
    for point in points:
        category = str(point.get("category") or "completeness")
        category_counts[category] = category_counts.get(category, 0) + 1
        if point.get("source_type") == "gold":
            gold_category_counts[category] = gold_category_counts.get(category, 0) + 1
        elif point.get("source_type") == "generated":
            method = str(point.get("method") or "generated")
            generated_category_counts_by_method.setdefault(method, {category: 0 for category in CATEGORIES})
            generated_category_counts_by_method[method][category] = (
                generated_category_counts_by_method[method].get(category, 0) + 1
            )

    gold_categories = sorted(category for category, count in gold_category_counts.items() if count > 0)
    generated_gold_category_coverage_by_method = {}
    generated_gold_category_count_by_method = {}
    for method in methods:
        generated_categories = {
            category
            for category, count in generated_category_counts_by_method.get(method, {}).items()
            if count > 0 and category in gold_categories
        }
        generated_gold_category_count_by_method[method] = len(generated_categories)
        generated_gold_category_coverage_by_method[method] = safe_ratio(len(generated_categories), len(gold_categories))

    embedding_metrics = embedding_summary_metrics(points, vectors=vectors)
    summary = {
        "projection": projection,
        "requested_projection": requested_projection or projection,
        "embedding_model": embedding_model,
        "gold_cluster_tau": gold_cluster_tau,
        "point_csv_columns": POINT_CSV_COLUMNS,
        "point_csv_schema_version": 3,
        "n_points": len(points),
        "n_gold": sum(1 for point in points if point.get("source_type") == "gold"),
        "n_generated": sum(1 for point in points if point.get("source_type") == "generated"),
        "methods": methods,
        "category_counts": category_counts,
        "gold_categories": gold_categories,
        "gold_category_counts": gold_category_counts,
        "generated_category_counts_by_method": generated_category_counts_by_method,
        "generated_gold_category_count_by_method": generated_gold_category_count_by_method,
        "generated_gold_category_coverage_by_method": generated_gold_category_coverage_by_method,
    }
    summary.update(embedding_metrics)
    add_method_delta(
        summary,
        "sft_rl_vs_sft_only_generated_gold_category_coverage_delta",
        generated_gold_category_coverage_by_method,
        left="sft_rl",
        right="sft_only",
    )
    add_method_delta(
        summary,
        "sft_rl_vs_sft_only_nearest_gold_category_coverage_delta",
        summary["nearest_gold_category_coverage_by_method"],
        left="sft_rl",
        right="sft_only",
    )
    add_method_delta(
        summary,
        "sft_rl_vs_sft_only_gold_cluster_coverage_delta",
        summary["nearest_gold_cluster_coverage_by_method"],
        left="sft_rl",
        right="sft_only",
    )
    add_method_delta(
        summary,
        "sft_rl_vs_sft_only_nearest_gold_cluster_entropy_delta",
        summary["nearest_gold_cluster_entropy_by_method"],
        left="sft_rl",
        right="sft_only",
    )
    add_method_delta(
        summary,
        "sft_rl_vs_sft_only_generated_dispersion_delta",
        summary["generated_dispersion_by_method"],
        left="sft_rl",
        right="sft_only",
    )
    add_method_delta(
        summary,
        "sft_rl_vs_sft_only_nearest_gold_similarity_delta",
        summary["mean_nearest_gold_similarity_by_method"],
        left="sft_rl",
        right="sft_only",
    )
    return summary


def resolve_embedding_model_name(embedder: Any, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    model_name = getattr(embedder, "model_name", None)
    if model_name:
        return str(model_name)
    return embedder.__class__.__name__


def attach_gold_cluster_fields(points: list[dict[str, Any]], vectors: np.ndarray | None, *, gold_cluster_tau: float) -> None:
    for point in points:
        point.setdefault("gold_cluster_id", "")
        point.setdefault("nearest_gold_cluster_id", "")
    if vectors is None or not points:
        return
    matrix = ensure_2d(np.asarray(vectors, dtype=np.float64))
    if matrix.shape[0] != len(points):
        return
    gold_indices = [idx for idx, point in enumerate(points) if point.get("source_type") == "gold"]
    if not gold_indices:
        return
    normed = normalize_rows(matrix)
    clusters = gold_cluster_components(normed, gold_indices=gold_indices, tau=gold_cluster_tau)
    for cluster_idx, component in enumerate(clusters):
        cluster_id = f"g{cluster_idx:03d}"
        for point_idx in component:
            points[point_idx]["gold_cluster_id"] = cluster_id


def gold_cluster_components(matrix: np.ndarray, *, gold_indices: Sequence[int], tau: float) -> list[list[int]]:
    gold_list = list(gold_indices)
    if not gold_list:
        return []
    similarities = cosine_similarity_matrix(matrix, source_indices=gold_list, target_indices=gold_list)
    visited: set[int] = set()
    components: list[list[int]] = []
    for offset, point_idx in enumerate(gold_list):
        if offset in visited:
            continue
        stack = [offset]
        visited.add(offset)
        component_offsets = []
        while stack:
            current = stack.pop()
            component_offsets.append(current)
            neighbors = np.where(similarities[current] >= tau)[0]
            for neighbor in neighbors:
                neighbor_int = int(neighbor)
                if neighbor_int not in visited:
                    visited.add(neighbor_int)
                    stack.append(neighbor_int)
        components.append(sorted(gold_list[item] for item in component_offsets))
    return sorted(components, key=lambda component: (component[0], len(component)))


def attach_nearest_gold_fields(points: list[dict[str, Any]], vectors: np.ndarray | None) -> None:
    for point in points:
        point.setdefault("nearest_gold_point_id", "")
        point.setdefault("nearest_gold_record_idx", "")
        point.setdefault("nearest_gold_rubric_idx", "")
        point.setdefault("nearest_gold_category", "")
        point.setdefault("nearest_gold_cluster_id", "")
        point.setdefault("nearest_gold_similarity", "")
        point.setdefault("nearest_gold_same_record", "")
        point.setdefault("nearest_gold_text", "")
    if vectors is None or not points:
        return
    matrix = ensure_2d(np.asarray(vectors, dtype=np.float64))
    if matrix.shape[0] != len(points):
        return
    gold_indices = [idx for idx, point in enumerate(points) if point.get("source_type") == "gold"]
    generated_indices = [idx for idx, point in enumerate(points) if point.get("source_type") == "generated"]
    if not gold_indices or not generated_indices:
        return
    normed = normalize_rows(matrix)
    similarities = cosine_similarity_matrix(normed, source_indices=generated_indices, target_indices=gold_indices)
    nearest_offsets = np.argmax(similarities, axis=1)
    for row_idx, source_idx in enumerate(generated_indices):
        gold_idx = gold_indices[int(nearest_offsets[row_idx])]
        gold_point = points[gold_idx]
        source_point = points[source_idx]
        source_point["nearest_gold_point_id"] = gold_point.get("point_id", gold_idx)
        source_point["nearest_gold_record_idx"] = gold_point.get("record_idx", "")
        source_point["nearest_gold_rubric_idx"] = gold_point.get("rubric_idx", "")
        source_point["nearest_gold_category"] = gold_point.get("category", "")
        source_point["nearest_gold_cluster_id"] = gold_point.get("gold_cluster_id", "")
        source_point["nearest_gold_similarity"] = float(similarities[row_idx, int(nearest_offsets[row_idx])])
        source_point["nearest_gold_same_record"] = bool(source_point.get("record_idx") == gold_point.get("record_idx"))
        source_point["nearest_gold_text"] = gold_point.get("text", "")


def embedding_summary_metrics(points: Sequence[dict[str, Any]], vectors: np.ndarray | None) -> dict[str, Any]:
    methods = sorted({str(point["method"]) for point in points if point.get("source_type") == "generated"})
    empty = {
        "mean_nearest_gold_similarity_by_method": {method: None for method in methods},
        "generated_dispersion_by_method": {method: None for method in methods},
        "nearest_gold_category_count_by_method": {method: None for method in methods},
        "nearest_gold_category_coverage_by_method": {method: None for method in methods},
        "n_gold_clusters": len({str(point.get("gold_cluster_id")) for point in points if point.get("gold_cluster_id")}),
        "gold_cluster_counts": {},
        "nearest_gold_cluster_count_by_method": {method: None for method in methods},
        "nearest_gold_cluster_coverage_by_method": {method: None for method in methods},
        "nearest_gold_cluster_distribution_by_method": {method: {} for method in methods},
        "nearest_gold_cluster_entropy_by_method": {method: None for method in methods},
    }
    if vectors is None or len(points) == 0:
        return empty

    matrix = ensure_2d(np.asarray(vectors, dtype=np.float64))
    if matrix.shape[0] != len(points):
        return empty
    normed = normalize_rows(matrix)
    gold_indices = [idx for idx, point in enumerate(points) if point.get("source_type") == "gold"]
    gold_categories = sorted({str(points[idx].get("category") or "") for idx in gold_indices if points[idx].get("category")})
    generated_indices_by_method = {
        method: [
            idx
            for idx, point in enumerate(points)
            if point.get("source_type") == "generated" and str(point.get("method")) == method
        ]
        for method in methods
    }

    nearest_gold = {}
    dispersion = {}
    nearest_gold_category_count = {}
    nearest_gold_category_coverage = {}
    gold_clusters = sorted({str(points[idx].get("gold_cluster_id") or "") for idx in gold_indices if points[idx].get("gold_cluster_id")})
    gold_cluster_counts = {cluster: 0 for cluster in gold_clusters}
    for idx in gold_indices:
        cluster_id = str(points[idx].get("gold_cluster_id") or "")
        if cluster_id:
            gold_cluster_counts[cluster_id] = gold_cluster_counts.get(cluster_id, 0) + 1
    nearest_gold_cluster_count = {}
    nearest_gold_cluster_coverage = {}
    nearest_gold_cluster_distribution = {}
    nearest_gold_cluster_entropy = {}
    for method, indices in generated_indices_by_method.items():
        nearest_gold[method] = mean_nearest_similarity(normed, source_indices=indices, target_indices=gold_indices)
        dispersion[method] = mean_pairwise_cosine_distance(normed, indices=indices)
        nearest_categories = nearest_gold_categories(normed, points, source_indices=indices, target_indices=gold_indices)
        nearest_gold_category_count[method] = len(nearest_categories)
        nearest_gold_category_coverage[method] = safe_ratio(len(nearest_categories), len(gold_categories))
        nearest_cluster_counts = nearest_gold_cluster_counts(
            normed,
            points,
            source_indices=indices,
            target_indices=gold_indices,
        )
        nearest_clusters = set(nearest_cluster_counts)
        nearest_gold_cluster_count[method] = len(nearest_clusters)
        nearest_gold_cluster_coverage[method] = safe_ratio(len(nearest_clusters), len(gold_clusters))
        nearest_gold_cluster_distribution[method] = nearest_cluster_counts
        nearest_gold_cluster_entropy[method] = normalized_entropy(nearest_cluster_counts, support_size=len(gold_clusters))
    return {
        "mean_nearest_gold_similarity_by_method": nearest_gold,
        "generated_dispersion_by_method": dispersion,
        "nearest_gold_category_count_by_method": nearest_gold_category_count,
        "nearest_gold_category_coverage_by_method": nearest_gold_category_coverage,
        "n_gold_clusters": len(gold_clusters),
        "gold_cluster_counts": gold_cluster_counts,
        "nearest_gold_cluster_count_by_method": nearest_gold_cluster_count,
        "nearest_gold_cluster_coverage_by_method": nearest_gold_cluster_coverage,
        "nearest_gold_cluster_distribution_by_method": nearest_gold_cluster_distribution,
        "nearest_gold_cluster_entropy_by_method": nearest_gold_cluster_entropy,
    }


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.nan_to_num(norms, nan=1.0, posinf=1.0, neginf=1.0)
    norms[norms <= 1e-12] = 1.0
    return np.nan_to_num(matrix / norms, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_similarity_matrix(
    matrix: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> np.ndarray:
    if not source_indices or not target_indices:
        return np.zeros((len(source_indices), len(target_indices)), dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        similarities = matrix[list(source_indices)] @ matrix[list(target_indices)].T
    return np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)


def mean_nearest_similarity(
    matrix: np.ndarray,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> float | None:
    if not source_indices or not target_indices:
        return None
    similarities = cosine_similarity_matrix(matrix, source_indices=source_indices, target_indices=target_indices)
    return float(np.max(similarities, axis=1).mean())


def nearest_gold_categories(
    matrix: np.ndarray,
    points: Sequence[dict[str, Any]],
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> set[str]:
    if not source_indices or not target_indices:
        return set()
    target_list = list(target_indices)
    similarities = cosine_similarity_matrix(matrix, source_indices=source_indices, target_indices=target_list)
    nearest_offsets = np.argmax(similarities, axis=1)
    return {
        str(points[target_list[int(offset)]].get("category") or "")
        for offset in nearest_offsets
        if points[target_list[int(offset)]].get("category")
    }


def nearest_gold_clusters(
    matrix: np.ndarray,
    points: Sequence[dict[str, Any]],
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> set[str]:
    if not source_indices or not target_indices:
        return set()
    target_list = list(target_indices)
    similarities = cosine_similarity_matrix(matrix, source_indices=source_indices, target_indices=target_list)
    nearest_offsets = np.argmax(similarities, axis=1)
    return {
        str(points[target_list[int(offset)]].get("gold_cluster_id") or "")
        for offset in nearest_offsets
        if points[target_list[int(offset)]].get("gold_cluster_id")
    }


def nearest_gold_cluster_counts(
    matrix: np.ndarray,
    points: Sequence[dict[str, Any]],
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> dict[str, int]:
    if not source_indices or not target_indices:
        return {}
    target_list = list(target_indices)
    similarities = cosine_similarity_matrix(matrix, source_indices=source_indices, target_indices=target_list)
    nearest_offsets = np.argmax(similarities, axis=1)
    counts: dict[str, int] = {}
    for offset in nearest_offsets:
        cluster_id = str(points[target_list[int(offset)]].get("gold_cluster_id") or "")
        if cluster_id:
            counts[cluster_id] = counts.get(cluster_id, 0) + 1
    return counts


def normalized_entropy(counts: dict[str, int], *, support_size: int) -> float | None:
    total = sum(count for count in counts.values() if count > 0)
    if total <= 0 or support_size <= 1:
        return None
    probabilities = np.asarray([count / total for count in counts.values() if count > 0], dtype=np.float64)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / float(np.log(support_size))


def mean_pairwise_cosine_distance(matrix: np.ndarray, *, indices: Sequence[int]) -> float | None:
    if len(indices) < 2:
        return None
    subset = matrix[list(indices)]
    sims = cosine_similarity_matrix(subset, source_indices=range(len(indices)), target_indices=range(len(indices)))
    upper = sims[np.triu_indices(len(indices), k=1)]
    return float((1.0 - upper).mean())


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def add_method_delta(
    summary: dict[str, Any],
    key: str,
    values: dict[str, float | None],
    *,
    left: str,
    right: str,
) -> None:
    left_value = values.get(left)
    right_value = values.get(right)
    if left_value is None or right_value is None:
        summary[key] = None
        return
    summary[key] = float(left_value) - float(right_value)


def render_svg(points: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    width = 980
    height = 720
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale_x(value: float) -> float:
        return margin + normalize(value, min_x, max_x) * plot_w

    def scale_y(value: float) -> float:
        return height - margin - normalize(value, min_y, max_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #222; }",
        ".axis { stroke: #d0d0d0; stroke-width: 1; }",
        ".gold { fill: white; stroke-width: 2.2; opacity: 0.92; }",
        ".generated { stroke: white; stroke-width: 0.8; opacity: 0.78; }",
        "</style>",
        '<rect x="0" y="0" width="980" height="720" fill="white"/>',
        '<text x="70" y="38" font-size="22" font-weight="700">Evaluation-Criteria Semantic Space</text>',
        (
            f'<text x="70" y="60" font-size="13">{svg_escape(str(summary["projection"]).upper())} projection; '
            f'{summary["n_gold"]} human-gold dimensions and {summary["n_generated"]} generated dimensions</text>'
        ),
        f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d9d9d9"/>',
        f'<line class="axis" x1="{margin}" y1="{height/2:.2f}" x2="{width-margin}" y2="{height/2:.2f}"/>',
        f'<line class="axis" x1="{width/2:.2f}" y1="{margin}" x2="{width/2:.2f}" y2="{height-margin}"/>',
    ]
    for point in points:
        category = str(point.get("category") or "completeness")
        color = CATEGORY_COLORS.get(category, SOURCE_COLORS.get(str(point.get("source_type")), "#777777"))
        x = scale_x(float(point["x"]))
        y = scale_y(float(point["y"]))
        label = svg_escape(f"{point.get('source_type')} | {point.get('method')} | {category}: {point.get('text')}")
        if point.get("source_type") == "gold":
            lines.append(f'<circle class="gold" cx="{x:.2f}" cy="{y:.2f}" r="5.0" stroke="{color}"><title>{label}</title></circle>')
        else:
            lines.append(f'<circle class="generated" cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{color}"><title>{label}</title></circle>')

    legend_x = width - 275
    legend_y = 88
    lines.append(f'<rect x="{legend_x - 16}" y="{legend_y - 28}" width="245" height="250" fill="white" stroke="#dddddd"/>')
    lines.append(f'<text x="{legend_x}" y="{legend_y - 8}" font-size="13" font-weight="700">Category color</text>')
    for idx, category in enumerate(CATEGORIES):
        y = legend_y + idx * 25 + 16
        color = CATEGORY_COLORS.get(category, "#777777")
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 14}" y="{y + 4}" font-size="12">{svg_escape(category)}</text>')
    lines.append(f'<circle cx="{legend_x}" cy="{legend_y + 205}" r="5" fill="white" stroke="#222" stroke-width="2.2"/>')
    lines.append(f'<text x="{legend_x + 14}" y="{legend_y + 209}" font-size="12">human gold: hollow</text>')
    lines.append(f'<circle cx="{legend_x}" cy="{legend_y + 230}" r="5" fill="#777" stroke="white" stroke-width="0.8"/>')
    lines.append(f'<text x="{legend_x + 14}" y="{legend_y + 234}" font-size="12">generated: filled</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_pdf(points: Sequence[dict[str, Any]], summary: dict[str, Any]) -> bytes:
    width = 980
    height = 720
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale_x(value: float) -> float:
        return margin + normalize(value, min_x, max_x) * plot_w

    def scale_y(value: float) -> float:
        svg_y = height - margin - normalize(value, min_y, max_y) * plot_h
        return height - svg_y

    commands = [
        "1 1 1 rg 0 0 980 720 re f",
        pdf_text(70, 682, "Evaluation-Criteria Semantic Space", size=22),
        pdf_text(
            70,
            660,
            f"{str(summary['projection']).upper()} projection; {summary['n_gold']} human-gold dimensions and {summary['n_generated']} generated dimensions",
            size=13,
        ),
        "0.98 0.98 0.98 rg 0.85 0.85 0.85 RG 70 70 840 580 re B",
        "0.82 0.82 0.82 RG 70 360 m 910 360 l S",
        "0.82 0.82 0.82 RG 490 70 m 490 650 l S",
    ]
    for point in points:
        category = str(point.get("category") or "completeness")
        color = CATEGORY_COLORS.get(category, SOURCE_COLORS.get(str(point.get("source_type")), "#777777"))
        r, g, b = hex_rgb(color)
        x = scale_x(float(point["x"]))
        y = scale_y(float(point["y"]))
        if point.get("source_type") == "gold":
            commands.append(f"1 1 1 rg {r:.4f} {g:.4f} {b:.4f} RG 2 w {x - 5:.2f} {y - 5:.2f} 10 10 re B")
        else:
            commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg 1 1 1 RG 0.8 w {x - 4:.2f} {y - 4:.2f} 8 8 re B")

    legend_x = width - 275
    legend_y = height - 88
    commands.append(f"1 1 1 rg 0.87 0.87 0.87 RG {legend_x - 16} {legend_y - 222} 245 250 re B")
    commands.append(pdf_text(legend_x, legend_y + 8, "Category color", size=13))
    for idx, category in enumerate(CATEGORIES):
        y = legend_y - idx * 25 - 16
        r, g, b = hex_rgb(CATEGORY_COLORS.get(category, "#777777"))
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg {legend_x - 5} {y - 5} 10 10 re f")
        commands.append(pdf_text(legend_x + 14, y - 4, category, size=12))
    commands.append(f"1 1 1 rg 0.13 0.13 0.13 RG 2 w {legend_x - 5} {legend_y - 210} 10 10 re B")
    commands.append(pdf_text(legend_x + 14, legend_y - 209, "human gold: hollow", size=12))
    commands.append(f"0.47 0.47 0.47 rg 1 1 1 RG 0.8 w {legend_x - 5} {legend_y - 235} 10 10 re B")
    commands.append(pdf_text(legend_x + 14, legend_y - 234, "generated: filled", size=12))
    return build_pdf("\n".join(commands).encode("latin-1", errors="replace"), width=width, height=height)


def build_pdf(content: bytes, *, width: int, height: int) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    xref = [b"xref\n", f"0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return b"".join(chunks + xref + [trailer])


def pdf_text(x: float, y: float, text: str, *, size: int) -> str:
    return f"0 0 0 rg BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET"


def pdf_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def hex_rgb(value: str) -> tuple[float, float, float]:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        return (0.47, 0.47, 0.47)
    return (int(raw[0:2], 16) / 255, int(raw[2:4], 16) / 255, int(raw[4:6], 16) / 255)


def normalize(value: float, min_value: float, max_value: float) -> float:
    if abs(max_value - min_value) <= 1e-12:
        return 0.5
    return (value - min_value) / (max_value - min_value)


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Input label is empty: {spec}")
        return label, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def parse_labeled_path_specs(specs: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        label, path = parse_input_spec(spec)
        if label in parsed:
            raise ValueError(f"Duplicate labeled path spec: {label}")
        parsed[label] = path
    return parsed


def input_metadata_for(
    *,
    label: str,
    path: Path,
    records: Sequence[dict[str, Any]],
    join_report: Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "label": label,
        "path": str(path),
        "sha256": file_sha256(path),
        "n_records": len(records),
        "n_unique_queries": len({str(pick_first(record, "query", "prompt", "instruction") or "") for record in records}),
        "split_counts": value_counts(first_split_value(record) for record in records),
        "data_source_counts": value_counts(str(record.get("data_source") or "") for record in records),
    }
    if join_report is not None:
        metadata["join_report"] = join_report_metadata(join_report)
    return metadata


def join_report_metadata(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "output_sha256": report.get("output_sha256"),
        "output": report.get("output"),
        "gold": report.get("gold"),
        "prediction": report.get("prediction"),
        "join_key": report.get("join_key"),
        "n_joined": report.get("n_joined"),
        "unmatched_gold": report.get("unmatched_gold"),
        "unmatched_prediction": report.get("unmatched_prediction"),
        "duplicate_gold_keys": report.get("duplicate_gold_keys"),
        "duplicate_prediction_keys": report.get("duplicate_prediction_keys"),
    }


def first_split_value(record: dict[str, Any]) -> str:
    for key in ("split", "data_split", "subset"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("split", "data_split", "subset"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def value_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def load_labeled_records(path: Path, label: str) -> list[dict[str, Any]]:
    records = []
    for record in load_records(path):
        row = dict(record)
        row["_input_label"] = label
        records.append(row)
    return records


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield ensure_record(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in unwrap_records(data):
            yield ensure_record(item)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required to read parquet files.") from exc
        for item in pd.read_parquet(path).to_dict(orient="records"):
            yield ensure_record(item)
        return
    raise ValueError(f"Unsupported input format: {path}")


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


def write_points_csv(path: Path, points: Sequence[dict[str, Any]]) -> None:
    if not points:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POINT_CSV_COLUMNS)
        writer.writeheader()
        for point in points:
            writer.writerow({key: point.get(key, "") for key in POINT_CSV_COLUMNS})


def output_artifact_metadata(*, points_path: Path, svg_path: Path, pdf_path: Path, n_points: int) -> dict[str, Any]:
    point_csv_rows_count = count_csv_rows(points_path)
    return {
        "output_artifacts_schema_version": 1,
        "point_csv": str(points_path),
        "point_csv_sha256": file_sha256(points_path),
        "point_csv_rows_count": point_csv_rows_count,
        "point_csv_rows_match_n_points": point_csv_rows_count == n_points,
        "svg": str(svg_path),
        "svg_sha256": file_sha256(svg_path),
        "pdf": str(pdf_path),
        "pdf_sha256": file_sha256(pdf_path),
    }


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
