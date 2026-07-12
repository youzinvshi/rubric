#!/usr/bin/env python3
"""Build multi-method BlindSpot-RL pipeline and audit manifest configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    config = load_methods_config(args.methods)
    pipeline = build_pipeline(config, manifest_path=str(args.manifest_output))
    manifest = build_manifest(config)

    args.pipeline_output.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_output.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote pipeline config to {args.pipeline_output}")
    print(f"Wrote audit manifest to {args.manifest_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-method experiment configs.")
    parser.add_argument("--methods", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    return parser.parse_args()


def load_methods_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Experiment matrix methods config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Experiment matrix methods config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Experiment matrix methods config must be a JSON object: {path}")
    return data


def build_pipeline(config: dict[str, Any], manifest_path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    common = config["common"]
    methods = config["methods"]
    output_root = common.get("output_root", "outputs")
    processed_root = common.get("processed_root", "data/processed")
    embedding_model = common.get("embedding_model", "BAAI/bge-large-en-v1.5")
    coverage_tau = common.get("coverage_tau", 0.75)
    redundancy_tau = common.get("redundancy_tau", 0.85)

    stages: list[dict[str, Any]] = []
    bsc_specs: list[str] = []
    downstream_specs: list[str] = []
    bsc_ci_specs: list[str] = []
    downstream_ci_specs: list[str] = []
    bsc_eval_paths: dict[str, str] = {}
    repair_summary_paths: list[str] = []
    trained_gate = trained_method_gate_stage(common, methods)
    if trained_gate:
        stages.append(trained_gate)
    for method in methods:
        name = method["name"]
        method_output = f"{output_root}/{name}"
        method_processed = f"{processed_root}/{name}"
        bsc_joined = f"{method_processed}/bsc_eval.jsonl"
        downstream_joined = f"{method_processed}/downstream_eval.jsonl"
        bsc_eval_paths[name] = bsc_joined

        stages.extend(
            [
                {
                    "name": f"{name}_prepare_bsc",
                    "type": "prepare_bsc",
                    "args": {
                        "gold": method.get("bsc_gold", common["bsc_gold"]),
                        "predictions": resolve_rubric_path(method, common, "bsc_rubrics"),
                        "output": bsc_joined,
                        "report": f"{method_output}/bsc_join_report.json",
                        "model": method.get("model_filter", name),
                        "data_source": method.get("bsc_data_source", common.get("bsc_data_source", "rubricbench")),
                    },
                },
                {
                    "name": f"{name}_bsc",
                    "type": "bsc",
                    "args": {
                        "input": bsc_joined,
                        "embedding_model": embedding_model,
                        "coverage_tau": coverage_tau,
                        "redundancy_tau": redundancy_tau,
                        "output_dir": f"{method_output}/bsc",
                    },
                },
            ]
        )
        if bsc_sweep_enabled(common, name):
            stages.append(
                bsc_sweep_stage(
                    name=f"{name}_bsc_sweep",
                    input_path=bsc_joined,
                    output_dir=f"{method_output}/bsc_sweep",
                    embedding_model=embedding_model,
                    common=common,
                )
            )
        if bsc_human_audit_pack_enabled(common, name):
            stages.append(
                bsc_human_audit_pack_stage(
                    name=f"{name}_bsc_human_audit_pack",
                    input_path=bsc_joined,
                    output_dir=f"{method_output}/bsc_human_audit_pack",
                    embedding_model=embedding_model,
                    coverage_tau=coverage_tau,
                    common=common,
                )
            )
        if not skip_downstream(method, common):
            downstream_type = common.get("downstream_type", "pairwise")
            prepare_type = "prepare_multicandidate" if downstream_type == "multicandidate" else "prepare_downstream"
            eval_type = "multicandidate_downstream" if downstream_type == "multicandidate" else "downstream"
            preference_arg = "benchmark" if downstream_type == "multicandidate" else "preferences"
            downstream_budget = downstream_budget_stage(
                name=f"{name}_downstream_api_budget",
                input_path=downstream_joined,
                output_dir=f"{method_output}/downstream_api_budget",
                downstream_type=downstream_type,
                common=common,
            )
            stages.append(
                {
                    "name": f"{name}_prepare_downstream",
                    "type": prepare_type,
                    "args": {
                        preference_arg: method.get("downstream_preferences", common["downstream_preferences"]),
                        "rubrics": resolve_rubric_path(method, common, "downstream_rubrics"),
                        "output": downstream_joined,
                        "report": f"{method_output}/downstream_join_report.json",
                        "model": method.get("model_filter", name),
                        "data_source": method.get(
                            "downstream_data_source",
                            common.get("downstream_data_source", "rewardbench"),
                        ),
                    },
                }
            )
            if downstream_budget:
                stages.append(downstream_budget)
            stages.append(
                {
                    "name": f"{name}_downstream",
                    "type": eval_type,
                    "args": downstream_args(
                        common,
                        downstream_joined,
                        f"{method_output}/downstream",
                        require_budget_report=(
                            downstream_budget["args"]["output"] if downstream_budget else None
                        ),
                    ),
                }
            )
        if bootstrap_ci_enabled(common):
            stages.extend(
                [
                    bootstrap_ci_stage(
                        name=f"{name}_bsc_bootstrap_ci",
                        input_path=f"{method_output}/bsc/per_item.csv",
                        output_dir=f"{method_output}/bsc_ci",
                        metrics=bootstrap_metrics(
                            common,
                            key="bsc_metrics",
                            default=["coverage", "blind", "redundancy", "hallucination", "reward"],
                        ),
                        common=common,
                    ),
                ]
            )
            if not skip_downstream(method, common):
                stages.append(
                    bootstrap_ci_stage(
                        name=f"{name}_downstream_bootstrap_ci",
                        input_path=f"{method_output}/downstream/per_item.csv",
                        output_dir=f"{method_output}/downstream_ci",
                        metrics=bootstrap_metrics(
                            common,
                            key="downstream_metrics",
                            default=["correct", "tie", "margin"],
                        ),
                        common=common,
                    )
                )
        bsc_specs.append(f"{name}={method_output}/bsc/summary.json")
        if not skip_downstream(method, common):
            downstream_specs.append(f"{name}={method_output}/downstream/summary.json")
        if bootstrap_ci_enabled(common):
            bsc_ci_specs.append(f"{name}={method_output}/bsc_ci/bootstrap_ci.json")
            if not skip_downstream(method, common):
                downstream_ci_specs.append(f"{name}={method_output}/downstream_ci/bootstrap_ci.json")

    if reward_ablation_enabled(common):
        stages.append(reward_ablation_stage(common=common, bsc_eval_paths=bsc_eval_paths, embedding_model=embedding_model))
    if teacher_union_ablation_enabled(common):
        stages.append(teacher_union_ablation_stage(common=common, embedding_model=embedding_model))
    if verifier_filter_ablation_enabled(common):
        stages.append(verifier_filter_ablation_stage(common=common, embedding_model=embedding_model))
    if blindspot_repair_enabled(common):
        repair_stages, repair_summary_paths = blindspot_repair_stages(
            common=common,
            bsc_eval_paths=bsc_eval_paths,
            embedding_model=embedding_model,
            output_root=output_root,
        )
        stages.extend(repair_stages)
    semantic_space_output_dir = ""
    if semantic_space_enabled(common):
        semantic_stage = semantic_space_stage(
            common=common,
            bsc_eval_paths=bsc_eval_paths,
            embedding_model=embedding_model,
            output_root=output_root,
        )
        semantic_space_output_dir = semantic_stage["args"]["output_dir"]
        stages.append(semantic_stage)

    summarize_args: dict[str, Any] = {
        "bsc": bsc_specs,
        "output_csv": f"{output_root}/main_table.csv",
        "output_md": f"{output_root}/main_table.md",
    }
    if downstream_specs:
        summarize_args["downstream"] = downstream_specs
    if bsc_ci_specs:
        summarize_args["bsc_ci"] = bsc_ci_specs
    if downstream_ci_specs:
        summarize_args["downstream_ci"] = downstream_ci_specs

    stages.extend(
        [
            {
                "name": "summarize",
                "type": "summarize",
                "args": summarize_args,
            },
            {
                "name": "audit",
                "type": "audit",
                "args": {
                    "manifest": manifest_path or config.get("manifest_path", "configs/experiment_matrix_manifest.local.json"),
                    "output": f"{output_root}/audit_report.json",
                },
            },
            {
                "name": "export",
                "type": "export",
                "args": export_args(
                    common,
                    output_root,
                    repair_summary_paths=repair_summary_paths,
                    semantic_space_output_dir=semantic_space_output_dir,
                ),
            },
        ]
    )
    return {"stages": stages}


def bootstrap_ci_enabled(common: dict[str, Any]) -> bool:
    return bool(common.get("bootstrap_ci", {}).get("enabled", False))


def skip_downstream(method: dict[str, Any], common: dict[str, Any]) -> bool:
    return bool(method.get("skip_downstream", common.get("skip_downstream", False)))


def bsc_sweep_enabled(common: dict[str, Any], method_name: str) -> bool:
    config = common.get("bsc_sweep", {})
    if not config.get("enabled", False):
        return False
    selected = config.get("methods")
    return selected is None or method_name in set(selected)


def bsc_human_audit_pack_enabled(common: dict[str, Any], method_name: str) -> bool:
    config = common.get("bsc_human_audit_pack", {})
    if not config.get("enabled", False):
        return False
    selected = config.get("methods")
    return selected is None or method_name in set(selected)


def trained_method_gate_stage(common: dict[str, Any], methods: list[dict[str, Any]]) -> dict[str, Any] | None:
    config = common.get("trained_method_gate", {})
    if not config.get("enabled", False):
        return None
    trained_filters = set(config.get("model_filters", ["sft_only", "sft_rl"]))
    required_methods = [
        str(method.get("model_filter") or method.get("name"))
        for method in methods
        if str(method.get("model_filter") or method.get("name")) in trained_filters
    ]
    required_methods = dedupe(required_methods)
    if not required_methods:
        return None
    args: dict[str, Any] = {
        "name": config.get("gate_name", config.get("name", "trained_method_gate")),
        "required_path": config.get("required_paths", []),
        "required_json": config.get("required_json", []),
        "required_json_contains": config.get("required_json_contains", []),
        "required_json_equals": config.get("required_json_equals", []),
        "required_json_sha256": config.get("required_json_sha256", []),
        "instructions": config.get("instructions", []),
        "output": config["output"],
        "output_md": config.get("output_md"),
        "strict": config.get("strict", True),
    }
    method_identity_specs = [config.get("served_methods"), config.get("served_generators")]
    method_identity_specs = [spec for spec in method_identity_specs if spec]
    if method_identity_specs:
        args["required_json_contains"] = list(args["required_json_contains"]) + [
            f"{spec}={','.join(required_methods)}" for spec in method_identity_specs
        ]
    return {
        "name": config.get("name", "trained_method_gate"),
        "type": "manual_gate",
        "args": args,
    }


def bsc_sweep_stage(
    name: str,
    input_path: str,
    output_dir: str,
    embedding_model: str,
    common: dict[str, Any],
) -> dict[str, Any]:
    config = common.get("bsc_sweep", {})
    return {
        "name": name,
        "type": "bsc_sweep",
        "args": {
            "input": input_path,
            "embedding_model": config.get("embedding_model", embedding_model),
            "coverage_tau": config.get("coverage_tau", [0.70, 0.75, 0.80]),
            "redundancy_tau": config.get("redundancy_tau", [0.80, 0.85, 0.90]),
            "output_dir": output_dir,
        },
    }


def bsc_human_audit_pack_stage(
    name: str,
    input_path: str,
    output_dir: str,
    embedding_model: str,
    coverage_tau: float,
    common: dict[str, Any],
) -> dict[str, Any]:
    config = common.get("bsc_human_audit_pack", {})
    return {
        "name": name,
        "type": "bsc_human_audit_pack",
        "args": {
            "input": input_path,
            "embedding_model": config.get("embedding_model", embedding_model),
            "coverage_tau": config.get("coverage_tau", coverage_tau),
            "matched": config.get("matched", 25),
            "unmatched": config.get("unmatched", 25),
            "seed": config.get("seed", 13),
            "output_dir": output_dir,
        },
    }


def reward_ablation_enabled(common: dict[str, Any]) -> bool:
    return bool(common.get("reward_ablation", {}).get("enabled", False))


def reward_ablation_stage(
    common: dict[str, Any],
    bsc_eval_paths: dict[str, str],
    embedding_model: str,
) -> dict[str, Any]:
    config = common.get("reward_ablation", {})
    method = config.get("method", "sft_rl")
    input_path = config.get("input") or bsc_eval_paths.get(method)
    if not input_path:
        raise ValueError(f"Missing BSC eval input for reward ablation method: {method}")
    output_dir = config.get("output_dir", "outputs/bsc_ablation")
    return {
        "name": config.get("name", "reward_component_ablation"),
        "type": "ablation",
        "args": {
            "input": input_path,
            "embedding_model": config.get("embedding_model", embedding_model),
            "coverage_tau": config.get("coverage_tau", common.get("coverage_tau", 0.75)),
            "redundancy_tau": config.get("redundancy_tau", common.get("redundancy_tau", 0.85)),
            "output_dir": output_dir,
        },
    }


def teacher_union_ablation_enabled(common: dict[str, Any]) -> bool:
    return bool(common.get("teacher_union_ablation", {}).get("enabled", False))


def teacher_union_ablation_stage(common: dict[str, Any], embedding_model: str) -> dict[str, Any]:
    config = common.get("teacher_union_ablation", {})
    output_dir = config.get("output_dir", "outputs/teacher_union_ablation")
    return {
        "name": config.get("name", "teacher_union_ablation"),
        "type": "teacher_union_ablation",
        "args": {
            "teachers": config.get("teachers", common.get("teacher_rubrics", "data/processed/teacher_rubrics_raw.jsonl")),
            "gold": config.get("gold", common["bsc_gold"]),
            "embedding_model": config.get("embedding_model", embedding_model),
            "coverage_tau": config.get("coverage_tau", common.get("coverage_tau", 0.75)),
            "redundancy_tau": config.get("redundancy_tau", common.get("redundancy_tau", 0.85)),
            "dedupe_tau": config.get("dedupe_tau", common.get("dedupe_tau", 0.85)),
            "min_teachers": config.get("min_teachers", 2),
            "output_dir": output_dir,
        },
    }


def verifier_filter_ablation_enabled(common: dict[str, Any]) -> bool:
    return bool(common.get("verifier_filter_ablation", {}).get("enabled", False))


def verifier_filter_ablation_stage(common: dict[str, Any], embedding_model: str) -> dict[str, Any]:
    config = common.get("verifier_filter_ablation", {})
    output_dir = config.get("output_dir", "outputs/verifier_filter_ablation")
    return {
        "name": config.get("name", "verifier_filter_ablation"),
        "type": "verifier_filter_ablation",
        "args": {
            "raw_teachers": config.get("raw_teachers", common.get("teacher_rubrics", "data/processed/teacher_rubrics_raw.jsonl")),
            "filtered_teachers": config.get("filtered_teachers", "data/processed/teacher_rubrics_filtered.jsonl"),
            "gold": config.get("gold", common["bsc_gold"]),
            "embedding_model": config.get("embedding_model", embedding_model),
            "coverage_tau": config.get("coverage_tau", common.get("coverage_tau", 0.75)),
            "redundancy_tau": config.get("redundancy_tau", common.get("redundancy_tau", 0.85)),
            "dedupe_tau": config.get("dedupe_tau", common.get("dedupe_tau", 0.85)),
            "min_teachers": config.get("min_teachers", 2),
            "output_dir": output_dir,
        },
    }


def dimension_transition_config(common: dict[str, Any]) -> dict[str, Any]:
    """Return the current dimension-transition config, accepting legacy local keys."""

    config = common.get("dimension_transition")
    if isinstance(config, dict):
        return config
    legacy_config = common.get("blindspot_repair")
    return legacy_config if isinstance(legacy_config, dict) else {}


def dimension_transition_enabled(common: dict[str, Any]) -> bool:
    return bool(dimension_transition_config(common).get("enabled", False))


def blindspot_repair_enabled(common: dict[str, Any]) -> bool:
    return dimension_transition_enabled(common)


def blindspot_repair_stages(
    common: dict[str, Any],
    bsc_eval_paths: dict[str, str],
    embedding_model: str,
    output_root: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    config = dimension_transition_config(common)
    baseline = config.get("baseline", "base")
    baseline_path = config.get("baseline_input") or bsc_eval_paths.get(baseline)
    if not baseline_path:
        raise ValueError(f"Missing baseline BSC eval input for dimension_transition baseline: {baseline}")

    candidates = list(config.get("candidates") or [name for name in bsc_eval_paths if name != baseline])
    if not candidates:
        raise ValueError("dimension_transition requires at least one candidate method")

    output_dir = config.get("output_dir", f"{output_root}/dimension_transition")
    stages: list[dict[str, Any]] = []
    summary_paths: list[str] = []
    for candidate in candidates:
        candidate_path = config.get("candidate_inputs", {}).get(candidate) if isinstance(config.get("candidate_inputs"), dict) else None
        candidate_path = candidate_path or bsc_eval_paths.get(candidate)
        if not candidate_path:
            raise ValueError(f"Missing candidate BSC eval input for dimension_transition candidate: {candidate}")
        pair_name = f"{baseline}_to_{candidate}"
        pair_output_dir = f"{output_dir}/{pair_name}"
        args: dict[str, Any] = {
            "baseline": baseline_path,
            "candidate": candidate_path,
            "output_dir": pair_output_dir,
            "embedding_model": config.get("embedding_model", embedding_model),
            "coverage_tau": config.get("coverage_tau", common.get("coverage_tau", 0.75)),
            "join_key": config.get("join_key", "query"),
            "baseline_label": config.get("baseline_label", baseline),
            "candidate_label": config.get("candidate_labels", {}).get(candidate, candidate)
            if isinstance(config.get("candidate_labels"), dict)
            else candidate,
        }
        if config.get("ignore_valid_flags"):
            args["ignore_valid_flags"] = True
        stages.append(
            {
                "name": config.get("name_prefix", "dimension_transition") + f"_{pair_name}",
                "type": "dimension_transition",
                "args": args,
            }
        )
        summary_paths.append(f"{pair_output_dir}/transition_summary.json")
    return stages, summary_paths


def semantic_space_enabled(common: dict[str, Any]) -> bool:
    return bool(common.get("semantic_space", {}).get("enabled", False))


def semantic_space_stage(
    common: dict[str, Any],
    bsc_eval_paths: dict[str, str],
    embedding_model: str,
    output_root: str,
) -> dict[str, Any]:
    config = common.get("semantic_space", {})
    methods = list(config.get("methods") or bsc_eval_paths.keys())
    if not methods:
        raise ValueError("semantic_space requires at least one method")
    input_specs = []
    join_report_specs = []
    for method in methods:
        path = config.get("inputs", {}).get(method) if isinstance(config.get("inputs"), dict) else None
        path = path or bsc_eval_paths.get(method)
        if not path:
            raise ValueError(f"Missing BSC eval input for semantic_space method: {method}")
        input_specs.append(f"{method}={path}")
        join_report = config.get("join_reports", {}).get(method) if isinstance(config.get("join_reports"), dict) else None
        join_report = join_report or f"{output_root}/{method}/bsc_join_report.json"
        join_report_specs.append(f"{method}={join_report}")
    args: dict[str, Any] = {
        "input": input_specs,
        "join_report": join_report_specs,
        "output_dir": config.get("output_dir", f"{output_root}/semantic_space"),
        "embedding_model": config.get("embedding_model", embedding_model),
        "projection": config.get("projection", "pca"),
        "gold_cluster_tau": config.get("gold_cluster_tau", common.get("coverage_tau", 0.75)),
    }
    if config.get("max_points") is not None:
        args["max_points"] = config["max_points"]
    return {
        "name": config.get("name", "semantic_space"),
        "type": "semantic_space",
        "args": args,
    }


def bootstrap_metrics(common: dict[str, Any], key: str, default: list[str]) -> list[str]:
    config = common.get("bootstrap_ci", {})
    metrics = config.get(key)
    if metrics is None:
        metrics = config.get("metrics")
    return list(metrics or default)


def bootstrap_ci_stage(
    name: str,
    input_path: str,
    output_dir: str,
    metrics: list[str],
    common: dict[str, Any],
) -> dict[str, Any]:
    config = common.get("bootstrap_ci", {})
    return {
        "name": name,
        "type": "bootstrap_ci",
        "args": {
            "input": input_path,
            "metric": metrics,
            "n_boot": config.get("n_boot", 1000),
            "seed": config.get("seed", 13),
            "confidence": config.get("confidence", 0.95),
            "output_json": f"{output_dir}/bootstrap_ci.json",
            "output_csv": f"{output_dir}/bootstrap_ci.csv",
            "output_md": f"{output_dir}/bootstrap_ci.md",
        },
    }


def downstream_budget_stage(
    name: str,
    input_path: str,
    output_dir: str,
    downstream_type: str,
    common: dict[str, Any],
) -> dict[str, Any] | None:
    if common.get("downstream_scorer") != "api":
        return None
    provider = common.get("judge_provider")
    if not provider:
        raise ValueError("downstream_scorer=api requires judge_provider for budget gating")
    config = common.get("judge_api_budget", {})
    validate_downstream_budget_contract(config, downstream_type)
    args: dict[str, Any] = {
        "input": input_path,
        "providers": provider,
        "unit_field": config.get("unit_field", "rubrics"),
        "calls_per_record_per_provider": config.get(
            "calls_per_record_per_provider",
            1 if downstream_type == "multicandidate" else 2,
        ),
        "default_qpm": config.get("default_qpm", 60),
        "default_tpm": config.get("default_tpm", 60000),
        "max_calls": config.get("max_calls", 500000),
        "max_total_tokens": config.get("max_total_tokens", 500000000),
        "max_cost_usd": config.get("max_cost_usd", 15000),
        "max_wallclock_minutes_serial": config.get("max_wallclock_minutes_serial", 40000),
        "output": f"{output_dir}/budget.json",
        "output_md": f"{output_dir}/budget.md",
        "strict": config.get("strict", True),
    }
    if downstream_type == "multicandidate":
        args["unit_multiplier_field"] = config.get("unit_multiplier_field", "candidates")
    return {"name": name, "type": "api_budget", "args": args}


def validate_downstream_budget_contract(config: dict[str, Any], downstream_type: str) -> None:
    expected = {
        "unit_field": "rubrics",
        "calls_per_record_per_provider": 1 if downstream_type == "multicandidate" else 2,
    }
    if downstream_type == "multicandidate":
        expected["unit_multiplier_field"] = "candidates"
    for key, expected_value in expected.items():
        if key in config and config[key] != expected_value:
            raise ValueError(
                f"judge_api_budget.{key} must be {expected_value!r} for {downstream_type} downstream"
            )
    if downstream_type != "multicandidate" and "unit_multiplier_field" in config:
        raise ValueError("judge_api_budget.unit_multiplier_field is only valid for multicandidate downstream")


def downstream_args(
    common: dict[str, Any],
    input_path: str,
    output_dir: str,
    require_budget_report: str | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"input": input_path, "output_dir": output_dir}
    if common.get("downstream_scorer"):
        args["scorer"] = common["downstream_scorer"]
    if common.get("judge_provider"):
        args["provider"] = common["judge_provider"]
    if common.get("judge_sleep") is not None:
        args["sleep"] = common["judge_sleep"]
    if require_budget_report:
        args["require_budget_report"] = require_budget_report
    return args


def resolve_rubric_path(method: dict[str, Any], common: dict[str, Any], key: str) -> str:
    path = method.get(key) or common.get(key) or common.get("model_rubrics")
    if not path:
        raise ValueError(f"Missing {key} for method {method.get('name', '<unknown>')}")
    return str(path)


def export_args(
    common: dict[str, Any],
    output_root: str,
    repair_summary_paths: list[str] | None = None,
    semantic_space_output_dir: str | None = None,
) -> dict[str, Any]:
    args = {
        "main_table_csv": f"{output_root}/main_table.csv",
        "main_table_md": f"{output_root}/main_table.md",
        "audit_report": f"{output_root}/audit_report.json",
        "output_dir": f"{output_root}/paper_artifacts",
    }
    if common.get("ablation_csv"):
        args["ablation_csv"] = common["ablation_csv"]
    elif reward_ablation_enabled(common):
        output_dir = common.get("reward_ablation", {}).get("output_dir", "outputs/bsc_ablation")
        args["ablation_csv"] = f"{output_dir}/ablation_summary.csv"
    if common.get("teacher_union_csv"):
        args["teacher_union_csv"] = common["teacher_union_csv"]
    elif teacher_union_ablation_enabled(common):
        output_dir = common.get("teacher_union_ablation", {}).get("output_dir", "outputs/teacher_union_ablation")
        args["teacher_union_csv"] = f"{output_dir}/teacher_union_ablation.csv"
    if common.get("verifier_filter_csv"):
        args["verifier_filter_csv"] = common["verifier_filter_csv"]
    elif verifier_filter_ablation_enabled(common):
        output_dir = common.get("verifier_filter_ablation", {}).get("output_dir", "outputs/verifier_filter_ablation")
        args["verifier_filter_csv"] = f"{output_dir}/verifier_filter_ablation.csv"
    if common.get("downstream_table_csv"):
        args["downstream_table_csv"] = as_list(common["downstream_table_csv"])
    if common.get("transition_summary_json"):
        args["transition_summary_json"] = as_list(common["transition_summary_json"])
    elif common.get("repair_summary_json"):
        args["transition_summary_json"] = as_list(common["repair_summary_json"])
    elif repair_summary_paths:
        args["transition_summary_json"] = repair_summary_paths
    if common.get("semantic_space_dir"):
        args["semantic_space_dir"] = common["semantic_space_dir"]
    elif semantic_space_output_dir:
        args["semantic_space_dir"] = semantic_space_output_dir
    if common.get("evidence_md"):
        args["evidence_md"] = common["evidence_md"]
    if common.get("evidence_json"):
        args["evidence_json"] = common["evidence_json"]
    if common.get("evidence_csv"):
        args["evidence_csv"] = common["evidence_csv"]
    return args


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    common = config["common"]
    methods = config["methods"]
    output_root = common.get("output_root", "outputs")
    processed_root = common.get("processed_root", "data/processed")

    required_files: list[str] = []
    summaries = []
    trained_gate = trained_method_gate_stage(common, methods)
    if trained_gate:
        gate_args = trained_gate["args"]
        required_files.extend([gate_args.get("output", ""), gate_args.get("output_md", "")])
    for method in methods:
        name = method["name"]
        method_output = f"{output_root}/{name}"
        method_processed = f"{processed_root}/{name}"
        required_files.extend(
            [
                f"{method_processed}/bsc_eval.jsonl",
                f"{method_output}/bsc_join_report.json",
                f"{method_output}/bsc/summary.json",
            ]
        )
        if not skip_downstream(method, common):
            required_files.extend(
                [
                    f"{method_processed}/downstream_eval.jsonl",
                    f"{method_output}/downstream_join_report.json",
                    f"{method_output}/downstream/summary.json",
                ]
            )
            if common.get("downstream_scorer") == "api":
                required_files.extend(
                    [
                        f"{method_output}/downstream_api_budget/budget.json",
                        f"{method_output}/downstream_api_budget/budget.md",
                    ]
                )
        if bootstrap_ci_enabled(common):
            required_files.extend(
                [
                    f"{method_output}/bsc_ci/bootstrap_ci.json",
                    f"{method_output}/bsc_ci/bootstrap_ci.csv",
                    f"{method_output}/bsc_ci/bootstrap_ci.md",
                ]
            )
            if not skip_downstream(method, common):
                required_files.extend(
                    [
                        f"{method_output}/downstream_ci/bootstrap_ci.json",
                        f"{method_output}/downstream_ci/bootstrap_ci.csv",
                        f"{method_output}/downstream_ci/bootstrap_ci.md",
                    ]
                )
        if bsc_sweep_enabled(common, name):
            required_files.extend(
                [
                    f"{method_output}/bsc_sweep/threshold_sweep.csv",
                    f"{method_output}/bsc_sweep/threshold_sweep.json",
                    f"{method_output}/bsc_sweep/threshold_sweep.md",
                ]
            )
        if bsc_human_audit_pack_enabled(common, name):
            required_files.extend(
                [
                    f"{method_output}/bsc_human_audit_pack/summary.json",
                    f"{method_output}/bsc_human_audit_pack/audit_items.csv",
                    f"{method_output}/bsc_human_audit_pack/audit_items.jsonl",
                    f"{method_output}/bsc_human_audit_pack/human_label_summary.json",
                    f"{method_output}/bsc_human_audit_pack/human_label_summary.md",
                ]
            )
        summaries.append(
            {
                "name": f"{name}_bsc",
                "path": f"{method_output}/bsc/summary.json",
                "required_keys": [
                    "n",
                    "input",
                    "embedding_model",
                    "coverage_tau",
                    "redundancy_tau",
                    "data_source_counts",
                    "verifier_source",
                    "verifier_source_counts",
                    "mean_coverage",
                    "mean_blind",
                    "mean_redundancy",
                    "mean_hallucination",
                ],
            }
        )
        if not skip_downstream(method, common):
            summaries.append(
                {
                    "name": f"{name}_downstream",
                    "path": f"{method_output}/downstream/summary.json",
                    "required_keys": ["accuracy", "tie_rate", "mean_margin"],
                }
            )
    required_files.extend([f"{output_root}/main_table.csv", f"{output_root}/main_table.md"])
    if reward_ablation_enabled(common):
        output_dir = common.get("reward_ablation", {}).get("output_dir", "outputs/bsc_ablation")
        required_files.extend(
            [
                f"{output_dir}/ablation_summary.csv",
                f"{output_dir}/ablation_summary.md",
                f"{output_dir}/variants/full_summary.json",
                f"{output_dir}/variants/no_red_summary.json",
                f"{output_dir}/variants/no_valid_summary.json",
                f"{output_dir}/variants/no_verifier_summary.json",
                f"{output_dir}/variants/cov_only_summary.json",
            ]
        )
    if teacher_union_ablation_enabled(common):
        output_dir = common.get("teacher_union_ablation", {}).get("output_dir", "outputs/teacher_union_ablation")
        required_files.extend(
            [
                f"{output_dir}/teacher_union_per_item.csv",
                f"{output_dir}/teacher_union_ablation.csv",
                f"{output_dir}/teacher_union_ablation.md",
                f"{output_dir}/teacher_union_ablation.json",
            ]
        )
    if verifier_filter_ablation_enabled(common):
        output_dir = common.get("verifier_filter_ablation", {}).get("output_dir", "outputs/verifier_filter_ablation")
        required_files.extend(
            [
                f"{output_dir}/verifier_filter_per_item.csv",
                f"{output_dir}/verifier_filter_ablation.csv",
                f"{output_dir}/verifier_filter_ablation.md",
                f"{output_dir}/verifier_filter_ablation.json",
            ]
        )
    if common.get("transition_summary_json"):
        required_files.extend(as_list(common["transition_summary_json"]))
    elif common.get("repair_summary_json"):
        required_files.extend(as_list(common["repair_summary_json"]))
    elif dimension_transition_enabled(common):
        transition_config = dimension_transition_config(common)
        output_dir = transition_config.get("output_dir", f"{output_root}/dimension_transition")
        baseline = transition_config.get("baseline", "base")
        candidates = transition_config.get("candidates")
        if candidates is None:
            candidates = [method["name"] for method in methods if method["name"] != baseline]
        for candidate in candidates:
            pair_dir = f"{output_dir}/{baseline}_to_{candidate}"
            required_files.extend(
                [
                    f"{pair_dir}/transition_summary.json",
                    f"{pair_dir}/transition_per_item.csv",
                    f"{pair_dir}/transition_by_category.csv",
                    f"{pair_dir}/transition_gold_items.jsonl",
                ]
            )
    if common.get("semantic_space_dir"):
        semantic_dir = common["semantic_space_dir"]
        required_files.extend(
            [
                f"{semantic_dir}/semantic_space.svg",
                f"{semantic_dir}/semantic_space.pdf",
                f"{semantic_dir}/semantic_space_points.csv",
                f"{semantic_dir}/semantic_space_summary.json",
            ]
        )
    elif semantic_space_enabled(common):
        semantic_dir = common.get("semantic_space", {}).get("output_dir", f"{output_root}/semantic_space")
        required_files.extend(
            [
                f"{semantic_dir}/semantic_space.svg",
                f"{semantic_dir}/semantic_space.pdf",
                f"{semantic_dir}/semantic_space_points.csv",
                f"{semantic_dir}/semantic_space_summary.json",
            ]
        )
    return {"required_files": dedupe([path for path in required_files if path]), "summaries": summaries}


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


if __name__ == "__main__":
    main()
