#!/usr/bin/env python3
"""Assemble the end-to-end real BlindSpot-RL run pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    config = read_json_object(args.config, "Real-run assembly config")
    pipeline = build_pipeline(config)
    manifest = build_manifest(config)

    args.pipeline_output.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_output.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote real-run pipeline to {args.pipeline_output}")
    print(f"Wrote real-run manifest to {args.manifest_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble full real-run BlindSpot-RL pipeline config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    return parser.parse_args()


def build_pipeline(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stages: list[dict[str, Any]] = []
    preflight_report = preflight_report_path(config["preflight_pipeline"])
    stages.extend(load_pipeline_stages(config["data_pipeline"]))
    if config.get("data_audit", {}).get("enabled", True):
        stages.append(audit_stage(config["data_audit"]))
    stages.extend(holdout_contamination_filter_stages(config.get("holdout_contamination_filters", [])))
    for audit_config in enabled_holdout_contamination_audit_configs(config, "pre_sft_holdout_contamination_audits"):
        stages.append(holdout_contamination_audit_stage(audit_config))
    stages.extend(load_pipeline_stages(config["preflight_pipeline"]))
    stages.extend(load_pipeline_stages(config["api_budget_pipeline"]))
    if config.get("sft_data_pipeline"):
        stages.extend(load_pipeline_stages(config["sft_data_pipeline"]))
    for audit_config in holdout_contamination_audit_configs(config):
        stages.append(holdout_contamination_audit_stage(audit_config))
    if config.get("training_commands", {}).get("enabled", True):
        stages.append(training_commands_stage(config["training_commands"]))
    if config.get("training_completion_gate", {}).get("enabled", True):
        stages.append(manual_gate_stage(config["training_completion_gate"]))
    if config.get("reward_component_ablation_eval", {}).get("enabled", False):
        stages.extend(
            reward_component_ablation_eval_stages(
                config["reward_component_ablation_eval"],
                require_preflight_report=preflight_report,
            )
        )
    if config.get("model_generation", {}).get("enabled", True):
        model_generation = dict(config["model_generation"])
        if preflight_report:
            model_generation.setdefault("require_preflight_report", preflight_report)
        stages.extend(model_generation_stages(model_generation))
    if config.get("model_verification", {}).get("enabled", False):
        model_verification = dict(config["model_verification"])
        if preflight_report:
            model_verification.setdefault("require_preflight_report", preflight_report)
        stages.extend(model_verification_stages(model_verification, config["model_generation"]))
    stages.extend(load_pipeline_stages(config["rubric_validation_pipeline"]))
    stages.extend(load_pipeline_stages(config["matrix_pipeline"]))
    for extra in config.get("extra_pipelines", []):
        stages.extend(load_pipeline_stages(extra["pipeline"]))
    if config.get("generalization_pipeline"):
        stages.extend(load_pipeline_stages(config["generalization_pipeline"]))
    if config.get("downstream_rlvr_data_pipeline"):
        stages.extend(load_pipeline_stages(config["downstream_rlvr_data_pipeline"]))
    if config.get("downstream_rlvr_commands", {}).get("enabled", False):
        stages.append(downstream_rlvr_commands_stage(config["downstream_rlvr_commands"]))
    if config.get("downstream_rlvr_completion_gate", {}).get("enabled", False):
        stages.append(manual_gate_stage(config["downstream_rlvr_completion_gate"]))
    stages.append(evidence_stage(config["evidence"]))
    if config.get("final_paper_export", {}).get("enabled", False):
        stages.append(final_paper_export_stage(config["final_paper_export"]))
    if config.get("paper_sync", {}).get("enabled", False):
        stages.append(sync_paper_stage(config["paper_sync"]))
        stages.append(paper_asset_index_check_stage(config["paper_sync"]))
    stages.extend(load_pipeline_stages(config["submission_readiness_pipeline"]))
    if config.get("rebuttal_pack", {}).get("enabled", False):
        stages.append(rebuttal_pack_stage(config["rebuttal_pack"]))
    if config.get("submission_gap_report", {}).get("enabled", False):
        stages.append(submission_gap_report_stage(config["submission_gap_report"], config.get("rebuttal_pack")))
    stages.append(dashboard_stage(config["dashboard"]))
    stages.append(result_card_stage(config["result_card"]))
    if config.get("paper_sync", {}).get("enabled", False):
        stages.append(
            result_card_sync_stage(
                config["paper_sync"],
                config["dashboard"],
                config["result_card"],
                config.get("rebuttal_pack"),
                config.get("submission_gap_report"),
                config.get("evidence"),
            )
        )
        stages.append(final_paper_asset_index_check_stage(config["paper_sync"]))
    return {"stages": stages}


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required_files: list[str] = []
    summaries: list[dict[str, Any]] = []
    manifest_paths = [config["data_manifest"], config["matrix_manifest"]]
    if config.get("sft_data_manifest"):
        manifest_paths.append(config["sft_data_manifest"])
    for extra in config.get("extra_pipelines", []):
        if extra.get("manifest"):
            manifest_paths.append(extra["manifest"])
    if config.get("generalization_manifest"):
        manifest_paths.append(config["generalization_manifest"])
    for path in manifest_paths:
        manifest = read_json_object(Path(path), "Real-run component manifest")
        required_files.extend(manifest.get("required_files", []))
        summaries.extend(manifest.get("summaries", []))

    if config.get("data_audit", {}).get("enabled", True):
        required_files.append(config["data_audit"]["output"])
    required_files.extend(holdout_contamination_filter_outputs(config.get("holdout_contamination_filters", [])))
    for audit_config in enabled_holdout_contamination_audit_configs(config, "pre_sft_holdout_contamination_audits"):
        required_files.extend(holdout_contamination_audit_outputs(audit_config))
    if config.get("model_generation", {}).get("enabled", True):
        required_files.extend(model_generation_outputs(config["model_generation"]))
    if config.get("model_verification", {}).get("enabled", False):
        required_files.extend(model_verification_outputs(config["model_verification"], config["model_generation"]))
    required_files.extend(collect_stage_outputs(load_pipeline_stages(config["preflight_pipeline"])))
    required_files.extend(collect_stage_outputs(load_pipeline_stages(config["api_budget_pipeline"])))
    if config.get("training_commands", {}).get("enabled", True):
        required_files.extend(training_command_outputs(config["training_commands"]))
    for audit_config in holdout_contamination_audit_configs(config):
        required_files.extend(holdout_contamination_audit_outputs(audit_config))
    if config.get("training_completion_gate", {}).get("enabled", True):
        required_files.extend(manual_gate_outputs(config["training_completion_gate"]))
    if config.get("reward_component_ablation_eval", {}).get("enabled", False):
        required_files.extend(reward_component_ablation_eval_outputs(config["reward_component_ablation_eval"]))
    if config.get("downstream_rlvr_data_pipeline"):
        required_files.extend(collect_stage_outputs(load_pipeline_stages(config["downstream_rlvr_data_pipeline"])))
    if config.get("downstream_rlvr_commands", {}).get("enabled", False):
        required_files.extend(downstream_rlvr_command_outputs(config["downstream_rlvr_commands"]))
    if config.get("downstream_rlvr_completion_gate", {}).get("enabled", False):
        required_files.extend(manual_gate_outputs(config["downstream_rlvr_completion_gate"]))
    required_files.extend(collect_validation_outputs(load_pipeline_stages(config["rubric_validation_pipeline"])))
    required_files.extend(
        [
            f"{config['evidence']['output_dir']}/evidence_matrix.json",
            f"{config['evidence']['output_dir']}/evidence_matrix.csv",
            f"{config['evidence']['output_dir']}/evidence_matrix.md",
        ]
    )
    if config.get("final_paper_export", {}).get("enabled", False):
        required_files.extend(final_paper_export_outputs(config["final_paper_export"]))
    if config.get("paper_sync", {}).get("enabled", False):
        required_files.extend(sync_paper_outputs(config["paper_sync"]))
        required_files.extend(paper_asset_index_check_outputs(config["paper_sync"]))
    required_files.extend(collect_stage_outputs(load_pipeline_stages(config["submission_readiness_pipeline"])))
    required_files.extend([config["dashboard"]["output_json"], config["dashboard"].get("output_md", "")])
    required_files.extend([config["result_card"]["output_json"], config["result_card"].get("output_md", "")])
    if config.get("rebuttal_pack", {}).get("enabled", False):
        required_files.extend(rebuttal_pack_outputs(config["rebuttal_pack"]))
    if config.get("submission_gap_report", {}).get("enabled", False):
        required_files.extend(submission_gap_report_outputs(config["submission_gap_report"]))
    return {
        "required_files": dedupe([path for path in required_files if path]),
        "summaries": summaries,
    }


def load_pipeline_stages(path: str | Path) -> list[dict[str, Any]]:
    pipeline_path = Path(path)
    pipeline = read_json_object(pipeline_path, "Real-run component pipeline")
    stages = pipeline.get("stages", [])
    if not isinstance(stages, list):
        raise SystemExit(f"Real-run component pipeline stages must be a list: {pipeline_path}")
    return list(stages)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"JSON file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON file is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}: line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return data


def audit_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "audit_data_readiness"),
        "type": "audit",
        "args": {
            "manifest": config["manifest"],
            "output": config["output"],
        },
    }


def model_generation_stages(config: dict[str, Any]) -> list[dict[str, Any]]:
    domains = config.get("domains")
    if not domains:
        return [model_generation_stage(config)]
    stages = []
    for domain in domains:
        domain_config = dict(config)
        domain_config.update(domain)
        domain_config.pop("domains", None)
        stages.append(model_generation_stage(domain_config))
    return stages


def model_generation_stage(config: dict[str, Any]) -> dict[str, Any]:
    args = {
        "input": config["input"],
        "providers": config["providers"],
        "output": config["output"],
        "data_source": config.get("data_source", "real_model_evaluation_criteria_elicitation"),
    }
    for key in [
        "system_prompt",
        "user_template",
        "limit",
        "sleep",
        "resume",
        "require_budget_report",
        "require_preflight_report",
    ]:
        if key in config and config[key] is not None:
            args[key] = config[key]
    return {"name": config.get("name", "generate_model_rubrics"), "type": "generate_model_rubrics", "args": args}


def model_generation_outputs(config: dict[str, Any]) -> list[str]:
    return [stage["args"]["output"] for stage in model_generation_stages(config)]


def model_verification_stages(config: dict[str, Any], generation_config: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    provider = config["provider"]
    for domain in model_generation_domains(generation_config):
        input_path = domain["output"]
        stem = Path(input_path).stem
        budget_output = config.get("budget_output_template", "outputs/api_budget/{stem}_verifier_budget.json").format(
            stem=stem,
            name=domain.get("name", stem),
        )
        budget_output_md = config.get(
            "budget_output_md_template",
            "outputs/api_budget/{stem}_verifier_budget.md",
        ).format(stem=stem, name=domain.get("name", stem))
        stats_output = config.get("stats_output_template", "outputs/verifier/{stem}_stats.jsonl").format(
            stem=stem,
            name=domain.get("name", stem),
        )
        budget_args = {
            "input": input_path,
            "providers": provider,
            "unit_field": "rubrics",
            "calls_per_record_per_provider": 1,
            "default_qpm": config.get("default_qpm", 60),
            "default_tpm": config.get("default_tpm", 60000),
            "max_calls": config.get("max_calls", 50000),
            "max_total_tokens": config.get("max_total_tokens", 150000000),
            "max_cost_usd": config.get("max_cost_usd", 5000),
            "max_wallclock_minutes_serial": config.get("max_wallclock_minutes_serial", 10000),
            "output": budget_output,
            "output_md": budget_output_md,
            "strict": config.get("strict_budget", True),
        }
        stages.append(
            {
                "name": f"api_budget_{stem}_verifier",
                "type": "api_budget",
                "args": budget_args,
            }
        )
        filter_args = {
            "input": input_path,
            "output": input_path,
            "stats_output": stats_output,
            "mode": config.get("mode", "api"),
            "provider": provider,
            "require_budget_report": budget_output,
            "sleep": config.get("sleep", 0.0),
            "annotate_only": config.get("annotate_only", True),
        }
        if config.get("require_preflight_report"):
            filter_args["require_preflight_report"] = config["require_preflight_report"]
        if config.get("drop_empty", False):
            filter_args["drop_empty"] = True
        stages.append(
            {
                "name": f"verify_{stem}",
                "type": "filter_verifier",
                "args": filter_args,
            }
        )
    return stages


def preflight_report_path(pipeline_path: str | Path) -> str | None:
    for stage in load_pipeline_stages(pipeline_path):
        if stage.get("type", stage.get("name")) == "preflight":
            output = stage.get("args", {}).get("output")
            if isinstance(output, str) and output:
                return output
    return None


def model_generation_domains(config: dict[str, Any]) -> list[dict[str, Any]]:
    domains = config.get("domains")
    if domains:
        merged = []
        for domain in domains:
            domain_config = dict(config)
            domain_config.update(domain)
            domain_config.pop("domains", None)
            merged.append(domain_config)
        return merged
    return [config]


def model_verification_outputs(config: dict[str, Any], generation_config: dict[str, Any]) -> list[str]:
    outputs: list[str] = []
    for stage in model_verification_stages(config, generation_config):
        args = stage.get("args", {})
        for key in ["output", "output_md", "stats_output"]:
            value = args.get(key)
            if isinstance(value, str):
                outputs.append(value)
    return outputs


def evidence_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "evidence_real"),
        "type": "evidence",
        "args": {
            "config": config["config"],
            "output_dir": config["output_dir"],
        },
    }


def sync_paper_stage(config: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "artifacts_dir": config["artifacts_dir"],
        "paper_dir": config.get("paper_dir", "paper"),
    }
    if config.get("extra_doc"):
        args["extra_doc"] = config["extra_doc"]
    return {
        "name": config.get("name", "sync_paper_artifacts"),
        "type": "sync_paper",
        "args": args,
    }


def final_paper_export_stage(config: dict[str, Any]) -> dict[str, Any]:
    args = {
        key: value
        for key, value in config.items()
        if key not in {"enabled", "name", "required_outputs"} and value is not None
    }
    return {
        "name": config.get("name", "final_paper_export"),
        "type": "export",
        "args": args,
    }


def final_paper_export_outputs(config: dict[str, Any]) -> list[str]:
    output_dir = config["output_dir"]
    defaults = [
        f"{output_dir}/main_table.tex",
        f"{output_dir}/rl_stage_ablation_table.tex",
        f"{output_dir}/downstream_utility_table.tex",
        f"{output_dir}/ablation_table.tex",
        f"{output_dir}/teacher_union_ablation_table.tex",
        f"{output_dir}/verifier_filter_ablation_table.tex",
        f"{output_dir}/dimension_transition_table.tex",
        f"{output_dir}/semantic_space.svg",
        f"{output_dir}/semantic_space.pdf",
        f"{output_dir}/semantic_space_points.csv",
        f"{output_dir}/semantic_space_summary.json",
        f"{output_dir}/experiment_summary.md",
    ]
    return list(config.get("required_outputs", defaults))


def sync_paper_outputs(config: dict[str, Any]) -> list[str]:
    paper_dir = config.get("paper_dir", "paper")
    return [f"{paper_dir}/asset_index.md"]


def paper_asset_index_check_stage(config: dict[str, Any]) -> dict[str, Any]:
    output, output_md = paper_asset_index_check_outputs(config)
    paper_dir = config.get("paper_dir", "paper")
    return {
        "name": config.get("asset_index_check_name", "paper_asset_index_check_real"),
        "type": "paper_asset_index_check",
        "args": {
            "asset_index": f"{paper_dir}/asset_index.md",
            "output": output,
            "output_md": output_md,
            "strict": True,
        },
    }


def final_paper_asset_index_check_stage(config: dict[str, Any]) -> dict[str, Any]:
    final_config = dict(config)
    final_config["asset_index_check_name"] = config.get(
        "final_asset_index_check_name",
        "paper_asset_index_check_final_real",
    )
    return paper_asset_index_check_stage(final_config)


def result_card_sync_stage(
    paper_sync: dict[str, Any],
    dashboard: dict[str, Any],
    result_card: dict[str, Any],
    rebuttal_pack: dict[str, Any] | None = None,
    submission_gap_report: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sync_config = dict(paper_sync)
    sync_config["name"] = paper_sync.get("result_card_sync_name", "sync_result_card_real")
    extra_docs = dashboard_outputs(dashboard)
    if evidence:
        extra_docs.extend(evidence_outputs(evidence))
    extra_docs.extend([result_card.get("output_json"), result_card.get("output_md")])
    extra_docs.extend(readiness_outputs(result_card))
    if rebuttal_pack:
        extra_docs.extend(readiness_outputs(rebuttal_pack))
    if submission_gap_report:
        extra_docs.extend(readiness_outputs(submission_gap_report))
    if rebuttal_pack and rebuttal_pack.get("enabled", False):
        extra_docs.extend(rebuttal_pack_outputs(rebuttal_pack))
    if submission_gap_report and submission_gap_report.get("enabled", False):
        extra_docs.extend(submission_gap_report_outputs(submission_gap_report))
    extra_docs.extend(sync_config.get("extra_doc", []))
    sync_config["extra_doc"] = [path for path in dedupe([str(path) for path in extra_docs if path])]
    return sync_paper_stage(sync_config)


def evidence_outputs(config: dict[str, Any]) -> list[str]:
    output_dir = config.get("output_dir", "outputs/evidence_real")
    return [
        f"{output_dir}/evidence_matrix.json",
        f"{output_dir}/evidence_matrix.csv",
        f"{output_dir}/evidence_matrix.md",
    ]


def readiness_outputs(config: dict[str, Any]) -> list[str]:
    readiness_json = config.get("readiness_report")
    if not readiness_json:
        return []
    outputs = [readiness_json]
    if str(readiness_json).endswith(".json"):
        outputs.append(str(readiness_json)[:-5] + ".md")
    return outputs


def rebuttal_pack_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "rebuttal_pack_real"),
        "type": "rebuttal_pack",
        "args": {
            "evidence_matrix": config["evidence_matrix"],
            "readiness_report": config.get("readiness_report"),
            "output_dir": config["output_dir"],
        },
    }


def rebuttal_pack_outputs(config: dict[str, Any]) -> list[str]:
    output_dir = config["output_dir"].rstrip("/")
    return [
        f"{output_dir}/rebuttal_pack.json",
        f"{output_dir}/rebuttal_pack.md",
        f"{output_dir}/rebuttal_pack_manifest.json",
    ]


def submission_gap_report_stage(
    config: dict[str, Any],
    rebuttal_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "readiness_report": config["readiness_report"],
        "evidence_matrix": config["evidence_matrix"],
        "output_dir": config["output_dir"],
    }
    rebuttal_manifest = config.get("rebuttal_manifest")
    if not rebuttal_manifest and rebuttal_pack and rebuttal_pack.get("enabled", False):
        rebuttal_manifest = rebuttal_pack_outputs(rebuttal_pack)[-1]
    if rebuttal_manifest:
        args["rebuttal_manifest"] = rebuttal_manifest
    if config.get("preflight_report"):
        args["preflight_report"] = config["preflight_report"]
    if config.get("gate_report"):
        args["gate_report"] = config["gate_report"]
    return {
        "name": config.get("name", "submission_gap_report_real"),
        "type": "submission_gap_report",
        "args": args,
    }


def submission_gap_report_outputs(config: dict[str, Any]) -> list[str]:
    output_dir = config["output_dir"].rstrip("/")
    return [
        f"{output_dir}/submission_gap_report.json",
        f"{output_dir}/submission_gap_report.md",
    ]


def dashboard_outputs(config: dict[str, Any]) -> list[str]:
    return [config.get("output_json", ""), config.get("output_md", "")]


def paper_asset_index_check_outputs(config: dict[str, Any]) -> list[str]:
    artifacts_dir = config["artifacts_dir"]
    return [
        config.get("asset_index_check_output", f"{artifacts_dir}/paper_asset_index_check.json"),
        config.get("asset_index_check_output_md", f"{artifacts_dir}/paper_asset_index_check.md"),
    ]


def training_commands_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "training_commands"),
        "type": "training_commands",
        "args": {
            "config": config["config"],
            "output_dir": config["output_dir"],
        },
    }


def holdout_contamination_filter_stages(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [holdout_contamination_filter_stage(config) for config in configs if config.get("enabled", True)]


def holdout_contamination_filter_stage(config: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "holdout": config["holdout"],
        "input": config["input"],
        "output": config["output"],
        "report": config["report"],
        "strict": config.get("strict", True),
    }
    if config.get("query_keys"):
        args["query_key"] = config["query_keys"]
    return {
        "name": config.get("name", "filter_holdout_contamination"),
        "type": "filter_holdout_contamination",
        "args": args,
    }


def holdout_contamination_filter_outputs(configs: list[dict[str, Any]]) -> list[str]:
    outputs: list[str] = []
    for config in configs:
        if not config.get("enabled", True):
            continue
        outputs.extend([config.get("output", ""), config.get("report", "")])
    return [path for path in outputs if path]


def holdout_contamination_audit_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    legacy = config.get("holdout_contamination_audit")
    if isinstance(legacy, dict) and legacy.get("enabled", False):
        configs.append(legacy)
    configs.extend(enabled_holdout_contamination_audit_configs(config, "holdout_contamination_audits"))
    return configs


def enabled_holdout_contamination_audit_configs(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [audit_config for audit_config in config.get(key, []) if audit_config.get("enabled", True)]


def holdout_contamination_audit_stage(config: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "holdout": config["holdout"],
        "training": config["training"],
        "output": config["output"],
        "strict": config.get("strict", True),
    }
    if config.get("output_csv"):
        args["output_csv"] = config["output_csv"]
    if config.get("query_keys"):
        args["query_key"] = config["query_keys"]
    if config.get("max_examples") is not None:
        args["max_examples"] = config["max_examples"]
    return {
        "name": config.get("name", "holdout_contamination_audit"),
        "type": "holdout_contamination_audit",
        "args": args,
    }


def holdout_contamination_audit_outputs(config: dict[str, Any]) -> list[str]:
    return [path for path in [config.get("output"), config.get("output_csv")] if path]


def downstream_rlvr_commands_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "downstream_rlvr_commands"),
        "type": "downstream_rlvr_commands",
        "args": {
            "config": config["config"],
            "output_dir": config["output_dir"],
        },
    }


def manual_gate_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "training_completion_gate"),
        "type": "manual_gate",
        "args": {
            "name": config.get("gate_name", config.get("name", "training_completion_gate")),
            "required_path": config.get("required_paths", []),
            "required_json": config.get("required_json", []),
            "required_json_contains": config.get("required_json_contains", []),
            "required_json_equals": config.get("required_json_equals", []),
            "required_json_sha256": config.get("required_json_sha256", []),
            "instructions": config.get("instructions", []),
            "output": config["output"],
            "output_md": config.get("output_md"),
            "strict": config.get("strict", True),
        },
    }


def reward_component_ablation_eval_stages(
    config: dict[str, Any],
    *,
    require_preflight_report: str | None = None,
) -> list[dict[str, Any]]:
    variants = reward_component_ablation_eval_variants(config)
    stages: list[dict[str, Any]] = []
    if config.get("gate", {}).get("enabled", True):
        stages.append(manual_gate_stage(config["gate"]))
    if config.get("budget", {}).get("enabled", True):
        budget_config = config["budget"]
        stages.append(
            {
                "name": budget_config.get("name", "api_budget_reward_component_ablation_rubrics"),
                "type": "api_budget",
                "args": {
                    "input": config["input"],
                    "providers": config["providers"],
                    "default_qpm": budget_config.get("default_qpm", 60),
                    "default_tpm": budget_config.get("default_tpm", 60000),
                    "max_calls": budget_config.get("max_calls", 500000),
                    "max_total_tokens": budget_config.get("max_total_tokens", 500000000),
                    "max_cost_usd": budget_config.get("max_cost_usd", 15000),
                    "max_wallclock_minutes_serial": budget_config.get("max_wallclock_minutes_serial", 40000),
                    "output": budget_config["output"],
                    "output_md": budget_config.get("output_md"),
                    "strict": budget_config.get("strict", True),
                },
            }
        )
    generation_config = {
        "name": config.get("generation_name", "generate_reward_component_ablation_rubrics"),
        "input": config["input"],
        "providers": config["providers"],
        "output": config["predictions"],
        "data_source": config.get(
            "data_source",
            "reward_component_ablation_evaluation_criteria_elicitation",
        ),
        "sleep": config.get("sleep", 0.5),
        "resume": config.get("resume", True),
    }
    if config.get("budget", {}).get("enabled", True):
        generation_config["require_budget_report"] = config["budget"]["output"]
    if require_preflight_report:
        generation_config["require_preflight_report"] = require_preflight_report
    stages.append(model_generation_stage(generation_config))
    for variant in variants:
        eval_input = f"{config['eval_data_dir'].rstrip('/')}/{variant}/bsc_eval.jsonl"
        output_dir = f"{config['output_dir'].rstrip('/')}/{variant}"
        stages.append(
            {
                "name": f"reward_ablation_{variant}_prepare_bsc",
                "type": "prepare_bsc",
                "args": {
                    "gold": config["gold"],
                    "predictions": config["predictions"],
                    "output": eval_input,
                    "report": f"{output_dir}/bsc_join_report.json",
                    "model": variant,
                    "data_source": config.get("bsc_data_source", "rubricbench"),
                },
            }
        )
        stages.append(
            {
                "name": f"reward_ablation_{variant}_bsc",
                "type": "bsc",
                "args": {
                    "input": eval_input,
                    "embedding_model": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    "coverage_tau": config.get("coverage_tau", 0.75),
                    "redundancy_tau": config.get("redundancy_tau", 0.85),
                    "output_dir": f"{output_dir}/bsc",
                },
            }
        )
    return stages


def reward_component_ablation_eval_variants(config: dict[str, Any]) -> list[str]:
    variants = config.get("variants", ["no_red", "no_valid", "no_verifier", "cov_only"])
    if not isinstance(variants, list) or not variants or not all(isinstance(item, str) and item for item in variants):
        raise SystemExit("reward_component_ablation_eval.variants must be a non-empty list of strings")
    return variants


def reward_component_ablation_eval_outputs(config: dict[str, Any]) -> list[str]:
    outputs: list[str] = []
    if config.get("gate", {}).get("enabled", True):
        outputs.extend(manual_gate_outputs(config["gate"]))
    if config.get("budget", {}).get("enabled", True):
        outputs.extend(
            [
                config["budget"]["output"],
                config["budget"].get("output_md", ""),
            ]
        )
    outputs.append(config["predictions"])
    eval_data_dir = config["eval_data_dir"].rstrip("/")
    output_dir = config["output_dir"].rstrip("/")
    for variant in reward_component_ablation_eval_variants(config):
        outputs.extend(
            [
                f"{eval_data_dir}/{variant}/bsc_eval.jsonl",
                f"{output_dir}/{variant}/bsc_join_report.json",
                f"{output_dir}/{variant}/bsc/summary.json",
                f"{output_dir}/{variant}/bsc/per_item.csv",
                f"{output_dir}/{variant}/bsc/per_item.jsonl",
            ]
        )
    return outputs


def dashboard_stage(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name", "dashboard_real"),
        "type": "dashboard",
        "args": {
            "config": config["config"],
            "output_json": config["output_json"],
            "output_md": config.get("output_md"),
        },
    }


def result_card_stage(config: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "config": config["config"],
        "output_json": config["output_json"],
        "output_md": config.get("output_md"),
    }
    if config.get("strict", True):
        args["strict"] = True
    return {
        "name": config.get("name", "result_card_real"),
        "type": "result_card",
        "args": args,
    }


def collect_stage_outputs(stages: list[dict[str, Any]]) -> list[str]:
    outputs = []
    for stage in stages:
        args = stage.get("args", {})
        for key in ["output", "output_json", "output_md"]:
            value = args.get(key)
            if isinstance(value, str):
                outputs.append(value)
    return outputs


def collect_validation_outputs(stages: list[dict[str, Any]]) -> list[str]:
    outputs = []
    for stage in stages:
        if stage.get("type") != "validate_rubrics":
            continue
        output_dir = stage.get("args", {}).get("output_dir")
        if not output_dir:
            continue
        outputs.extend(
            [
                f"{output_dir}/validation_report.json",
                f"{output_dir}/validation_report.md",
                f"{output_dir}/per_record.jsonl",
            ]
        )
    return outputs


def training_command_outputs(config: dict[str, Any]) -> list[str]:
    output_dir = config["output_dir"].rstrip("/")
    outputs = [
        f"{output_dir}/run_sft.sh",
        f"{output_dir}/run_grpo.sh",
        f"{output_dir}/training_done.template.json",
        f"{output_dir}/training_manifest.json",
    ]
    outputs.extend(
        [
            f"{output_dir}/run_grpo_no_red.sh",
            f"{output_dir}/run_grpo_no_valid.sh",
            f"{output_dir}/run_grpo_no_verifier.sh",
            f"{output_dir}/run_grpo_cov_only.sh",
            "outputs/reward_component_training_ablation/training_done.template.json",
        ]
    )
    return outputs


def downstream_rlvr_command_outputs(config: dict[str, Any]) -> list[str]:
    output_dir = config["output_dir"].rstrip("/")
    return [
        f"{output_dir}/downstream_rlvr_manifest.json",
        f"{output_dir}/downstream_rlvr_done.template.json",
        f"{output_dir}/run_healthbench_hard_rlvr.sh",
        f"{output_dir}/run_healthbench_hard_eval.sh",
        f"{output_dir}/run_arenahard_rlvr.sh",
        f"{output_dir}/run_arenahard_eval.sh",
    ]


def manual_gate_outputs(config: dict[str, Any]) -> list[str]:
    outputs = [config["output"]]
    if config.get("output_md"):
        outputs.append(config["output_md"])
    return outputs


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
