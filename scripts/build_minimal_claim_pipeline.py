#!/usr/bin/env python3
"""Build the minimal BlindSpot-RL motivation experiment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result_card_enabled = config.get("result_card", {}).get("enabled", False)
    result_card_output = args.result_card_output
    if result_card_enabled and result_card_output is None:
        result_card_output = args.evidence_output.with_name("minimal_result_card_config.json")
    pipeline = build_pipeline(
        config,
        evidence_config_path=str(args.evidence_output),
        manifest_path=str(args.manifest_output),
        pipeline_config_path=str(args.pipeline_output),
        result_card_config_path=str(result_card_output) if result_card_output else None,
    )
    manifest = build_manifest(config)
    evidence_config = build_evidence_config(config)
    result_card_config = build_result_card_config(config, manifest_path=str(args.manifest_output)) if result_card_enabled else None

    args.pipeline_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_output.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.evidence_output.write_text(json.dumps(evidence_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result_card_enabled and result_card_output:
        result_card_output.parent.mkdir(parents=True, exist_ok=True)
        result_card_output.write_text(json.dumps(result_card_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote minimal claim pipeline to {args.pipeline_output}")
    print(f"Wrote minimal claim manifest to {args.manifest_output}")
    print(f"Wrote minimal claim evidence config to {args.evidence_output}")
    if result_card_enabled and result_card_output:
        print(f"Wrote minimal claim result card config to {result_card_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal BSC motivation experiment pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--result-card-output", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Minimal claim config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Minimal claim config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Minimal claim config must be a JSON object: {path}")
    return data


def normalize_contract_value(value: Any) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return str(value)


def validate_generation_budget_contract(
    api_budget: dict[str, Any],
    generation_input: str,
    budget_providers: Any,
    generation_providers: Any,
) -> None:
    """Keep the minimal-claim budget report compatible with model criteria elicitation."""
    budget_input = api_budget.get("input", generation_input)
    if normalize_contract_value(budget_input) != normalize_contract_value(generation_input):
        raise ValueError("api_budget.input must match the generate_model_rubrics input for minimal-claim runs.")
    if normalize_contract_value(budget_providers) != normalize_contract_value(generation_providers):
        raise ValueError("api_budget.providers must match providers for minimal-claim model criteria elicitation.")
    if api_budget.get("unit_field") is not None:
        raise ValueError("api_budget.unit_field must be omitted for model criteria elicitation.")
    if api_budget.get("unit_multiplier_field") is not None:
        raise ValueError("api_budget.unit_multiplier_field must be omitted for model criteria elicitation.")
    calls_per_record = api_budget.get("calls_per_record_per_provider", 1)
    if calls_per_record != 1:
        raise ValueError("api_budget.calls_per_record_per_provider must be 1 for model criteria elicitation.")


def required_provenance_args(config: dict[str, Any]) -> list[str]:
    configured = config.get("required_provenance_values", config.get("required_provenance", {}))
    values: dict[str, str] = {}
    if isinstance(configured, dict):
        values.update({str(key): str(value) for key, value in configured.items() if value not in (None, "")})
    elif isinstance(configured, list):
        for item in configured:
            text = str(item)
            if "=" in text:
                key, value = text.split("=", 1)
                if key.strip() and value.strip():
                    values[key.strip()] = value.strip()
    return [f"{key}={value}" for key, value in sorted(values.items())]


def build_pipeline(
    config: dict[str, Any],
    evidence_config_path: str,
    manifest_path: str,
    pipeline_config_path: str | None = None,
    result_card_config_path: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    paths = resolve_paths(config)
    method = config.get("method", "base")
    model_filter = config.get("model_filter", method)
    model_verification = config.get("model_verification", {})
    bsc_predictions = paths["verified_model_rubrics"] if model_verification.get("enabled", False) else paths["model_rubrics"]
    stages: list[dict[str, Any]] = []
    generation_input = config.get("queries", config["gold"])
    result_card = config.get("result_card", {})
    local_config = config.get("local_config_init", {})
    pilot = config.get("pilot_sample", {})
    default_min_joined = (
        pilot.get("n")
        if pilot.get("enabled", False)
        else config.get("claim_gates", {}).get("min_n", 100)
    )

    if local_config.get("enabled", False):
        stages.append(
            {
                "name": "init_data_source_local_config",
                "type": "init_data_source_config",
                "args": {
                    "template": local_config["template"],
                    "output": local_config.get("output", "configs/data_sources_real.local.json"),
                    "report_json": local_config.get("report_json", "outputs/data_sources/local_config_init.json"),
                    "report_md": local_config.get("report_md", "outputs/data_sources/local_config_init.md"),
                    "required_dataset": local_config.get("required_datasets", []),
                    "fill_present_sha256": local_config.get("fill_present_sha256", False),
                    "update_existing": local_config.get("update_existing", False),
                    "strict": local_config.get("strict", False),
                },
            }
        )

    for item in result_card.get("data_source_reports", []):
        if item.get("config") and item.get("path"):
            source_args: dict[str, Any] = {
                "config": item["config"],
                "output_json": item["path"],
            }
            if item.get("output_md"):
                source_args["output_md"] = item["output_md"]
            if item.get("required_datasets"):
                source_args["required_dataset"] = item["required_datasets"]
            if item.get("strict", False):
                source_args["strict"] = True
            stages.append(
                {
                    "name": item.get("stage_name", "data_source_report"),
                    "type": "data_source_report",
                    "args": source_args,
                }
            )

    gold_validation = config.get("gold_validation", {})
    if gold_validation.get("enabled", False):
        validation_args: dict[str, Any] = {
            "input": gold_validation.get("input", config["gold"]),
            "target": "gold",
            "output_json": gold_validation.get("output_json", "outputs/data_validation/rubricbench_gold.json"),
            "output_md": gold_validation.get("output_md", "outputs/data_validation/rubricbench_gold.md"),
            "min_records": gold_validation.get("min_records", config.get("claim_gates", {}).get("min_n", 100)),
            "min_rubrics_per_query": gold_validation.get("min_rubrics_per_query", 1),
        }
        for key in ["require_provenance", "allow_missing_data_source", "strict"]:
            if gold_validation.get(key, False):
                validation_args[key] = True
        for item in gold_validation.get("required_data_source", []):
            validation_args.setdefault("required_data_source", []).append(item)
        for item in gold_validation.get("forbidden_data_source", []):
            validation_args.setdefault("forbidden_data_source", []).append(item)
        for item in required_provenance_args(gold_validation):
            validation_args.setdefault("required_provenance", []).append(item)
        stages.append(
            {
                "name": gold_validation.get("stage_name", "validate_rubricbench_gold"),
                "type": "validate_gold",
                "args": validation_args,
            }
        )

    query_validation = config.get("query_validation", {})
    if query_validation.get("enabled", False):
        query_validation_args: dict[str, Any] = {
            "input": query_validation.get("input", config.get("queries", config["gold"])),
            "target": "query_pool",
            "output_json": query_validation.get("output_json", "outputs/data_validation/rubricbench_queries.json"),
            "output_md": query_validation.get("output_md", "outputs/data_validation/rubricbench_queries.md"),
            "min_records": query_validation.get("min_records", config.get("claim_gates", {}).get("min_n", 100)),
            "min_rubrics_per_query": query_validation.get("min_rubrics_per_query", 1),
        }
        for key in ["require_provenance", "allow_missing_data_source", "strict"]:
            if query_validation.get(key, False):
                query_validation_args[key] = True
        for item in query_validation.get("required_data_source", []):
            query_validation_args.setdefault("required_data_source", []).append(item)
        for item in query_validation.get("forbidden_data_source", []):
            query_validation_args.setdefault("forbidden_data_source", []).append(item)
        for item in required_provenance_args(query_validation):
            query_validation_args.setdefault("required_provenance", []).append(item)
        stages.append(
            {
                "name": query_validation.get("stage_name", "validate_rubricbench_queries"),
                "type": "validate_gold",
                "args": query_validation_args,
            }
        )

    bsc_gold_sanity = config.get("bsc_gold_sanity", {})
    if bsc_gold_sanity.get("enabled", False):
        sanity_args: dict[str, Any] = {
            "gold": bsc_gold_sanity.get("gold", config["gold"]),
            "output_dir": bsc_gold_sanity.get("output_dir", paths["bsc_gold_sanity_dir"]),
            "data_source": bsc_gold_sanity.get("data_source", config.get("bsc_data_source", "rubricbench")),
            "min_joined": bsc_gold_sanity.get(
                "min_joined",
                config.get("bsc_join", {}).get("min_joined", config.get("claim_gates", {}).get("min_n", 100)),
            ),
        }
        if bsc_gold_sanity.get("limit") is not None:
            sanity_args["limit"] = bsc_gold_sanity["limit"]
        stages.append({"name": "bsc_gold_sanity", "type": "bsc_gold_sanity", "args": sanity_args})

    if pilot.get("enabled", False):
        generation_input = paths["pilot_queries"]
        stages.append(
            {
                "name": "sample_queries",
                "type": "sample_records",
                "args": {
                    "input": config.get("queries", config["gold"]),
                    "output": paths["pilot_queries"],
                    "n": pilot.get("n", 100),
                    "seed": pilot.get("seed", 13),
                    "stratify_key": pilot.get("stratify_key", "data_source"),
                    "dedupe_key": pilot.get("dedupe_key", "query"),
                    "report": paths["pilot_report"],
                    "report_md": paths["pilot_report_md"],
                },
            }
        )

    preflight = config.get("preflight", {})
    if preflight.get("enabled", False):
        preflight_inputs = preflight.get("inputs", [config["gold"], generation_input])
        preflight_args: dict[str, Any] = {
            "input": preflight_inputs,
            "min_records": preflight.get("min_records", 1),
            "output": paths["preflight_report"],
            "output_md": paths["preflight_report_md"],
        }
        providers = preflight.get("providers", config.get("providers"))
        if providers:
            preflight_args["providers"] = providers if isinstance(providers, list) else [providers]
        if preflight.get("training_config"):
            preflight_args["training_config"] = preflight["training_config"]
        if preflight.get("required_env"):
            preflight_args["required_env"] = preflight["required_env"]
        if preflight.get("required_provider"):
            preflight_args["required_provider"] = preflight["required_provider"]
        if preflight.get("required_provider_in"):
            preflight_args["required_provider_in"] = preflight["required_provider_in"]
        if preflight.get("check_local_provider_health", False):
            preflight_args["check_local_provider_health"] = True
        if preflight.get("strict", False):
            preflight_args["strict"] = True
        stages.append({"name": "preflight", "type": "preflight", "args": preflight_args})

    api_budget = config.get("api_budget", {})
    require_budget_report = None
    require_preflight_report = paths["preflight_report"] if preflight.get("enabled", False) else None
    if api_budget.get("enabled", False):
        budget_providers = api_budget.get("providers", config.get("providers"))
        if not budget_providers:
            raise ValueError("api_budget.enabled requires providers in api_budget or top-level config.")
        validate_generation_budget_contract(api_budget, generation_input, budget_providers, config.get("providers"))
        budget_args: dict[str, Any] = {
            "input": api_budget.get("input", generation_input),
            "providers": budget_providers,
            "resume_output": api_budget.get("resume_output", paths["model_rubrics"]),
            "method_key": api_budget.get("method_key", "method"),
            "calls_per_record_per_provider": api_budget.get("calls_per_record_per_provider", 1),
            "default_qpm": api_budget.get("default_qpm", 60),
            "default_tpm": api_budget.get("default_tpm", 60000),
            "output": paths["api_budget_report"],
            "output_md": paths["api_budget_report_md"],
        }
        for key in [
            "limit",
            "system_prompt",
            "user_template",
            "max_calls",
            "max_total_tokens",
            "max_cost_usd",
            "max_wallclock_minutes_serial",
            "strict",
        ]:
            if key in api_budget and api_budget[key] is not None:
                budget_args[key] = api_budget[key]
        stages.append({"name": "api_budget", "type": "api_budget", "args": budget_args})
        require_budget_report = paths["api_budget_report"]

    api_handoff = config.get("minimal_api_handoff", {})
    if api_handoff.get("enabled", False):
        handoff_args: dict[str, Any] = {
            "pipeline": api_handoff.get(
                "pipeline",
                pipeline_config_path or "configs/pipeline_minimal_claim.generated.json",
            ),
            "preflight": api_handoff.get("preflight", paths["preflight_report"]),
            "api_budget": api_handoff.get("api_budget", paths["api_budget_report"]),
            "bsc_gold_sanity": api_handoff.get("bsc_gold_sanity", paths["bsc_gold_sanity_summary"]),
            "output_json": api_handoff.get("output_json", paths["api_handoff_json"]),
            "output_md": api_handoff.get("output_md", paths["api_handoff_md"]),
            "start_stage": api_handoff.get("start_stage", "generate_model_rubrics"),
            "end_stage": api_handoff.get("end_stage", "result_card"),
        }
        if api_handoff.get("strict", False):
            handoff_args["strict"] = True
        stages.append({"name": "minimal_api_handoff", "type": "minimal_api_handoff", "args": handoff_args})

    if config.get("providers"):
        generate_args: dict[str, Any] = {
            "input": generation_input,
            "providers": config["providers"],
            "output": paths["model_rubrics"],
            "data_source": config.get("generation_data_source", "minimal_claim_generation"),
        }
        for key in ["limit", "sleep", "resume", "system_prompt", "user_template"]:
            if key in config and config[key] is not None:
                generate_args[key] = config[key]
        if require_budget_report:
            generate_args["require_budget_report"] = require_budget_report
        if require_preflight_report:
            generate_args["require_preflight_report"] = require_preflight_report
        stages.append({"name": "generate_model_rubrics", "type": "generate_model_rubrics", "args": generate_args})

    model_output_validation = config.get("model_output_validation", {})
    if model_output_validation.get("enabled", False):
        validation_args: dict[str, Any] = {
            "input": paths["model_rubrics"],
            "output_dir": paths["model_validation_dir"],
            "min_rubrics": model_output_validation.get("min_rubrics", 1),
            "max_rubrics": model_output_validation.get("max_rubrics", 50),
        }
        for key in [
            "redundancy_tau",
            "allow_exact_duplicates",
            "allow_generic_terms",
            "allow_semantic_redundancy",
            "strict",
        ]:
            if key in model_output_validation and model_output_validation[key] is not None:
                validation_args[key] = model_output_validation[key]
        stages.append({"name": "validate_model_rubrics", "type": "validate_rubrics", "args": validation_args})

    if model_verification.get("enabled", False):
        verifier_provider = model_verification.get("provider")
        if not verifier_provider:
            raise ValueError("model_verification.enabled requires model_verification.provider.")
        verifier_budget_args: dict[str, Any] = {
            "input": paths["model_rubrics"],
            "providers": verifier_provider,
            "unit_field": "rubrics",
            "calls_per_record_per_provider": model_verification.get("calls_per_record_per_provider", 1),
            "default_qpm": model_verification.get("default_qpm", 60),
            "default_tpm": model_verification.get("default_tpm", 60000),
            "output": paths["verifier_budget_report"],
            "output_md": paths["verifier_budget_report_md"],
        }
        for key in [
            "limit",
            "max_calls",
            "max_total_tokens",
            "max_cost_usd",
            "max_wallclock_minutes_serial",
            "strict",
        ]:
            if key in model_verification and model_verification[key] is not None:
                verifier_budget_args[key] = model_verification[key]
        stages.append({"name": "verifier_api_budget", "type": "api_budget", "args": verifier_budget_args})
        verifier_args: dict[str, Any] = {
            "input": paths["model_rubrics"],
            "output": paths["verified_model_rubrics"],
            "stats_output": paths["verifier_stats"],
            "mode": model_verification.get("mode", "api"),
            "provider": verifier_provider,
            "require_budget_report": paths["verifier_budget_report"],
            "sleep": model_verification.get("sleep", 0.0),
            "annotate_only": model_verification.get("annotate_only", True),
        }
        if require_preflight_report:
            verifier_args["require_preflight_report"] = require_preflight_report
        if model_verification.get("drop_empty", False):
            verifier_args["drop_empty"] = True
        stages.append({"name": "verify_model_rubrics", "type": "filter_verifier", "args": verifier_args})

    verified_output_validation = config.get("verified_output_validation", {})
    if verified_output_validation.get("enabled", False):
        validation_args = {
            "input": paths["verified_model_rubrics"],
            "output_dir": paths["verified_model_validation_dir"],
            "min_rubrics": verified_output_validation.get("min_rubrics", 1),
            "max_rubrics": verified_output_validation.get("max_rubrics", 50),
        }
        for key in [
            "redundancy_tau",
            "require_valid_flags",
            "allow_exact_duplicates",
            "allow_generic_terms",
            "allow_semantic_redundancy",
            "strict",
        ]:
            if key in verified_output_validation and verified_output_validation[key] is not None:
                validation_args[key] = verified_output_validation[key]
        stages.append({"name": "validate_verified_model_rubrics", "type": "validate_rubrics", "args": validation_args})

    stages.extend(
        [
            {
                "name": "prepare_bsc",
                "type": "prepare_bsc",
                "args": {
                    "gold": config["gold"],
                    "predictions": bsc_predictions,
                    "output": paths["bsc_eval"],
                    "report": paths["join_report"],
                    "model": model_filter,
                    "data_source": config.get("bsc_data_source", "rubricbench"),
                    "min_joined": config.get("bsc_join", {}).get("min_joined", default_min_joined),
                },
            },
            {
                "name": "bsc",
                "type": "bsc",
                "args": {
                    "input": paths["bsc_eval"],
                    "embedding_model": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    "coverage_tau": config.get("coverage_tau", 0.75),
                    "redundancy_tau": config.get("redundancy_tau", 0.85),
                    "output_dir": paths["bsc_dir"],
                },
            },
        ]
    )

    if config.get("sweep", {}).get("enabled", True):
        sweep = config.get("sweep", {})
        stages.append(
            {
                "name": "bsc_sweep",
                "type": "bsc_sweep",
                "args": {
                    "input": paths["bsc_eval"],
                    "embedding_model": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    "coverage_tau": sweep.get("coverage_tau", [0.7, 0.75, 0.8]),
                    "redundancy_tau": sweep.get("redundancy_tau", [0.8, 0.85, 0.9]),
                    "output_dir": paths["sweep_dir"],
                },
            }
        )

    blindspot_map = config.get("blindspot_map", {"enabled": True})
    if blindspot_map.get("enabled", True):
        stages.append(
            {
                "name": "blindspot_map",
                "type": "blindspot_map",
                "args": {
                    "input": paths["bsc_eval"],
                    "embedding_model": blindspot_map.get(
                        "embedding_model",
                        config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    ),
                    "coverage_tau": blindspot_map.get("coverage_tau", config.get("coverage_tau", 0.75)),
                    "model": config.get("model_filter", config.get("method", "base")),
                    "output_dir": paths["blindspot_map_dir"],
                },
            }
        )

    budget_curve = config.get("budget_curve", {"enabled": True})
    if budget_curve.get("enabled", True):
        stages.append(
            {
                "name": "budget_curve",
                "type": "budget_curve",
                "args": {
                    "input": paths["bsc_eval"],
                    "embedding_model": budget_curve.get(
                        "embedding_model",
                        config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    ),
                    "coverage_tau": budget_curve.get("coverage_tau", config.get("coverage_tau", 0.75)),
                    "redundancy_tau": budget_curve.get("redundancy_tau", config.get("redundancy_tau", 0.85)),
                    "k": budget_curve.get("k", [3, 5, 8, 10, 15]),
                    "output_dir": paths["budget_curve_dir"],
                },
            }
        )

    human_audit_pack = config.get("human_audit_pack", {})
    if human_audit_pack.get("enabled", False):
        stages.append(
            {
                "name": "bsc_human_audit_pack",
                "type": "bsc_human_audit_pack",
                "args": {
                    "input": paths["bsc_eval"],
                    "embedding_model": human_audit_pack.get(
                        "embedding_model",
                        config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                    ),
                    "coverage_tau": human_audit_pack.get("coverage_tau", config.get("coverage_tau", 0.75)),
                    "matched": human_audit_pack.get("matched", 25),
                    "unmatched": human_audit_pack.get("unmatched", 25),
                    "seed": human_audit_pack.get("seed", 13),
                    "output_dir": paths["human_audit_pack_dir"],
                },
            }
        )

    bootstrap_ci = config.get("bootstrap_ci", {})
    if bootstrap_ci.get("enabled", False):
        stages.append(
            {
                "name": "bsc_bootstrap_ci",
                "type": "bootstrap_ci",
                "args": {
                    "input": paths["bsc_per_item"],
                    "metric": bootstrap_ci.get(
                        "metrics",
                        ["coverage", "blind", "redundancy", "hallucination", "reward"],
                    ),
                    "n_boot": bootstrap_ci.get("n_boot", 1000),
                    "seed": bootstrap_ci.get("seed", 13),
                    "confidence": bootstrap_ci.get("confidence", 0.95),
                    "output_json": paths["bsc_ci_json"],
                    "output_csv": paths["bsc_ci_csv"],
                    "output_md": paths["bsc_ci_md"],
                },
            }
        )

    summarize_args: dict[str, Any] = {
        "bsc": [f"{method}={paths['bsc_summary']}"],
        "output_csv": paths["main_table_csv"],
        "output_md": paths["main_table_md"],
    }
    if config.get("bootstrap_ci", {}).get("enabled", False):
        summarize_args["bsc_ci"] = [f"{method}={paths['bsc_ci_json']}"]

    stages.extend(
        [
            {
                "name": "summarize",
                "type": "summarize",
                "args": summarize_args,
            },
            {
                "name": "evidence",
                "type": "evidence",
                "args": {
                    "config": evidence_config_path,
                    "output_dir": paths["evidence_dir"],
                },
            },
            {
                "name": "audit",
                "type": "audit",
                "args": {
                    "manifest": manifest_path,
                    "output": paths["audit_report"],
                    "non_strict": True,
                },
            },
            {
                "name": "export",
                "type": "export",
                "args": {
                    "main_table_csv": paths["main_table_csv"],
                    "main_table_md": paths["main_table_md"],
                    "audit_report": paths["audit_report"],
                    "handoff_json": paths["api_handoff_json"],
                    "handoff_md": paths["api_handoff_md"],
                    "evidence_json": paths["evidence_json"],
                    "evidence_csv": paths["evidence_csv"],
                    "evidence_md": paths["evidence_md"],
                    "output_dir": paths["paper_artifacts_dir"],
                },
            },
        ]
    )
    paper_sync = config.get("paper_sync", {})
    if paper_sync.get("enabled", False):
        required_file = paper_sync.get("required_file", ["main_table.tex", "experiment_summary.md"])
        stages.append(
            {
                "name": "sync_paper",
                "type": "sync_paper",
                "args": {
                    "artifacts_dir": paths["paper_artifacts_dir"],
                    "paper_dir": paper_sync.get("paper_dir", "paper"),
                    "required_file": required_file,
                },
            }
        )
        stages.append(
            {
                "name": "paper_asset_index_check",
                "type": "paper_asset_index_check",
                "args": {
                    "asset_index": f"{paper_sync.get('paper_dir', 'paper')}/asset_index.md",
                    "output": paths["paper_asset_index_check_json"],
                    "output_md": paths["paper_asset_index_check_md"],
                },
            }
        )

    readiness = config.get("readiness", {})
    if readiness.get("enabled", False):
        readiness_args: dict[str, Any] = {
            "audit_report": paths["audit_report"],
            "evidence_matrix": paths["evidence_json"],
            "paper_dir": readiness.get("paper_dir", paper_sync.get("paper_dir", "paper")),
            "raw_gate": build_readiness_raw_gate_specs(config),
            "output_json": paths["readiness_report"],
            "output_md": paths["readiness_report_md"],
        }
        if readiness.get("strict", False):
            readiness_args["strict"] = True
        stages.append({"name": "submission_readiness", "type": "submission_readiness", "args": readiness_args})

    result_card = config.get("result_card", {})
    if result_card.get("enabled", False):
        if not result_card_config_path:
            raise ValueError("result_card.enabled requires result_card_config_path.")
        result_card_args: dict[str, Any] = {
            "config": result_card_config_path,
            "output_json": paths["result_card_json"],
            "output_md": paths["result_card_md"],
        }
        if result_card.get("strict", False):
            result_card_args["strict"] = True
        stages.append({"name": "result_card", "type": "result_card", "args": result_card_args})
        if paper_sync.get("enabled", False):
            stages.append(
                {
                    "name": "sync_result_card",
                    "type": "sync_paper",
                    "args": {
                        "artifacts_dir": paths["paper_artifacts_dir"],
                        "paper_dir": paper_sync.get("paper_dir", "paper"),
                        "required_file": required_file,
                    },
                }
            )
            stages.append(
                {
                    "name": "paper_asset_index_check_post_sync",
                    "type": "paper_asset_index_check",
                    "args": {
                        "asset_index": f"{paper_sync.get('paper_dir', 'paper')}/asset_index.md",
                        "output": paths["paper_asset_index_check_json"],
                        "output_md": paths["paper_asset_index_check_md"],
                    },
                }
            )
            stages.append(
                {
                    "name": "paper_asset_index_check_final",
                    "type": "paper_asset_index_check",
                    "args": {
                        "asset_index": f"{paper_sync.get('paper_dir', 'paper')}/asset_index.md",
                        "output": paths["paper_asset_index_check_json"],
                        "output_md": paths["paper_asset_index_check_md"],
                        "strict": True,
                    },
                }
            )
    return {"stages": stages}


def build_readiness_raw_gate_specs(config: dict[str, Any]) -> list[str]:
    paths = resolve_paths(config)
    result_card = config.get("result_card", {})
    specs: list[str] = []
    local_config = config.get("local_config_init", {})
    if local_config.get("enabled", False):
        specs.append(
            "Data Source Local Config|data_source_local_config|"
            + local_config.get("report_json", "outputs/data_sources/local_config_init.json")
        )
    for item in result_card.get("data_source_reports", []):
        if item.get("path"):
            gate_type = "data_source_report"
            required_datasets = item.get("required_datasets", [])
            if required_datasets:
                gate_type = "data_source_report[" + ",".join(required_datasets) + "]"
            specs.append(f"{item.get('name', 'Data Source Report')}|{gate_type}|{item['path']}")
    for item in gold_validation_items(config):
        if item.get("path"):
            specs.append(f"{item.get('name', 'Gold Validation')}|gold_validation|{item['path']}")
    for item in query_validation_items(config):
        if item.get("path"):
            specs.append(f"{item.get('name', 'Query Pool Validation')}|query_validation|{item['path']}")
    if config.get("preflight", {}).get("enabled", False):
        specs.append(f"Minimal Claim Preflight|preflight|{paths['preflight_report']}")
    if config.get("api_budget", {}).get("enabled", False):
        specs.append(f"Minimal Claim API Budget|api_budget|{paths['api_budget_report']}")
    if config.get("minimal_api_handoff", {}).get("enabled", False):
        specs.append(f"Minimal Claim API Handoff|minimal_api_handoff|{paths['api_handoff_json']}")
    if config.get("model_output_validation", {}).get("enabled", False):
        specs.append(f"Minimal Claim Model Evaluation-Criteria Validation|validation|{paths['model_validation_report']}")
    if config.get("model_verification", {}).get("enabled", False):
        specs.append(f"Minimal Claim Verifier API Budget|api_budget|{paths['verifier_budget_report']}")
        specs.append(f"Minimal Claim Verified Model Evaluation-Criteria|generic|{paths['verified_model_rubrics']}")
        specs.append(f"Minimal Claim Verifier Stats|generic|{paths['verifier_stats']}")
    if config.get("verified_output_validation", {}).get("enabled", False):
        specs.append(
            f"Minimal Claim Verified Evaluation-Criteria Validation|validation|{paths['verified_model_validation_report']}"
        )
    if config.get("bsc_gold_sanity", {}).get("enabled", False):
        specs.append(f"BSC Gold-as-Prediction Sanity|bsc_gold_sanity|{paths['bsc_gold_sanity_summary']}")
    specs.extend(config.get("readiness", {}).get("raw_gate", []))
    return specs


def gold_validation_items(config: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    gold_validation = config.get("gold_validation", {})
    if gold_validation.get("enabled", False):
        path = gold_validation.get("output_json", "outputs/data_validation/rubricbench_gold.json")
        items.append(
            {
                "name": gold_validation.get("name", "RubricBench Gold Validation"),
                "path": path,
            }
        )
        seen_paths.add(path)
    for item in config.get("result_card", {}).get("gold_validations", []):
        path = item.get("path")
        if not path or path in seen_paths:
            continue
        items.append({"name": item.get("name", "Gold Validation"), "path": path})
        seen_paths.add(path)
    return items


def query_validation_items(config: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    query_validation = config.get("query_validation", {})
    if query_validation.get("enabled", False):
        path = query_validation.get("output_json", "outputs/data_validation/rubricbench_queries.json")
        items.append(
            {
                "name": query_validation.get("name", "RubricBench Query Pool Validation"),
                "path": path,
            }
        )
        seen_paths.add(path)
    for item in config.get("result_card", {}).get("query_validations", []):
        path = item.get("path")
        if not path or path in seen_paths:
            continue
        items.append({"name": item.get("name", "Query Pool Validation"), "path": path})
        seen_paths.add(path)
    return items


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    paths = resolve_paths(config)
    required_files = [
        paths["model_rubrics"],
        paths["bsc_eval"],
        paths["join_report"],
        paths["bsc_summary"],
        paths["bsc_per_item"],
        paths["main_table_csv"],
        paths["main_table_md"],
        paths["evidence_json"],
        paths["evidence_csv"],
        paths["evidence_md"],
    ]
    if config.get("pilot_sample", {}).get("enabled", False):
        required_files.extend([paths["pilot_queries"], paths["pilot_report"], paths["pilot_report_md"]])
    local_config = config.get("local_config_init", {})
    if local_config.get("enabled", False):
        required_files.extend(
            [
                local_config.get("output", "configs/data_sources_real.local.json"),
                local_config.get("report_json", "outputs/data_sources/local_config_init.json"),
            ]
        )
    for item in config.get("result_card", {}).get("data_source_reports", []):
        if item.get("path"):
            required_files.append(item["path"])
    gold_validation = config.get("gold_validation", {})
    if gold_validation.get("enabled", False):
        required_files.append(gold_validation.get("output_json", "outputs/data_validation/rubricbench_gold.json"))
        if gold_validation.get("output_md"):
            required_files.append(gold_validation["output_md"])
    for item in config.get("result_card", {}).get("gold_validations", []):
        if item.get("path") and item["path"] not in required_files:
            required_files.append(item["path"])
    query_validation = config.get("query_validation", {})
    if query_validation.get("enabled", False):
        required_files.append(query_validation.get("output_json", "outputs/data_validation/rubricbench_queries.json"))
        if query_validation.get("output_md"):
            required_files.append(query_validation["output_md"])
    for item in config.get("result_card", {}).get("query_validations", []):
        if item.get("path") and item["path"] not in required_files:
            required_files.append(item["path"])
    if config.get("bsc_gold_sanity", {}).get("enabled", False):
        required_files.extend(
            [
                paths["bsc_gold_sanity_summary"],
                paths["bsc_gold_sanity_join_report"],
                paths["bsc_gold_sanity_eval"],
                paths["bsc_gold_sanity_predictions"],
                paths["bsc_gold_sanity_per_item"],
            ]
        )
    if config.get("preflight", {}).get("enabled", False):
        required_files.extend([paths["preflight_report"], paths["preflight_report_md"]])
    if config.get("api_budget", {}).get("enabled", False):
        required_files.extend([paths["api_budget_report"], paths["api_budget_report_md"]])
    if config.get("minimal_api_handoff", {}).get("enabled", False):
        required_files.extend([paths["api_handoff_json"], paths["api_handoff_md"]])
    if config.get("model_output_validation", {}).get("enabled", False):
        required_files.extend([paths["model_validation_report"], paths["model_validation_md"]])
    if config.get("model_verification", {}).get("enabled", False):
        required_files.extend(
            [
                paths["verified_model_rubrics"],
                paths["verifier_budget_report"],
                paths["verifier_budget_report_md"],
                paths["verifier_stats"],
            ]
        )
    if config.get("verified_output_validation", {}).get("enabled", False):
        required_files.extend([paths["verified_model_validation_report"], paths["verified_model_validation_md"]])
    if config.get("sweep", {}).get("enabled", True):
        required_files.extend([paths["sweep_csv"], paths["sweep_json"], paths["sweep_md"]])
    if config.get("blindspot_map", {"enabled": True}).get("enabled", True):
        required_files.extend(
            [
                paths["blindspot_map_summary"],
                paths["blindspot_map_category_summary"],
                paths["blindspot_map_jsonl"],
            ]
        )
    if config.get("budget_curve", {"enabled": True}).get("enabled", True):
        required_files.extend([paths["budget_curve_json"], paths["budget_curve_csv"]])
    if config.get("human_audit_pack", {}).get("enabled", False):
        required_files.extend(
            [
                paths["human_audit_pack_summary"],
                paths["human_audit_pack_csv"],
                paths["human_audit_pack_jsonl"],
            ]
        )
    if config.get("bootstrap_ci", {}).get("enabled", False):
        required_files.extend([paths["bsc_ci_json"], paths["bsc_ci_csv"], paths["bsc_ci_md"]])
    if config.get("result_card", {}).get("enabled", False):
        required_files.extend([paths["result_card_json"], paths["result_card_md"]])

    required_summary_keys = [
        "n",
        "coverage_tau",
        "redundancy_tau",
        "mean_coverage",
        "mean_blind",
        "median_blind",
        "queries_coverage_le_0_5",
        "queries_blind_ge_0_5",
        "queries_zero_coverage",
        "mean_redundancy",
        "mean_hallucination",
    ]

    return {
        "required_files": required_files,
        "summaries": [
            {
                "name": "minimal_bsc",
                "path": paths["bsc_summary"],
                "required_keys": required_summary_keys,
            }
        ],
    }


def build_evidence_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = resolve_paths(config)
    min_n = config.get("claim_gates", {}).get("min_n", 100)
    min_blind = config.get("claim_gates", {}).get("min_blind", 0.2)
    bsc_data_source = config.get("bsc_data_source", "rubricbench")
    claims: list[dict[str, Any]] = [
        {
            "id": "C1",
            "section": "Motivation",
            "claim": "A single-model evaluation-criteria policy leaves measurable blind spots against human-gold evaluation dimensions.",
            "artifacts": [
                {"label": "BSC summary", "path": paths["bsc_summary"]},
                {"label": "BSC per-item diagnostics", "path": paths["bsc_per_item"]},
            ],
            "metrics": [
                {
                    "label": "Evaluated gold-query count",
                    "path": paths["bsc_summary"],
                    "metric": "n",
                    "op": ">=",
                    "value": min_n,
                },
                {
                    "label": "Blind spot is non-trivial",
                    "path": paths["bsc_summary"],
                    "metric": "mean_blind",
                    "op": ">=",
                    "value": min_blind,
                },
                {
                    "label": "Median query blind spot is non-trivial",
                    "path": paths["bsc_summary"],
                    "metric": "median_blind",
                    "op": ">=",
                    "value": min_blind,
                },
                {
                    "label": "Most diagnostic queries have at most half coverage",
                    "path": paths["bsc_summary"],
                    "metric": "queries_coverage_le_0_5",
                    "op": ">=",
                    "value": max(1, int(min_n * 0.5)),
                },
                {
                    "label": "BSC coverage threshold matches protocol",
                    "path": paths["bsc_summary"],
                    "metric": "coverage_tau",
                    "op": "==",
                    "value": config.get("coverage_tau", 0.75),
                },
                {
                    "label": "BSC redundancy threshold matches protocol",
                    "path": paths["bsc_summary"],
                    "metric": "redundancy_tau",
                    "op": "==",
                    "value": config.get("redundancy_tau", 0.85),
                },
                {
                    "label": f"Hard-gold source has enough evaluated records ({bsc_data_source})",
                    "path": paths["bsc_summary"],
                    "metric": f"data_source_counts.{bsc_data_source}",
                    "op": ">=",
                    "value": min_n,
                },
            ],
            "values": [
                {
                    "label": "BSC embedding model matches protocol",
                    "path": paths["bsc_summary"],
                    "key": "embedding_model",
                    "op": "==",
                    "value": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                },
            ],
            "notes": "This is the first paper gate: run before investing in SFT/GRPO training.",
        }
    ]
    if config.get("blindspot_map", {"enabled": True}).get("enabled", True):
        claims[0]["artifacts"].extend(
            [
                {"label": "Blind-spot attribution summary", "path": paths["blindspot_map_summary"]},
                {"label": "Blind-spot attribution category table", "path": paths["blindspot_map_category_summary"]},
                {"label": "Blind-spot attribution per-query map", "path": paths["blindspot_map_jsonl"]},
            ]
        )
        claims[0].setdefault("csv_checks", []).append(
            {
                "label": "Blind-spot attribution table has all coarse categories",
                "path": paths["blindspot_map_category_summary"],
                "columns": ["category", "total_gold", "covered_gold", "uncovered_gold", "coverage", "blind_rate"],
                "min_rows": 7,
                "numeric": ["total_gold", "covered_gold", "uncovered_gold", "coverage", "blind_rate"],
            }
        )
        claims[0]["metrics"].extend(
            [
                {
                    "label": "Blind-spot attribution uses diagnostic sample",
                    "path": paths["blindspot_map_summary"],
                    "metric": "n",
                    "op": ">=",
                    "value": min_n,
                },
                {
                    "label": "Blind-spot attribution coverage threshold matches protocol",
                    "path": paths["blindspot_map_summary"],
                    "metric": "coverage_tau",
                    "op": "==",
                    "value": config.get("coverage_tau", 0.75),
                },
                {
                    "label": "Blind-spot attribution covers human-gold dimensions",
                    "path": paths["blindspot_map_summary"],
                    "metric": "total_gold",
                    "op": ">",
                    "value": 0,
                },
            ]
        )

    if config.get("budget_curve", {"enabled": True}).get("enabled", True):
        claims[0]["artifacts"].extend(
            [
                {"label": "Criteria-budget curve JSON", "path": paths["budget_curve_json"]},
                {"label": "Criteria-budget curve CSV", "path": paths["budget_curve_csv"]},
            ]
        )
        claims[0].setdefault("csv_checks", []).append(
            {
                "label": "Criteria-budget curve has evaluated K rows",
                "path": paths["budget_curve_csv"],
                "columns": ["k", "n", "mean_coverage", "mean_blind", "mean_redundancy", "total_gen"],
                "min_rows": 5,
                "numeric": ["k", "n", "mean_coverage", "mean_blind", "mean_redundancy", "total_gen"],
            }
        )
        claims[0]["values"].extend(
            [
                {
                    "label": "Criteria-budget curve embedding model matches protocol",
                    "path": paths["budget_curve_json"],
                    "key": "embedding_model",
                    "op": "==",
                    "value": config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
                },
                {
                    "label": "Criteria-budget curve K grid matches Section 2 diagnostic",
                    "path": paths["budget_curve_json"],
                    "key": "budgets",
                    "op": "==",
                    "value": config.get("budget_curve", {}).get("k", [3, 5, 8, 10, 15]),
                },
            ]
        )
        claims[0]["metrics"].extend(
            [
                {
                    "label": "Criteria-budget curve coverage threshold matches protocol",
                    "path": paths["budget_curve_json"],
                    "metric": "coverage_tau",
                    "op": "==",
                    "value": config.get("coverage_tau", 0.75),
                },
                {
                    "label": "Criteria-budget curve redundancy threshold matches protocol",
                    "path": paths["budget_curve_json"],
                    "metric": "redundancy_tau",
                    "op": "==",
                    "value": config.get("redundancy_tau", 0.85),
                },
            ]
        )
    if config.get("bootstrap_ci", {}).get("enabled", False):
        claims[0]["artifacts"].append({"label": "BSC bootstrap CI", "path": paths["bsc_ci_json"]})
        claims[0]["metrics"].append(
            {
                "label": "Blind spot CI lower bound is non-trivial",
                "path": paths["bsc_ci_json"],
                "metric": "metrics[metric=blind].ci_lower",
                "op": ">=",
                "value": config.get("claim_gates", {}).get("min_blind_ci_lower", min_blind),
            }
        )
    frozen = config.get("claim_gates", {}).get("frozen_diagnostic", {})
    if frozen:
        frozen_summary_metrics = {
            "n": "Frozen diagnostic evaluated gold-query count",
            "mean_coverage": "Frozen diagnostic mean coverage",
            "mean_blind": "Frozen diagnostic mean blind-spot rate",
            "median_blind": "Frozen diagnostic median blind-spot rate",
            "queries_coverage_le_0_5": "Frozen diagnostic queries with at most half coverage",
            "queries_zero_coverage": "Frozen diagnostic zero-coverage queries",
            "mean_redundancy": "Frozen diagnostic mean redundancy",
            "mean_hallucination": "Frozen diagnostic mean hallucination/invalidity",
        }
        for metric, label in frozen_summary_metrics.items():
            if metric in frozen:
                claims[0]["metrics"].append(
                    {
                        "label": label,
                        "path": paths["bsc_summary"],
                        "metric": metric,
                        "op": "==",
                        "value": frozen[metric],
                    }
                )
        if config.get("bootstrap_ci", {}).get("enabled", False):
            frozen_ci_metrics = {
                "blind_ci_lower": ("metrics[metric=blind].ci_lower", "Frozen diagnostic blind CI lower bound"),
                "blind_ci_upper": ("metrics[metric=blind].ci_upper", "Frozen diagnostic blind CI upper bound"),
            }
            for key, (metric, label) in frozen_ci_metrics.items():
                if key in frozen:
                    claims[0]["metrics"].append(
                        {
                            "label": label,
                            "path": paths["bsc_ci_json"],
                            "metric": metric,
                            "op": "==",
                            "value": frozen[key],
                        }
                    )
    if config.get("model_verification", {}).get("enabled", False):
        claims[0]["values"].append(
            {
                "label": "Minimal BSC hallucination source is verifier-backed",
                "path": paths["bsc_summary"],
                "key": "verifier_source",
                "op": "==",
                "value": "valid_flags",
            }
        )
        claims[0]["metrics"].append(
            {
                "label": "Minimal BSC verifier-backed records",
                "path": paths["bsc_summary"],
                "metric": "verifier_source_counts.valid_flags",
                "op": ">=",
                "value": min_n,
            }
        )
    frozen_attribution = config.get("claim_gates", {}).get("frozen_blindspot_attribution", {})
    if frozen_attribution:
        for category, expected in frozen_attribution.get("categories", {}).items():
            for key in ["total_gold", "coverage", "blind_rate"]:
                if key not in expected:
                    continue
                claims[0].setdefault("table_values", []).append(
                    {
                        "label": f"Frozen attribution {category} {key}",
                        "path": paths["blindspot_map_category_summary"],
                        "row_key": "category",
                        "row_value": category,
                        "key": key,
                        "op": "==",
                        "value": expected[key],
                    }
                )
        for metric in ["total_gold", "uncovered_gold", "mean_blind_over_gold"]:
            if metric in frozen_attribution:
                claims[0]["metrics"].append(
                    {
                        "label": f"Frozen attribution {metric}",
                        "path": paths["blindspot_map_summary"],
                        "metric": metric,
                        "op": "==",
                        "value": frozen_attribution[metric],
                    }
                )
    frozen_budget_curve = config.get("claim_gates", {}).get("frozen_budget_curve", {})
    if frozen_budget_curve:
        for row in frozen_budget_curve.get("rows", []):
            k = row["k"]
            for key in ["n", "mean_coverage", "mean_blind", "mean_redundancy", "total_gen"]:
                if key not in row:
                    continue
                claims[0].setdefault("table_values", []).append(
                    {
                        "label": f"Frozen budget curve K={k} {key}",
                        "path": paths["budget_curve_csv"],
                        "row_key": "k",
                        "row_value": k,
                        "key": key,
                        "op": "==",
                        "value": row[key],
                    }
                )
    if config.get("sweep", {}).get("enabled", True):
        c2_claim: dict[str, Any] = {
            "id": "C2",
            "section": "Robustness",
            "claim": "The blind-spot diagnostic can be audited across semantic threshold settings.",
            "artifacts": [
                {"label": "Threshold sweep CSV", "path": paths["sweep_csv"]},
                {"label": "Threshold sweep Markdown", "path": paths["sweep_md"]},
            ],
            "metrics": [
                {
                    "label": "Threshold sweep has evaluated records",
                    "path": paths["sweep_json"],
                    "metric": "0.n",
                    "op": ">",
                    "value": 0,
                },
                {
                    "label": "Loose coverage-threshold blind spot remains non-trivial",
                    "path": paths["sweep_json"],
                    "metric": "0.mean_blind",
                    "op": ">=",
                    "value": min_blind,
                },
                {
                    "label": "Strict coverage-threshold blind spot remains non-trivial",
                    "path": paths["sweep_json"],
                    "metric": "6.mean_blind",
                    "op": ">=",
                    "value": min_blind,
                },
            ],
        }
        human_audit_pack = config.get("human_audit_pack", {})
        if human_audit_pack.get("enabled", False):
            c2_claim["artifacts"].append(
                {"label": "BSC human audit annotation pack", "path": paths["human_audit_pack_csv"]}
            )
            c2_claim["values"] = [
                {
                    "label": "BSC human audit pack is annotation-ready",
                    "path": paths["human_audit_pack_summary"],
                    "key": "status",
                    "op": "==",
                    "value": "annotation_pack_ready",
                }
            ]
            c2_claim["metrics"].extend(
                [
                    {
                        "label": "BSC human audit pack has matched samples",
                        "path": paths["human_audit_pack_summary"],
                        "metric": "sampled_matched",
                        "op": ">=",
                        "value": human_audit_pack.get("matched", 25),
                    },
                    {
                        "label": "BSC human audit pack has unmatched samples",
                        "path": paths["human_audit_pack_summary"],
                        "metric": "sampled_unmatched",
                        "op": ">=",
                        "value": human_audit_pack.get("unmatched", 25),
                    },
                ]
            )
        claims.append(c2_claim)
    return {"claims": claims}


def build_result_card_config(config: dict[str, Any], manifest_path: str) -> dict[str, Any]:
    paths = resolve_paths(config)
    method = config.get("method", "base")
    result_card = config.get("result_card", {})
    raw_audit_gates = []
    local_config = config.get("local_config_init", {})
    if local_config.get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Data Source Local Config",
                "type": "data_source_local_config",
                "path": local_config.get("report_json", "outputs/data_sources/local_config_init.json"),
            }
        )
    if config.get("preflight", {}).get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Minimal Claim Preflight",
                "type": "preflight",
                "path": paths["preflight_report"],
            }
        )
    if config.get("api_budget", {}).get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Minimal Claim API Budget",
                "type": "api_budget",
                "path": paths["api_budget_report"],
            }
        )
    if config.get("minimal_api_handoff", {}).get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Minimal Claim API Handoff",
                "type": "minimal_api_handoff",
                "path": paths["api_handoff_json"],
            }
        )
    if config.get("model_output_validation", {}).get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Minimal Claim Model Evaluation-Criteria Validation",
                "type": "validation",
                "path": paths["model_validation_report"],
            }
        )
    if config.get("model_verification", {}).get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Minimal Claim Verifier API Budget",
                "type": "api_budget",
                "path": paths["verifier_budget_report"],
            }
        )
        raw_audit_gates.append(
            {
                "name": "Minimal Claim Verified Model Evaluation-Criteria",
                "type": "generic",
                "path": paths["verified_model_rubrics"],
            }
        )
        raw_audit_gates.append(
            {
                "name": "Minimal Claim Verifier Stats",
                "type": "generic",
                "path": paths["verifier_stats"],
            }
        )
    if config.get("verified_output_validation", {}).get("enabled", False):
        raw_audit_gates.append(
            {
                "name": "Minimal Claim Verified Evaluation-Criteria Validation",
                "type": "validation",
                "path": paths["verified_model_validation_report"],
            }
        )
    raw_audit_gates.append(
        {
            "name": "Minimal Claim Audit",
            "type": "audit",
            "path": paths["audit_report"],
        }
    )
    return {
        "title": result_card.get("title", "BlindSpot-RL Minimal Claim Result Card"),
        "experiment_id": result_card.get("experiment_id", f"minimal_claim_{method}"),
        "scope": result_card.get(
            "scope",
            "Minimal motivation experiment: single-model BSC against hard-gold evaluation dimensions.",
        ),
        "safe_claim": result_card.get(
            "safe_claim",
            "A single-model evaluation-criteria policy leaves measurable blind spots only if Raw Audit Gates and evidence gates pass.",
        ),
        "deferred_claim": result_card.get(
            "deferred_claim",
            "Keep the blind-spot magnitude claim deferred until hard-gold data, criteria elicitation, BSC diagnostics, and evidence gates pass.",
        ),
        "raw_audit_gates": raw_audit_gates,
        "data_source_reports": result_card.get("data_source_reports", []),
        "gold_validations": gold_validation_items(config),
        "query_validations": query_validation_items(config),
        "manifests": [
            {
                "name": "Minimal Claim Manifest",
                "path": manifest_path,
            }
        ]
        + result_card.get("manifests", []),
        "bsc_summaries": [
            {
                "method": method,
                "path": paths["bsc_summary"],
            }
        ],
        "confidence_intervals": [
            {
                "name": "BSC Bootstrap CI",
                "path": paths["bsc_ci_json"],
            }
        ]
        if config.get("bootstrap_ci", {}).get("enabled", False)
        else result_card.get("confidence_intervals", []),
        "downstream_summaries": result_card.get("downstream_summaries", []),
        "evidence_matrix": paths["evidence_json"],
        "readiness_report": paths["readiness_report"] if config.get("readiness", {}).get("enabled", False) else None,
        "dashboard": result_card.get("dashboard"),
        "require_downstream": result_card.get("require_downstream", False),
        "notes": result_card.get(
            "notes",
            [
                "Generated by build_minimal_claim_pipeline.py.",
                "Do not upgrade paper claims unless this card is safe_to_claim.",
            ],
        ),
    }


def resolve_paths(config: dict[str, Any]) -> dict[str, str]:
    method = config.get("method", "base")
    output_root = config.get("output_root", f"outputs/minimal_claim/{method}")
    processed_root = config.get("processed_root", f"data/processed/minimal_claim/{method}")
    model_rubrics = config.get("model_rubrics", f"{processed_root}/model_rubrics.jsonl")
    verified_model_rubrics = config.get("verified_model_rubrics", f"{processed_root}/model_rubrics_verified.jsonl")
    bsc_dir = f"{output_root}/bsc"
    bsc_gold_sanity_dir = f"{output_root}/bsc_gold_sanity"
    api_handoff_dir = f"{output_root}/handoff"
    sweep_dir = f"{output_root}/bsc_sweep"
    blindspot_map_dir = f"{output_root}/blindspot_map"
    budget_curve_dir = f"{output_root}/budget_curve"
    human_audit_pack_dir = f"{output_root}/bsc_human_audit_pack"
    evidence_dir = f"{output_root}/evidence"
    model_validation_dir = f"{output_root}/validation/model_rubrics"
    verified_model_validation_dir = f"{output_root}/validation/model_rubrics_verified"
    return {
        "model_rubrics": model_rubrics,
        "verified_model_rubrics": verified_model_rubrics,
        "bsc_eval": f"{processed_root}/bsc_eval.jsonl",
        "join_report": f"{output_root}/bsc_join_report.json",
        "bsc_dir": bsc_dir,
        "bsc_summary": f"{bsc_dir}/summary.json",
        "bsc_per_item": f"{bsc_dir}/per_item.csv",
        "bsc_gold_sanity_dir": bsc_gold_sanity_dir,
        "bsc_gold_sanity_summary": f"{bsc_gold_sanity_dir}/summary.json",
        "bsc_gold_sanity_join_report": f"{bsc_gold_sanity_dir}/join_report.json",
        "bsc_gold_sanity_eval": f"{bsc_gold_sanity_dir}/bsc_eval.jsonl",
        "bsc_gold_sanity_predictions": f"{bsc_gold_sanity_dir}/gold_as_prediction.jsonl",
        "bsc_gold_sanity_per_item": f"{bsc_gold_sanity_dir}/per_item.csv",
        "api_handoff_json": f"{api_handoff_dir}/api_handoff.json",
        "api_handoff_md": f"{api_handoff_dir}/api_handoff.md",
        "sweep_dir": sweep_dir,
        "sweep_csv": f"{sweep_dir}/threshold_sweep.csv",
        "sweep_json": f"{sweep_dir}/threshold_sweep.json",
        "sweep_md": f"{sweep_dir}/threshold_sweep.md",
        "blindspot_map_dir": blindspot_map_dir,
        "blindspot_map_summary": f"{blindspot_map_dir}/summary.json",
        "blindspot_map_category_summary": f"{blindspot_map_dir}/category_summary.csv",
        "blindspot_map_jsonl": f"{blindspot_map_dir}/blindspots.jsonl",
        "budget_curve_dir": budget_curve_dir,
        "budget_curve_json": f"{budget_curve_dir}/coverage_by_k.json",
        "budget_curve_csv": f"{budget_curve_dir}/coverage_by_k.csv",
        "human_audit_pack_dir": human_audit_pack_dir,
        "human_audit_pack_summary": f"{human_audit_pack_dir}/summary.json",
        "human_audit_pack_csv": f"{human_audit_pack_dir}/audit_items.csv",
        "human_audit_pack_jsonl": f"{human_audit_pack_dir}/audit_items.jsonl",
        "bsc_ci_json": f"{output_root}/bsc_ci/bootstrap_ci.json",
        "bsc_ci_csv": f"{output_root}/bsc_ci/bootstrap_ci.csv",
        "bsc_ci_md": f"{output_root}/bsc_ci/bootstrap_ci.md",
        "main_table_csv": f"{output_root}/main_table.csv",
        "main_table_md": f"{output_root}/main_table.md",
        "evidence_dir": evidence_dir,
        "evidence_json": f"{evidence_dir}/evidence_matrix.json",
        "evidence_csv": f"{evidence_dir}/evidence_matrix.csv",
        "evidence_md": f"{evidence_dir}/evidence_matrix.md",
        "model_validation_dir": model_validation_dir,
        "model_validation_report": f"{model_validation_dir}/validation_report.json",
        "model_validation_md": f"{model_validation_dir}/validation_report.md",
        "verified_model_validation_dir": verified_model_validation_dir,
        "verified_model_validation_report": f"{verified_model_validation_dir}/validation_report.json",
        "verified_model_validation_md": f"{verified_model_validation_dir}/validation_report.md",
        "audit_report": f"{output_root}/audit_report.json",
        "paper_artifacts_dir": f"{output_root}/paper_artifacts",
        "paper_asset_index_check_json": f"{output_root}/paper_artifacts/paper_asset_index_check.json",
        "paper_asset_index_check_md": f"{output_root}/paper_artifacts/paper_asset_index_check.md",
        "pilot_queries": f"{processed_root}/queries_pilot.jsonl",
        "pilot_report": f"{output_root}/sampling/queries_pilot_report.json",
        "pilot_report_md": f"{output_root}/sampling/queries_pilot_report.md",
        "preflight_report": f"{output_root}/preflight/preflight.json",
        "preflight_report_md": f"{output_root}/preflight/preflight.md",
        "api_budget_report": f"{output_root}/api_budget/model_rubrics_budget.json",
        "api_budget_report_md": f"{output_root}/api_budget/model_rubrics_budget.md",
        "verifier_budget_report": f"{output_root}/api_budget/model_rubrics_verifier_budget.json",
        "verifier_budget_report_md": f"{output_root}/api_budget/model_rubrics_verifier_budget.md",
        "verifier_stats": f"{output_root}/verifier/model_rubrics_stats.jsonl",
        "readiness_report": f"{output_root}/readiness/readiness_report.json",
        "readiness_report_md": f"{output_root}/readiness/readiness_report.md",
        "result_card_json": f"{output_root}/paper_artifacts/result_card.json",
        "result_card_md": f"{output_root}/paper_artifacts/result_card.md",
    }


if __name__ == "__main__":
    main()
