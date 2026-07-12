#!/usr/bin/env python3
"""Build reproducible data acquisition/normalization pipeline configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.include_dataset:
        config = scope_config(config, args.include_dataset)
    pipeline = build_pipeline(config)
    manifest = build_manifest(config)

    args.pipeline_output.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_output.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote data pipeline to {args.pipeline_output}")
    print(f"Wrote data readiness manifest to {args.manifest_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BlindSpot-RL data pipeline config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument(
        "--include-dataset",
        action="append",
        default=[],
        help="Build a scoped pipeline containing only the named dataset. Repeatable.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Data pipeline config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Data pipeline config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Data pipeline config must be a JSON object: {path}")
    return data


def scope_config(config: dict[str, Any], include_datasets: list[str]) -> dict[str, Any]:
    scoped = copy.deepcopy(config)
    requested = list(dict.fromkeys(include_datasets))
    datasets = scoped.get("datasets", [])
    by_name = {dataset.get("name"): dataset for dataset in datasets}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(f"Data pipeline config is missing requested dataset(s): {', '.join(missing)}")
    scoped["datasets"] = [by_name[name] for name in requested]
    for key in [
        "local_config_init",
        "post_download_local_config_init",
        "pre_normalization_source_report",
        "source_report",
    ]:
        section = scoped.get(key, {})
        if section.get("enabled", False):
            section["required_datasets"] = requested
            apply_scoped_report_suffix(section, key, requested)
    return scoped


def apply_scoped_report_suffix(section: dict[str, Any], section_key: str, requested: list[str]) -> None:
    suffix = "_".join(requested)
    defaults = {
        "local_config_init": {
            "report_json": "outputs/data_sources/local_config_init.json",
            "report_md": "outputs/data_sources/local_config_init.md",
        },
        "post_download_local_config_init": {
            "report_json": "outputs/data_sources/local_config_post_download.json",
            "report_md": "outputs/data_sources/local_config_post_download.md",
        },
        "pre_normalization_source_report": {
            "output_json": "outputs/data_sources/source_report_pre_normalization.json",
            "output_md": "outputs/data_sources/source_report_pre_normalization.md",
        },
        "source_report": {
            "output_json": "outputs/data_sources/source_report.json",
            "output_md": "outputs/data_sources/source_report.md",
        },
    }
    for field, default_path in defaults.get(section_key, {}).items():
        if section.get(field, default_path) == default_path:
            section[field] = suffixed_path(default_path, suffix)


def suffixed_path(path: str, suffix: str) -> str:
    item = Path(path)
    return str(item.with_name(f"{item.stem}_{suffix}{item.suffix}"))


def build_pipeline(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stages: list[dict[str, Any]] = []
    local_config = config.get("local_config_init", {})
    if local_config.get("enabled", False):
        stages.append(init_data_source_stage(local_config, "init_data_source_local_config"))
    stages.extend(download_stages(config))

    post_download_local_config = config.get("post_download_local_config_init", {})
    if post_download_local_config.get("enabled", False):
        stages.append(init_data_source_stage(post_download_local_config, "init_data_source_local_config_post_download"))

    pre_source_report = config.get("pre_normalization_source_report", {})
    if pre_source_report.get("enabled", False):
        stages.append(data_source_report_stage(pre_source_report, "data_source_report_pre_normalization"))

    for dataset in config.get("datasets", []):
        name = dataset["name"]
        source = dataset["source"]
        raw_path = raw_output_path(dataset)
        if source["type"] not in {"hf", "url", "manual"}:
            raise ValueError(f"Unsupported source type for {name}: {source['type']}")

        if dataset.get("profile"):
            stages.append(
                {
                    "name": f"profile_{name}",
                    "type": "profile_data",
                    "args": profile_args(raw_path, dataset["profile"]),
                }
            )

        if dataset.get("schema_contract"):
            stages.append(
                {
                    "name": f"schema_contract_{name}",
                    "type": "schema_contract",
                    "args": schema_contract_args(raw_path, dataset["schema_contract"], dataset_name=name),
                }
            )

        for idx, normalization in enumerate(dataset.get("normalizations", []), start=1):
            stages.append(
                {
                    "name": f"normalize_{name}_{normalization['target']}_{idx}",
                    "type": "normalize",
                    "args": normalize_args(
                        raw_path=raw_path,
                        dataset_name=name,
                        source=source,
                        normalization=normalization,
                    ),
                }
            )
            validation = normalization.get("validation", {})
            if normalization["target"] in {"gold", "query_pool"} and validation.get("enabled", False):
                stages.append(
                    {
                        "name": f"validate_{name}_{normalization['target']}_{idx}",
                        "type": "validate_gold",
                        "args": validate_gold_args(
                            input_path=normalization["output"],
                            validation=validation,
                            dataset_name=name,
                            source=source,
                            normalization=normalization,
                        ),
                    }
                )
            split = normalization.get("split", {})
            if split.get("enabled", False):
                stages.append(
                    {
                        "name": f"split_{name}_{normalization['target']}_{idx}",
                        "type": "split_dataset",
                        "args": split_args(input_path=normalization["output"], split=split),
                    }
                )
    source_report = config.get("source_report", {})
    if source_report.get("enabled", False):
        stages.append(data_source_report_stage(source_report, "data_source_report"))
    return {"stages": stages}


def init_data_source_stage(config: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "init_data_source_config",
        "args": {
            "template": config["template"],
            "output": config.get("output", "configs/data_sources_real.local.json"),
            "report_json": config.get("report_json", "outputs/data_sources/local_config_init.json"),
            "report_md": config.get("report_md", "outputs/data_sources/local_config_init.md"),
            "required_dataset": config.get("required_datasets", []),
            "fill_present_sha256": config.get("fill_present_sha256", False),
            "update_existing": config.get("update_existing", False),
            "strict": config.get("strict", False),
        },
    }


def data_source_report_stage(config: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "data_source_report",
        "args": {
            "config": config.get("config", "configs/data_sources_real.local.json"),
            "output_json": config.get("output_json", "outputs/data_sources/source_report.json"),
            "output_md": config.get("output_md", "outputs/data_sources/source_report.md"),
            "required_dataset": config.get("required_datasets", []),
            "strict": config.get("strict", False),
        },
    }


def download_stages(config: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for dataset in config.get("datasets", []):
        name = dataset["name"]
        source = dataset["source"]
        raw_path = raw_output_path(dataset)
        if source["type"] in {"hf", "url"}:
            stages.append(
                {
                    "name": f"download_{name}",
                    "type": "download",
                    "args": download_args(source, raw_path),
                }
            )
        elif source["type"] == "manual" and source.get("download_enabled", False):
            stages.append(
                {
                    "name": f"download_{name}",
                    "type": "download",
                    "args": manual_download_args(source, raw_path),
                }
            )
        elif source["type"] != "manual":
            raise ValueError(f"Unsupported source type for {name}: {source['type']}")
    return stages


def build_manifest(config: dict[str, Any]) -> dict[str, list[str]]:
    required_files = []
    local_config = config.get("local_config_init", {})
    if local_config.get("enabled", False):
        required_files.append(local_config.get("output", "configs/data_sources_real.local.json"))
        required_files.append(local_config.get("report_json", "outputs/data_sources/local_config_init.json"))
    post_download_local_config = config.get("post_download_local_config_init", {})
    if post_download_local_config.get("enabled", False):
        required_files.append(post_download_local_config.get("output", "configs/data_sources_real.local.json"))
        required_files.append(
            post_download_local_config.get("report_json", "outputs/data_sources/local_config_post_download.json")
        )
    pre_source_report = config.get("pre_normalization_source_report", {})
    if pre_source_report.get("enabled", False):
        required_files.append(
            pre_source_report.get("output_json", "outputs/data_sources/source_report_pre_normalization.json")
        )
    source_report = config.get("source_report", {})
    if source_report.get("enabled", False):
        required_files.append(source_report.get("output_json", "outputs/data_sources/source_report.json"))
    for dataset in config.get("datasets", []):
        required_files.append(raw_output_path(dataset))
        if dataset.get("profile"):
            required_files.append(dataset["profile"]["output"])
        if dataset.get("schema_contract"):
            required_files.append(dataset["schema_contract"]["output_json"])
        for normalization in dataset.get("normalizations", []):
            required_files.append(normalization["output"])
            validation = normalization.get("validation", {})
            if normalization["target"] in {"gold", "query_pool"} and validation.get("enabled", False):
                output_json, output_md = validation_output_paths(dataset["name"], validation, normalization)
                required_files.append(output_json)
                required_files.append(output_md)
            split = normalization.get("split", {})
            if split.get("enabled", False):
                manifest = split.get("manifest")
                if manifest:
                    required_files.append(manifest)
                required_files.extend(split_output_paths(split))
    return {"required_files": required_files, "summaries": []}


def raw_output_path(dataset: dict[str, Any]) -> str:
    source = dataset["source"]
    if source["type"] == "manual":
        return source["raw_path"]
    output = source.get("output")
    if not output:
        raise ValueError(f"HF dataset {dataset['name']} must define source.output")
    return output


def download_args(source: dict[str, Any], output: str) -> dict[str, Any]:
    args: dict[str, Any] = {"output": output}
    for key in ["preset", "hf_dataset", "url", "name", "split", "limit", "streaming"]:
        if key in source and source[key] is not None:
            args[key] = source[key]
    return args


def manual_download_args(source: dict[str, Any], output: str) -> dict[str, Any]:
    url = source.get("download_url") or source.get("official_url")
    if not url:
        raise ValueError("manual source download_enabled requires download_url or official_url")
    args: dict[str, Any] = {"url": url, "output": output}
    for key in ["limit"]:
        if key in source and source[key] is not None:
            args[key] = source[key]
    return args


def normalize_args(
    raw_path: str,
    dataset_name: str,
    source: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "input": raw_path,
        "output": normalization["output"],
        "target": normalization["target"],
        "data_source": normalization.get("data_source", dataset_name),
    }
    if source.get("official_url") and "source_url" not in normalization:
        args["source_url"] = source["official_url"]
    if source.get("paper_url") and "paper_url" not in normalization:
        args["paper_url"] = source["paper_url"]
    for key in [
        "query_key",
        "gold_key",
        "chosen_key",
        "rejected_key",
        "candidates_key",
        "label_key",
        "provenance_key",
        "provenance",
        "source_url",
        "paper_url",
        "dataset_version",
        "license",
        "split",
        "limit",
        "dedupe_query",
    ]:
        if key == "split" and isinstance(normalization.get(key), dict):
            continue
        if key in normalization and normalization[key] is not None:
            args[key] = normalization[key]
    return args


def split_args(input_path: str, split: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "input": input_path,
        "split": split["splits"],
    }
    for key in [
        "output_dir",
        "output_prefix",
        "group_key",
        "stratify_key",
        "gold_type",
        "main_eval_split",
        "seed",
        "manifest",
    ]:
        if key in split and split[key] is not None:
            args[key] = split[key]
    return args


def split_output_paths(split: dict[str, Any]) -> list[str]:
    configured = split.get("outputs")
    if configured:
        return list(configured)
    output_dir = split.get("output_dir")
    if not output_dir:
        return []
    output_prefix = split.get("output_prefix", "")
    outputs: list[str] = []
    for spec in split.get("splits", []):
        name = str(spec).split(":", 1)[0].strip()
        if name:
            outputs.append(str(Path(output_dir) / f"{output_prefix}{name}.jsonl"))
    return outputs


def profile_args(raw_path: str, profile: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"input": raw_path, "output": profile["output"]}
    if profile.get("limit") is not None:
        args["limit"] = profile["limit"]
    return args


def schema_contract_args(raw_path: str, contract: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "input": raw_path,
        "target": contract.get("target", ["preference", "multicandidate"]),
        "data_source": contract.get("data_source", dataset_name),
        "min_records": contract.get("min_records", 1),
        "output_json": contract["output_json"],
    }
    if contract.get("output_md"):
        args["output_md"] = contract["output_md"]
    for key in ["query_key", "chosen_key", "rejected_key", "candidates_key", "label_key", "limit", "strict"]:
        if key in contract and contract[key] is not None:
            args[key] = contract[key]
    return args


def validate_gold_args(
    input_path: str,
    validation: dict[str, Any],
    dataset_name: str,
    source: dict[str, Any] | None = None,
    normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = source or {}
    normalization = normalization or {}
    target = validation.get("target", "query_pool" if normalization.get("target") == "query_pool" else "gold")
    output_json, output_md = validation_output_paths(dataset_name, validation, normalization)
    args: dict[str, Any] = {
        "input": input_path,
        "target": target,
        "output_json": output_json,
        "output_md": output_md,
        "min_records": validation.get("min_records", 1),
        "min_rubrics_per_query": validation.get("min_rubrics_per_query", 1),
    }
    for key in ["require_provenance", "allow_missing_data_source", "strict"]:
        if validation.get(key, False):
            args[key] = True
    for item in validation.get("required_data_source", []):
        args.setdefault("required_data_source", []).append(item)
    for item in validation.get("forbidden_data_source", []):
        args.setdefault("forbidden_data_source", []).append(item)
    for item in required_provenance_values(validation, source, normalization):
        args.setdefault("required_provenance", []).append(item)
    return args


def validation_output_paths(
    dataset_name: str,
    validation: dict[str, Any],
    normalization: dict[str, Any],
) -> tuple[str, str]:
    target = validation.get("target", "query_pool" if normalization.get("target") == "query_pool" else "gold")
    output_stem = "queries" if target == "query_pool" else "gold"
    return (
        validation.get("output_json", f"outputs/data_validation/{dataset_name}_{output_stem}.json"),
        validation.get("output_md", f"outputs/data_validation/{dataset_name}_{output_stem}.md"),
    )


def required_provenance_values(
    validation: dict[str, Any],
    source: dict[str, Any],
    normalization: dict[str, Any],
) -> list[str]:
    values: dict[str, str] = {}
    configured = validation.get("required_provenance_values", validation.get("required_provenance", {}))
    if isinstance(configured, dict):
        values.update({str(key): str(value) for key, value in configured.items() if value not in (None, "")})
    elif isinstance(configured, list):
        for item in configured:
            if isinstance(item, str):
                if "=" not in item:
                    raise ValueError(f"required_provenance item must use KEY=VALUE format: {item!r}")
                key, value = item.split("=", 1)
                if key.strip() and value.strip():
                    values[key.strip()] = value.strip()
            elif isinstance(item, dict):
                key = item.get("key")
                value = item.get("value")
                if key and value not in (None, ""):
                    values[str(key)] = str(value)
            else:
                raise ValueError(f"Unsupported required_provenance item: {item!r}")
    elif configured:
        raise ValueError("required_provenance must be a dict, list, or empty value.")

    if normalization.get("paper_url"):
        values.setdefault("paper_url", str(normalization["paper_url"]))
    if source.get("paper_url"):
        values.setdefault("paper_url", str(source["paper_url"]))
    if source.get("require_official_url") and source.get("official_url"):
        values.setdefault("source_url", str(source["official_url"]))
    return [f"{key}={value}" for key, value in sorted(values.items())]


if __name__ == "__main__":
    main()
