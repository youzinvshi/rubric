#!/usr/bin/env python3
"""Generate executable SFT and GRPO training command scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_script = build_sft_script(config)
    grpo_script = build_grpo_script(config)
    sft_path = args.output_dir / "run_sft.sh"
    grpo_path = args.output_dir / "run_grpo.sh"
    sft_path.write_text(sft_script, encoding="utf-8")
    grpo_path.write_text(grpo_script, encoding="utf-8")
    sft_path.chmod(0o755)
    grpo_path.chmod(0o755)

    done_template_path = args.output_dir / "training_done.template.json"
    done_template_path.write_text(
        json.dumps(build_training_done_template(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ablation_outputs = write_reward_component_ablation_outputs(config, args.output_dir)
    manifest = build_manifest(config, sft_path, grpo_path, done_template_path, ablation_outputs=ablation_outputs)
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote training scripts to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BlindSpot-RL training command scripts.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Training command config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Training command config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Training command config must be a JSON object: {path}")
    return data


def build_sft_script(config: dict[str, Any]) -> str:
    sft = config["sft"]
    command = sft.get("command", "llamafactory-cli train")
    yaml_path = sft["config"]
    log_path = sft.get("log", "outputs/logs/sft.log")
    exports = shell_exports(sft.get("env", {}))
    dataset_note = sft.get("dataset_info_note", "")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mkdir -p $(dirname " + shell_quote(log_path) + ")",
        *exports,
    ]
    if dataset_note:
        lines.append(f"# {dataset_note}")
    lines.append(f"{command} {shell_quote(yaml_path)} 2>&1 | tee {shell_quote(log_path)}")
    return "\n".join(lines) + "\n"


def build_grpo_script(config: dict[str, Any]) -> str:
    return build_grpo_script_from_section(config["grpo"])


def build_grpo_script_from_section(grpo: dict[str, Any]) -> str:
    command = grpo.get("command", "python3 -m verl.trainer.main_ppo")
    yaml_path = grpo["config"]
    log_path = grpo.get("log", "outputs/logs/grpo.log")
    exports = shell_exports(grpo.get("env", {}))
    overrides = " ".join(str(item) for item in grpo.get("overrides", []))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mkdir -p $(dirname " + shell_quote(log_path) + ")",
        *exports,
        f"{command} --config-path {shell_quote(yaml_path)} {overrides} 2>&1 | tee {shell_quote(log_path)}".rstrip(),
    ]
    return "\n".join(lines) + "\n"


def build_manifest(
    config: dict[str, Any],
    sft_path: Path,
    grpo_path: Path,
    done_template_path: Path,
    *,
    ablation_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    done_json_path = done_template_path.with_name("training_done.json")
    manifest = {
        "sft_script": str(sft_path),
        "grpo_script": str(grpo_path),
        "training_done_template": str(done_template_path),
        "expected_training_done": str(done_json_path),
        "training_done_sha256_command": (
            "python3 scripts/fill_training_done_sha256.py "
            f"--input {shell_quote(str(done_json_path))} "
            f"--report {shell_quote(str(done_json_path.with_suffix('.sha256_fill_report.json')))}"
        ),
        "sft_config": config["sft"]["config"],
        "grpo_config": config["grpo"]["config"],
        "sft_output_dir": config["sft"].get("output_dir", ""),
        "grpo_output_dir": config["grpo"].get("output_dir", ""),
        "sft_data": config["sft"].get("sft_data", ""),
        "rl_data": config["grpo"].get("rl_data", ""),
        "rl_data_report": config["grpo"].get("rl_data_report", ""),
        "reward_function": config["grpo"].get("reward_function", ""),
        "notes": config.get("notes", []),
    }
    if ablation_outputs:
        manifest["reward_component_ablation"] = ablation_outputs
    return manifest


def build_training_done_template(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "sft_checkpoint": config["sft"].get("output_dir", ""),
        "rl_checkpoint": config["grpo"].get("output_dir", ""),
        "served_methods": ["base", "sft_only", "sft_rl"],
        "served_generators": ["base", "sft_only", "sft_rl"],
        "serving": {
            "base": "",
            "sft_only": "",
            "sft_rl": "",
        },
        "sft_config": config["sft"]["config"],
        "grpo_config": config["grpo"]["config"],
        "sft_data": config["sft"].get("sft_data", ""),
        "rl_data": config["grpo"].get("rl_data", ""),
        "rl_data_report": config["grpo"].get("rl_data_report", ""),
        "reward_function": config["grpo"].get("reward_function", ""),
        "sft_config_sha256": "",
        "grpo_config_sha256": "",
        "sft_data_sha256": "",
        "rl_data_sha256": "",
        "rl_data_report_sha256": "",
        "reward_function_sha256": "",
        "operator": "",
        "date": "",
        "notes": "",
    }


def reward_component_ablation_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("reward_component_ablation", {}).get("enabled", False))


def reward_component_ablation_variants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ablation = config.get("reward_component_ablation", {})
    variants = ablation.get("variants", {})
    if not isinstance(variants, dict):
        raise SystemExit("reward_component_ablation.variants must be a JSON object")
    return variants


def write_reward_component_ablation_outputs(config: dict[str, Any], command_output_dir: Path) -> dict[str, Any] | None:
    if not reward_component_ablation_enabled(config):
        return None

    ablation = config["reward_component_ablation"]
    variants = reward_component_ablation_variants(config)
    if not variants:
        raise SystemExit("reward_component_ablation.enabled requires at least one variant")

    scripts: dict[str, str] = {}
    for variant_name, variant in variants.items():
        variant_grpo = build_grpo_variant_section(config, variant_name, variant)
        script_path = command_output_dir / f"run_grpo_{variant_name}.sh"
        script_path.write_text(build_grpo_script_from_section(variant_grpo), encoding="utf-8")
        script_path.chmod(0o755)
        scripts[variant_name] = str(script_path)

    ablation_output_dir = Path(ablation.get("output_dir", "outputs/reward_component_training_ablation"))
    ablation_output_dir.mkdir(parents=True, exist_ok=True)
    done_template_path = ablation_output_dir / "training_done.template.json"
    done_json_path = ablation_output_dir / "training_done.json"
    done_template_path.write_text(
        json.dumps(build_reward_component_ablation_done_template(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "scripts": scripts,
        "training_done_template": str(done_template_path),
        "expected_training_done": str(done_json_path),
        "sha256_fill_command": (
            "python3 scripts/fill_training_done_sha256.py "
            f"--input {shell_quote(str(done_json_path))} "
            f"--report {shell_quote(str(done_json_path.with_suffix('.sha256_fill_report.json')))}"
        ),
        "output_dir": str(ablation_output_dir),
        "reward_variants": ["full", *variants.keys()],
    }


def build_grpo_variant_section(config: dict[str, Any], variant_name: str, variant: Any) -> dict[str, Any]:
    if not isinstance(variant, dict):
        raise SystemExit(f"reward_component_ablation variant must be an object: {variant_name}")
    base = dict(config["grpo"])
    base["env"] = {**config["grpo"].get("env", {}), **variant.get("env", {})}
    base["overrides"] = [*config["grpo"].get("overrides", []), *variant.get("overrides", [])]
    if "log" in variant:
        base["log"] = variant["log"]
    if "output_dir" in variant:
        base["output_dir"] = variant["output_dir"]
    return base


def build_reward_component_ablation_done_template(config: dict[str, Any]) -> dict[str, Any]:
    variants = reward_component_ablation_variants(config)
    variant_entries: dict[str, Any] = {
        "full": {
            "checkpoint": config["grpo"].get("output_dir", ""),
            "serving": "",
            "grpo_config": config["grpo"]["config"],
            "rl_data": config["grpo"].get("rl_data", ""),
            "rl_data_report": config["grpo"].get("rl_data_report", ""),
            "reward_function": config["grpo"].get("reward_function", ""),
            "grpo_config_sha256": "",
            "rl_data_sha256": "",
            "rl_data_report_sha256": "",
            "reward_function_sha256": "",
            "env": config["grpo"].get("env", {}),
        }
    }
    for variant_name, variant in variants.items():
        variant_grpo = build_grpo_variant_section(config, variant_name, variant)
        variant_entries[variant_name] = {
            "checkpoint": variant_grpo.get("output_dir", ""),
            "serving": "",
            "grpo_config": variant_grpo["config"],
            "rl_data": variant_grpo.get("rl_data", ""),
            "rl_data_report": variant_grpo.get("rl_data_report", ""),
            "reward_function": variant_grpo.get("reward_function", ""),
            "grpo_config_sha256": "",
            "rl_data_sha256": "",
            "rl_data_report_sha256": "",
            "reward_function_sha256": "",
            "env": variant_grpo.get("env", {}),
            "operator": "",
            "date": "",
            "notes": "",
        }
    return {
        "reward_variants": list(variant_entries.keys()),
        "variants": variant_entries,
        "sft_checkpoint": config["sft"].get("output_dir", ""),
        "sft_data": config["sft"].get("sft_data", ""),
        "sft_data_sha256": "",
        "rl_data": config["grpo"].get("rl_data", ""),
        "rl_data_report": config["grpo"].get("rl_data_report", ""),
        "reward_function": config["grpo"].get("reward_function", ""),
        "rl_data_sha256": "",
        "rl_data_report_sha256": "",
        "reward_function_sha256": "",
        "operator": "",
        "date": "",
        "notes": "Fill this file after running the full, no_red, no_valid, no_verifier, and cov_only GRPO variants.",
    }


def shell_exports(env: dict[str, Any]) -> list[str]:
    return [f"export {key}={shell_quote(str(value))}" for key, value in env.items()]


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=+-")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
