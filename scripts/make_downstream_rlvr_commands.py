#!/usr/bin/env python3
"""Generate downstream policy-RLVR command scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scripts = build_scripts(config)
    for rel_path, content in scripts.items():
        path = args.output_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    manifest = build_manifest(config, args.output_dir, scripts, workspace=Path.cwd())
    done_template_path = args.output_dir / "downstream_rlvr_done.template.json"
    done_template_path.write_text(
        json.dumps(build_done_template(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "downstream_rlvr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote downstream RLVR scripts to {args.output_dir}")
    if args.strict and not manifest["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate downstream policy-RLVR command scripts.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if required downstream RLVR inputs are missing.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Downstream RLVR command config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Downstream RLVR command config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Downstream RLVR command config must be a JSON object: {path}")
    return data


def build_scripts(config: dict[str, Any]) -> dict[str, str]:
    scripts: dict[str, str] = {}
    common = config.get("common", {})
    for benchmark in config.get("benchmarks", []):
        name = benchmark["name"]
        scripts[f"run_{name}_rlvr.sh"] = build_rlvr_script(common, benchmark)
        if benchmark.get("eval_command"):
            scripts[f"run_{name}_eval.sh"] = build_eval_script(common, benchmark)
    return scripts


def build_rlvr_script(common: dict[str, Any], benchmark: dict[str, Any]) -> str:
    command = benchmark.get("command", common.get("command", "python3 -m verl.trainer.main_ppo"))
    config_path = benchmark["config"]
    log_path = benchmark.get("log", f"outputs/logs/{benchmark['name']}_policy_rlvr.log")
    env = dict(common.get("env", {}))
    env.update(benchmark.get("env", {}))
    checkpoint = criteria_policy_checkpoint(benchmark)
    if checkpoint:
        env.setdefault("BSC_POLICY_CRITERIA_POLICY_CHECKPOINT", checkpoint)
    if benchmark.get("rubric_file"):
        env.setdefault("BSC_POLICY_RUBRIC_FILE", benchmark["rubric_file"])
    overrides = list(common.get("overrides", [])) + list(benchmark.get("overrides", []))
    overrides.extend(required_overrides(benchmark))
    joined_overrides = " ".join(str(item) for item in overrides)
    return shell_script(
        log_path=log_path,
        env=env,
        command=f"{command} --config-path {shell_quote(config_path)} {joined_overrides}".rstrip(),
        comments=benchmark.get("notes", []),
    )


def build_eval_script(common: dict[str, Any], benchmark: dict[str, Any]) -> str:
    log_path = benchmark.get("eval_log", f"outputs/logs/{benchmark['name']}_policy_eval.log")
    env = dict(common.get("env", {}))
    env.update(benchmark.get("eval_env", {}))
    return shell_script(
        log_path=log_path,
        env=env,
        command=benchmark["eval_command"],
        comments=benchmark.get("eval_notes", []),
    )


def required_overrides(benchmark: dict[str, Any]) -> list[str]:
    mapping = {
        "train_data": "data.train_files",
        "val_data": "data.val_files",
        "criteria_policy_checkpoint": "reward_model.rubric_generator",
        "rubric_file": "reward_model.rubric_file",
        "output_dir": "trainer.default_local_dir",
    }
    overrides = [f"{target}={benchmark[key]}" for key, target in mapping.items() if benchmark.get(key)]
    if not benchmark.get("criteria_policy_checkpoint") and benchmark.get("rubric_generator"):
        overrides.append(f"reward_model.rubric_generator={benchmark['rubric_generator']}")
    return overrides


def shell_script(log_path: str, env: dict[str, Any], command: str, comments: list[str] | None = None) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mkdir -p $(dirname " + shell_quote(log_path) + ")",
    ]
    lines.extend(shell_exports(env))
    for comment in comments or []:
        lines.append(f"# {comment}")
    lines.append(f"{command} 2>&1 | tee {shell_quote(log_path)}")
    return "\n".join(lines) + "\n"


def build_manifest(
    config: dict[str, Any],
    output_dir: Path,
    scripts: dict[str, str],
    workspace: Path | None = None,
) -> dict[str, Any]:
    benchmarks = []
    blockers = []
    workspace = workspace or Path.cwd()
    for benchmark in config.get("benchmarks", []):
        name = benchmark["name"]
        checks = dependency_checks(benchmark, workspace)
        benchmark_blockers = [
            f"{name}: missing required {item['key']} at {item['path']}"
            for item in checks
            if item["required"] and not item["present"]
        ]
        blockers.extend(benchmark_blockers)
        benchmarks.append(
            {
                "name": name,
                "rlvr_script": str(output_dir / f"run_{name}_rlvr.sh"),
                "eval_script": str(output_dir / f"run_{name}_eval.sh") if benchmark.get("eval_command") else "",
                "config": benchmark["config"],
                "train_data": benchmark.get("train_data", ""),
                "val_data": benchmark.get("val_data", ""),
                "criteria_policy_checkpoint": criteria_policy_checkpoint(benchmark),
                "rubric_file": benchmark.get("rubric_file", ""),
                "reward_function": benchmark.get("reward_function", ""),
                "output_dir": benchmark.get("output_dir", ""),
                "eval_output": benchmark.get("eval_output", ""),
                "checks": checks,
                "blockers": benchmark_blockers,
                "notes": benchmark.get("notes", []),
            }
        )
    return {
        "ok": not blockers,
        "scripts": [str(output_dir / rel_path) for rel_path in scripts],
        "downstream_rlvr_done_template": str(output_dir / "downstream_rlvr_done.template.json"),
        "benchmarks": benchmarks,
        "blockers": blockers,
        "notes": config.get("notes", []),
    }


def build_done_template(config: dict[str, Any]) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    template: dict[str, Any] = {
        "operator": "",
        "date": "",
        "notes": "",
        "benchmarks": benchmarks,
    }
    for benchmark in config.get("benchmarks", []):
        name = benchmark["name"]
        policy_key = f"{name}_policy"
        eval_key = f"{name}_eval"
        template[policy_key] = benchmark.get("output_dir", "")
        template[eval_key] = benchmark.get("eval_output", "")
        benchmarks[name] = {
            "config": benchmark.get("config", ""),
            "train_data": benchmark.get("train_data", ""),
            "val_data": benchmark.get("val_data", ""),
            "criteria_policy_checkpoint": criteria_policy_checkpoint(benchmark),
            "rubric_file": benchmark.get("rubric_file", ""),
            "reward_function": benchmark.get("reward_function", ""),
        }
    return template


def dependency_checks(benchmark: dict[str, Any], workspace: Path) -> list[dict[str, Any]]:
    required_keys = ["config", "train_data", "val_data", "criteria_policy_checkpoint", "rubric_file"]
    checks = []
    for key in required_keys:
        raw_path = criteria_policy_checkpoint(benchmark) if key == "criteria_policy_checkpoint" else benchmark.get(key, "")
        if not raw_path:
            checks.append({"key": key, "path": "", "required": True, "present": False})
            continue
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else workspace / path
        checks.append({"key": key, "path": str(raw_path), "required": True, "present": resolved.exists()})
    return checks


def criteria_policy_checkpoint(benchmark: dict[str, Any]) -> str:
    return str(benchmark.get("criteria_policy_checkpoint") or benchmark.get("rubric_generator") or "")


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
