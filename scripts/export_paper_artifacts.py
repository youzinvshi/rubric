#!/usr/bin/env python3
"""Export paper-ready tables and a compact experiment artifact bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from scripts.check_paper_asset_index import semantic_space_contract_blockers
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution
    from check_paper_asset_index import semantic_space_contract_blockers


MANAGED_DOC_FILES = [
    "api_handoff.json",
    "api_handoff.md",
    "audit_report.json",
    "evidence_matrix.csv",
    "evidence_matrix.json",
    "evidence_matrix.md",
    "readiness_report.json",
    "readiness_report.md",
    "rebuttal_pack_manifest.json",
    "rebuttal_pack.json",
    "rebuttal_pack.md",
    "result_card.md",
    "result_card.json",
    "submission_gap_report.json",
    "submission_gap_report.md",
]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_managed_generated_files(args.output_dir)
    downstream_table_specs = parse_labeled_path_specs(args.downstream_table_csv)
    transition_summary_json = args.transition_summary_json or args.repair_summary_json

    copied = []
    blockers = []
    warnings = []
    generated = []
    for path in [
        args.main_table_csv,
        args.main_table_md,
        *(path for _, path in downstream_table_specs),
        args.ablation_csv,
        args.teacher_union_csv,
        args.verifier_filter_csv,
        *(transition_summary_json or []),
        args.audit_report,
        args.handoff_json,
        args.handoff_md,
        args.evidence_json,
        args.evidence_csv,
        args.evidence_md,
        args.result_card_md,
        args.result_card_json,
    ]:
        if path and path.exists():
            target = args.output_dir / path.name
            shutil.copyfile(path, target)
            copied.append(target.name)

    main_rows, main_blockers, main_warnings = read_csv_checked(args.main_table_csv, "main table", required=True)
    blockers.extend(main_blockers)
    warnings.extend(main_warnings)
    main_downstream_blockers = sanitize_main_downstream_metrics(main_rows)
    blockers.extend(main_downstream_blockers)
    downstream_rows, downstream_blockers, downstream_warnings = read_downstream_tables(downstream_table_specs)
    blockers.extend(downstream_blockers)
    warnings.extend(downstream_warnings)
    ablation_rows, ablation_blockers, ablation_warnings = read_csv_checked(
        args.ablation_csv,
        "ablation table",
        required=False,
    )
    blockers.extend(ablation_blockers)
    warnings.extend(ablation_warnings)
    teacher_union_rows, teacher_blockers, teacher_warnings = read_csv_checked(
        args.teacher_union_csv,
        "teacher-union ablation table",
        required=False,
    )
    blockers.extend(teacher_blockers)
    warnings.extend(teacher_warnings)
    verifier_filter_rows, verifier_blockers, verifier_warnings = read_csv_checked(
        args.verifier_filter_csv,
        "verifier-filter ablation table",
        required=False,
    )
    blockers.extend(verifier_blockers)
    warnings.extend(verifier_warnings)
    repair_rows, repair_blockers, repair_warnings = read_transition_summaries(transition_summary_json or [])
    blockers.extend(repair_blockers)
    warnings.extend(repair_warnings)
    semantic_files, semantic_blockers, semantic_warnings = collect_semantic_space_artifacts(args.semantic_space_dir)
    blockers.extend(semantic_blockers)
    warnings.extend(semantic_warnings)
    audit = read_json(args.audit_report) if args.audit_report and args.audit_report.exists() else {}
    if is_load_error(audit):
        blockers.append(f"audit report is not readable: {audit['_load_error']}")

    if main_rows:
        main_rows_for_latex = add_display_columns(
            main_rows,
            {
                "cov": "cov_ci",
                "blind": "blind_ci",
                "red": "red_ci",
                "hall": "hall_ci",
                "accuracy": "accuracy_ci",
            },
        )
        main_table_tex = args.output_dir / "main_table.tex"
        main_table_tex.write_text(
            latex_table(
                rows=main_rows_for_latex,
                columns=[
                    ("method", "Method"),
                    ("cov_display", "Cov$\\uparrow$"),
                    ("blind_display", "Blind$\\downarrow$"),
                    ("red_display", "Red$\\downarrow$"),
                    ("hall_display", "Hall$\\downarrow$"),
                    ("accuracy_display", "Acc$\\uparrow$"),
                ],
                caption="Evidence-gated BlindSpot-RL main matrix. Trained and downstream claims are reportable only after their evidence gates pass.",
                label="tab:blindspot_main",
            ),
            encoding="utf-8",
        )
        generated.append(main_table_tex.name)
    rl_stage_rows, rl_stage_warnings = build_rl_stage_ablation_rows(main_rows)
    warnings.extend(rl_stage_warnings)
    if rl_stage_rows:
        rl_stage_table_tex = args.output_dir / "rl_stage_ablation_table.tex"
        rl_stage_table_tex.write_text(
            latex_table(
                rows=rl_stage_rows,
                columns=[
                    ("comparison", "Comparison"),
                    ("baseline_cov", "SFT Cov"),
                    ("candidate_cov", "RL Cov"),
                    ("delta_cov", "$\\Delta$Cov"),
                    ("baseline_cov_per_gen", "SFT Cov/Gen"),
                    ("candidate_cov_per_gen", "RL Cov/Gen"),
                    ("delta_cov_per_gen", "$\\Delta$Cov/Gen"),
                    ("delta_red", "$\\Delta$Red"),
                    ("delta_hall", "$\\Delta$Hall"),
                    ("delta_accuracy", "$\\Delta$Acc"),
                ],
                caption=(
                    "Evidence-gated SFT-only versus SFT+GRPO stage comparison. "
                    "RL-stage support is reportable only after C14 and the shared "
                    "hard-gold protocol gates pass."
                ),
                label="tab:rl_stage_ablation",
            ),
            encoding="utf-8",
        )
        generated.append(rl_stage_table_tex.name)
    if downstream_rows:
        downstream_rows_for_latex = add_display_columns(
            downstream_rows,
            {
                "accuracy": "accuracy_ci",
                "tie_rate": "tie_rate_ci",
                "mean_margin": "mean_margin_ci",
            },
        )
        downstream_table_tex = args.output_dir / "downstream_utility_table.tex"
        downstream_table_tex.write_text(
            latex_table(
                rows=downstream_rows_for_latex,
                columns=[
                    ("benchmark", "Benchmark"),
                    ("method", "Method"),
                    ("downstream_n", "N"),
                    ("accuracy_display", "Acc$\\uparrow$"),
                    ("tie_rate_display", "Tie$\\downarrow$"),
                    ("mean_margin_display", "Margin$\\uparrow$"),
                ],
                caption=(
                    "Evidence-gated downstream judge-utility rows on held-out "
                    "benchmarks; rows are included only when paper-eligibility "
                    "metadata passes."
                ),
                label="tab:downstream_utility",
            ),
            encoding="utf-8",
        )
        generated.append(downstream_table_tex.name)
    if ablation_rows:
        ablation_table_tex = args.output_dir / "ablation_table.tex"
        ablation_table_tex.write_text(
            latex_table(
                rows=ablation_rows,
                columns=[
                    ("variant", "Variant"),
                    ("mean_coverage", "Cov"),
                    ("mean_blind", "Blind"),
                    ("mean_redundancy", "Red"),
                    ("mean_hallucination", "Hall"),
                    ("mean_reward", "Reward"),
                ],
                caption=(
                    "Evidence-gated reward-component ablations. Attribution to "
                    "reward terms is reportable only after C7 passes with matched "
                    "hard-gold evaluation."
                ),
                label="tab:blindspot_ablation",
            ),
            encoding="utf-8",
        )
        generated.append(ablation_table_tex.name)
    if teacher_union_rows:
        teacher_union_table_tex = args.output_dir / "teacher_union_ablation_table.tex"
        teacher_union_table_tex.write_text(
            latex_table(
                rows=teacher_union_rows,
                columns=[
                    ("variant", "Variant"),
                    ("mean_coverage", "Cov"),
                    ("mean_blind", "Blind"),
                    ("mean_redundancy", "Red"),
                    ("mean_reward", "Reward"),
                    ("mean_n_gen", "N Gen"),
                    ("coverage_gain_vs_best_single", "$\\Delta$Cov"),
                ],
                caption=(
                    "Evidence-gated single-teacher versus multi-teacher union "
                    "ablation for proxy-gold construction."
                ),
                label="tab:teacher_union_ablation",
            ),
            encoding="utf-8",
        )
        generated.append(teacher_union_table_tex.name)
    if verifier_filter_rows:
        verifier_filter_table_tex = args.output_dir / "verifier_filter_ablation_table.tex"
        verifier_filter_table_tex.write_text(
            latex_table(
                rows=verifier_filter_rows,
                columns=[
                    ("variant", "Variant"),
                    ("mean_coverage", "Cov"),
                    ("mean_blind", "Blind"),
                    ("mean_redundancy", "Red"),
                    ("mean_hallucination", "Hall"),
                    ("mean_reward", "Reward"),
                    ("mean_n_gen", "N Gen"),
                    ("coverage_delta_vs_no_verifier", "$\\Delta$Cov"),
                    ("hallucination_delta_vs_no_verifier", "$\\Delta$Hall"),
                ],
                caption=(
                    "Evidence-gated verifier-filtering ablation for proxy-gold "
                    "construction; proxy-quality claims require C7 support."
                ),
                label="tab:verifier_filter_ablation",
            ),
            encoding="utf-8",
        )
        generated.append(verifier_filter_table_tex.name)
    if repair_rows:
        transition_table_tex = args.output_dir / "dimension_transition_table.tex"
        transition_table_tex.write_text(
            latex_table(
                rows=repair_rows,
                columns=[
                    ("comparison", "Comparison"),
                    ("baseline_coverage", "Base Cov"),
                    ("candidate_coverage", "Cand. Cov"),
                    ("recovered_dimension_rate", "Recovered$\\uparrow$"),
                    ("loss_rate", "Loss$\\downarrow$"),
                    ("net_transition_rate", "Net Transition"),
                ],
                caption=(
                    "Evidence-gated gold-dimension transition audit over recovered "
                    "and lost human-gold dimensions; dimension-level recovery "
                    "wording requires C12 support."
                ),
                label="tab:dimension_transition",
            ),
            encoding="utf-8",
        )
        write_csv(args.output_dir / "dimension_transition_table.csv", repair_rows)
        write_markdown_table(args.output_dir / "dimension_transition_table.md", repair_rows)
        generated.extend([transition_table_tex.name, "dimension_transition_table.csv", "dimension_transition_table.md"])
    for source in semantic_files:
        target = args.output_dir / source.name
        shutil.copyfile(source, target)
        generated.append(target.name)

    generated.append("experiment_summary.md")
    summary_md = build_summary_markdown(
        main_rows=main_rows,
        downstream_rows=downstream_rows,
        rl_stage_rows=rl_stage_rows,
        ablation_rows=ablation_rows,
        teacher_union_rows=teacher_union_rows,
        verifier_filter_rows=verifier_filter_rows,
        audit=audit,
        copied=copied,
        repair_rows=repair_rows,
        semantic_files=[path.name for path in semantic_files],
        blockers=blockers,
        warnings=warnings,
        generated_files=generated,
    )
    (args.output_dir / "experiment_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Exported paper artifacts to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BlindSpot-RL paper artifacts.")
    parser.add_argument("--main-table-csv", required=True, type=Path)
    parser.add_argument("--main-table-md", type=Path)
    parser.add_argument(
        "--downstream-table-csv",
        action="append",
        default=[],
        metavar="BENCHMARK=PATH",
        help="Optional benchmark-level main table CSV to combine into a downstream utility table.",
    )
    parser.add_argument("--ablation-csv", type=Path)
    parser.add_argument("--teacher-union-csv", type=Path)
    parser.add_argument("--verifier-filter-csv", type=Path)
    parser.add_argument("--transition-summary-json", action="append", default=[], type=Path)
    parser.add_argument("--repair-summary-json", action="append", default=[], type=Path)
    parser.add_argument("--semantic-space-dir", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--handoff-json", type=Path)
    parser.add_argument("--handoff-md", type=Path)
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--evidence-csv", type=Path)
    parser.add_argument("--evidence-md", type=Path)
    parser.add_argument("--result-card-md", type=Path)
    parser.add_argument("--result-card-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_labeled_path_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            label = Path(spec).parent.name or Path(spec).stem
            parsed.append((label, Path(spec)))
            continue
        label, raw_path = spec.split("=", 1)
        if not label:
            raise SystemExit(f"Missing benchmark label in downstream table spec: {spec}")
        if not raw_path:
            raise SystemExit(f"Missing CSV path in downstream table spec: {spec}")
        parsed.append((label, Path(raw_path)))
    return parsed


def semantic_artifact(directory: Path | None, name: str) -> Path | None:
    return directory / name if directory else None


def clear_managed_generated_files(output_dir: Path) -> None:
    for name in [
        "main_table.tex",
        "rl_stage_ablation_table.tex",
        "downstream_utility_table.tex",
        "ablation_table.tex",
        "teacher_union_ablation_table.tex",
        "verifier_filter_ablation_table.tex",
        "dimension_transition_table.tex",
        "dimension_transition_table.csv",
        "dimension_transition_table.md",
        "repair_table.tex",
        "repair_table.csv",
        "repair_table.md",
        "semantic_space.svg",
        "semantic_space.pdf",
        "semantic_space_points.csv",
        "semantic_space_summary.json",
        "experiment_summary.md",
        *MANAGED_DOC_FILES,
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def read_csv_checked(
    path: Path | None,
    label: str,
    required: bool,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    if path is None:
        message = f"{label} path is not configured"
        return [], [message] if required else [], [] if required else [message]
    if not path.exists() or path.stat().st_size == 0:
        message = f"{label} CSV is missing: {path}"
        return [], [message] if required else [], [] if required else [message]
    try:
        return read_csv(path), [], []
    except OSError as exc:
        message = f"{label} CSV is not readable: {path}: {exc}"
        return [], [message] if required else [], [] if required else [message]


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"_load_error": f"{path}: not valid JSON at line {exc.lineno} column {exc.colno}"}
    except OSError as exc:
        return {"_load_error": f"{path}: {exc}"}
    if not isinstance(data, dict):
        return {"_load_error": f"{path}: audit JSON must be an object"}
    return data


def read_downstream_tables(
    specs: list[tuple[str, Path]],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not specs:
        return rows, blockers, ["downstream table paths are not configured"]
    for benchmark, path in specs:
        table_rows, table_blockers, table_warnings = read_csv_checked(
            path,
            f"{benchmark} downstream table",
            required=False,
        )
        blockers.extend(table_blockers)
        warnings.extend(table_warnings)
        for row in table_rows:
            method = row.get("method", "<unknown>")
            if row.get("downstream_status") != "pass" or normalize_bool(row.get("downstream_paper_claim_eligible")) is not True:
                blockers.append(
                    f"{benchmark} downstream row for {method} is not paper-eligible; "
                    "downstream_status must be pass and downstream_paper_claim_eligible must be true"
                )
                continue
            out = dict(row)
            out["benchmark"] = benchmark
            out["source_table"] = str(path)
            rows.append(out)
    return rows, blockers, warnings


def sanitize_main_downstream_metrics(rows: list[dict[str, str]]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        status = row.get("downstream_status", "")
        eligible = normalize_bool(row.get("downstream_paper_claim_eligible"))
        has_downstream_metric = any(
            row.get(key, "")
            for key in [
                "accuracy",
                "accuracy_ci",
                "accuracy_ci_lower",
                "accuracy_ci_upper",
                "tie_rate",
                "tie_rate_ci",
                "tie_rate_ci_lower",
                "tie_rate_ci_upper",
                "mean_margin",
                "mean_margin_ci",
                "mean_margin_ci_lower",
                "mean_margin_ci_upper",
                "downstream_n",
            ]
        )
        if not has_downstream_metric:
            continue
        if status == "pass" and eligible is True:
            continue
        method = row.get("method", "<unknown>")
        blockers.append(
            f"main table downstream metrics for {method} are not paper-eligible; "
            "clearing accuracy/tie/margin cells"
        )
        for key in [
            "accuracy",
            "accuracy_ci",
            "accuracy_ci_lower",
            "accuracy_ci_upper",
            "tie_rate",
            "tie_rate_ci",
            "tie_rate_ci_lower",
            "tie_rate_ci_upper",
            "mean_margin",
            "mean_margin_ci",
            "mean_margin_ci_lower",
            "mean_margin_ci_upper",
            "downstream_n",
        ]:
            row[key] = ""
    return blockers


def normalize_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def build_rl_stage_ablation_rows(
    main_rows: list[dict[str, str]],
    baseline_method: str = "sft_only",
    candidate_method: str = "sft_rl",
) -> tuple[list[dict[str, str]], list[str]]:
    if not main_rows:
        return [], []
    by_method = {row.get("method", ""): row for row in main_rows}
    baseline = by_method.get(baseline_method)
    candidate = by_method.get(candidate_method)
    if not baseline or not candidate:
        return [], [f"RL-stage ablation requires {baseline_method} and {candidate_method} rows in main table"]
    return [
        {
            "comparison": "SFT-only -> SFT+GRPO",
            "baseline_cov": baseline.get("cov", ""),
            "candidate_cov": candidate.get("cov", ""),
            "baseline_red": baseline.get("red", ""),
            "candidate_red": candidate.get("red", ""),
            "baseline_hall": baseline.get("hall", ""),
            "candidate_hall": candidate.get("hall", ""),
            "baseline_accuracy": baseline.get("accuracy", ""),
            "candidate_accuracy": candidate.get("accuracy", ""),
            "baseline_cov_per_gen": baseline.get("coverage_per_generated_criterion", ""),
            "candidate_cov_per_gen": candidate.get("coverage_per_generated_criterion", ""),
            "delta_cov": delta(candidate.get("cov"), baseline.get("cov")),
            "delta_cov_per_gen": delta(
                candidate.get("coverage_per_generated_criterion"),
                baseline.get("coverage_per_generated_criterion"),
            ),
            "delta_blind": delta(candidate.get("blind"), baseline.get("blind")),
            "delta_red": delta(candidate.get("red"), baseline.get("red")),
            "delta_hall": delta(candidate.get("hall"), baseline.get("hall")),
            "delta_accuracy": delta(candidate.get("accuracy"), baseline.get("accuracy")),
        }
    ], []


def delta(left: str | None, right: str | None) -> str:
    left_value = parse_float(left)
    right_value = parse_float(right)
    if left_value is None or right_value is None:
        return ""
    return f"{left_value - right_value:.4f}"


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def read_transition_summaries(paths: list[Path]) -> tuple[list[dict[str, str]], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    if not paths:
        return rows, blockers, ["dimension-transition summary path is not configured"]
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            warnings.append(f"dimension-transition summary JSON is missing: {path}")
            continue
        data = read_json(path)
        if is_load_error(data):
            blockers.append(f"dimension-transition summary is not readable: {data['_load_error']}")
            continue
        rows.append(transition_summary_to_row(data, path))
    return rows, blockers, warnings


def read_repair_summaries(paths: list[Path]) -> tuple[list[dict[str, str]], list[str], list[str]]:
    return read_transition_summaries(paths)


def collect_semantic_space_artifacts(directory: Path | None) -> tuple[list[Path], list[str], list[str]]:
    if directory is None:
        return [], [], ["semantic-space directory is not configured"]
    required = ["semantic_space.svg", "semantic_space.pdf", "semantic_space_points.csv", "semantic_space_summary.json"]
    files: list[Path] = []
    blockers: list[str] = []
    warnings: list[str] = []
    for name in required:
        path = directory / name
        if path.exists() and path.stat().st_size > 0:
            files.append(path)
        else:
            warnings.append(f"semantic-space artifact is missing: {path}")
    if len(files) == len(required):
        blockers.extend(
            semantic_space_contract_blockers(
                summary_path=directory / "semantic_space_summary.json",
                points_path=directory / "semantic_space_points.csv",
                svg_path=directory / "semantic_space.svg",
                pdf_path=directory / "semantic_space.pdf",
                paper_path=str(directory / "semantic_space_summary.json"),
            )
        )
        if blockers:
            files = []
    return files, blockers, warnings


def transition_summary_to_row(data: dict[str, Any], path: Path) -> dict[str, str]:
    baseline = str(data.get("baseline_label") or "baseline")
    candidate = str(data.get("candidate_label") or path.parent.name or "candidate")
    comparison = f"{baseline} -> {candidate}"
    return {
        "comparison": comparison,
        "total_gold": str(data.get("total_gold", "")),
        "baseline_blind_gold": str(data.get("baseline_blind_gold", "")),
        "recovered_gold": str(data.get("recovered_gold", data.get("repaired_gold", ""))),
        "lost_gold": str(data.get("lost_gold", "")),
        "baseline_coverage": str(data.get("baseline_coverage", "")),
        "candidate_coverage": str(data.get("candidate_coverage", "")),
        "recovered_dimension_rate": str(data.get("recovered_dimension_rate", data.get("repair_rate", ""))),
        "loss_rate": str(data.get("loss_rate", "")),
        "net_transition_rate": str(data.get("net_transition_rate", data.get("net_repair_rate", ""))),
        "source": str(path),
    }


def repair_summary_to_row(data: dict[str, Any], path: Path) -> dict[str, str]:
    return transition_summary_to_row(data, path)


def is_load_error(data: Any) -> bool:
    return isinstance(data, dict) and bool(data.get("_load_error"))


def latex_table(
    rows: list[dict[str, str]],
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
) -> str:
    align = "l" + "r" * (len(columns) - 1)
    header = " & ".join(title for _, title in columns) + r" \\"
    body = []
    for row in rows:
        body.append(" & ".join(format_cell(row.get(key, "")) for key, _ in columns) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_display_columns(rows: list[dict[str, str]], display_map: dict[str, str]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        new_row = dict(row)
        for raw_key, ci_key in display_map.items():
            new_row[f"{raw_key}_display"] = row.get(ci_key) or row.get(raw_key, "")
        out.append(new_row)
    return out


def format_cell(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value.replace("_", r"\_")


def build_summary_markdown(
    main_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    teacher_union_rows: list[dict[str, str]],
    audit: dict[str, Any],
    copied: list[str],
    downstream_rows: list[dict[str, str]] | None = None,
    rl_stage_rows: list[dict[str, str]] | None = None,
    verifier_filter_rows: list[dict[str, str]] | None = None,
    repair_rows: list[dict[str, str]] | None = None,
    semantic_files: list[str] | None = None,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    generated_files: list[str] | None = None,
) -> str:
    blockers = blockers or []
    warnings = warnings or []
    repair_rows = repair_rows or []
    verifier_filter_rows = verifier_filter_rows or []
    downstream_rows = downstream_rows or []
    rl_stage_rows = rl_stage_rows or []
    semantic_files = semantic_files or []
    generated_files = generated_files or [
        "main_table.tex",
        "rl_stage_ablation_table.tex",
        "downstream_utility_table.tex",
        "ablation_table.tex",
        "teacher_union_ablation_table.tex",
        "verifier_filter_ablation_table.tex",
        "dimension_transition_table.tex",
        "semantic_space.svg",
        "semantic_space.pdf",
        "evidence_matrix.md",
        "result_card.md",
        "experiment_summary.md",
    ]
    lines = ["# BlindSpot-RL Experiment Summary", ""]
    lines.append(f"- Main table rows: {len(main_rows)}")
    lines.append(f"- RL-stage ablation rows: {len(rl_stage_rows)}")
    lines.append(f"- Downstream utility rows: {len(downstream_rows)}")
    lines.append(f"- Ablation rows: {len(ablation_rows)}")
    lines.append(f"- Teacher-union ablation rows: {len(teacher_union_rows)}")
    lines.append(f"- Verifier-filter ablation rows: {len(verifier_filter_rows)}")
    lines.append(f"- Dimension-transition rows: {len(repair_rows)}")
    lines.append(f"- Semantic-space artifacts: {len(semantic_files)}")
    if audit:
        if is_load_error(audit):
            lines.append(f"- Audit status: blocked")
            lines.append(f"- Audit error: {audit['_load_error']}")
        else:
            lines.append(f"- Audit ok: {audit.get('ok')}")
            lines.append(f"- Missing files: {len(audit.get('missing_files', []))}")
    lines.append(f"- Copied artifacts: {', '.join(copied) if copied else 'none'}")
    lines.append(f"- Blockers: {len(blockers)}")
    lines.append(f"- Warnings: {len(warnings)}")
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    lines.extend([f"- {item}" for item in warnings] or ["- none"])
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.extend([f"- `{item}`" for item in generated_files] or ["- none"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
