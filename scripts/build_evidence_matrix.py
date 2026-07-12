#!/usr/bin/env python3
"""Build a claim-to-evidence matrix for BlindSpot-RL paper writing."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path
from typing import Any


OPS = {
    "==": lambda actual, expected: actual == expected,
    "!=": lambda actual, expected: actual != expected,
    ">": lambda actual, expected: actual > expected,
    ">=": lambda actual, expected: actual >= expected,
    "<": lambda actual, expected: actual < expected,
    "<=": lambda actual, expected: actual <= expected,
}
VALUE_OPS = set(OPS) | {"in", "not in"}


def configured_fail_status(item: dict[str, Any]) -> str:
    status = str(item.get("fail_status", "fail"))
    if status not in {"fail", "missing"}:
        raise ValueError(f"Unsupported fail_status: {status}")
    return status


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = build_matrix(config, root=args.root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evidence_matrix.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "evidence_matrix.csv", rows)
    (args.output_dir / "evidence_matrix.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} evidence rows to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BlindSpot-RL claim-evidence matrix.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", default=Path("."), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Evidence Matrix config is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Evidence Matrix config is not valid JSON: {path}: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Evidence Matrix config must be a JSON object: {path}")
    return data


def build_matrix(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for claim in config.get("claims", []):
        artifact_checks = [check_artifact(item, root) for item in claim.get("artifacts", [])]
        metric_checks = [check_metric(item, root) for item in claim.get("metrics", [])]
        value_checks = [check_value(item, root) for item in claim.get("values", [])]
        csv_checks = [check_csv(item, root) for item in claim.get("csv_checks", [])]
        file_sha256_checks = [check_file_sha256(item, root) for item in claim.get("file_sha256_checks", [])]
        table_value_checks = [check_table_value(item, root) for item in claim.get("table_values", [])]
        comparison_checks = [check_comparison(item, root) for item in claim.get("comparisons", [])]
        value_comparison_checks = [
            check_value_comparison(item, root) for item in claim.get("value_comparisons", [])
        ]
        checks = (
            artifact_checks
            + metric_checks
            + value_checks
            + csv_checks
            + file_sha256_checks
            + table_value_checks
            + comparison_checks
            + value_comparison_checks
        )
        failed = [item for item in checks if item["status"] != "pass"]
        status = "safe_to_claim" if checks and not failed else "not_yet_supported"
        if any(item["status"] == "fail" for item in failed):
            status = "contradicted"
        elif any(item["status"] == "missing" for item in failed):
            status = "missing_evidence"

        rows.append(
            {
                "claim_id": claim["id"],
                "claim": claim["claim"],
                "paper_section": claim.get("section", ""),
                "status": status,
                "evidence": "; ".join(format_check(item) for item in checks) or "no checks configured",
                "notes": claim.get("notes", ""),
            }
        )
    return rows


def check_artifact(item: str | dict[str, Any], root: Path) -> dict[str, Any]:
    path = item if isinstance(item, str) else item["path"]
    label = path if isinstance(item, str) else item.get("label", path)
    full_path = root / path
    if full_path.exists() and full_path.stat().st_size > 0:
        return {"kind": "artifact", "label": label, "status": "pass", "detail": path}
    return {"kind": "artifact", "label": label, "status": "missing", "detail": path}


def check_metric(item: dict[str, Any], root: Path) -> dict[str, Any]:
    metric = item["metric"]
    label = item.get("label", f"{item['path']}::{metric}")
    metric_result = read_metric(root=root, path=item["path"], metric=metric, label=label)
    if metric_result["status"] != "pass":
        return {
            "kind": "metric",
            "label": label,
            "status": metric_result["status"],
            "detail": metric_result["detail"],
        }

    op = item.get("op", ">=")
    if op not in OPS:
        raise ValueError(f"Unsupported metric op: {op}")
    actual = float(metric_result["value"])
    expected = float(item["value"])
    passed = OPS[op](actual, expected)
    return {
        "kind": "metric",
        "label": label,
        "status": "pass" if passed else configured_fail_status(item),
        "detail": f"{metric}={actual:.4f} {op} {expected:.4f}",
    }


def check_value(item: dict[str, Any], root: Path) -> dict[str, Any]:
    key = item["key"]
    label = item.get("label", f"{item['path']}::{key}")
    value_result = read_value(root=root, path=item["path"], key=key, label=label)
    if value_result["status"] != "pass":
        return {
            "kind": "value",
            "label": label,
            "status": value_result["status"],
            "detail": value_result["detail"],
        }

    op = item.get("op", "==")
    if op not in VALUE_OPS:
        raise ValueError(f"Unsupported value op: {op}")
    expected = item["value"]
    actual = value_result["value"]
    if op in {"==", "!="}:
        passed = OPS[op](stringify_json_value(actual), stringify_json_value(expected))
    elif op in {"in", "not in"}:
        if not isinstance(expected, list):
            return {"kind": "value", "label": label, "status": "fail", "detail": f"{key} expected value is not a list"}
        actual_text = stringify_json_value(actual)
        expected_values = {stringify_json_value(value) for value in expected}
        passed = actual_text in expected_values if op == "in" else actual_text not in expected_values
    else:
        try:
            passed = OPS[op](float(actual), float(expected))
        except (TypeError, ValueError):
            return {"kind": "value", "label": label, "status": "fail", "detail": f"{key} is non-numeric"}
    return {
        "kind": "value",
        "label": label,
        "status": "pass" if passed else configured_fail_status(item),
        "detail": f"{key}={stringify_json_value(actual)} {op} {stringify_json_value(expected)}",
    }


def check_table_value(item: dict[str, Any], root: Path) -> dict[str, Any]:
    key = item["key"]
    row_key = item.get("row_key", "method")
    row_value = str(item["row_value"])
    label = item.get("label", f"{item['path']}::{row_key}={row_value}::{key}")
    value_result = read_table_value(
        root=root,
        path=item["path"],
        row_key=row_key,
        row_value=row_value,
        key=key,
        label=label,
    )
    if value_result["status"] != "pass":
        return {
            "kind": "table_value",
            "label": label,
            "status": value_result["status"],
            "detail": value_result["detail"],
        }

    op = item.get("op", "==")
    if op not in VALUE_OPS:
        raise ValueError(f"Unsupported table value op: {op}")
    expected = item["value"]
    actual = value_result["value"]
    if op in {"==", "!="}:
        passed = OPS[op](stringify_json_value(actual), stringify_json_value(expected))
    elif op in {"in", "not in"}:
        if not isinstance(expected, list):
            return {
                "kind": "table_value",
                "label": label,
                "status": "fail",
                "detail": f"{key} expected value is not a list",
            }
        actual_text = stringify_json_value(actual)
        expected_values = {stringify_json_value(value) for value in expected}
        passed = actual_text in expected_values if op == "in" else actual_text not in expected_values
    else:
        try:
            passed = OPS[op](float(actual), float(expected))
        except (TypeError, ValueError):
            return {"kind": "table_value", "label": label, "status": "fail", "detail": f"{key} is non-numeric"}
    return {
        "kind": "table_value",
        "label": label,
        "status": "pass" if passed else configured_fail_status(item),
        "detail": (
            f"{row_key}={row_value} {key}={stringify_json_value(actual)} "
            f"{op} {stringify_json_value(expected)}"
        ),
    }


def check_csv(item: dict[str, Any], root: Path) -> dict[str, Any]:
    label = item.get("label", item["path"])
    result = read_csv_rows(root=root, path=item["path"], label=label)
    if result["status"] != "pass":
        return {"kind": "csv", "label": label, "status": result["status"], "detail": result["detail"]}

    rows = result["rows"]
    fieldnames = result["fieldnames"]
    expected_columns = item.get("columns")
    if expected_columns is not None:
        expected = [str(column) for column in expected_columns]
        if item.get("column_mode", "contains") == "exact":
            if fieldnames != expected:
                return {
                    "kind": "csv",
                    "label": label,
                    "status": "fail",
                    "detail": f"columns={fieldnames} != {expected}",
                }
        else:
            missing = [column for column in expected if column not in fieldnames]
            if missing:
                return {
                    "kind": "csv",
                    "label": label,
                    "status": "missing",
                    "detail": f"missing columns: {', '.join(missing)}",
                }

    filtered_rows = rows
    where = item.get("where", {})
    if where:
        filtered_rows = [
            row
            for row in rows
            if all(str(row.get(key, "")) == str(value) for key, value in where.items())
        ]

    min_rows = int(item.get("min_rows", 0))
    if len(filtered_rows) < min_rows:
        return {
            "kind": "csv",
            "label": label,
            "status": "missing",
            "detail": f"{len(filtered_rows)} rows match {where or '<all>'}; expected at least {min_rows}",
        }

    for column in item.get("non_empty", []):
        if column not in fieldnames:
            return {"kind": "csv", "label": label, "status": "missing", "detail": f"missing column {column}"}
        empty_count = sum(1 for row in filtered_rows if str(row.get(column, "")).strip() == "")
        if empty_count:
            return {
                "kind": "csv",
                "label": label,
                "status": "fail",
                "detail": f"{empty_count} rows have empty {column}",
            }

    for column in item.get("numeric", []):
        if column not in fieldnames:
            return {"kind": "csv", "label": label, "status": "missing", "detail": f"missing column {column}"}
        for row_idx, row in enumerate(filtered_rows):
            try:
                float(row.get(column, ""))
            except (TypeError, ValueError):
                return {
                    "kind": "csv",
                    "label": label,
                    "status": "fail",
                    "detail": f"row {row_idx} has non-numeric {column}",
                }

    return {
        "kind": "csv",
        "label": label,
        "status": "pass",
        "detail": f"{len(filtered_rows)} rows match {where or '<all>'}",
    }


def check_file_sha256(item: dict[str, Any], root: Path) -> dict[str, Any]:
    label = item.get("label", "file SHA256 check")
    summary_result = read_value(
        root=root,
        path=item["json_path"],
        key=item["json_key"],
        label=f"{label}:json",
    )
    if summary_result["status"] != "pass":
        return {"kind": "file_sha256", "label": label, "status": summary_result["status"], "detail": summary_result["detail"]}

    file_path = root / item["file_path"]
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {"kind": "file_sha256", "label": label, "status": "missing", "detail": str(file_path)}

    expected = stringify_json_value(summary_result["value"])
    if expected == "":
        return {
            "kind": "file_sha256",
            "label": label,
            "status": "missing",
            "detail": f"{item['json_key']}({item['json_path']}) is empty",
        }
    actual = file_sha256(file_path)
    op = item.get("op", "==")
    if op not in {"==", "!="}:
        raise ValueError(f"Unsupported file_sha256 op: {op}")
    passed = OPS[op](actual, expected)
    return {
        "kind": "file_sha256",
        "label": label,
        "status": "pass" if passed else configured_fail_status(item),
        "detail": f"sha256({item['file_path']})={actual} {op} {item['json_key']}({item['json_path']})={expected}",
    }


def check_comparison(item: dict[str, Any], root: Path) -> dict[str, Any]:
    label = item.get("label", "metric comparison")
    left = read_metric(
        root=root,
        path=item["left_path"],
        metric=item["left_metric"],
        label=f"{label}:left",
    )
    right = read_metric(
        root=root,
        path=item["right_path"],
        metric=item["right_metric"],
        label=f"{label}:right",
    )
    if left["status"] != "pass":
        return {"kind": "comparison", "label": label, "status": left["status"], "detail": left["detail"]}
    if right["status"] != "pass":
        return {"kind": "comparison", "label": label, "status": right["status"], "detail": right["detail"]}

    mode = item.get("mode", "diff")
    left_value = float(left["value"])
    right_value = float(right["value"])
    if mode == "diff":
        actual = left_value - right_value
        detail = (
            f"{item['left_metric']}({item['left_path']}) - "
            f"{item['right_metric']}({item['right_path']}) = {actual:.4f}"
        )
    elif mode == "ratio":
        if right_value == 0:
            return {"kind": "comparison", "label": label, "status": "fail", "detail": "division by zero"}
        actual = left_value / right_value
        detail = (
            f"{item['left_metric']}({item['left_path']}) / "
            f"{item['right_metric']}({item['right_path']}) = {actual:.4f}"
        )
    else:
        raise ValueError(f"Unsupported comparison mode: {mode}")

    op = item.get("op", ">=")
    if op not in OPS:
        raise ValueError(f"Unsupported comparison op: {op}")
    expected = float(item["value"])
    passed = OPS[op](actual, expected)
    return {
        "kind": "comparison",
        "label": label,
        "status": "pass" if passed else configured_fail_status(item),
        "detail": f"{detail} {op} {expected:.4f}",
    }


def check_value_comparison(item: dict[str, Any], root: Path) -> dict[str, Any]:
    label = item.get("label", "value comparison")
    left_key = item["left_key"]
    right_key = item["right_key"]
    left = read_value(
        root=root,
        path=item["left_path"],
        key=left_key,
        label=f"{label}:left",
    )
    right = read_value(
        root=root,
        path=item["right_path"],
        key=right_key,
        label=f"{label}:right",
    )
    if left["status"] != "pass":
        return {"kind": "value_comparison", "label": label, "status": left["status"], "detail": left["detail"]}
    if right["status"] != "pass":
        return {"kind": "value_comparison", "label": label, "status": right["status"], "detail": right["detail"]}

    op = item.get("op", "==")
    if op not in OPS:
        raise ValueError(f"Unsupported value comparison op: {op}")
    left_value = left["value"]
    right_value = right["value"]
    if stringify_json_value(left_value) == "" or stringify_json_value(right_value) == "":
        return {
            "kind": "value_comparison",
            "label": label,
            "status": "missing",
            "detail": (
                f"{left_key}({item['left_path']})={stringify_json_value(left_value)} or "
                f"{right_key}({item['right_path']})={stringify_json_value(right_value)} is empty"
            ),
        }
    if op in {"==", "!="}:
        passed = OPS[op](stringify_json_value(left_value), stringify_json_value(right_value))
    else:
        try:
            passed = OPS[op](float(left_value), float(right_value))
        except (TypeError, ValueError):
            return {
                "kind": "value_comparison",
                "label": label,
                "status": "fail",
                "detail": f"{left_key} or {right_key} is non-numeric",
            }
    return {
        "kind": "value_comparison",
        "label": label,
        "status": "pass" if passed else configured_fail_status(item),
        "detail": (
            f"{left_key}({item['left_path']})={stringify_json_value(left_value)} {op} "
            f"{right_key}({item['right_path']})={stringify_json_value(right_value)}"
        ),
    }


def read_metric(root: Path, path: str, metric: str, label: str) -> dict[str, Any]:
    full_path = root / path
    if not full_path.exists():
        return {"label": label, "status": "missing", "detail": str(full_path)}
    try:
        data = json.loads(full_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "label": label,
            "status": "missing",
            "detail": f"{full_path}: not valid JSON at line {exc.lineno} column {exc.colno}",
        }
    except OSError as exc:
        return {"label": label, "status": "missing", "detail": f"{full_path}: {exc}"}
    value = nested_get(data, metric)
    if value is None:
        return {"label": label, "status": "missing", "detail": metric}
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return {"label": label, "status": "fail", "detail": f"{metric} is non-numeric"}
    return {"label": label, "status": "pass", "detail": f"{metric}={numeric:.4f}", "value": numeric}


def read_value(root: Path, path: str, key: str, label: str) -> dict[str, Any]:
    full_path = root / path
    if not full_path.exists():
        return {"label": label, "status": "missing", "detail": str(full_path)}
    try:
        data = json.loads(full_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "label": label,
            "status": "missing",
            "detail": f"{full_path}: not valid JSON at line {exc.lineno} column {exc.colno}",
        }
    except OSError as exc:
        return {"label": label, "status": "missing", "detail": f"{full_path}: {exc}"}
    value = nested_get(data, key)
    if value is None:
        return {"label": label, "status": "missing", "detail": key}
    return {"label": label, "status": "pass", "detail": f"{key}={stringify_json_value(value)}", "value": value}


def read_table_value(
    root: Path,
    path: str,
    row_key: str,
    row_value: str,
    key: str,
    label: str,
) -> dict[str, Any]:
    full_path = root / path
    if not full_path.exists():
        return {"label": label, "status": "missing", "detail": str(full_path)}
    try:
        with full_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        return {"label": label, "status": "missing", "detail": f"{full_path}: {exc}"}
    if not rows:
        return {"label": label, "status": "missing", "detail": f"{path}: no rows"}
    if row_key not in rows[0]:
        return {"label": label, "status": "missing", "detail": f"{path}: missing column {row_key}"}
    row = next((item for item in rows if item.get(row_key) == row_value), None)
    if row is None:
        return {"label": label, "status": "missing", "detail": f"{path}: missing row {row_key}={row_value}"}
    if key not in row:
        return {"label": label, "status": "missing", "detail": f"{path}: missing column {key}"}
    return {"label": label, "status": "pass", "detail": f"{key}={row[key]}", "value": row[key]}


def read_csv_rows(root: Path, path: str, label: str) -> dict[str, Any]:
    full_path = root / path
    if not full_path.exists():
        return {"label": label, "status": "missing", "detail": str(full_path)}
    try:
        with full_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    except OSError as exc:
        return {"label": label, "status": "missing", "detail": f"{full_path}: {exc}"}
    if not fieldnames:
        return {"label": label, "status": "missing", "detail": f"{path}: missing header"}
    return {"label": label, "status": "pass", "detail": f"{len(rows)} rows", "rows": rows, "fieldnames": fieldnames}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stringify_json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def nested_get(data: Any, dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        selector = parse_selector(part)
        if selector:
            list_key, match_key, match_value = selector
            if isinstance(current, dict):
                current = current.get(list_key)
            else:
                return None
            if not isinstance(current, list):
                return None
            current = next(
                (
                    item
                    for item in current
                    if isinstance(item, dict) and str(item.get(match_key)) == match_value
                ),
                None,
            )
            if current is None:
                return None
            continue
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def parse_selector(part: str) -> tuple[str, str, str] | None:
    if not part.endswith("]") or "[" not in part:
        return None
    list_key, selector = part[:-1].split("[", 1)
    if "=" not in selector or not list_key:
        return None
    match_key, match_value = selector.split("=", 1)
    if not match_key:
        return None
    return list_key, match_key, match_value


def format_check(item: dict[str, Any]) -> str:
    return f"[{item['status']}] {item['label']} ({item['detail']})"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["claim_id", "claim", "paper_section", "status", "evidence", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Claim ID | Section | Status | Claim | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(row["claim_id"]),
                    escape_md(row["paper_section"]),
                    escape_md(row["status"]),
                    escape_md(row["claim"]),
                    escape_md(row["evidence"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
