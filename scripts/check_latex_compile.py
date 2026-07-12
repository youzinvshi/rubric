#!/usr/bin/env python3
"""Check that the paper compiles under the official AAAI style."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ERROR_MARKERS = [
    "Fatal error occurred",
    "Emergency stop",
    "LaTeX Error",
]


def main() -> None:
    args = parse_args()
    report = build_report(
        paper_dir=args.paper_dir,
        compile_pdf=args.compile,
        require_official_style=args.require_official_style,
        require_anonymous=args.require_anonymous,
        max_pages=args.max_pages,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"LaTeX compile check ok={report['ok']} report={args.output_json}")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AAAI LaTeX compile artifacts.")
    parser.add_argument("--paper-dir", default=Path("paper"), type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--compile", action="store_true", help="Run pdflatex before checking artifacts.")
    parser.add_argument("--require-official-style", action="store_true")
    parser.add_argument("--require-anonymous", action="store_true")
    parser.add_argument("--max-pages", default=0, type=int, help="Optional page limit; 0 disables this check.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def build_report(
    paper_dir: Path,
    compile_pdf: bool = False,
    require_official_style: bool = False,
    require_anonymous: bool = False,
    max_pages: int = 0,
) -> dict[str, Any]:
    tex = paper_dir / "main.tex"
    pdf = paper_dir / "main.pdf"
    log = paper_dir / "main.log"
    aux = paper_dir / "main.aux"
    sty = paper_dir / "aaai2026.sty"
    bst = paper_dir / "aaai2026.bst"
    blockers: list[str] = []
    warnings: list[str] = []
    compile_result = run_pdflatex(paper_dir) if compile_pdf else {"ran": False}

    if compile_result.get("returncode") not in (None, 0):
        blockers.append(f"pdflatex failed with returncode={compile_result.get('returncode')}")
    if compile_result.get("missing_binary"):
        blockers.append("pdflatex binary is not available")

    tex_text = tex.read_text(encoding="utf-8", errors="replace") if tex.exists() else ""
    if not tex.exists():
        blockers.append(f"missing LaTeX source: {tex}")
    submission_mode_declared = r"\usepackage[submission]{aaai2026}" in tex_text
    anonymous_author_declared = r"\author{Anonymous Submission}" in tex_text
    bibliography_style_declared = r"\bibliographystyle{aaai2026}" in tex_text
    if require_official_style and not submission_mode_declared:
        blockers.append("main.tex does not declare AAAI submission mode")
    if require_official_style and not bibliography_style_declared:
        blockers.append("main.tex does not declare AAAI bibliography style")
    if require_anonymous and not anonymous_author_declared:
        blockers.append("main.tex does not declare Anonymous Submission author")
    official_style_files_present = sty.exists() and bst.exists()
    if require_official_style and not sty.exists():
        blockers.append("missing official AAAI style file: paper/aaai2026.sty")
    if require_official_style and not bst.exists():
        blockers.append("missing official AAAI bibliography style: paper/aaai2026.bst")

    pdf_exists = pdf.exists() and pdf.stat().st_size > 0
    pdf_header_ok = pdf_exists and pdf.read_bytes().startswith(b"%PDF")
    if not pdf_exists:
        blockers.append(f"missing compiled PDF: {pdf}")
    elif not pdf_header_ok:
        blockers.append(f"compiled PDF does not start with %PDF: {pdf}")

    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    aux_text = aux.read_text(encoding="utf-8", errors="replace") if aux.exists() else ""
    if not log.exists():
        blockers.append(f"missing LaTeX log: {log}")
    for marker in ERROR_MARKERS:
        if marker in log_text:
            blockers.append(f"LaTeX log contains error marker: {marker}")

    official_style_active = "aaai2026.sty" in log_text
    if require_official_style and not official_style_active:
        blockers.append("official AAAI style was not observed in paper/main.log")
    bibliography_style_active = r"\bibstyle{aaai2026}" in aux_text or "aaai2026.bst" in log_text
    if require_official_style and not aux.exists():
        blockers.append(f"missing LaTeX aux file: {aux}")
    elif require_official_style and not bibliography_style_active:
        blockers.append("official AAAI bibliography style was not observed in paper/main.aux")

    page_count = count_pdf_pages(pdf) if pdf_exists else 0
    if pdf_exists and page_count == 0:
        warnings.append("could not infer PDF page count from main.pdf")
    if max_pages > 0 and page_count > max_pages:
        blockers.append(f"compiled PDF has {page_count} pages, exceeding max_pages={max_pages}")

    return {
        "ok": not blockers,
        "paper_dir": str(paper_dir),
        "tex_path": str(tex),
        "pdf_path": str(pdf),
        "log_path": str(log),
        "aux_path": str(aux),
        "compile": compile_result,
        "official_style_files_present": official_style_files_present,
        "official_style_active": official_style_active,
        "require_official_style": require_official_style,
        "submission_mode_declared": submission_mode_declared,
        "bibliography_style_declared": bibliography_style_declared,
        "bibliography_style_active": bibliography_style_active,
        "require_anonymous": require_anonymous,
        "anonymous_author_declared": anonymous_author_declared,
        "pdf_exists": pdf_exists,
        "pdf_bytes": pdf.stat().st_size if pdf.exists() else 0,
        "pdf_header_ok": pdf_header_ok,
        "page_count": page_count,
        "max_pages": max_pages,
        "blockers": blockers,
        "warnings": warnings,
    }


def run_pdflatex(paper_dir: Path) -> dict[str, Any]:
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    try:
        result = subprocess.run(
            command,
            cwd=paper_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return {"ran": True, "command": command, "missing_binary": True}
    return {
        "ran": True,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def count_pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    return max(data.count(b"/Type /Page") - data.count(b"/Type /Pages"), 0)


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LaTeX Compile Check",
        "",
        f"- ok: `{str(report['ok']).lower()}`",
        f"- pdf: `{report['pdf_path']}`",
        f"- pdf_bytes: `{report['pdf_bytes']}`",
        f"- page_count: `{report['page_count']}`",
        f"- official_style_files_present: `{str(report['official_style_files_present']).lower()}`",
        f"- official_style_active: `{str(report['official_style_active']).lower()}`",
        f"- submission_mode_declared: `{str(report['submission_mode_declared']).lower()}`",
        f"- bibliography_style_declared: `{str(report['bibliography_style_declared']).lower()}`",
        f"- bibliography_style_active: `{str(report['bibliography_style_active']).lower()}`",
        f"- anonymous_author_declared: `{str(report['anonymous_author_declared']).lower()}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item}" for item in report.get("blockers", []))
    if not report.get("blockers"):
        lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    lines.extend(f"- {item}" for item in report.get("warnings", []))
    if not report.get("warnings"):
        lines.append("- none")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
