from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_latex_compile import build_report, count_pdf_pages


class CheckLatexCompileTest(unittest.TestCase):
    def test_build_report_passes_with_official_style_pdf_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)
            (paper / "main.tex").write_text(
                "\\usepackage[submission]{aaai2026}\n"
                "\\author{Anonymous Submission}\n"
                "\\bibliographystyle{aaai2026}\n",
                encoding="utf-8",
            )
            (paper / "aaai2026.sty").write_text("style", encoding="utf-8")
            (paper / "aaai2026.bst").write_text("bst", encoding="utf-8")
            (paper / "main.log").write_text("(./aaai2026.sty)\nOutput written on main.pdf\n", encoding="utf-8")
            (paper / "main.aux").write_text("\\bibstyle{aaai2026}\n", encoding="utf-8")
            (paper / "main.pdf").write_bytes(b"%PDF-1.4\n1 0 obj << /Type /Page >> endobj\n%%EOF\n")

            report = build_report(paper, require_official_style=True, require_anonymous=True)

        self.assertTrue(report["ok"])
        self.assertTrue(report["official_style_files_present"])
        self.assertTrue(report["official_style_active"])
        self.assertTrue(report["submission_mode_declared"])
        self.assertTrue(report["bibliography_style_declared"])
        self.assertTrue(report["bibliography_style_active"])
        self.assertTrue(report["anonymous_author_declared"])
        self.assertTrue(report["pdf_header_ok"])
        self.assertEqual(report["page_count"], 1)

    def test_build_report_blocks_missing_official_style_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)
            (paper / "main.tex").write_text("x", encoding="utf-8")

            report = build_report(paper, require_official_style=True)

        self.assertFalse(report["ok"])
        self.assertTrue(any("aaai2026.sty" in blocker for blocker in report["blockers"]))
        self.assertTrue(any("missing compiled PDF" in blocker for blocker in report["blockers"]))

    def test_build_report_blocks_non_anonymous_main_tex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)
            (paper / "main.tex").write_text(
                "\\usepackage{aaai2026}\n\\author{Real Author}\n",
                encoding="utf-8",
            )
            (paper / "aaai2026.sty").write_text("style", encoding="utf-8")
            (paper / "aaai2026.bst").write_text("bst", encoding="utf-8")
            (paper / "main.log").write_text("(./aaai2026.sty)\nOutput written on main.pdf\n", encoding="utf-8")
            (paper / "main.aux").write_text("\\bibstyle{plain}\n", encoding="utf-8")
            (paper / "main.pdf").write_bytes(b"%PDF-1.4\n1 0 obj << /Type /Page >> endobj\n%%EOF\n")

            report = build_report(paper, require_official_style=True, require_anonymous=True)

        self.assertFalse(report["ok"])
        self.assertFalse(report["submission_mode_declared"])
        self.assertFalse(report["bibliography_style_declared"])
        self.assertFalse(report["bibliography_style_active"])
        self.assertFalse(report["anonymous_author_declared"])
        self.assertIn("main.tex does not declare AAAI submission mode", report["blockers"])
        self.assertIn("main.tex does not declare AAAI bibliography style", report["blockers"])
        self.assertIn("official AAAI bibliography style was not observed in paper/main.aux", report["blockers"])
        self.assertIn("main.tex does not declare Anonymous Submission author", report["blockers"])

    def test_build_report_blocks_page_limit_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)
            (paper / "main.tex").write_text(
                "\\usepackage[submission]{aaai2026}\n"
                "\\author{Anonymous Submission}\n"
                "\\bibliographystyle{aaai2026}\n",
                encoding="utf-8",
            )
            (paper / "aaai2026.sty").write_text("style", encoding="utf-8")
            (paper / "aaai2026.bst").write_text("bst", encoding="utf-8")
            (paper / "main.log").write_text("(./aaai2026.sty)\nOutput written on main.pdf\n", encoding="utf-8")
            (paper / "main.aux").write_text("\\bibstyle{aaai2026}\n", encoding="utf-8")
            pages = "\n".join("1 0 obj << /Type /Page >> endobj" for _ in range(9))
            (paper / "main.pdf").write_bytes(("%PDF-1.4\n" + pages + "\n%%EOF\n").encode("ascii"))

            report = build_report(
                paper,
                require_official_style=True,
                require_anonymous=True,
                max_pages=8,
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["page_count"], 9)
        self.assertIn("compiled PDF has 9 pages, exceeding max_pages=8", report["blockers"])

    def test_count_pdf_pages_ignores_pages_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "main.pdf"
            pdf.write_bytes(b"%PDF-1.4\n/Type /Pages\n/Type /Page\n/Type /Page\n%%EOF")

            self.assertEqual(count_pdf_pages(pdf), 2)


if __name__ == "__main__":
    unittest.main()
