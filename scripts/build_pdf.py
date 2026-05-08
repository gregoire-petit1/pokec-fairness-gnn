"""Render report/2_pager.md (or any .md) to PDF via markdown-pdf, no LaTeX.

Usage::

    .venv/bin/python scripts/build_pdf.py [SRC_MD]

Defaults to ``report/2_pager.md`` -> ``report/2_pager.pdf``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

REPO_ROOT = Path(__file__).resolve().parent.parent


_USER_CSS = """
@page { size: A4; margin: 1.0cm 1.3cm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 8.8pt; line-height: 1.18; }
h1 { font-size: 12.5pt; margin: 0 0 0.25em 0; }
h2 { font-size: 10pt; margin: 0.5em 0 0.2em 0; }
h3 { font-size: 9.5pt; }
p, li { margin: 0.1em 0; }
ul, ol { margin: 0.15em 0; padding-left: 1.2em; }
table { font-size: 8pt; margin: 0.25em 0; border-collapse: collapse; }
th, td { padding: 1px 5px; white-space: nowrap; }
img { max-width: 92%; height: auto; }
em { font-size: 8pt; color: #555; }
hr { display: none; }
"""


_PAGEBREAK_MARKER = "<!-- PAGEBREAK -->"


def main() -> None:
    src = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else REPO_ROOT / "report" / "2_pager.md"
    )
    out = src.with_suffix(".pdf")
    text = src.read_text()
    # markdown-pdf creates a hard page break between Sections, so we split
    # on a custom marker (rendered as a comment by GitHub's md preview, so
    # the markdown stays readable on the GH side too).
    chunks = [c.strip() for c in text.split(_PAGEBREAK_MARKER)]

    pdf = MarkdownPdf(toc_level=0, optimize=True)
    for chunk in chunks:
        if not chunk:
            continue
        # root=src.parent so relative image paths resolve next to the markdown.
        pdf.add_section(Section(chunk, root=str(src.parent)), user_css=_USER_CSS)
    pdf.meta["title"] = "Pokec-z — fairness multi-axes par composition post-hoc"
    pdf.meta["author"] = "Mini-projet IADATA708"
    pdf.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
