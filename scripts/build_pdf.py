"""Render report/2_pager.md to PDF using markdown-pdf (pure-python, no LaTeX)."""

from __future__ import annotations

from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "report" / "2_pager.md"
OUT = REPO_ROOT / "report" / "2_pager.pdf"


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
    text = SRC.read_text()
    # markdown-pdf creates a hard page break between Sections, so we split
    # on a custom marker (rendered as a comment by GitHub's md preview, so
    # the markdown stays readable on the GH side too).
    chunks = [c.strip() for c in text.split(_PAGEBREAK_MARKER)]

    pdf = MarkdownPdf(toc_level=0, optimize=True)
    for chunk in chunks:
        if not chunk:
            continue
        # root=SRC.parent so relative image paths (``fig1_toolbox_per_metric.png``)
        # resolve next to the markdown — the report/ directory is self-contained
        # and renders correctly both via this script and on GitHub's md preview.
        pdf.add_section(Section(chunk, root=str(SRC.parent)), user_css=_USER_CSS)
    pdf.meta["title"] = "Pokec-z — fairness multi-axes par composition post-hoc"
    pdf.meta["author"] = "Mini-projet IADATA708"
    pdf.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
