"""Exporta docs/APEX_ARCHITECTURE.md a PDF."""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "APEX_ARCHITECTURE.md"
PDF_PATH = ROOT / "docs" / "APEX_ARCHITECTURE.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 1.8cm;
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 20pt;
    color: #0d47a1;
    border-bottom: 2px solid #0d47a1;
    padding-bottom: 6px;
    margin-top: 0;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    color: #1565c0;
    margin-top: 18px;
    page-break-after: avoid;
}
h3 {
    font-size: 11pt;
    color: #1976d2;
    page-break-after: avoid;
}
h4 { font-size: 10pt; color: #333; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}
th, td {
    border: 1px solid #bbb;
    padding: 6px 8px;
    text-align: left;
}
th {
    background-color: #e3f2fd;
    font-weight: bold;
}
tr:nth-child(even) { background-color: #f5f5f5; }
code, pre {
    font-family: Courier, monospace;
    font-size: 8pt;
    background: #f5f5f5;
}
pre {
    padding: 8px;
    border: 1px solid #ddd;
    white-space: pre-wrap;
    word-wrap: break-word;
}
blockquote {
    border-left: 4px solid #1565c0;
    margin: 12px 0;
    padding: 8px 12px;
    background: #e8f4fc;
    font-style: italic;
}
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
ul, ol { margin: 6px 0; padding-left: 22px; }
.diagram-note {
    background: #fff3e0;
    border: 1px solid #ffb74d;
    padding: 10px;
    font-size: 9pt;
    margin: 10px 0;
}
.cover {
    text-align: center;
    padding: 80px 20px 40px;
}
.cover h1 { border: none; font-size: 26pt; }
.cover p { font-size: 12pt; color: #555; }
"""


def preprocess_md(text: str) -> str:
    """Convierte bloques mermaid a nota legible en PDF."""

    def repl_mermaid(m: re.Match[str]) -> str:
        block = m.group(1).strip()
        first_line = block.split("\n")[0][:60]
        return (
            f'\n<div class="diagram-note">'
            f"<strong>Diagrama (Mermaid)</strong> — {first_line}… "
            f"<br><em>Ver diagrama interactivo en APEX_ARCHITECTURE.md (Markdown).</em>"
            f"</div>\n"
        )

    text = re.sub(r"```mermaid\s*\n(.*?)```", repl_mermaid, text, flags=re.DOTALL)
    return text


def md_to_html(md_text: str) -> str:
    md_text = preprocess_md(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>AURION X 3.0 — Apex Architecture</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def html_to_pdf(html: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        status = pisa.CreatePDF(html.encode("utf-8"), dest=f, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"Error generando PDF: {status.err}")


def main() -> None:
    if not MD_PATH.is_file():
        raise FileNotFoundError(MD_PATH)
    md_text = MD_PATH.read_text(encoding="utf-8")
    html = md_to_html(md_text)
    html_to_pdf(html, PDF_PATH)
    print(f"PDF generado: {PDF_PATH}")
    print(f"Tamaño: {PDF_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
