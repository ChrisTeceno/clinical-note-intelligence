#!/usr/bin/env python3
"""Convert Jupyter .ipynb notebook to standalone dark-themed HTML.

Uses the same CSS and layout as convert_databricks_notebook.py so notebooks
rendered from either source look identical in the dashboard.
"""

import argparse
import html as html_lib
import json
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, BashLexer

_py_lexer = PythonLexer()
_bash_lexer = BashLexer()
_formatter = HtmlFormatter(noclasses=True, style="monokai", nowrap=True)

# Reuse the exact same CSS from the Databricks converter
CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1b1b2f;
  color: #e0e0e8;
  line-height: 1.6;
}
.nb-header {
  background: #2d2d44;
  border-bottom: 2px solid #3a3a55;
  padding: 10px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  position: sticky;
  top: 0;
  z-index: 100;
}
.nb-header .logo {
  font-weight: 800;
  color: #ff6f61;
  font-size: 13px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}
.nb-header .sep { color: #555; font-size: 18px; }
.nb-header .name { color: #ccc; font-size: 14px; font-weight: 500; }
.nb-header .badge {
  background: #2ecc71;
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 10px;
  margin-left: auto;
}
.nb-header .cell-count {
  color: #888;
  font-size: 12px;
}
.nb-container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px 60px;
}
.cell-md {
  padding: 20px 28px;
  margin-bottom: 8px;
}
.cell-md h1 { font-size: 26px; font-weight: 700; color: #fff; margin: 16px 0 10px; }
.cell-md h2 { font-size: 21px; font-weight: 600; color: #ddd; margin: 20px 0 8px; border-bottom: 1px solid #3a3a55; padding-bottom: 6px; }
.cell-md h3 { font-size: 17px; font-weight: 600; color: #ccc; margin: 14px 0 6px; }
.cell-md p { color: #bbb; margin-bottom: 8px; }
.cell-md ul, .cell-md ol { color: #bbb; margin-left: 24px; margin-bottom: 8px; }
.cell-md li { margin-bottom: 3px; }
.cell-md strong { color: #e0e0e8; }
.cell-md code { background: #2d2d44; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #e8b86d; }
.cell-md a { color: #64b5f6; }
.cell-md hr { border: none; border-top: 1px solid #3a3a55; margin: 16px 0; }
.cell-md table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.cell-md th { background: #2d2d44; color: #aaa; text-align: left; padding: 8px 14px; border: 1px solid #3a3a55; font-size: 13px; font-weight: 600; }
.cell-md td { padding: 8px 14px; border: 1px solid #3a3a55; color: #ccc; font-size: 13px; }
.cell-md pre { background: #1e1e32; padding: 12px 16px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
.cell-md pre code { background: transparent; padding: 0; color: #ccc; }
.cell-code {
  background: #1e1e32;
  border: 1px solid #3a3a55;
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
}
.cell-toolbar {
  background: #262640;
  padding: 6px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid #3a3a55;
  font-size: 12px;
}
.cell-num {
  color: #666;
  font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px;
  min-width: 32px;
}
.cell-lang {
  color: #888;
  font-size: 11px;
  margin-left: auto;
  background: #2d2d44;
  padding: 2px 10px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.status-ok { color: #2ecc71; font-size: 14px; }
.status-none { color: #555; }
.cell-source pre {
  padding: 16px 20px;
  overflow-x: auto;
  font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
  font-size: 13px;
  line-height: 1.6;
  background: transparent;
  white-space: pre;
  tab-size: 4;
}
.cell-output {
  border-top: 1px solid #3a3a55;
  background: #191930;
  max-height: 350px;
  overflow-y: auto;
}
.cell-output pre {
  padding: 12px 20px;
  font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px;
  line-height: 1.5;
  color: #aaa;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.cell-output-table { padding: 0; overflow-x: auto; }
.cell-output-table table { border-collapse: collapse; width: 100%; font-size: 12px; }
.cell-output-table th {
  background: #232340; color: #aaa; text-align: left; padding: 8px 12px;
  border: 1px solid #3a3a55; font-weight: 600; position: sticky; top: 0;
}
.cell-output-table td {
  padding: 6px 12px; border: 1px solid #2d2d44; color: #ccc;
  max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cell-output-table tr:nth-child(even) td { background: #1d1d35; }
.cell-output-table tr:hover td { background: #252545; }
.cell-output img { max-width: 100%; border-radius: 6px; margin: 8px 0; }
"""


def render_markdown(source: str) -> str:
    rendered = markdown.markdown(source, extensions=["tables", "fenced_code"])
    return f'<div class="cell cell-md">{rendered}</div>'


def render_code(source: str, cell_number: int) -> str:
    is_magic = source.lstrip().startswith(("%pip", "%sh", "!"))
    lexer = _bash_lexer if is_magic else _py_lexer
    lang = "shell" if is_magic else "python"
    highlighted = highlight(source, lexer, _formatter)

    return (
        f'<div class="cell cell-code">'
        f'  <div class="cell-toolbar">'
        f'    <span class="cell-num">[{cell_number}]</span>'
        f'    <span class="status-none">&#8212;</span>'
        f'    <span class="cell-lang">{lang}</span>'
        f'  </div>'
        f'  <div class="cell-source"><pre>{highlighted}</pre></div>'
        f'</div>'
    )


def convert(ipynb_path: str, output_path: str, name: str | None = None) -> None:
    nb = json.loads(Path(ipynb_path).read_text(encoding="utf-8"))
    nb_name = name or Path(ipynb_path).stem

    cells_html = []
    code_num = 0

    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if not source.strip():
            continue

        if cell["cell_type"] == "markdown":
            cells_html.append(render_markdown(source))
        elif cell["cell_type"] == "code":
            code_num += 1
            cells_html.append(render_code(source, code_num))

    cells_joined = "\n".join(cells_html)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_lib.escape(nb_name)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="nb-header">
  <span class="logo">Databricks</span>
  <span class="sep">/</span>
  <span class="name">{html_lib.escape(nb_name)}</span>
  <span class="cell-count">{len(cells_html)} cells</span>
  <span class="badge">Notebook</span>
</div>
<div class="nb-container">
{cells_joined}
</div>
</body>
</html>"""

    Path(output_path).write_text(html_doc, encoding="utf-8")
    print(f"Converted {nb_name}: {len(cells_html)} cells -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert .ipynb to dark-themed HTML")
    parser.add_argument("input", help="Path to .ipynb file")
    parser.add_argument("output", help="Path for output HTML file")
    parser.add_argument("--name", help="Override notebook display name")
    args = parser.parse_args()
    convert(args.input, args.output, args.name)


if __name__ == "__main__":
    main()
