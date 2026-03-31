#!/usr/bin/env python3
"""Convert Databricks HTML notebook export to standalone dark-themed HTML."""

import argparse
import base64
import html as html_lib
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, BashLexer


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_notebook_json(html_path: str) -> dict:
    """Extract and decode the notebook JSON from Databricks HTML export."""
    content = Path(html_path).read_text(encoding="utf-8")
    marker = "var __DATABRICKS_NOTEBOOK_MODEL = '"
    start = content.index(marker) + len(marker)
    end = content.index("';", start)
    encoded = content[start:end]
    url_encoded = base64.b64decode(encoded).decode("utf-8")
    return json.loads(urllib.parse.unquote(url_encoded))


# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------

@dataclass
class CellInfo:
    cell_type: str  # "markdown", "python", "pip", "empty"
    source: str
    state: str
    results: list = field(default_factory=list)
    duration_ms: int = 0
    collapsed: bool = False
    hide_code: bool = False
    hide_result: bool = False


def classify_cell(command: dict) -> CellInfo:
    source = command.get("command", "")
    state = command.get("state", "")
    results_obj = command.get("results")
    results = []
    if results_obj and isinstance(results_obj, dict):
        results = results_obj.get("data", [])

    start_t = command.get("startTime", 0)
    finish_t = command.get("finishTime", 0)
    duration = finish_t - start_t if start_t and finish_t else 0

    if not source.strip() and not results:
        cell_type = "empty"
    elif source.startswith("%md"):
        cell_type = "markdown"
    elif source.startswith("%pip") or source.startswith("%sh"):
        cell_type = "pip"
    else:
        cell_type = "python"

    return CellInfo(
        cell_type=cell_type,
        source=source,
        state=state,
        results=results,
        duration_ms=duration,
        collapsed=command.get("collapsed", False),
        hide_code=command.get("hideCommandCode", False),
        hide_result=command.get("hideCommandResult", False),
    )


# ---------------------------------------------------------------------------
# ANSI stripping
# ---------------------------------------------------------------------------

def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[hl]|\x1b\[2K|\r", "", text)


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

def format_duration(ms: int) -> str:
    if ms <= 0:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f}s"
    mins = int(secs // 60)
    remaining = secs % 60
    return f"{mins}m {remaining:.0f}s"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_py_lexer = PythonLexer()
_bash_lexer = BashLexer()
_formatter = HtmlFormatter(noclasses=True, style="monokai", nowrap=True)


def render_markdown_cell(cell: CellInfo) -> str:
    source = cell.source
    # Strip %md prefix
    if source.startswith("%md\n"):
        source = source[4:]
    elif source.startswith("%md"):
        source = source[3:]
    # Strip leading ---
    source = source.strip()

    rendered = markdown.markdown(
        source,
        extensions=["tables", "fenced_code"],
    )
    return f'<div class="cell cell-md">{rendered}</div>'


def render_code_cell(cell: CellInfo, cell_number: int) -> str:
    source = cell.source
    is_pip = cell.cell_type == "pip"

    # Status indicator
    if cell.state == "finished":
        status_html = '<span class="status-ok">&#10003;</span>'
    elif cell.state in ("error", "failed"):
        status_html = '<span class="status-err">&#10007;</span>'
    else:
        status_html = '<span class="status-none">&#8212;</span>'

    duration = format_duration(cell.duration_ms)
    duration_html = f'<span class="cell-duration">{duration}</span>' if duration else ""

    lang_label = "shell" if is_pip else "python"
    lexer = _bash_lexer if is_pip else _py_lexer

    # Syntax highlight
    highlighted = highlight(source, lexer, _formatter)

    parts = [
        f'<div class="cell cell-code">',
        f'  <div class="cell-toolbar">',
        f'    <span class="cell-num">[{cell_number}]</span>',
        f'    {status_html}',
        f'    {duration_html}',
        f'    <span class="cell-lang">{lang_label}</span>',
        f'  </div>',
        f'  <div class="cell-source"><pre>{highlighted}</pre></div>',
    ]

    # Render outputs
    if cell.results and not cell.hide_result:
        for item in cell.results:
            parts.append(render_result_item(item))

    parts.append("</div>")
    return "\n".join(parts)


def render_result_item(item: dict) -> str:
    item_type = item.get("type", "")
    data = item.get("data", "")

    if item_type == "ansi":
        cleaned = strip_ansi(str(data))
        escaped = html_lib.escape(cleaned).strip()
        if not escaped:
            return ""
        return f'<div class="cell-output"><pre>{escaped}</pre></div>'

    elif item_type == "table":
        return render_table(item)

    elif item_type in ("html", "htmlSandbox"):
        return f'<div class="cell-output cell-output-html">{data}</div>'

    elif item_type == "text":
        escaped = html_lib.escape(str(data)).strip()
        if not escaped:
            return ""
        return f'<div class="cell-output"><pre>{escaped}</pre></div>'

    else:
        # Unknown type — render as text
        if data:
            escaped = html_lib.escape(str(data)).strip()
            if escaped:
                return f'<div class="cell-output"><pre>{escaped}</pre></div>'
        return ""


def render_table(item: dict) -> str:
    schema = item.get("schema", [])
    rows = item.get("data", [])

    if not schema:
        return ""

    # Column names
    col_names = []
    for col in schema:
        if isinstance(col, dict):
            col_names.append(col.get("name", ""))
        else:
            col_names.append(str(col))

    parts = ['<div class="cell-output cell-output-table"><table>']
    parts.append("<thead><tr>")
    for name in col_names:
        parts.append(f"<th>{html_lib.escape(str(name))}</th>")
    parts.append("</tr></thead>")

    parts.append("<tbody>")
    for row in rows[:50]:  # Limit to 50 rows
        parts.append("<tr>")
        if isinstance(row, (list, tuple)):
            for val in row:
                cell_val = format_table_value(val)
                parts.append(f"<td>{cell_val}</td>")
        elif isinstance(row, dict):
            for name in col_names:
                val = row.get(name, "")
                cell_val = format_table_value(val)
                parts.append(f"<td>{cell_val}</td>")
        parts.append("</tr>")

    if len(rows) > 50:
        parts.append(
            f'<tr><td colspan="{len(col_names)}" class="truncated">'
            f"... {len(rows) - 50} more rows</td></tr>"
        )

    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def format_table_value(val) -> str:
    if val is None:
        return '<span class="null">null</span>'
    if isinstance(val, list):
        items = ", ".join(str(v) for v in val[:5])
        if len(val) > 5:
            items += f", ... +{len(val) - 5}"
        return html_lib.escape(f"[{items}]")
    s = str(val)
    if len(s) > 150:
        s = s[:147] + "..."
    return html_lib.escape(s)


# ---------------------------------------------------------------------------
# Full HTML document
# ---------------------------------------------------------------------------

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
/* Markdown cells */
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
/* Code cells */
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
.cell-duration { color: #888; font-size: 11px; }
.status-ok { color: #2ecc71; font-size: 14px; }
.status-err { color: #e74c3c; font-size: 14px; }
.status-none { color: #555; }
.cell-source {
  padding: 0;
}
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
/* Output cells */
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
.cell-output-table {
  padding: 0;
  overflow-x: auto;
}
.cell-output-table table {
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}
.cell-output-table th {
  background: #232340;
  color: #aaa;
  text-align: left;
  padding: 8px 12px;
  border: 1px solid #3a3a55;
  font-weight: 600;
  position: sticky;
  top: 0;
}
.cell-output-table td {
  padding: 6px 12px;
  border: 1px solid #2d2d44;
  color: #ccc;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cell-output-table tr:nth-child(even) td { background: #1d1d35; }
.cell-output-table tr:hover td { background: #252545; }
.cell-output-table .truncated { color: #666; font-style: italic; text-align: center; }
.cell-output-table .null { color: #666; font-style: italic; }
"""


def build_html(notebook_name: str, cells_html: list[str], cell_count: int) -> str:
    cells_joined = "\n".join(cells_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_lib.escape(notebook_name)} — Databricks Notebook</title>
<style>{CSS}</style>
</head>
<body>
<div class="nb-header">
  <span class="logo">Databricks</span>
  <span class="sep">/</span>
  <span class="name">{html_lib.escape(notebook_name)}</span>
  <span class="cell-count">{cell_count} cells</span>
  <span class="badge">Serverless</span>
</div>
<div class="nb-container">
{cells_joined}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Databricks HTML notebook export to standalone dark-themed HTML"
    )
    parser.add_argument("input", help="Path to Databricks HTML export file")
    parser.add_argument("output", help="Path for output HTML file")
    parser.add_argument("--name", help="Override notebook name")
    args = parser.parse_args()

    notebook = extract_notebook_json(args.input)
    name = args.name or notebook.get("name", "Untitled")
    commands = notebook.get("commands", [])

    cells_html = []
    code_cell_num = 0

    for command in commands:
        cell = classify_cell(command)
        if cell.cell_type == "empty":
            continue
        elif cell.cell_type == "markdown":
            cells_html.append(render_markdown_cell(cell))
        elif cell.cell_type in ("python", "pip"):
            code_cell_num += 1
            cells_html.append(render_code_cell(cell, code_cell_num))

    output_html = build_html(name, cells_html, len(cells_html))
    Path(args.output).write_text(output_html, encoding="utf-8")
    print(f"Converted {name}: {len(cells_html)} cells -> {args.output}")


if __name__ == "__main__":
    main()
