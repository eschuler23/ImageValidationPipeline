#!/usr/bin/env python3
"""Convert a Data Card markdown file into print-ready HTML.

This exporter is intentionally dependency-free so it works in offline setups.
It supports the markdown structures used in this project:
- headings (# to ####)
- unordered lists (- item)
- ordered lists (1. item)
- fenced code blocks (``` ... ```)
- paragraphs and inline code (`code`)
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


INLINE_CODE_RE = re.compile(r"`([^`]+)`")
ORDERED_RE = re.compile(r"^\d+\.\s+")


def inline_format(text: str) -> str:
    """Escape text and render inline code spans."""
    escaped = html.escape(text)
    return INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)


def flush_paragraph(lines: list[str], out: list[str]) -> None:
    if not lines:
        return
    joined = " ".join(part.strip() for part in lines if part.strip())
    if joined:
        out.append(f"<p>{inline_format(joined)}</p>")
    lines.clear()


def close_list(state: dict[str, str | None], out: list[str]) -> None:
    list_type = state.get("list")
    if list_type == "ul":
        out.append("</ul>")
    elif list_type == "ol":
        out.append("</ol>")
    state["list"] = None


def markdown_to_html(md_text: str) -> str:
    out: list[str] = []
    paragraph_buffer: list[str] = []
    state: dict[str, str | None] = {"list": None, "code": None}

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # Code-fence handling
        if stripped.startswith("```"):
            flush_paragraph(paragraph_buffer, out)
            close_list(state, out)
            if state["code"] is None:
                state["code"] = "open"
                out.append("<pre><code>")
            else:
                state["code"] = None
                out.append("</code></pre>")
            continue

        if state["code"] is not None:
            out.append(html.escape(line))
            continue

        # Blank line ends paragraph/list context.
        if not stripped:
            flush_paragraph(paragraph_buffer, out)
            close_list(state, out)
            continue

        # Headings
        if stripped.startswith("#"):
            flush_paragraph(paragraph_buffer, out)
            close_list(state, out)
            level = len(stripped) - len(stripped.lstrip("#"))
            if 1 <= level <= 6:
                text = stripped[level:].strip()
                out.append(f"<h{level}>{inline_format(text)}</h{level}>")
                continue

        # Unordered list items
        if stripped.startswith("- "):
            flush_paragraph(paragraph_buffer, out)
            if state["list"] != "ul":
                close_list(state, out)
                out.append("<ul>")
                state["list"] = "ul"
            item = stripped[2:].strip()
            out.append(f"<li>{inline_format(item)}</li>")
            continue

        # Ordered list items
        if ORDERED_RE.match(stripped):
            flush_paragraph(paragraph_buffer, out)
            if state["list"] != "ol":
                close_list(state, out)
                out.append("<ol>")
                state["list"] = "ol"
            item = ORDERED_RE.sub("", stripped, count=1).strip()
            out.append(f"<li>{inline_format(item)}</li>")
            continue

        # Regular paragraph content
        paragraph_buffer.append(line)

    flush_paragraph(paragraph_buffer, out)
    close_list(state, out)

    if state["code"] is not None:
        out.append("</code></pre>")

    return "\n".join(out)


def build_document(body_html: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    @page {{
      size: A4;
      margin: 16mm;
    }}
    :root {{
      --text: #121212;
      --muted: #404040;
      --accent: #0f4c5c;
      --code-bg: #f2f4f7;
      --rule: #d6dbe1;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      color: var(--text);
      background: #ffffff;
      font-family: "Source Serif 4", "Georgia", "Times New Roman", serif;
      line-height: 1.45;
      font-size: 11pt;
    }}
    main {{
      max-width: 180mm;
      margin: 0 auto;
      padding: 0;
    }}
    h1, h2, h3, h4 {{
      line-height: 1.25;
      margin: 0.85em 0 0.35em;
      color: var(--accent);
      break-after: avoid-page;
    }}
    h1 {{
      font-size: 24pt;
      margin-top: 0;
      border-bottom: 2px solid var(--rule);
      padding-bottom: 0.22em;
    }}
    h2 {{
      font-size: 16pt;
      border-bottom: 1px solid var(--rule);
      padding-bottom: 0.18em;
      margin-top: 1.2em;
    }}
    h3 {{ font-size: 13pt; }}
    h4 {{
      font-size: 11.5pt;
      color: var(--muted);
      margin-top: 0.9em;
    }}
    p {{
      margin: 0.25em 0 0.6em;
      text-align: left;
      orphans: 3;
      widows: 3;
    }}
    ul, ol {{
      margin: 0.25em 0 0.7em 1.25em;
      padding: 0;
    }}
    li {{
      margin: 0.16em 0;
    }}
    code {{
      font-family: "SF Mono", "Menlo", "Consolas", monospace;
      background: var(--code-bg);
      border-radius: 3px;
      padding: 0.05em 0.3em;
      font-size: 0.95em;
    }}
    pre {{
      background: var(--code-bg);
      border: 1px solid var(--rule);
      border-radius: 6px;
      padding: 0.65em;
      overflow-x: auto;
      break-inside: avoid-page;
    }}
    pre code {{
      background: transparent;
      padding: 0;
    }}
  </style>
</head>
<body>
  <main>
{body_html}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Data Card markdown to print-ready HTML."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to source markdown file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = args.input.read_text(encoding="utf-8")
    body_html = markdown_to_html(markdown)

    # Use first heading as title when available.
    title = args.input.stem
    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    document = build_document(body_html, title)
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote print-ready HTML: {args.output}")


if __name__ == "__main__":
    main()
