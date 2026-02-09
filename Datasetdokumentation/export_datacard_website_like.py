#!/usr/bin/env python3
"""Export markdown into a Card Builder-like HTML document.

This script is dependency-free and designed for offline use in this repository.
It recreates the Data Card DOM structure (sections/subsections/fields) and
compiles the local `src/styles/default.scss` into flat CSS for browser/PDF use.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


ICON_INFO = (
    '<svg aria-hidden="true" class="tooltip-icon" fill="#3c4f50" '
    'viewBox="0 0 24 24" height="18px" width="18px" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<path d="M11,7h2v2h-2V7z M11,11h2v6h-2V11z M12,2C6.48,2,2,6.48,2,12s4.48,10,10,10s10-4.48,10-10S17.52,2,12,2z M12,20 c-4.41,0-8-3.59-8-8s3.59-8,8-8s8,3.59,8,8S16.41,20,12,20z"></path>'
    "</svg>"
)

ICON_EXPAND = (
    '<svg aria-hidden="true" fill="#3c4f50" viewBox="0 0 24 24" '
    'height="24px" width="24px" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M0 0h24v24H0z" fill="none"></path>'
    '<path d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z"></path>'
    "</svg>"
)

INLINE_CODE_RE = re.compile(r"`([^`]+)`")
ORDERED_RE = re.compile(r"^\d+\.\s+")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
VAR_RE = re.compile(r"^\s*\$([-\w]+)\s*:\s*(.+?)\s*;\s*$", re.MULTILINE)


@dataclass
class FieldNode:
    title: str
    lines: list[str] = field(default_factory=list)


@dataclass
class SubsectionNode:
    title: str
    lines: list[str] = field(default_factory=list)
    fields: list[FieldNode] = field(default_factory=list)


@dataclass
class SectionNode:
    title: str
    lines: list[str] = field(default_factory=list)
    subsections: list[SubsectionNode] = field(default_factory=list)


@dataclass
class CardNode:
    title: str
    lines: list[str] = field(default_factory=list)
    sections: list[SectionNode] = field(default_factory=list)


@dataclass
class ScssBlock:
    header: str
    children: list[Union[str, "ScssBlock"]]


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    return INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)


def markdown_fragment_to_html(lines: list[str]) -> str:
    out: list[str] = []
    paragraph: list[str] = []
    in_code = False
    list_state: str | None = None

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph if part.strip())
        if text:
            out.append(f"<p>{inline_markdown(text)}</p>")
        paragraph.clear()

    def close_list() -> None:
        nonlocal list_state
        if list_state == "ul":
            out.append("</ul>")
        elif list_state == "ol":
            out.append("</ol>")
        list_state = None

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            if not in_code:
                in_code = True
                out.append("<pre><code>")
            else:
                in_code = False
                out.append("</code></pre>")
            continue

        if in_code:
            out.append(html.escape(line))
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if list_state != "ul":
                close_list()
                out.append("<ul>")
                list_state = "ul"
            out.append(f"<li>{inline_markdown(stripped[2:].strip())}</li>")
            continue

        if ORDERED_RE.match(stripped):
            flush_paragraph()
            if list_state != "ol":
                close_list()
                out.append("<ol>")
                list_state = "ol"
            item = ORDERED_RE.sub("", stripped, count=1).strip()
            out.append(f"<li>{inline_markdown(item)}</li>")
            continue

        paragraph.append(line)

    flush_paragraph()
    close_list()
    if in_code:
        out.append("</code></pre>")
    return "".join(out)


def slug_from_title(title: str) -> str:
    return "-".join(title.strip().lower().split())


def parse_card_markdown(markdown: str) -> CardNode:
    card = CardNode(title="")
    current_section: SectionNode | None = None
    current_subsection: SubsectionNode | None = None
    current_field: FieldNode | None = None

    for raw_line in markdown.splitlines():
        heading = HEADING_RE.match(raw_line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()

            if level == 1:
                card.title = title
                current_section = None
                current_subsection = None
                current_field = None
            elif level == 2:
                section = SectionNode(title=title)
                card.sections.append(section)
                current_section = section
                current_subsection = None
                current_field = None
            elif level == 3:
                if current_section is None:
                    current_section = SectionNode(title="Section")
                    card.sections.append(current_section)
                subsection = SubsectionNode(title=title)
                current_section.subsections.append(subsection)
                current_subsection = subsection
                current_field = None
            elif level == 4:
                if current_subsection is None:
                    if current_section is None:
                        current_section = SectionNode(title="Section")
                        card.sections.append(current_section)
                    current_subsection = SubsectionNode(title="Subsection")
                    current_section.subsections.append(current_subsection)
                field_node = FieldNode(title=title)
                current_subsection.fields.append(field_node)
                current_field = field_node
            continue

        target_lines: list[str]
        if current_field is not None:
            target_lines = current_field.lines
        elif current_subsection is not None:
            target_lines = current_subsection.lines
        elif current_section is not None:
            target_lines = current_section.lines
        else:
            target_lines = card.lines
        target_lines.append(raw_line)

    return card


def render_field(field_node: FieldNode) -> str:
    body = markdown_fragment_to_html(field_node.lines)
    return (
        '<div class="card-field">'
        '<div class="card-field-top">'
        f'<h4 class="card-field-title" id="{html.escape(slug_from_title(field_node.title))}">{html.escape(field_node.title)}</h4>'
        "</div>"
        '<div class="card-field-body">'
        f'<div class="card-content-wrapper">{body}</div>'
        "</div>"
        "</div>"
    )


def render_subsection(subsection: SubsectionNode) -> str:
    body_parts: list[str] = []
    if subsection.lines:
        body_parts.append(
            f'<div class="card-content-wrapper">{markdown_fragment_to_html(subsection.lines)}</div>'
        )

    if subsection.fields:
        fields = "".join(render_field(field_node) for field_node in subsection.fields)
        body_parts.append(f'<div class="card-field-wrapper">{fields}</div>')

    body_html = "".join(body_parts)
    return (
        '<div class="card-subsection">'
        '<div class="card-subsection-top">'
        f'<h3 class="card-subsection-title" id="{html.escape(slug_from_title(subsection.title))}">{html.escape(subsection.title)}</h3>'
        "</div>"
        f'<div class="card-subsection-body">{body_html}</div>'
        "</div>"
    )


def render_section(section: SectionNode, open_by_default: bool = True) -> str:
    subtitle = ", ".join(sub.title for sub in section.subsections)
    tooltip_html = ""
    if subtitle:
        tooltip_html = (
            '<div class="tooltip">'
            f"{ICON_INFO}"
            f'<div class="tooltip-text">{html.escape(subtitle)}</div>'
            "</div>"
        )

    subsection_html = "".join(render_subsection(sub) for sub in section.subsections)
    body_parts: list[str] = []
    if section.lines:
        body_parts.append(
            f'<div class="card-content-wrapper">{markdown_fragment_to_html(section.lines)}</div>'
        )
    if subsection_html:
        body_parts.append(f'<div class="card-subsection-wrapper">{subsection_html}</div>')

    classes = "card-section open" if open_by_default else "card-section"
    return (
        f'<section class="{classes}">'
        '<div class="card-section-top">'
        '<div class="card-section-title-wrapper">'
        f'<h2 class="card-section-title" id="{html.escape(slug_from_title(section.title))}">{html.escape(section.title)}</h2>'
        f"{tooltip_html}"
        "</div>"
        f'<div class="icon-wrapper expand-more">{ICON_EXPAND}</div>'
        "</div>"
        f'<div class="card-section-body">{"".join(body_parts)}</div>'
        "</section>"
    )


def render_card(card: CardNode, open_by_default: bool = True) -> str:
    summary = markdown_fragment_to_html(card.lines)
    sections = "".join(render_section(section, open_by_default) for section in card.sections)
    return (
        '<div class="datacard-wrapper">'
        '<article class="datacard" id="datacard-preview">'
        '<section class="card-summary">'
        '<div class="datacard-header">'
        f'<h1 class="datacard-title" id="{html.escape(slug_from_title(card.title))}">{html.escape(card.title)}</h1>'
        f'<div class="card-content-wrapper">{summary}</div>'
        "</div>"
        "</section>"
        f"{sections}"
        "</article>"
        "</div>"
    )


def strip_mixin_block(scss: str, mixin_name: str) -> str:
    marker = f"@mixin {mixin_name}"
    start = scss.find(marker)
    if start < 0:
        return scss

    brace_start = scss.find("{", start)
    if brace_start < 0:
        return scss

    depth = 0
    idx = brace_start
    while idx < len(scss):
        if scss[idx] == "{":
            depth += 1
        elif scss[idx] == "}":
            depth -= 1
            if depth == 0:
                return scss[:start] + scss[idx + 1 :]
        idx += 1
    return scss


def parse_scss_items(source: str, start: int = 0) -> tuple[list[Union[str, ScssBlock]], int]:
    items: list[Union[str, ScssBlock]] = []
    token = []
    i = start
    paren_depth = 0
    quote_char = ""
    escaped = False
    while i < len(source):
        ch = source[i]

        if quote_char:
            token.append(ch)
            if ch == quote_char and not escaped:
                quote_char = ""
            escaped = (ch == "\\") and not escaped
            i += 1
            continue

        if ch in ("'", '"'):
            quote_char = ch
            token.append(ch)
            i += 1
            continue

        if ch == "(":
            paren_depth += 1
            token.append(ch)
            i += 1
            continue

        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            token.append(ch)
            i += 1
            continue

        if ch == "{" and paren_depth == 0:
            header = "".join(token).strip()
            token = []
            children, i = parse_scss_items(source, i + 1)
            items.append(ScssBlock(header=header, children=children))
            continue
        if ch == "}" and paren_depth == 0:
            stmt = "".join(token).strip()
            if stmt:
                items.append(stmt)
            return items, i + 1
        if ch == ";" and paren_depth == 0:
            token.append(ch)
            stmt = "".join(token).strip()
            if stmt:
                items.append(stmt)
            token = []
            i += 1
            continue
        token.append(ch)
        i += 1

    stmt = "".join(token).strip()
    if stmt:
        items.append(stmt)
    return items, i


def split_selectors(selector_text: str) -> list[str]:
    parts = []
    current = []
    depth = 0
    for ch in selector_text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = normalize_space("".join(current))
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    part = normalize_space("".join(current))
    if part:
        parts.append(part)
    return parts


def combine_selectors(parents: list[str], children: list[str]) -> list[str]:
    if not parents:
        return children

    combined: list[str] = []
    for parent in parents:
        for child in children:
            if "&" in child:
                combined.append(normalize_space(child.replace("&", parent)))
            else:
                combined.append(normalize_space(f"{parent} {child}"))
    return combined


def emit_css_items(
    items: list[Union[str, ScssBlock]],
    selectors: list[str] | None = None,
    at_rules: list[str] | None = None,
) -> str:
    selectors = selectors or []
    at_rules = at_rules or []
    out: list[str] = []

    declarations: list[str] = []
    children: list[ScssBlock] = []
    top_statements: list[str] = []

    for item in items:
        if isinstance(item, ScssBlock):
            children.append(item)
        else:
            text = normalize_space(item)
            if not text:
                continue
            if selectors:
                if not text.endswith(";"):
                    text += ";"
                declarations.append(text)
            else:
                top_statements.append(text if text.endswith(";") else f"{text};")

    out.extend(top_statements)

    if selectors and declarations:
        block = f"{', '.join(selectors)} {{{' '.join(declarations)}}}"
        for at_rule in reversed(at_rules):
            block = f"{at_rule} {{{block}}}"
        out.append(block)

    for child in children:
        header = normalize_space(child.header)
        if not header:
            continue

        if header.startswith("@media"):
            out.append(
                emit_css_items(
                    child.children,
                    selectors=selectors,
                    at_rules=[*at_rules, header],
                )
            )
            continue

        if header.startswith("@"):
            # Unsupported nested at-rule for this stylesheet.
            continue

        child_selectors = combine_selectors(selectors, split_selectors(header))
        out.append(
            emit_css_items(
                child.children,
                selectors=child_selectors,
                at_rules=at_rules,
            )
        )

    return "\n".join(part for part in out if part.strip())


def compile_scss_to_css(scss_path: Path) -> str:
    source = scss_path.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = strip_mixin_block(source, "viewport-small")

    variables = {match.group(1): match.group(2).strip() for match in VAR_RE.finditer(source)}
    source = VAR_RE.sub("", source)

    for key, value in variables.items():
        clean = value.strip().strip('"').strip("'")
        source = re.sub(
            rf"#\{{\s*\${re.escape(key)}\s*\}}",
            clean,
            source,
        )

    source = source.replace(
        "@include viewport-small {", "@media screen and (max-width: 600px) {"
    )

    for key in sorted(variables, key=len, reverse=True):
        source = re.sub(rf"\${re.escape(key)}\b", variables[key], source)

    items, _ = parse_scss_items(source, 0)
    css = emit_css_items(items)

    print_overrides = """
@media print {
  .datacard {
    overflow: visible !important;
    max-width: none !important;
    box-shadow: none !important;
  }
  .card-section-body {
    height: max-content !important;
    overflow: visible !important;
  }
  .icon-wrapper,
  .tooltip {
    display: none !important;
  }
}
"""
    return css + "\n" + print_overrides


def build_document(card_html: str, css: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
{css}
  </style>
</head>
<body>
{card_html}
<script>
for (const icon of document.getElementsByClassName('expand-more')) {{
  icon.addEventListener('click', (e) => {{
    const section = e.currentTarget.parentElement.parentElement;
    section.classList.toggle('open');
  }});
}}
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export markdown as website-like Data Card HTML.")
    parser.add_argument("--input", required=True, type=Path, help="Input markdown file.")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML file.")
    parser.add_argument(
        "--scss",
        type=Path,
        default=Path(__file__).parent / "src/styles/default.scss",
        help="Path to card SCSS style file.",
    )
    parser.add_argument(
        "--collapse",
        action="store_true",
        help="Render sections collapsed by default (website-like interaction mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = args.input.read_text(encoding="utf-8")
    card = parse_card_markdown(markdown)
    if not card.title:
        raise SystemExit("Input markdown must start with an H1 title.")

    card_html = render_card(card, open_by_default=not args.collapse)
    css = compile_scss_to_css(args.scss)
    document = build_document(card_html, css, card.title)
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote Data Card HTML: {args.output}")


if __name__ == "__main__":
    main()
