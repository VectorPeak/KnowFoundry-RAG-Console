"""Build static HTML documentation from docs/ markdown lecture files.

Reads mkdocs.yml for nav structure and markdown extension config,
converts every .md file in docs/ to .html under static/docs/,
and wraps each page in a template with sidebar navigation.

Usage:  python scripts/build_docs.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import mermaid_formatter
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from markdown import Markdown
from mermaid_formatter import mermaid_div_format


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = PROJECT_ROOT / "static" / "docs"
MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"

# ---------------------------------------------------------------------------
# Markdown converter setup (mirrors mkdocs.yml config)
# ---------------------------------------------------------------------------

def make_markdown_converter() -> Markdown:
    return Markdown(
        extensions=[
            "admonition",
            "codehilite",
            "toc",
            "tables",
            "pymdownx.highlight",
            "pymdownx.inlinehilite",
            "pymdownx.snippets",
            "pymdownx.superfences",
            "pymdownx.arithmatex",
        ],
        extension_configs={
            "toc": {
                "permalink": True,
                "toc_depth": 3,
            },
            "pymdownx.highlight": {
                "anchor_linenums": True,
                "linenums": True,
            },
            "pymdownx.superfences": {
                "custom_fences": [
                    {
                        "name": "mermaid",
                        "class": "mermaid",
                        "format": mermaid_div_format,
                    }
                ],
            },
            "pymdownx.arithmatex": {
                "generic": True,
            },
        },
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def parse_nav(mkdocs_path: Path) -> list[dict]:
    """Parse mkdocs.yml nav into a flat ordered list of {title, path} dicts.

    The nav in mkdocs.yml is a list of mixed entries:
      - "首页: index.md"        → str
      - "第一阶段：基础概念:"   → dict with list value
    """
    with open(mkdocs_path, encoding="utf-8") as f:
        config = yaml.unsafe_load(f)
    return config.get("nav", [])


def flatten_nav(nav_entries: list) -> list[dict]:
    """Flatten nested nav structure into ordered list of {title, path, section}."""
    pages: list[dict] = []

    def walk(entries: list, section: str = ""):
        for entry in entries:
            if isinstance(entry, str):
                # "首页: index.md" or bare path
                if ": " in entry:
                    title, path = entry.split(": ", 1)
                else:
                    title = entry
                    path = entry
                pages.append({"title": title.strip(), "path": path.strip(), "section": section})
            elif isinstance(entry, dict):
                for key, value in entry.items():
                    if isinstance(value, str):
                        pages.append({"title": key.strip(), "path": value.strip(), "section": section})
                    elif isinstance(value, list):
                        walk(value, section=key.strip())
                    else:
                        pages.append({"title": key.strip(), "path": str(value), "section": section})

    walk(nav_entries)
    return pages


def build_nav_html(page_list: list[dict], current_path: str) -> str:
    """Build sidebar navigation HTML from flattened page list.

    Groups pages by their `section` field.  `current_path` is the
    relative path of the page being rendered (from docs/ root) with
    .html extension already.
    """
    # Group by section
    sections: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    for p in page_list:
        sec = p.get("section", "")
        if sec not in seen:
            seen.add(sec)
            sections.append((sec, [p]))
        else:
            for s_name, s_pages in sections:
                if s_name == sec:
                    s_pages.append(p)
                    break

    parts: list[str] = ['<ul class="nav-list">']

    for sec_name, sec_pages in sections:
        if sec_name:
            parts.append(
                '<li class="nav-section">'
                f'<span class="nav-section-title">{_escape(sec_name)}</span>'
                '<ul class="nav-list">'
            )
        for p in sec_pages:
            href = "/docs/" + p["path"].replace(".md", ".html")
            label = _escape(p["title"])
            cls = 'nav-link active' if p["path"].replace(".md", ".html") == current_path else 'nav-link'
            parts.append(f'<li class="nav-item"><a class="{cls}" href="{href}">{label}</a></li>')
        if sec_name:
            parts.append("</ul></li>")

    parts.append("</ul>")
    return "\n".join(parts)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Link rewriting
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r'href="([^"]*)"')


def _rewrite_link(match: re.Match) -> str:
    href = match.group(1)
    # Skip external URLs, mailto, and pure anchors
    if href.startswith(("http://", "https://", "mailto:", "#")):
        return match.group(0)
    # Handle fragment anchors on .md files: file.md#anchor → file.html#anchor
    if "#" in href:
        base, anchor = href.split("#", 1)
        if base.endswith(".md"):
            return f'href="{base[:-3]}.html#{anchor}"'
        return match.group(0)
    if href.endswith(".md"):
        return f'href="{href[:-3]}.html"'
    return match.group(0)


def fix_internal_links(html: str) -> str:
    return _LINK_RE.sub(_rewrite_link, html)


# ---------------------------------------------------------------------------
# Prev/Next computation
# ---------------------------------------------------------------------------

def build_prev_next(page_list: list[dict]) -> dict[str, dict]:
    """Return {rel_path_without_ext: {prev: {title,path}|None, next: ...}}."""
    result: dict[str, dict] = {}
    paths = [p["path"].replace(".md", "") for p in page_list]
    titles = [p["title"] for p in page_list]
    for i, key in enumerate(paths):
        prev_info = None
        next_info = None
        if i > 0:
            prev_info = {"title": titles[i - 1], "path": paths[i - 1] + ".html"}
        if i < len(paths) - 1:
            next_info = {"title": titles[i + 1], "path": paths[i + 1] + ".html"}
        result[key] = {"prev": prev_info, "next": next_info}
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def render_page(
    *,
    title: str,
    content_html: str,
    nav_html: str,
    prev_link: dict | None,
    next_link: dict | None,
    current_rel: str,
) -> str:
    """Render a complete HTML page."""

    prev_block = ""
    if prev_link:
        prev_block = (
            f'<a class="pager-link prev" href="/docs/{prev_link["path"]}">'
            f'<i class="fas fa-arrow-left"></i> {_escape(prev_link["title"])}</a>'
        )

    next_block = ""
    if next_link:
        next_block = (
            f'<a class="pager-link next" href="/docs/{next_link["path"]}">'
            f'{_escape(next_link["title"])} <i class="fas fa-arrow-right"></i></a>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(title)} — KnowForge RAG Platform 讲义</title>
<script src="/static/js/vendor/mermaid.min.js"></script>
<script>if (window.mermaid) {{ mermaid.initialize({{startOnLoad:false,theme:"default"}}); }}</script>
<script>
window.MathJax = {{
  loader: {{ paths: {{ mathjax: "/static/js/vendor/mathjax" }} }},
  options: {{ enableAssistiveMml: false }}
}};
</script>
<script src="/static/js/vendor/mathjax/tex-mml-chtml.js"></script>
<link rel="stylesheet" href="/static/css/fontawesome-shim.css?v=1">
<link rel="stylesheet" href="/static/css/doc.css?v=3">
</head>
<body>
<div class="doc-shell">
  <header class="doc-topbar">
    <button class="sidebar-toggle" aria-label="切换侧边栏"><i class="fas fa-bars"></i></button>
    <a class="doc-brand" href="/docs/index.html"><i class="fas fa-layer-group"></i><span>KnowForge 讲义</span></a>
    <a class="btn btn-ghost" href="/"><i class="fas fa-comment-dots"></i> 进入问答</a>
  </header>
  <div class="doc-layout">
    <aside class="doc-sidebar" id="sidebar">
      <nav class="sidebar-nav">
{nav_html}
      </nav>
    </aside>
    <main class="doc-content">
{content_html}
      <div class="doc-pager">
        {prev_block}
        {next_block}
      </div>
    </main>
  </div>
</div>
<button class="doc-back-to-top" id="backToTop" title="回到顶部"><i class="fas fa-arrow-up"></i></button>
<script>
(function() {{
  // Mobile sidebar toggle
  document.querySelector(".sidebar-toggle").addEventListener("click", function() {{
    document.getElementById("sidebar").classList.toggle("open");
  }});
  // Close sidebar on nav link click (mobile)
  document.querySelectorAll(".nav-link").forEach(function(el) {{
    el.addEventListener("click", function() {{
      document.getElementById("sidebar").classList.remove("open");
    }});
  }});
  // Back-to-top button
  var btn = document.getElementById("backToTop");
  window.addEventListener("scroll", function() {{
    btn.classList.toggle("visible", window.scrollY > 400);
  }});
  btn.addEventListener("click", function() {{ window.scrollTo({{top:0,behavior:"smooth"}}); }});
  // Render mermaid diagrams
  if (window.mermaid) {{
    mermaid.run({{querySelector:".mermaid"}});
  }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Building documentation HTML from docs/")
    print("=" * 60)

    # 1. Parse nav and build page list
    nav_entries = parse_nav(MKDOCS_YML)
    page_list = flatten_nav(nav_entries)
    nav_paths = {p["path"].replace(".md", "") for p in page_list}

    prev_next = build_prev_next(page_list)
    md_converter = make_markdown_converter()

    # 2. Collect all .md files under docs/
    md_files = sorted(DOCS_DIR.rglob("*.md"))
    print(f"\nFound {len(md_files)} markdown files\n")

    processed = 0
    errors: list[tuple[str, str]] = []

    for md_abs in md_files:
        md_rel = md_abs.relative_to(DOCS_DIR)
        out_rel = md_rel.with_suffix(".html")
        out_abs = OUTPUT_DIR / out_rel

        # Ensure output dir exists
        out_abs.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Read source
            with open(md_abs, encoding="utf-8") as f:
                source = f.read()

            # Convert markdown to HTML body
            body = md_converter.convert(source)
            md_converter.reset()

            # Rewrite internal .md links to .html
            body = fix_internal_links(body)

            # Extract title from first H1 or use filename
            title_match = re.search(r"<h1[^>]*>(.*?)</h1>", body)
            if title_match:
                # Strip HTML tags and entities like &para; from TOC permalink
                raw = title_match.group(1)
                raw = re.sub(r"<[^>]+>", "", raw)
                raw = raw.replace("&para;", "").replace("¶", "")
                title = raw.strip()
            else:
                title = md_rel.stem

            # Determine prev/next for this page (only for nav-listed pages)
            lookup_key = str(md_rel).replace("\\", "/").replace(".md", "")
            pn = prev_next.get(lookup_key, {"prev": None, "next": None})

            # Build nav HTML (highlight current page)
            current_rel_str = str(md_rel).replace("\\", "/").replace(".md", ".html")
            nav_html = build_nav_html(page_list, current_rel_str)

            # Render full page
            full_html = render_page(
                title=title,
                content_html=body,
                nav_html=nav_html,
                prev_link=pn.get("prev"),
                next_link=pn.get("next"),
                current_rel=current_rel_str,
            )

            # Write
            with open(out_abs, "w", encoding="utf-8") as f:
                f.write(full_html)

            processed += 1
            print(f"  OK  {out_rel}")

        except Exception as exc:
            errors.append((str(md_rel), str(exc)))
            print(f"  FAIL  {md_rel}  —  {exc}")

    print(f"\n{'=' * 60}")
    print(f"  Done: {processed} converted, {len(errors)} errors")
    if errors:
        print("  Errors:")
        for path, err in errors:
            print(f"    - {path}: {err}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
