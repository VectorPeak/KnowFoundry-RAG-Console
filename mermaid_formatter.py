"""Custom Mermaid formatter that preserves raw content.

The built-in pymdownx.superfences formatters HTML-escape the code content,
which breaks Mermaid syntax (e.g. <br/> becomes &lt;br/&gt;).
This formatter outputs raw, unescaped content in a <div class="mermaid"> wrapper.
"""


def mermaid_div_format(source, language, class_name, options, md, **kwargs):
    """Format mermaid source as a raw div WITHOUT HTML-escaping.

    Unlike fence_code_format and fence_div_format, this does NOT call
    html.escape / _escape on the source. Mermaid.js needs raw syntax.
    """
    return '<div class="%s">%s</div>' % (class_name, source)
