// Mermaid initialization for mkdocs-material with instant navigation
document$.subscribe(function () {
  mermaid.initialize({ startOnLoad: true, theme: "default" });
  mermaid.run({ querySelector: ".mermaid" });
});
