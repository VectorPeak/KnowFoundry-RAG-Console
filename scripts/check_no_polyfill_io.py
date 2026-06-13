"""Fail if runtime sources contain forbidden third-party polyfill services."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_TARGETS = (
    ROOT / "app.py",
    ROOT / "Dockerfile",
    ROOT / "docker-compose.yml",
    ROOT / "docker-compose.milvus.yml",
    ROOT / "mermaid_formatter.py",
    ROOT / "mkdocs.yml",
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "qa_core",
    ROOT / "scripts",
    ROOT / "site",
    ROOT / "static",
    ROOT / "tests",
)
BANNED_MARKERS = (
    "polyfill" + ".io",
    "cdn." + "polyfill" + ".io",
    "polyfill-fastly" + ".io",
    "polyfill" + "-service",
)
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}
SKIP_SUFFIXES = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".log",
    ".map",
    ".pdf",
    ".png",
    ".pyc",
    ".pyo",
    ".svg",
    ".webp",
    ".zip",
}
SCAN_SUFFIXES = {
    ".css",
    ".env",
    ".html",
    ".htm",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part.lower() in SKIP_DIRS for part in path.parts):
        return False
    suffix = path.suffix.lower()
    if suffix in SKIP_SUFFIXES:
        return False
    return suffix in SCAN_SUFFIXES or path.name.lower().startswith(".env")


def iter_scan_paths():
    for target in SCAN_TARGETS:
        if target.is_file():
            yield target
        elif target.is_dir():
            yield from target.rglob("*")
    yield from ROOT.glob(".env*")


def main() -> int:
    hits: list[str] = []
    for path in iter_scan_paths():
        if not should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for marker in BANNED_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(ROOT)}: contains {marker}")

    if hits:
        print("Forbidden polyfill service references found:")
        for hit in hits:
            print(f"  - {hit}")
        return 1

    print("No forbidden polyfill service references found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
