"""Validate chapter map metadata against the codealong chapter files."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

from sync_chapter_animations import CHAPTERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_SYMBOLS = {
    "DEFAULT_*",
    "retrieval_info['prompt_profile']",
    "list/activate/archive payload",
    "python -m unittest / acceptance_smoke",
    "store.delete_ids() / add_documents()",
    "search_many(..., data_scope=...)",
    "active_kb_version",
    "run_guardrails()",
    "unittest",
    "router",
    "BM25BuiltInFunction",
    "MilvusHybridStore",
    "Collection.hybrid_search()",
    "os.walk()",
}

SYMBOL_ALIASES = {
    "QueryServiceContext.from_ws_payload()": ("class QueryServiceContext", "def from_ws_payload"),
    "from_debug_request()": "def from_debug_request",
    "FeedbackStore.add_feedback()": ("class FeedbackStore", "def add_feedback"),
    "QAService.stream_query()": ("class QAService", "def stream_query"),
    "resolve_scenario()": "def resolve_scenario",
    "Settings": "class Settings",
    "RetrievalPlan": "class RetrievalPlan",
    "IntentResult": "class IntentResult",
    "RouteDecision": "class RouteDecision",
    "DataScope": "class DataScope",
    "KnowledgeBaseVersion": "class KnowledgeBaseVersion",
    "KnowledgeBaseVersionStore": "class KnowledgeBaseVersionStore",
    "IndexManifest": "class IndexManifest",
    "RAGQueryContext": "class RAGQueryContext",
    "PromptProfile": "class PromptProfile",
    "QueryVariants": "class QueryVariants",
    "TestSystemChapter18Test": "class TestSystemChapter18Test",
    "QualityReportChapter17Test": "class QualityReportChapter17Test",
    "_MySqlStore": "class _MySqlStore",
}


@dataclass(frozen=True)
class MapIssue:
    chapter: str
    title: str
    file_path: str
    symbol: str
    message: str

    def format(self) -> str:
        return f"第 {self.chapter} 章 {self.title}: {self.file_path} / {self.symbol} - {self.message}"


def _split_file_paths(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+\+\s+", value) if part.strip()]


def _split_symbols(value: str) -> list[str]:
    if value in SKIP_SYMBOLS:
        return []
    symbols: list[str] = []
    for part in value.split("/"):
        item = part.strip()
        if item and item not in SKIP_SYMBOLS:
            symbols.append(item)
    return symbols


def _symbol_tokens(symbol: str) -> tuple[str, ...]:
    alias = SYMBOL_ALIASES.get(symbol)
    if alias:
        return (alias,) if isinstance(alias, str) else tuple(alias)
    cleaned = symbol.strip()
    if not cleaned or cleaned in SKIP_SYMBOLS:
        return ()
    if cleaned.endswith("()"):
        name = cleaned[:-2].split(".")[-1]
        return (f"def {name}",)
    if "." in cleaned and cleaned.endswith(")"):
        name = cleaned.split(".")[-1].removesuffix("()")
        return (f"def {name}",)
    if cleaned.endswith("*"):
        return ()
    return (cleaned,)


def _matching_files(chapter_root: Path, raw_path: str) -> list[Path]:
    if any(char in raw_path for char in "*?[]"):
        return sorted(path for path in chapter_root.rglob("*") if fnmatch.fnmatch(path.as_posix(), f"*{raw_path}"))
    path = chapter_root / raw_path
    return [path] if path.exists() else []


def _symbol_exists(files: list[Path], symbol: str) -> bool:
    tokens = _symbol_tokens(symbol)
    if not tokens:
        return True
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files if path.suffix == ".py")
    return all(token in combined for token in tokens)


def validate_maps() -> list[MapIssue]:
    issues: list[MapIssue] = []
    for chapter in CHAPTERS:
        chapter_no = str(chapter["no"])
        chapter_root = PROJECT_ROOT / str(chapter["path"])
        for _, title, file_value, symbol_value, _ in chapter["nodes"]:  # type: ignore[index]
            file_paths = _split_file_paths(str(file_value))
            matched_files: list[Path] = []
            for raw_path in file_paths:
                matches = _matching_files(chapter_root, raw_path)
                if not matches:
                    issues.append(MapIssue(chapter_no, str(title), raw_path, str(symbol_value), "文件不存在"))
                matched_files.extend(matches)
            if not matched_files:
                continue
            for symbol in _split_symbols(str(symbol_value)):
                if not _symbol_exists(matched_files, symbol):
                    issues.append(MapIssue(chapter_no, str(title), str(file_value), symbol, "符号未在对应文件中找到"))
    return issues


def main() -> int:
    issues = validate_maps()
    if issues:
        print("章节地图校验失败：")
        for issue in issues:
            print(f"- {issue.format()}")
        return 1
    print("章节地图校验通过：05-19 地图节点均能对齐到对应章节代码。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
