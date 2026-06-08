"""chunk 质量检测规则。不依赖 LLM，适合入库前后高频运行。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_table_metadata
from qa_core.utils import stable_hash


def _content_hash(content: str) -> str:
    """对 chunk 正文生成短 hash，用于重复内容检测。"""
    return stable_hash(content.strip())[:16]


def analyze_chunk_quality(chunks: list[Document]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """检测 chunk 级别的常见质量问题（empty/too_short/too_long/low_unique_ratio/duplicate_content）。
    Returns (issues_list, stats_dict)."""
    settings = get_settings()
    issues: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    lengths: list[int] = []
    for index, chunk in enumerate(chunks, start=1):
        content = (chunk.page_content or "").strip()
        metadata = dict(chunk.metadata or {})
        file_name = metadata.get("file_name") or metadata.get("source") or "unknown"
        length = len(content)
        lengths.append(length)
        digest = _content_hash(content)
        seen[digest] = seen.get(digest, 0) + 1
        base = {
            "index": index,
            "file_name": file_name,
            "source": metadata.get("source"),
            "chunk_id": metadata.get("chunk_id"),
            "length": length,
            "content_preview": content[:120],
        }
        if not content:
            issues.append({**base, "issue": "empty", "reason": "chunk 为空，不能提供有效语义。"})
            continue
        if length < 30 and not is_table_metadata(metadata):
            issues.append({**base, "issue": "too_short", "reason": "chunk 过短，可能只包含标题、页码或碎片。"})
        if length > max(settings.parent_chunk_size * 2, 2000):
            issues.append({**base, "issue": "too_long", "reason": "chunk 过长，会降低召回精度并增加 prompt 成本。"})
        compact = "".join(content.split())
        unique_ratio = round(len(set(compact)) / len(compact), 4) if compact else 0.0
        if length >= 50 and unique_ratio < 0.08 and not is_table_metadata(metadata):
            issues.append(
                {
                    **base,
                    "issue": "low_unique_ratio",
                    "unique_ratio": unique_ratio,
                    "reason": "字符重复比例过高，可能是 OCR 噪声、表格线或解析失败内容。",
                }
            )

    duplicate_count = 0
    for index, chunk in enumerate(chunks, start=1):
        content = (chunk.page_content or "").strip()
        digest = _content_hash(content)
        if content and seen.get(digest, 0) > 1:
            duplicate_count += 1
            metadata = dict(chunk.metadata or {})
            issues.append(
                {
                    "index": index,
                    "file_name": metadata.get("file_name") or metadata.get("source") or "unknown",
                    "source": metadata.get("source"),
                    "chunk_id": metadata.get("chunk_id"),
                    "length": len(content),
                    "issue": "duplicate_content",
                    "reason": "chunk 正文重复，可能造成重复召回和答案引用噪声。",
                    "content_preview": content[:120],
                }
            )

    stats = {
        "chunk_count": len(chunks),
        "duplicate_chunk_count": duplicate_count,
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "avg_chunk_length": round(sum(lengths) / max(len(lengths), 1), 2),
        "low_quality_issue_count": len(issues),
    }
    return issues, stats
