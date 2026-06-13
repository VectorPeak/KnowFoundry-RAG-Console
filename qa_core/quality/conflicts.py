"""FAQ 标准答案与正文资料潜在冲突检测。"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
from langchain_core.documents import Document

from qa_core.document_metadata import is_table_metadata
from qa_core.quality.faq import read_faq_records


NUMERIC_FACT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|小时|天|周|个月|月|年|%|折|次|人|条|分钟)?")
VERSION_NUMBER_RE = re.compile(r"v?20\d{2}\.\d{1,2}", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"(不支持|不能|不可以|不可|不应|不得|无需|不需要|禁止|无法|不会|未开放|不允许|不代表|不一定|资料不足|未经|不得作为)")
POSITIVE_RE = re.compile(r"(支持|可以|可用|需要|允许|开放|能够|必须|会|建议|应当|应|负责)")


def _keyword_tokens(text: str) -> list[str]:
    """抽取用于 FAQ 与正文关联的中文关键词。使用 jieba.cut_for_search 搜索分词。"""
    stop_words = {
        "什么",
        "怎么",
        "如何",
        "是否",
        "可以",
        "需要",
        "相关",
        "业务",
        "问题",
        "进行",
        "支持",
        "通常",
        "应该",
        "说明",
    }
    result: list[str] = []
    for token in jieba.cut_for_search((text or "").lower()):
        token = token.strip()
        if len(token) < 2 or token in stop_words or token in result:
            continue
        result.append(token)
    return result[:12]


def _numbers(text: str) -> set[str]:
    """抽取文本中的数字事实，用于发现价格、时间、比例等口径不一致。"""
    cleaned_text = VERSION_NUMBER_RE.sub("", text or "")
    return {item.replace(" ", "") for item in NUMERIC_FACT_RE.findall(cleaned_text) if item.strip()}


def _polarity(text: str) -> str:
    """判断文本是否包含明显正向或否定口径。同时出现两者时返回 mixed_or_unknown。"""
    has_negative = bool(NEGATIVE_RE.search(text or ""))
    has_positive = bool(POSITIVE_RE.search(text or ""))
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return "mixed_or_unknown"


def _related_threshold(keywords: list[str]) -> int:
    """根据关键词数量决定 FAQ 与正文相关性的最低命中数。关键词越多，阈值越高。"""
    if len(keywords) >= 6:
        return 3
    if len(keywords) >= 3:
        return 2
    return 1


def _has_table_specific_overlap(content: str, keywords: list[str]) -> bool:
    """判断表格行是否命中了足够具体的 FAQ 关键词。至少命中一个长度>=3 的关键词或 2+ 普通关键词。"""
    normalized = content.lower()
    hits = [keyword for keyword in keywords if keyword and keyword in normalized]
    return any(len(keyword) >= 3 for keyword in hits) or len(hits) >= 2


def _check_record_conflicts(record, indexed_chunks, conflicts: list, limit: int) -> bool:
    """检查一条 FAQ 记录与所有 indexed chunks 的冲突。
    返回 True 表示应继续检查下一条记录，False 表示已达到 limit 应停止。
    """
    question = record["question"]
    answer = record["answer"]
    if not question or not answer:
        return True
    keywords = _keyword_tokens(f"{question} {answer}")
    related = []
    threshold = _related_threshold(keywords)
    for item in indexed_chunks:
        if record["source"] and item["source"] and record["source"] != item["source"]:
            continue
        content = item["content"]
        if is_table_metadata(item["metadata"]) and not _has_table_specific_overlap(content, keywords):
            continue
        hit_count = sum(1 for keyword in keywords if keyword and keyword in content.lower())
        if hit_count >= threshold:
            related.append((hit_count, item))
    related.sort(key=lambda pair: pair[0], reverse=True)
    top_related = [item for _, item in related[:5]]
    if not top_related:
        conflicts.append(
            {
                "issue": "no_related_document",
                "row": record["row"],
                "question": question,
                "source": record["source"],
                "reason": "FAQ 有标准答案，但正文资料中没有找到明显相关片段，需要确认 FAQ 是否有文档依据。",
            }
        )
        return True

    answer_numbers = _numbers(answer)
    answer_polarity = _polarity(answer)
    for item in top_related:
        content = item["content"]
        doc_numbers = _numbers(content)
        if answer_numbers and doc_numbers and answer_numbers.isdisjoint(doc_numbers):
            conflicts.append(
                {
                    "issue": "numeric_mismatch",
                    "row": record["row"],
                    "question": question,
                    "source": record["source"],
                    "faq_numbers": sorted(answer_numbers),
                    "document_numbers": sorted(doc_numbers),
                    "file_name": item["file_name"],
                    "content_preview": content[:160],
                    "reason": "FAQ 答案和相关正文出现不同数字，可能存在价格、时间、比例或数量口径冲突。",
                }
            )
        doc_polarity = _polarity(content)
        if {answer_polarity, doc_polarity} == {"positive", "negative"}:
            conflicts.append(
                {
                    "issue": "polarity_mismatch",
                    "row": record["row"],
                    "question": question,
                    "source": record["source"],
                    "faq_polarity": answer_polarity,
                    "document_polarity": doc_polarity,
                    "file_name": item["file_name"],
                    "content_preview": content[:160],
                    "reason": "FAQ 答案和相关正文存在肯定/否定口径差异，需要人工确认。",
                }
            )
        if len(conflicts) >= limit:
            return False
    return True


def detect_faq_document_conflicts(faq_csv: str | Path, chunks: list[Document], *, limit: int = 100) -> dict[str, Any]:
    """检测 FAQ 标准答案和正文资料之间的潜在冲突（no_related_document/numeric_mismatch/polarity_mismatch）。"""
    # 原因： FAQ 标准答案由业务人员维护，正文资料由另外的团队提供——两套来源可能口径不一致（价格/否定/支持范围），激活前自动比对可以提前发现冲突而不依赖人工逐条复核
    records = read_faq_records(faq_csv)
    conflicts: list[dict[str, Any]] = []
    indexed_chunks = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        indexed_chunks.append(
            {
                "content": chunk.page_content or "",
                "metadata": metadata,
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name") or metadata.get("source"),
            }
        )

    for record in records:
        if not _check_record_conflicts(record, indexed_chunks, conflicts, limit):
            break

    summary = Counter(item["issue"] for item in conflicts)
    return {
        "checked_faq_count": len([item for item in records if item.get("question") and item.get("answer")]),
        "conflict_count": len(conflicts),
        "issue_counts": dict(summary),
        "items": conflicts[:limit],
    }

