"""检索候选合并与重排，不依赖 Milvus 连接的纯逻辑。

这里放的是去重、查询清洗、排序和 CrossEncoder 重排等纯逻辑。它和 `store.py` 分开，
是为了让 Milvus 交互只留在 store 层，而这些函数可以在没有 Milvus 的情况下单独测试。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document

from qa_core.retrieval.results import RetrievalHit


def document_key(document: Document) -> str:
    """返回用于合并重复命中文档的稳定标识（chunk_id > faq_id > 内容前 120 字符）。

    参数：
        document: LangChain Document。

    返回：
        稳定去重 key。
    """
    metadata = document.metadata or {}
    return str(metadata.get("chunk_id") or metadata.get("faq_id") or document.page_content[:120])


def normalize_queries(queries: Iterable[str]) -> list[str]:
    """清洗查询变体列表：去空白、去空串、按顺序去重，保持第一个查询（原问题）用于后续 rerank。

    参数：
        queries: 查询变体文本列表。

    返回：
        清洗后且按原顺序去重的查询列表。
    """
    result: list[str] = []
    for query in queries:
        cleaned = str(query or "").strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def merge_hits_by_document(merged: dict[str, RetrievalHit], hits: list[RetrievalHit]) -> None:
    """把一批候选命中合并到已有结果中，同一 chunk 被多次命中时只保留最高分。

    参数：
        merged: 已累计的命中字典，会被原地更新。
        hits: 本批新增候选命中。
    """
    for hit in hits:
        key = document_key(hit.document)
        previous = merged.get(key)
        # 同一文档被多个查询变体命中时只保留最高分，避免召回阶段排序膨胀
        if previous is None or hit.score > previous.score:
            merged[key] = hit


def sort_hits_by_score(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    """按分数从高到低排序候选结果。

    参数：
        hits: 待排序的候选命中。

    返回：
        按分数降序排列的候选列表。
    """
    return sorted(hits, key=lambda item: item.score, reverse=True)


def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    *,
    reranker: Any,
    top_n: int,
) -> list[RetrievalHit]:
    """使用 CrossEncoder 对 query-passage 对打分排序，比向量相似度更准确。

    执行流程：
      1. hits 为空时直接返回空列表。
      2. 检索计划要求重排但 reranker 为空时抛错，避免静默降级。
      3. 为每个候选构造 (query, passage_text)。
      4. 调用 CrossEncoder predict() 得到相关性分数。
      5. 按新分数降序排序。
      6. 截断为 top_n 条。

    参数：
        query: 原始用户问题。
        hits: 待重排候选。
        reranker: CrossEncoder 模型实例，需实现 predict()。
        top_n: 重排后最多返回条数。

    返回：
        重排并截断后的候选列表。

    异常：
        RuntimeError: 检索计划要求重排但 reranker 未初始化。
    """
    if not hits:
        return []
    # 检索计划要求重排但 reranker 未配置，此时不应静默降级（会显著降低排序质量）
    if reranker is None:
        raise RuntimeError("Reranker 未初始化，但当前检索计划要求重排。")
    pairs = [(query, hit.document.page_content) for hit in hits]
    # 用 CrossEncoder 对 query-passage 对重新打分，比向量余弦距离更准确，用于最终排序决策
    scores = reranker.predict(pairs)
    reranked = [
        RetrievalHit(document=hit.document, score=float(score))
        for hit, score in sorted(zip(hits, scores), key=lambda item: float(item[1]), reverse=True)
    ]
    return reranked[:top_n]
