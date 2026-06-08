"""Milvus 混合检索适配器，统一封装 FAQ 和文档集合。

QAService 不需要关心 Milvus 连接、BM25 配置、数据隔离表达式和 CrossEncoder 重排细节，
只调用这里提供的 search/search_many 即可。

依赖分层：
- qa_core.retrieval.filters：构造 Milvus 过滤表达式。
- qa_core.retrieval.ranking：候选去重、查询清洗和重排。
- qa_core.retrieval.models：embedding 和 reranker 模型提供器。
- qa_core.governance.data_scope：租户、数据集、角色的数据隔离。
"""

from __future__ import annotations

import time
from typing import Literal

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus.exceptions import MilvusException

from qa_core.governance.data_scope import DataScope
from qa_core.config.logging_config import get_logger
from qa_core.retrieval.milvus_compat import (
    bm25_function,
    collection_alias,
    ensure_milvus_database,
    ensure_orm_alias_connection,
    langchain_connection_args,
    patch_milvus_client_connection,
)
from qa_core.retrieval.filters import build_source_expr
from qa_core.retrieval.models import get_embeddings, get_reranker
from qa_core.retrieval.ranking import merge_hits_by_document, normalize_queries, rerank_hits, sort_hits_by_score
from qa_core.retrieval.results import RetrievalHit, RetrievalResult
from qa_core.config.settings import get_settings


logger = get_logger(__name__)


class MilvusHybridStore:
    """单个混合检索集合的 LangChain Milvus 封装。

    它封装单个 Milvus collection，可以是 FAQ 集合，也可以是文档集合。集合支持稠密向量
    + Milvus BM25 稀疏向量的混合检索，并在首次访问时才创建连接。

    字段说明：
      - collection_name：目标 Milvus collection 名称.
    """

    def __init__(self, collection_name: str) -> None:
        """保存集合配置，延迟创建 Milvus 连接以使模块导入保持轻量。（★★ 理解）

        构造阶段只保存集合名和全局配置，不立即连接 Milvus；真正连接发生在首次访问 store 属性时。

        参数：
            collection_name: 目标 Milvus collection 名称.
        """
        # 加载应用全局设置（Milvus URI、rerank_top_n 等检索配置）
        self.settings = get_settings()
        self.collection_name = collection_name
        self._store: Milvus | None = None

    @property
    def store(self):
        """为当前集合懒加载 LangChain Milvus 存储对象，每次连接前先打兼容补丁。（★★ 理解）

        首次访问时创建并缓存 LangChain Milvus wrapper。创建前会打 PyMilvus 兼容补丁，
        确保数据库存在，并为每个 collection 使用独立连接别名。

        执行流程：
          1. 应用 PyMilvus 连接参数兼容补丁。
          2. 未缓存时确认数据库存在，构造连接参数并连接 Milvus。
          3. 创建 Milvus wrapper，配置 BGE embedding、Milvus BM25、向量字段、文本字段和主键。
          4. 缓存并返回 Milvus wrapper。

        返回：
            当前集合对应的 LangChain Milvus wrapper.
        """
        if self._store is None:
            # 修复 langchain-milvus 的 PyMilvus 连接参数兼容性问题，仅在首次建连时执行。
            patch_milvus_client_connection()
            # 确保 Milvus 数据库存在（当前项目使用默认数据库）
            ensure_milvus_database()
            alias = collection_alias(self.collection_name)
            connection_args = langchain_connection_args(alias)
            # 业务检索走 langchain-milvus；这里仅预注册底层 ORM alias，兼容 hybrid search。
            ensure_orm_alias_connection(alias)
            self._store = Milvus(
                # 获取已缓存的 BGE 向量模型，用于稠密向量检索
                embedding_function=get_embeddings(),
                # 获取 Milvus 2.5+ 服务端 BM25 内置函数，用于稀疏向量关键词检索
                builtin_function=bm25_function(),
                collection_name=self.collection_name,
                connection_args=connection_args,
                vector_field=["dense", "sparse"],
                text_field="text",
                primary_field="pk",
                auto_id=False,
                enable_dynamic_field=True,
                consistency_level="Session",
                drop_old=False,
            )
        return self._store

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        """把文档写入 Milvus 集合，服务端自动生成稠密+稀疏向量。

        写入时 LangChain Milvus 会生成稠密向量，Milvus 服务端 BM25 会生成稀疏向量。

        参数：
            documents: 要写入的 LangChain Document 列表。
            ids: 可选主键 ID 列表。

        返回：
            写入文档的 ID 列表；documents 为空时返回空列表.
        """
        if not documents:
            return []
        # 将文档批量写入 Milvus 集合
        return self.store.add_documents(documents=documents, ids=ids)

    def delete_ids(self, ids: list[str]) -> bool:
        """按主键删除文档，供增量重建使用（先删旧 chunk 再写新 chunk）。

        增量重建时，文件变化后需要先删除旧 chunk，再写入新 chunk。

        参数：
            ids: 待删除主键列表。

        返回：
            删除成功返回 True；ids 为空也视为成功.
        """
        if not ids:
            return True
        try:
            # 从 Milvus 集合中按主键列表删除
            return bool(self.store.delete(ids=ids))
        except MilvusException as exc:
            if "collection not found" in str(exc).lower():
                logger.warning(
                    "跳过按 ID 删除：集合 %s 尚不存在，无需清理。", self.collection_name
                )
                return True
            raise

    def delete_by_expr(self, expr: str) -> bool:
        """按 Milvus 布尔表达式批量删除文档。

        适合按 source、kb_version 或数据域批量清理，不需要逐个 chunk_id 删除。

        参数：
            expr: Milvus boolean expr，例如 ``source == "hr"``。

        返回：
            删除成功返回 True；expr 为空也视为成功.
        """
        if not expr:
            return True
        try:
            # 按布尔表达式批量删除文档
            return bool(self.store.delete(expr=expr))
        except MilvusException as exc:
            if "collection not found" in str(exc).lower():
                logger.warning(
                    "跳过按表达式删除：集合 %s 尚不存在，无需清理。", self.collection_name
                )
                return True
            raise

    def search(
        self,
        query: str,
        *,
        k: int,
        source_filter: str | None,
        kb_version: str | None = None,
        valid_sources: list[str] | None = None,
        data_scope: DataScope | None = None,
        source_type: Literal["faq", "doc"],
        rerank: bool = True,
    ) -> RetrievalResult:
        """执行一次 Milvus 混合检索并按需重排，使用 weighted ranker 融合稠密+稀疏召回。（★★★ 核心）

        Dense+Sparse 融合是核心技术：稠密向量捕捉语义相似，稀疏 BM25 捕捉关键词精确匹配，二者互补。

        执行流程：
          1. 开始计时。
          2. 根据 source_filter、kb_version、valid_sources 和 data_scope 构造 Milvus expr。
          3. 调用 similarity_search_with_score，使用 weighted ranker 融合 dense/sparse。
          4. 将原始结果转换成 RetrievalHit。
          5. rerank=True 时调用 CrossEncoder 二阶段重排。
          6. 包装成 RetrievalResult 返回。

        参数：
            query: 用户检索问题。
            k: 从 Milvus 初始召回的数量。
            source_filter: 业务分类过滤项。
            kb_version: 知识库版本过滤项。
            valid_sources: source 白名单。
            data_scope: 租户、数据集、可见级别和角色过滤。
            source_type: faq 或 doc，用于标记检索来源类型。
            rerank: 是否启用 CrossEncoder 重排。

        返回：
            RetrievalResult，包含命中列表、查询文本、来源类型和耗时。
        """
        started = time.perf_counter()
        clean_query = (query or "").strip()
        if not clean_query or k <= 0:
            return RetrievalResult(
                query=clean_query,
                source_type=source_type,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        # 将 source、kb_version、tenant_id、dataset_id、visibility、allowed_roles 等合并为 Milvus 布尔表达式
        expr = build_source_expr(source_filter, kb_version, valid_sources, data_scope)
        # 稠密向量（0.55）权重略高于稀疏 BM25（0.45），语义匹配优先于关键词匹配，但关键词仍保留贡献
        kwargs = {
            "ranker_type": "weighted",
            "ranker_params": {"weights": [0.55, 0.45]},
        }
        # 执行 Milvus 稠密+稀疏混合检索，使用 weighted ranker 融合两路召回结果
        raw_hits = self._similarity_search_with_score(clean_query, k=k, expr=expr, kwargs=kwargs)

        # 转成内部 RetrievalHit 格式，隔离上层对 langchain-milvus 的依赖
        hits = [RetrievalHit(document=doc, score=float(score or 0.0)) for doc, score in raw_hits]
        if rerank and hits:
            # 二阶段 CrossEncoder 重排精度远高于向量相似度（约+10~15%），但计算成本高，仅按需启用
            hits = self._rerank(clean_query, hits)
        return RetrievalResult(
            hits=hits,
            query=clean_query,
            source_type=source_type,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _similarity_search_with_score(self, query: str, *, k: int, expr: str, kwargs: dict) -> list[tuple[Document, float]]:
        """执行 Milvus 检索；混合检索遇到空向量请求时回退到 dense-only。"""
        try:
            return self.store.similarity_search_with_score(query, k=k, expr=expr, **kwargs)
        except MilvusException as exc:
            message = str(exc)
            is_empty_query_vector_error = "nq [0] is invalid" in message or (
                "number of search vector" in message and "got 0" in message
            )
            if not is_empty_query_vector_error:
                raise
            logger.warning(
                "Milvus hybrid search returned an empty query-vector request; "
                "falling back to dense search. collection=%s query=%r",
                self.collection_name,
                query,
            )
            embedding = get_embeddings().embed_query(query)
            if not embedding:
                return []
            return self.store.similarity_search_with_score_by_vector(
                embedding,
                k=k,
                expr=expr,
                anns_field="dense",
            )

    def search_many(
        self,
        queries: list[str],
        *,
        k: int,
        source_filter: str | None,
        kb_version: str | None = None,
        valid_sources: list[str] | None = None,
        data_scope: DataScope | None = None,
        source_type: Literal["faq", "doc"],
        rerank: bool = True,
    ) -> RetrievalResult:
        """搜索多个查询变体并合并重复 chunk 命中，减少 CrossEncoder 重排次数。（★★ 理解）

        多变体搜索的核心业务价值：用户同一个问题有多种表述，变体可提升召回率；合并后统一重排避免 N 倍计算成本。

        执行流程：
          1. 开始计时。
          2. 清洗并去重查询变体。
          3. 每个变体先以 rerank=False 检索，避免每个变体单独重排。
          4. 按 chunk_id/faq_id 合并重复命中，同一 chunk 保留最高分。
          5. 合并结果按分数排序；启用 rerank 时先限制候选量，再用原问题统一重排。
          6. 截断为 k 条并包装 RetrievalResult。

        参数：
            queries: 查询变体列表，第一条应为用户原问题。
            k: 合并和重排后返回的数量。
            source_filter: 业务分类过滤项。
            kb_version: 知识库版本过滤项。
            valid_sources: source 白名单。
            data_scope: 数据隔离过滤。
            source_type: faq 或 doc。
            rerank: 是否启用 CrossEncoder 重排。

        返回：
            RetrievalResult，包含去重、重排后的命中列表和耗时。
        """
        started = time.perf_counter()
        merged: dict[str, RetrievalHit] = {}
        # 清洗查询变体列表：去空白、去空串、按顺序去重
        searched_queries = normalize_queries(queries)
        for clean_query in searched_queries:
            # 单变体先不做重排：同一批候选被多个变体重复召回时，合并后再统一重排比分别重排节省 N-1 倍计算
            result = self.search(
                clean_query,
                k=k,
                source_filter=source_filter,
                kb_version=kb_version,
                valid_sources=valid_sources,
                data_scope=data_scope,
                source_type=source_type,
                rerank=False,
            )
            # 按稳定 key（chunk_id/faq_id）合并同一文档的多次命中，只保留分数更高的那次 —— 防止同一个 chunk 在最终上下文中重复出现
            merge_hits_by_document(merged, result.hits)
        # 按分数从高到低排序候选结果
        hits = sort_hits_by_score(merged.values())
        if rerank and hits:
            # 先截断候选量：rerank_top_n × 变体数，避免无用候选浪费 CrossEncoder 计算预算
            candidate_limit = max(self.settings.rerank_top_n * max(len(searched_queries), 1), self.settings.rerank_top_n)
            hits = hits[:candidate_limit]
        if rerank and hits:
            # 用原始问题（首个变体）统一做 CrossEncoder 相关性打分，避免每个变体单独重排造成 N 倍开销
            hits = self._rerank(searched_queries[0], hits)
        return RetrievalResult(
            hits=hits[:k],
            query=" | ".join(searched_queries),
            source_type=source_type,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """使用 CrossEncoder 对候选结果二阶段排序，比向量相似度更精准。（★★ 理解）

        CrossEncoder 是检索链的精度瓶颈：一对一问+答打分，效果最好但最慢，所以只在候选数减少后才调用。

        参数：
            query: 用于相关性打分的原始问题。
            hits: 待重排候选列表。

        返回：
            按 rerank_top_n 限制后的重排结果。
        """
        return rerank_hits(
            query,
            hits,
            reranker=get_reranker(),
            top_n=self.settings.rerank_top_n,
        )



