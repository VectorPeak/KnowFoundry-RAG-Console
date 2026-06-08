"""RAG 主流程中的检索执行步骤。

`steps.py` 负责意图、改写、Prompt 等准备工作；本文件只负责把准备好的
`RetrievalPreparation` 落到 FAQ 和文档检索上。这样学生阅读主链路时可以清楚区分：
先决定“怎么查”，再执行“真正去哪里查”。
"""

from __future__ import annotations

from qa_core.pipeline.context import direct_faq_answer
from qa_core.pipeline.runtime import RAGQueryContext
from qa_core.pipeline.steps import RetrievalPreparation
from qa_core.retrieval.factory import get_doc_store, get_faq_store
from qa_core.retrieval.results import RetrievalResult


def search_faq(context: RAGQueryContext, prepared: RetrievalPreparation) -> RetrievalResult:
    """按检索计划执行 FAQ 混合检索，并把耗时、最高分写入检索诊断信息。★★★ 核心

    使用场景：
    - FAQ 标准问答优先召回，用于判断是否可以直接返回标准答案；
    - FAQ 未直出时，FAQ 片段仍可和文档片段一起进入 Prompt，补充标准口径。

    为什么单独封装：FAQ 检索涉及 fast path 结果复用、source 过滤、版本过滤和数据隔离。
    这些属于“执行检索”的细节，不应该挤在 `rag.py` 的流程编排里。
    """

    def do_search() -> RetrievalResult:
        if not prepared.plan.run_faq:
            return RetrievalResult(query=prepared.rewritten_query, source_type="faq")
        if (
            context.fast_faq_result is not None
            and prepared.rewritten_query == context.query
            and prepared.query_variants == [context.query]
            and prepared.effective_source_filter == context.fast_faq_source_filter
        ):
            context.retrieval_info["faq_reused_from_fast_path"] = True
            return context.fast_faq_result
        context.retrieval_info["faq_reused_from_fast_path"] = False
        return get_faq_store(context.scenario.faq_collection).search_many(
            prepared.query_variants,
            k=prepared.plan.faq_top_k,
            source_filter=prepared.effective_source_filter,
            kb_version=context.active_kb_version,
            valid_sources=context.scenario.valid_sources,
            data_scope=context.data_scope,
            source_type="faq",
            rerank=prepared.plan.rerank,
        )

    faq_result = context.run_stage("faq_retrieval", do_search)
    context.retrieval_info["faq_elapsed_ms"] = round(faq_result.elapsed_ms, 2)
    context.retrieval_info["faq_top_score"] = faq_result.top_score
    return faq_result


def get_faq_direct_answer(
    context: RAGQueryContext,
    prepared: RetrievalPreparation,
    faq_result: RetrievalResult,
) -> str | None:
    """判断 FAQ top 命中是否足够可靠，可靠时直接返回标准答案。★★★ 核心

    使用场景：用户问的是制度型、流程型、标准口径型问题，例如“报销需要哪些材料”。
    如果 FAQ 中已经有高置信标准答案，就不必再让 LLM 重新生成，减少幻觉和延迟。
    """

    threshold = float("inf") if prepared.plan.faq_direct_exact_only else prepared.plan.faq_direct_threshold
    return direct_faq_answer(
        context.query,
        faq_result.top_document,
        faq_result.top_score,
        threshold,
    )


def search_doc(context: RAGQueryContext, prepared: RetrievalPreparation) -> RetrievalResult:
    """按检索计划执行文档混合检索，并把耗时、最高分写入检索诊断信息。★★★ 核心

    使用场景：
    - FAQ 没有直接命中，需要从制度、合同、规范、表格等正文资料里召回证据；
    - 表格问题、复杂资料问题通常依赖文档检索而不是 FAQ 直出。
    """

    def do_search() -> RetrievalResult:
        if not prepared.plan.run_doc:
            return RetrievalResult(query=prepared.rewritten_query, source_type="doc")
        return get_doc_store(context.scenario.doc_collection).search_many(
            prepared.query_variants,
            k=prepared.plan.doc_top_k,
            source_filter=prepared.effective_source_filter,
            kb_version=context.active_kb_version,
            valid_sources=context.scenario.valid_sources,
            data_scope=context.data_scope,
            source_type="doc",
            rerank=prepared.plan.rerank,
        )

    doc_result = context.run_stage("doc_retrieval", do_search)
    context.retrieval_info["doc_elapsed_ms"] = round(doc_result.elapsed_ms, 2)
    context.retrieval_info["doc_top_score"] = doc_result.top_score
    return doc_result

