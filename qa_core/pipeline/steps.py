"""RAG 主流程中的业务步骤。

这里的函数名称刻意写得直白：准备检索、检索 FAQ、检索文档、准备生成参数。
主流程 `rag.py` 只负责把这些步骤串起来，细节放在这里，阅读时不用在一个超长函数里
来回跳。
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from qa_core.intent.classifier import IntentResult, classify_intent, infer_source
from qa_core.llm.client import get_chat_model
from qa_core.memory.history import format_messages
from qa_core.pipeline.context import (
    build_context,
    direct_faq_answer,
    effective_source_filter as resolve_effective_source_filter,
    select_context_docs,
)
from qa_core.pipeline.query_variants import generate_query_variants
from qa_core.pipeline.rewrite import rewrite_query_if_needed
from qa_core.pipeline.runtime import RAGQueryContext
from qa_core.prompts.profiles import PromptProfile
from qa_core.prompts.selector import build_answer_prompt_profile
from qa_core.retrieval.factory import get_faq_store
from qa_core.retrieval.results import RetrievalResult
from qa_core.retrieval.strategy import RetrievalPlan, build_retrieval_plan
from qa_core.scenarios.boundary import detect_scenario_boundary, detect_source_boundary

# 匹配"怎么办/如何/怎么/能不能"等标准问答句式特征，命中后值得先尝试 FAQ 精确直出
FAQ_FAST_PATH_HINTS = re.compile(r"(怎么办|如何|怎么|需要什么|需要哪些|需要谁|有哪些|为什么|什么时候|能不能|可以吗|会不会|是否|吗|什么|谁|怎么处理)")

@dataclass
class RetrievalPreparation:
    """prepare_retrieval() 的输出包，供下游 search_faq / search_doc / prepare_answer 消费。
    """

    history_messages: list[Any]       # 压缩后的对话上下文，用于拼接 user_prompt
    intent: IntentResult              # 意图识别结果，rag.py 据此判断是否直接回答
    effective_source_filter: str | None  # 最终生效的 source 过滤项
    rewritten_query: str              # 改写后的独立检索问题（或原问题）
    plan: RetrievalPlan               # ★ 检索计划，控制后续所有检索参数
    query_variants: list[str]         # 同义检索表达列表，原问题排第一位
    prompt_profile: PromptProfile     # 回答 Prompt 模板档位


@dataclass
class AnswerPreparation:
    """调用大模型前需要准备好的 Prompt 和来源信息。"""

    context_docs: list[Document]
    sources: list[dict[str, Any]]
    hit_type: str
    system_prompt: str
    user_prompt: str


def should_try_faq_fast_path(query: str, scenario) -> bool:
    """判断问题是否短小完整且像标准问答，值得先尝试 FAQ 精确直出。

    参数：
        query: 用户提出的原始问题
        scenario: ScenarioDefinition 业务场景定义

    返回：
        bool: True 表示问题满足精确直出前置条件（短、无换行、含 FAQ 句式特征或可推断 source）
    """
    compact_query = (query or "").strip()
    if not compact_query or len(compact_query) > 48 or "\n" in compact_query:
        return False
    return bool(FAQ_FAST_PATH_HINTS.search(compact_query) or infer_source(compact_query, scenario))


def _exact_faq_answer(query: str, faq_result: RetrievalResult) -> tuple[str | None, RetrievalResult]:
    """从 FAQ 候选中寻找与标准问题完全一致的答案（仅允许精确匹配直出）。

    参数：
        query: 用户原始提问
        faq_result: FAQ 检索结果（含 hits 列表）

    返回：
        tuple: (answer_str or None, 可能重排后的 faq_result)
    """
    for index, hit in enumerate(faq_result.hits):
        answer = direct_faq_answer(query, hit.document, hit.score, threshold=float("inf"))
        if not answer:
            continue
        if index:
            # 精确匹配项不在首位时将其提到列表最前，保证来源展示顺序一致
            reordered = [hit, *faq_result.hits[:index], *faq_result.hits[index + 1 :]]
            faq_result = RetrievalResult(
                hits=reordered,
                query=faq_result.query,
                source_type=faq_result.source_type,
                elapsed_ms=faq_result.elapsed_ms,
            )
        return answer, faq_result
    return None, faq_result


def try_fast_faq_direct_answer(context: RAGQueryContext) -> str | None:
    """意图识别前置快速路径：短问题精确匹配标准 FAQ 时直接返回，跳过后续所有 RAG 步骤。★★★ 核心

    执行流程：
    1. 检查问题是否符合 fast path 前置条件（短/无换行/标准问答句式）
    2. 校验前端 source_filter 合法性
    3. 构建轻量检索计划（仅 FAQ，不查文档，不 rerank）
    4. 执行 FAQ 混合检索（仅单变体）
    5. 从候选中寻找精确匹配的标准问题答案
    6. 命中的直接设置 hit_type 并返回答案；未命中的缓存结果供主链路复用

    参数：
        context: RAGQueryContext 请求级状态

    返回：
        str | None: 精确匹配的答案文本；未命中时返回 None
    """
    if not should_try_faq_fast_path(context.query, context.scenario):
        return None

    # 校验前端 source_filter 是否在场景白名单内，拒绝非法业务分类
    context.run_stage("fast_validate_source", lambda: context.validate_source(context.source_filter, context.scenario))

    # 根据问题关键词推断可能的业务分类过滤项
    suggested_source = infer_source(context.query, context.scenario)
    intent = IntentResult(
        intent="FAQ_QUERY",
        confidence=0.98,
        reason="faq_fast_exact_match_probe",
        requires_rewrite=False,
        suggested_source=suggested_source,
    )
    context.intent_payload = intent.as_dict()
    context.rewritten_query = context.query

    # 确定 fast path 的 source 过滤项：快路径只用前端显式选择，不自动推断
    effective_source_filter = context.run_stage(
        "fast_resolve_source_filter",
        lambda: resolve_effective_source_filter(context.source_filter, None, context.scenario),
    )

    # 为 FAQ 快路径构建轻量检索计划：只查 FAQ，不查文档，不启用 rerank
    plan = context.run_stage("fast_build_retrieval_plan", lambda: build_retrieval_plan(context.query, intent))

    # 选择 FAQ 直出使用的提示词模板档位
    prompt_profile = context.run_stage(
        "fast_select_prompt_profile",
        lambda: build_answer_prompt_profile(intent.intent, context.scenario, context.query),
    )

    # 构造检索诊断信息快照，供后续 trace 和错误排查使用
    context.retrieval_info = {
        "fast_path": "faq_exact_match",
        "plan": plan.as_dict(),
        "query_variants": [context.query],
        "scenario_id": context.scenario.scenario_id,
        "scenario_name": context.scenario.display_name,
        "data_scope": context.data_scope.as_dict(),
        "source_filter": effective_source_filter,
        "kb_version": context.active_kb_version,
        "prompt_profile": prompt_profile.as_dict(),
    }

    # 获取当前场景的 FAQ Milvus 混合检索集合，执行单查询变体的快速检索
    faq_result = context.run_stage(
        "faq_fast_retrieval",
        lambda: get_faq_store(context.scenario.faq_collection).search_many(
            [context.query],
            k=min(plan.faq_top_k, 12),
            source_filter=effective_source_filter,
            kb_version=context.active_kb_version,
            valid_sources=context.scenario.valid_sources,
            data_scope=context.data_scope,
            source_type="faq",
            rerank=False,
        ),
    )
    context.retrieval_info["faq_elapsed_ms"] = round(faq_result.elapsed_ms, 2)
    context.retrieval_info["faq_top_score"] = faq_result.top_score

    # 从 FAQ 候选中找与用户问题标准问题完全一致的答案，只允许精确匹配直出
    answer, faq_result = _exact_faq_answer(context.query, faq_result)
    if not answer:
        # 无精确命中时，用边界规则判断是否跨场景/跨 source 提问
        boundary_answer = detect_and_apply_boundary_answer(context)
        if boundary_answer:
            return boundary_answer
        # 缓存 fast path 的 FAQ 召回结果，主链路同参数时复用避免重复检索
        context.fast_faq_result = faq_result
        context.fast_faq_source_filter = effective_source_filter
        return None
    context.hit_type = "faq_direct"
    context.sources = faq_result.source_payloads()
    context.retrieval_info["fast_path_hit"] = True
    return answer


def prepare_retrieval(context: RAGQueryContext) -> RetrievalPreparation:
    """意图识别+检索参数准备：识别意图、改写追问、构建检索策略，为下游检索生成完整参数包。★★★ 核心

    执行流程：
    1. 校验前端 source_filter 合法性
    2. 检测场景/source 边界问题，若跨域则返回引导提示
    3. 加载历史摘要 + 最近消息作为对话上下文
    4. 规则优先 + LLM 补充的意图识别
    5. 确定 source 过滤项（前端 > 意图推断 > 不过滤）
    6. 追问改写（将依赖上下文的问法转为独立检索问题）
    7. 构建检索策略（top_k、阈值、是否重排等）
    8. 生成同义检索表达列表（查询变体）
    9. 选择最终回答提示词模板档位
    10. 汇总检索诊断信息快照

    参数：
        context: RAGQueryContext 请求级状态

    返回：
        RetrievalPreparation: 包含检索所需全部参数的数据包
    """
    # 校验前端 source_filter 是否在场景白名单内，拒绝非法业务分类
    context.run_stage("validate_source", lambda: context.validate_source(context.source_filter, context.scenario))

    # 识别场景或 source 边界问题，若跨域则返回引导提示
    boundary_answer = detect_and_apply_boundary_answer(context)
    if boundary_answer:
        intent = IntentResult(
            intent="OUT_OF_SCOPE",
            direct_answer=boundary_answer,
            confidence=0.98,
            reason=context.retrieval_info.get("boundary_reason") or "scenario_boundary",
            requires_rewrite=False,
            suggested_source=None,
        )
        context.intent_payload = intent.as_dict()
        # 根据意图和问题类别选择最终回答提示词模板档位
        prompt_profile = build_answer_prompt_profile(intent.intent, context.scenario, context.query)
        context.retrieval_info["prompt_profile"] = prompt_profile.as_dict()
        return RetrievalPreparation(
            history_messages=[],
            intent=intent,
            effective_source_filter=context.source_filter,
            rewritten_query=context.query,
            plan=build_retrieval_plan(context.query, intent),
            query_variants=[context.query],
            prompt_profile=prompt_profile,
        )

    # 从 MySQL 加载"历史摘要 + 最近消息"作为压缩后的对话上下文
    history_messages = context.run_stage("load_history", lambda: context.history.get_context_messages(context.session_id))

    # 规则优先 + LLM 补充的意图识别（问候、追问、FAQ 查询等）
    intent = context.run_stage("classify_intent", lambda: classify_intent(context.query, history_messages, context.scenario))
    context.intent_payload = intent.as_dict()

    # 确定 source 过滤项：前端显式选择 > 意图推断 > 不过滤
    effective_source_filter = context.run_stage(
        "resolve_source_filter",
        lambda: resolve_effective_source_filter(context.source_filter, intent.suggested_source, context.scenario),
    )

    # 将依赖上下文的追问改写成独立检索问题
    context.rewritten_query = context.run_stage(
        "rewrite_query",
        lambda: rewrite_query_if_needed(context.query, history_messages, intent.requires_rewrite),
    )

    # 构建检索策略（FAQ/doc top_k、阈值、是否重排等）
    plan = context.run_stage("build_retrieval_plan", lambda: build_retrieval_plan(context.rewritten_query, intent))

    # 生成同义检索表达（如"Webhook"→"回调"），同时传给 FAQ 和文档检索
    query_variants = context.run_stage(
        "generate_query_variants",
        lambda: generate_query_variants(context.rewritten_query, enabled=plan.use_query_variants),
    )

    # 选择最终回答提示词模板档位
    prompt_profile = context.run_stage(
        "select_prompt_profile",
        lambda: build_answer_prompt_profile(intent.intent, context.scenario, context.rewritten_query),
    )

    # 构造检索诊断信息快照，供后续 trace 和错误排查使用
    context.retrieval_info = {
        "plan": plan.as_dict(),
        "query_variants": query_variants,
        "scenario_id": context.scenario.scenario_id,
        "scenario_name": context.scenario.display_name,
        "data_scope": context.data_scope.as_dict(),
        "source_filter": effective_source_filter,
        "kb_version": context.active_kb_version,
        "prompt_profile": prompt_profile.as_dict(),
    }
    return RetrievalPreparation(
        history_messages=history_messages,
        intent=intent,
        effective_source_filter=effective_source_filter,
        rewritten_query=context.rewritten_query,
        plan=plan,
        query_variants=query_variants,
        prompt_profile=prompt_profile,
    )


def prepare_answer(
    context: RAGQueryContext,
    prepared: RetrievalPreparation,
    faq_result: RetrievalResult,
    doc_result: RetrievalResult,
) -> AnswerPreparation:
    """将 FAQ+文档检索结果整理为 LLM Prompt、引用来源和命中类型，为流式生成做准备。★★★ 核心

    执行流程：
    1. 调用 _build_answer_context 筛选上下文、确定来源和命中类型
    2. 记录上下文统计指标（条数/字符数/来源数/分数）到检索诊断信息
    3. 将历史消息和上下文填充到提示词模板中

    参数：
        context: RAGQueryContext 请求级状态
        prepared: RetrievalPreparation 检索参数包
        faq_result: FAQ 检索结果
        doc_result: 文档检索结果

    返回：
        AnswerPreparation: 包含 system_prompt、user_prompt、context_docs、sources、hit_type
    """
    # 整理上下文、引用来源、命中类型和最高分数，为 LLM 生成准备
    context_docs, sources, hit_type, top_score = context.run_stage(
        "build_answer_context",
        lambda: _build_answer_context(prepared, faq_result, doc_result),
    )
    # 记录上下文统计指标（数量/字符数/来源数/分数），供状态页和 trace 诊断使用
    context.retrieval_info["context_count"] = len(context_docs)
    context.retrieval_info["context_chars"] = sum(len(doc.page_content or "") for doc in context_docs)
    context.retrieval_info["context_source_count"] = len({str((doc.metadata or {}).get("source") or "") for doc in context_docs})
    context.retrieval_info["context_min_score"] = prepared.plan.min_context_score
    context.retrieval_info["context_top_score"] = top_score

    # 将历史消息转为中文对话文本，填充到提示词模板中
    user_prompt = prepared.prompt_profile.user_template.format(
        history=format_messages(prepared.history_messages),
        question=prepared.rewritten_query,
        context=build_context(context_docs) or "无可用上下文。必须明确回答：信息不足，无法确认。",
    )
    return AnswerPreparation(
        context_docs=context_docs,
        sources=sources,
        hit_type=hit_type,
        system_prompt=prepared.prompt_profile.system_template,
        user_prompt=user_prompt,
    )


def _build_answer_context(
    prepared: RetrievalPreparation,
    faq_result: RetrievalResult,
    doc_result: RetrievalResult,
) -> tuple[list[Document], list[dict[str, Any]], str, float]:
    """整理上下文文档、引用来源、命中类型和最高分数，为 prepare_answer 提供核心数据。★★★ 核心

    执行流程：
    1. 按分数阈值和条数限制从 FAQ/文档候选中筛选上下文 Doc 列表
    2. prefer_table 时来源以文档为主、FAQ 为辅（表格行优先展示）
    3. 取 FAQ 和文档两端最高分中的较大值
    4. 无上下文通过过滤时命中类型标记为 insufficient_context

    参数：
        prepared: RetrievalPreparation 检索参数包（含 plan）
        faq_result: FAQ 检索结果
        doc_result: 文档检索结果

    返回：
        tuple: (context_docs, sources, hit_type, top_score) 四元组
    """
    # 按分数阈值和条数限制从 FAQ/文档候选中选择最终进入 prompt 的上下文片段
    context_docs = select_context_docs(faq_result.hits, doc_result.hits, prepared.plan)
    # 表格查询时优先展示表格行，来源列表以文档为主、FAQ 为辅
    if prepared.plan.prefer_table:
        sources = doc_result.source_payloads(limit=5) + faq_result.source_payloads(limit=2)
    else:
        sources = faq_result.source_payloads(limit=2) + doc_result.source_payloads(limit=5)
    top_score = max(
        [score for score in [faq_result.top_score, doc_result.top_score] if score is not None],
        default=0.0,
    )
    # 无上下文通过分数过滤时返回 insufficient_context，上游据此返回确定性信息不足回答
    return context_docs, sources, "rag" if context_docs else "insufficient_context", top_score


def build_insufficient_context_answer(context: RAGQueryContext) -> str:
    """无可用上下文时返回确定性"信息不足"回答，避免 LLM 幻觉。

    参数：
        context: RAGQueryContext 请求级状态（含 scenario.support_contact）

    返回：
        str: 确定性"信息不足，无法确认"提示文案
    """
    context.retrieval_info["insufficient_context_reason"] = "no_context_after_score_filter"
    return f"信息不足，无法确认。当前知识库没有召回到足够可靠的依据，请联系{context.scenario.support_contact}。"


def stream_llm_answer(system_prompt: str, user_prompt: str):
    """调用 LangChain ChatModel 流式生成答案片段。

    参数：
        system_prompt: SystemMessage 系统提示词
        user_prompt: HumanMessage 用户提示词（含上下文和问题）

    返回：
        BaseMessage stream iterable，yield 每个 AIMessageChunk 直到流结束
    """
    # 获取已缓存的流式 ChatOpenAI 客户端，按 token 逐步推送生成结果
    llm = get_chat_model(streaming=True)
    # 以 SystemMessage + HumanMessage 调用 LLM 流式生成
    return llm.stream([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])


def detect_and_apply_boundary_answer(context: RAGQueryContext) -> str | None:
    """识别场景或 source 边界问题（如跨场景提问、source 与问题不匹配），并返回确定性提示。

    执行流程：
    1. 调用 detect_scenario_boundary 判断当前问题是否明显属于其他业务场景
    2. 若跨场景，返回"请切换到对应场景后查询"引导
    3. 调用 detect_source_boundary 判断前端 source 是否和问题明显不匹配
    4. 若不匹配，返回"请切换分类后查询"引导
    5. 无边界问题时返回 None，继续正常 RAG 流程

    参数：
        context: RAGQueryContext 请求级状态（含 query、scenario、source_filter）

    返回：
        str | None: 边界问题引导文案；无边界问题时返回 None
    """
    # 判断当前问题是否明显属于其他业务场景（如选了保险但问了 AI 相关问题）
    scenario_boundary = detect_scenario_boundary(context.query, context.scenario)
    context.retrieval_info["scenario_boundary"] = scenario_boundary.as_dict()
    if scenario_boundary.crossed:
        context.hit_type = "scenario_boundary"
        context.retrieval_info["boundary_reason"] = "scenario_boundary"
        return (
            f"当前场景是「{context.scenario.display_name}」，未在该场景资料中确认这个问题的依据。"
            f"这个问题更像「{scenario_boundary.matched_scenario_name}」中的"
            f"「{scenario_boundary.matched_source_label}」分类。请切换到对应场景后再查询。"
        )

    # 判断用户显式选择的 source 是否和问题明显不匹配（如选了 hr 但问题属于 legal）
    source_boundary = detect_source_boundary(context.query, context.scenario, context.source_filter)
    context.retrieval_info["source_boundary"] = source_boundary.as_dict()
    if source_boundary.mismatched:
        context.hit_type = "source_boundary"
        context.retrieval_info["boundary_reason"] = "source_boundary"
        return (
            f"当前选择的是「{source_boundary.selected_source_label}」，但问题更像当前场景下的"
            f"「{source_boundary.matched_source_label}」分类。为避免按错误资料回答，请切换分类后再查询。"
        )
    return None
