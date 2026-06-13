"""RAG 主流程编排：`stream_query`（流式问答）和 `debug_retrieval`（检索调试）。
"""
from __future__ import annotations
from collections.abc import Generator
from typing import Any
from qa_core.config.logging_config import get_logger
from qa_core.pipeline.events import (
    status_event as build_status_event,
    token_event as build_token_event,
)
from qa_core.pipeline.citations import enforce_answer_citations
from qa_core.pipeline.runtime import (
    create_query_context,
    finish_error,
    finish_success,
    start_event as build_query_start_event,
)
from qa_core.pipeline.steps import (
    build_insufficient_context_answer,
    decide_route,
    prepare_answer,
    prepare_retrieval,
    stream_llm_answer,
)
from qa_core.pipeline.retrieval_steps import get_faq_direct_answer, search_doc, search_faq
logger = get_logger(__name__)

def stream_query(
    history,
    validate_source,
    query: str,
    source_filter: str | None,
    session_id: str | None,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    user_role: str | None = None,
    user_roles: list[str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """一次完整问答请求的编排主入口，协调 Stage 0-7 管线并持续产出 WebSocket 事件流。★★★ 核心

    执行流程：
    Stage 0: 创建运行时上下文（场景/数据域/会话/trace/知识库版本）
    Stage 1: 低成本查询路由（直答/边界、FAQ 精确命中、继续检索）
    Stage 2: 检索准备（历史、意图、source、按需改写、检索计划、查询变体、Prompt Profile）
    Stage 3: FAQ 检索，判断是否直出
    Stage 4: 文档检索
    Stage 5: 上下文构建
    Stage 6: LLM 流式生成
    Stage 7: 写历史记录、写 trace、发送结束事件

    参数：
        history: 对话历史管理器
        validate_source: source_filter 合法性校验回调函数
        query: 用户原始提问文本
        source_filter: 前端选择的业务分类过滤项
        session_id: 会话 ID
        kb_version: 请求指定的知识库版本号
        scenario_id: 业务场景 ID
        tenant_id: 租户 ID
        dataset_id: 数据集 ID
        visibility: 数据可见级别
        user_role: 用户主角色
        user_roles: 用户的全部角色列表

    返回：
        Generator yielding WebSocket 事件 dict（包含事件类型和数据）
    """
    # ── Stage 0: Create runtime context (scenario, data scope, session, trace_id, kb version) ──
    context = create_query_context(
        history=history,
        validate_source=validate_source,
        query=query,
        source_filter=source_filter,
        session_id=session_id,
        requested_kb_version=kb_version,
        scenario_id=scenario_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_role=user_role,
        user_roles=user_roles,
    )
    # 向前端发送"请求已接收"事件
    yield build_query_start_event(context)

    try:
        # ── Stage 1: Low-cost route decision ──
        # 统一处理确定性直答、边界拦截和 FAQ 精确命中；未命中才进入检索准备。
        yield build_status_event("正在进行查询路由...", context.session_id)
        route = decide_route(context)
        if route.answer:
            yield from _finish_with_single_answer(context, history, query, route.answer)
            return

        # ── Stage 2: Intent recognition + retrieval parameter preparation ──
        # 向前端发送"正在识别问题意图"状态事件
        yield build_status_event("正在识别问题意图...", context.session_id)
        # 生成下游检索参数包：历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile。
        prepared = prepare_retrieval(context)
        # 兜底保护：调试/未来扩展若在 prepare_retrieval 中产出直答，主链路仍直接收口。
        if prepared.intent.direct_answer:
            if context.hit_type == "unknown":
                context.hit_type = prepared.intent.intent.lower()
            yield from _finish_with_single_answer(context, history, query, prepared.intent.direct_answer)
            return

        # ── Stage 3-6: FAQ retrieval → doc retrieval → context building → LLM streaming ──
        # 执行检索+生成并产生状态/token/结束事件，返回 None 表示已内部收尾
        helper_result = yield from _search_and_generate(context, prepared, query, history)
        # _search_and_generate 内已 yield 收尾事件，无需继续走引用补强
        if helper_result is None:
            return

        answer_prepared = helper_result
        raw_answer = context.answer
        # 确保 RAG 答案带有可见来源编号，模型漏写时在末尾补充"参考来源"
        answer = enforce_answer_citations(raw_answer, answer_prepared.context_docs)
        # 引用补强后为空说明答案被来源验证完全清除（如全部编造），此时返回确定性信息不足提示
        if not answer:
            answer = f"信息不足，无法确认，请联系人工支持：{context.scenario.support_contact}。"
            yield from _finish_with_single_answer(context, history, query, answer, record_save_stage=True)
            return
        # 引用补强追加了新文本（如"参考来源：..."），将增量部分推送给前端
        elif answer != raw_answer:
            extra_token = answer[len(raw_answer):] if answer.startswith(raw_answer) else answer
            # 将补充的引用来源片段推送给前端
            yield build_token_event(extra_token, context.session_id)
            context.answer_parts = [answer]

        # ── Stage 7: Save history, write trace, send end event ──
        with context.stage("save_history"):
            # 将本轮问答写入 MySQL 历史表
            history.add_turn(context.session_id, query, answer)
        # 向前端发送 success 结束事件（含来源、耗时、意图、检索诊断信息）并写入 trace
        yield finish_success(context, answer=answer)
    except Exception as exc:
        logger.exception("QA stream failed")
        # 向前端发送 error 结束事件（WebSocket 不断连，前端显示可恢复的失败提示）
        yield finish_error(context, exc)


def _search_and_generate(context, prepared, query, history) -> Generator:
    """检索+生成子管线：FAQ 直出判断 → 文档检索 → 上下文构建 → LLM 流式生成。★★★ 核心

    执行流程：
    1. FAQ 检索：按查询变体召回 FAQ，判断是否分数直出（无需 LLM）
    2. 文档检索：按查询变体召回文档
    3. 上下文构建：合并 FAQ+文档候选中筛选出最终 prompt 上下文
    4. 上下文为空时返回"信息不足"兜底，不走 LLM
    5. LLM 流式生成：逐 chunk yield token 事件

    参数：
        context: RAGQueryContext 请求级状态
        prepared: RetrievalPreparation 检索参数包
        query: 用户原始提问
        history: 对话历史管理器

    返回：
        Generator yielding token/status/end 事件；
        function return value 为 None（已收尾）或 AnswerPreparation（需上游继续引用补强）
    """
    # ── Stage 3: FAQ retrieval with direct-answer bypass ──
    # 向前端发送"正在检索 FAQ 知识库"状态事件
    yield build_status_event("正在检索业务 FAQ 知识库...", context.session_id)
    # 按检索计划查询 FAQ 集合
    faq_result = search_faq(context, prepared)
    # 判断 FAQ 是否达到直出条件（精确匹配或分数超阈值）
    direct_answer = get_faq_direct_answer(context, prepared, faq_result)
    # FAQ 检索分数超阈值时可直接返回标准答案，无需 LLM 生成
    if direct_answer:
        context.hit_type = "faq_direct"
        context.sources = faq_result.source_payloads()
        yield from _finish_with_single_answer(context, history, query, direct_answer)
        return None

    # ── Stage 4: Document retrieval ──
    # 向前端发送"正在匹配业务资料"状态事件
    yield build_status_event("正在匹配相关业务资料...", context.session_id)
    # 按检索计划查询文档集合
    doc_result = search_doc(context, prepared)
    # ── Stage 5: Answer context preparation ──
    # 将检索结果整理成最终 Prompt、引用来源和命中类型
    answer_prepared = prepare_answer(context, prepared, faq_result, doc_result)
    context.sources = answer_prepared.sources
    context.hit_type = answer_prepared.hit_type

    # 合并后上下文仍为空（所有候选低于分数阈值），直接返回确定性"信息不足"避免 LLM 幻觉
    if context.hit_type == "insufficient_context":
        answer = build_insufficient_context_answer(context)
        yield from _finish_with_single_answer(context, history, query, answer, record_save_stage=True)
        return None

    # ── Stage 6: LLM streaming generation ──
    # 向前端发送"正在生成回答"状态事件
    yield build_status_event("正在生成回答...", context.session_id)
    with context.stage("llm_generation"):
        # 调用 LangChain ChatOpenAI 流式接口，逐 chunk 产生 token
        for chunk in stream_llm_answer(answer_prepared.system_prompt, answer_prepared.user_prompt):
            token = str(getattr(chunk, "content", "") or "")
            # LangChain 可能产出空 content 的 chunk（如 finish_reason 块），跳过
            if not token:
                continue
            context.answer_parts.append(token)
            context.mark_first_token()
            # 向 WebSocket 推送当前 token 片段，前端逐字展示
            yield build_token_event(token, context.session_id)

    return answer_prepared


def _finish_with_single_answer(
    context,
    history,
    query: str,
    answer: str,
    *,
    record_save_stage: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """无需 LLM 流式的答案收口：发 token → 写历史 → 写 trace → 发 end，四类直出分支共用。★★★ 核心

    处理步骤：发 token → 写历史 → 写 trace → 发 end。
    FAQ 直出、直接意图、信息不足兜底四类已有完整答案的分支统一走此收尾。

    参数：
        context: RAGQueryContext 请求级状态
        history: 对话历史管理器
        query: 用户原始提问
        answer: 完整答案文本（非流式，一次性推送）
        record_save_stage: 是否包裹 stage 计时上下文（信息不足等兜底分支需要）

    返回：
        Generator yielding token 事件 + success 结束事件
    """
    context.answer_parts = [answer]
    context.mark_first_token()
    # 将完整答案作为单次 token 推送（非流式场景直接透传）
    yield build_token_event(answer, context.session_id)
    # record_save_stage=True 时包裹 stage 计时（信息不足等兜底分支），
    # 其他简单直出路径不加额外包装直接写入历史
    if record_save_stage:
        with context.stage("save_history"):
            history.add_turn(context.session_id, query, answer)
    else:
        history.add_turn(context.session_id, query, answer)
    # 向前端发送 success 结束事件并写入 trace
    yield finish_success(context, answer=answer)

def debug_retrieval(
    history,
    validate_source,
    query: str,
    source_filter: str | None,
    session_id: str | None = None,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    user_role: str | None = None,
    user_roles: list[str] | None = None,
) -> dict[str, Any]:
    """检索半链路调试入口：只做检索准备+FAQ/文档检索，不调用 LLM，用于诊断召回质量和耗时。★★★ 核心

    执行流程：
    1. 创建运行时上下文（场景/数据域/会话/trace/知识库版本）
    2. 检索准备：历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile
    3. FAQ 检索 + 文档检索（检索计划禁用某一路时返回空结果）
    4. 汇总阶段耗时和检索诊断信息

    参数：
        history: 对话历史管理器
        validate_source: source_filter 合法性校验回调
        query: 用户原始提问
        source_filter: 前端选择的业务分类过滤项
        session_id: 会话 ID
        kb_version: 请求指定的知识库版本号
        scenario_id: 业务场景 ID
        tenant_id: 租户 ID
        dataset_id: 数据集 ID
        visibility: 数据可见级别
        user_role: 用户主角色
        user_roles: 用户的全部角色列表

    返回：
        dict: 检索调试诊断数据包，包含完整检索链路中的参数和结果
    """
    # ── Stage 0: Create runtime context ──
    context = create_query_context(
        history=history,
        validate_source=validate_source,
        query=query,
        source_filter=source_filter,
        session_id=session_id,
        requested_kb_version=kb_version,
        scenario_id=scenario_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_role=user_role,
        user_roles=user_roles,
    )
    # ── Stage 1: Intent, rewrite, retrieval plan, query variants ──
    # 完成意图识别、改写、检索计划和查询变体生成
    prepared = prepare_retrieval(context)

    # ── Stage 2: FAQ + doc retrieval (each returns empty result if plan disables it) ──
    # 按检索计划并行查询 FAQ 和文档集合
    faq_result = search_faq(context, prepared)
    doc_result = search_doc(context, prepared)
    # 将各阶段耗时写回 retrieval_info
    context.finalize_timings()

    # 构造调试诊断数据包，包含完整检索链路中的参数和结果
    return {
        "query": query,
        "rewritten_query": prepared.rewritten_query,
        "scenario_id": context.scenario.scenario_id,
        "scenario_name": context.scenario.display_name,
        "data_scope": context.data_scope.as_dict(),
        "tenant_id": context.data_scope.tenant_id,
        "dataset_id": context.data_scope.dataset_id,
        "visibility": context.data_scope.visibility,
        "source_filter": prepared.effective_source_filter,
        "kb_version": context.active_kb_version,
        "intent": prepared.intent.as_dict(),
        "classification": context.retrieval_info.get("classification") or {},
        "retrieval_plan": {
            **prepared.plan.as_dict(),
            "query_variants": prepared.query_variants,
            "prompt_profile": prepared.prompt_profile.as_dict(),
        },
        "stage_timings_ms": context.retrieval_info.get("stage_timings_ms") or {},
        "slowest_stage": context.retrieval_info.get("slowest_stage") or {},
        "faq_sources": faq_result.source_payloads(limit=10),
        "doc_sources": doc_result.source_payloads(limit=10),
    }
