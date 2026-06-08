"""RAG 单次请求运行上下文：请求级状态、阶段计时、统一收尾（LangSmith trace/事件）。
"""

from __future__ import annotations
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from qa_core.governance.data_scope import DataScope, resolve_data_scope
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.observability.langsmith_adapter import record_query_trace
from qa_core.pipeline.events import end_event as build_end_event
from qa_core.pipeline.events import error_event as build_error_event
from qa_core.pipeline.events import start_event as build_start_event
from qa_core.scenarios.registry import ScenarioDefinition, resolve_scenario
from qa_core.retrieval.results import RetrievalResult

ValidateSource = Callable[[str | None, ScenarioDefinition], None]
T = TypeVar("T")

@dataclass
class RAGQueryContext:
    """一次 RAG 请求在后端流转时共享的请求级状态，不跨请求复用。★★★ 核心
    核心职责：
    - 承载用户输入和运行时推导出的场景、数据域、trace 等元信息
    - 记录各阶段耗时（record_stage / run_stage / stage）用于慢链路诊断
    - 提供 mark_first_token 记录首 token 耗时（关键用户体验指标）
    - 提供 record_trace / finalize_timings 用于请求结束时的持久化归档
    - 缓存 FAQ fast path 检索结果避免重复检索
    """

    history: Any                                    # 对话历史管理器（用于加载历史+写入新对话）
    validate_source: ValidateSource                 # 前端 source_filter 合法性校验回调
    query: str                                      # 用户原始提问
    source_filter: str | None                       # 前端选择的业务分类过滤项
    scenario: ScenarioDefinition                    # 当前解析到的业务场景定义
    data_scope: DataScope                           # 数据域隔离信息（租户/数据集/可见级别）
    session_id: str                                 # 会话 ID（前端传入或自动生成）
    trace_id: str                                   # 全链路追踪 ID（自动生成 UUID）
    started: float                                  # time.perf_counter() 请求开始时间戳
    active_kb_version: str                          # 当前生效的知识库版本号
    answer_parts: list[str] = field(default_factory=list)   # LLM 流式输出的 token 片段列表
    sources: list[dict[str, Any]] = field(default_factory=list)   # 引用来源列表（供前端展示）
    hit_type: str = "unknown"                       # 命中类型: unknown/faq_direct/scenario_boundary/rag/insufficient_context
    rewritten_query: str | None = None              # 追问改写后的独立检索问题
    intent_payload: dict[str, Any] = field(default_factory=dict)  # 意图识别结果快照
    retrieval_info: dict[str, Any] = field(default_factory=dict)  # 检索诊断信息（耗时/分数/参数）
    stage_timings_ms: dict[str, float] = field(default_factory=dict)  # 各阶段耗时（毫秒）
    first_token_ms: float | None = None             # 首 token 耗时（记录后不再更新）
    fast_faq_result: RetrievalResult | None = None  # FAQ fast path 检索结果缓存，供主链路复用
    fast_faq_source_filter: str | None = None       # fast path 缓存对应的 source_filter 参数

    @property
    def answer(self) -> str:
        """拼接流式输出的 token 片段并返回已生成的完整答案文本。

        返回：
            str: 当前已累积的所有 token 拼接后的完整文本（首尾去空白）
        """
        return "".join(self.answer_parts).strip()

    def record_stage(self, stage_name: str, started: float) -> float:
        """记录一个主链路阶段的耗时，写入 stage_timings_ms 供 trace 和诊断使用。

        参数：
            stage_name: 阶段名称（如"faq_retrieval"、"classify_intent"）
            started: time.perf_counter() 起始时间戳（由调用侧记录）

        返回：
            float: 本阶段经过的毫秒数（保留 2 位小数）
        """
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.stage_timings_ms[stage_name] = round(elapsed_ms, 2)
        return elapsed_ms

    def run_stage(self, stage_name: str, action: Callable[[], T]) -> T:
        """执行一个阶段并自动记录耗时，避免每个步骤重复写计时代码。

        参数：
            stage_name: 阶段名称（用于 stage_timings_ms 的 key）
            action: 阶段执行回调，返回值即为 run_stage 的返回值

        返回：
            T: action 回调的返回值
        """
        started = time.perf_counter()
        try:
            return action()
        finally:
            self.record_stage(stage_name, started)

    @contextmanager
    def stage(self, stage_name: str):
        """上下文管理器：包裹多行阶段并自动记录耗时（如 LLM 流式生成循环）。"""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_stage(stage_name, started)

    def mark_first_token(self) -> None:
        """记录首次 token 时间，仅首次调用生效。

        首 token 耗比比总耗时更能代表用户体感（用户感知的等待时间约等于首 token 到达时间）。
        """
        if self.first_token_ms is None:
            self.first_token_ms = round((time.perf_counter() - self.started) * 1000, 2)

    def record_trace(self, answer: str, elapsed_ms: float, error: str | None = None) -> None:
        """将完整 trace metadata 发送到 LangSmith。★★★ 核心

        参数：
            answer: 最终答案文本（流式拼接后的完整内容）
            elapsed_ms: 请求总耗时（毫秒）
            error: 异常信息（正常请求为 None，异常请求传入 str(error)）
        """
        record_query_trace(
            trace_id=self.trace_id,
            session_id=self.session_id,
            question=self.query,
            answer=answer,
            hit_type=self.hit_type,
            scenario=self.scenario,
            data_scope=self.data_scope.as_dict(),
            source_filter=self.retrieval_info.get("source_filter") or self.source_filter,
            kb_version=self.active_kb_version,
            rewritten_query=self.rewritten_query,
            intent=self.intent_payload,
            retrieval=self.retrieval_info,
            sources=self.sources,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    def finalize_timings(self) -> None:
        """汇总各阶段耗时到 retrieval_info，供结束事件和 trace 使用。

        执行流程：
        1. 计算总耗时（当前时间 - started）
        2. 将 stage_timings_ms 和 total_elapsed_ms 写回 retrieval_info
        3. 找出耗时最长阶段标记为 slowest_stage
        """
        total_elapsed_ms = round((time.perf_counter() - self.started) * 1000, 2)
        self.retrieval_info["stage_timings_ms"] = dict(self.stage_timings_ms)
        self.retrieval_info["first_token_ms"] = self.first_token_ms
        self.retrieval_info["total_elapsed_ms"] = total_elapsed_ms
        if self.stage_timings_ms:
            slowest_stage = max(self.stage_timings_ms.items(), key=lambda item: item[1])
            self.retrieval_info["slowest_stage"] = {
                "name": slowest_stage[0],
                "elapsed_ms": slowest_stage[1],
            }


def create_query_context(
    *,
    history: Any,
    validate_source: ValidateSource,
    query: str,
    source_filter: str | None,
    session_id: str | None,
    requested_kb_version: str | None,
    scenario_id: str | None,
    tenant_id: str | None,
    dataset_id: str | None,
    visibility: str | None,
    user_role: str | None,
    user_roles: list[str] | None,
) -> RAGQueryContext:
    """创建单次 RAG 请求上下文：解析业务场景、数据域隔离、会话号、trace_id 和知识库版本。★★★ 核心

    执行流程：
    1. 根据 scenario_id 解析当前请求的业务场景配置
    2. 构建数据域隔离信息（租户/数据集/可见级别/角色）
    3. 实例化 RAGQueryContext，含自动生成的 session_id 和 trace_id
    4. 解析知识库版本（请求指定 > 环境变量 > 版本清单 active）

    参数：
        history: 对话历史管理器
        validate_source: source_filter 合法性校验回调
        query: 用户原始提问
        source_filter: 前端选择的业务分类过滤项
        session_id: 前端传入的会话 ID（None 时自动生成）
        requested_kb_version: 请求指定的知识库版本号
        scenario_id: 业务场景 ID（None 时使用默认场景）
        tenant_id: 租户 ID
        dataset_id: 数据集 ID
        visibility: 数据可见级别
        user_role: 用户主角色
        user_roles: 用户的全部角色列表

    返回：
        RAGQueryContext: 包含完整请求级状态和计时起点的上下文字典
    """
    # 解析当前请求使用的业务场景配置
    scenario = resolve_scenario(scenario_id)
    # 构建数据域隔离信息（租户、数据集、可见级别、角色），用于 Milvus 检索过滤
    data_scope = resolve_data_scope(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_role=user_role,
        user_roles=user_roles,
    )
    return RAGQueryContext(
        history=history,
        validate_source=validate_source,
        query=query,
        source_filter=source_filter,
        scenario=scenario,
        data_scope=data_scope,
        session_id=session_id or str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        started=time.perf_counter(),
        # 解析当前请求可用的知识库版本（请求指定 > 环境变量 > 版本清单 active）
        active_kb_version=resolve_active_kb_version(requested_kb_version, scenario.scenario_id),
    )


def start_event(context: RAGQueryContext) -> dict[str, Any]:
    """构造本轮请求的 WebSocket start 事件。

    参数：
        context: RAGQueryContext（提供 session_id、trace_id、scenario、data_scope、active_kb_version）

    返回：
        dict: WebSocket start 事件（含会话号、trace_id、场景、数据域、知识库版本）
    """
    # 构造 WebSocket start 事件（含会话号、trace_id、场景、数据域）
    return build_start_event(
        session_id=context.session_id,
        trace_id=context.trace_id,
        scenario_id=context.scenario.scenario_id,
        scenario_name=context.scenario.display_name,
        data_scope=context.data_scope.as_dict(),
        kb_version=context.active_kb_version,
    )


def finish_success(context: RAGQueryContext, *, answer: str) -> dict[str, Any]:
    """成功结束事件构造并写入 LangSmith trace，FAQ 直出和 RAG 生成两类场景共用此收尾。★★★ 核心

    执行流程：
    1. 调用 finalize_timings 汇总各阶段耗时
    2. 构造 WebSocket end 事件（含来源列表、耗时、意图、检索诊断信息）
    3. 将 trace metadata 发送到 LangSmith 供复盘使用

    参数：
        context: RAGQueryContext 当前请求上下文
        answer: 最终答案文本

    返回：
        dict: WebSocket end 事件，供 rag.py yield 给前端
    """
    context.finalize_timings()
    # 构造 WebSocket 结束事件（含来源、耗时、意图、检索诊断信息）
    final_event = build_end_event(
        session_id=context.session_id,
        hit_type=context.hit_type,
        sources=context.sources,
        started=context.started,
        rewritten_query=context.rewritten_query,
        trace_id=context.trace_id,
        intent=context.intent_payload,
        retrieval=context.retrieval_info,
    )
    # 将 trace metadata 发送到 LangSmith，供诊断和 dataset 沉淀使用
    context.record_trace(answer=answer, elapsed_ms=final_event["processing_time"] * 1000)
    return final_event


def finish_error(context: RAGQueryContext, error: Exception) -> dict[str, Any]:
    """异常结束处理：记录异常 LangSmith trace 并构造 error 事件（WebSocket 不断连，前端显示可恢复提示）。★★★ 核心

    参数：
        context: RAGQueryContext 当前请求上下文
        error: 捕获的异常对象

    返回：
        dict: WebSocket error 事件，供 rag.py yield 给前端
    """
    context.finalize_timings()
    elapsed_ms = (time.perf_counter() - context.started) * 1000
    # 将异常 trace metadata 发送到 LangSmith，包含错误信息
    context.record_trace(answer="", elapsed_ms=elapsed_ms, error=str(error))
    # 构造 WebSocket error 事件（可恢复的失败提示，页面不崩溃）
    return build_error_event(error=str(error), session_id=context.session_id, trace_id=context.trace_id)
