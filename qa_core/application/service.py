"""应用层 RAG 编排服务。接收 API 层的 HTTP/WebSocket 请求，依次完成场景解析、数据域隔离、意图分类、
预检直答或流式 RAG 主链路（改写 -> FAQ 检索 -> 置信度判断 -> 文档检索 -> 生成 -> 保存），
并将最终结果以同步 QAResult 或异步 Generator 事件流的形式返回给 API 层。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from typing import Any

from qa_core.application.results import QAResult
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.memory.history import get_history_store
from qa_core.intent.classifier import classify_intent
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.retrieval.filters import validate_source_filter
from qa_core.config.logging_config import get_logger
from qa_core.pipeline.rag import debug_retrieval as rag_debug_retrieval
from qa_core.pipeline.rag import stream_query as rag_stream_query
from qa_core.scenarios.registry import ScenarioDefinition, resolve_scenario
from qa_core.config.settings import get_settings
from qa_core.pipeline.runtime import RAGQueryContext

logger = get_logger(__name__)

class QAService:
    """核心 RAG 问答工作流，按业务场景分两条路径编排：

    执行流程（双路径）：
    路径 A — 预检直答（preview_query）：
      1. resolve_scenario() 解析请求使用哪个业务场景配置
      2. resolve_data_scope() 构建数据域隔离对象（租户/数据集/可见级别/角色）
      3. validate_source() 校验 source_filter 白名单
      4. get_context_messages() 加载会话历史
      5. classify_intent() 规则+LLM 意图分类
      6. 问候/越界/客服电话等 → 直接构造 QAResult 并 record_trace() 后返回
      7. 复杂问题 → 返回 None，前端转 WebSocket 走路径 B

    路径 B — 流式 RAG 主链路（stream_query）：
      1. 复用路径 A 的步骤 1-5（由 API 层确保已执行过预检）
      2. rag_stream_query() 执行改写 -> FAQ 检索 -> 置信度判断 -> 文档检索 -> LLM 生成
      3. token 逐块通过 Generator[dict] 推送给 API 层 WebSocket 转发
    """

    def __init__(self) -> None:
        """初始化共享服务层依赖。

        执行流程：
        1. get_settings() 加载应用全局配置（限流、历史摘要开关等）
        2. get_history_store() 获取 MySQL 历史记录适配器单例

        说明： 构造函数不保存任何请求级状态，QAService 实例由
        factory.get_qa_service() 以进程级单例方式管理。
        """
        self.settings = get_settings()
        self.history = get_history_store()

    def validate_source(self, source_filter: str | None, scenario: ScenarioDefinition) -> None:
        """校验 source_filter 是否在当前业务场景的合法来源白名单内。

        在 Milvus 检索之前调用，提前拒绝非法分类过滤条件。

        参数：
            source_filter: 前端传入的业务分类过滤项（可空，空时跳过校验）。
            scenario: resolve_scenario() 解析出的当前请求的场景定义，
                      从中读取 valid_sources 白名单。

        异常：
            ValueError: 当 source_filter 不为 None 且不在 scenario.valid_sources 中时抛出。
        """
        # 委托至 governance 层做实际的集合包含关系判断
        validate_source_filter(source_filter, scenario.valid_sources)

    def preview_query(
        self,
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
    ) -> QAResult | None:
        """轻量预检入口：问候/越界/客服等可直接回答的场景直接返回 QAResult；
        复杂 RAG 问题返回 None，由 API 层让前端改用 WebSocket 走流式主链路。

        执行流程：
        1. resolve_scenario(scenario_id)           — 解析业务场景配置
        2. resolve_data_scope(...)                 — 构建数据域隔离（租户/数据集/可见级别/角色）
        3. self.history.get_context_messages()     — 从 MySQL 加载会话历史摘要+最近消息
        4. classify_intent(...)                    — 规则优先+LLM 补充的意图识别
        5. intent.direct_answer 非空 → 直答路径：
           a. history.add_turn() 写入本轮问答
           b. resolve_active_kb_version() 解析当前 KB 版本
           c. 构造 RAGQueryContext, record_trace() 记录 trace
           d. 返回 QAResult
        6. intent.direct_answer 为空 → 返回 None，留给 stream_query 处理

        参数：
            query: 用户输入的问题文本。
            source_filter: 前端选择的业务分类过滤项（可为 None）。
            session_id: 会话 ID，未传则自动生成 UUID。
            kb_version: 知识库版本号（可选，不传则自动解析最新版）。
            scenario_id: 业务场景标识（可选，不传则使用默认场景）。
            tenant_id: 租户 ID（可选，用于多租户数据隔离）。
            dataset_id: 数据集 ID（可选，用于数据域隔离）。
            visibility: 可见级别（可选，用于数据域隔离）。
            user_role: 用户角色（可选，用于数据域隔离）。
            user_roles: 用户角色列表（可选，用于数据域隔离）。

        返回：
            QAResult: 直答结果（问候/越界/客服等），包含 answer、hit_type、session_id、intent 等。
            None: 表示需要走 WebSocket 流式 RAG 主链路。
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
        session_id = session_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        self.validate_source(source_filter, scenario)
        # 从 MySQL 获取历史摘要+最近消息，用于意图识别和追问改写
        history_messages = self.history.get_context_messages(session_id)
        # 规则优先+LLM 补充的意图识别
        intent = classify_intent(query, history_messages, scenario)
        # 业务决策：问候/越界/客服电话等可直接回答的场景走直答，否则留待流式链路
        if intent.direct_answer:
            # 将本轮问答写入 MySQL 历史表
            self.history.add_turn(session_id, query, intent.direct_answer)
            # 解析当前可用的知识库版本
            active_kb_version = resolve_active_kb_version(kb_version, scenario.scenario_id)
            elapsed_ms = (time.perf_counter() - started) * 1000
            # 将本轮问答 trace 写入持久化存储
            ctx = RAGQueryContext(
                history=history_messages,
                validate_source=self.validate_source,
                query=query,
                source_filter=source_filter,
                scenario=scenario,
                data_scope=data_scope,
                session_id=session_id,
                trace_id=trace_id,
                started=started,
                active_kb_version=active_kb_version,
            )
            ctx.hit_type = intent.intent.lower()
            ctx.intent_payload = intent.as_dict()
            ctx.retrieval_info = {"path": "preview_direct"}
            ctx.record_trace(answer=intent.direct_answer, elapsed_ms=elapsed_ms)
            return QAResult(
                answer=intent.direct_answer,
                hit_type=intent.intent.lower(),
                session_id=session_id,
                intent=intent.as_dict(),
                retrieval={
                    "scenario_id": scenario.scenario_id,
                    "scenario_name": scenario.display_name,
                    "data_scope": data_scope.as_dict(),
                    "kb_version": active_kb_version,
                    "trace_id": trace_id,
                },
                elapsed_ms=elapsed_ms,
            )
        # 复杂问题统一走 WebSocket，检索和大模型生成只执行一次，token 流式推送
        return None

    def stream_query(
        self,
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
        """完整流式问答入口。委托 RAGPipeline 执行改写 -> 检索 -> 生成全链路，
        以事件生成器（status / token / end / error）形式逐块产出，由 API 层 WebSocket 转发。

        执行流程（由 rag_stream_query 内部完成）：
        1. rewrite_query()            — 结合历史对话改写用户追问
        2. retrieve_faq()             — 从 Milvus FAQ 集合检索
        3. confidence_check()          — 对 FAQ 结果做置信度判断
        4. retrieve_docs()            — 低置信度时执行文档 RAG 检索
        5. llm_generate()             — 调用大模型生成回答，token 逐块 yield
        6. history_add_turn()         — 将本轮问答写入 MySQL 历史表
        7. record_trace()             — 将完整 trace 写入持久化存储

        参数：
            query: 用户输入的问题文本。
            source_filter: 前端选择的业务分类过滤项（可为 None）。
            session_id: 会话 ID，未传则自动生成 UUID。
            kb_version: 知识库版本号（可选）。
            scenario_id: 业务场景标识（可选）。
            tenant_id: 租户 ID（可选）。
            dataset_id: 数据集 ID（可选）。
            visibility: 可见级别（可选）。
            user_role: 用户角色（可选）。
            user_roles: 用户角色列表（可选）。

        Yields:
            dict[str, Any]: 流式事件，包含 type 字段，可能取值：
                - "status": 阶段状态变更事件
                - "token": LLM 生成的文本片段
                - "end": 问答结束事件
                - "error": 错误事件
        """
        # 委托至 pipeline 层 rag_stream_query 执行完整 RAG 全链路，逐事件 yield
        yield from rag_stream_query(
            self.history,
            self.validate_source,
            query,
            source_filter,
            session_id,
            kb_version=kb_version,
            scenario_id=scenario_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_role=user_role,
            user_roles=user_roles,
        )

    def debug_retrieval(
        self,
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
        """检索调试半链路入口。委托 RAGPipeline 执行改写 + FAQ 检索 + 文档检索，
        但不调用最终回答 LLM，用于开发者诊断检索质量。

        执行流程（由 rag_debug_retrieval 内部完成）：
        1. rewrite_query()            — 结合历史对话改写用户追问
        2. retrieve_faq()             — 从 Milvus FAQ 集合检索
        3. retrieve_docs()            — 从 Milvus 文档集合检索
        4. 组装调试信息并返回（含意图、改写、检索计划、FAQ/文档命中）
        5. record_trace()             — 记录调试 trace（如启用）

        参数：
            query: 用户输入的问题文本。
            source_filter: 前端选择的业务分类过滤项（可为 None）。
            session_id: 会话 ID（可选）。
            kb_version: 知识库版本号（可选）。
            scenario_id: 业务场景标识（可选）。
            tenant_id: 租户 ID（可选）。
            dataset_id: 数据集 ID（可选）。
            visibility: 可见级别（可选）。
            user_role: 用户角色（可选）。
            user_roles: 用户角色列表（可选）。

        返回：
            dict[str, Any]: 检索诊断结果，包含意图分类、改写结果、FAQ 命中列表、
            文档命中列表、检索计划等调试信息。
        """
        # 委托至 pipeline 层 rag_debug_retrieval 执行检索半链路（不含 LLM 生成）
        return rag_debug_retrieval(
            self.history,
            self.validate_source,
            query,
            source_filter,
            session_id=session_id,
            kb_version=kb_version,
            scenario_id=scenario_id,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            visibility=visibility,
            user_role=user_role,
            user_roles=user_roles,
        )


