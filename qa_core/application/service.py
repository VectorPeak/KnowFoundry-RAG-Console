"""应用层 RAG 编排服务。接收 API 层的 WebSocket 或检索调试请求，依次完成场景解析、
数据域隔离、查询路由、检索准备、FAQ 检索、置信度判断、文档检索、生成和保存，
并将最终结果以 Generator 事件流或检索诊断字典的形式返回给 API 层。"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from qa_core.memory.history import get_history_store
from qa_core.retrieval.filters import validate_source_filter
from qa_core.config.logging_config import get_logger
from qa_core.pipeline.rag import debug_retrieval as rag_debug_retrieval
from qa_core.pipeline.rag import stream_query as rag_stream_query
from qa_core.scenarios.registry import ScenarioDefinition, resolve_scenario
from qa_core.config.settings import get_settings

logger = get_logger(__name__)

class QAService:
    """核心 RAG 问答工作流，统一通过流式主链路编排：

    执行流程：
      1. API 层通过 WebSocket 接收问题并构造 QueryServiceContext。
      2. stream_query() 委托 pipeline 执行上下文创建、查询路由、检索准备、
         FAQ/文档检索、LLM 生成、历史保存和 Trace。
      3. token 与状态事件逐块通过 Generator[dict] 推送给 API 层 WebSocket 转发。
      4. debug_retrieval() 复用同一套检索准备和检索逻辑，但不调用 LLM。
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
        """完整流式问答入口。委托 RAGPipeline 执行查询路由 -> 检索准备 -> 检索 -> 生成全链路，
        以事件生成器（status / token / end / error）形式逐块产出，由 API 层 WebSocket 转发。

        执行流程（由 rag_stream_query 内部完成）：
        1. decide_route()             — 统一处理直答、FAQ 精确路由或继续检索
        2. prepare_retrieval()        — 历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile
        3. search_faq()               — 从 Milvus FAQ 集合检索并判断是否标准直出
        4. search_doc()               — 低置信度时执行文档 RAG 检索
        5. stream_llm_answer()        — 调用大模型生成回答，token 逐块 yield
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
        """检索调试半链路入口。委托 RAGPipeline 执行检索准备 + FAQ 检索 + 文档检索，
        但不调用最终回答 LLM，用于开发者诊断检索质量。

        执行流程（由 rag_debug_retrieval 内部完成）：
        1. prepare_retrieval()        — 历史、意图、source、按需改写、检索计划、查询变体和 Prompt Profile
        2. search_faq()               — 从 Milvus FAQ 集合检索
        3. search_doc()               — 从 Milvus 文档集合检索
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


