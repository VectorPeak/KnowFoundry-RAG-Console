"""API 层传给 QAService 的公共调用上下文。

WebSocket stream 和 retrieval debug 都需要同一组字段：场景、租户、数据集、
可见性、角色和知识库版本。集中到这个小对象后，路由层只负责解析请求，业务规则仍在
QAService 和 pipeline 中。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qa_core.schemas import RetrievalDebugRequest


@dataclass(frozen=True)
class QueryServiceContext:
    """一次 API 请求调用 QAService 所需的稳定参数包。"""

    query: str
    source_filter: str | None
    session_id: str
    kb_version: str | None
    scenario_id: str | None
    tenant_id: str | None
    dataset_id: str | None
    visibility: str | None
    user_role: str | None
    user_roles: list[str] | None

    @classmethod
    def from_debug_request(
        cls,
        request: RetrievalDebugRequest,
        *,
        session_id: str | None = None,
    ) -> "QueryServiceContext":
        """从 HTTP 检索诊断请求模型构造 QAService 调用上下文。"""
        return cls(
            query=request.query.strip(),
            source_filter=request.source_filter,
            session_id=session_id or request.session_id or str(uuid.uuid4()),
            kb_version=request.kb_version,
            scenario_id=request.scenario_id,
            tenant_id=request.tenant_id,
            dataset_id=request.dataset_id,
            visibility=request.visibility,
            user_role=request.user_role,
            user_roles=request.user_roles,
        )

    @classmethod
    def from_ws_payload(cls, payload: dict[str, Any]) -> "QueryServiceContext":
        """从 WebSocket JSON 载荷构造 QAService 调用上下文。"""
        return cls(
            query=str(payload.get("query") or "").strip(),
            source_filter=payload.get("source_filter"),
            session_id=payload.get("session_id") or str(uuid.uuid4()),
            kb_version=payload.get("kb_version"),
            scenario_id=payload.get("scenario_id"),
            tenant_id=payload.get("tenant_id"),
            dataset_id=payload.get("dataset_id"),
            visibility=payload.get("visibility"),
            user_role=payload.get("user_role"),
            user_roles=payload.get("user_roles") or [],
        )

    def service_args(self) -> tuple[Any, ...]:
        """返回 QAService 方法当前使用的顺序参数。"""
        return (
            self.query,
            self.source_filter,
            self.session_id,
            self.kb_version,
            self.scenario_id,
            self.tenant_id,
            self.dataset_id,
            self.visibility,
            self.user_role,
            self.user_roles,
        )
