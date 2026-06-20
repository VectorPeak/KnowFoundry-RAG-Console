"""FastAPI 对外请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalDebugRequest(BaseModel):
    """HTTP 检索诊断接口的请求体。在线问答不复用该模型。"""

    query: str = Field(..., min_length=1)
    source_filter: str | None = None
    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    user_role: str | None = None
    user_roles: list[str] = Field(default_factory=list)
    kb_version: str | None = None


class FeedbackRequest(BaseModel):
    """用户反馈载荷，rating 约束为 useful/not_useful。"""

    session_id: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(useful|not_useful)$")
    comment: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalDebugResponse(BaseModel):
    """检索调试响应，不包含最终答案，faq 和 doc 来源分开返回。"""

    query: str
    rewritten_query: str
    source_filter: str | None = None
    scenario_id: str | None = None
    tenant_id: str | None = None
    dataset_id: str | None = None
    visibility: str | None = None
    data_scope: dict[str, Any] | None = None
    kb_version: str | None = None
    intent: dict[str, Any]
    retrieval_plan: dict[str, Any]
    faq_sources: list[dict[str, Any]] = Field(default_factory=list)
    doc_sources: list[dict[str, Any]] = Field(default_factory=list)
