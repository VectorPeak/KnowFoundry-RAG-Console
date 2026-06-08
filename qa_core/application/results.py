"""应用服务层返回给 API 层的内部结果对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QAResult:
    """非流式预览路径返回的内部答案对象。

    它不是 FastAPI 的响应模型，而是 `QAService.preview_query()` 和 API 路由之间的内部
    传递对象。单独放在 application 层，可以让 `schemas.py` 只保留对外接口模型，降低
    初学者阅读时的概念混杂。
    """

    answer: str
    hit_type: str
    session_id: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    rewritten_query: str | None = None
    intent: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
