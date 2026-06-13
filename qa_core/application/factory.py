"""共享问答运行时对象的工厂方法。API 层（HTTP 路由和 WebSocket 路由）通过这里
获取进程级 lru_cache 缓存的服务实例，避免每次请求都重新初始化 QAService。

典型调用链：
  chat.py (API 路由) → factory.get_qa_service() → QAService 单例
"""

from __future__ import annotations

from functools import lru_cache

from qa_core.application.service import QAService

@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    """返回 HTTP 和 WebSocket 路由共用的 QAService 单例。

    由 @lru_cache(maxsize=1) 保证进程内只初始化一次。
    API 层通过此函数获取服务实例，而非直接实例化 QAService。

    执行流程：
    1. 首次调用 → QAService.__init__() 加载共享配置和历史记录适配器
    2. 后续调用 → lru_cache 命中，直接返回缓存实例

    返回：
        QAService: 进程级共享的问答编排服务实例。
    """
    return QAService()

