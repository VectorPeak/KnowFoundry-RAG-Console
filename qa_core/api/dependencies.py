"""FastAPI 共享依赖：管理令牌校验和轻量限流。属于协议层横切能力。"""

from __future__ import annotations
import time
from collections import defaultdict, deque
from fastapi import Header, Request, WebSocket
from qa_core.api.error_handlers import raise_server_error, raise_too_many_requests, raise_unauthorized
from qa_core.config.settings import get_settings
settings = get_settings()
RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)

def client_key(scope: Request | WebSocket) -> str:
    """从 HTTP/WebSocket 连接中提取客户端 IP 作为限流 key。
    """
    client = getattr(scope, "client", None)
    if client and getattr(client, "host", None):
        return str(client.host)
    return "local"

def check_rate_limit(key: str) -> bool:
    """执行进程内滑动窗口限流，避免频繁请求打爆 LLM/Milvus。
    """
    limit = max(int(settings.api_rate_limit_per_minute or 0), 0)
    # 原因： 滑动窗口比固定窗口更平滑，不会在整点边界出现突发流量；滑动窗口只在每次请求时惰性清理过期记录，省去定时器开销
    if limit <= 0:
        return True
    now = time.time()
    bucket = RATE_BUCKETS[key]
    while bucket and now - bucket[0] >= 60:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True

def enforce_http_rate_limit(request: Request) -> None:
    """HTTP 请求限流依赖，超限时返回 429。
    """
    if not check_rate_limit(client_key(request)):
        raise_too_many_requests("请求过于频繁，请稍后再试。")

def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    """校验管理接口令牌，未配置或令牌不匹配时返回 500/401。
    """
    # 原因： 管理接口（重索引、版本激活、数据清理）一旦被滥用可能破坏生产数据，需要独立于用户 token 的专用令牌鉴权
    expected = settings.admin_api_token.strip()
    if not expected:
        raise_server_error("ADMIN_API_TOKEN 未配置")
    if x_admin_token != expected:
        raise_unauthorized("管理接口令牌无效")
