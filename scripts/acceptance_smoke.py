"""本地服务真实链路验收脚本。

该脚本假设 FastAPI 服务已经启动，例如：

python -m uvicorn app:app --host 127.0.0.1 --port 8000

它不再使用服务桩、假 QAService 或内存替身，而是通过 HTTP/WebSocket 访问真实服务。
因此它会验证当前项目是否真正通电：健康检查、场景列表、页面、管理摘要、流式回答
事件，以及真实 RAG 主链路能否返回 end 事件。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

import websocket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.config.settings import get_settings
from scripts.common import configure_utf8_stdio, write_optional_json


def http_json(base_url: str, path: str, token: str = "") -> dict:
    """读取 JSON 接口，失败时抛出带路径的异常。"""
    req = urlrequest.Request(base_url.rstrip("/") + path, headers=_headers(token))
    try:
        with urlrequest.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"HTTP check failed: {path}: {exc}") from exc


def http_text(base_url: str, path: str) -> str:
    """读取文本页面，确认页面可访问。"""
    req = urlrequest.Request(base_url.rstrip("/") + path)
    try:
        with urlrequest.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Page check failed: {path}: {exc}") from exc


def websocket_events(
    base_url: str,
    token: str = "",
    query: str = "新人入职流程怎么走",
    scenario_id: str = "enterprise_knowledge",
    max_events: int = 1000,
    max_seconds: float = 180.0,
) -> list[dict]:
    """连接真实 WebSocket 服务并收集一次问答事件。

    这里要求 `websocket-client` 依赖已经安装。它不会 monkey patch 应用对象，也不会绕过
    MySQL、Milvus、LLM 或本地模型，因此可以作为“核心功能是否闭环”的验收入口。
    """
    parsed = urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{scheme}://{parsed.netloc}/api/stream"
    headers = [f"X-Admin-Token: {token}"] if token else []
    ws = websocket.create_connection(ws_url, timeout=90, header=headers)
    try:
        ws.send(
            json.dumps(
                {
                    "query": query,
                    "session_id": "acceptance-smoke-session",
                    "scenario_id": scenario_id,
                },
                ensure_ascii=False,
            )
        )
        events: list[dict] = []
        deadline = time.monotonic() + max_seconds
        while len(events) < max_events and time.monotonic() < deadline:
            event = json.loads(ws.recv())
            events.append(event)
            if event.get("type") == "error":
                raise RuntimeError(f"WebSocket returned error: {event.get('error')}")
            if event.get("type") == "end":
                return events
        raise RuntimeError(f"WebSocket did not finish, received {len(events)} events")
    finally:
        ws.close()


def _headers(token: str) -> dict[str, str]:
    """构造管理令牌 header。"""
    return {"X-Admin-Token": token} if token else {}


def resolve_admin_token(raw_token: str | None) -> str:
    """解析本次验收使用的管理令牌。

    管理接口已经要求强令牌保护，但验收脚本不应该要求每次手动复制令牌：
    - 命令行显式传入时优先使用，方便临时验收远端环境；
    - 未传入时读取当前运行配置的 `ADMIN_API_TOKEN`，保持本地和容器内一键验收体验；
    - 返回报告只展示验收结果，不展示令牌本身，避免敏感信息进入日志。
    """
    return (raw_token or get_settings().admin_api_token or "").strip()


def main() -> None:
    """执行真实服务验收并打印 JSON 摘要。"""
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run real acceptance smoke checks against the FastAPI service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--admin-token", default="", help="为空时自动读取当前运行配置中的 ADMIN_API_TOKEN。")
    parser.add_argument("--query", default="新人入职流程怎么走")
    parser.add_argument("--scenario", default="enterprise_knowledge", help="Scenario id used by the WebSocket smoke query.")
    parser.add_argument("--max-events", type=int, default=1000, help="最多接收的 WebSocket 事件数量。")
    parser.add_argument("--max-seconds", type=float, default=180.0, help="最多等待 WebSocket end 事件的秒数。")
    parser.add_argument("--output", default="", help="可选 JSON 报告输出路径。")
    args = parser.parse_args()
    admin_token = resolve_admin_token(args.admin_token)

    checks: dict[str, object] = {}
    health = http_json(args.base_url, "/health")
    scenarios = http_json(args.base_url, "/api/scenarios")
    page = http_text(args.base_url, "/")
    admin = http_text(args.base_url, "/admin")
    langsmith = http_json(args.base_url, "/api/admin/langsmith", admin_token)
    events = websocket_events(args.base_url, admin_token, args.query, args.scenario, args.max_events, args.max_seconds)
    event_types = [event.get("type") for event in events]

    checks["health"] = health.get("status") == "healthy"
    checks["scenarios"] = bool(scenarios.get("scenarios"))
    checks["page"] = "KnowForge RAG Platform" in page
    checks["admin_page"] = "KnowForge 状态页" in admin
    checks["admin_langsmith"] = "enabled" in langsmith and "project" in langsmith
    checks["websocket_events"] = "start" in event_types and "token" in event_types and "end" in event_types
    ok = all(bool(value) for value in checks.values())
    payload = {
        "report_type": "acceptance_smoke",
        "ok": ok,
        "base_url": args.base_url,
        "checks": checks,
        "active_scenario_id": health.get("active_scenario_id"),
        "smoke_scenario_id": args.scenario,
        "websocket_event_types": event_types,
    }
    write_optional_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
