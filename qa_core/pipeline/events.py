"""WebSocket 流式问答事件构造器：start / status / token / end / error 五种事件格式集中管理。
"""

from __future__ import annotations
import time
from typing import Any

def start_event(
    *,
    session_id: str,
    trace_id: str,
    scenario_id: str,
    scenario_name: str,
    data_scope: dict[str, Any],
    kb_version: str | None,
) -> dict[str, Any]:
    """通知前端 WebSocket 连接已建立、请求已接收，前端据此展示加载状态。
    """
    return {
        "type": "start",
        "session_id": session_id,
        "trace_id": trace_id,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "data_scope": data_scope,
        "kb_version": kb_version,
    }


def status_event(message: str, session_id: str) -> dict[str, Any]:
    """让前端展示当前处理阶段（意图识别/检索/生成），缓解用户等待焦虑。"""
    return {"type": "status", "message": message, "session_id": session_id}


def token_event(token: str, session_id: str) -> dict[str, Any]:
    """流式逐字推送生成结果，让用户逐步看到内容而非等待完整响应。"""
    return {"type": "token", "token": token, "session_id": session_id}


def end_event(
    *,
    session_id: str,
    hit_type: str,
    sources: list[dict[str, Any]],
    started: float,
    rewritten_query: str | None,
    trace_id: str | None = None,
    intent: dict[str, Any] | None = None,
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """通知前端生成结束以关闭加载状态，同时附带耗时/来源等诊断数据供后续分析。
    """
    return {
        "type": "end",
        "session_id": session_id,
        "trace_id": trace_id,
        "is_complete": True,
        "hit_type": hit_type,
        "sources": sources,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "retrieval": retrieval,
        "stage_timings_ms": (retrieval or {}).get("stage_timings_ms") if retrieval else {},
        "first_token_ms": (retrieval or {}).get("first_token_ms") if retrieval else None,
        "slowest_stage": (retrieval or {}).get("slowest_stage") if retrieval else None,
        "processing_time": time.perf_counter() - started,
    }


def error_event(*, error: str, session_id: str, trace_id: str) -> dict[str, Any]:
    """以事件形式将异常推送给前端，避免 WebSocket 断开，前端可展示友好提示。
    """
    return {"type": "error", "error": error, "session_id": session_id, "trace_id": trace_id}
