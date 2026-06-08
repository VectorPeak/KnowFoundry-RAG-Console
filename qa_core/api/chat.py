"""API 层问答、历史、反馈和检索调试路由。将浏览器 HTTP/WebSocket 请求转换成 QAService 调用，
并组装响应。是整个系统的 HTTP 入口层，负责参数校验、限流、异常转 HTTP 错误码、WebSocket 事件转发。

路由结构：
  GET    /api/history/{session_id}       — 获取会话历史
  DELETE /api/history/{session_id}       — 清理会话历史
  POST   /api/query                      — 预检直答（同步、限流）
  WS     /api/stream                     — 流式问答（WebSocket、限流）
  POST   /api/retrieval/debug            — 检索诊断（同步、限流）
  POST   /api/feedback                   — 用户反馈（同步、限流）
  GET    /api/sources                    — 业务分类过滤项
  GET    /api/scenarios                  — 业务场景列表
"""

from __future__ import annotations
import asyncio
import json
import time
from collections.abc import Iterator
from typing import Any
from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState
from qa_core.api.dependencies import check_rate_limit, client_key, enforce_http_rate_limit
from qa_core.api.error_handlers import raise_bad_request
from qa_core.api.service_context import QueryServiceContext
from qa_core.application.factory import get_qa_service
from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.memory.feedback import get_feedback_store
from qa_core.memory.history import get_history_store
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario
from qa_core.schemas import FeedbackRequest, QueryRequest, QueryResponse, RetrievalDebugResponse
router = APIRouter()
# 加载应用全局设置（限流、历史摘要开关等 API 层配置）
settings = get_settings()
# 获取当前模块的结构化日志记录器
logger = get_logger(__name__)

def _next_stream_event(iterator: Iterator[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
    """在 asyncio.to_thread 中安全地推进同步生成器一步，捕获 StopIteration 以避免其穿过线程边界导致异常丢失。

    这是同步生成器与异步事件循环之间的适配层：asyncio.to_thread 会在线程中执行本函数，
    返回 (has_next, event) 二元组供调用方判断是否继续。

    执行流程：
    1. next(iterator) 推进同步生成器一步
    2. 成功 → 返回 (True, event_dict)
    3. StopIteration → 返回 (False, None)，通知调用方终止

    参数：
        iterator: QAService.stream_query 返回的同步生成器。

    返回：
        (True, dict): 还有事件，dict 为事件负载。
        (False, None): 生成器已耗尽。
    """
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _log_background_task_result(task: asyncio.Task) -> None:
    """asyncio.Task.add_done_callback 回调函数。在后台摘要刷新任务完成时被调用，
    捕获并记录可能出现的异常，避免异常被事件循环静默吞噬。

    执行流程：
    1. task.result() 获取后台任务结果
    2. 成功 → 无操作（正常返回）
    3. 异常 → logger.exception() 记录完整堆栈

    参数：
        task: 已完成的 asyncio.Task 对象（由 _schedule_summary_refresh 创建）。
    """
    try:
        task.result()
    except Exception:
        logger.exception("Refresh chat history summary failed")


def _schedule_summary_refresh(session_id: str) -> None:
    """在 WebSocket 流式问答 end 事件后，异步刷新会话摘要记忆，供下一轮追问上下文使用。

    执行流程：
    1. 检查 settings.history_summary_enabled 开关，关闭则直接返回
    2. asyncio.create_task(asyncio.to_thread(...)) 将同步刷新任务提交到线程池
       —— 避免阻塞 WebSocket end 事件的发送
    3. task.add_done_callback(_log_background_task_result) 注册异常捕获回调

    参数：
        session_id: 需要刷新摘要的会话 ID。
    """
    if not settings.history_summary_enabled:
        return
    # 原因： 摘要刷新是 CPU/IO 密集型操作（需 LLM 调用），放入线程池避免阻塞事件循环
    task = asyncio.create_task(asyncio.to_thread(get_history_store().refresh_summary_if_needed, session_id))
    # 原因： 后台任务的异常默认不会被任何地方捕获，必须显式注册回调才能记录
    task.add_done_callback(_log_background_task_result)


async def _send_stream_events(websocket: WebSocket, context: QueryServiceContext) -> bool:
    """将 QAService 的同步事件生成器逐事件转发到 WebSocket。

    执行流程：
    1. get_qa_service().stream_query(...) 创建同步事件生成器
    2. asyncio.to_thread(_next_stream_event, stream) 在线程池中安全推进生成器
    3. 生成器耗尽 → 返回 True（正常结束）
    4. WebSocket 断开 → 返回 False（通知调用方关闭连接）
    5. 每收到一个事件就通过 websocket.send_json() 推送给前端
    6. await asyncio.sleep(0) 出让控制权，让事件循环处理其他任务
    7. type 为 "end" → _schedule_summary_refresh() 刷新会话摘要
    8. type 为 "end" 或 "error" → 返回 True（流结束）

    参数：
        websocket: FastAPI WebSocket 连接对象，用于向浏览器推送事件。
        context: 请求上下文，包含 session_id 等参数，用于构造 service_args。

    返回：
        bool: True 表示正常结束或连接断开，False 表示 WebSocket 提前断开。
    """
    # 创建同步生成器，推进完整 RAG 流式问答链路
    stream = get_qa_service().stream_query(*context.service_args())
    while True:
        # 原因： 在线程池中推进同步生成器，避免阻塞 asyncio 事件循环
        has_event, event = await asyncio.to_thread(_next_stream_event, stream)
        if not has_event or event is None:
            return True
        if websocket.client_state != WebSocketState.CONNECTED:
            return False
        await websocket.send_json(event)
        # 原因： 出让控制权让事件循环处理其他等待任务，防止单个 WebSocket 独占事件循环
        await asyncio.sleep(0)
        if event.get("type") in {"end", "error"}:
            if event.get("type") == "end":
                _schedule_summary_refresh(str(event.get("session_id") or context.session_id))
            return True

@router.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取指定会话的历史问答对列表，供前端侧边栏展示。
    执行流程：
    1. get_history_store().as_pairs() 从 MySQL 加载会话消息并转为问答对列表
    2. 异常由全局 HTTP 异常处理器记录并返回 500

    参数：
        session_id: 会话 ID（路径参数）。

    返回：
        dict: {"session_id": str, "history": list[dict]}。
               history 列表每个元素包含 question、answer、timestamp 等字段。
    """
    # 从 MySQL 加载会话消息并转为前端展示用的问答对列表
    history = get_history_store().as_pairs(session_id, limit=settings.max_history_messages)
    return {"session_id": session_id, "history": history}

@router.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """清理指定会话的持久化聊天记录及其派生的摘要记忆。

    执行流程：
    1. get_history_store().clear() 删除 MySQL 中该会话的消息记录和派生摘要
    2. 异常由全局 HTTP 异常处理器记录并返回 500

    参数：
        session_id: 会话 ID（路径参数）。

    返回：
        dict: {"status": "success", "message": "历史记录已清除"}。
    """
    # 删除 MySQL 中该会话的消息记录和派生摘要
    get_history_store().clear(session_id)
    return {"status": "success", "message": "历史记录已清除"}


@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, _: None = Depends(enforce_http_rate_limit)):
    """HTTP 同步预检端点。问候/离题等直接返回答案，复杂业务知识问题返回
    is_streaming=True 告知前端改用 WebSocket 获取流式响应。

    执行流程：
    1. Depends(enforce_http_rate_limit)   — 限流中间件，超限返回 429
    2. QueryServiceContext.from_request()  — 构造请求上下文
    3. 空查询校验 → 返回 HTTP 400
    4. get_qa_service().preview_query()    — 调用应用层预检
    5. ValueError 由全局异常处理器转为 HTTP 400
    6. 其他异常由全局异常处理器记录并返回 HTTP 500
    7. preview 非空 → 直接组装 QueryResponse（同步回答）
    8. preview 为空 → 返回 is_streaming=True 引导前端走 WebSocket

    参数：
        request: QueryRequest（JSON body），包含 query、session_id、scenario_id 等字段。
        _: 由 Depends 注入，仅用于触发限流检查。

    返回：
        QueryResponse: 包含 answer、is_streaming、session_id、hit_type、sources 等。
    """
    started = time.perf_counter()
    # 从 HTTP 请求构造 QAService 调用上下文
    context = QueryServiceContext.from_request(request)
    if not context.query:
        raise_bad_request("查询内容不能为空")
    # 原因： HTTP 同步预检优先处理简单问题（问候/越界），避免每次查询都建立 WebSocket 连接浪费资源
    # 调用应用服务层的轻量预检。这里不再包 try/except：
    # 参数错误和未预期异常由 app.py 注册的全局 HTTP 异常处理器统一转换。
    preview = get_qa_service().preview_query(*context.service_args())

    if preview is not None:
        # 直答路径：问候/越界/客服等，同步返回完整答案
        return QueryResponse(
            answer=preview.answer,
            is_streaming=False,
            session_id=context.session_id,
            scenario_id=request.scenario_id,
            processing_time=time.perf_counter() - started,
            hit_type=preview.hit_type,
            sources=preview.sources,
            rewritten_query=preview.rewritten_query,
            intent=preview.intent,
            retrieval=preview.retrieval,
        )

    # 复杂问题路径：告知前端 is_streaming=True，需建立 WebSocket 获取流式回答
    return QueryResponse(
        answer="请使用 WebSocket 接口获取流式响应",
        is_streaming=True,
        session_id=context.session_id,
        scenario_id=request.scenario_id,
        processing_time=time.perf_counter() - started,
        hit_type="stream_required",
        sources=[],
    )

@router.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 流式问答端点。接收浏览器查询请求，逐事件推送 RAG 流式回答。

    执行流程：
    1. websocket.accept()                          — 接受 WebSocket 连接
    2. while True 循环（每个连接可处理多次问答）：
       a. websocket.receive_text()                 — 接收前端 JSON 文本
       b. check_rate_limit(client_key(websocket))  — WebSocket 级别限流检查
       c. json.loads() 解析请求体                  — 无效 JSON 返回 type=error
       d. QueryServiceContext.from_ws_payload()    — 构造请求上下文
       e. 空查询校验 → 返回 type=error
       f. _send_stream_events()                    — 转发 RAG 事件流到 WebSocket
          - 返回 False 表示连接已断开 → return
          - 返回 True 表示正常结束 → 继续等待下一轮问答
    3. WebSocketDisconnect → logger.info() 记录断开
    4. 其他异常 → 连接未断开时推送 type=error 事件

    参数：
        websocket: FastAPI WebSocket 连接对象，路径参数自动注入。
    """
    # 原因： 两层设计——HTTP POST /api/query 优先处理简单问题（问候/越界/客服等），无需建立 WebSocket 即可返回；只有复杂业务知识才走 WebSocket 流式链路，降低全量请求的 WebSocket 建连开销
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            # 原因： WebSocket 连接也需限流，使用 client IP 作为 key
            if not check_rate_limit(client_key(websocket)):
                await websocket.send_json({"type": "error", "error": "请求过于频繁，请稍后再试。"})
                continue
            try:
                request_data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "无效的 JSON 数据"})
                continue

            # 从 WebSocket JSON payload 解析查询参数
            context = QueryServiceContext.from_ws_payload(request_data)
            if not context.query:
                await websocket.send_json({"type": "error", "error": "查询内容不能为空"})
                continue

            if not await _send_stream_events(websocket, context):
                return
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket error")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": str(exc)})


@router.post("/api/retrieval/debug", response_model=RetrievalDebugResponse)
async def debug_retrieval(request: QueryRequest, _: None = Depends(enforce_http_rate_limit)):
    """检索诊断端点。返回意图分类、改写结果、FAQ/文档命中列表等调试信息，
    但不调用最终回答 LLM，用于开发者评估检索质量。

    执行流程：
    1. Depends(enforce_http_rate_limit) − 限流中间件
    2. QueryServiceContext.from_request() − 构造请求上下文
    3. asyncio.to_thread(get_qa_service().debug_retrieval, ...)
       − 在线程池中执行同步检索半链路
    4. ValueError 由全局异常处理器转为 HTTP 400
    5. 其他异常由全局异常处理器记录并返回 HTTP 500

    参数：
        request: QueryRequest（JSON body），复用预检请求结构。
        _: 由 Depends 注入，仅用于触发限流检查。

    返回：
        RetrievalDebugResponse: 包含意图分类、改写后查询、FAQ 和文档检索命中列表等。
    """
    context = QueryServiceContext.from_request(request)
    if not context.query:
        raise_bad_request("查询内容不能为空")
    # 原因： debug_retrieval 是同步阻塞调用（含 Milvus 检索），放入线程池避免阻塞事件循环
    return await asyncio.to_thread(
        get_qa_service().debug_retrieval,
        *context.service_args(),
    )


@router.post("/api/feedback")
async def add_feedback(request: FeedbackRequest, _: None = Depends(enforce_http_rate_limit)):
    """保存用户对答案的赞/踩反馈，供后续质量分析和模型微调使用。

    执行流程：
    1. Depends(enforce_http_rate_limit) − 限流中间件
    2. get_feedback_store().add_feedback() − 将反馈写入 MySQL 反馈表
    3. 异常由全局 HTTP 异常处理器记录并返回 500

    参数：
        request: FeedbackRequest（JSON body），包含 session_id/rating/comment 等。
        _: 由 Depends 注入，仅用于触发限流检查。

    返回：
        dict: {"status": "success", "feedback_id": str}。
    """
    # 将用户反馈写入 MySQL 反馈表
    feedback_id = get_feedback_store().add_feedback(
        session_id=request.session_id,
        question=request.question.strip(),
        answer=request.answer.strip(),
        rating=request.rating,
        comment=(request.comment or "").strip() or None,
        sources=request.sources,
        scenario_id=request.scenario_id,
        tenant_id=request.tenant_id,
        dataset_id=request.dataset_id,
    )
    return {"status": "success", "feedback_id": feedback_id}


@router.get("/api/sources")
async def get_sources(scenario_id: str | None = None):
    """获取业务分类过滤项列表，供前端页面的分类下拉框使用。

    执行流程：
    1. resolve_scenario(scenario_id) − 解析业务场景配置
    2. 组装含 scenario_id、valid_sources、source_options 的响应

    参数：
        scenario_id: 业务场景标识（可选的查询参数，不传则使用默认场景）。

    返回：
        dict: 包含 scenario_id、sources（有效来源列表）、source_options（带标签的选项列表）。
    """
    # 解析请求对应的业务场景配置
    scenario = resolve_scenario(scenario_id)
    return {
        "scenario_id": scenario.scenario_id,
        "sources": scenario.valid_sources,
        "source_options": scenario.source_options(),
    }


@router.get("/api/scenarios")
async def list_scenarios():
    """获取所有可切换的业务场景列表，供前端场景切换下拉框使用。

    执行流程：
    1. get_scenario_registry().as_payload() − 从场景注册表获取序列化后的场景列表

    返回：
        dict: 包含所有可用场景的配置列表，每个场景含 id、display_name、description 等。
    """
    return get_scenario_registry().as_payload()
