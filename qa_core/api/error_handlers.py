"""API 层统一异常处理。

HTTP 路由里不再为每个服务调用重复编写 `try/except Exception`。统一处理的好处是：
- 路由函数只保留正常业务流程，可读性更高；
- `ValueError` 这类可预期的参数/业务校验错误统一返回 400；
- 未预期异常统一记录堆栈，但响应给前端时不暴露内部异常细节。

WebSocket 不走 FastAPI 的 HTTP 异常响应机制，所以 WebSocket 端点仍需要自己发送
`type=error` 事件。
"""

from __future__ import annotations

from typing import NoReturn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from qa_core.config.logging_config import get_logger


logger = get_logger(__name__)


def raise_http_error(status_code: int, detail: str) -> NoReturn:
    """抛出带统一响应结构的 HTTPException。"""
    raise HTTPException(status_code=status_code, detail=detail)


def raise_bad_request(detail: str) -> NoReturn:
    """请求参数或业务条件不满足，返回 HTTP 400。"""
    raise_http_error(400, detail)


def raise_unauthorized(detail: str) -> NoReturn:
    """认证失败，返回 HTTP 401。"""
    raise_http_error(401, detail)


def raise_not_found(detail: str) -> NoReturn:
    """目标资源不存在，返回 HTTP 404。"""
    raise_http_error(404, detail)


def raise_too_many_requests(detail: str) -> NoReturn:
    """请求过于频繁，返回 HTTP 429。"""
    raise_http_error(429, detail)


def raise_server_error(detail: str) -> NoReturn:
    """服务端配置或运行状态不满足要求，返回 HTTP 500。"""
    raise_http_error(500, detail)


def register_api_exception_handlers(app: FastAPI) -> None:
    """注册 HTTP API 的全局异常转换规则。"""

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """业务参数或白名单校验失败，统一返回 HTTP 400。"""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """未预期异常统一记录日志，前端只收到稳定的 500 文案。"""
        logger.exception("Unhandled API error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "服务内部错误，请查看后端日志"})
