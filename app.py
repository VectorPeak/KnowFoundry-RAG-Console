"""KnowForge RAG Platform 的 FastAPI 应用入口。

本文件现在只负责四件事：
1. 创建 FastAPI 应用；
2. 配置 CORS 和静态资源；
3. 启动时执行必需环境校验和检索栈预热；
4. 注册 `qa_core.api` 下拆分后的路由。

为什么要保持入口文件很薄：
- `app.py` 是服务启动点，不应该继续堆 HTTP、WebSocket、管理接口和 RAG 细节；
- 接口按页面、聊天、管理、知识库版本拆分后，后续二期 Agent 增加路由时不会污染
  一期 RAG 主链路；
- 入口越薄，越容易确认当前项目没有旧链路、没有技术降级路径、没有隐藏旁路。

不适合放在这里的内容：
- 不要在这里实现意图识别、检索策略、提示词拼接或 Milvus 查询；
- 不要在这里保存用户会话状态；
- 不要在这里重新接入旧版 mysql_qa/rag_qa 流程。
"""

from __future__ import annotations
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from qa_core.api import admin, chat, kb_versions, pages
from qa_core.api.error_handlers import register_api_exception_handlers
from qa_core.config.logging_config import get_logger
from qa_core.config.preflight import validate_runtime_environment
from qa_core.config.settings import get_settings
from qa_core.observability.langsmith_adapter import configure_langsmith_environment
from qa_core.retrieval.factory import warmup_retrieval_stack

settings = get_settings()
configure_langsmith_environment()
logger = get_logger(__name__)
app = FastAPI(
    title="KnowForge RAG Platform API",
    description="LangChain + Milvus Hybrid 企业级多场景 RAG 知识平台",
)
register_api_exception_handlers(app)

# 当前前端和 API 默认同源部署，但保留 CORS 配置是为了方便本地调试：
# 例如单独启动 Vite/React 页面时，只需要在 .env 中追加允许来源即可。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def warmup_runtime() -> None:
    """服务启动时执行前置校验，并预热本地检索模型和 Milvus 连接。

    当前项目的环境是必需前置条件：LLM Key、Milvus、MySQL、本地模型、场景配置和
    active 知识库版本缺失时，服务直接启动失败。这样可以避免页面看似能打开，但真正
    提问时才发现核心链路没有通电。
    """
    summary = validate_runtime_environment()
    logger.info("Runtime preflight passed: %s", summary)
    await asyncio.to_thread(warmup_retrieval_stack)

app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(kb_versions.router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
