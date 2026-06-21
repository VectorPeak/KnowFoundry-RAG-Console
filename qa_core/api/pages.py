"""页面、健康检查和会话创建路由。不参与 RAG 检索和答案生成。"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from qa_core.scenarios.registry import resolve_scenario


router = APIRouter()


def _static_page(path: str):
    """返回一个禁止浏览器缓存的静态文件响应。"""
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.get("/")
async def read_root():
    """提供单页聊天界面，并禁止浏览器缓存旧版 JS。"""
    return _static_page("static/index.html")


@router.get("/admin")
async def read_admin_page():
    """提供本地教学状态页。"""
    return _static_page("static/admin.html")


@router.get("/health")
async def health_check():
    """容器与本地健康检查接口。"""
    scenario = resolve_scenario()
    return {
        "status": "healthy",
        "engine": "langchain_milvus_hybrid",
        "active_scenario_id": scenario.scenario_id,
        "active_scenario_name": scenario.display_name,
    }


@router.post("/api/create_session")
async def create_session(scenario_id: str | None = None):
    """创建页面端使用的会话 ID。"""
    scenario = resolve_scenario(scenario_id)
    return {"session_id": f"{scenario.scenario_id}:{uuid.uuid4()}", "scenario_id": scenario.scenario_id}


@router.get("/project-docs")
@router.get("/project-docs/{full_path:path}")
async def read_docs(full_path: str = ""):
    """提供 MkDocs 构建的项目讲义页面（site/ 目录）。"""
    docs_dir = Path("site")
    if not full_path or full_path.endswith("/"):
        full_path = os.path.join(full_path, "index.html")
    elif not full_path.endswith(".html") and "." not in Path(full_path).suffix:
        full_path = full_path + ".html" if not full_path.endswith("/") else os.path.join(full_path, "index.html")

    file_path = docs_dir / full_path
    if not file_path.exists() or not file_path.is_file():
        file_path = docs_dir / "index.html"

    return FileResponse(str(file_path), headers={"Cache-Control": "no-store"})
