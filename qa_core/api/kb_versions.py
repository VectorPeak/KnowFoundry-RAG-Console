"""知识库版本管理路由。只管理本地版本清单，不直接删除 Milvus 数据。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from qa_core.api.dependencies import require_admin_token
from qa_core.api.error_handlers import raise_bad_request, raise_not_found
from qa_core.governance.kb_versions import get_kb_version_store
router = APIRouter()

@router.get("/api/kb_versions")
async def list_kb_versions(scenario_id: str | None = None):
    """返回知识库版本清单和当前生效版本。"""
    return get_kb_version_store(scenario_id).as_payload()

@router.post("/api/kb_versions/{kb_version}/activate")
async def activate_kb_version(kb_version: str, scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """把指定知识库版本切换为 active。"""
    try:
        version = get_kb_version_store(scenario_id).activate_version(kb_version)
        return {"status": "success", "version": version.as_dict()}
    except ValueError as exc:
        raise_not_found(str(exc))

@router.post("/api/kb_versions/{kb_version}/archive")
async def archive_kb_version(kb_version: str, scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """归档一个非 active 知识库版本。"""
    try:
        version = get_kb_version_store(scenario_id).archive_version(kb_version)
        return {"status": "success", "version": version.as_dict()}
    except ValueError as exc:
        raise_bad_request(str(exc))
