"""Lightweight admin routes.

Enterprise LLMOps capabilities are delegated to LangSmith. The local admin API
therefore exposes only service status, local quality artifacts, and links/status
for LangSmith instead of re-implementing trace storage, annotation queues, or
evaluation dashboards.
"""

from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Depends

from qa_core.api.dependencies import require_admin_token
from qa_core.config.settings import PROJECT_ROOT
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.observability.langsmith_adapter import langsmith_status
from qa_core.quality.ingestion import list_ingestion_reports
from qa_core.scenarios.registry import get_scenario_registry
router = APIRouter()
REPORT_DIR = PROJECT_ROOT / "reports" / "verification"

def _latest_file_summary(pattern: str) -> dict[str, Any]:
    candidates = sorted(REPORT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {"available": False, "file": None}
    path = candidates[0]
    return {
        "available": True,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "updated_at": path.stat().st_mtime,
    }

@router.get("/api/admin/status")
async def admin_status(_: None = Depends(require_admin_token)):
    """Return the lightweight local admin status."""
    registry = get_scenario_registry()
    scenario_ids = [scenario.scenario_id for scenario in registry.list_scenarios()]
    active_versions: dict[str, str] = {}
    for scenario_id in scenario_ids:
        versions = get_kb_version_store(scenario_id).list_versions()
        active = next((item for item in versions if item.status == "ACTIVE"), None)
        active_versions[scenario_id] = active.kb_version if active else ""
    return {
        "status": "ok",
        "scenarios": scenario_ids,
        "active_kb_versions": active_versions,
        "langsmith": langsmith_status(),
    }

@router.get("/api/admin/langsmith")
async def admin_langsmith(_: None = Depends(require_admin_token)):
    """Return LangSmith configuration and project link hints."""
    return langsmith_status()

@router.get("/api/admin/ingestion_reports")
async def admin_ingestion_reports(limit: int = 20, scenario_id: str | None = None, _: None = Depends(require_admin_token)):
    """Return recent local ingestion quality reports."""
    return {"reports": list_ingestion_reports(scenario_id=scenario_id, limit=limit)}

@router.get("/api/admin/kb_version_compare")
async def admin_kb_version_compare(_: None = Depends(require_admin_token)):
    """Return lightweight KB version comparison pointer."""
    return {
        "report_type": "kb_version_compare",
        "comparison": _latest_file_summary("kb_version_compare*_latest.json"),
        "langsmith": langsmith_status(),
    }

@router.get("/api/admin/gate_reports")
async def admin_gate_reports(_: None = Depends(require_admin_token)):
    """Return lightweight gate summary file pointers."""
    return {"reports": [_latest_file_summary("*gate*_latest.json")], "langsmith": langsmith_status()}

@router.get("/api/admin/performance_reports")
async def admin_performance_reports(_: None = Depends(require_admin_token)):
    """Return lightweight performance summary file pointers."""
    return {"reports": [_latest_file_summary("*performance*_latest.json")], "langsmith": langsmith_status()}

@router.get("/api/admin/enterprise_governance")
async def admin_enterprise_governance(_: None = Depends(require_admin_token)):
    """Return lightweight enterprise governance summary file pointers."""
    return {
        "report_type": "enterprise_governance",
        "data_realism": _latest_file_summary("enterprise_data_realism_latest.json"),
        "overlay_readiness": _latest_file_summary("enterprise_overlay_readiness_latest.json"),
        "langsmith": langsmith_status(),
    }