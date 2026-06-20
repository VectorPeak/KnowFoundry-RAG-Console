"""真实 API E2E 验收脚本。

`acceptance_smoke.py` 重点验证页面、管理摘要和 WebSocket 流式事件。这个脚本更偏 API
合同验收：逐个检查当前一期暴露的 HTTP 管理、质量报告和 LangSmith 状态接口能否稳定返回预期字段。
LangSmith 未启用不代表本地服务不可用；本脚本只要求状态接口结构正确，是否启用写入
details，供正式企业环境另行确认。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.config.settings import get_settings
from scripts.common import configure_utf8_stdio, write_optional_json


def headers(token: str) -> dict[str, str]:
    """构造管理接口 header。"""
    return {"X-Admin-Token": token} if token else {}


def resolve_admin_token(raw_token: str | None) -> str:
    """解析管理接口令牌。

    使用场景：
    - 本地手动烟测时，通常只传 `--base-url`，不希望每次都把管理令牌写在命令行里；
    - 质量检查脚本会统一调用本脚本，如果命令行没有显式传令牌，就应该复用当前运行配置中的
      `ADMIN_API_TOKEN`；
    - CI 或临时环境仍可以通过 `--admin-token` 覆盖，便于验证不同服务实例。

    这里不会把令牌写入报告，避免验收文件泄露敏感信息。
    """
    return (raw_token or get_settings().admin_api_token or "").strip()


def fetch_json(base_url: str, path: str, token: str = "") -> dict:
    """读取 JSON 接口并返回 dict。"""
    req = urlrequest.Request(base_url.rstrip("/") + path, headers=headers(token))
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"API check failed: {path}: {exc}") from exc


def check_fields(payload: dict, fields: list[str]) -> bool:
    """检查响应是否包含必需字段。"""
    return all(field in payload for field in fields)


def run_checks(args: argparse.Namespace) -> dict:
    """执行 API 合同验收。"""
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    admin_token = resolve_admin_token(args.admin_token)

    health = fetch_json(args.base_url, "/health")
    checks["health"] = check_fields(health, ["status", "active_scenario_id"])
    details["health"] = health

    scenarios = fetch_json(args.base_url, "/api/scenarios")
    checks["scenarios"] = bool(scenarios.get("scenarios"))
    details["scenario_count"] = len(scenarios.get("scenarios") or [])

    versions = fetch_json(args.base_url, f"/api/kb_versions?scenario_id={args.scenario}", admin_token)
    checks["kb_versions"] = check_fields(versions, ["scenario_id", "versions", "metadata_store"])
    details["kb_active"] = versions.get("effective_active_version")

    admin_status = fetch_json(args.base_url, "/api/admin/status", admin_token)
    checks["admin_status"] = check_fields(admin_status, ["status", "scenarios", "active_kb_versions", "langsmith"])
    details["admin_scenario_count"] = len(admin_status.get("scenarios") or [])

    langsmith = fetch_json(args.base_url, "/api/admin/langsmith", admin_token)
    checks["langsmith"] = check_fields(langsmith, ["enabled", "project", "endpoint", "project_url"])
    details["langsmith_enabled"] = langsmith.get("enabled")
    details["langsmith_has_api_key"] = langsmith.get("has_api_key")
    details["langsmith_project"] = langsmith.get("project")

    ingestion = fetch_json(args.base_url, f"/api/admin/ingestion_reports?scenario_id={args.scenario}&limit=5", admin_token)
    checks["ingestion_reports"] = "reports" in ingestion
    details["ingestion_report_count"] = len(ingestion.get("reports") or [])

    gates = fetch_json(args.base_url, "/api/admin/gate_reports?limit=5", admin_token)
    checks["gate_reports"] = check_fields(gates, ["reports", "langsmith"])
    details["gate_report_count"] = len(gates.get("reports") or [])

    performance = fetch_json(args.base_url, "/api/admin/performance_reports?limit=5", admin_token)
    checks["performance_reports"] = check_fields(performance, ["reports", "langsmith"])
    details["performance_report_count"] = len(performance.get("reports") or [])

    enterprise_governance = fetch_json(args.base_url, "/api/admin/enterprise_governance", admin_token)
    checks["enterprise_governance"] = check_fields(enterprise_governance, ["report_type", "data_realism", "overlay_readiness", "langsmith"])

    kb_version_compare = fetch_json(args.base_url, "/api/admin/kb_version_compare", admin_token)
    checks["kb_version_compare"] = check_fields(kb_version_compare, ["report_type", "comparison", "langsmith"])

    return {
        "report_type": "api_e2e_smoke",
        "ok": all(checks.values()),
        "base_url": args.base_url,
        "checks": checks,
        "details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Run HTTP API E2E checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--admin-token", default="", help="为空时自动读取当前运行配置中的 ADMIN_API_TOKEN。")
    parser.add_argument("--scenario", default="enterprise_knowledge")
    parser.add_argument("--output", default="", help="可选 JSON 报告输出路径。")
    return parser


def main() -> None:
    """执行 API E2E 检查。"""
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    payload = run_checks(args)
    write_optional_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
