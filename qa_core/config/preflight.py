"""主链路运行环境前置校验。Milvus、MySQL、本地模型、LLM Key 等均为启动前置条件，不满足直接报错。"""

from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

from qa_core.config.settings import get_settings
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.llm.client import validate_llm_connectivity
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario


PLACEHOLDER_VALUES = {"", "replace-with-real-key", "replace-with-random-token", "changeme", "change-me"}
PLACEHOLDER_MARKERS = ("请替换", "replace", "changeme", "change-me", "your-", "placeholder")

def _is_placeholder(value: str | None) -> bool:
    """判断配置值是否为空或仍是示例占位符。"""
    normalized = str(value or "").strip()
    lower_value = normalized.lower()
    return lower_value in PLACEHOLDER_VALUES or any(marker in lower_value for marker in PLACEHOLDER_MARKERS)


def _require_tcp(name: str, host: str, port: int, timeout: float = 3.0) -> None:
    """校验 TCP 端口可连接。

    这里只做连接性检查，不做业务读写。真实集合、表结构和模型预热会在后续 warmup 中
    完成。把端口检查放在这里，是为了让"服务没启动"这类基础问题在最早阶段暴露。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(f"{name} 不可连接：{host}:{port}。请先启动必需环境。") from exc


def _require_path(name: str, raw_path: str) -> None:
    """校验本地目录或文件存在。"""
    path = Path(raw_path)
    if not path.exists():
        raise RuntimeError(f"{name} 不存在：{path}")


def _require_milvus_uri() -> None:
    """校验 Milvus URI 格式和 TCP 可达性。"""
    settings = get_settings()
    parsed = urlparse(settings.milvus_uri)
    host = parsed.hostname
    port = parsed.port or 19530
    if not host:
        raise RuntimeError(f"MILVUS_URI 无效：{settings.milvus_uri}")
    _require_tcp("Milvus", host, port)


def validate_runtime_environment() -> dict[str, object]:
    """校验主链路全部前置条件，任一不满足则抛出 RuntimeError。

    【fail-fast 设计】
    在服务接收任何用户请求之前，完整验证所有外部依赖和配置项。这里的核心权衡是：
    启动时多花几百毫秒做全面检查，换取"线上零配置事故"的保障——如果等到第一个用户
    请求才暴露 Milvus 连接失败或模型路径错误，故障影响范围会从"启动失败"扩散到
    "服务降级、隐式报错、数据不一致"。

    【检查顺序说明】
    1. 占位符检测（API Key / Token）——纯内存操作，零成本，最先拦截最常见的人为错误。
    2. 本地路径检查（模型目录、场景目录、FAQ 文件）——文件系统调用，比网络 I/O 快一到
       两个数量级，优先暴露开发环境的常见配置遗漏。
    3. TCP 连接检查（Milvus、MySQL、LLM API）——网络 I/O 最慢且可能 hang，放在最后，
       让前面快速失败的检查先阻断，网络层面的问题留到最后集中暴露。
    4. 业务逻辑检查（active KB version 解析）——在基础依赖就绪后才做，避免
       "数据库连不上但提示版本不存在"这样的误导性错误信息。
    """
    settings = get_settings()
    # 获取场景注册器并解析当前场景的 TOML 配置
    registry = get_scenario_registry()
    scenario = resolve_scenario(settings.active_scenario_id)

    if _is_placeholder(settings.llm_api_key):
        raise RuntimeError("DASHSCOPE_API_KEY 未配置。当前架构必须通过 LangChain ChatOpenAI 调用真实 LLM。")
    if _is_placeholder(settings.admin_api_token):
        raise RuntimeError("ADMIN_API_TOKEN 未配置。管理接口必须显式设置令牌。")

    _require_path("Embedding 模型目录", settings.embedding_model_path)
    _require_path("Reranker 模型目录", settings.reranker_model_path)

    if not Path(settings.scenario_config_dir).exists():
        raise RuntimeError(f"SCENARIO_CONFIG_DIR 不存在：{settings.scenario_config_dir}")
    if scenario.scenario_id not in {item.scenario_id for item in registry.list_scenarios()}:
        raise RuntimeError(f"ACTIVE_SCENARIO_ID 无效：{settings.active_scenario_id}")
    _require_path("场景文档目录", scenario.data_root)
    _require_path("场景 FAQ 文件", scenario.faq_csv_path)

    _require_milvus_uri()
    _require_tcp("MySQL", settings.mysql_host, settings.mysql_port)
    # 校验 LLM API Key 和网络连通性
    validate_llm_connectivity()

    # 获取知识库版本管理器并解析当前 active 版本
    version_store = get_kb_version_store(scenario.scenario_id)
    try:
        active_version = version_store.resolve_active_version()
    except ValueError as exc:
        raise RuntimeError(
            f"{exc}。请先执行入库并激活版本，例如 "
            "scripts/rebuild_kb_version.py --new-version --force --activate。"
        ) from exc

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "milvus_uri": settings.milvus_uri,
        "mysql": f"{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}",
        "embedding_model_path": settings.embedding_model_path,
        "reranker_model_path": settings.reranker_model_path,
        "active_kb_version": active_version,
        "available_scenarios": [item.scenario_id for item in registry.list_scenarios()],
    }
