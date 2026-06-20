"""本地运行环境诊断。

该脚本用于回答“项目为什么没有通电”的基础问题：Docker/WSL 是否可用，Milvus、
MySQL、API 端口是否可连接，8 个场景是否都有 active 知识库版本。

它只做诊断，不做自动修复，也不提供技术降级。当前项目的前置条件就是必须具备
Docker、Milvus、MySQL、本地模型、LLM Key 和 active 知识库版本；缺一项就应该明确
失败，而不是绕开核心能力继续运行。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa_core.config.settings import get_settings
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.scenarios.registry import get_scenario_registry
from scripts.common import configure_utf8_stdio, preview_text, print_json, write_optional_json


WINDOWS_FEATURES = {
    "Microsoft-Windows-Subsystem-Linux": "WSL",
    "VirtualMachinePlatform": "虚拟机平台",
}
DEFAULT_COMPOSE_FILE = "docker-compose.yml"
ALT_COMPOSE_FILES = ("docker-compose.milvus.yml",)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PLACEHOLDER_HINTS = ("请替换", "replace", "changeme", "your-", "sk-your")
DEFAULT_ADMIN_TOKENS = {"local-admin-token", "admin", "123456", "token"}
POWERSHELL_CANDIDATES = (
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    str(Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"),
)
HOST_PORT_PATTERN = re.compile(r":(\d+)->")


@dataclass(frozen=True)
class RuntimeCheck:
    """一条本地环境诊断结果。"""

    name: str
    ok: bool
    required: bool
    detail: str
    suggestion: str = ""


def update_check(checks: list[RuntimeCheck], name: str, **changes: object) -> None:
    """按名称原位更新某条诊断结果。"""

    for index, item in enumerate(checks):
        if item.name != name:
            continue
        payload = asdict(item)
        payload.update(changes)
        checks[index] = RuntimeCheck(**payload)
        return


@dataclass(frozen=True)
class ComposeServiceSnapshot:
    """一条 compose service 的最小运行摘要。

    这里只保留环境诊断真正需要的字段：service 名、container_name 和宿主机端口。
    这样可以在不引入额外 YAML 依赖的情况下，对当前仓库固定格式的 compose 文件做
    足够稳定的检查。
    """

    compose_file: str
    service_name: str
    container_name: str
    host_ports: tuple[int, ...]
    raw_ports: tuple[str, ...]


@dataclass(frozen=True)
class DockerContainerSnapshot:
    """`docker ps` 返回的一条容器摘要。"""

    name: str
    ports: str
    host_ports: tuple[int, ...]


def decode_probe_output(raw_output: bytes) -> str:
    """按常见 Windows 命令编码解码输出。"""
    for encoding in ("utf-8", "gbk", "utf-16-le"):
        try:
            text = raw_output.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text[:80] or encoding == "utf-16-le":
            return text
    return raw_output.decode("utf-8", errors="replace").replace("\x00", "")


def run_probe(command: list[str]) -> tuple[bool, str, int]:
    """执行只读探测命令，并返回是否成功、输出摘要和退出码。"""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), 1
    stdout = decode_probe_output(completed.stdout or b"")
    stderr = decode_probe_output(completed.stderr or b"")
    output = "\n".join(part for part in [stdout, stderr] if part)
    return completed.returncode == 0, preview_text(output, 800), completed.returncode


def parse_env_file(path: Path) -> dict[str, str]:
    """解析 `.env` 文件中的键值对。

    运行时配置最后仍以 `get_settings()` 为准；这里额外读取 `.env`，是为了在诊断报告中
    直接展示“当前文件里写的就是这套端口/主机”，便于解释为什么和 compose 暴露端口不一致。
    """

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def extract_host_ports_from_text(raw_ports: str) -> tuple[int, ...]:
    """从 `docker ps` 或 compose 端口文本中提取宿主机端口。"""

    found = [int(match.group(1)) for match in HOST_PORT_PATTERN.finditer(raw_ports or "")]
    return tuple(sorted(set(found)))


def parse_port_mapping(raw_mapping: str) -> int | None:
    """解析 compose 里的 `host:container` 端口映射，只返回宿主机端口。

    当前项目 compose 文件只使用固定字符串端口映射，不涉及复杂变量展开。这里保留最小
    解析能力，是为了避免为了诊断脚本再引入一层额外依赖。
    """

    mapping = raw_mapping.strip().strip('"').strip("'")
    if not mapping:
        return None
    mapping = mapping.split("/", 1)[0]
    parts = [part.strip() for part in mapping.split(":") if part.strip()]
    if len(parts) < 2:
        return None
    host_candidate = parts[-2]
    return int(host_candidate) if host_candidate.isdigit() else None


def parse_compose_services(path: Path) -> list[ComposeServiceSnapshot]:
    """读取 compose 文件中的 service、container_name 和端口映射。

    诊断目标不是做一个通用 YAML 解析器，而是把当前仓库里两份 compose 文件的关键差异
    摘出来：MySQL/Milvus/API 各自暴露到宿主机的端口是什么。
    """

    if not path.exists():
        return []
    services: list[ComposeServiceSnapshot] = []
    current_service = ""
    current_container_name = ""
    current_ports: list[int] = []
    current_raw_ports: list[str] = []
    inside_services = False
    inside_ports = False

    def flush_current() -> None:
        nonlocal current_service, current_container_name, current_ports, current_raw_ports
        if not current_service:
            return
        services.append(
            ComposeServiceSnapshot(
                compose_file=path.name,
                service_name=current_service,
                container_name=current_container_name or current_service,
                host_ports=tuple(sorted(set(current_ports))),
                raw_ports=tuple(current_raw_ports),
            )
        )
        current_service = ""
        current_container_name = ""
        current_ports = []
        current_raw_ports = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "services:":
            inside_services = True
            continue
        if inside_services and indent == 0 and stripped.endswith(":") and stripped != "services:":
            flush_current()
            inside_services = False
            inside_ports = False
            continue
        if not inside_services:
            continue
        if indent == 2 and stripped.endswith(":"):
            flush_current()
            current_service = stripped[:-1].strip()
            inside_ports = False
            continue
        if not current_service:
            continue
        if indent <= 2:
            flush_current()
            inside_ports = False
            continue
        if stripped.startswith("container_name:"):
            current_container_name = stripped.split(":", 1)[1].strip()
            continue
        if stripped == "ports:":
            inside_ports = True
            continue
        if inside_ports:
            if indent >= 6 and stripped.startswith("-"):
                raw_mapping = stripped[1:].strip()
                current_raw_ports.append(raw_mapping.strip('"').strip("'"))
                host_port = parse_port_mapping(raw_mapping)
                if host_port is not None:
                    current_ports.append(host_port)
                continue
            inside_ports = False
    flush_current()
    return services


def list_compose_summaries(project_root: Path) -> list[dict[str, Any]]:
    """返回当前仓库内可见的 compose 摘要。"""

    summaries: list[dict[str, Any]] = []
    for file_name in (DEFAULT_COMPOSE_FILE, *ALT_COMPOSE_FILES):
        services = parse_compose_services(project_root / file_name)
        if not services:
            continue
        summaries.append(
            {
                "compose_file": file_name,
                "services": [
                    {
                        "service_name": item.service_name,
                        "container_name": item.container_name,
                        "host_ports": list(item.host_ports),
                        "raw_ports": list(item.raw_ports),
                    }
                    for item in services
                ],
            }
        )
    return summaries


def list_running_containers() -> tuple[list[DockerContainerSnapshot], str]:
    """读取 `docker ps`，用于补充“端口为什么不通”的解释。"""

    ok, output, _ = run_probe(["docker", "ps", "--format", "{{.Names}}|{{.Ports}}"])
    if not ok:
        return [], output
    containers: list[DockerContainerSnapshot] = []
    for line in output.splitlines():
        if "|" not in line:
            continue
        name, ports = line.split("|", 1)
        containers.append(
            DockerContainerSnapshot(
                name=name.strip(),
                ports=ports.strip(),
                host_ports=extract_host_ports_from_text(ports),
            )
        )
    return containers, ""


def normalize_host(host: str) -> str:
    """统一 host 比较格式。"""

    return (host or "").strip().lower()


def is_local_host(host: str) -> bool:
    """判断当前 host 是否表示宿主机直连。"""

    return normalize_host(host) in LOCAL_HOSTS


def detect_runtime_mode(mysql_host: str, milvus_host: str) -> tuple[str, bool, str]:
    """识别当前 `.env` 更接近哪种运行方式。

    - `local_api_host_ports`：API 运行在宿主机，MySQL/Milvus 通过宿主机端口访问；
    - `compose_api_network`：API 运行在 compose 网络里，依赖 service/container 名互联；
    - `mixed_runtime_mode`：一部分走 localhost，一部分走容器名，排查成本最高，应尽量避免。
    """

    mysql_is_local = is_local_host(mysql_host)
    milvus_is_local = is_local_host(milvus_host)
    if mysql_is_local and milvus_is_local:
        return (
            "local_api_host_ports",
            True,
            "当前 .env 是“本机启动 API + 宿主机端口访问 MySQL/Milvus”的模式。",
        )
    if (not mysql_is_local) and (not milvus_is_local):
        return (
            "compose_api_network",
            True,
            "当前 .env 是“API 在 compose 网络中运行，通过 service/container 名访问依赖”的模式。",
        )
    return (
        "mixed_runtime_mode",
        False,
        "当前 .env 混用了 localhost 和容器名，建议统一为本机模式或 compose 网络模式，避免排查时端口和网络语义混乱。",
    )


def classify_compose_service(service_name: str, container_name: str) -> str:
    """把 compose service 归类为 mysql/milvus/api 三类。"""

    haystack = f"{service_name} {container_name}".lower()
    if "mysql" in haystack:
        return "mysql"
    if "milvus" in haystack:
        return "milvus"
    if service_name == "api" or " api" in haystack or "-api" in haystack or "backend" in haystack:
        return "api"
    return ""


def service_port_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """把 compose 摘要转换成按角色索引的字典。"""

    mapping: dict[str, dict[str, Any]] = {}
    for service in summary.get("services", []):
        role = classify_compose_service(service["service_name"], service["container_name"])
        if role:
            mapping[role] = service
    return mapping


def summarize_compose_alignment(
    settings,
    api_host: str,
    api_port: int,
    compose_summaries: list[dict[str, Any]],
) -> RuntimeCheck:
    """检查 `.env` 与标准 compose 是否处于同一套运行语义。"""

    parsed_milvus = urlparse(settings.milvus_uri)
    milvus_host = parsed_milvus.hostname or "localhost"
    milvus_port = parsed_milvus.port or 19530
    runtime_mode, runtime_mode_ok, runtime_mode_detail = detect_runtime_mode(settings.mysql_host, milvus_host)
    runtime_check = RuntimeCheck(
        name="运行模式识别",
        ok=runtime_mode_ok,
        required=False,
        detail=f"{runtime_mode}: {runtime_mode_detail}",
        suggestion="" if runtime_mode_ok else "统一为一种运行方式：本机 API 模式用 localhost；compose 模式用 service/container 名。",
    )
    if not compose_summaries:
        return runtime_check

    standard_summary = next((item for item in compose_summaries if item["compose_file"] == DEFAULT_COMPOSE_FILE), compose_summaries[0])
    standard_map = service_port_map(standard_summary)
    details = [runtime_check.detail]
    suggestions: list[str] = []

    if runtime_mode == "local_api_host_ports":
        expected_ports = {"mysql": settings.mysql_port, "milvus": milvus_port, "api": api_port}
        mismatches: list[str] = []
        alternate_hits: list[str] = []
        for role, expected_port in expected_ports.items():
            standard_ports = standard_map.get(role, {}).get("host_ports", [])
            if expected_port not in standard_ports:
                mismatches.append(f"{role} 期望宿主机端口 {expected_port}，但 {standard_summary['compose_file']} 暴露的是 {standard_ports or '无'}")
                for summary in compose_summaries:
                    if summary["compose_file"] == standard_summary["compose_file"]:
                        continue
                    alt_ports = service_port_map(summary).get(role, {}).get("host_ports", [])
                    if expected_port in alt_ports:
                        alternate_hits.append(f"{role} 的 {expected_port} 出现在 {summary['compose_file']}")
        if mismatches:
            details.extend(mismatches)
            if alternate_hits:
                details.append("可见当前 .env 更像另一套 compose 端口：" + "；".join(alternate_hits))
            suggestions.append(
                "建议统一按 docker-compose.yml 运行：本机启动 API 时把 MYSQL_PORT 调整为 3306，并保持 MILVUS_URI=http://localhost:19530。"
            )
            return RuntimeCheck(
                name="Compose / .env 对齐",
                ok=False,
                required=False,
                detail="；".join(details),
                suggestion=" ".join(suggestions),
            )
        details.append(f".env 端口与 {standard_summary['compose_file']} 的宿主机暴露一致。")
        return RuntimeCheck(name="Compose / .env 对齐", ok=True, required=False, detail="；".join(details))

    if runtime_mode == "compose_api_network":
        standard_mysql = standard_map.get("mysql", {})
        standard_milvus = standard_map.get("milvus", {})
        mysql_targets = {standard_mysql.get("service_name", ""), standard_mysql.get("container_name", "")}
        milvus_targets = {standard_milvus.get("service_name", ""), standard_milvus.get("container_name", "")}
        mismatches: list[str] = []
        if settings.mysql_host not in mysql_targets:
            mismatches.append(f"MYSQL_HOST={settings.mysql_host}，但 {standard_summary['compose_file']} 的 MySQL 标识是 {sorted(filter(None, mysql_targets))}")
        if milvus_host not in milvus_targets:
            mismatches.append(f"MILVUS_URI 指向 {milvus_host}，但 {standard_summary['compose_file']} 的 Milvus 标识是 {sorted(filter(None, milvus_targets))}")
        if mismatches:
            return RuntimeCheck(
                name="Compose / .env 对齐",
                ok=False,
                required=False,
                detail="；".join(details + mismatches),
                suggestion="如果 API 运行在 compose 网络里，建议直接使用 .env.compose.example 中的 mysql / milvus 作为主机名。",
            )
        return RuntimeCheck(
            name="Compose / .env 对齐",
            ok=True,
            required=False,
            detail="；".join(details + [f"当前运行配置中的容器网络主机名与 {standard_summary['compose_file']} 一致。"]),
        )

    return RuntimeCheck(
        name="Compose / .env 对齐",
        ok=False,
        required=False,
        detail="；".join(details),
        suggestion="请先消除 mixed_runtime_mode，再判断 compose 端口或容器名是否一致。",
    )


def container_candidates(containers: list[DockerContainerSnapshot], role: str) -> list[DockerContainerSnapshot]:
    """从 `docker ps` 快照中过滤出和某类依赖相关的容器。"""

    candidates: list[DockerContainerSnapshot] = []
    for item in containers:
        name = item.name.lower()
        if role == "mysql" and "mysql" in name:
            candidates.append(item)
        elif role == "milvus" and "milvus" in name and "redis" not in name and "etcd" not in name and "minio" not in name:
            candidates.append(item)
        elif role == "api" and ("api" in name or "backend" in name):
            candidates.append(item)
    return candidates


def summarize_container_mapping(role: str, expected_port: int, containers: list[DockerContainerSnapshot]) -> RuntimeCheck:
    """解释目标端口为什么无法从宿主机访问。"""

    role_label = {"mysql": "MySQL", "milvus": "Milvus", "api": "API"}[role]
    if role == "api":
        return RuntimeCheck(
            name=f"Docker 端口映射：{role_label}",
            ok=True,
            required=False,
            detail="当前主线采用“本机启动 FastAPI + 宿主机访问 MySQL/Milvus”模式，API 不要求由 Docker 暴露 8000。",
        )
    for item in containers:
        if expected_port in item.host_ports:
            return RuntimeCheck(
                name=f"Docker 端口映射：{role_label}",
                ok=True,
                required=False,
                detail=f"{item.name} 已暴露宿主机端口 {expected_port}，docker ps 端口信息：{item.ports or '无'}",
            )
    candidates = container_candidates(containers, role)
    if candidates:
        candidate_text = "；".join(f"{item.name}（{item.ports or '无端口信息'}）" for item in candidates)
        return RuntimeCheck(
            name=f"Docker 端口映射：{role_label}",
            ok=False,
            required=False,
            detail=f"未发现宿主机端口 {expected_port} 的映射；发现相关容器：{candidate_text}",
            suggestion=f"相关容器存在但没有把 {expected_port} 暴露到宿主机，localhost:{expected_port} 无法直连。",
        )
    return RuntimeCheck(
        name=f"Docker 端口映射：{role_label}",
        ok=False,
        required=False,
        detail=f"docker ps 中未发现与 {role_label} 相关且暴露宿主机端口 {expected_port} 的容器。",
        suggestion=f"启动标准 compose 服务后，再确认 localhost:{expected_port} 可连通。",
    )


def is_placeholder_secret(value: str) -> bool:
    """判断当前配置是否仍然是占位符。"""

    lowered = (value or "").strip().lower()
    if not lowered:
        return True
    return any(hint in lowered for hint in PLACEHOLDER_HINTS)


def check_llm_key_configured(llm_api_key: str) -> RuntimeCheck:
    """检查 LLM Key 是否已配置为可用值。"""

    ok = not is_placeholder_secret(llm_api_key)
    return RuntimeCheck(
        name="LLM Key 配置",
        ok=ok,
        required=True,
        detail="已配置（已隐藏）" if ok else "未配置或仍是占位符",
        suggestion="" if ok else "在当前运行配置中填写真实可用的 DASHSCOPE_API_KEY；本机调试写 .env，Compose 部署写 .env.compose。",
    )


def check_admin_token_strength(admin_token: str) -> list[RuntimeCheck]:
    """检查管理口令是否既存在又不是默认弱口令。"""

    configured = bool((admin_token or "").strip())
    normalized = (admin_token or "").strip().lower()
    strong_enough = configured and normalized not in DEFAULT_ADMIN_TOKENS and len(admin_token.strip()) >= 16
    checks = [
        RuntimeCheck(
            name="ADMIN_API_TOKEN 配置",
            ok=configured,
            required=True,
            detail="已配置（已隐藏）" if configured else "未配置",
            suggestion="" if configured else "在当前运行配置中设置随机长令牌；本机调试写 .env，Compose 部署写 .env.compose。",
        )
    ]
    checks.append(
        RuntimeCheck(
            name="管理令牌强度",
            ok=strong_enough,
            required=False,
            detail="已使用自定义强令牌" if strong_enough else "当前仍使用默认、过短或过弱令牌",
            suggestion="" if strong_enough else "上线前请把 ADMIN_API_TOKEN 改成长度至少 16 位的随机长令牌，避免状态页被弱口令保护。",
        )
    )
    return checks


def resolve_powershell() -> str:
    """寻找当前机器可用的 PowerShell 可执行文件。"""
    for candidate in POWERSHELL_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.exists():
            return str(path)
    return ""


def check_windows_feature(feature_name: str, display_name: str) -> RuntimeCheck:
    """检查 Windows 可选功能状态。

    Docker Desktop 的 Linux 容器依赖 WSL 和虚拟机平台。常见问题是功能已经启用但处于
    `EnablePending`，这时必须重启 Windows，Docker 后端才会真正可用。
    """
    powershell = resolve_powershell()
    if not powershell:
        return RuntimeCheck(
            name=f"Windows 功能：{display_name}",
            ok=False,
            required=True,
            detail="未找到 powershell.exe 或 pwsh.exe",
            suggestion="确认 PowerShell 可执行文件在 PATH 中，或手动检查 Windows 功能状态。",
        )
    ok, output, _ = run_probe(
        [
            powershell,
            "-NoProfile",
            "-Command",
            f"(Get-WindowsOptionalFeature -Online -FeatureName {feature_name}).State",
        ]
    )
    state = output.strip().splitlines()[0] if output.strip() else ""
    if not ok:
        return RuntimeCheck(
            name=f"Windows 功能：{display_name}",
            ok=False,
            required=True,
            detail=output,
            suggestion="在 Windows 环境下运行该脚本，或手动确认 WSL / 虚拟机平台已启用。",
        )
    if state == "Enabled":
        return RuntimeCheck(name=f"Windows 功能：{display_name}", ok=True, required=True, detail=state)
    suggestion = "启用该 Windows 功能后重启系统，再启动 Docker Desktop。"
    if state == "EnablePending":
        suggestion = "该功能处于待生效状态，请先重启 Windows，再启动 Docker Desktop。"
    return RuntimeCheck(name=f"Windows 功能：{display_name}", ok=False, required=True, detail=state, suggestion=suggestion)


def check_command(name: str, command: list[str], suggestion: str, *, required: bool = True) -> RuntimeCheck:
    """运行一个只读命令式诊断。"""
    ok, output, returncode = run_probe(command)
    if name == "WSL 状态" and ("不支持 WSL2" in output or "enablevirtualization" in output.lower()):
        ok = False
        suggestion = "启用虚拟机平台并确认 BIOS 虚拟化开启，重启 Windows 后再启动 Docker Desktop。"
    return RuntimeCheck(
        name=name,
        ok=ok,
        required=required,
        detail=output or f"returncode={returncode}",
        suggestion="" if ok else suggestion,
    )


def check_tcp(name: str, host: str, port: int, suggestion: str, *, required: bool = True) -> RuntimeCheck:
    """检查 TCP 端口是否可连接。"""
    try:
        with socket.create_connection((host, port), timeout=3):
            return RuntimeCheck(name=name, ok=True, required=required, detail=f"{host}:{port}")
    except OSError as exc:
        return RuntimeCheck(name=name, ok=False, required=required, detail=f"{host}:{port}，{exc}", suggestion=suggestion)


def check_path(name: str, raw_path: str, suggestion: str) -> RuntimeCheck:
    """检查本地模型、场景目录等必需路径是否存在。"""
    path = Path(raw_path)
    return RuntimeCheck(name=name, ok=path.exists(), required=True, detail=str(path), suggestion="" if path.exists() else suggestion)


def relax_compose_network_host_checks(checks: list[RuntimeCheck], runtime_mode: str) -> None:
    """在宿主机执行脚本但 `.env` 使用 compose 网络名时，避免误报硬失败。

    `MYSQL_HOST=mysql`、`MILVUS_URI=http://milvus:19530` 和 `/app/models/...`
    是 API 容器内部视角。用户在 Windows 宿主机直接运行该诊断脚本时，这些名称和路径
    本来就不可解析。此时应把它们降级为模式提示，由 compose 对齐检查承担配置一致性判断。
    """

    if runtime_mode != "compose_api_network":
        return
    targets = {
        "Milvus 端口": "compose 网络模式下，该端口应在 API 容器内检查；宿主机侧可用 docker compose --env-file .env.compose ps 或 API healthcheck 验证。",
        "MySQL 端口": "compose 网络模式下，该端口应在 API 容器内检查；宿主机侧可用 docker compose --env-file .env.compose ps 或 API healthcheck 验证。",
        "Embedding 模型目录": "compose 网络模式下，/app/models 由 docker-compose.yml 的 ./models:/app/models 挂载提供。",
        "Reranker 模型目录": "compose 网络模式下，/app/models 由 docker-compose.yml 的 ./models:/app/models 挂载提供。",
    }
    for name, suggestion in targets.items():
        for item in checks:
            if item.name == name and not item.ok:
                update_check(
                    checks,
                    name,
                    required=False,
                    detail=f"{item.detail}（当前 .env 使用 compose 网络/容器路径，宿主机直接运行诊断时此项仅作提示）",
                    suggestion=suggestion,
                )
                break


def check_active_versions() -> RuntimeCheck:
    """检查冻结的 8 个业务场景是否都存在 active 知识库版本。"""
    missing: list[str] = []
    active_versions: dict[str, str] = {}
    for scenario in get_scenario_registry().list_scenarios():
        store = get_kb_version_store(scenario.scenario_id)
        try:
            active_versions[scenario.scenario_id] = store.resolve_active_version()
        except ValueError as exc:
            missing.append(f"{scenario.scenario_id}: {exc}")
    if not missing:
        return RuntimeCheck(
            name="知识库 active 版本",
            ok=True,
            required=True,
            detail=json.dumps(active_versions, ensure_ascii=False),
        )
    return RuntimeCheck(
        name="知识库 active 版本",
        ok=False,
        required=True,
        detail="；".join(missing),
        suggestion="先执行 scripts/rebuild_kb_version.py --scenario <场景ID> --new-version --force --quality-gate --activate。",
    )


def relax_windows_feature_requirement_if_runtime_ready(checks: list[RuntimeCheck]) -> None:
    """当 WSL、Docker 和核心端口已经真实可用时，放宽 Windows 特性硬失败。

    某些 Windows 环境里，`Get-WindowsOptionalFeature` 读取到的状态和 Docker/WSL 的实际
    可用性并不完全一致。既然当前脚本的目标是判断“项目能不能通电”，就应该优先相信
    已经跑起来的事实，而不是让一个过严的前置标签阻断 live 验收。
    """

    status = {item.name: item for item in checks}
    runtime_ready = all(
        status.get(name) and status[name].ok
        for name in ("WSL 状态", "Docker 服务端", "Milvus 端口", "MySQL 端口")
    )
    feature = status.get("Windows 功能：虚拟机平台")
    if not runtime_ready or feature is None or feature.ok:
        return
    update_check(
        checks,
        "Windows 功能：虚拟机平台",
        required=False,
        detail=f"{feature.detail}（当前 WSL、Docker、Milvus、MySQL 已实际可用，因此作为提示项展示）",
        suggestion="建议后续仍修正该 Windows 功能状态，避免 Docker/WSL 在系统更新后再次异常。",
    )


def build_runtime_report(args: argparse.Namespace) -> dict[str, object]:
    """生成本地运行环境诊断报告。"""
    settings = get_settings()
    if args.admin_token_only:
        checks = check_admin_token_strength(settings.admin_api_token)
        failed_required = [check for check in checks if check.required and not check.ok]
        failed_optional = [check for check in checks if not check.required and not check.ok]
        return {
            "report_type": "admin_token_strength_check",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ok": not failed_required and not failed_optional,
            "required_failed_count": len(failed_required),
            "optional_failed_count": len(failed_optional),
            "checks": [asdict(check) for check in checks],
        }
    env_values = parse_env_file(Path(settings.model_config.get("env_file", "")))
    parsed_milvus = urlparse(settings.milvus_uri)
    milvus_host = parsed_milvus.hostname or "localhost"
    milvus_port = parsed_milvus.port or 19530
    compose_summaries = list_compose_summaries(Path(__file__).resolve().parents[2])
    containers, docker_ps_error = list_running_containers()
    runtime_mode, _, _ = detect_runtime_mode(settings.mysql_host, milvus_host)
    checks: list[RuntimeCheck] = []
    for feature_name, display_name in WINDOWS_FEATURES.items():
        checks.append(check_windows_feature(feature_name, display_name))
    checks.extend(
        [
            check_command("WSL 状态", ["wsl", "--status"], "修复 WSL 后再启动 Docker Desktop。"),
            check_command("Docker 服务端", ["docker", "version", "--format", "{{.Server.Version}}"], "启动 Docker Desktop，并确认 Linux 容器后端正常。"),
            check_tcp("Milvus 端口", milvus_host, milvus_port, "执行 docker compose --env-file .env.compose up -d etcd minio milvus。"),
            check_tcp("MySQL 端口", settings.mysql_host, settings.mysql_port, "执行 docker compose --env-file .env.compose up -d mysql，并确认账号密码与当前运行配置一致。"),
            check_llm_key_configured(settings.llm_api_key),
            check_path("Embedding 模型目录", settings.embedding_model_path, "下载或放置 models/bge-m3。"),
            check_path("Reranker 模型目录", settings.reranker_model_path, "下载或放置 models/bge-reranker-large。"),
            check_path("场景配置目录", settings.scenario_config_dir, "确认 scenarios 目录存在且包含 8 个冻结场景。"),
            check_active_versions(),
        ]
    )
    checks.extend(check_admin_token_strength(settings.admin_api_token))
    checks.append(
        check_tcp(
            "FastAPI 端口",
            args.api_host,
            args.api_port,
            "启动 API：python -m uvicorn app:app --host 127.0.0.1 --port 8000。",
            required=args.require_api,
        )
    )
    checks.append(summarize_compose_alignment(settings, args.api_host, args.api_port, compose_summaries))
    relax_compose_network_host_checks(checks, runtime_mode)
    if runtime_mode == "local_api_host_ports":
        checks.extend(
            [
                summarize_container_mapping("mysql", settings.mysql_port, containers),
                summarize_container_mapping("milvus", milvus_port, containers),
                summarize_container_mapping("api", args.api_port, containers),
            ]
        )
    relax_windows_feature_requirement_if_runtime_ready(checks)
    failed_required = [check for check in checks if check.required and not check.ok]
    failed_optional = [check for check in checks if not check.required and not check.ok]
    return {
        "report_type": "local_runtime_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ok": not failed_required,
        "required_failed_count": len(failed_required),
        "optional_failed_count": len(failed_optional),
        "runtime_mode": runtime_mode,
        "env_snapshot": {
            "env_file_found": bool(env_values),
            "active_scenario_id": settings.active_scenario_id,
            "mysql_host": settings.mysql_host,
            "mysql_port": settings.mysql_port,
            "milvus_uri": settings.milvus_uri,
            "api_host": args.api_host,
            "api_port": args.api_port,
            "llm_key_configured": not is_placeholder_secret(settings.llm_api_key),
            "admin_token_configured": bool((settings.admin_api_token or "").strip()),
            "admin_token_is_default": (settings.admin_api_token or "").strip().lower() in DEFAULT_ADMIN_TOKENS,
        },
        "compose_summaries": compose_summaries,
        "running_containers": [
            {"name": item.name, "ports": item.ports, "host_ports": list(item.host_ports)}
            for item in containers
        ],
        "docker_ps_error": docker_ps_error,
        "checks": [asdict(check) for check in checks],
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Check local Docker / Milvus / MySQL / API runtime.")
    parser.add_argument("--output", default="", help="可选 JSON 报告输出路径。")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--require-api", action="store_true", help="把 FastAPI 端口也作为必过项。")
    parser.add_argument("--admin-token-only", action="store_true", help="只检查 ADMIN_API_TOKEN 配置和强度，不探测 Milvus/MySQL/模型目录。")
    return parser


def main() -> None:
    """运行本地环境诊断。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = build_runtime_report(args)
    write_optional_json(args.output, report)
    print_json(report)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

