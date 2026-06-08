"""脚本层公共工具。

`scripts/` 里的文件越来越多，如果每个脚本都重复处理 JSON 读写、UTF-8 输出、命令执行
和报告保存，学生阅读时会被样板代码淹没。本模块只放“脚本基础设施”，不放 RAG 业务
规则；业务规则仍留在各自脚本里。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SUMMARY_KEYS = (
    "total",
    "errors",
    "recall_at_k",
    "mrr",
    "avg_keyword_coverage",
    "source_inference_accuracy",
    "prompt_profile_accuracy",
    "scenario_isolation_accuracy",
    "avg_total_ms",
    "p95_total_ms",
    "avg_first_token_ms",
    "p95_first_token_ms",
    "scenario_count",
    "comparable_scenario_count",
    "total_regressions",
    "total_errors",
    "skipped_scenario_count",
    "failed_count",
    "ready_count",
    "pending_count",
    "executed_count",
    "skipped_count",
)


@dataclass(frozen=True)
class CommandStepResult:
    """一条命令式验收步骤的执行结果。"""

    name: str
    command: list[str]
    ok: bool
    elapsed_ms: float
    stdout_preview: str
    stderr_preview: str
    returncode: int


def configure_utf8_stdio() -> None:
    """把脚本标准输出统一成 UTF-8。

    Windows PowerShell 的默认编码可能不是 UTF-8，一键验收里又会输出中文 JSON。这里集中
    处理，避免每个脚本单独写一遍编码保护。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_json_file(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件并返回对象。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    """返回 UTC 时间字符串。

    发布、验收和质量报告都用这个函数生成时间，避免每个脚本重复导入 datetime，也避免
    一部分报告使用本地时间、一部分报告使用 UTC。
    """
    return datetime.now(timezone.utc).isoformat()


def read_json_report(path: str | Path) -> dict[str, Any]:
    """读取报告 JSON；缺失或损坏时返回统一失败对象。

    使用场景：
    - 就绪总报告、发布可读报告和交付报告都只读取已有离线报告；
    - 报告缺失不应该抛异常中断页面或脚本，而应该以 `missing=true` 明确展示；
    - 调用方可以继续用同一套 `ok/path/reason` 字段生成 Markdown 或状态页摘要。
    """
    report_path = Path(path)
    if not report_path.exists():
        return {"ok": False, "missing": True, "path": str(report_path), "reason": "报告不存在"}
    try:
        payload = read_json_file(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "missing": False, "path": str(report_path), "reason": str(exc)}
    payload.setdefault("path", str(report_path))
    return payload


def compact_report_summary(report: dict[str, Any], *, keys: tuple[str, ...] = REPORT_SUMMARY_KEYS) -> dict[str, Any]:
    """抽取报告中的高信号字段，避免汇总报告被完整 JSON 撑大。

    不同报告有两种常见形态：指标直接在顶层，或放在 `metrics` 下。这里统一处理，并额外
    统计 `steps/checks` 的通过数量，让质量检查和 API 冒烟也能用同一套摘要口径。
    """
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    summary = {key: metrics.get(key) for key in keys if key in metrics}
    if isinstance(report.get("checks"), dict):
        checks = dict(report["checks"])
        summary["checks_passed"] = sum(1 for value in checks.values() if value)
        summary["checks_total"] = len(checks)
    if isinstance(report.get("steps"), list):
        steps = list(report["steps"])
        summary["steps_passed"] = sum(1 for item in steps if isinstance(item, dict) and item.get("ok"))
        summary["steps_total"] = len(steps)
    return summary


def status_text(value: Any) -> str:
    """把布尔状态转成人能快速扫描的中文。"""
    if value is True:
        return "通过"
    if value is False:
        return "未通过"
    return "未生成"


def report_status_text(report: dict[str, Any]) -> str:
    """把报告对象转换成状态文本，优先区分缺失报告。"""
    if report.get("missing"):
        return "缺失"
    return status_text(report.get("ok"))


def number_text(value: Any, digits: int = 4) -> str:
    """格式化报告中的数值指标。"""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return "未生成"


def write_json_file(path: str | Path, payload: dict[str, Any]) -> str:
    """把对象写成中文友好的 JSON 文件，并返回写入路径。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def print_json(payload: dict[str, Any]) -> None:
    """按统一格式打印 JSON。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_optional_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    """当调用方提供路径时写 JSON；未提供时什么都不做。"""
    if path:
        write_json_file(path, payload)


def preview_text(text: str, limit: int = 1200) -> str:
    """截断长输出，避免验收报告被命令日志撑爆。"""
    compact = (text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "\n..."


def run_command_step(name: str, command: list[str], *, preview_limit: int = 1200) -> CommandStepResult:
    """运行一个验收命令并记录结果。

    这里不用 shell 拼接命令，是为了减少 Windows 下路径、引号和转义问题。每个步骤独立
    执行，某一步失败后仍继续跑后续步骤，最后统一汇总失败项。
    """
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return CommandStepResult(
        name=name,
        command=command,
        ok=completed.returncode == 0,
        elapsed_ms=elapsed_ms,
        stdout_preview=preview_text(completed.stdout, preview_limit),
        stderr_preview=preview_text(completed.stderr, preview_limit),
        returncode=completed.returncode,
    )
