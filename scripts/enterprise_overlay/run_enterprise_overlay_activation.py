"""执行企业 clean overlay 上线计划。

`plan_enterprise_overlay_activation.py` 只负责生成命令，不连接 Milvus；本脚本负责在发布
门禁中按计划执行这些命令。这样 clean overlay 的流程可以保持三段清晰：

1. 预检候选资料是否干净；
2. 生成每个场景的标准重建命令；
3. 在真实环境中重建并激活 active 知识库版本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import CommandStepResult, configure_utf8_stdio, read_json_file, run_command_step, utc_now, write_json_file


DEFAULT_PLAN = PROJECT_ROOT / "reports" / "verification" / "enterprise_overlay_activation_plan_latest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "verification" / "enterprise_overlay_activation_run_latest.json"
REBUILD_SCRIPT = "scripts/rebuild_kb_version.py"
REQUIRED_FLAGS = ("--scenario", "--data-dir", "--faq-csv", "--new-version", "--force", "--quality-gate", "--activate")


def resolve_path(value: str | Path) -> Path:
    """把相对路径解析到项目根目录下。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_activation_command(command: list[Any]) -> list[str]:
    """校验并规范化单条 overlay 激活命令。

    使用场景：
    - 上线计划里保存的是可读的 `python scripts/rebuild_kb_version.py ...`；
    - 自动化执行时需要使用当前虚拟环境的 Python，避免 Windows 上命中其它解释器；
    - 只允许执行 `rebuild_kb_version.py`，并且必须带 `--quality-gate` 和 `--activate`。

    这不是通用命令执行器。它只接受本项目生成的知识库重建命令，避免把本地 JSON 计划
    变成任意命令执行入口。
    """
    normalized = [str(item) for item in command]
    if len(normalized) < 2:
        raise ValueError("overlay 激活命令为空或格式不完整。")
    script = normalized[1].replace("\\", "/")
    if script != REBUILD_SCRIPT:
        raise ValueError(f"overlay 激活只允许执行 {REBUILD_SCRIPT}，当前为：{normalized[1]}")
    missing_flags = [flag for flag in REQUIRED_FLAGS if flag not in normalized]
    if missing_flags:
        raise ValueError(f"overlay 激活命令缺少必要参数：{missing_flags}")
    normalized[0] = sys.executable
    return normalized


def command_argument(command: list[str], flag: str) -> str:
    """读取命令参数值。

    overlay 计划里的命令是由本项目生成的标准参数列表。这里不做 shell 字符串解析，只按
    参数数组查找，避免 Windows 引号和空格路径导致误判。
    """
    if flag not in command:
        return ""
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else ""


def hash_file(path: Path) -> str:
    """计算单个文件内容指纹。

    使用内容 hash，而不是 mtime。这样只要资料内容不变，即使文件被复制或重新生成，也
    不会触发无意义的 overlay 重建。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_fingerprint(path: Path) -> dict[str, Any]:
    """计算文件或目录的稳定指纹。

    data 目录会包含多个 source 文档；FAQ 是单个 CSV。自动化检查需要知道“上线输入资料”
    是否真的变化，所以把相对路径、文件大小和内容 hash 一起写入签名对象。
    """
    actual_path = resolve_path(path)
    if not actual_path.exists():
        return {"exists": False, "path": str(actual_path)}
    if actual_path.is_file():
        return {
            "exists": True,
            "path": str(actual_path),
            "kind": "file",
            "size": actual_path.stat().st_size,
            "sha256": hash_file(actual_path),
        }
    files = []
    for file_path in sorted(item for item in actual_path.rglob("*") if item.is_file()):
        files.append(
            {
                "rel_path": file_path.relative_to(actual_path).as_posix(),
                "size": file_path.stat().st_size,
                "sha256": hash_file(file_path),
            }
        )
    return {"exists": True, "path": str(actual_path), "kind": "directory", "files": files}


def activation_signature(command: list[str]) -> str:
    """为单条 overlay 激活命令生成资料签名。

    签名只关注会影响知识库内容的输入：场景、命令参数、FAQ CSV 和 data 目录。当前 Python
    解释器路径不参与签名，否则换虚拟环境会导致同一批资料被误判为变化。
    """
    payload = {
        "command": command[1:],
        "scenario": command_argument(command, "--scenario"),
        "faq": path_fingerprint(Path(command_argument(command, "--faq-csv"))),
        "data": path_fingerprint(Path(command_argument(command, "--data-dir"))),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def selected_plan_items(plan: dict[str, Any], selected_scenarios: set[str]) -> list[dict[str, Any]]:
    """按命令行场景过滤上线计划。"""
    items: list[dict[str, Any]] = []
    for item in list(plan.get("commands") or []):
        scenario_id = str(item.get("scenario_id") or "")
        if selected_scenarios and scenario_id not in selected_scenarios:
            continue
        items.append(dict(item))
    return items


def build_activation_signatures(items: list[dict[str, Any]]) -> dict[str, str]:
    """生成本次计划内每个场景的 overlay 资料签名。"""
    signatures: dict[str, str] = {}
    for item in items:
        scenario_id = str(item.get("scenario_id") or "")
        command = normalize_activation_command(list(item.get("command") or []))
        signatures[scenario_id] = activation_signature(command)
    return signatures


def matching_previous_report(output_path: str | Path, signatures: dict[str, str]) -> dict[str, Any]:
    """读取可复用的上一次激活报告。

    只有上次执行成功，并且本次所有场景的资料签名完全一致，才允许跳过激活。这样既避免
    重复执行高成本入库，也不会在资料变化时误用旧 active 版本。
    """
    report_path = resolve_path(output_path)
    if not report_path.exists() or not signatures:
        return {}
    try:
        report = read_json_file(report_path)
    except Exception:
        return {}
    previous_signatures = dict(report.get("overlay_signatures") or {})
    if report.get("ok") is not True:
        return {}
    if all(previous_signatures.get(scenario_id) == signature for scenario_id, signature in signatures.items()):
        return report
    return {}


def skipped_activation_report(
    *,
    plan_path: Path,
    previous: dict[str, Any],
    signatures: dict[str, str],
) -> dict[str, Any]:
    """构造资料未变化时的跳过报告。"""
    previous_results = list(previous.get("results") or [])
    selected_scenarios = set(signatures)
    results = []
    for item in previous_results:
        scenario_id = str(item.get("scenario_id") or "")
        if scenario_id not in selected_scenarios:
            continue
        row = dict(item)
        row["ok"] = True
        row["skipped"] = True
        row["stdout_preview"] = "overlay 资料指纹未变化，复用上次成功激活结果。"
        row["stderr_preview"] = ""
        results.append(row)
    return {
        "report_type": "enterprise_overlay_activation_run",
        "created_at": utc_now(),
        "ok": bool(results),
        "plan_path": str(plan_path),
        "scenario_count": len(results),
        "succeeded_count": len(results),
        "failed_count": 0,
        "executed_count": 0,
        "skipped_count": len(results),
        "idempotent": True,
        "overlay_signatures": signatures,
        "results": results,
    }


def command_result_payload(scenario_id: str, result: CommandStepResult) -> dict[str, Any]:
    """把命令结果转换成报告行。"""
    payload = asdict(result)
    payload["scenario_id"] = scenario_id
    return payload


def run_activation_plan(args: argparse.Namespace) -> dict[str, Any]:
    """读取上线计划并执行全部场景的 overlay 激活命令。"""
    plan_path = resolve_path(args.plan)
    plan = read_json_file(plan_path)
    selected_scenarios = {item.strip() for item in args.scenarios.split(",") if item.strip()}
    if plan.get("report_type") != "enterprise_overlay_activation_plan":
        raise ValueError(f"不是企业 overlay 上线计划：{plan_path}")
    if plan.get("ok") is not True:
        return {
            "report_type": "enterprise_overlay_activation_run",
            "created_at": utc_now(),
            "ok": False,
            "plan_path": str(plan_path),
            "scenario_count": 0,
            "succeeded_count": 0,
            "failed_count": 1,
            "results": [],
            "failure": "上线计划未通过，禁止执行激活。",
        }

    items = selected_plan_items(plan, selected_scenarios)
    signatures = build_activation_signatures(items)
    if not args.force:
        previous = matching_previous_report(args.output, signatures)
        if previous:
            report = skipped_activation_report(plan_path=plan_path, previous=previous, signatures=signatures)
            write_json_file(args.output, report)
            return report

    results: list[dict[str, Any]] = []
    for item in items:
        scenario_id = str(item.get("scenario_id") or "")
        try:
            command = normalize_activation_command(list(item.get("command") or []))
            result = run_command_step(f"激活 clean overlay：{scenario_id}", command, preview_limit=args.preview_limit)
        except Exception as exc:
            result = CommandStepResult(
                name=f"激活 clean overlay：{scenario_id or 'unknown'}",
                command=[str(part) for part in list(item.get("command") or [])],
                ok=False,
                elapsed_ms=0,
                stdout_preview="",
                stderr_preview=str(exc),
                returncode=1,
            )
        results.append(command_result_payload(scenario_id, result))
        if not result.ok:
            break

    failed_count = sum(1 for item in results if not item["ok"])
    report = {
        "report_type": "enterprise_overlay_activation_run",
        "created_at": utc_now(),
        "ok": bool(results) and failed_count == 0,
        "plan_path": str(plan_path),
        "scenario_count": len(results),
        "succeeded_count": sum(1 for item in results if item["ok"]),
        "failed_count": failed_count,
        "executed_count": len(results),
        "skipped_count": 0,
        "idempotent": False,
        "overlay_signatures": signatures,
        "results": results,
    }
    write_json_file(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Run enterprise clean overlay activation plan.")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="plan_enterprise_overlay_activation.py 生成的计划文件。")
    parser.add_argument("--scenarios", default="", help="逗号分隔的场景 ID；为空时执行计划中的全部场景。")
    parser.add_argument("--preview-limit", type=int, default=1600, help="每条命令 stdout/stderr 写入报告的最大字符数。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="执行报告输出路径。")
    parser.add_argument("--force", action="store_true", help="即使资料指纹未变化，也强制重新执行 overlay 激活。")
    return parser


def main() -> None:
    """执行企业 clean overlay 激活计划。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = run_activation_plan(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

