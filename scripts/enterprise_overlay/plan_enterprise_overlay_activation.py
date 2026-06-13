"""生成企业 clean overlay 从预检到激活的上线计划。

该脚本不执行入库、不连接 Milvus，只读取 `enterprise_overlay_build_latest.json`，为每个
已通过预检的场景生成标准 `rebuild_kb_version.py` 命令。这样可以把“候选资料预检通过”
和“正式激活 active 知识库版本”两个动作清楚分开。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, read_json_file, write_optional_json


DEFAULT_OVERLAY_REPORT = PROJECT_ROOT / "reports" / "verification" / "enterprise_overlay_build_latest.json"


def _resolve_path(value: str | Path) -> Path:
    """把命令行路径解析成项目内绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def activation_command(row: dict[str, Any], *, description: str, activate: bool) -> list[str]:
    """根据单个 overlay 预检结果生成知识库重建命令。"""
    command = [
        "python",
        "scripts/rebuild_kb_version.py",
        "--scenario",
        str(row["scenario_id"]),
        "--data-dir",
        str(row["data_dir"]),
        "--faq-csv",
        str(row["faq_csv"]),
        "--new-version",
        "--force",
        "--quality-gate",
        "--description",
        description,
    ]
    if activate:
        command.append("--activate")
    return command


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """构建 overlay 上线计划。"""
    report_path = _resolve_path(args.overlay_report)
    report = read_json_file(report_path)
    rows = [row for row in report.get("rows") or [] if (row.get("gate") or {}).get("ok") is True]
    blocked = [row for row in report.get("rows") or [] if (row.get("gate") or {}).get("ok") is not True]
    commands = [
        {
            "scenario_id": row["scenario_id"],
            "command": activation_command(row, description=args.description, activate=not args.no_activate),
        }
        for row in rows
    ]
    return {
        "report_type": "enterprise_overlay_activation_plan",
        "ok": bool(report.get("ok")) and not blocked,
        "overlay_report": str(report_path),
        "activate": not args.no_activate,
        "scenario_count": len(commands),
        "blocked_scenario_count": len(blocked),
        "commands": commands,
        "post_activation_checks": [
            [
                "python",
                "scripts/check_evaluation_gate.py",
                "--dataset",
                "eval_sets/enterprise_overlay_regression.json",
                "--limit",
                "24",
                "--output",
                "reports/verification/enterprise_overlay_evaluation_latest.json",
                "--gate-output",
                "reports/verification/enterprise_overlay_evaluation_gate_latest.json",
                "--min-recall-at-k",
                "0.8",
                "--min-source-inference-accuracy",
                "1.0",
                "--min-prompt-profile-accuracy",
                "0.85",
            ]
        ],
        "recommendation": "先逐条执行 commands 重建并激活，再执行 post_activation_checks 验证增强资料真实链路效果。",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate activation plan for enterprise clean overlay.")
    parser.add_argument("--overlay-report", default=str(DEFAULT_OVERLAY_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--description", default="企业 clean overlay 增强资料版本")
    parser.add_argument("--no-activate", action="store_true", help="只生成 staged 重建命令，不带 --activate。")
    parser.add_argument("--output", default="", help="计划输出路径。")
    return parser


def main() -> None:
    """输出 overlay 上线计划。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    plan = build_plan(args)
    write_optional_json(args.output, plan)
    print_json(plan)
    if not plan["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

