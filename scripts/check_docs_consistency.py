"""检查文档和当前一期工程边界是否一致。

这不是语法检查，而是防止 README、架构文档和验收命令在多轮优化后滞后。当前项目已经
冻结为 8 个业务场景，一期只做 RAG；这些边界如果文档里说错，会让学习路径和交付口径
同时跑偏，所以保留一个轻量自检脚本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_utf8_stdio, utc_now, write_optional_json


FROZEN_SCENARIO_COUNT = 8
REQUIRED_PATHS = (
    "README.md",
    "docs/index.md",
    "docs/course-outline.md",
    "docs/01-project-overview.md",
    "docs/19-observability-tracing.md",
    "docs/appendix/appendix-h-tool-foundations.md",
    "scripts/enterprise_overlay/run_enterprise_overlay_activation.py",
    "eval_sets/business_depth_regression.json",
    "eval_sets/enterprise_overlay_regression.json",
    "eval_sets/phase1_performance_baseline.json",
)
README_REQUIRED_SNIPPETS = (
    "LangChain + Milvus Hybrid Search + FastAPI",
    "Bad Case 闭环",
    "一期源码不提前放 Agent 预留实现",
    "GraphRAG",
)
COURSE_REQUIRED_SNIPPETS = (
    "19 讲系统化课程",
    "P3 扩展方向",
    "GraphRAG Agent",
    "Router/Planner",
    "01 → 19",
)


def text_of(path: Path) -> str:
    """读取文本文件；不存在时返回空字符串。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def check_required_paths() -> list[dict[str, Any]]:
    """检查文档和关键脚本是否存在。"""
    failures: list[dict[str, Any]] = []
    for rel_path in REQUIRED_PATHS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            failures.append({"metric": "required_path", "path": rel_path, "message": "必需文件不存在"})
    return failures


def check_readme() -> list[dict[str, Any]]:
    """检查 README 是否仍反映当前一期边界。"""
    failures: list[dict[str, Any]] = []
    readme = text_of(PROJECT_ROOT / "README.md")
    for snippet in README_REQUIRED_SNIPPETS:
        if snippet not in readme:
            failures.append({"metric": "readme_snippet", "path": "README.md", "message": f"README 缺少关键说明：{snippet}"})
    scenario_section = readme.split("## 2. 业务场景", 1)[-1].split("\n## ", 1)[0]
    scenario_rows = [line for line in scenario_section.splitlines() if line.startswith("| `") and "` |" in line]
    if len(scenario_rows) != FROZEN_SCENARIO_COUNT:
        failures.append(
            {
                "metric": "readme_scenario_count",
                "path": "README.md",
                "message": f"README 场景表应为 {FROZEN_SCENARIO_COUNT} 行，当前 {len(scenario_rows)} 行",
            }
        )
    return failures


def check_course_outline() -> list[dict[str, Any]]:
    """检查课程大纲是否保留当前 19 讲和二期边界表达。"""
    failures: list[dict[str, Any]] = []
    outline = text_of(PROJECT_ROOT / "docs" / "course-outline.md")
    for snippet in COURSE_REQUIRED_SNIPPETS:
        if snippet not in outline:
            failures.append(
                {
                    "metric": "course_outline_snippet",
                    "path": "docs/course-outline.md",
                    "message": f"课程大纲缺少关键说明：{snippet}",
                }
            )
    return failures


def build_report() -> dict[str, Any]:
    """生成文档一致性检查报告。"""
    failures = [*check_required_paths(), *check_readme(), *check_course_outline()]
    return {
        "report_type": "docs_consistency_check",
        "created_at": utc_now(),
        "ok": not failures,
        "frozen_scenario_count": FROZEN_SCENARIO_COUNT,
        "checked_path_count": len(REQUIRED_PATHS),
        "failed_count": len(failures),
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Check documentation consistency for phase-1 RAG project.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "verification" / "docs_consistency_latest.json"))
    return parser


def main() -> None:
    """执行文档一致性检查。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    payload = build_report()
    write_optional_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
