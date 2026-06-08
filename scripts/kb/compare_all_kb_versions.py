"""批量对比全部场景的新旧知识库版本召回结果。

单场景 `compare_kb_versions.py` 适合排查某个场景；本脚本适合发布前总览 8 个冻结场景。
它仍然复用 QAService.debug_retrieval，不新建检索逻辑，也不绕过 kb_version 和数据隔离。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.application.factory import get_qa_service
from qa_core.scenarios.registry import get_scenario_registry
from scripts.common import configure_utf8_stdio, utc_now, write_json_file
from scripts.kb.compare_kb_versions import compare_case, resolve_versions
from scripts.eval_common import load_eval_items


def selected_scenario_ids(raw_value: str) -> list[str]:
    """解析命令行中的场景列表；未传时使用全部冻结场景。"""
    registry = get_scenario_registry()
    if raw_value.strip():
        requested = [item.strip() for item in raw_value.split(",") if item.strip()]
        available = {scenario.scenario_id for scenario in registry.list_scenarios()}
        unknown = [scenario_id for scenario_id in requested if scenario_id not in available]
        if unknown:
            raise ValueError(f"未知场景：{unknown}")
        return requested
    return [scenario.scenario_id for scenario in registry.list_scenarios()]


def scenario_items(items: list[dict[str, Any]], scenario_id: str, limit: int) -> list[dict[str, Any]]:
    """取出某个场景的评测样本，并按每场景 limit 截断。"""
    filtered = [item for item in items if item.get("scenario_id") == scenario_id]
    return filtered[:limit] if limit > 0 else filtered


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总单场景版本对比结果。"""
    comparable_rows = [row for row in rows if row.get("candidate_recall_hit") is not None and not row.get("error")]
    regressions = [row for row in comparable_rows if row.get("base_recall_hit") and not row.get("candidate_recall_hit")]
    improvements = [row for row in comparable_rows if not row.get("base_recall_hit") and row.get("candidate_recall_hit")]
    return {
        "total": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "top_source_changed_count": sum(1 for row in rows if row.get("top_source_changed")),
        "base_recall_at_k": round(sum(1 for row in comparable_rows if row.get("base_recall_hit")) / max(len(comparable_rows), 1), 4),
        "candidate_recall_at_k": round(sum(1 for row in comparable_rows if row.get("candidate_recall_hit")) / max(len(comparable_rows), 1), 4),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "regressions": regressions,
        "improvements": improvements,
    }


def compare_scenario(service, items: list[dict[str, Any]], scenario_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """对比一个场景的 base/candidate 版本。"""
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.scenario = scenario_id
    scenario_args.kb_version = None
    filtered_items = scenario_items(items, scenario_id, args.per_scenario_limit)
    if not filtered_items:
        return {"scenario_id": scenario_id, "skipped": True, "reason": "评测集中没有该场景样本。", "ok": True, "rows": []}
    try:
        base_version, candidate_version = resolve_versions(scenario_args)
    except Exception as exc:
        return {"scenario_id": scenario_id, "skipped": True, "reason": str(exc), "ok": not args.require_all, "rows": []}

    rows = [
        compare_case(service, item, index, scenario_args, base_version, candidate_version)
        for index, item in enumerate(filtered_items, start=1)
    ]
    summary = summarize_rows(rows)
    return {
        "scenario_id": scenario_id,
        "skipped": False,
        "reason": "",
        "base_version": base_version,
        "candidate_version": candidate_version,
        "ok": summary["regression_count"] == 0 and summary["errors"] == 0,
        **summary,
        "rows": rows,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """执行全场景版本对比并返回报告。"""
    items = load_eval_items(args.dataset, args.limit)
    service = get_qa_service()
    scenario_ids = selected_scenario_ids(args.scenarios)
    scenarios = [compare_scenario(service, items, scenario_id, args) for scenario_id in scenario_ids]
    comparable = [scenario for scenario in scenarios if not scenario.get("skipped")]
    skipped = [scenario for scenario in scenarios if scenario.get("skipped")]
    hard_failures = [scenario for scenario in scenarios if not scenario.get("ok")]
    return {
        "report_type": "all_scenario_kb_version_retrieval_compare",
        "created_at": utc_now(),
        "dataset": str(Path(args.dataset)),
        "scenario_count": len(scenarios),
        "comparable_scenario_count": len(comparable),
        "skipped_scenario_count": len(skipped),
        "total_cases": sum(int(scenario.get("total") or 0) for scenario in scenarios),
        "total_regressions": sum(int(scenario.get("regression_count") or 0) for scenario in scenarios),
        "total_errors": sum(int(scenario.get("errors") or 0) for scenario in scenarios),
        "ok": not hard_failures,
        "skipped": skipped,
        "scenarios": scenarios,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Compare retrieval between previous and active KB versions for all scenarios.")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "business_depth_regression.json"))
    parser.add_argument("--limit", type=int, default=999, help="从评测集读取的最大样本数。")
    parser.add_argument("--per-scenario-limit", type=int, default=4, help="每个场景最多对比多少条样本，0 表示不限制。")
    parser.add_argument("--scenarios", default="", help="逗号分隔的场景 ID；为空时对比全部场景。")
    parser.add_argument("--require-all", action="store_true", help="要求每个场景都必须存在 previous/candidate 版本并完成对比。")
    parser.add_argument("--base-version", default="", help="传给单场景解析；全场景通常不使用。")
    parser.add_argument("--candidate-version", default="", help="传给单场景解析；全场景通常不使用。")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--user-role", default=None)
    parser.add_argument("--kb-version", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "verification" / "kb_version_compare_all_latest.json"))
    return parser


def main() -> None:
    """执行全场景知识库版本召回对比。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = build_report(args)
    write_json_file(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

