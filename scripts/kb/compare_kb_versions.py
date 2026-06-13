"""知识库版本召回对比报告。

该脚本复用 QAService 的 `debug_retrieval`，对同一批问题分别指定旧版和新版 kb_version，
比较 FAQ/doc 来源排名、预期来源召回和 top source 是否变化。它不生成答案，专注判断
新版知识库上线前是否出现召回退化。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa_core.application.factory import get_qa_service
from qa_core.governance.kb_versions import get_kb_version_store
from scripts.common import configure_utf8_stdio, utc_now, write_json_file
from scripts.eval_common import EvalCaseRuntime, load_eval_items
from scripts.evaluate_core_chain import find_expected_source_rank


def source_signature(source: dict[str, Any]) -> str:
    """把一个来源压缩成可比较的稳定标签。"""
    metadata = source.get("metadata") or {}
    return str(
        source.get("citation")
        or metadata.get("file_name")
        or metadata.get("standard_question")
        or metadata.get("source")
        or source.get("content")
        or ""
    )[:160]


def ranked_sources_for_compare(sources: list[dict[str, Any]], *, prefer_table: bool) -> list[dict[str, Any]]:
    """返回用于版本对比的来源顺序。

    表格问题的对比目标是确认行级文档证据是否稳定，因此 top source 也应优先看 doc
    来源。否则泛化 FAQ 会排在报告第一位，让版本对比读起来像没有命中表格。
    """
    if not prefer_table:
        return sources
    doc_sources = [source for source in sources if source.get("source_type") == "doc"]
    return doc_sources or sources


def resolve_versions(args: argparse.Namespace) -> tuple[str, str]:
    """解析对比用的旧版和新版版本号。"""
    store = get_kb_version_store(args.scenario)
    payload = store.as_payload()
    candidate_version = args.candidate_version or payload.get("effective_active_version")
    base_version = args.base_version or payload.get("previous_version")
    if not candidate_version:
        raise ValueError("未找到 candidate 版本，请传入 --candidate-version 或先激活一个版本。")
    if not base_version:
        raise ValueError("未找到 base 版本，请传入 --base-version。")
    return str(base_version), str(candidate_version)


def debug_case(service, runtime: EvalCaseRuntime, kb_version: str) -> dict[str, Any]:
    """运行单条样本的检索调试链路。"""
    return service.debug_retrieval(
        runtime.question,
        runtime.source_filter,
        runtime.session_id,
        kb_version=kb_version,
        scenario_id=runtime.scenario_id,
        tenant_id=runtime.tenant_id,
        dataset_id=runtime.dataset_id,
        visibility=runtime.visibility,
        user_role=runtime.user_role,
    )


def compare_case(service, item: dict[str, Any], index: int, args: argparse.Namespace, base_version: str, candidate_version: str) -> dict[str, Any]:
    """对比单条问题在两个知识库版本下的召回结果。"""
    runtime = EvalCaseRuntime.from_item(item, index, args, session_prefix="kb-compare")
    expected_sources = [str(value) for value in item.get("expected_source_contains", [])]
    row: dict[str, Any] = {
        "case_id": runtime.case_id,
        "question": runtime.question,
        "scenario_id": runtime.scenario_id,
        "source_filter": runtime.source_filter,
        "base_version": base_version,
        "candidate_version": candidate_version,
    }
    try:
        base_debug = debug_case(service, runtime, base_version)
        candidate_debug = debug_case(service, runtime, candidate_version)
        base_sources = list(base_debug.get("faq_sources") or []) + list(base_debug.get("doc_sources") or [])
        candidate_sources = list(candidate_debug.get("faq_sources") or []) + list(candidate_debug.get("doc_sources") or [])
        prefer_table = bool((candidate_debug.get("retrieval_plan") or {}).get("prefer_table"))
        base_ranked_sources = ranked_sources_for_compare(base_sources, prefer_table=prefer_table)
        candidate_ranked_sources = ranked_sources_for_compare(candidate_sources, prefer_table=prefer_table)
        base_rank = find_expected_source_rank(base_sources, expected_sources, prefer_table=prefer_table)
        candidate_rank = find_expected_source_rank(candidate_sources, expected_sources, prefer_table=prefer_table)
        row.update(
            {
                "base_top_source": source_signature(base_ranked_sources[0]) if base_ranked_sources else "",
                "candidate_top_source": source_signature(candidate_ranked_sources[0]) if candidate_ranked_sources else "",
                "top_source_changed": (source_signature(base_ranked_sources[0]) if base_ranked_sources else "") != (source_signature(candidate_ranked_sources[0]) if candidate_ranked_sources else ""),
                "base_expected_rank": base_rank,
                "candidate_expected_rank": candidate_rank,
                "base_recall_hit": base_rank is not None if expected_sources else None,
                "candidate_recall_hit": candidate_rank is not None if expected_sources else None,
                "mrr_delta": round((1 / candidate_rank if candidate_rank else 0.0) - (1 / base_rank if base_rank else 0.0), 4),
                "candidate_query_variants": (candidate_debug.get("retrieval_plan") or {}).get("query_variants", []),
                "error": "",
            }
        )
    except Exception as exc:
        row["error"] = str(exc)
    return row


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """执行版本对比并返回报告。"""
    base_version, candidate_version = resolve_versions(args)
    items = load_eval_items(args.dataset, args.limit)
    service = get_qa_service()
    rows = [
        compare_case(service, item, index, args, base_version, candidate_version)
        for index, item in enumerate(items, start=1)
    ]
    comparable_rows = [row for row in rows if row.get("candidate_recall_hit") is not None and not row.get("error")]
    regressions = [
        row
        for row in comparable_rows
        if row.get("base_recall_hit") and not row.get("candidate_recall_hit")
    ]
    improvements = [
        row
        for row in comparable_rows
        if not row.get("base_recall_hit") and row.get("candidate_recall_hit")
    ]
    return {
        "report_type": "kb_version_retrieval_compare",
        "created_at": utc_now(),
        "dataset": str(Path(args.dataset)),
        "scenario_id": args.scenario,
        "base_version": base_version,
        "candidate_version": candidate_version,
        "total": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "top_source_changed_count": sum(1 for row in rows if row.get("top_source_changed")),
        "base_recall_at_k": round(sum(1 for row in comparable_rows if row.get("base_recall_hit")) / max(len(comparable_rows), 1), 4),
        "candidate_recall_at_k": round(sum(1 for row in comparable_rows if row.get("candidate_recall_hit")) / max(len(comparable_rows), 1), 4),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "ok": not regressions and not any(row.get("error") for row in rows),
        "regressions": regressions,
        "improvements": improvements,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Compare retrieval between two knowledge base versions.")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "business_depth_regression.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--scenario", default=None, help="单场景对比时指定场景 ID。")
    parser.add_argument("--base-version", default="", help="旧版本；不传时使用该场景 previous_version。")
    parser.add_argument("--candidate-version", default="", help="候选版本；不传时使用该场景 active version。")
    parser.add_argument("--output", default=str(Path("reports") / "verification" / "kb_version_compare_latest.json"))
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--user-role", default=None)
    parser.add_argument("--kb-version", default=None, help="保留给 EvalCaseRuntime，版本对比时不使用。")
    return parser


def main() -> None:
    """执行知识库版本召回对比。"""
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    report = build_report(args)
    write_json_file(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

