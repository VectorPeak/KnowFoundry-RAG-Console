"""清理已删除本地文件对应的 Milvus 文档 chunk。

增量入库只能处理“文件新增/修改”，不能自动知道某个本地文件已经被删除。该脚本读取
场景 manifest，找出本地路径不存在的记录，并在显式 `--apply` 时删除对应 Milvus chunk。

默认 dry-run，只输出将要清理的内容，不做删除。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.indexing.cleanup import cleanup_missing_document_chunks, write_cleanup_report
from qa_core.scenarios.registry import get_scenario_registry


def cleanup_targets(args: argparse.Namespace) -> list[str | None]:
    """解析本次要检查的场景列表。"""
    if args.all_scenarios:
        return [scenario.scenario_id for scenario in get_scenario_registry().list_scenarios()]
    return [args.scenario]


def main() -> None:
    """解析命令行参数并执行缺失文件清理。"""
    parser = argparse.ArgumentParser(description="Cleanup Milvus document chunks for local files that no longer exist.")
    parser.add_argument("--scenario", default=None, help="业务场景 ID，默认使用 ACTIVE_SCENARIO_ID。")
    parser.add_argument("--source", default=None, help="可选业务分类，只清理该 source。")
    parser.add_argument("--kb-version", default=None, help="可选知识库版本，默认清理当前 active 版本。")
    parser.add_argument("--apply", action="store_true", help="真正删除 Milvus chunk 并更新 manifest；不传则只预览。")
    parser.add_argument("--all-scenarios", action="store_true", help="检查全部冻结场景。")
    parser.add_argument("--output", default="", help="清理差异报告输出路径，默认写入 reports/ingestion。")
    args = parser.parse_args()

    results = [
        cleanup_missing_document_chunks(
            scenario_id=scenario_id,
            source=args.source,
            kb_version=args.kb_version,
            dry_run=not args.apply,
        )
        for scenario_id in cleanup_targets(args)
    ]
    result = {
        "mode": "all_scenarios" if args.all_scenarios else "single_scenario",
        "dry_run": not args.apply,
        "scenario_count": len(results),
        "records_checked": sum(item["records_checked"] for item in results),
        "missing_file_count": sum(item["missing_file_count"] for item in results),
        "affected_chunk_count": sum(item["affected_chunk_count"] for item in results),
        "deleted_chunk_count": sum(item["deleted_chunk_count"] for item in results),
        "failed_count": sum(len(item["failed_records"]) for item in results),
        "results": results,
    }
    report_path = write_cleanup_report(result, args.output or None)
    result["report_path"] = str(report_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
