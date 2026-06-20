"""入库质量门禁检查。

质量报告负责“看见问题”，质量门禁负责“阻止问题进入线上版本”。本脚本可以读取已经
生成的入库质量报告，也可以现场生成一份报告后立即判断是否允许继续激活知识库版本。

它检查的不是模型回答效果，而是入库前后最容易造成 RAG 失真的基础问题：
- 文件解析失败；
- 不支持或未纳入场景白名单的文件；
- 空文件、空 FAQ、重复 FAQ；
- 低质量 chunk；
- FAQ 标准答案和正文资料潜在冲突；
- 知识库版本号、embedding 版本、chunk schema 版本是否记录完整。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import print_json, read_json_file, write_optional_json
from scripts.gate_utils import add_max_failure, add_required_failure, to_count
from qa_core.quality.ingestion import build_ingestion_quality_report, save_ingestion_quality_report


@dataclass(frozen=True)
class IngestionQualityThresholds:
    """入库质量门禁阈值。

    默认值采用严格策略：解析失败、低质量 chunk、FAQ 冲突等问题都不允许进入 active
    版本。真实企业项目里可以按资料治理阶段临时放宽某个阈值，但放宽应通过命令行显式
    写出来，不能在代码里悄悄吞掉。
    """

    max_failed_files: int = 0
    max_unsupported_files: int = 0
    max_empty_files: int = 0
    max_low_quality_issues: int = 0
    max_duplicate_chunks: int = 0
    max_empty_faq_questions: int = 0
    max_empty_faq_answers: int = 0
    max_duplicate_faq_questions: int = 0
    max_invalid_faq_sources: int = 0
    max_faq_document_conflicts: int = 0
    require_faq_file: bool = True
    require_kb_version: bool = True
    require_model_versions: bool = True


def load_quality_report(path: str | Path) -> dict[str, Any]:
    """读取入库质量报告 JSON。

    使用独立函数是为了让 `rebuild_kb_version.py` 和单元测试可以复用同一套加载逻辑。
    """
    return read_json_file(path)


def summarize_report_counts(report: dict[str, Any]) -> dict[str, int]:
    """抽取门禁需要比较的核心数量。

    这里不把报告原文全部带入门禁摘要，是为了输出更稳定，也避免终端里刷出大量 chunk
    预览。要看详情时仍然打开原始 report JSON。
    """
    faq_quality = report.get("faq_quality") or {}
    chunk_quality = report.get("chunk_quality") or {}
    conflicts = report.get("faq_document_conflicts") or {}
    return {
        "failed_files": to_count(report.get("failed_files_count")),
        "unsupported_files": to_count(report.get("unsupported_files_count")),
        "empty_files": to_count(report.get("empty_files_count")),
        "ocr_risk_files": to_count(report.get("ocr_risk_files_count")),
        "low_quality_issues": to_count(chunk_quality.get("low_quality_issue_count")),
        "duplicate_chunks": to_count(chunk_quality.get("duplicate_chunk_count")),
        "empty_faq_questions": to_count(faq_quality.get("empty_question_rows")),
        "empty_faq_answers": to_count(faq_quality.get("empty_answer_rows")),
        "duplicate_faq_questions": to_count(faq_quality.get("duplicate_questions")),
        "invalid_faq_sources": to_count(faq_quality.get("invalid_sources")),
        "faq_document_conflicts": to_count(conflicts.get("conflict_count")),
    }


def evaluate_report_against_gate(
    report: dict[str, Any],
    thresholds: IngestionQualityThresholds,
    *,
    report_path: str = "",
) -> dict[str, Any]:
    """用阈值判断入库质量报告是否通过。

    返回结构同时适合命令行打印和 `rebuild_kb_version.py` 调用：
    - `ok=True` 表示可以继续激活；
    - `failures` 给出明确失败原因；
    - `counts` 保留关键指标，便于状态页或 CI 展示。
    """
    counts = summarize_report_counts(report)
    faq_quality = report.get("faq_quality") or {}
    failures: list[dict[str, Any]] = []

    add_max_failure(
        failures,
        metric="failed_files",
        actual=counts["failed_files"],
        maximum=thresholds.max_failed_files,
        message="存在解析失败文件，必须修复后重新生成知识库版本。",
    )
    add_max_failure(
        failures,
        metric="unsupported_files",
        actual=counts["unsupported_files"],
        maximum=thresholds.max_unsupported_files,
        message="存在未支持或未纳入 source 白名单的文件，可能导致资料漏入库。",
    )
    add_max_failure(
        failures,
        metric="empty_files",
        actual=counts["empty_files"],
        maximum=thresholds.max_empty_files,
        message="存在 loader 未解析出内容的空文件。",
    )
    add_max_failure(
        failures,
        metric="ocr_risk_files",
        actual=counts["ocr_risk_files"],
        maximum=0,
        message="存在疑似 OCR/扫描件风险文件，默认不能直接进入 active 知识库。",
    )
    add_max_failure(
        failures,
        metric="low_quality_issues",
        actual=counts["low_quality_issues"],
        maximum=thresholds.max_low_quality_issues,
        message="存在空 chunk、过短 chunk、重复 chunk 或疑似 OCR 噪声。",
    )
    add_max_failure(
        failures,
        metric="duplicate_chunks",
        actual=counts["duplicate_chunks"],
        maximum=thresholds.max_duplicate_chunks,
        message="存在重复 chunk，可能造成重复召回和答案引用噪声。",
    )
    add_max_failure(
        failures,
        metric="empty_faq_questions",
        actual=counts["empty_faq_questions"],
        maximum=thresholds.max_empty_faq_questions,
        message="FAQ 存在空问题，无法稳定命中标准问答。",
    )
    add_max_failure(
        failures,
        metric="empty_faq_answers",
        actual=counts["empty_faq_answers"],
        maximum=thresholds.max_empty_faq_answers,
        message="FAQ 存在空答案，高置信命中后无法直出可靠结果。",
    )
    add_max_failure(
        failures,
        metric="duplicate_faq_questions",
        actual=counts["duplicate_faq_questions"],
        maximum=thresholds.max_duplicate_faq_questions,
        message="FAQ 存在重复标准问题，可能造成口径冲突。",
    )
    add_max_failure(
        failures,
        metric="invalid_faq_sources",
        actual=counts["invalid_faq_sources"],
        maximum=thresholds.max_invalid_faq_sources,
        message="FAQ source 不在当前场景白名单中，检索过滤会不稳定。",
    )
    add_max_failure(
        failures,
        metric="faq_document_conflicts",
        actual=counts["faq_document_conflicts"],
        maximum=thresholds.max_faq_document_conflicts,
        message="FAQ 标准答案和正文资料存在潜在冲突或缺少正文依据。",
    )

    add_required_failure(
        failures,
        metric="faq_file",
        actual=faq_quality.get("exists"),
        enabled=thresholds.require_faq_file,
        message="FAQ CSV 不存在，FAQ 直出能力不可验收。",
    )
    add_required_failure(
        failures,
        metric="kb_version",
        actual=report.get("kb_version"),
        enabled=thresholds.require_kb_version,
        message="报告缺少 kb_version，无法追溯这次入库版本。",
    )
    add_required_failure(
        failures,
        metric="embedding_model_version",
        actual=report.get("embedding_model_version"),
        enabled=thresholds.require_model_versions,
        message="报告缺少 embedding_model_version，无法判断向量化版本。",
    )
    add_required_failure(
        failures,
        metric="chunk_schema_version",
        actual=report.get("chunk_schema_version"),
        enabled=thresholds.require_model_versions,
        message="报告缺少 chunk_schema_version，无法判断切分方案版本。",
    )

    return {
        "ok": not failures,
        "report_type": "ingestion_quality_gate",
        "report_path": report_path,
        "scenario_id": report.get("scenario_id"),
        "kb_version": report.get("kb_version"),
        "counts": counts,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def thresholds_from_args(args: argparse.Namespace) -> IngestionQualityThresholds:
    """把命令行参数转换成门禁阈值对象。"""
    return IngestionQualityThresholds(
        max_failed_files=args.max_failed_files,
        max_unsupported_files=args.max_unsupported_files,
        max_empty_files=args.max_empty_files,
        max_low_quality_issues=args.max_low_quality_issues,
        max_duplicate_chunks=args.max_duplicate_chunks,
        max_empty_faq_questions=args.max_empty_faq_questions,
        max_empty_faq_answers=args.max_empty_faq_answers,
        max_duplicate_faq_questions=args.max_duplicate_faq_questions,
        max_invalid_faq_sources=args.max_invalid_faq_sources,
        max_faq_document_conflicts=args.max_faq_document_conflicts,
        require_faq_file=not args.allow_missing_faq,
        require_kb_version=not args.allow_missing_kb_version,
        require_model_versions=not args.allow_missing_model_versions,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Check ingestion quality report against strict gate thresholds.")
    parser.add_argument("--report", default="", help="已有入库质量报告路径。未提供时会现场生成报告。")
    parser.add_argument("--scenario", default=None, help="业务场景 ID，默认使用 ACTIVE_SCENARIO_ID。")
    parser.add_argument("--data-dir", default=None, help="文档根目录，默认使用场景配置 data_root。")
    parser.add_argument("--faq-csv", default=None, help="FAQ CSV 路径，默认使用场景配置 faq_csv_path。")
    parser.add_argument("--kb-version", default=None, help="报告关联的知识库版本，默认使用当前 active 版本。")
    parser.add_argument("--tenant-id", default=None, help="报告中记录的租户 ID。")
    parser.add_argument("--dataset-id", default=None, help="报告中记录的数据集 ID。")
    parser.add_argument("--visibility", default=None, help="报告中记录的可见级别。")
    parser.add_argument("--allowed-role", action="append", default=None, help="允许检索该资料的角色，可重复。")
    parser.add_argument("--output", default="", help="现场生成报告时的输出路径。")
    parser.add_argument("--gate-output", default="", help="门禁判定摘要输出路径。")
    parser.add_argument("--max-failed-files", type=int, default=0)
    parser.add_argument("--max-unsupported-files", type=int, default=0)
    parser.add_argument("--max-empty-files", type=int, default=0)
    parser.add_argument("--max-low-quality-issues", type=int, default=0)
    parser.add_argument("--max-duplicate-chunks", type=int, default=0)
    parser.add_argument("--max-empty-faq-questions", type=int, default=0)
    parser.add_argument("--max-empty-faq-answers", type=int, default=0)
    parser.add_argument("--max-duplicate-faq-questions", type=int, default=0)
    parser.add_argument("--max-invalid-faq-sources", type=int, default=0)
    parser.add_argument("--max-faq-document-conflicts", type=int, default=0)
    parser.add_argument("--allow-missing-faq", action="store_true", help="允许没有 FAQ CSV。")
    parser.add_argument("--allow-missing-kb-version", action="store_true", help="允许报告缺少 kb_version。")
    parser.add_argument("--allow-missing-model-versions", action="store_true", help="允许报告缺少模型和切分版本。")
    return parser


def main() -> None:
    """执行入库质量门禁并按结果设置退出码。"""
    parser = build_parser()
    args = parser.parse_args()
    if args.report:
        report_path = args.report
        report = load_quality_report(report_path)
    else:
        report = build_ingestion_quality_report(
            scenario_id=args.scenario,
            data_dir=args.data_dir,
            faq_csv=args.faq_csv,
            kb_version=args.kb_version,
            tenant_id=args.tenant_id,
            dataset_id=args.dataset_id,
            visibility=args.visibility,
            allowed_roles=args.allowed_role,
        )
        report_path = save_ingestion_quality_report(report, args.output or None)

    result = evaluate_report_against_gate(report, thresholds_from_args(args), report_path=report_path)
    write_optional_json(args.gate_output, result)
    print_json(result)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
