"""一次性构建完整知识库版本。

该脚本用于“多版本知识库”的标准全量构建：先创建一个统一 kb_version，然后把 FAQ
和所有业务文档都写入这个版本，最后按需激活。相比分别运行 FAQ/文档入库脚本，它能
避免 FAQ 和文档进入两个不同版本。

推荐用法：
`python scripts/rebuild_kb_version.py --new-version --force --activate --description "2026-05-06 全量重建"`

执行完成后，在线检索会通过 `kb_version == active_version` 只查该版本；旧版本仍保留，
可通过管理脚本或 API 回滚。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.indexing.faq_ingestion import ingest_faq_csv
from qa_core.indexing.service import ingest_directory
from qa_core.quality.ingestion import build_ingestion_quality_report, save_ingestion_quality_report
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.scenarios.registry import resolve_scenario
from scripts.check_ingestion_quality_gate import IngestionQualityThresholds, evaluate_report_against_gate


def main() -> None:
    """创建/复用一个 kb_version，并把 FAQ 和文档写入同一版本。"""
    parser = argparse.ArgumentParser(description="Rebuild a complete multi-scenario RAG knowledge base version.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root data directory. Defaults to the selected scenario data_root.",
    )
    parser.add_argument(
        "--faq-csv",
        default=None,
        help="FAQ CSV path. Defaults to the selected scenario faq_csv_path.",
    )
    parser.add_argument("--scenario", default=None, help="Business scenario id. Defaults to ACTIVE_SCENARIO_ID.")
    parser.add_argument("--kb-version", default=None, help="Explicit knowledge base version id.")
    parser.add_argument("--new-version", action="store_true", help="Create a new staged version before ingest.")
    parser.add_argument("--force", action="store_true", help="Rebuild files even when fingerprint is unchanged.")
    parser.add_argument("--skip-faq", action="store_true", help="Skip FAQ ingest.")
    parser.add_argument("--skip-docs", action="store_true", help="Skip document ingest.")
    parser.add_argument("--skip-quality-report", action="store_true", help="Skip ingestion quality report generation.")
    parser.add_argument("--quality-gate", action="store_true", help="Run strict ingestion quality gate before activation.")
    parser.add_argument("--activate", action="store_true", help="Activate this version after successful ingest.")
    parser.add_argument("--description", default="", help="Human readable version description.")
    parser.add_argument("--tenant-id", default=None, help="Tenant/org id written into metadata. Defaults to default.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id written into metadata. Defaults to default.")
    parser.add_argument("--visibility", default=None, help="Visibility written into metadata: public/internal/private.")
    parser.add_argument("--allowed-role", action="append", default=None, help="Role allowed to retrieve this data. Can repeat.")
    parser.add_argument("--max-failed-files", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-unsupported-files", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-empty-files", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-low-quality-issues", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-duplicate-chunks", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-empty-faq-questions", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-empty-faq-answers", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-duplicate-faq-questions", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-invalid-faq-sources", type=int, default=0, help="Quality gate threshold.")
    parser.add_argument("--max-faq-document-conflicts", type=int, default=0, help="Quality gate threshold.")
    args = parser.parse_args()
    if args.quality_gate and args.skip_quality_report:
        parser.error("--quality-gate requires quality report generation; remove --skip-quality-report.")

    scenario = resolve_scenario(args.scenario)
    version_store = get_kb_version_store(scenario.scenario_id)
    if args.new_version or not args.kb_version:
        version = version_store.ensure_version(
            args.kb_version,
            create_new=args.new_version or not bool(version_store.active_version_candidate()),
            description=args.description,
            created_by="rebuild_kb_version",
        )
    else:
        version = version_store.ensure_version(
            args.kb_version,
            create_new=False,
            description=args.description,
            created_by="rebuild_kb_version",
        )
    kb_version = version.kb_version

    faq_count = 0
    doc_chunks = 0
    if not args.skip_faq:
        faq_count = ingest_faq_csv(
            args.faq_csv or scenario.faq_csv_path,
            scenario_id=scenario.scenario_id,
            tenant_id=args.tenant_id,
            dataset_id=args.dataset_id,
            visibility=args.visibility,
            allowed_roles=args.allowed_role,
            kb_version=kb_version,
            create_new_version=False,
            activate=False,
            description=args.description,
        )

    if not args.skip_docs:
        root = Path(args.data_dir or scenario.data_root)
        for source in scenario.valid_sources:
            source_dir = root / f"{source}_data"
            if source_dir.exists():
                doc_chunks += ingest_directory(
                    str(source_dir),
                    source=source,
                    scenario_id=scenario.scenario_id,
                    tenant_id=args.tenant_id,
                    dataset_id=args.dataset_id,
                    visibility=args.visibility,
                    allowed_roles=args.allowed_role,
                    force=args.force,
                    kb_version=kb_version,
                    create_new_version=False,
                    activate=False,
                    description=args.description,
                )

    report_path = ""
    activated = False
    if not args.skip_quality_report:
        # 质量报告复用主链路的 LangChain loader 和 splitter，但不写 Milvus。它记录本次版本
        # 对应的文件解析、FAQ 质量、低质量 chunk 和模型/切分版本，便于后续面试讲解、
        # bad case 排查和版本回滚对比。
        report = build_ingestion_quality_report(
            scenario_id=scenario.scenario_id,
            data_dir=args.data_dir or scenario.data_root,
            faq_csv=args.faq_csv or scenario.faq_csv_path,
            kb_version=kb_version,
            tenant_id=args.tenant_id,
            dataset_id=args.dataset_id,
            visibility=args.visibility,
            allowed_roles=args.allowed_role,
        )
        report["actual_ingest"] = {
            "faq_records_written": faq_count,
            "doc_chunks_written": doc_chunks,
            "activated": False,
        }

        if args.quality_gate:
            # 版本激活必须放在质量门禁之后。这样即使 Milvus 已经写入了 staged 版本，
            # 只要报告发现解析失败、FAQ 冲突或低质量 chunk，线上 active 版本也不会切换。
            thresholds = IngestionQualityThresholds(
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
            )
            gate_result = evaluate_report_against_gate(report, thresholds)
            report["quality_gate"] = gate_result
            if not gate_result["ok"]:
                report_path = save_ingestion_quality_report(report)
                print(
                    "Ingestion quality gate failed; knowledge base version was not activated: "
                    f"{kb_version}, quality_report={report_path}"
                )
                sys.exit(1)

        if args.activate:
            version_store.activate_version(kb_version)
            activated = True
            report["actual_ingest"]["activated"] = True
        report_path = save_ingestion_quality_report(report)
    elif args.activate:
        version_store.activate_version(kb_version)
        activated = True

    print(
        "Rebuilt knowledge base version: "
        f"{kb_version}, faq_records={faq_count}, doc_chunks={doc_chunks}, "
        f"activated={activated}, quality_report={report_path or 'skipped'}"
    )


if __name__ == "__main__":
    main()



