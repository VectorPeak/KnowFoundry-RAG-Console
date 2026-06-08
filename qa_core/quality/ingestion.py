"""知识库入库质量报告编排层。扫描文件、调用 loader/normalizer/chunking，汇总 chunk 质量、FAQ 质量、冲突和异常文件。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from qa_core.common import list_json_reports, path_updated_at, read_json_dict, utc_file_stamp, utc_now, write_json
from qa_core.config.logging_config import get_logger
from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import get_document_loader_spec, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.indexing.table_documents import is_table_file, looks_like_ocr_risk, looks_like_table_text
from qa_core.quality.chunk import analyze_chunk_quality
from qa_core.quality.conflicts import detect_faq_document_conflicts
from qa_core.quality.faq import analyze_faq_csv
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import normalize_source_from_path


logger = get_logger(__name__)
INGESTION_REPORT_DIR = PROJECT_ROOT / "reports" / "ingestion"


def _iter_candidate_files(root: Path) -> list[Path]:
    """列出入库候选目录下的全部文件，按路径排序保证报告稳定。"""
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def _process_candidate_file(
    path: Path,
    source: str,
    scenario: Any,
    active_kb_version: str,
    scope: Any,
    allowed_roles: list[str] | None,
) -> dict[str, Any]:
    """解析单个候选文件并返回质量报告需要的中间结果。

    该函数只做“试解析 + 试切分”，不会写入 Milvus。返回值中包含：
    - ok：文件是否成功解析。
    - raw_docs：Loader 原始文档结果。
    - normalized_chunks：标准化并切分后的 chunk。
    - is_table / is_ocr_risk：是否疑似表格资料或 OCR 风险资料。
    - error：解析失败时的错误信息。
    """
    try:
        raw_docs = load_file(path)
        if not raw_docs:
            return {
                "ok": True,
                "raw_docs": [],
                "is_table": False,
                "is_ocr_risk": False,
                "normalized_chunks": [],
                "error": None,
                "path": str(path),
                "source": source,
            }
        combined_text = "\n".join(doc.page_content or "" for doc in raw_docs)
        is_table = is_table_file(path) or looks_like_table_text(path, combined_text)
        is_ocr_risk = looks_like_ocr_risk(path, combined_text)
        normalized = normalize_documents(
            raw_docs,
            path,
            source,
            active_kb_version,
            scenario.scenario_id,
            scope,
            allowed_roles,
        )
        chunks, _ = split_documents(normalized)
        return {
            "ok": True,
            "raw_docs": raw_docs,
            "is_table": is_table,
            "is_ocr_risk": is_ocr_risk,
            "normalized_chunks": chunks,
            "error": None,
            "path": str(path),
            "source": source,
            "combined_text": combined_text,
        }
    except Exception as exc:
        logger.warning("构建入库质量报告时文件解析失败：%s，错误：%s", path, exc)
        return {
            "ok": False,
            "raw_docs": None,
            "is_table": False,
            "is_ocr_risk": False,
            "normalized_chunks": [],
            "error": str(exc),
            "path": str(path),
            "source": source,
        }


def build_ingestion_quality_report(
    *,
    scenario_id: str | None = None,
    data_dir: str | None = None,
    faq_csv: str | None = None,
    kb_version: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
) -> dict[str, Any]:
    """构建一次完整的入库质量报告（只解析和切分，不写 Milvus）。"""
    # 原因： 在激活 kb_version 之前先运行质量检测，拦截切分异常/FAQ 冲突/低质量 chunk，防止有问题的数据进入线上检索链路
    scenario = resolve_scenario(scenario_id)
    scope = resolve_data_scope(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        user_roles=allowed_roles,
    )
    active_kb_version = kb_version or resolve_active_kb_version(None, scenario.scenario_id)
    settings = get_settings()
    root = Path(data_dir or scenario.data_root)
    faq_path = Path(faq_csv or scenario.faq_csv_path)
    started_at = utc_now()

    all_chunks: list[Document] = []
    source_counts: Counter[str] = Counter()
    files_loaded: list[dict[str, Any]] = []
    unsupported_files: list[dict[str, Any]] = []
    failed_files: list[dict[str, Any]] = []
    empty_files: list[dict[str, Any]] = []
    table_files: list[dict[str, Any]] = []
    ocr_risk_files: list[dict[str, Any]] = []
    candidate_files = _iter_candidate_files(root)

    for path in candidate_files:
        spec = get_document_loader_spec(path)
        source = normalize_source_from_path(path.parent)
        if spec is None:
            unsupported_files.append({"path": str(path), "suffix": path.suffix.lower(), "reason": "未注册 LangChain loader。"})
            continue
        if source not in scenario.valid_sources:
            unsupported_files.append({"path": str(path), "source": source, "reason": "文件目录无法映射到当前场景 source 白名单。"})
            continue
        result = _process_candidate_file(path, source, scenario, active_kb_version, scope, allowed_roles)
        if result["error"]:
            failed_files.append({"path": result["path"], "error": result["error"]})
            continue
        if not result["raw_docs"]:
            empty_files.append({"path": result["path"], "reason": "loader 未返回任何 Document。"})
            continue
        if result["is_table"]:
            table_files.append({
                "path": result["path"],
                "suffix": path.suffix.lower(),
                "source": result["source"],
                "raw_documents": len(result["raw_docs"]),
                "reason": "表格型资料已按表头、行号和单元格键值保留结构化语义。",
            })
        if result["is_ocr_risk"]:
            ocr_risk_files.append({
                "path": result["path"],
                "suffix": path.suffix.lower(),
                "source": result["source"],
                "reason": "疑似扫描件或 OCR 噪声，默认不执行复杂 OCR，应先人工复核或走独立 OCR 清洗流程。",
                "preview": result["combined_text"][:160],
            })
        all_chunks.extend(result["normalized_chunks"])
        source_counts[result["source"]] += len(result["normalized_chunks"])
        files_loaded.append({
            "path": result["path"],
            "source": result["source"],
            "raw_documents": len(result["raw_docs"]),
            "chunks": len(result["normalized_chunks"]),
        })

    chunk_issues, chunk_stats = analyze_chunk_quality(all_chunks)
    faq_quality = analyze_faq_csv(faq_path, scenario.valid_sources)
    faq_document_conflicts = detect_faq_document_conflicts(faq_path, all_chunks)
    warnings: list[str] = []
    if unsupported_files:
        warnings.append("存在未支持或未纳入白名单的文件，需要确认是否应扩展 loader 或调整目录。")
    if failed_files:
        warnings.append("存在解析失败文件，需要单独修复后重新入库。")
    if chunk_issues:
        warnings.append("存在低质量 chunk，需要检查切分策略、文档格式或 OCR 质量。")
    if faq_quality.get("duplicate_questions"):
        warnings.append("FAQ 存在重复标准问题，高置信直出可能产生口径冲突。")
    if faq_quality.get("invalid_sources"):
        warnings.append("FAQ 存在不在当前场景白名单内的 source。")
    if faq_document_conflicts.get("conflict_count"):
        warnings.append("FAQ 标准答案与正文资料存在潜在冲突，需要人工复核后再激活知识库版本。")
    if table_files:
        warnings.append("检测到表格型资料，已按行列语义入库或标记，适合清单、金额、状态和字段类问题检索。")
    if ocr_risk_files:
        warnings.append("检测到疑似 OCR/扫描件风险文件，默认不执行复杂 OCR，需要人工复核后再进入 active 知识库。")

    return {
        "report_type": "ingestion_quality",
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "kb_version": active_kb_version,
        "data_scope": scope.as_dict(),
        "embedding_model_version": settings.embedding_model_version,
        "reranker_model_version": settings.reranker_model_version,
        "chunk_schema_version": settings.chunk_schema_version,
        "data_root": str(root),
        "faq_csv_path": str(faq_path),
        "started_at": started_at,
        "finished_at": utc_now(),
        "files_scanned": len(candidate_files),
        "files_loaded_count": len(files_loaded),
        "unsupported_files_count": len(unsupported_files),
        "failed_files_count": len(failed_files),
        "empty_files_count": len(empty_files),
        "table_files_count": len(table_files),
        "ocr_risk_files_count": len(ocr_risk_files),
        "source_chunk_counts": dict(source_counts),
        "chunk_quality": chunk_stats,
        "low_quality_chunks": chunk_issues[:200],
        "faq_quality": faq_quality,
        "faq_document_conflicts": faq_document_conflicts,
        "files_loaded": files_loaded,
        "unsupported_files": unsupported_files,
        "failed_files": failed_files,
        "empty_files": empty_files,
        "table_files": table_files,
        "ocr_risk_files": ocr_risk_files,
        "warnings": warnings,
    }


def save_ingestion_quality_report(report: dict[str, Any], output: str | None = None) -> str:
    """保存入库质量报告并返回文件路径。"""
    scenario_id = str(report.get("scenario_id") or "default")
    kb_version = str(report.get("kb_version") or "preview").replace(":", "_")
    if output:
        path = Path(output)
    else:
        stamp = utc_file_stamp()
        path = INGESTION_REPORT_DIR / scenario_id / f"{stamp}_{kb_version}.json"
    return write_json(path, report)


def list_ingestion_reports(*, scenario_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """列出最近的入库质量报告。"""
    root = INGESTION_REPORT_DIR
    fetch_limit = 0 if scenario_id else limit
    entries = list_json_reports(root, "**/*.json", limit=fetch_limit)
    reports: list[dict[str, Any]] = []
    for entry in entries:
        payload = entry["payload"]
        if scenario_id and payload.get("scenario_id") != scenario_id:
            continue
        reports.append(
            {
                "path": entry["path"],
                "file_name": entry["file_name"],
                "scenario_id": payload.get("scenario_id"),
                "kb_version": payload.get("kb_version"),
                "updated_at": entry["updated_at"],
                "summary": {
                    "files_scanned": payload.get("files_scanned"),
                    "files_loaded_count": payload.get("files_loaded_count"),
                    "unsupported_files_count": payload.get("unsupported_files_count"),
                    "failed_files_count": payload.get("failed_files_count"),
                    "table_files_count": payload.get("table_files_count"),
                    "ocr_risk_files_count": payload.get("ocr_risk_files_count"),
                    "table_files": payload.get("table_files", [])[:5],
                    "ocr_risk_files": payload.get("ocr_risk_files", [])[:5],
                    "chunk_quality": payload.get("chunk_quality"),
                    "faq_quality": payload.get("faq_quality"),
                    "faq_document_conflicts": payload.get("faq_document_conflicts"),
                    "warnings": payload.get("warnings", []),
                },
            }
        )
        if len(reports) >= limit:
            break
    return reports


