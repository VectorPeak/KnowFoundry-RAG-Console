"""Quality-gate tests kept after the LangSmith migration.

Generic trace storage, bad-case review queues, and evaluation trend dashboards
now live in LangSmith, so local tests focus on project-specific RAG quality
rules that remain in this repository.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.observability.langsmith_adapter import langsmith_enabled, langsmith_status
from qa_core.quality.conflicts import detect_faq_document_conflicts
from qa_core.quality.ingestion import build_ingestion_quality_report
from scripts.check_ingestion_quality_gate import IngestionQualityThresholds
from scripts.check_ingestion_quality_gate import evaluate_report_against_gate as evaluate_ingestion_gate


def _clean_ingestion_report() -> dict:
    return {
        "scenario_id": "enterprise_knowledge",
        "files_total": 1,
        "loaded_files_count": 1,
        "unsupported_files_count": 0,
        "failed_files_count": 0,
        "empty_files_count": 0,
        "low_quality_issues_count": 0,
        "duplicate_chunks_count": 0,
        "faq_quality": {
            "exists": True,
            "empty_question_rows": 0,
            "empty_answer_rows": 0,
            "duplicate_questions": 0,
            "invalid_sources": 0,
        },
        "chunk_quality": {
            "low_quality_issue_count": 0,
            "duplicate_chunk_count": 0,
        },
        "faq_document_conflicts": {"conflict_count": 0},
        "metadata_quality": {
            "missing_kb_version_count": 0,
            "missing_embedding_model_version_count": 0,
            "missing_reranker_model_version_count": 0,
            "missing_chunk_schema_version_count": 0,
            "missing_data_scope_count": 0,
        },
        "table_files_count": 0,
        "ocr_risk_files_count": 0,
        "kb_version": "kb_test",
        "embedding_model_version": "bge-m3-local-v1",
        "chunk_schema_version": "parent_child_v1",
    }


class QualityGateTests(unittest.TestCase):
    """Project-specific quality rules that remain local."""

    def test_ingestion_gate_rejects_faq_document_conflicts(self) -> None:
        report = _clean_ingestion_report()
        report["faq_document_conflicts"] = {"conflict_count": 1}
        result = evaluate_ingestion_gate(report, IngestionQualityThresholds())
        self.assertFalse(result["ok"])
        self.assertEqual(result["failures"][0]["metric"], "faq_document_conflicts")

    def test_ingestion_gate_passes_clean_report(self) -> None:
        result = evaluate_ingestion_gate(_clean_ingestion_report(), IngestionQualityThresholds())
        self.assertTrue(result["ok"])

    def test_faq_document_conflict_uses_chinese_search_segmentation(self) -> None:
        docs = [
            Document(
                page_content="管理员忘记密码时，可以在登录页选择忘记密码，并通过绑定邮箱重置。",
                metadata={"source": "account", "file_name": "password.md"},
            ),
            Document(
                page_content="成员离职后应先禁用账号，再回收角色权限、应用授权、API Token 和数据导出权限。",
                metadata={"source": "account", "file_name": "member_offboarding.md"},
            ),
        ]
        report = detect_faq_document_conflicts("scenarios/saas_support/faq.csv", docs)
        missing_account = [
            item
            for item in report["items"]
            if item["source"] == "account" and item["issue"] == "no_related_document"
        ]
        self.assertEqual(missing_account, [])

    def test_csv_table_loader_keeps_row_and_header_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "materials.csv"
            path.write_text("材料,状态,金额\n验收记录,缺失,50000\n付款申请,待复核,120000\n", encoding="utf-8")
            docs = load_file(path)
        self.assertEqual(len(docs), 2)
        self.assertIn("表头：材料 / 状态 / 金额", docs[0].page_content)
        self.assertIn("- 状态：缺失", docs[0].page_content)
        self.assertEqual(docs[0].metadata["content_type"], "table_row")
        self.assertEqual(docs[0].metadata["row_number"], 1)

    def test_table_chunk_is_not_split_like_normal_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "materials.csv"
            path.write_text("材料,状态,金额\n验收记录,缺失,50000\n", encoding="utf-8")
            normalized = normalize_documents(load_file(path), path, "quality", "kb_test", "engineering_project_qa")
            chunks, ids = split_documents(normalized)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(ids), 1)
        self.assertEqual(chunks[0].metadata["content_type"], "table_row")
        self.assertIn("验收记录", chunks[0].metadata["parent_content"])

    def test_ingestion_quality_report_marks_table_and_ocr_risk_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            quality_dir = data_dir / "quality_data"
            quality_dir.mkdir(parents=True)
            (quality_dir / "materials.csv").write_text("材料,状态\n隐蔽验收记录,缺失\n", encoding="utf-8")
            (quality_dir / "scan_noise.txt").write_text("扫描件 OCR 识别存在 O 和 0 混淆、断行和错字。", encoding="utf-8")
            faq_path = root / "faq.csv"
            faq_path.write_text("question,answer,source\n隐蔽工程验收需要哪些资料,需要隐蔽验收记录。,quality\n", encoding="utf-8")
            report = build_ingestion_quality_report(
                scenario_id="engineering_project_qa",
                data_dir=str(data_dir),
                faq_csv=str(faq_path),
                kb_version="kb_test",
            )
        self.assertEqual(report["table_files_count"], 1)
        self.assertEqual(report["ocr_risk_files_count"], 1)
        result = evaluate_ingestion_gate(report, IngestionQualityThresholds(max_low_quality_issues=10))
        self.assertFalse(result["ok"])
        self.assertIn("ocr_risk_files", {item["metric"] for item in result["failures"]})

    def test_langsmith_status_is_available_without_api_key(self) -> None:
        self.assertFalse(langsmith_enabled())
        status = langsmith_status()
        self.assertEqual(status["provider"], "langsmith")
        self.assertIn("project", status)


if __name__ == "__main__":
    unittest.main()
