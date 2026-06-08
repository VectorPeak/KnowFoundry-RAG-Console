"""入库文档元数据标准化。补充 source、scenario、kb_version、tenant、dataset 等元数据。"""

from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from qa_core.governance.data_scope import DataScope, resolve_data_scope
from qa_core.governance.kb_versions import version_metadata
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint

def normalize_documents(
    documents: list[Document],
    file_path: Path,
    source: str,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    data_scope: DataScope | None = None,
    allowed_roles: list[str] | None = None,
) -> list[Document]:
    """为文档补充项目标准元数据（source/scenario_id/数据域/文件信息/doc_id/版本信息）。"""
    doc_id = file_fingerprint(file_path)
    scenario = resolve_scenario(scenario_id)
    scope = data_scope or resolve_data_scope()
    version_meta = version_metadata(kb_version, scenario.scenario_id)
    normalized: list[Document] = []
    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata or {})
        metadata.update(
            {
                "source": source,
                "scenario_id": scenario.scenario_id,
                **scope.metadata(allowed_roles=allowed_roles),
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "doc_id": doc_id,
                "page_index": metadata.get("page", index),
                "content_type": metadata.get("content_type") or "text",
                **version_meta,
            }
        )
        normalized.append(Document(page_content=doc.page_content, metadata=metadata))
    return normalized

