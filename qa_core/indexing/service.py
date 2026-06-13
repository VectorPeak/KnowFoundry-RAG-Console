"""本地业务文档入库编排服务。

这个文件负责“离线入库链路”，不参与在线问答时的实时生成。它把一个业务场景目录下的
本地资料按固定流程写入 Milvus：解析场景配置 → 确认数据隔离范围 → 确认知识库版本 →
加载文件 → 标准化元数据 → 切分 chunk → 写入向量库 → 更新本地索引清单。

这里单独成一个 service 文件，是为了把入库流程和在线 QAService 分开：
- 在线问答只负责检索、重排、Prompt 和流式返回；
- 离线入库只负责资料治理、增量构建、版本记录和 Milvus 写入。
"""
from __future__ import annotations

import os
from pathlib import Path

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import get_document_loader_spec, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.indexing.manifest import IndexManifest
from qa_core.retrieval.factory import get_doc_store
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint, normalize_source_from_path


logger = get_logger(__name__)


def _ingest_single_file(
    path: Path,
    resolved_source: str,
    active_kb_version: str,
    scenario,
    data_scope,
    allowed_roles: list[str] | None,
    doc_store,
    manifest: IndexManifest,
    force: bool,
) -> tuple[int, bool]:
    """处理单个文件的增量入库。

    这个函数是 `ingest_directory()` 的最小执行单元。之所以拆出来，是因为“目录遍历”
    和“单文件是否需要重建”是两层不同逻辑：目录层只负责找文件，文件层负责判断是否跳过、
    是否删除旧 chunk、是否重新加载和写入。

    执行流程：
      1. 校验文件类型是否有 LangChain Loader 支持，不支持就直接报错。
      2. 根据文件路径、修改时间、大小生成 fingerprint，用于判断文件是否变化。
      3. 从 manifest 中查找同一场景、同一版本、同一 source、同一路径的旧入库记录。
      4. 如果文件指纹、embedding 版本、chunk schema 都没变，并且没有强制重建，则跳过。
      5. 如果文件已变化或 schema 升级，先删除旧 chunk，避免 Milvus 中同一文件新旧内容并存。
      6. 通过 LangChain Loader 读取文件，再补齐场景、版本、数据域等 metadata。
      7. 调用统一切分器生成 chunk 和稳定 chunk_id。
      8. 将 chunk 写入 Milvus，并把新 chunk_id 记录回 manifest。

    返回值：
      - 第一个值：本次实际写入的 chunk 数。
      - 第二个值：是否因为文件未变化而跳过。
    """
    if get_document_loader_spec(path) is None:
        raise ValueError(f"不支持的文档类型：{path}")
    fingerprint = file_fingerprint(path)
    existing = manifest.get(resolved_source, path, active_kb_version, scenario.scenario_id)
    # 文件指纹 + embedding/chunk schema 均未变 → 跳过入库，实现增量式索引更新
    if (
        not force
        and existing
        and existing.fingerprint == fingerprint
        and existing.embedding_model_version == get_settings().embedding_model_version
        and existing.chunk_schema_version == get_settings().chunk_schema_version
    ):
        return 0, True
    # 文件已变更或 schema 升级，清理旧 chunk 后重新入库，防止向量数据版本混乱
    if existing and existing.chunk_ids:
        doc_store.delete_ids(existing.chunk_ids)
    docs = normalize_documents(
        load_file(path),
        path,
        resolved_source,
        active_kb_version,
        scenario.scenario_id,
        data_scope,
        allowed_roles,
    )
    chunks, ids = split_documents(docs)
    if chunks:
        doc_store.add_documents(chunks, ids=ids)
        manifest.update(
            resolved_source,
            path,
            fingerprint,
            ids,
            scenario_id=scenario.scenario_id,
            kb_version=active_kb_version,
            embedding_model_version=get_settings().embedding_model_version,
            chunk_schema_version=get_settings().chunk_schema_version,
        )
        return len(chunks), False
    return 0, False


def ingest_directory(
    directory_path: str,
    source: str | None = None,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    force: bool = False,
    kb_version: str | None = None,
    create_new_version: bool = False,
    activate: bool = False,
    description: str = "",
) -> int:
    """把某个目录下的业务文档增量写入 Milvus。

    这是文档入库的主入口，通常由脚本调用，例如重建某个场景的知识库版本时会走这里。
    它不负责 FAQ CSV，FAQ 有单独的 `faq_ingestion.py`；这里专注处理普通业务文档、
    表格行、OCR 后的文本等“文档型资料”。

    主要职责：
      1. 解析当前业务场景，拿到 doc_collection、valid_sources、版本清单路径等配置。
      2. 构建 DataScope，把 tenant/dataset/visibility/roles 写入 metadata，支持隔离检索。
      3. 校验 source 必须属于当前场景的 valid_sources，防止跨场景数据写错集合。
      4. 确认或创建知识库版本，新旧版本可以并存，线上只检索 active 版本。
      5. 递归遍历目录，对每个文件调用 `_ingest_single_file()` 做增量判断和写入。
      6. 保存 manifest，让下次入库可以跳过未变化文件，并能删除旧 chunk。
      7. 记录本次入库统计；如果传入 activate，则把该版本切换为线上检索版本。

    参数说明：
      - directory_path：要入库的目录。
      - source：业务分类；不传时从目录名推断，例如 `finance_data` 推断为 `finance`。
      - scenario_id：目标业务场景，例如 `enterprise_knowledge`。
      - tenant_id/dataset_id/visibility/allowed_roles：数据隔离字段，会进入 Milvus metadata。
      - force：是否忽略 fingerprint，强制重建所有文件。
      - kb_version/create_new_version/activate：知识库多版本管理相关参数。

    返回：
      实际写入 Milvus 的 chunk 总数，不包含被增量跳过的文件。
    """
    scenario = resolve_scenario(scenario_id)
    data_scope = resolve_data_scope(tenant_id=tenant_id, dataset_id=dataset_id, visibility=visibility, user_roles=allowed_roles)
    root = Path(directory_path)
    # 未显式传 source 时，从当前目录名推断业务分类，例如 finance_data → finance。
    # 推断结果仍要经过场景 valid_sources 校验，防止目录名写错后把数据写进错误分类。
    resolved_source = source or normalize_source_from_path(root)
    if resolved_source not in scenario.valid_sources:
        raise ValueError(f"无效的业务分类：{resolved_source}，当前场景支持：{scenario.valid_sources}")
    version_store = get_kb_version_store(scenario.scenario_id)
    version = version_store.ensure_version(
        kb_version,
        create_new=create_new_version,
        description=description,
        created_by="ingest_directory",
    )
    active_kb_version = version.kb_version
    manifest = IndexManifest(path=scenario.index_manifest_path)
    doc_store = get_doc_store(scenario.doc_collection)
    total_chunks = 0
    skipped_files = 0
    for current_root, _, files in os.walk(root):
        for file_name in files:
            path = Path(current_root) / file_name
            chunks, skipped = _ingest_single_file(
                path, resolved_source, active_kb_version, scenario, data_scope,
                allowed_roles, doc_store, manifest, force,
            )
            if skipped:
                skipped_files += 1
            else:
                total_chunks += chunks
    # 将索引清单持久化到本地 JSON 文件
    manifest.save()
    # 在版本清单中记录本次入库统计
    version_store.record_ingest_result(active_kb_version, content_type="doc", count=total_chunks, source=resolved_source)
    if activate:
        # 将当前版本切换为在线检索的 active 版本
        version_store.activate_version(active_kb_version)  # 激活后新入库数据立即可被在线检索命中
    logger.info(
        "文档入库完成：写入 chunk=%s，目录=%s，跳过未变化文件=%s，kb_version=%s",
        total_chunks,
        directory_path,
        skipped_files,
        active_kb_version,
    )
    return total_chunks



