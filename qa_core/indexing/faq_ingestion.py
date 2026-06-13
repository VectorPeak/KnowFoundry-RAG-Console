"""FAQ CSV 入库链路。FAQ 的 page_content 是标准问题，标准答案放在 metadata.answer。"""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.quality.faq import _resolve_csv_source
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store, version_metadata
from qa_core.retrieval.factory import get_faq_store
from qa_core.scenarios.registry import ScenarioDefinition, resolve_scenario
from qa_core.utils import stable_hash
logger = get_logger(__name__)

def faq_documents_from_csv(
    csv_path: str,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
) -> tuple[list[Document], list[str]]:
    """把 FAQ CSV 转换为可写入 Milvus 的问题文档。page_content=标准问题，metadata.answer=标准答案。"""
    scenario = resolve_scenario(scenario_id)
    data_scope = resolve_data_scope(tenant_id=tenant_id, dataset_id=dataset_id, visibility=visibility, user_roles=allowed_roles)
    version_meta = version_metadata(kb_version, scenario.scenario_id)
    # 原因： pandas 自动处理 BOM/编码推断/空值填充，而 csv.DictReader 只做逐行原始解析，遇到同名列合并或编码抖动需要额外编排才能达到同等健壮性
    data = pd.read_csv(csv_path, encoding="utf-8")
    docs: list[Document] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for _, row in data.iterrows():
        # 原因： 中文客户 CSV 可能用中文列名（问题/答案）也可能用英文列名（question/answer），同时兼容两种 header 减少运维沟通成本
        question = str(row.get("问题") or row.get("question") or "").strip()
        answer = str(row.get("答案") or row.get("answer") or "").strip()
        subject = _resolve_csv_source(dict(row))
        if not question or not answer:
            continue

        source = normalize_faq_source(subject, scenario=scenario, question=question)
        faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question)
        if faq_id in seen_ids:
            # 同一标准问题但答案不同，使用答案参与 hash，避免 id 冲突。
            faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question, answer)
        if faq_id in seen_ids:
            continue
        seen_ids.add(faq_id)
        docs.append(
            Document(
                page_content=question,
                metadata={
                    "faq_id": faq_id,
                    "scenario_id": scenario.scenario_id,
                    **data_scope.metadata(allowed_roles=allowed_roles),
                    "standard_question": question,
                    "answer": answer,
                    "source": source,
                    "subject_name": subject,
                    "status": "published",
                    **version_meta,
                },
            )
        )
        ids.append(faq_id)
    return docs, ids


def normalize_faq_source(
    subject: str,
    *,
    scenario: ScenarioDefinition,
    question: str = "",
) -> str:
    """按当前业务场景把 FAQ 分类归一化为 Milvus metadata.source。

    当前项目是多业务场景知识问答平台。FAQ 的 source 必须来自场景包 `scenario.toml` 中的
    `valid_sources`，或者由同一场景的 `source_patterns` 从分类/问题文本中推断出来。

    使用场景：
      - CSV 里已经写了标准 source，例如 `finance`，直接通过白名单校验。
      - CSV 里写的是中文分类，例如"财务报销"，通过 source_patterns 映射为 `finance`。
      - CSV 分类为空但问题本身包含强业务词，例如"VPN 账号权限怎么申请"，可映射为 `it`。

    这样做的原因：
      - source 会进入 Milvus 过滤表达式，必须严格受当前场景白名单约束。
      - 新增业务场景时只改场景配置，不改 Python 代码，降低学习成本和维护成本。
      - 主链路只认当前场景配置，避免业务分类规则散落在代码里。
    """
    normalized = subject.strip().lower()
    if normalized in scenario.valid_sources:
        return normalized

    for source, pattern in scenario.compiled_source_patterns().items():
        if pattern.search(subject) or pattern.search(question):
            return source

    raise ValueError(
        f"FAQ 分类无法映射到场景 {scenario.scenario_id} 的 valid_sources："
        f"subject={subject!r}, question={question!r}"
    )

def ingest_faq_csv(
    csv_path: str,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    kb_version: str | None = None,
    create_new_version: bool = False,
    activate: bool = False,
    description: str = "",
) -> int:
    """从 CSV 重新构建 FAQ 记录并写入 FAQ 混合集合。FAQ id 包含 kb_version，新版本不会覆盖旧版本。"""
    scenario = resolve_scenario(scenario_id)
    version_store = get_kb_version_store(scenario.scenario_id)
    version = version_store.ensure_version(
        kb_version,
        create_new=create_new_version,
        description=description,
        created_by="ingest_faq_csv",
    )
    active_kb_version = version.kb_version
    docs, ids = faq_documents_from_csv(
        csv_path,
        active_kb_version,
        scenario.scenario_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        allowed_roles=allowed_roles,
    )
    store = get_faq_store(scenario.faq_collection)
    store.delete_ids(ids)
    store.add_documents(docs, ids=ids)
    version_store.record_ingest_result(active_kb_version, content_type="faq", count=len(docs))
    if activate:
        version_store.activate_version(active_kb_version)
    logger.info("Ingested %s FAQ records from %s, kb_version: %s", len(docs), csv_path, active_kb_version)
    return len(docs)

