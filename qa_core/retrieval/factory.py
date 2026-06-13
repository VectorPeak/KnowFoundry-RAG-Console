"""检索集合工厂与启动预热。

这个模块只负责“拿到检索对象”和“启动时预热检索依赖”。真正执行 Milvus 查询的逻辑在
`store.py`。这里通过 collection_name 缓存 MilvusHybridStore，避免每次请求重复创建
连接对象。
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Literal

from qa_core.config.logging_config import get_logger
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.retrieval.milvus_compat import milvus_endpoint_available
from qa_core.retrieval.models import get_embeddings, get_reranker
from qa_core.retrieval.store import MilvusHybridStore
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario

logger = get_logger(__name__)

@lru_cache(maxsize=32)
def get_hybrid_store(collection_name: str) -> MilvusHybridStore:
    """按 collection_name 返回已缓存的 Milvus 混合检索封装，多业务场景间隔离集合连接。

    LRU 缓存最多保留 32 个集合实例，足够覆盖当前冻结场景包，同时避免无界增长。

    把 lru_cache 放在 get_hybrid_store 而非顶层，是为了让 warmup_retrieval_stack 可以
    显式遍历每个 collection 触发缓存填充，而顶层调用方的无参 get_faq_store/get_doc_store
    仍享受缓存复用——预热与按需获取共享同一缓存空间。

    参数：
        collection_name: Milvus collection 名称。

    返回：
        已缓存或新创建的 MilvusHybridStore。
    """
    return MilvusHybridStore(collection_name)

def _active_scenario_collection(kind: Literal["faq", "doc"]) -> str:
    """返回当前默认场景的 FAQ 或文档 collection 名。

    参数：
        kind: faq 表示 FAQ 集合，doc 表示文档集合。

    返回：
        当前默认场景配置的 collection 名。
    """
    # 解析当前业务场景，获取其对应的集合配置
    scenario = resolve_scenario()
    return scenario.faq_collection if kind == "faq" else scenario.doc_collection

def get_faq_store(collection_name: str | None = None) -> MilvusHybridStore:
    """返回已缓存的 FAQ 混合集合封装。

    参数：
        collection_name: 可选集合名；为空时使用当前默认场景的 faq_collection。

    返回：
        FAQ MilvusHybridStore。
    """
    return get_hybrid_store(collection_name or _active_scenario_collection("faq"))


def get_doc_store(collection_name: str | None = None) -> MilvusHybridStore:
    """返回已缓存的文档混合集合封装。

    参数：
        collection_name: 可选集合名；为空时使用当前默认场景的 doc_collection。

    返回：
        文档 MilvusHybridStore。
    """
    return get_hybrid_store(collection_name or _active_scenario_collection("doc"))


def warmup_retrieval_stack() -> None:
    """服务启动时加载检索模型、全部冻结场景集合和当前 active 版本。任一预热失败直接阻断启动。

    执行流程：
      1. 加载 BGE embedding 模型，并对样例问题做一次向量化。
      2. 检查 Milvus 端点是否可达。
      3. 遍历所有场景，解析 active 知识库版本，并触发 FAQ/文档集合懒加载。
      4. 加载 CrossEncoder reranker，并对样例 query-passage 做一次预测。
      5. 记录预热耗时、场景数量、集合数量和 active 版本。

    异常：
        RuntimeError: Milvus 不可达或关键依赖不可用。
    """
    sample_query = "当前业务资料有哪些处理流程"
    started = time.perf_counter()
    # 获取场景注册表，遍历所有业务场景进行预热
    registry = get_scenario_registry()

    # 加载 BGE 向量模型并将样例查询向量化，验证 embedding 模型可用
    get_embeddings().embed_query(sample_query)

    # 检查 Milvus 服务端点是否可达，不可达则启动失败
    if not milvus_endpoint_available(timeout=3.0):
        raise RuntimeError("Milvus 服务不可达：请先启动 Milvus 2.5+ 服务。")

    warmed_collections: list[str] = []
    active_versions: dict[str, str] = {}
    # 遍历所有场景逐一预热，保证任意场景的第一个用户请求零冷启动延迟
    for scenario in registry.list_scenarios():
        # 解析每个场景的 active 知识库版本，若缺失则启动时直接暴露
        active_versions[scenario.scenario_id] = resolve_active_kb_version(None, scenario.scenario_id)
        for collection_name in (scenario.faq_collection, scenario.doc_collection):
            # 触发 Milvus 集合的懒加载连接，提前初始化 collection
            _ = get_hybrid_store(collection_name).store
            warmed_collections.append(collection_name)

    # 加载 CrossEncoder 重排模型并对样例对做预测，验证 reranker 可用
    get_reranker().predict([(sample_query, "业务资料包含处理流程、常见问题和操作规范。")])

    logger.info(
        "检索栈预热完成：耗时 %.2fs，场景数=%s，集合数=%s，active_versions=%s",
        time.perf_counter() - started,
        len(active_versions),
        len(warmed_collections),
        active_versions,
    )
