"""Milvus 连接、BM25 函数与客户端兼容补丁。
底层适配逻辑，使 store.py 可专注混合检索流程。"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pymilvus
from langchain_milvus import BM25BuiltInFunction
from pymilvus import MilvusClient, connections

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings


logger = get_logger(__name__)
_MILVUS_CLIENT_PATCHED = False


def collection_alias(collection_name: str) -> str:
    """为单个 collection 生成稳定的 PyMilvus ORM 连接别名。"""
    return f"{collection_name}_alias"


def langchain_connection_args(alias: str) -> dict[str, str]:
    """构建传给 langchain-milvus 的连接参数。

    业务检索统一走 langchain-milvus；这里集中保留底层 PyMilvus 连接参数，是为了
    避免 store.py 直接关心 ORM alias、database 这类驱动细节。
    """
    settings = get_settings()
    args = {"uri": settings.milvus_uri, "alias": alias}
    if settings.milvus_database:
        args["db_name"] = settings.milvus_database
    return args


def ensure_orm_alias_connection(alias: str, uri: str | None = None) -> None:
    """确保 PyMilvus ORM 连接注册表中存在指定 alias。

    langchain-milvus 是业务层 VectorStore 入口，但底层 hybrid search 仍可能按
    PyMilvus ORM alias 查找连接。把 alias 注册逻辑集中在这里，可以让 store.py
    保持“只使用 LangChain Milvus 封装”的教学口径。
    """
    settings = get_settings()
    target_uri = uri or settings.milvus_uri
    if connections.has_connection(alias):
        return
    kwargs = {"alias": alias, "uri": target_uri}
    if settings.milvus_database:
        kwargs["db_name"] = settings.milvus_database
    connections.connect(**kwargs)


def ensure_milvus_database() -> None:
    """在服务端支持数据库时创建配置中的 Milvus database，失败直接阻断启动。"""
    settings = get_settings()
    # 创建临时 MilvusClient 用于检查数据库列表
    client = MilvusClient(uri=settings.milvus_uri)
    # 获取当前 Milvus 服务中已有的数据库列表
    databases = client.list_databases()
    if settings.milvus_database and settings.milvus_database not in databases:
        # 如果配置的数据库不存在则创建
        client.create_database(settings.milvus_database)


def bm25_function():
    """构建 Milvus 2.5+ 内置 BM25 稀疏向量函数，替换旧版本地 BM25 方案。

    analyzer_params={"type": "chinese"} 是中文场景的必选项，不是可选的性能调优：
    中文文本词之间没有空格分隔，必须经过分词器才能产生有意义的 token。如果使用默认
    的英文分词器（按空白符切分），BM25 稀疏检索对中文 query 几乎失效。
    """
    return BM25BuiltInFunction(
        input_field_names="text",
        output_field_names="sparse",
        analyzer_params={"type": "chinese"},
        enable_match=True,
    )


def milvus_endpoint_available(timeout: float = 1.5) -> bool:
    """快速判断 Milvus TCP 端口是否可达，用于启动前置校验。"""
    settings = get_settings()
    # 从 Milvus URI 中解析主机和端口
    parsed = urlparse(settings.milvus_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 19530
    try:
        # 尝试 TCP 连接 Milvus 服务端口
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def patch_milvus_client_connection() -> None:
    """修补 langchain-milvus 与 pymilvus ORM 的连接别名兼容性问题。

    langchain-milvus 的 MilvusHybridStore 内部创建 MilvusClient 实例时不会自动向
    pymilvus.connections 注册连接别名，但 store.py 中的混合检索查询（hybrid search）
    依赖 pymilvus ORM 连接别名来协调稠密/稀疏向量子查询。不修补会导致
    "connection not found" 异常。这个补丁让两个库共享同一个 TCP 连接。
    """
    global _MILVUS_CLIENT_PATCHED
    if _MILVUS_CLIENT_PATCHED:
        return
    try:
        original_init = pymilvus.MilvusClient.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            uri = kwargs.get("uri")
            if uri is None and args:
                uri = args[0]
            if uri and getattr(self, "_using", None):
                if not connections.has_connection(self._using):
                    # 让 LangChain Milvus 和 pymilvus ORM 共享连接别名；失败应在预热阶段暴露
                    ensure_orm_alias_connection(self._using, uri)

        if getattr(pymilvus.MilvusClient.__init__, "__name__", "") != "patched_init":
            pymilvus.MilvusClient.__init__ = patched_init
        _MILVUS_CLIENT_PATCHED = True
    except Exception as exc:
        logger.debug("MilvusClient alias patch not applied: %s", exc)

