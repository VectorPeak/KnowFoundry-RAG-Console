"""Milvus 连接、BM25 函数与数据库初始化工具。
底层适配逻辑，使 store.py 可专注混合检索流程。"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from langchain_milvus import BM25BuiltInFunction
from pymilvus import MilvusClient, connections

from qa_core.config.settings import get_settings


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
    保持“业务层只使用 LangChain Milvus 封装”的工程边界。
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

