"""检索链路模型加载器，集中管理 embedding 和 CrossEncoder。
进程级重资源使用 lru_cache 缓存，不缓存用户级结果。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# 防止 transformers 将 SentencePiece 转 Tiktoken 失败导致崩溃，需在导入前设置
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings


logger = get_logger(__name__)


def resolve_device() -> str:
    """选择本机可用推理设备（CUDA > CPU），CPU 是正常执行设备而非降级方案。

    BGE embedding 和 CrossEncoder 在典型批大小下 CPU 推理延迟完全可接受，
    所以 CPU 是一等执行设备；CUDA 只是可有可无的加速，不是必要条件。
    这样设计避免了对 GPU 环境的硬依赖，降低部署门槛。
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embeddings():
    """返回已缓存的 BGE 向量模型，用于 Milvus 稠密向量检索。

    模型加载涉及从磁盘读取权重文件到 GPU/CPU 内存（数百 MB），开销巨大。
    lru_cache 保证整个进程生命周期只加载一次，所有请求共享同一个模型实例。
    """
    # 加载应用全局设置（embedding 模型路径等配置）
    settings = get_settings()
    # 创建 BGE HuggingFaceEmbeddings 实例，用于生成稠密向量
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_path,
        # 自动选择 CUDA 或 CPU 作为推理设备
        model_kwargs={"device": resolve_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_reranker():
    """返回已缓存的 CrossEncoder 重排模型，用于 Milvus 召回后的二阶段精细排序。

    与 get_embeddings 同理：CrossEncoder 权重文件通常在 1GB 以上，加载是进程级
    重操作。lru_cache 确保只加载一次，所有检索请求复用同一个重排模型实例。
    """
    # 加载应用全局设置（reranker 模型路径等配置）
    settings = get_settings()
    model_path = Path(settings.reranker_model_path)
    if not model_path.exists():
        raise RuntimeError(f"Reranker model path does not exist: {model_path}")
    vocab_file = model_path / "sentencepiece.bpe.model"
    if not vocab_file.exists():
        raise RuntimeError(f"Reranker tokenizer vocab file does not exist: {vocab_file}")
    # 创建 CrossEncoder 重排模型实例，用于 Milvus 召回后的二阶段精细排序
    return CrossEncoder(
        str(model_path),
        # 自动选择 CUDA 或 CPU 作为推理设备
        device=resolve_device(),
        local_files_only=True,
        tokenizer_kwargs={
            "use_fast": False,
            # 新版 transformers 读 tokenizer_config.json 相对路径可能拼错，显式传完整路径
            "vocab_file": str(vocab_file),
        },
    )
