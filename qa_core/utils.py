"""入库和检索共用的小型确定性工具函数，用于生成 doc_id、chunk_id、faq_id 等 ID。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def stable_hash(*parts: object) -> str:
    """根据任意值创建稳定的 SHA-256 标识，用于 Milvus 主键和清单 key。"""
    raw = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """根据路径、修改时间和大小生成本地文件指纹，用于增量入库的变化判断。"""
    p = Path(path)
    stat = p.stat()
    return stable_hash(str(p.resolve()), stat.st_mtime_ns, stat.st_size)


def normalize_source_from_path(path: str | Path) -> str:
    """根据 `<source>_data` 目录名去掉 _data 后缀，得到来源名。"""
    name = os.path.basename(str(path)).replace("_data", "")
    return name or "default"
