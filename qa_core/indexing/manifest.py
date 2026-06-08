"""增量文档索引用的文件指纹清单。记录哪些本地文件已切分写入 Milvus，用于增量更新时删除旧 chunk。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qa_core.common import utc_now
from qa_core.config.settings import get_settings
from qa_core.json_store import JsonFileStore
from qa_core.utils import stable_hash


@dataclass
class ManifestRecord:
    """一个已入库文件及其对应的 Milvus chunk id 列表。chunk_ids 是删除旧数据的关键字段。"""

    key: str
    scenario_id: str
    source: str
    path: str
    fingerprint: str
    chunk_ids: list[str]
    updated_at: str
    kb_version: str = ""
    embedding_model_version: str = ""
    chunk_schema_version: str = ""

class IndexManifest(JsonFileStore):
    """供本地入库脚本使用的 JSON 清单。记录 files: {<hash>: {source, path, fingerprint, chunk_ids, ...}}。"""
    def __init__(self, path: str | None = None) -> None:
        """打开配置的清单文件，并创建其父目录。"""
        super().__init__(path or get_settings().index_manifest_path)

    def empty_data(self) -> dict[str, Any]:
        """返回空增量清单。"""
        return {"files": {}}

    def normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """补齐历史增量清单缺失字段。"""
        data.setdefault("files", {})
        return data

    @staticmethod
    def key(
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> str:
        """根据来源和绝对文件路径生成稳定的清单键。source/kb_version/scenario_id 均参与 key，支持多场景多版本并存。"""
        return stable_hash(scenario_id or "", source, kb_version or "", str(Path(file_path).resolve()))

    def get(
        self,
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> ManifestRecord | None:
        """如果文件曾经入库，返回对应清单记录。"""
        key = self.key(source, file_path, kb_version, scenario_id)
        raw = self.data.get("files", {}).get(key)
        if not raw:
            return None
        raw.setdefault("scenario_id", scenario_id or "")
        return ManifestRecord(key=key, **raw)

    def is_unchanged(
        self,
        source: str,
        file_path: str | Path,
        fingerprint: str,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> bool:
        """检查当前文件指纹是否与清单一致。"""
        record = self.get(source, file_path, kb_version, scenario_id)
        return bool(record and record.fingerprint == fingerprint)

    def update(
        self,
        source: str,
        file_path: str | Path,
        fingerprint: str,
        chunk_ids: list[str],
        *,
        scenario_id: str = "",
        kb_version: str = "",
        embedding_model_version: str = "",
        chunk_schema_version: str = "",
    ) -> None:
        """记录一次成功入库及其生成的 chunk id。只有 Milvus 写入成功后才应调用。"""
        # 原因： 版本信息决定 chunk id 中的 hash 因子——embedding 模型或切分策略变更后，同一文件会生成不同的 chunk_id，manifest 据此判断旧 chunk 需要重建而非复用
        key = self.key(source, file_path, kb_version, scenario_id)
        self.data.setdefault("files", {})[key] = {
            "scenario_id": scenario_id,
            "source": source,
            "path": str(Path(file_path).resolve()),
            "fingerprint": fingerprint,
            "chunk_ids": chunk_ids,
            "updated_at": utc_now(),
            "kb_version": kb_version,
            "embedding_model_version": embedding_model_version,
            "chunk_schema_version": chunk_schema_version,
        }

    def remove(
        self,
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> ManifestRecord | None:
        """从清单中移除一个文件，并返回其旧记录。适合未来做删除本地文件后同步清理 Milvus chunk。"""
        key = self.key(source, file_path, kb_version, scenario_id)
        raw = self.data.setdefault("files", {}).pop(key, None)
        if not raw:
            return None
        raw.setdefault("scenario_id", scenario_id or "")
        return ManifestRecord(key=key, **raw)

    def iter_records(
        self,
        *,
        scenario_id: str | None = None,
        source: str | None = None,
        kb_version: str | None = None,
    ) -> list[ManifestRecord]:
        """按条件列出清单记录。支持按 scenario_id/source/kb_version 过滤。"""
        records: list[ManifestRecord] = []
        for key, raw in self.data.get("files", {}).items():
            payload = dict(raw)
            payload.setdefault("scenario_id", "")
            record = ManifestRecord(key=key, **payload)
            if scenario_id and record.scenario_id != scenario_id:
                continue
            if source and record.source != source:
                continue
            if kb_version and record.kb_version != kb_version:
                continue
            records.append(record)
        return records

    def remove_by_key(self, key: str) -> ManifestRecord | None:
        """按 manifest key 删除记录。用于文件路径已不存在但 key 仍存在时的清理。"""
        raw = self.data.setdefault("files", {}).pop(key, None)
        if not raw:
            return None
        raw.setdefault("scenario_id", "")
        return ManifestRecord(key=key, **raw)

