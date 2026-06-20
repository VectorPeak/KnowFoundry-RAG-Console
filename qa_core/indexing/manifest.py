"""增量文档索引用的 MySQL 清单。

清单记录本地文件指纹和对应 Milvus chunk id。增量入库时通过它判断文件是否需要重建，
文件删除清理时也通过它定位旧 chunk。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from qa_core.common import utc_now
from qa_core.config.settings import get_settings
from qa_core.memory.base import _MySqlStore, safe_sql_identifier
from qa_core.utils import stable_hash

INDEX_MANIFEST_TABLE = "kb_document_manifests"


def _json_dumps_list(value: list[str]) -> str:
    """序列化 chunk id 列表。"""
    return json.dumps(value or [], ensure_ascii=False)


def _json_loads_list(value: Any) -> list[str]:
    """反序列化 chunk id 列表。"""
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


@dataclass
class ManifestRecord:
    """一个已入库文件及其对应的 Milvus chunk id 列表。"""

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

    @classmethod
    def from_row(cls, row: Any) -> "ManifestRecord":
        """从 SQLAlchemy RowMapping 恢复 manifest 记录。"""
        payload = dict(row)
        payload["key"] = payload.pop("manifest_key")
        payload["chunk_ids"] = _json_loads_list(payload.pop("chunk_ids_json", "[]"))
        return cls(**payload)


class IndexManifest(_MySqlStore):
    """供入库脚本使用的 MySQL 清单。"""

    def __init__(self) -> None:
        """初始化 MySQL manifest。"""
        super().__init__()
        self.settings = get_settings()
        self.table_name = safe_sql_identifier(INDEX_MANIFEST_TABLE, label="Index manifest table")
        self._table_ready = False

    def ensure_table(self) -> None:
        """按需创建文档入库 manifest 表。"""
        if self._table_ready:
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            manifest_key VARCHAR(64) PRIMARY KEY,
            scenario_id VARCHAR(128) NOT NULL,
            source VARCHAR(128) NOT NULL,
            path TEXT NOT NULL,
            fingerprint VARCHAR(191) NOT NULL,
            chunk_ids_json LONGTEXT NOT NULL,
            kb_version VARCHAR(191) NOT NULL,
            embedding_model_version VARCHAR(191) NOT NULL,
            chunk_schema_version VARCHAR(191) NOT NULL,
            updated_at VARCHAR(40) NOT NULL,
            row_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_manifest_scenario_version (scenario_id, kb_version),
            INDEX idx_manifest_source (scenario_id, source),
            INDEX idx_manifest_updated_at (scenario_id, updated_at)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
        self._execute_ddl(ddl)
        self._table_ready = True

    def save(self) -> None:
        """兼容旧文件版接口；MySQL 每次 update 已经持久化。"""
        return None

    def reload(self) -> None:
        """兼容旧文件版接口；MySQL 查询始终读取最新数据。"""
        return None

    @staticmethod
    def key(
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> str:
        """根据来源和绝对文件路径生成稳定的清单键。"""
        return stable_hash(scenario_id or "", source, kb_version or "", str(Path(file_path).resolve()))

    def get(
        self,
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> ManifestRecord | None:
        """如果文件曾经入库，返回对应清单记录。"""
        self.ensure_table()
        key = self.key(source, file_path, kb_version, scenario_id)
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    f"""
                    SELECT
                        manifest_key, scenario_id, source, path, fingerprint,
                        chunk_ids_json, updated_at, kb_version,
                        embedding_model_version, chunk_schema_version
                    FROM {self.table_name}
                    WHERE manifest_key = :manifest_key
                    """
                ),
                {"manifest_key": key},
            ).mappings().fetchone()
        return ManifestRecord.from_row(row) if row else None

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
        """记录一次成功入库及其生成的 chunk id。"""
        self.ensure_table()
        key = self.key(source, file_path, kb_version, scenario_id)
        sql = f"""
        INSERT INTO {self.table_name}
            (
                manifest_key, scenario_id, source, path, fingerprint,
                chunk_ids_json, updated_at, kb_version,
                embedding_model_version, chunk_schema_version
            )
        VALUES
            (
                :manifest_key, :scenario_id, :source, :path, :fingerprint,
                :chunk_ids_json, :updated_at, :kb_version,
                :embedding_model_version, :chunk_schema_version
            )
        ON DUPLICATE KEY UPDATE
            scenario_id=VALUES(scenario_id),
            source=VALUES(source),
            path=VALUES(path),
            fingerprint=VALUES(fingerprint),
            chunk_ids_json=VALUES(chunk_ids_json),
            updated_at=VALUES(updated_at),
            kb_version=VALUES(kb_version),
            embedding_model_version=VALUES(embedding_model_version),
            chunk_schema_version=VALUES(chunk_schema_version),
            row_updated_at=CURRENT_TIMESTAMP
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(sql),
                {
                    "manifest_key": key,
                    "scenario_id": scenario_id,
                    "source": source,
                    "path": str(Path(file_path).resolve()),
                    "fingerprint": fingerprint,
                    "chunk_ids_json": _json_dumps_list(chunk_ids),
                    "updated_at": utc_now(),
                    "kb_version": kb_version,
                    "embedding_model_version": embedding_model_version,
                    "chunk_schema_version": chunk_schema_version,
                },
            )

    def remove(
        self,
        source: str,
        file_path: str | Path,
        kb_version: str | None = None,
        scenario_id: str | None = None,
    ) -> ManifestRecord | None:
        """从清单中移除一个文件，并返回其旧记录。"""
        key = self.key(source, file_path, kb_version, scenario_id)
        return self.remove_by_key(key)

    def iter_records(
        self,
        *,
        scenario_id: str | None = None,
        source: str | None = None,
        kb_version: str | None = None,
    ) -> list[ManifestRecord]:
        """按条件列出清单记录。支持按 scenario_id/source/kb_version 过滤。"""
        self.ensure_table()
        filters: list[str] = []
        params: dict[str, Any] = {}
        if scenario_id:
            filters.append("scenario_id = :scenario_id")
            params["scenario_id"] = scenario_id
        if source:
            filters.append("source = :source")
            params["source"] = source
        if kb_version:
            filters.append("kb_version = :kb_version")
            params["kb_version"] = kb_version
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = f"""
        SELECT
            manifest_key, scenario_id, source, path, fingerprint,
            chunk_ids_json, updated_at, kb_version,
            embedding_model_version, chunk_schema_version
        FROM {self.table_name}
        {where}
        ORDER BY updated_at DESC, manifest_key ASC
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [ManifestRecord.from_row(row) for row in rows]

    def remove_by_key(self, key: str) -> ManifestRecord | None:
        """按 manifest key 删除记录。"""
        self.ensure_table()
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    f"""
                    SELECT
                        manifest_key, scenario_id, source, path, fingerprint,
                        chunk_ids_json, updated_at, kb_version,
                        embedding_model_version, chunk_schema_version
                    FROM {self.table_name}
                    WHERE manifest_key = :manifest_key
                    """
                ),
                {"manifest_key": key},
            ).mappings().fetchone()
            if not row:
                return None
            conn.execute(
                text(f"DELETE FROM {self.table_name} WHERE manifest_key = :manifest_key"),
                {"manifest_key": key},
            )
        return ManifestRecord.from_row(row)
