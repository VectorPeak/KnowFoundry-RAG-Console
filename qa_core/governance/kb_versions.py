"""知识库多版本管理。

MySQL 保存版本控制面数据，Milvus 保存可检索的 FAQ/文档 chunk。版本切换只更新
MySQL 中的 active 指针，不修改 Milvus 数据，因此仍然保持 O(1) 激活和快速回滚。

核心能力：
- 生成带时间戳和配置 hash 的版本号。
- 跟踪版本生命周期：STAGED -> ACTIVE -> ARCHIVED。
- 按“请求参数 > 环境变量 > MySQL active 指针”的优先级解析检索版本。
- 记录每个版本的 FAQ/文档入库统计。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from sqlalchemy import text

from qa_core.common import utc_file_stamp, utc_now
from qa_core.config.settings import get_settings
from qa_core.memory.base import _MySqlStore, safe_sql_identifier
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import stable_hash

KB_VERSIONS_TABLE = "kb_versions"
KB_ACTIVE_TABLE = "kb_active_versions"
KB_VERSION_STATUS_STAGED = "STAGED"
KB_VERSION_STATUS_ACTIVE = "ACTIVE"
KB_VERSION_STATUS_ARCHIVED = "ARCHIVED"


def _resolve_version_scenario(scenario_id: str | None = None):
    """解析知识库版本所属的场景配置，避免模块循环依赖。"""
    return resolve_scenario(scenario_id)


def _json_dumps(value: Any) -> str:
    """按项目统一方式序列化 JSON 字段。"""
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _json_loads_dict(value: Any) -> dict[str, Any]:
    """把数据库 JSON/TEXT 字段恢复为 dict，坏数据按空字典处理。"""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_loads_list(value: Any) -> list[str]:
    """把数据库 JSON/TEXT 字段恢复为字符串列表。"""
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


def generate_kb_version(prefix: str = "kb", scenario_id: str | None = None) -> str:
    """生成含时间戳和配置 hash 的知识库版本号。（★★ 理解）"""
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)
    stamp = utc_file_stamp()
    config_hash = stable_hash(
        scenario.scenario_id,
        settings.embedding_model_version,
        settings.reranker_model_version,
        settings.chunk_schema_version,
        scenario.doc_collection,
        scenario.faq_collection,
    )[:8]
    return f"{prefix}_{scenario.scenario_id}_{stamp}_{config_hash}"


@dataclass
class KnowledgeBaseVersion:
    """可检索知识库版本的元数据。"""

    kb_version: str
    scenario_id: str = ""
    status: str = KB_VERSION_STATUS_STAGED
    description: str = ""
    created_at: str = field(default_factory=utc_now)
    activated_at: str | None = None
    archived_at: str | None = None
    doc_collection: str = ""
    faq_collection: str = ""
    embedding_model_version: str = ""
    reranker_model_version: str = ""
    chunk_schema_version: str = ""
    created_by: str = "local"
    sources: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeBaseVersion":
        """从 dict 恢复版本对象，老记录缺失的字段自动使用默认值。"""
        fields = cls.__dataclass_fields__
        data = {name: payload.get(name) for name in fields if name in payload}
        version = cls(**data)
        if version.sources is None:
            version.sources = []
        if version.stats is None:
            version.stats = {}
        return version

    @classmethod
    def from_row(cls, row: Any) -> "KnowledgeBaseVersion":
        """从 SQLAlchemy RowMapping 恢复版本对象。"""
        payload = dict(row)
        payload["sources"] = _json_loads_list(payload.get("sources_json"))
        payload["stats"] = _json_loads_dict(payload.get("stats_json"))
        payload.pop("sources_json", None)
        payload.pop("stats_json", None)
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的版本信息。"""
        return asdict(self)


class KnowledgeBaseVersionStore(_MySqlStore):
    """知识库版本状态机的 MySQL 存储实现。"""

    def __init__(self, scenario_id: str | None = None) -> None:
        """绑定一个业务场景，按需创建 MySQL 表。"""
        super().__init__()
        self.settings = get_settings()
        self.scenario = _resolve_version_scenario(scenario_id)
        self.version_table = safe_sql_identifier(KB_VERSIONS_TABLE, label="KB versions table")
        self.active_table = safe_sql_identifier(KB_ACTIVE_TABLE, label="KB active table")
        self._tables_ready = False

    def reload(self) -> None:
        """保留旧文件版接口；MySQL 每次查询都读取最新数据，无需 reload。"""
        return None

    @property
    def data(self) -> dict[str, Any]:
        """兼容旧调用方的只读数据视图。新代码应使用显式方法。"""
        active, previous = self._active_pointer()
        return {
            "scenario_id": self.scenario.scenario_id,
            "active_version": active,
            "previous_version": previous,
            "versions": {item.kb_version: item.as_dict() for item in self.list_versions()},
        }

    def ensure_tables(self) -> None:
        """按需创建知识库版本表和 active 指针表。"""
        if self._tables_ready:
            return
        ddl_versions = f"""
        CREATE TABLE IF NOT EXISTS {self.version_table} (
            scenario_id VARCHAR(128) NOT NULL,
            kb_version VARCHAR(191) NOT NULL,
            status VARCHAR(32) NOT NULL,
            description TEXT NULL,
            created_at VARCHAR(40) NOT NULL,
            activated_at VARCHAR(40) NULL,
            archived_at VARCHAR(40) NULL,
            doc_collection VARCHAR(191) NOT NULL,
            faq_collection VARCHAR(191) NOT NULL,
            embedding_model_version VARCHAR(191) NOT NULL,
            reranker_model_version VARCHAR(191) NOT NULL,
            chunk_schema_version VARCHAR(191) NOT NULL,
            created_by VARCHAR(128) NOT NULL,
            sources_json LONGTEXT NULL,
            stats_json LONGTEXT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (scenario_id, kb_version),
            INDEX idx_kb_versions_status (scenario_id, status),
            INDEX idx_kb_versions_created_at (scenario_id, created_at)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
        ddl_active = f"""
        CREATE TABLE IF NOT EXISTS {self.active_table} (
            scenario_id VARCHAR(128) PRIMARY KEY,
            active_kb_version VARCHAR(191) NOT NULL DEFAULT '',
            previous_kb_version VARCHAR(191) NOT NULL DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """
        self._execute_ddl(ddl_versions)
        self._execute_ddl(ddl_active)
        self._tables_ready = True
        self._reconcile_active_status()

    def list_versions(self) -> list[KnowledgeBaseVersion]:
        """按创建时间倒序返回当前场景全部版本。"""
        self.ensure_tables()
        sql = f"""
        SELECT
            scenario_id, kb_version, status, description, created_at,
            activated_at, archived_at, doc_collection, faq_collection,
            embedding_model_version, reranker_model_version, chunk_schema_version,
            created_by, sources_json, stats_json
        FROM {self.version_table}
        WHERE scenario_id = :scenario_id
        ORDER BY created_at DESC, kb_version DESC
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), {"scenario_id": self.scenario.scenario_id}).mappings().all()
        return [KnowledgeBaseVersion.from_row(row) for row in rows]

    def get(self, kb_version: str | None) -> KnowledgeBaseVersion | None:
        """按版本号读取版本记录。"""
        if not kb_version:
            return None
        self.ensure_tables()
        sql = f"""
        SELECT
            scenario_id, kb_version, status, description, created_at,
            activated_at, archived_at, doc_collection, faq_collection,
            embedding_model_version, reranker_model_version, chunk_schema_version,
            created_by, sources_json, stats_json
        FROM {self.version_table}
        WHERE scenario_id = :scenario_id AND kb_version = :kb_version
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                text(sql),
                {"scenario_id": self.scenario.scenario_id, "kb_version": kb_version},
            ).mappings().fetchone()
        return KnowledgeBaseVersion.from_row(row) if row else None

    def exists(self, kb_version: str | None) -> bool:
        """判断版本是否存在于 MySQL。"""
        return self.get(kb_version) is not None

    def active_version_candidate(self) -> str:
        """返回配置声明的 active 版本号（允许为空）。"""
        configured = self.settings.active_kb_version.strip()
        if configured:
            return configured
        active, _ = self._active_pointer()
        return active

    def resolve_active_version(self, requested: str | None = None) -> str:
        """解析检索应使用的知识库版本。（★★ 理解）"""
        if requested:
            candidate = requested.strip()
            if not self.exists(candidate):
                raise ValueError(f"请求的知识库版本不存在：{candidate}")
            return candidate
        active = self.active_version_candidate()
        if not active:
            raise ValueError(f"场景 {self.scenario.scenario_id} 没有 active 知识库版本")
        if not self.exists(active):
            if self.settings.active_kb_version.strip() == active:
                raise ValueError(f"ACTIVE_KB_VERSION 不存在于版本表：{active}")
            raise ValueError(f"active 知识库版本不存在于版本表：{active}")
        return active

    def ensure_version(
        self,
        kb_version: str | None = None,
        *,
        create_new: bool = False,
        description: str = "",
        created_by: str = "local",
    ) -> KnowledgeBaseVersion:
        """确保版本记录存在（不自动覆盖已有版本）。（★★★ 核心）"""
        self.ensure_tables()
        candidate = (kb_version or "").strip()
        if create_new:
            candidate = candidate or generate_kb_version(scenario_id=self.scenario.scenario_id)
        elif not candidate:
            active = self.active_version_candidate()
            candidate = active or generate_kb_version(scenario_id=self.scenario.scenario_id)

        existing = self.get(candidate)
        if existing is not None:
            if description and not existing.description:
                existing.description = description
                self._upsert_version(existing)
            return existing

        settings = self.settings
        should_auto_activate = not self._active_pointer()[0] and not settings.active_kb_version.strip()
        record = KnowledgeBaseVersion(
            kb_version=candidate,
            scenario_id=self.scenario.scenario_id,
            status=KB_VERSION_STATUS_ACTIVE if should_auto_activate else KB_VERSION_STATUS_STAGED,
            description=description,
            activated_at=utc_now() if should_auto_activate else None,
            doc_collection=self.scenario.doc_collection,
            faq_collection=self.scenario.faq_collection,
            embedding_model_version=settings.embedding_model_version,
            reranker_model_version=settings.reranker_model_version,
            chunk_schema_version=settings.chunk_schema_version,
            created_by=created_by,
            stats={"created_reason": "ingest_or_manual"},
        )
        with self.engine.begin() as conn:
            self._upsert_version_with_conn(conn, record)
            if should_auto_activate:
                self._set_active_pointer_with_conn(conn, record.kb_version, "")
        return record

    def activate_version(self, kb_version: str) -> KnowledgeBaseVersion:
        """把指定版本切为当前在线检索版本。（★★★ 核心）"""
        self.ensure_tables()
        record = self.get(kb_version)
        if record is None:
            raise ValueError(f"知识库版本不存在：{kb_version}")

        now = utc_now()
        with self.engine.begin() as conn:
            pointer = conn.execute(
                text(f"SELECT active_kb_version FROM {self.active_table} WHERE scenario_id=:scenario_id FOR UPDATE"),
                {"scenario_id": self.scenario.scenario_id},
            ).mappings().fetchone()
            previous = str(pointer["active_kb_version"] or "") if pointer else ""
            conn.execute(
                text(
                    f"""
                    UPDATE {self.version_table}
                    SET status=:status
                    WHERE scenario_id=:scenario_id AND status=:active_status
                    """
                ),
                {
                    "status": KB_VERSION_STATUS_STAGED,
                    "scenario_id": self.scenario.scenario_id,
                    "active_status": KB_VERSION_STATUS_ACTIVE,
                },
            )
            record.status = KB_VERSION_STATUS_ACTIVE
            record.activated_at = now
            self._upsert_version_with_conn(conn, record)
            self._set_active_pointer_with_conn(
                conn,
                kb_version,
                previous if previous != kb_version else self._active_pointer_with_conn(conn)[1],
            )
        return self.get(kb_version) or record

    def archive_version(self, kb_version: str) -> KnowledgeBaseVersion:
        """归档非 active 版本，仅改状态不删 Milvus 数据。"""
        self.ensure_tables()
        if self.active_version_candidate() == kb_version:
            raise ValueError("不能归档当前 active 知识库版本")
        record = self.get(kb_version)
        if record is None:
            raise ValueError(f"知识库版本不存在：{kb_version}")
        record.status = KB_VERSION_STATUS_ARCHIVED
        record.archived_at = utc_now()
        self._upsert_version(record)
        return record

    def record_ingest_result(
        self,
        kb_version: str,
        *,
        content_type: str,
        count: int,
        source: str | None = None,
    ) -> KnowledgeBaseVersion:
        """记录某次入库结果统计。"""
        self.ensure_tables()
        record = self.get(kb_version)
        if record is None:
            record = self.ensure_version(kb_version)
        if source and source not in record.sources:
            record.sources.append(source)
        key = f"last_{content_type}_count"
        runs_key = f"{content_type}_ingest_runs"
        total_key = f"total_{content_type}_written"
        record.stats[key] = count
        record.stats[runs_key] = int(record.stats.get(runs_key, 0)) + 1
        record.stats[total_key] = int(record.stats.get(total_key, 0)) + count
        record.stats["last_ingested_at"] = utc_now()
        self._upsert_version(record)
        return record

    def as_payload(self) -> dict[str, Any]:
        """返回 API 和脚本可以直接打印的版本管理视图。"""
        active, previous = self._active_pointer()
        try:
            effective_active = self.resolve_active_version()
        except ValueError:
            effective_active = None
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.display_name,
            "active_version": active or None,
            "effective_active_version": effective_active,
            "previous_version": previous or None,
            "active_version_source": "env" if self.settings.active_kb_version.strip() else "mysql",
            "metadata_store": "mysql",
            "versions": [item.as_dict() for item in self.list_versions()],
        }

    def _active_pointer(self) -> tuple[str, str]:
        """读取当前场景 active/previous 指针。"""
        self.ensure_tables()
        with self.engine.begin() as conn:
            return self._active_pointer_with_conn(conn)

    def _active_pointer_with_conn(self, conn) -> tuple[str, str]:
        row = conn.execute(
            text(
                f"""
                SELECT active_kb_version, previous_kb_version
                FROM {self.active_table}
                WHERE scenario_id = :scenario_id
                """
            ),
            {"scenario_id": self.scenario.scenario_id},
        ).mappings().fetchone()
        if not row:
            return "", ""
        return str(row["active_kb_version"] or ""), str(row["previous_kb_version"] or "")

    def _set_active_pointer_with_conn(self, conn, active: str, previous: str) -> None:
        sql = f"""
        INSERT INTO {self.active_table}
            (scenario_id, active_kb_version, previous_kb_version)
        VALUES
            (:scenario_id, :active_kb_version, :previous_kb_version)
        ON DUPLICATE KEY UPDATE
            active_kb_version=VALUES(active_kb_version),
            previous_kb_version=VALUES(previous_kb_version),
            updated_at=CURRENT_TIMESTAMP
        """
        conn.execute(
            text(sql),
            {
                "scenario_id": self.scenario.scenario_id,
                "active_kb_version": active,
                "previous_kb_version": previous,
            },
        )

    def _upsert_version(self, record: KnowledgeBaseVersion) -> None:
        self.ensure_tables()
        with self.engine.begin() as conn:
            self._upsert_version_with_conn(conn, record)

    def _upsert_version_with_conn(self, conn, record: KnowledgeBaseVersion) -> None:
        sql = f"""
        INSERT INTO {self.version_table}
            (
                scenario_id, kb_version, status, description, created_at,
                activated_at, archived_at, doc_collection, faq_collection,
                embedding_model_version, reranker_model_version, chunk_schema_version,
                created_by, sources_json, stats_json
            )
        VALUES
            (
                :scenario_id, :kb_version, :status, :description, :created_at,
                :activated_at, :archived_at, :doc_collection, :faq_collection,
                :embedding_model_version, :reranker_model_version, :chunk_schema_version,
                :created_by, :sources_json, :stats_json
            )
        ON DUPLICATE KEY UPDATE
            status=VALUES(status),
            description=VALUES(description),
            activated_at=VALUES(activated_at),
            archived_at=VALUES(archived_at),
            doc_collection=VALUES(doc_collection),
            faq_collection=VALUES(faq_collection),
            embedding_model_version=VALUES(embedding_model_version),
            reranker_model_version=VALUES(reranker_model_version),
            chunk_schema_version=VALUES(chunk_schema_version),
            created_by=VALUES(created_by),
            sources_json=VALUES(sources_json),
            stats_json=VALUES(stats_json),
            updated_at=CURRENT_TIMESTAMP
        """
        conn.execute(
            text(sql),
            {
                "scenario_id": record.scenario_id or self.scenario.scenario_id,
                "kb_version": record.kb_version,
                "status": record.status,
                "description": record.description,
                "created_at": record.created_at,
                "activated_at": record.activated_at,
                "archived_at": record.archived_at,
                "doc_collection": record.doc_collection,
                "faq_collection": record.faq_collection,
                "embedding_model_version": record.embedding_model_version,
                "reranker_model_version": record.reranker_model_version,
                "chunk_schema_version": record.chunk_schema_version,
                "created_by": record.created_by,
                "sources_json": _json_dumps(record.sources),
                "stats_json": _json_dumps(record.stats),
            },
        )

    def _reconcile_active_status(self) -> None:
        """保证 active 指针对应的版本行状态为 ACTIVE。"""
        with self.engine.begin() as conn:
            active, _ = self._active_pointer_with_conn(conn)
            if not active:
                return
            conn.execute(
                text(
                    f"""
                    UPDATE {self.version_table}
                    SET status=:staged_status
                    WHERE scenario_id=:scenario_id
                      AND status=:active_status
                      AND kb_version != :active_kb_version
                    """
                ),
                {
                    "staged_status": KB_VERSION_STATUS_STAGED,
                    "active_status": KB_VERSION_STATUS_ACTIVE,
                    "scenario_id": self.scenario.scenario_id,
                    "active_kb_version": active,
                },
            )
            conn.execute(
                text(
                    f"""
                    UPDATE {self.version_table}
                    SET status=:active_status
                    WHERE scenario_id=:scenario_id AND kb_version=:active_kb_version
                    """
                ),
                {
                    "active_status": KB_VERSION_STATUS_ACTIVE,
                    "scenario_id": self.scenario.scenario_id,
                    "active_kb_version": active,
                },
            )


def get_kb_version_store(scenario_id: str | None = None) -> KnowledgeBaseVersionStore:
    """返回新的版本表访问对象（不缓存，保证看到最新 MySQL 状态）。"""
    return KnowledgeBaseVersionStore(scenario_id=scenario_id)


def resolve_active_kb_version(requested: str | None = None, scenario_id: str | None = None) -> str:
    """解析当前请求应使用的知识库版本。"""
    return get_kb_version_store(scenario_id).resolve_active_version(requested)


def version_metadata(kb_version: str | None, scenario_id: str | None = None) -> dict[str, str]:
    """构建写入 FAQ/chunk metadata 的版本字段，记录模型版本和切分方案。"""
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)
    return {
        "scenario_id": scenario.scenario_id,
        "kb_version": kb_version or "",
        "embedding_model_version": settings.embedding_model_version,
        "reranker_model_version": settings.reranker_model_version,
        "chunk_schema_version": settings.chunk_schema_version,
    }


@lru_cache(maxsize=1)
def _dummy_cache_marker() -> bool:
    """保留模块级 lru_cache 依赖占位，便于旧测试清理缓存时不报错。"""
    return True
