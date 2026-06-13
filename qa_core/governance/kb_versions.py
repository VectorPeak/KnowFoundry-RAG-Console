"""知识库多版本管理。

当前项目用本地 JSON manifest 管理知识库版本状态。版本切换只改变检索过滤使用的
kb_version，不直接修改 Milvus 数据，因此可以低成本支持激活、回滚和评测对比。

核心能力：
- 生成带时间戳和配置 hash 的版本号。
- 跟踪版本生命周期：STAGED -> ACTIVE -> ARCHIVED。
- 按“请求参数 > 环境变量 > manifest”的优先级解析检索版本。
- 记录每个版本的 FAQ/文档入库统计。
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qa_core.common import utc_file_stamp, utc_now
from qa_core.config.settings import get_settings
from qa_core.json_store import JsonFileStore
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import stable_hash

def _resolve_version_scenario(scenario_id: str | None = None):
    """解析知识库版本所属的场景配置，避免模块循环依赖。

    这里包一层 resolve_scenario，是为了避免版本管理模块和场景注册模块形成复杂依赖。

    参数：
        scenario_id: 可选场景 ID；为空时解析默认场景。

    返回：
        解析后的场景配置对象.
    """
    return resolve_scenario(scenario_id)


def generate_kb_version(prefix: str = "kb", scenario_id: str | None = None) -> str:
    """生成含时间戳和配置 hash 的知识库版本号。（★★ 理解）

    版本号由场景 ID、创建时间戳和检索配置 hash 组成。配置 hash 覆盖 embedding 模型、
    reranker 模型、chunk schema 和集合名，用来区分不同入库配置。

    版本号中包含配置 hash 的决策意义：如果 embedding/reranker/chunk 配置发生变化，生成的新版本号必然不同，
    确保检索时使用的模型配置与入库时一致，避免模型升级后对旧数据的检索质量下降。

    示例：
        ``generate_kb_version("kb", "enterprise_knowledge")`` 可能返回
        ``"kb_enterprise_knowledge_20250101_120000_a1b2c3d4"``。

    参数：
        prefix: 版本号前缀，默认 kb。
        scenario_id: 可选场景 ID；为空时解析默认场景。

    返回：
        生成的版本号.
    """
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
    """可检索知识库版本的元数据。

    实际 chunk 内容存放在 Milvus 中；这个对象只记录版本生命周期、配置快照和入库统计。

    字段说明：
      - kb_version：版本号。
      - scenario_id：所属业务场景。
      - status：生命周期状态，支持 STAGED、ACTIVE、ARCHIVED。
      - description：版本描述。
      - created_at / activated_at / archived_at：创建、激活、归档时间。
      - doc_collection / faq_collection：创建版本时使用的 Milvus 集合名。
      - embedding_model_version / reranker_model_version / chunk_schema_version：入库配置快照。
      - created_by：创建来源。
      - sources：该版本已入库的 source 列表。
      - stats：入库统计信息。
    """

    kb_version: str
    scenario_id: str = ""
    status: str = "STAGED"
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
        """从 JSON 字典恢复版本对象，老记录缺失的字段自动使用默认值。

        从 JSON 记录恢复版本对象。旧版本 manifest 缺失的字段会使用 dataclass 默认值补齐。

        示例：
            ``from_dict({"kb_version": "v1"})`` 会得到 sources=[]、stats={} 的版本对象。

        参数：
            payload: 从 JSON manifest 读取的字典。

        返回：
            补齐默认值后的 KnowledgeBaseVersion。
        """
        fields = cls.__dataclass_fields__
        data = {name: payload.get(name) for name in fields if name in payload}
        version = cls(**data)
        if version.sources is None:
            version.sources = []
        if version.stats is None:
            version.stats = {}
        return version

    def as_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的版本信息。

        将版本对象转换成 JSON 兼容字典，用于写入 manifest 或 API 返回。

        返回：
            包含 dataclass 全部字段的字典.
        """
        return asdict(self)


class KnowledgeBaseVersionStore(JsonFileStore):
    """知识库版本清单读写器，基于本地 JSON 文件维护版本状态。

    这个类负责读取和写入某个场景的版本 manifest，提供版本列表、创建、激活、归档和
    入库统计记录能力。

    manifest 结构示例：

        {
            "scenario_id": "hr",
            "active_version": "kb_hr_20250101_120000_a1b2c3d4",
            "previous_version": "kb_hr_20241201_000000_xxxx",
            "versions": {
                "kb_hr_20250101_120000_a1b2c3d4": { ... },
                ...
            }
        }
    """

    def __init__(self, path: str | Path | None = None, scenario_id: str | None = None) -> None:
        """打开版本清单文件，path 可指定临时路径，默认使用当前场景的版本清单。

        未显式传 path 时，默认使用当前场景配置里的 kb_versions_manifest_path。

        参数：
            path: 可选 JSON manifest 路径。
            scenario_id: 可选场景 ID；为空时解析默认场景。
        """
        self.scenario = _resolve_version_scenario(scenario_id)
        super().__init__(path or self.scenario.kb_versions_manifest_path)

    def empty_data(self) -> dict[str, Any]:
        """返回空版本清单结构。

        manifest 文件不存在时，JsonFileStore 会使用这个结构初始化空清单。

        返回：
            包含 scenario_id、active_version、previous_version 和 versions 的空清单.
        """
        return {"scenario_id": self.scenario.scenario_id, "active_version": "", "previous_version": "", "versions": {}}

    def normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """补齐历史版本清单缺失字段，避免多次入库间互相覆盖统计信息。

        读取旧格式 manifest 时补齐顶层字段，避免后续写回时丢失结构。

        参数：
            data: 从 JSON 读取的原始 manifest 字典。

        返回：
            保证包含 active_version、previous_version、scenario_id、versions 的清单.
        """
        data.setdefault("active_version", "")
        data.setdefault("previous_version", "")
        data.setdefault("scenario_id", self.scenario.scenario_id)
        data.setdefault("versions", {})
        return data

    def list_versions(self) -> list[KnowledgeBaseVersion]:
        """按创建时间倒序返回所有版本。

        返回 manifest 中全部版本，按 created_at 倒序排列。

        返回：
            最新版本在前的 KnowledgeBaseVersion 列表.
        """
        versions = [KnowledgeBaseVersion.from_dict(raw) for raw in self.data.get("versions", {}).values()]
        return sorted(versions, key=lambda item: item.created_at or "", reverse=True)

    def get(self, kb_version: str | None) -> KnowledgeBaseVersion | None:
        """按版本号读取版本记录。

        根据版本号读取单个版本记录。

        参数：
            kb_version: 版本号，允许为空。

        返回：
            找到时返回 KnowledgeBaseVersion，否则返回 None.
        """
        if not kb_version:
            return None
        raw = self.data.get("versions", {}).get(kb_version)
        if not raw:
            return None
        return KnowledgeBaseVersion.from_dict(raw)

    def exists(self, kb_version: str | None) -> bool:
        """判断版本是否存在于清单中。

        判断指定版本号是否存在于 manifest。

        参数：
            kb_version: 待检查版本号，允许为空。

        返回：
            存在返回 True，否则返回 False.
        """
        return bool(kb_version and kb_version in self.data.get("versions", {}))

    def active_version_candidate(self) -> str:
        """返回配置声明的 active 版本号（允许为空）。

        优先从环境变量 active_kb_version 读取候选版本；为空时读取 manifest.active_version。
        返回值可能为空，在线检索应调用 resolve_active_version() 获取最终权威版本。

        返回：
            候选版本号，可能为空字符串.
        """
        configured = get_settings().active_kb_version.strip()
        if configured:
            return configured
        return str(self.data.get("active_version") or "").strip()

    def resolve_active_version(self, requested: str | None = None) -> str:
        """解析检索应使用的知识库版本。（★★ 理解）

        按“请求参数 > 环境变量 active_kb_version > manifest.active_version”的优先级
        解析本次检索使用的知识库版本。

        三优先级决策：API 请求参数（在线临时切换）> 环境变量（部署时固定版本）> manifest（默认版本）。
        环境变量优先级高于 manifest，支持灰度发布：线上用 env 指定新版本，manifest 保持老版本做回退基线。

        参数：
            requested: API 请求中显式指定的版本；传入时必须存在于 manifest。

        返回：
            解析后的 active 版本号。

        异常：
            ValueError: 指定版本不存在，或当前场景没有可用 active 版本。
        """
        if requested:
            candidate = requested.strip()
            if not self.exists(candidate):
                raise ValueError(f"请求的知识库版本不存在：{candidate}")
            return candidate
        active = self.active_version_candidate()
        if not active:
            raise ValueError(f"场景 {self.scenario.scenario_id} 没有 active 知识库版本")
        # 验证版本存在性：env 或 manifest 中声明的版本必须在清单中有记录，防止引用了已删除或从未入库的版本
        if not self.exists(active):
            if get_settings().active_kb_version.strip() == active:
                raise ValueError(f"ACTIVE_KB_VERSION 不存在于版本清单：{active}")
            raise ValueError(f"active 知识库版本不存在于版本清单：{active}")
        return active

    def ensure_version(
        self,
        kb_version: str | None = None,
        *,
        create_new: bool = False,
        description: str = "",
        created_by: str = "local",
    ) -> KnowledgeBaseVersion:
        """确保版本记录存在（不自动激活）。（★★★ 核心）

        确保 manifest 里有指定版本记录。除首次入库外，它不会自动激活新版本。

        执行流程：
          1. 确定候选版本号：create_new 时生成新版本；否则使用传入版本、active 候选或新版本。
          2. 候选版本不存在时创建记录，状态为 STAGED；首次入库且没有 active 版本时自动设为 ACTIVE。
          3. 创建时快照当前模型、集合和 chunk schema 配置。
          4. 候选版本已存在时只补填空 description，不覆盖已有描述。

        版本管理是入库关键：每次数据入库对应一个版本号，记录当时使用的 embedding/reranker/chunk 配置快照。
        这确保检索时使用与入库时一致的模型配置，防止模型升级后旧数据检索效果不一致。

        参数：
            kb_version: 显式版本号；为空时按规则自动确定。
            create_new: 是否总是创建新版本号。
            description: 新版本描述。
            created_by: 创建来源。

        返回：
            已存在或新创建的版本记录.
        """
        candidate = (kb_version or "").strip()
        # 开启 create_new 时总是生成新版本号，避免覆盖已有版本数据
        if create_new:
            candidate = candidate or generate_kb_version(scenario_id=self.scenario.scenario_id)
        elif not candidate:
            # 未指定版本时尝试复用当前 active 版本或自动生成
            active = self.active_version_candidate()
            candidate = active or generate_kb_version(scenario_id=self.scenario.scenario_id)

        raw_versions = self.data.setdefault("versions", {})
        if candidate not in raw_versions:
            settings = get_settings()
            # 自动激活逻辑：只有首次入库（没有任何 active 版本）才自动激活；已有 active 则新版本先保持 STAGED 等待审核
            should_auto_activate = not self.data.get("active_version") and not settings.active_kb_version.strip()
            record = KnowledgeBaseVersion(
                kb_version=candidate,
                scenario_id=self.scenario.scenario_id,
                status="ACTIVE" if should_auto_activate else "STAGED",
                description=description,
                activated_at=utc_now() if should_auto_activate else None,
                # 快照当前配置到版本记录：即使未来模型升级，此版本的检索行为仍然可复现
                doc_collection=self.scenario.doc_collection,
                faq_collection=self.scenario.faq_collection,
                embedding_model_version=settings.embedding_model_version,
                reranker_model_version=settings.reranker_model_version,
                chunk_schema_version=settings.chunk_schema_version,
                created_by=created_by,
                stats={"created_reason": "ingest_or_manual"},
            )
            raw_versions[candidate] = record.as_dict()
            if should_auto_activate:
                self.data["active_version"] = candidate
            self.save()
            return record

        # 版本已存在：只补填 description（不覆盖已有值），避免重复入库覆盖手动填写的描述信息
        record = KnowledgeBaseVersion.from_dict(raw_versions[candidate])
        changed = False
        if description and not record.description:
            record.description = description
            changed = True
        if changed:
            raw_versions[candidate] = record.as_dict()
            self.save()
        return record

    def activate_version(self, kb_version: str) -> KnowledgeBaseVersion:
        """把指定版本切为当前在线检索版本。（★★★ 核心）

        激活版本只修改本地 JSON manifest，不修改 Milvus 数据。在线检索会把 active 版本写入
        Milvus 过滤表达式，从而只检索对应版本的资料。

        执行流程：
          1. 重新加载 manifest，避免使用旧状态。
          2. 校验目标版本存在。
          3. 把当前 active 记录为 previous_version。
          4. 目标版本设为 ACTIVE，其他原 ACTIVE 版本降为 STAGED。
          5. 保存 manifest 并返回激活后的版本记录。

        版本切换只改 manifest 不改 Milvus 数据，代价极低（毫秒级），支持快速回滚。
        老 active 版本自动降为 STAGED 而非直接删除，保留回退路径。

        参数：
            kb_version: 要激活的版本号。

        返回：
            激活后的版本记录。

        异常：
            ValueError: 版本不存在。
        """
        self.reload()
        # 验证目标版本存在后再操作，防止激活不存在的版本导致检索异常
        record = self.get(kb_version)
        if record is None:
            raise ValueError(f"知识库版本不存在：{kb_version}")

        previous = str(self.data.get("active_version") or "")
        now = utc_now()
        for vid, raw in self.data.setdefault("versions", {}).items():
            item = KnowledgeBaseVersion.from_dict(raw)
            if vid == kb_version:
                item.status = "ACTIVE"
                item.activated_at = now
            elif item.status == "ACTIVE":
                # 将此前 ACTIVE 版本降为 STAGED，保留回退能力而不删除数据
                item.status = "STAGED"
            self.data["versions"][vid] = item.as_dict()

        # 记录前一个 active 版本，用于快速回滚对比和 A/B 评估
        self.data["previous_version"] = previous if previous != kb_version else str(self.data.get("previous_version") or "")
        self.data["active_version"] = kb_version
        self.save()
        return self.get(kb_version) or record

    def archive_version(self, kb_version: str) -> KnowledgeBaseVersion:
        """归档非 active 版本，仅改状态不删 Milvus 数据。

        归档只是把版本状态改为 ARCHIVED，不删除 Milvus 底层数据，后续仍可恢复。

        参数：
            kb_version: 要归档的版本号。

        返回：
            归档后的版本记录。

        异常：
            ValueError: 版本不存在，或试图归档当前 active 版本。
        """
        self.reload()
        if self.active_version_candidate() == kb_version:
            raise ValueError("不能归档当前 active 知识库版本")
        record = self.get(kb_version)
        if record is None:
            raise ValueError(f"知识库版本不存在：{kb_version}")
        record.status = "ARCHIVED"
        record.archived_at = utc_now()
        self.data.setdefault("versions", {})[kb_version] = record.as_dict()
        self.save()
        return record

    def record_ingest_result(
        self,
        kb_version: str,
        *,
        content_type: str,
        count: int,
        source: str | None = None,
    ) -> KnowledgeBaseVersion:
        """记录某次入库结果统计。

        这里只记录统计信息，实际可检索内容仍在 Milvus 中。统计包括本次写入数量、累计写入数量、
        入库次数和最近入库时间。

        参数：
            kb_version: 要更新的版本号。
            content_type: 入库内容类型，例如 faq 或 doc。
            count: 本次写入数量。
            source: 本次入库涉及的 source。

        返回：
            更新后的版本记录.
        """
        self.reload()
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
        self.data.setdefault("versions", {})[kb_version] = record.as_dict()
        self.save()
        return record

    def as_payload(self) -> dict[str, Any]:
        """返回 API 和脚本可以直接打印的版本管理视图。

        构建 API 和调试页面使用的完整版本视图，包含场景信息、active/previous 版本、
        active 来源、manifest 路径和全部版本记录。

        返回：
            完整版本管理状态字典.
        """
        try:
            effective_active = self.resolve_active_version()
        except ValueError:
            effective_active = None
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.display_name,
            "active_version": self.data.get("active_version") or None,
            "effective_active_version": effective_active,
            "previous_version": self.data.get("previous_version") or None,
            "active_version_source": "env" if get_settings().active_kb_version.strip() else "manifest",
            "manifest_path": str(self.path),
            "versions": [item.as_dict() for item in self.list_versions()],
        }


def get_kb_version_store(scenario_id: str | None = None) -> KnowledgeBaseVersionStore:
    """返回新的版本清单访问对象（不缓存，保证看到最新 JSON）。

    每次调用都创建新对象，不做缓存，确保读取到最新 JSON 内容。

    参数：
        scenario_id: 可选场景 ID；为空时解析默认场景。

    返回：
        新的 KnowledgeBaseVersionStore 实例.
    """
    return KnowledgeBaseVersionStore(scenario_id=scenario_id)


def resolve_active_kb_version(requested: str | None = None, scenario_id: str | None = None) -> str:
    """解析当前请求应使用的知识库版本。

    便捷函数，内部委托 KnowledgeBaseVersionStore.resolve_active_version()。

    参数：
        requested: API 请求中显式指定的版本。
        scenario_id: 可选场景 ID。

    返回：
        解析后的 active 版本号。

    异常：
        ValueError: 版本解析失败时抛出。
    """
    return get_kb_version_store(scenario_id).resolve_active_version(requested)


def version_metadata(kb_version: str | None, scenario_id: str | None = None) -> dict[str, str]:
    """构建写入 FAQ/chunk metadata 的版本字段，记录模型版本和切分方案。

    构建写入 Milvus metadata 的版本字段，便于后续追踪每个 FAQ/chunk 的入库配置。

    参数：
        kb_version: 知识库版本号，允许为空。
        scenario_id: 可选场景 ID。

    返回：
        包含 scenario_id、kb_version、embedding_model_version、reranker_model_version、
        chunk_schema_version 的字典.
    """
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)
    return {
        "scenario_id": scenario.scenario_id,
        "kb_version": kb_version or "",
        "embedding_model_version": settings.embedding_model_version,
        "reranker_model_version": settings.reranker_model_version,
        "chunk_schema_version": settings.chunk_schema_version,
    }

