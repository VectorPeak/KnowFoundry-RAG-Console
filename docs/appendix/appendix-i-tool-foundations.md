# 附录 I：项目工具类开发详解

**前置阅读**：[第 13 讲：应用入口与环境前置校验](../13-app-entry-preflight.md)

本附录覆盖项目中所有跨模块复用的基础工具类，包括设计思路、实现细节和使用模式。

---

## 一、common.py：UTC 时间与 JSON 读写

### 设计思路

整个项目的时间戳统一使用 **UTC 时间**，避免时区混乱。所有需要持久化的时间（trace 记录、版本号、入库报告）都用 `utc_now()`，所有文件系统中的时间标识都用 `utc_file_stamp()`。

```python
# qa_core/common.py
import json
from datetime import datetime, timezone
from pathlib import Path

def utc_now() -> str:
    """返回 UTC ISO 时间字符串，用于版本管理、报告和 trace 的时间统一排序。
    格式: '2025-01-15T10:30:00.123456+00:00'
    """
    return datetime.now(timezone.utc).isoformat()

def utc_file_stamp() -> str:
    """返回适合放进文件名和版本号的 UTC 时间戳。
    格式: '20250115_103000'
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def path_updated_at(path: str | Path) -> str:
    """返回文件修改时间的 UTC ISO 字符串，用于报告列表的排序。"""
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()
```

### JSON 读写封装

项目统一使用 `ensure_ascii=False, indent=2` 格式写 JSON，保证中文字符可读、层级清晰。

```python
# qa_core/common.py

def read_json(path: str | Path, default: Any = None) -> Any:
    """读取 JSON 文件，读取失败时返回 default —— 避免文件损坏导致进程崩溃。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default

def read_json_dict(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取对象型 JSON，非对象或读取失败时返回字典默认值。
    用于版本清单、索引清单、质量报告等核心配置文件。
    """
    payload = read_json(path, default=None)
    if isinstance(payload, dict):
        return payload
    return dict(default or {})

def write_json(path: str | Path, payload: Any) -> str:
    """按项目统一格式写入 JSON。自动创建父目录。返回绝对路径。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return str(output_path)
```

### 列表报告工具

模板方法模式：通用的 JSON 文件扫描 + 按修改时间排序逻辑，被质量报告和状态页复用。

```python
def list_json_reports(root: Path, glob_pattern: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """列出目录下最近的 JSON 报告。
    每个条目包含 path / file_name / updated_at / payload 四个字段。
    被 `quality/ingestion.py` 的列表页复用。
    """
    if not root.exists():
        return []
    reports = []
    for path in sorted(root.glob(glob_pattern),
                       key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json_dict(path)
        reports.append({
            "path": str(path), "file_name": path.name,
            "updated_at": path_updated_at(path), "payload": payload,
        })
        if limit and len(reports) >= limit:
            break
    return reports
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | `write_json()` 的统一格式设计（ensure_ascii=False, indent=2） | 整个项目的持久化都依赖这个约定 |
| ★★★ 必会 | `read_json(path, default=...)` 的容错模式 | 避免文件损坏导致进程崩溃 |
| ★★ 理解 | `utc_now()` vs `utc_file_stamp()` 的使用场景区别 | UTC 规范是系统间协作的基础 |
| ★ 了解 | `list_json_reports()` 的模板方法模式 | 用于状态页报告列表 |

---

## 二、utils.py：稳定哈希与指纹

### 设计思路

入库和检索都需要稳定的 ID 生成——同一个文件重新入库时产生相同的 chunk_id，才能正确覆盖旧数据。`stable_hash()` 用 SHA-256 将任意数量的参数拼接后生成确定性的 hex 字符串。`file_fingerprint()` 只读文件元数据（路径 + 修改时间 + 大小），不读内容，大幅减少增量入库的 IO。

```python
# qa_core/utils.py
import hashlib
import os
from pathlib import Path

def stable_hash(*parts: object) -> str:
    """根据任意值创建稳定的 SHA-256 hex 字符串。
    
    关键特性：相同输入永远产生相同输出（确定性）。
    用于生成 chunk_id、faq_id、parent_id 等 Milvus 主键。
    
    例: stable_hash("hr_v2", "/data/入职流程.md", "chunk_3") 
        → 'a1b2c3d4e5f6...' (64 位 hex)
    """
    raw = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def file_fingerprint(path: str | Path) -> str:
    """根据文件路径、修改时间和大小生成指纹，用于增量入库的变化判断。
    
    为什么不用内容哈希 (SHA-256 of content)？
    - 大文件（几十 MB 的 PDF）读内容做哈希太慢
    - 教学机器 CPU 有限，每轮入库读全量文件成本太高
    - 路径 + mtime + size 的组合在实际使用中已经足够可靠
    """
    p = Path(path)
    stat = p.stat()
    return stable_hash(str(p.resolve()), stat.st_mtime_ns, stat.st_size)

def normalize_source_from_path(path: str | Path) -> str:
    """从 '<source>_data' 目录名提取 source 标识。
    
    例: 'scenarios/enterprise_knowledge/data/hr_data' → 'hr'
        'data/legal_data' → 'legal'
    """
    name = os.path.basename(str(path)).replace("_data", "")
    return name or "default"
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | `stable_hash()` 的确定性设计 | chunk_id/faq_id 的稳定性依赖于此 |
| ★★★ 必会 | `file_fingerprint()` 为什么用 mtime+size 而不是内容哈希 | IO 成本与可靠性之间的工程权衡 |
| ★★ 理解 | `normalize_source_from_path()` 的命名约定 | 入库脚本的 source 推断依赖目录命名规范 |

---

## 三、json_store.py：JSON 文件型 Store 基类

### 设计思路

知识库版本清单 (`kb_versions.json`) 和文件索引清单 (`documents.json`) 都是本地 JSON 文件存储。它们共享相同的能力：加载 → 补齐字段（兼容历史版本） → 读/写 → reload。`JsonFileStore` 把这些共性抽成基类，子类只需实现 `empty_data()` 和 `normalize_data()`。

```python
# qa_core/json_store.py
from pathlib import Path
from typing import Any
from qa_core.common import read_json_dict, write_json

class JsonFileStore:
    """对象型 JSON 文件 Store 的最小公共能力。
    
    子类：
    - KnowledgeBaseVersionStore (governance/kb_versions.py)
    - IndexManifest (indexing/manifest.py)
    
    子类需要实现：
    - empty_data() → 返回空结构（如 {"versions": {}, "active_version": ""}）
    - normalize_data(data) → 补齐历史文件缺失字段
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def empty_data(self) -> dict[str, Any]:
        return {}

    def normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self.empty_data()
        return self.normalize_data(read_json_dict(self.path, self.empty_data()))

    def reload(self) -> None:
        """重新读取磁盘文件。入库脚本中多次写入同一清单时，
        每次操作前 reload 可以避免内存对象盖掉别人刚写入的数据。"""
        self.data = self._load()

    def save(self) -> None:
        write_json(self.path, self.data)
```

### 子类示例：KnowledgeBaseVersionStore

```python
# qa_core/governance/kb_versions.py
class KnowledgeBaseVersionStore(JsonFileStore):
    def empty_data(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "active_version": "",
            "previous_version": "",
            "versions": {}
        }

    def normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # 补齐旧版本清单中可能缺失的字段
        data.setdefault("active_version", "")
        data.setdefault("previous_version", "")
        data.setdefault("scenario_id", self.scenario.scenario_id)
        data.setdefault("versions", {})
        return data
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | `JsonFileStore` 的模板方法模式（empty + normalize） | 两个核心清单 Store 都基于此模式 |
| ★★ 理解 | `reload()` 的使用场景（多步入库之间的数据同步） | 避免入库脚本多步写入时互相覆盖 |
| ★ 了解 | 为什么用 JSON 而不是 SQLite | 教学项目优先可调试性（JSON 可直接打开查看） |

---

## 四、schemas.py：Pydantic 数据模型

### 设计思路

所有 API 层的请求/响应都用 Pydantic BaseModel 定义，FastAPI 自动做参数校验和序列化。内部流转的数据结构用 Python dataclass（性能更好，不需要校验）。

```python
# qa_core/schemas.py
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """查询请求体 — 同时服务于 /api/query、/api/stream、/api/retrieval/debug。"""
    query: str = Field(..., min_length=1)        # 必填，最短 1 字符
    source_filter: str | None = None              # 前端选择的业务分类
    session_id: str | None = None                 # 会话 ID，未传则自动生成
    scenario_id: str | None = None                # 场景 ID，未传用默认场景
    tenant_id: str | None = None                  # 租户 ID，未传默认 "default"
    dataset_id: str | None = None                 # 数据集 ID，未传默认 "default"
    visibility: str | None = None                 # 可见级别，未传默认 "public"
    user_role: str | None = None                  # 单一角色（前端简化入口）
    user_roles: list[str] = Field(default_factory=list)  # 多角色（正式入口）
    kb_version: str | None = None                 # 知识库版本，未传用 active 版本

class QueryResponse(BaseModel):
    """HTTP 查询响应。is_streaming=True 时前端应改为 WebSocket 连接。"""
    answer: str
    is_streaming: bool                    # True = 需走 WebSocket 流式
    session_id: str
    processing_time: float
    scenario_id: str | None = None
    hit_type: str = "unknown"             # greeting / faq_direct / rag / insufficient
    sources: list[dict] = Field(default_factory=list)
    rewritten_query: str | None = None
    intent: dict | None = None
    retrieval: dict | None = None

class FeedbackRequest(BaseModel):
    """用户反馈载荷。rating 约束为 useful/not_useful，不允许自定义值。"""
    session_id: str | None = None
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(useful|not_useful)$")   # 仅允许这两个值
    comment: str | None = None
    sources: list[dict] = Field(default_factory=list)
```

### 内部流转的 dataclass

```python
# qa_core/schemas.py (节选)
from dataclasses import dataclass, field

@dataclass
class RetrievalHit:
    """一次检索命中的封装：document + score。"""
    document: Any       # LangChain Document 对象
    score: float

@dataclass
class RetrievalResult:
    """一次完整检索的结果：命中列表 + 元信息。"""
    hits: list[RetrievalHit] = field(default_factory=list)
    query: str = ""
    source_type: str = ""      # "faq" 或 "doc"
    elapsed_ms: float = 0.0
    
    @property
    def top_score(self) -> float | None:
        """最高分，无命中时为 None。"""
        return self.hits[0].score if self.hits else None

@dataclass
class QAResult:
    """preview_query() 的返回结果。"""
    answer: str
    hit_type: str
    session_id: str
    intent: dict | None = None
    retrieval: dict | None = None
    elapsed_ms: float = 0.0
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | `QueryRequest` 的所有字段含义 | 是整个系统的对外接口 |
| ★★★ 必会 | `QueryResponse(is_streaming=True)` 的双入口设计 | 前端根据此字段决定走 HTTP 还是 WebSocket |
| ★★ 理解 | `FeedbackRequest.rating` 的 pattern 约束 | 防止前端传自定义值污染数据 |
| ★ 了解 | `RetrievalHit` / `RetrievalResult` 的 dataclass 设计 | 内部流转的数据结构 |

---

## 五、config/settings.py：配置管理

### 设计思路

使用 pydantic-settings 从 `.env` 文件和环境变量加载全部配置。每个配置项有明确的默认值，注释标注了用途。`model_config` 设置 `env_file=".env"` 和 `case_sensitive=False` 使得环境变量大小写不敏感。

```python
# qa_core/config/settings.py (关键字段)
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # ── 场景 ──
    active_scenario_id: str = Field(default="enterprise_knowledge", ...)
    scenario_config_dir: str = Field(default="scenarios", ...)

    # ── Milvus ──
    milvus_uri: str = Field(default="http://localhost:19530", ...)
    milvus_database: str = Field(default="", ...)

    # ── MySQL ──
    mysql_host: str = Field(default="localhost", ...)
    mysql_database: str = Field(default="qa_system", ...)

    # ── LLM (DashScope / OpenAI 兼容) ──
    llm_api_key: str = Field(default="", validation_alias="DASHSCOPE_API_KEY")
    llm_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", ...)
    llm_model: str = Field(default="qwen-plus", ...)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_timeout: int = Field(default=60, ge=1)

    # ── 检索参数（所有默认值可用于 .env 覆盖）──
    faq_top_k: int = Field(default=20, ge=1)
    doc_top_k: int = Field(default=20, ge=1)
    faq_direct_score_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=1)
    max_context_doc_chars: int = Field(default=1600, ge=1)
    short_query_max_chars: int = Field(default=20, ge=1)

    # ── 历史摘要 ──
    history_summary_enabled: bool = Field(default=True)
    history_summary_after_messages: int = Field(default=14, ge=1)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级 Settings 单例。lru_cache 避免重复读取 .env。"""
    return Settings()
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | `get_settings()` 的 `@lru_cache(maxsize=1)` 单例模式 | 整个项目的配置入口 |
| ★★★ 必会 | `validation_alias`（如 DASHSCOPE_API_KEY）的用途 | `.env` 变量名与代码字段名的映射 |
| ★★ 理解 | 检索参数组的默认值和可覆盖性 | 调整这些参数不需要改代码 |
| ★ 了解 | `model_config` 的 `case_sensitive=False` | Windows/Linux 环境变量兼容 |

---

## 六、config/logging_config.py：结构化日志

### 设计思路

项目统一使用 Python 标准 logging + 自定义格式化器，输出 JSON 格式的结构化日志。`get_logger(name)` 为每个模块提供带模块名的 logger 实例。

```python
# qa_core/config/logging_config.py
import logging
import json
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★ 理解 | JSON 格式化日志的设计目的 | 方便 log 分析工具解析 |
| ★ 了解 | `get_logger(name)` 的 handler 防重复逻辑 | 避免多次调用时添加重复 handler |

---

## 七、memory/base.py：MySQL 惰性连接基类

### 设计思路

`ChatHistoryStore` 和 `FeedbackStore` 都需要 MySQL 连接，但不是每个请求都立即需要。`_MySqlStore` 基类提供惰性初始化的引擎属性——第一次访问 `engine` 时才创建连接，创建后缓存复用。`pool_pre_ping=True` 在连接被 MySQL 服务端断开后自动探活重连。

```python
# qa_core/memory/base.py
from sqlalchemy import create_engine, text

class _MySqlStore:
    """MySQL 存储的公共基类。
    
    子类必须在访问 engine 之前设置 self.settings（提供 mysql_sync_uri）。
    子类: ChatHistoryStore, FeedbackStore
    """

    def __init__(self) -> None:
        self._engine = None

    @property
    def engine(self):
        """惰性创建的 SQLAlchemy 同步引擎。
        
        pool_pre_ping=True: 每次从连接池取连接前先发送 ping，
        避免 "MySQL server has gone away" 错误。
        """
        if self._engine is None:
            self._engine = create_engine(
                self.settings.mysql_sync_uri,
                pool_pre_ping=True
            )
        return self._engine

    def _execute_ddl(self, sql: str) -> None:
        """执行 DDL 语句（如 CREATE TABLE IF NOT EXISTS）。
        在事务内执行，自动提交。
        """
        with self.engine.begin() as conn:
            conn.execute(text(sql))
```

### 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | 惰性引擎的 `pool_pre_ping=True` 设计 | 生产环境中 "MySQL gone away" 是高频问题 |
| ★★ 理解 | `_execute_ddl()` 封装的 DDL 执行模式 | 被 ChatHistoryStore 和 FeedbackStore 的建表逻辑复用 |
| ★ 了解 | 为什么不用 ORM 而用原生 SQL | 教学项目优先可读性，ORM 增加学习成本 |

---

## 本讲小结

- `common.py` 提供了项目级的 **UTC 时间规范** 和 **JSON 读写容错**
- `utils.py` 的 `stable_hash()` 是基于 SHA-256 的 **确定性 ID 生成器**
- `json_store.py` 通过 **模板方法模式**（empty_data + normalize_data）提供可复用的 JSON 文件 Store
- `schemas.py` 用 **Pydantic** 定义请求/响应模型，FastAPI 自动校验
- `settings.py` 用 **pydantic-settings** 加载 `.env` 配置，`@lru_cache` 单例
- `memory/base.py` 的 `_MySqlStore` 提供了 **惰性 MySQL 连接** 的公共能力
