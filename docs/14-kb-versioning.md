# 第14讲：知识库多版本管理

**上一讲**：[应用入口与环境前置校验](./13-app-entry-preflight.md)  
**下一讲**：[数据隔离与多租户设计](./15-data-isolation.md)

## 本讲目标

- 理解为什么 RAG 系统需要知识库版本管理
- 掌握版本状态机的设计（STAGED → ACTIVE → ARCHIVED）
- 理解版本切换的实现（O(1) 操作，不批量更新 Milvus）
- 理解版本号生成的设计（时间戳 + 配置哈希）

---

## 第一部分：前置知识 — 为什么 RAG 需要版本管理

### 1.1 不加版本管理的风险

假设你用一个脚本把 500 个业务文档写入 Milvus。运行完毕后：

```
场景 A：一切正常，问答效果好

场景 B：你发现新的 Embedding 模型（bge-m3-v2）效果更好，想切换
  → 重新入库，覆盖了旧的向量数据
  → 新模型效果反而更差（新的切分策略导致 chunk 太碎）
  → 无法回滚，因为旧数据已经被覆盖了

场景 C：你修改了文档切分参数（chunk_size 从 500 改为 300）
  → 想对比新旧切分方案的效果
  → 没有版本机制，你只能删除旧数据重建，对比无从谈起
```

**版本管理的核心价值**：让知识库的更新成为**可逆操作**。

### 1.2 版本管理的典型需求

| 需求 | 说明 |
|------|------|
| 安全入库 | 新版本先写入，不影响线上正在使用的版本 |
| 灰度验证 | 新版本入库后先评测，确定没问题再切换 |
| 快速回滚 | 新版本效果不好，一键切回旧版本 |
| 对比评测 | 同一个问题可以分别在新旧版本上验证召回效果 |
| 长期保留 | 历史版本不删除，作为 A/B 测试和故障分析的依据 |

---

## 第二部分：版本状态机

### 2.1 三种状态

```mermaid
stateDiagram-v2
    [*] --> STAGED : 入库完成

    STAGED --> ACTIVE : 激活版本<br/>仅修改 JSON 文件
    STAGED --> ARCHIVED : 直接归档<br/>从未激活的版本

    ACTIVE --> STAGED : 回滚/新版本激活<br/>旧 ACTIVE 降级
    ACTIVE --> ARCHIVED : 长期不用后归档

    ARCHIVED --> [*] : Milvus 数据保留<br/>可手动清理

    note right of ACTIVE
        在线检索表达式：
        kb_version == "active_version"
        同一场景只有一个 ACTIVE
    end note

    note right of STAGED
        已写入 Milvus
        评测验证中
        用户检索不可见
    end note
```

### 安全入库与激活流程

```mermaid
flowchart TD
    Start(["执行 rebuild_kb_version.py"]) --> Create["1️⃣ 创建 STAGED 版本<br/>版本号含时间戳+配置哈希"]

    Create --> Ingest["2️⃣ 入库存入 STAGED<br/>FAQ 入库 + 文档入库<br/>写入 kb_version=新版本号"]

    Ingest --> Report["3️⃣ 生成入库质量报告<br/>解析失败/低质量chunk/<br/>FAQ空值/冲突检测"]

    Report --> Gate{"4️⃣ 入库质量检查检查"}

    Gate -->|"✅ 通过"| Activate["5️⃣ 激活版本<br/>修改 JSON: active_version=新版本<br/>旧 ACTIVE 降为 STAGED<br/>Milvus 数据不更新"]

    Gate -->|"❌ 不通过"| Abort["❌ 终止激活<br/>STAGED 版本保留<br/>线上仍用旧 ACTIVE<br/>用户无感知"]

    Activate --> Online["✅ 新版本上线<br/>下一次检索自动使用<br/>expr: kb_version==新版本号"]

    style Create fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style Gate fill:#FFFBEB,stroke:#D97706,stroke-width:2px
    style Activate fill:#ECFDF5,stroke:#059669,stroke-width:2px
    style Abort fill:#FEF2F2,stroke:#DC2626,stroke-width:2px
    style Online fill:#ECFDF5,stroke:#059669,stroke-width:3px
```

- **STAGED**：版本已写入 Milvus，但线上检索不使用。通常用于新入库的版本，等待评测验证。
- **ACTIVE**：当前在线检索使用的版本。同一场景只有一个 ACTIVE 版本。
- **ARCHIVED**：不再使用的历史版本。数据和 Milvus chunk 都保留，但不参与在线检索。

### 2.2 状态转换实现

下面的代码展示的是步骤 5（激活）和归档操作。步骤 1（创建版本）见 `ensure_version()`，步骤 2（入库写入）见第 15 讲，步骤 3-4（质量报告和门控）见第 16 讲。

```python
# qa_core/governance/kb_versions.py

def activate_version(self, kb_version: str) -> KnowledgeBaseVersion:
    """把指定版本切为当前在线检索版本。

    激活只修改版本清单 JSON 文件，不更新 Milvus 数据。
    """  # → 流程图节点 5️⃣：激活版本
    self.reload()  # 重新加载，避免内存中的旧数据覆盖磁盘
    record = self.get(kb_version)
    if record is None:
        raise ValueError(f"知识库版本不存在：{kb_version}")

    previous = str(self.data.get("active_version") or "")
    now = utc_now()

    # 更新所有版本的状态
    for vid, raw in self.data["versions"].items():
        item = KnowledgeBaseVersion.from_dict(raw)
        if vid == kb_version:
            item.status = "ACTIVE"
            item.activated_at = now
        elif item.status == "ACTIVE":
            item.status = "STAGED"  # 旧 active 降级为 STAGED
        self.data["versions"][vid] = item.as_dict()

    # 更新清单的指针
    self.data["previous_version"] = previous if previous != kb_version else ...
    self.data["active_version"] = kb_version
    self.save()  # 写入磁盘
    return self.get(kb_version) or record


def archive_version(self, kb_version: str) -> KnowledgeBaseVersion:
    """归档一个非 active 版本。

    归档不会删除 Milvus 数据，只是状态标记。
    """  # → 流程图未展示的附加操作：归档
    if self.active_version_candidate() == kb_version:
        raise ValueError("不能归档当前 active 知识库版本")

    record = self.get(kb_version)
    record.status = "ARCHIVED"
    record.archived_at = utc_now()
    self.data["versions"][kb_version] = record.as_dict()
    self.save()
    return record
```

### 2.3 激活操作的轻量性

**关键设计**：激活版本只修改一个 JSON 文件，不碰 Milvus。

```
激活前的在线检索：
  Milvus expr 包含 kb_version == "v1"

激活后的在线检索：
  Milvus expr 包含 kb_version == "v2"
  v1 的 chunk 数据仍在 Milvus 中，只是不再被查到
```

如果激活需要修改所有 chunk 的 metadata，一个 10 万条 chunk 的知识库需要很长时间。通过把版本切换放在检索表达式中，版本切换变成了 O(1) 操作。

---

## 第三部分：版本号设计

### 3.1 版本号生成

```python
def generate_kb_version(prefix="kb", scenario_id=None) -> str:
    """生成一个适合人读和机器过滤的知识库版本号。

    包含：
    - 时间戳：便于肉眼判断版本先后
    - 配置短 hash：把 embedding、reranker、chunk_schema、collection
      等关键配置纳入标识
    """
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)  # 私有 helper，避免 settings -> scenarios -> kb_versions 循环导入

    stamp = utc_file_stamp()  # 如 20260506_103000（年月日_时分秒）
    config_hash = stable_hash(
        scenario.scenario_id,
        settings.embedding_model_version,    # 如 "bge-m3-local-v1"
        settings.reranker_model_version,     # 如 "bge-reranker-v1"
        settings.chunk_schema_version,       # 如 "parent_child_v1"
        scenario.doc_collection,
        scenario.faq_collection,
    )[:8]  # 只取前 8 位

    return f"{prefix}_{scenario.scenario_id}_{stamp}_{config_hash}"
    # 例：kb_enterprise_knowledge_20260506_103000_9f2a1b3c
```

### 3.2 为什么版本号包含配置哈希

设计意图：从版本号可以直接判断两个版本是否使用同一套配置。

```
kb_enterprise_knowledge_20260506_103000_9f2a1b3c
kb_enterprise_knowledge_20260507_150000_7d3e8f1a
                           不同日期 ↑         不同 hash ↑
```

如果两个版本的 hash 相同但日期不同，说明是同一套配置下的数据更新（新增/修改了文档）。
如果 hash 不同，说明 Embedding 模型、Reranker 模型或 Chunk 方案有变化，需要重点关注召回质量的对比。

---

## 第四部分：版本清单文件

### 4.1 文件结构

```json
{
    "scenario_id": "enterprise_knowledge",
    "active_version": "kb_enterprise_knowledge_20260507_150000_7d3e8f1a",
    "previous_version": "kb_enterprise_knowledge_20260506_103000_9f2a1b3c",
    "versions": {
        "kb_enterprise_knowledge_20260506_103000_9f2a1b3c": {
            "kb_version": "kb_enterprise_knowledge_20260506_103000_9f2a1b3c",
            "status": "STAGED",
            "created_at": "2026-05-06T10:30:00Z",
            "activated_at": "2026-05-06T10:35:00Z",
            "doc_collection": "enterprise_knowledge_doc",
            "faq_collection": "enterprise_knowledge_faq",
            "embedding_model_version": "bge-m3-local-v1",
            "reranker_model_version": "bge-reranker-v1",
            "chunk_schema_version": "parent_child_v1",
            "sources": ["hr", "it", "admin"],
            "stats": {
                "last_doc_count": 1523,
                "last_faq_count": 85,
                "total_doc_written": 1523,
                "total_faq_written": 85
            }
        },
        "kb_enterprise_knowledge_20260507_150000_7d3e8f1a": {
            "status": "ACTIVE",
            ...
        }
    }
}
```

### 4.2 KnowledgeBaseVersionStore 类

```python
class KnowledgeBaseVersionStore(JsonFileStore):
    """知识库版本清单读写器。

    每个场景有独立的版本清单文件：
    .index_manifest/<scenario_id>/kb_versions.json

    这样多场景教学项目可以为每个行业场景独立激活、
    回滚和对比知识库版本。
    """

    def __init__(self, path=None, scenario_id=None):
        self.scenario = _resolve_version_scenario(scenario_id)
        super().__init__(path or self.scenario.kb_versions_manifest_path)

    def resolve_active_version(self, requested=None) -> str:
        """解析一次检索应该使用的知识库版本。

        优先级：
        1. 请求显式传入的 kb_version（用于评测/灰度）
        2. 环境变量 ACTIVE_KB_VERSION
        3. 版本清单中的 active_version

        当前在线检索必须带版本过滤。没有 active 版本直接报错。
        """
        if requested:
            if not self.exists(requested):
                raise ValueError(f"请求的知识库版本不存在：{requested}")
            return requested

        active = self.active_version_candidate()
        if not active:
            raise ValueError(
                f"场景 {self.scenario.scenario_id} 没有 active 知识库版本"
            )
        if not self.exists(active):
            raise ValueError(
                f"active 知识库版本不存在于版本清单：{active}"
            )
        return active
```

---

## 第五部分：与 Milvus 检索的集成

### 5.1 写入时携带版本信息

```python
# qa_core/governance/kb_versions.py
def version_metadata(kb_version, scenario_id=None):
    """构建写入每个 FAQ/chunk metadata 的版本字段。"""
    settings = get_settings()
    scenario = _resolve_version_scenario(scenario_id)
    return {
        "scenario_id": scenario.scenario_id,
        "kb_version": kb_version,
        "embedding_model_version": settings.embedding_model_version,
        "reranker_model_version": settings.reranker_model_version,
        "chunk_schema_version": settings.chunk_schema_version,
    }
```

每条 FAQ 和 chunk 入库时，这些字段都会被写入 metadata：

```python
chunk = Document(
    page_content="入职流程包括以下步骤...",
    metadata={
        "source": "hr",
        "chunk_id": "abc123",
        "kb_version": "kb_enterprise_knowledge_20260507_150000_7d3e8f1a",
        "embedding_model_version": "bge-m3-local-v1",
        ...
    }
)
```

### 5.2 检索时过滤版本

```python
# 在线检索
active_version = resolve_active_kb_version(requested, scenario.scenario_id)
# active_version = "kb_enterprise_knowledge_20260507_150000_7d3e8f1a"

# 拼入 Milvus 表达式
expr = f'kb_version == "{active_version}" and source == "hr" and ...'

# 只检索 active 版本的 chunk
results = milvus_store.similarity_search(query, expr=expr)
```

### 5.3 评测用历史版本

```python
# 评测脚本可以显式指定历史版本
service.debug_retrieval(
    query="入职流程有哪些步骤",
    kb_version="kb_enterprise_knowledge_20260506_103000_9f2a1b3c",  # 旧版本
    ...
)

# 对比两个版本对同一批问题的召回效果
for question in eval_set:
    old_result = service.debug_retrieval(query=question, kb_version=old_version)
    new_result = service.debug_retrieval(query=question, kb_version=new_version)
    compare(old_result, new_result)
```

---

## 第六部分：全量重建的安全流程

```bash
# 完整的安全入库 + 激活流程
python scripts/rebuild_kb_version.py \
    --scenario enterprise_knowledge \
    --new-version --force \
    --quality-gate --activate
```

执行顺序：

```
1. 创建 STAGED 版本（version = "kb_...20260507_150000_xxxx"）
2. FAQ 入库（写入 STAGED 版本的 kb_version）
3. 文档入库（写入 STAGED 版本的 kb_version）
4. 生成入库质量报告
5. 执行入库质量检查
   ├─ 通过 → 激活版本（将 STAGED 切换为 ACTIVE）
   └─ 不通过 → 终止流程，STAGED 版本仍保留（不激活）
```

**关键安全点**：即使新的 STAGED 版本已经写入了 Milvus，只要没有执行激活步骤，线上检索仍然使用旧的 ACTIVE 版本。用户完全无感知。

---

## 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | 版本状态机：STAGED（已入库待验证）→ ACTIVE（在线检索使用）→ ARCHIVED（归档保留） | 知识库版本管理的核心模型 |
| ★★★ 必会 | O(1) 版本切换：激活只修改 JSON 文件的 active_version 指针，不更新 Milvus 数据 | 理解版本切换为什么是即时的，面试高频 |
| ★★★ 必会 | 版本号设计：`kb_{scenario_id}_{timestamp}_{config_hash}`，配置哈希体现版本差异 | 从版本号直接判断配置是否变化 |
| ★★ 理解 | resolve_active_version() 的三级优先级：请求传入 > 环境变量 > 版本清单 active_version | 理解版本解析流程 |
| ★★ 理解 | version_metadata() 将版本信息写入每个 chunk 的 metadata，检索时通过 expr 过滤 | 写入和检索的版本关联机制 |
| ★★ 理解 | 安全入库流程：创建 STAGED → 入库 → 生成质量报告 → 检查 → 激活 | 生产环境的完整安全操作 |
| ★ 了解 | 版本清单 JSON 文件结构 | 了解持久化格式 |
| ★ 了解 | 评测时可以显式指定历史版本做对比 | 了解版本管理的扩展用途 |

## 本讲小结

- **版本管理让知识库更新成为可逆操作**：入库 → 评测 → 激活 →（效果不好）→ 回滚
- **版本状态机**：STAGED（待验证）→ ACTIVE（在线使用）→ ARCHIVED（归档保留）
- **版本切换是 O(1) 操作**：只修改 JSON 文件中的 active_version，不更新 Milvus 数据
- **版本号** = 时间戳 + 配置哈希，可以肉眼判断先后和配置差异
- **每个场景独立版本清单**，不同行业场景可以独立管理知识库版本
- **评测脚本可以显式指定历史版本**，实现新老版本对比

**下一讲**：[数据隔离与多租户](./15-data-isolation.md) — 租户/数据集/角色隔离、Milvus 表达式过滤
