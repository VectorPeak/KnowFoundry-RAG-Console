# 知识库多版本管理方案

本文档说明当前项目已经落地的知识库多版本管理能力。该能力覆盖文档入库、FAQ 入库、
在线检索、调试接口、版本切换和回滚。

## 1. 设计边界

以 `qa_core` 新架构为准，多版本管理不是把多份数据写到 MySQL，也不是维护多套
RedisSearch 索引，而是在 Milvus chunk metadata 中写入版本字段，并由本地版本清单控制
当前 active 版本。

核心文件：

- `qa_core/governance/kb_versions.py`：知识库版本清单、创建、激活、归档。
- `qa_core/indexing/`：文档和 FAQ 入库时写入版本 metadata。
- `qa_core/retrieval/filters.py`：检索时追加 `kb_version == active_version` 过滤。
- `qa_core/application/service.py`：问答主链路解析当前版本，并在诊断信息里返回。
- `scripts/rebuild_kb_version.py`：一次性构建完整知识库版本。
- `scripts/manage_kb_versions.py`：命令行查看、创建、激活、归档版本。

## 2. 已写入的版本字段

每条文档 chunk 和 FAQ 记录都会写入：

```json
{
  "kb_version": "kb_20260506_103000_9f2a1b3c",
  "embedding_model_version": "bge-m3-local-v1",
  "reranker_model_version": "bge-reranker-large-local-v1",
  "chunk_schema_version": "parent_child_v1"
}
```

文档 chunk 还会继续保留：

```json
{
  "scenario_id": "enterprise_knowledge",
  "source": "hr",
  "file_name": "onboarding.md",
  "doc_id": "...",
  "parent_id": "...",
  "parent_content": "...",
  "chunk_id": "..."
}
```

FAQ 记录还会继续保留：

```json
{
  "faq_id": "...",
  "scenario_id": "enterprise_knowledge",
  "standard_question": "新人入职需要完成哪些流程？",
  "answer": "以 HR 制度和入职清单为准。",
  "source": "hr"
}
```

## 3. 检索如何按版本隔离

普通用户请求不传 `kb_version`，系统会读取 active 版本：

```text
kb_version == "kb_20260506_103000_9f2a1b3c"
```

如果用户选择业务分类，例如 `source_filter=hr`，最终 Milvus 表达式是：

```text
source == "hr" and kb_version == "kb_20260506_103000_9f2a1b3c"
```

如果评测或灰度请求显式传入历史版本：

```json
{
  "query": "新人入职流程怎么走",
  "source_filter": "hr",
  "kb_version": "kb_20260430_090000_abcd1234"
}
```

则只检索该历史版本，不影响全局 active 版本。

## 4. 标准全量构建命令

推荐使用统一脚本一次性构建完整知识库版本，避免 FAQ 和文档进入不同版本：

```powershell
python scripts/rebuild_kb_version.py --new-version --force --activate --description "2026-05-06 全量重建"
```

该命令会：

1. 创建一个新的 `kb_version`；
2. 把 FAQ CSV 写入该版本；
3. 扫描当前场景的 `scenarios/<scenario_id>/data/<source>_data` 目录，把文档 chunk 写入该版本；
4. 入库完成后激活该版本。

如果只想构建但不立即上线，去掉 `--activate`：

```powershell
python scripts/rebuild_kb_version.py --new-version --force --description "候选版本"
```

## 5. 单独入库命令

只入库文档：

```powershell
python scripts/rebuild_kb_version.py --kb-version kb_20260506_103000_9f2a1b3c --force --skip-faq
```

只入库 FAQ：

```powershell
python scripts/rebuild_kb_version.py --kb-version kb_20260506_103000_9f2a1b3c --skip-docs
```

如果不传 `--kb-version`，脚本会写入当前 active 版本；没有 active 版本时会自动创建第一个版本。

## 6. 查看、激活和回滚

查看版本：

```powershell
python scripts/manage_kb_versions.py list
```

激活某个版本：

```powershell
python scripts/manage_kb_versions.py activate kb_20260506_103000_9f2a1b3c
```

回滚到上一版本质上就是重新激活旧版本：

```powershell
python scripts/manage_kb_versions.py activate kb_20260430_090000_abcd1234
```

归档非 active 版本：

```powershell
python scripts/manage_kb_versions.py archive kb_20260430_090000_abcd1234
```

归档不会删除 Milvus 数据，只改变版本状态。物理清理应单独做，避免误删可回滚数据。

## 7. API 管理接口

查看版本：

```http
GET /api/kb_versions
```

激活版本：

```http
POST /api/kb_versions/{kb_version}/activate
```

归档版本：

```http
POST /api/kb_versions/{kb_version}/archive
```

检索调试时指定版本：

```http
POST /api/retrieval/debug
Content-Type: application/json

{
  "query": "新人入职流程怎么走",
  "source_filter": "hr",
  "kb_version": "kb_20260506_103000_9f2a1b3c"
}
```

## 8. 当前实现的取舍

当前版本切换只修改 `.index_manifest/kb_versions.json`，不会批量更新 Milvus 数据。这是刻意设计：

- 切换速度快；
- 旧版本可回滚；
- 新旧版本可同时评测；
- 不需要为 active 状态重写所有 chunk。

如果后续进入多节点部署，应把版本清单迁移到 MySQL 或对象存储元数据表，并增加入库锁。

## 9. 版本召回对比

新增知识库版本后，可以先跑召回对比，再决定是否激活或是否保留 active：

```powershell
python scripts\kb\compare_kb_versions.py --scenario engineering_project_qa --dataset eval_sets\table_regression.json --limit 1 --base-version <旧版本> --candidate-version <新版本> --output reports\verification\kb_version_compare_engineering_latest.json
```

报告重点看：

- `candidate_recall_at_k` 是否低于旧版本；
- `regression_count` 是否大于 0；
- `top_source_changed_count` 是否异常升高；
- 每条样本的 `base_expected_rank` 与 `candidate_expected_rank` 是否退化。

该脚本复用 `QAService.debug_retrieval()`，不会绕过现有意图识别、source 过滤、版本过滤、
Milvus Hybrid 和 rerank。

## 10. 注意事项

1. 更换 embedding 模型后必须创建新版本并重新入库。
2. 修改 chunk_size、overlap、父子块策略后必须创建新版本并重新入库。
3. 普通在线请求不建议传 `kb_version`，由 active 版本统一控制。
4. 评测、灰度、历史回放可以显式指定 `kb_version`。
5. 归档不等于删除；删除历史版本数据需要单独清理 Milvus。

