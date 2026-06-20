# 旧版链路退场治理说明

本文档只保留旧版链路的架构问题和退场结论。旧版 `mysql_qa`、`rag_qa`、`old_main.py`、
`new_main.py` 等代码已经从工程目录中移除，不再作为可运行代码保留。

## 1. 当前结论

当前唯一主链路是：

```text
app.py
  -> qa_core.application.factory.get_qa_service
  -> qa_core.application.service.QAService
  -> qa_core.intent / qa_core.pipeline / qa_core.retrieval.strategy
  -> qa_core.retrieval.store.MilvusHybridStore
  -> qa_core.memory.history.HistoryStore
```

旧链路不再保留运行入口，也不再提供环境变量开关恢复运行。这样做是为了降低学习成本，
避免新同学误以为项目同时存在两套可选架构。

## 2. 旧版链路为什么退场

旧版链路大致是：

```text
old_main.py / new_main.py
  -> mysql_qa.retrieval.BM25Search
  -> Redis 缓存
  -> MySQL FAQ 表
  -> rag_qa.core.RagSystem
  -> BERT QueryClassifier
  -> LLM StrategySelector
  -> rag_qa.core.VectorStore
  -> 旧版 Milvus schema
```

主要问题：

- MySQL、Redis、Milvus 三套知识状态需要同步，容易不一致。
- 本地 `rank_bm25 + jieba` 已被 Milvus 2.6.x 内置 BM25 替代。
- RedisSearch 会引入第二套索引，不符合当前核心链路简化目标。
- 旧版手写 Milvus schema 与当前 LangChain Milvus 集合不一致。
- BERT 分类器和 LLM 策略选择器增加启动成本和不可控性。
- 旧版流式输出不是当前页面使用的 WebSocket 事件协议。
- 旧历史表与当前 LangChain `SQLChatMessageHistory` 不是同一套。
- 旧文档入库没有 `IndexManifest` 增量清单，容易重复或保留过期 chunk。

因此，旧版代码的正确方向是删除，不是继续兼容或继续功能优化。

## 3. 保留什么

保留的是“架构演进说明”，不是旧代码：

- 优化前问题分析归档：`docs/archive/optimization_implementation.md`
- 当前架构文档：`docs/PROJECT_ARCHITECTURE.md`
- 当前流程图：`docs/current_architecture_flow.md`

如果后续确实需要迁移历史数据，应通过当前脚本入口处理，例如：

```powershell
python scripts/rebuild_kb_version.py --faq-csv <faq.csv> --data-dir <data_dir> --new-version --force --quality-gate --activate
```

不要再恢复旧 BM25、旧 RedisSearch 或旧 RAG 编排器。

## 4. 验收方式

当前主链路验收：

```powershell
python -m compileall app.py qa_core scripts tests
python -m pytest tests -q
python scripts/check_langchain_stack.py
```

目录边界验收：

```powershell
Get-ChildItem
```

预期结果：根目录只保留当前主链路、配置、文档、脚本、场景包、模型和测试目录，不再出现旧版代码目录或旧版入口文件。
