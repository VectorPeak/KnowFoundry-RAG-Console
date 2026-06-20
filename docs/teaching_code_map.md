# 课堂代码地图

这份地图用于控制课堂阅读范围。第一遍学习不要从目录树开始扫代码，只按每讲的“必读文件”进入。

## P0 主链路

| 讲次 | 主题 | 必读文件 | 只需扫读 |
|---|---|---|---|
| 03 | 意图分类 | `qa_core/intent/classifier.py` | `qa_core/scenarios/boundary.py` |
| 04 | 检索策略 | `qa_core/retrieval/strategy.py` | `qa_core/retrieval/results.py` |
| 05 | 查询改写 | `qa_core/pipeline/rewrite.py`, `qa_core/pipeline/query_variants.py` | `qa_core/memory/history.py` |
| 06 | Milvus 混合检索 | `qa_core/retrieval/store.py`, `qa_core/retrieval/factory.py` | `qa_core/retrieval/milvus_compat.py` |
| 07 | QAService 编排 | `qa_core/application/service.py` | `qa_core/api/chat.py` |
| 08 | RAG Pipeline | `qa_core/pipeline/rag.py`, `qa_core/pipeline/runtime.py` | `qa_core/pipeline/context.py`, `qa_core/pipeline/citations.py` |
| 09 | Prompt Profile | `qa_core/prompts/profiles.py`, `qa_core/prompts/selector.py`, `qa_core/prompts/templates.py` | `qa_core/intent/question_category.py` |

## P1 工程能力

| 讲次 | 主题 | 必读文件 | 只需扫读 |
|---|---|---|---|
| 10 | 应用入口与前置校验 | `qa_core/config/preflight.py`, `qa_core/config/settings.py` | `scripts/check_langchain_stack.py` |
| 11 | LangChain 生态 | `qa_core/llm/client.py`, `qa_core/retrieval/models.py` | `qa_core/retrieval/store.py` |
| 12 | 知识库版本 | `qa_core/governance/kb_versions.py`, `scripts/rebuild_kb_version.py` | `qa_core/api/kb_versions.py` |
| 13 | 数据隔离 | `qa_core/governance/data_scope.py`, `qa_core/retrieval/filters.py` | `docs/data_scope_isolation.md` |
| 14 | 文档入库 | `qa_core/indexing/service.py`, `qa_core/indexing/document_loaders.py`, `qa_core/indexing/chunking.py` | `qa_core/indexing/table_documents.py` |
| 15 | RAG 回归验收与入库质量 | `scripts/check_evaluation_gate.py`, `scripts/check_ingestion_quality_gate.py` | `scripts/evaluate_core_chain.py` |
| 16 | 测试与接口验收 | `tests/test_quality_gates.py`, `tests/test_retrieval_and_prompt.py` | `scripts/check_project_guardrails.py` |

## P2 企业增强

| 讲次 | 主题 | 必读文件 | 只需扫读 |
|---|---|---|---|
| 17 | LangSmith 观测与 Trace | `qa_core/observability/langsmith_adapter.py`, `qa_core/pipeline/runtime.py` | `qa_core/api/admin.py`, `static/admin.html` |

## P1 Web 服务基础设施

| 讲次 | 主题 | 必读文件 | 只需扫读 |
|---|---|---|---|
| 11 | FastAPI 与异步 | `app.py`, `qa_core/api/chat.py` | `qa_core/api/pages.py` |
| 12 | 应用入口与前置校验 | `app.py`, `qa_core/config/preflight.py` | `qa_core/config/settings.py` |

## 不建议第一遍逐行读

| 范围 | 原因 |
|---|---|
| `scripts/tools/check_local_runtime.py` | 本地排障分支多，适合遇到环境问题时查。 |
| `scripts/enterprise_overlay/` | 企业资料增强专题，不影响 P0 主链路。 |
| `scripts/ocr/` | OCR 依赖和异常场景多，适合作为复杂资料附录。 |
| `scripts/kb/compare_all_kb_versions.py` | 封版前批量检查工具，先理解单场景版本规则即可。 |

## 第一遍推荐阅读顺序

```text
qa_core/application/service.py
  -> qa_core/pipeline/rag.py
  -> qa_core/retrieval/strategy.py
  -> qa_core/retrieval/store.py
  -> qa_core/prompts/selector.py
  -> scripts/rebuild_kb_version.py
  -> scripts/evaluate_core_chain.py
  -> scripts/api_e2e_smoke.py
```

这条线能覆盖“用户提问如何变成一次可靠、可追踪、可回滚的 RAG 回答”。
