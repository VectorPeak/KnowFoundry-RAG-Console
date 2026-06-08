# 第19讲：LangSmith 观测与 Trace

**上一讲**：[测试与接口验收](./18-testing-system.md)

## 本讲目标

- 理解 RAG 系统为什么必须具备可观测性
- 掌握 LangSmith Trace 的核心字段和业务 metadata 设计
- 理解 Bad Case 如何通过 LangSmith annotation/dataset 沉淀
- 了解项目如何通过轻量 adapter 接入 LangSmith，而非自建 LLMOps 平台

---

## 第一部分：前置知识 — 可观测性的三个支柱

### 1.1 什么是可观测性

在 RAG 系统中，用户说"答案不对"，单看最终回答无法判断问题出在哪里：

```
答案不对的 5 种可能原因：
1. 意图识别错了（本该 FAQ 直出，却走了文档 RAG）
2. source 推断错了（该搜 HR 文档却搜了 IT 文档）
3. 检索召回了无关内容（Embedding 或 BM25 失效）
4. 上下文构建截断了关键信息（max_context_chars 太小）
5. LLM 生成了幻觉（Prompt 约束不够）
```

没有可观测数据，你只能**猜测**。有了可观测数据，你可以**定位**。

### 1.2 本项目的可观测架构

本项目采用 **LangSmith 委托** 架构——业务代码只负责写入领域 metadata，存储、查询、可视化、评测和标注全部由 LangSmith 平台完成。

```
RAG 运行时                     LangSmith 平台
┌──────────────────┐          ┌─────────────────────┐
│ record_query_trace│  ───→   │ Trace 存储 + 过滤    │
│ (147行 adapter)   │  metadata│ Dataset 管理         │
│                  │          │ Evaluation 自动评分   │
│ langsmith_status()│          │ Annotation 人工标注   │
└──────────────────┘          └─────────────────────┘
```

**核心原则**：项目不复刻 LLMOps 平台。自研追踪存储、状态页 Dashboard 和评测 UI 属于平台级工程，超出 RAG 教学项目的范围。企业路线下，LangSmith 提供成熟的 tracing、dataset、evaluation 和 annotation 能力，项目只负责写入业务 metadata。

---

## 第二部分：langsmith_adapter.py — 核心代码

项目中与 LangSmith 交互的唯一模块是 `qa_core/observability/langsmith_adapter.py`（147 行）。它包含四个函数：

### 2.1 环境配置：configure_langsmith_environment()

```python
# qa_core/observability/langsmith_adapter.py
def configure_langsmith_environment() -> None:
    settings = get_settings()
    os.environ.setdefault("LANGSMITH_TRACING", "true" if settings.langsmith_tracing else "false")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
```

在 `app.py` 启动时调用一次，将 Pydantic Settings 中的 LangSmith 配置写入环境变量，使 LangChain 集成能自动感知。

### 2.2 开关检测：langsmith_enabled()

```python
def langsmith_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)
```

所有 trace 写入操作的守卫。LangSmith 未启用时直接跳过，不影响请求主链路。

### 2.3 状态查询：langsmith_status()

返回轻量状态字典供状态页使用，包含 `provider`、`enabled`、`project`、`endpoint` 和项目 URL。

### 2.4 核心函数：record_query_trace()

```python
def record_query_trace(
    *,
    trace_id: str,
    session_id: str,
    question: str,
    answer: str,
    hit_type: str,
    scenario,
    data_scope: dict[str, Any],
    source_filter: str | None,
    kb_version: str,
    rewritten_query: str | None,
    intent: dict[str, Any] | None,
    retrieval: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    elapsed_ms: float,
    error: str | None = None,
) -> None:
```

**执行流程**：

1. `configure_langsmith_environment()` — 确保环境变量已注入
2. `langsmith_enabled()` 检查 → 未启用直接返回
3. 构建 `metadata` 字典（18 个业务字段，见下方）
4. 构建 `inputs`（question / scenario_id / source_filter / kb_version）
5. 构建 `outputs`（answer_preview[:800] / hit_type / sources / error）
6. 通过 `langsmith.run_helpers.trace()` 上下文管理器写入 LangSmith
7. 异常只记日志不抛出——trace 写入失败不影响用户请求

### 2.5 写入 LangSmith 的业务 metadata

```python
metadata = {
    "trace_id": trace_id,           # 与项目内部 trace_id 一致，可跨系统关联
    "session_id": session_id,
    "scenario_id": ...,             # 当前业务场景
    "scenario_name": ...,
    "source_filter": ...,           # 前端选择的业务分类
    "effective_source": ...,        # 最终生效的 source 过滤
    "kb_version": ...,              # 知识库版本号
    "tenant_id": ...,               # 数据隔离四维
    "dataset_id": ...,
    "visibility": ...,
    "user_role": ...,
    "intent": ...,                  # 意图分类结果
    "intent_reason": ...,           # 意图判断原因（rule / llm_structured）
    "hit_type": ...,                # faq_direct / rag / insufficient_context
    "prompt_profile": ...,          # 使用的 Prompt 档位
    "question_category": ...,       # 风险类别（pricing / compliance / ...）
    "sources_count": ...,           # 引用来源数量
    "top_source_score": ...,        # 最高召回分数
    "first_token_ms": ...,          # 首 token 延迟
    "stage_timings_ms": ...,        # 各阶段耗时
    "slowest_stage": ...,           # 最慢阶段
    "elapsed_ms": ...,              # 总耗时
    "error": ...,                   # 错误信息（如有）
}
```

**设计要点**：

- **metadata 不存完整 prompt/上下文**——敏感资料（合同条款、薪酬信息）不应进入外部平台
- **trace_id 使用项目 UUID**——可在 LangSmith UI 中搜索 `trace_id` 直接定位
- **tags** 自动包含 `scenario_id` 和 `hit_type`，支持在 LangSmith 中按场景和命中路径过滤

---

## 第三部分：Bad Case 沉淀

### 3.1 LangSmith 闭环流程

```
全量 Trace ──→ 过滤（error / no_sources / low_score）──→ Annotation（人工标注）
                                                              │
                                                              ↓
                    Gate 验收 ←── Evaluation（自动评分）←── Dataset（评估集）
```

1. **Trace 发现**：在 LangSmith 中按 `hit_type=insufficient_context` 或 `sources_count=0` 过滤失败案例
2. **Annotation 标注**：为失败案例标注 `expected_output` 和 `is_correct`，加入 Dataset
3. **Dataset 管理**：每个场景维护一个回归评估集，新增标注后自动触发 Evaluation
4. **Gate 验收**：在 CI 或发版前跑 Evaluation，对比基线分数判断是否退化

### 3.2 与自建闭环的对比

| | LangSmith 平台 | LangSmith 委托 |
|---|---|---|
| Trace 存储 | 本地文件，需自行管理轮转 | 云端，自动保留 |
| 检索/过滤 | grep + 自建状态页 API | LangSmith UI 多条件过滤 |
| 标注 | 需自建标注界面 | LangSmith Annotation Queue |
| 评估 | 需自建评估跑分 | LangSmith Evaluation 自动跑 |
| 适用场景 | 教学演示底层机制 | 企业真实项目 |

本项目的教学定位：**讲 LangSmith 企业闭环为主线，JSonL 底层原理只作为概念类比。**

---

## 第四部分：RAGQueryContext 中的 trace 调用

trace 不只是在问答结束时写一次，而是在整个 Pipeline 生命周期中逐步累积数据。调用链：

```
app.py 启动时
  └── configure_langsmith_environment()   ← 注入环境变量

Pipeline 执行中（qa_core/pipeline/runtime.py）
  └── RAGQueryContext.run_stage()         ← 每个阶段自动计时
  └── RAGQueryContext.retrieval_info      ← 累积检索诊断数据
  └── RAGQueryContext.mark_first_token()  ← 记录首 token 时刻

Pipeline 结束时（qa_core/pipeline/rag.py）
  └── finish_success() / finish_error()
      └── RAGQueryContext.record_trace()
          └── record_query_trace()        ← 汇总所有数据写入 LangSmith
```

`RAGQueryContext.run_stage()` 是阶段自动计时的关键：它执行回调并记录 `time.perf_counter()` 差值，最终汇总为 `stage_timings_ms`。

---

## 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | `record_query_trace()` 的 18 个业务 metadata 字段 | 面试常问"你往 trace 里记了什么" |
| ★★★ 必会 | LangSmith trace → annotation → dataset → evaluation 闭环 | 体现项目有RAG 回归与入库质量体系 |
| ★★ 理解 | `configure_langsmith_environment()` 的环境变量注入机制 | 理解 LangChain 集成如何感知 LangSmith |
| ★★ 理解 | `RAGQueryContext.run_stage()` 的阶段计时方式 | 能解释"怎么知道哪个阶段最慢" |
| ★ 了解 | 自建追踪存储方案的设计思路 | 作为系统设计题的备选方案 |

---

## 本讲小结

- **项目不复刻 LLMOps 平台**：trace 存储、过滤、可视化、标注和评估全部委托给 LangSmith
- **147 行 adapter** 是项目中与 LangSmith 交互的唯一代码
- **18 个 metadata 字段** 覆盖了场景、数据隔离、检索策略、耗时和错误信息
- **LangSmith 闭环**：Trace 发现 → Annotation 标注 → Dataset 沉淀 → Evaluation 回归 → Gate 验收
