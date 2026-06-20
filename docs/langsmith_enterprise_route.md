# LangSmith 企业路线与自研边界

本项目后续采用 **LangSmith 优先** 的企业工程路线：通用平台能力不继续自研，项目自研只保留 RAG 主链路和业务差异化能力。

这个决策的目标是降低学习成本和授课成本，让学生把精力放在企业 RAG 的核心业务工程，而不是学习一套自研版观测、评测和状态页平台。

## 1. 总原则

企业落地时，通用 LLMOps 能力优先交给 LangSmith：

- tracing / span / LLM 调用观测；
- prompt 版本与实验；
- dataset 管理；
- evaluation experiment；
- trace 到 bad case / dataset 的沉淀；
- annotation / 人工复核；
- 评测趋势和实验对比。

本项目自研只保留：

- RAG 主链路编排；
- 业务 source 推断；
- 数据权限过滤；
- 知识库版本规则；
- FAQ 命中策略；
- Prompt Profile 路由；
- 检索后处理；
- 表格/文档解析策略；
- 领域评测指标。

一句话边界：

> LangSmith 负责平台能力，项目代码负责业务 RAG 能力。

## 2. 为什么切到 LangSmith

项目曾经规划过本地观测、评测、Bad Case、趋势报告和状态页诊断能力。当前版本已经切到 LangSmith-first 路线，课程只保留业务 RAG 主链路和领域指标。

继续自研会带来三个问题：

1. 学生需要理解大量非核心平台代码，学习成本高。
2. 授课时容易把重点从 RAG 主链路带偏到报表和状态页。
3. 企业中通常已有 LangSmith、Langfuse、Phoenix、Datadog、Grafana 或内部平台，不会让业务团队从零造一套完整观测评测平台。

切到 LangSmith 后，课程表达更贴近企业：

```text
业务系统
  -> 自研 RAG 主链路
  -> LangSmith tracing / dataset / evaluation / annotation
  -> CI 回归检查读取评测结果
```

## 3. 保留自研的核心业务能力

下面这些能力不能简单交给 LangSmith，因为它们决定项目的业务效果和工程差异化。

| 能力 | 保留原因 |
| --- | --- |
| RAG 主链路编排 | 决定 FAQ、文档 RAG、追问、边界和流式输出如何组合。 |
| 业务 source 推断 | 不同场景的分类规则来自业务资料，不是通用平台能力。 |
| 数据权限过滤 | tenant、dataset、visibility、role 必须进入检索过滤。 |
| 知识库版本规则 | active/staged/rollback 是本项目知识治理核心。 |
| FAQ 命中策略 | 标准答案直出和文档 RAG 分层是项目主设计。 |
| Prompt Profile 路由 | 费用、合规、赔付、安全等风险边界必须由业务规则控制。 |
| 检索后处理 | 去重、parent-child chunk、表格行优先、上下文预算属于 RAG 效果核心。 |
| 表格/文档解析策略 | CSV/XLSX 行级证据、OCR 风险、文档规范化依赖业务资料形态。 |
| 领域评测指标 | source 推断准确率、场景隔离率、FAQ 直出准确率、表格行召回等需要项目自定义。 |

## 4. 迁移范围

优先迁移或瘦身：

| 当前自研能力 | LangSmith 路线 |
| --- | --- |
| `qa_core/observability` trace 存储与查询 | 改为 LangSmith trace，项目只保留 trace metadata 组装。 |
| `/api/admin` 复杂链路诊断 | 状态页改为系统状态 + LangSmith trace 链接。 |
| Bad Case 草稿导出与复核队列 | 使用 LangSmith trace -> dataset / annotation 工作流。 |
| 评测实验和趋势报告 | 使用 LangSmith datasets / experiments，项目保留领域指标 evaluator。 |
| Prompt 调试和版本说明 | 使用 LangSmith prompt/experiment，项目保留 Prompt Profile 选择逻辑。 |
| 本地长篇报告 | 缩减为本地回归摘要，详细结果跳转 LangSmith experiment。 |

暂不迁移：

- 入库质量检查；
- Milvus 集合和 metadata 过滤；
- 知识库版本激活和回滚；
- 表格/文档 loader；
- 场景配置和 source 白名单；
- Prompt Profile 路由代码；
- 核心单元测试。

## 5. 建议迁移步骤

### 阶段一：接入 LangSmith tracing

目标：每次问答在 LangSmith 中能看到完整 trace。

环境变量：

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=请替换为 LangSmith API Key
LANGSMITH_PROJECT=knowforge-rag-platform
```

trace metadata 至少包含：

- `scenario_id`
- `source_filter`
- `kb_version`
- `tenant_id`
- `dataset_id`
- `visibility`
- `user_role`
- `intent`
- `hit_type`
- `prompt_profile`
- `effective_source`

### 阶段二：迁移评测集

目标：把 `eval_sets/*.json` 映射为 LangSmith dataset。

保留项目领域字段：

- `expected_source_contains`
- `expected_hit_type`
- `expected_effective_source`
- `expected_prompt_profile`
- `expected_keywords`

这些字段用于自定义 evaluator，而不是丢掉。

### 阶段三：瘦身状态页和报告

目标：状态页不再复刻 LangSmith。

保留：

- 服务状态；
- active kb_version；
- 入库质量摘要；
- 最近质量检查摘要；
- LangSmith project / trace / experiment 链接。

移除或降级：

- 本地 trace 明细页；
- 本地 evaluation trend；
- 本地 bad case review queue；
- 大量 Markdown 报告生成器。

### 阶段四：清理自研平台代码

目标：删除或归档已被 LangSmith 替代的观测、评测平台代码。

保留少量适配层：

```text
qa_core/observability/langsmith_adapter.py
scripts/langsmith/sync_eval_dataset.py
scripts/langsmith/run_evaluation.py
scripts/langsmith/check_experiment_gate.py
```

## 6. 授课表达

课堂上这样讲：

> 本项目不从零自研 LLMOps 平台。企业里通常使用 LangSmith 这类成熟平台管理 trace、dataset、evaluation 和 annotation。我们自研的重点放在业务 RAG 主链路：source 推断、权限过滤、FAQ 策略、Prompt Profile、表格证据、知识库版本和领域评测指标。

这样学生要学的内容更聚焦：

1. 如何设计企业 RAG 主链路。
2. 如何把业务规则接进检索和 Prompt。
3. 如何把运行过程交给 LangSmith 观测。
4. 如何用 LangSmith dataset/evaluation 做回归。
5. 哪些指标必须由业务系统自定义。

## 7. 代码量预期

迁移完成后，预计可以减少：

- 保守：2500-3500 行；
- 合理目标：3500-5000 行；
- 激进：6000 行以上，但不建议第一阶段追求。

合理目标是删除平台复刻代码，而不是削弱 RAG 主链路。
