# 教学最小路径

这份文档用于授课减法。项目代码保持企业级完整度，但第一遍教学只讲能支撑学生理解和复述 RAG 主链路的内容。

核心原则：

- 平台能力交给 LangSmith，项目代码聚焦业务 RAG。
- 第一遍只讲主链路，不讲所有脚本。
- 先让学生跑通一次可信问答，再解释治理、评测和版本。
- 高级治理只讲 LangSmith 工作流和项目保留的领域指标，不讲自研平台实现。

## 1. 2 小时体验课

目标：让学生知道这个项目解决什么问题，并能看懂一次问答从页面到答案返回的大致路径。

只使用两个场景：

| 场景 | 用途 |
| --- | --- |
| `enterprise_knowledge` | 讲普通企业制度问答，容易理解。 |
| `engineering_project_qa` | 讲复杂资料、表格行、规范冲突和来源引用。 |

课堂流程：

1. 打开问答页，选择 `engineering_project_qa`。
2. 提问：`施工图纸和强制性规范冲突时怎么办？`
3. 观察流式输出、来源引用、命中路径和 Prompt Profile。
4. 只画一张主链路图：页面 -> API -> QAService -> Pipeline -> 检索 -> Prompt -> LLM -> 引用返回。
5. 打开 LangSmith Trace 或状态页核心视图，只看一次问答的 Trace、来源、Prompt Profile 和回归摘要。

本阶段不讲：

- LangSmith Bad Case 沉淀细节。
- 观测与评测平台底层实现。
- 企业 overlay。
- OCR 提升。
- 容量评估。
- 历史报告复盘。
- 全部 8 个场景资料细节。
- 二期 Agent 设计。

## 2. 1 天主链路课

目标：学生能讲清楚一次 RAG 请求为什么能找到正确资料，并生成有边界、有引用的答案。

代码阅读只走这条线：

```text
app.py
  -> qa_core/api/chat.py
  -> qa_core/application/service.py
  -> qa_core/pipeline/rag.py
  -> qa_core/pipeline/steps.py
  -> qa_core/pipeline/retrieval_steps.py
  -> qa_core/retrieval/
  -> qa_core/prompts/
  -> qa_core/indexing/
```

必须讲清楚的 8 个点：

| 主题 | 学生要能回答的问题 |
| --- | --- |
| 意图识别 | 为什么不是所有问题都直接进向量检索？ |
| source 推断 | 为什么要先判断业务分类？ |
| FAQ 直出 | 标准答案为什么要优先于长文档 RAG？ |
| 文档检索 | dense、sparse、rerank 分别解决什么问题？ |
| 上下文构建 | 哪些证据能进入 Prompt，哪些会被过滤？ |
| Prompt Profile | 费用、合规、赔付、安全类问题为什么要更严格？ |
| 来源引用 | 为什么答案必须带来源？ |
| 流式返回 | WebSocket 事件如何让页面逐步显示答案？ |

建议只跑轻量检查：

```powershell
python -m compileall app.py qa_core scripts tests
python -m unittest discover -s tests -p "test_*.py"
python scripts/check_project_guardrails.py
```

## 3. 3 天项目课

目标：把项目从“能跑的 RAG”讲成“可验收、可回归、可维护的企业 RAG”。

推荐节奏：

| 天数 | 主线 | 内容 |
| --- | --- | --- |
| 第 1 天 | RAG 主链路 | 意图、检索、rerank、Prompt、引用、流式输出。 |
| 第 2 天 | 入库与版本 | 文档加载、切分、FAQ/文档入库、kb_version、active 版本切换。 |
| 第 3 天 | 质量闭环 | LangSmith Dataset/Evaluation/Trace、领域指标、入库质量检查、RAG 回归验收。 |

第 3 天只讲结果和机制，不逐行讲所有脚本。学生需要理解这些脚本为什么存在，而不是背下每个参数。

## 4. 功能分层

授课时按下面三层处理。

| 层级 | 范围 | 授课方式 |
| --- | --- | --- |
| 必须掌握 | `qa_core/api`、`qa_core/application`、`qa_core/pipeline`、`qa_core/retrieval`、`qa_core/prompts`、`qa_core/indexing` | 逐步讲主流程，配合一次真实问答。 |
| 必须知道 | 知识库版本、数据隔离、入库质量、LangSmith Trace/Evaluation、领域指标回归验收 | 讲机制和价值，少量看适配层或平台结果。 |
| 选修展示 | enterprise overlay、dirty samples、OCR、容量评估、历史报告复盘、二期 Agent | 只展示入口和用途，面试追问时再展开。 |

## 5. LangSmith 路线下的自研边界

企业授课路线默认不讲自研 LLMOps 平台。自研只保留业务 RAG 的核心差异：

| 保留自研 | 不再主推自研 |
| --- | --- |
| RAG 主链路编排 | trace 存储和查询平台 |
| 业务 source 推断 | 评测实验平台 |
| 数据权限过滤 | Bad Case 人审队列 |
| 知识库版本规则 | Prompt playground |
| FAQ 命中策略 | 评测趋势 Dashboard |
| Prompt Profile 路由 | 大量 Markdown 发布报告 |
| 检索后处理 | 复杂状态页诊断页 |
| 表格/文档解析策略 |  |
| 领域评测指标 |  |

详细边界见 [LangSmith 企业路线与自研边界](langsmith_enterprise_route.md)。

## 6. 对学生的最低要求

完成第一遍后，学生只需要能做到：

1. 跑通一次问答。
2. 解释一次请求的主链路。
3. 找到一次回答的来源引用。
4. 说明 FAQ 直出和文档 RAG 的区别。
5. 说明为什么 Trace 和通用评测平台交给 LangSmith，而领域指标仍由项目定义。

这些完成后，再进入版本治理、Bad Case 和二期扩展。不要要求学生第一遍掌握所有脚本和所有状态页卡片。

## 7. 教师提醒

不要把项目的完整度等同于第一遍授课内容。这个项目的重心应放在业务 RAG 主链路；Trace、Dataset、Evaluation、Annotation 这类平台能力用 LangSmith 承接，课堂入口保持窄，让学生先抓住一条能复述、能调试、能改动的主线。
