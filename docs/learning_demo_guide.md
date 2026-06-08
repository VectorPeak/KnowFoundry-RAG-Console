# 学习与演示总入口

本文档给学生和面试演示使用。它不重复替代 README，而是告诉你按什么顺序理解、运行和讲解这个项目。

## 1. 学习顺序

两种学习方式，按需选择：

**先做减法**

第一次学习只看“主链路 + LangSmith 验收链路”。企业资料治理、LangSmith Bad Case 沉淀、
overlay 激活、OCR 提升和性能检查都属于高级能力，先知道它们存在即可，不要一开始就深入。

如果用于授课，先按 [教学最小路径](TEACHING_MINIMAL_PATH.md) 组织内容：2 小时体验课只讲两个场景和一条主链路，1 天课程再补入库、Prompt 和检索细节，3 天课程最后讲质量闭环。

**方式一：系统学习（推荐零基础学生）**

从 [18 讲系统课程](course-outline.md) 开始，按 01 → 18 顺序学习。每讲包含前置知识讲解、Mermaid 图解、代码详解和预计学习时间。

**方式二：快速浏览（适合有经验的开发者）**

建议按下面顺序学习，不要一上来就看所有代码：

1. 先看项目根目录的 `course-outline.md`，理解项目定位、8 个业务场景和启动方式。
2. 再看 [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)，理解从浏览器提问到答案返回的主链路。
3. 然后看 [standard_demo_runbook.md](standard_demo_runbook.md)，按 10 到 15 分钟标准路径完成一次可复用演示。
4. 再看 [evaluation_and_quality.md](evaluation_and_quality.md)，理解为什么这个项目不是只靠页面演示，而是有评测和回归验收。
5. 接着看 [phase1_quick_review.md](phase1_quick_review.md)，用最短路径确认一期 RAG 是否具备交付闭环。
6. 然后看 [resume_project_pack.md](resume_project_pack.md)，学习怎么把同一套平台包装成不同业务背景的简历项目。
7. 最后看 [phase2_agent_boundary.md](phase2_agent_boundary.md)，理解二期 Agent 为什么不直接塞进一期 RAG 主链路。

> 📖 **教师备课速查**：如果你是有经验的开发者想快速浏览全貌，可以看 [`lecture_notes.md`](lecture_notes.md)（13 章凝练版，含教学要点提示），然后挑薄弱讲次精读对应的 18 讲。

## 2. 演示路径

一次完整演示推荐控制在 10 到 15 分钟：

1. 打开问答页：`http://127.0.0.1:8000/`
2. 切换到 `engineering_project_qa`，提问“施工图纸和强制性规范冲突时怎么办？”
3. 展示流式输出、来源引用、命中路径和 Prompt Profile。
4. 切换到 `insurance_claims`，提问“收款账户和被保险人不一致可以打款吗？”
5. 说明费用/赔付类问题为什么不能让模型自由承诺。
6. 打开 LangSmith 项目，查看本次问答 trace、metadata、LLM 调用和检索阶段。
7. 再打开状态页核心视图，只看服务状态、active 版本、入库质量和 LangSmith 链接。
8. 面试官追问工程治理时，说明企业路线以 LangSmith dataset/evaluation/annotation 为主，本地只保留领域指标和轻量状态页。

## 3. 代码阅读路线

如果要读代码，按这条线走：

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

各模块重点：

| 模块 | 看什么 |
| --- | --- |
| `qa_core/api/chat.py` | HTTP 轻量预检和 WebSocket 流式事件怎么转发 |
| `qa_core/application/service.py` | QAService 为什么只做服务门面，不保存请求状态 |
| `qa_core/pipeline/steps.py` | 意图识别、边界判断、Prompt 准备、上下文构建 |
| `qa_core/pipeline/retrieval_steps.py` | FAQ 检索、FAQ 标准答案直出、文档检索 |
| `qa_core/retrieval/` | Milvus Hybrid Search、过滤表达式和重排 |
| `qa_core/prompts/` | 不同问题类别如何选择 Prompt Profile |
| `qa_core/indexing/` | 文档加载、切分、FAQ/文档入库和质量报告 |

## 4. 面试讲解主线

可以用这段话开场：

> 这个项目不是简单把文档向量化后问答，而是把企业级 RAG 的核心工程问题做成闭环：FAQ 标准答案和文档 RAG 分层检索，Milvus dense + BM25 sparse 混合召回，BGE rerank 重排，知识库多版本、数据隔离、Prompt Profile 和流式输出由项目自研；Trace、Dataset、Evaluation 和 Annotation 交给 LangSmith。

继续展开时，按这 5 个点讲：

1. **检索设计**：为什么 FAQ 和文档分开，为什么用 Milvus 内置 BM25，不自研 BM25。
2. **业务边界**：为什么要 source 推断、source_boundary 和 scenario_boundary。
3. **风险控制**：费用、合同、隐私、赔付、安全类问题为什么使用更严格的 Prompt Profile。
4. **质量闭环**：入库质量、FAQ/正文冲突、低质量 chunk、领域评测指标如何进入 LangSmith Evaluation。
5. **工程边界**：一期只做稳定 RAG，二期用 Agent 调用 RAG 证据，不让 Agent 绕过检索和版本隔离。

## 5. 一期与二期边界

一期已经完成的是 RAG 工程闭环：

- 文档加载、切分、入库；
- FAQ + 文档混合检索；
- Milvus dense + BM25 sparse；
- rerank；
- 意图识别和问题类别识别；
- Prompt Profile；
- 知识库版本和数据隔离；
- 流式输出；
- 历史和反馈；
- LangSmith trace、评测、入库质量检查和 Bad Case 沉淀。

二期应该做的是 Agent 工作流：

- 任务识别；
- 工具调用；
- 多步骤规划；
- 人工确认；
- 风险处置草稿；
- 工单、合同、理赔、工程资料核查等流程输出。

二期 Agent 不提前放入一期源码。真正进入二期时，再新建 Agent 模块，通过 QAService 调用一期 RAG 证据，不能直接访问 Milvus，也不能绕过 `kb_version`、tenant、dataset、role 过滤。

## 6. 常用验收命令

开发后先跑轻量检查：

```powershell
python -m compileall app.py qa_core scripts tests
python -m pytest tests -q
python scripts/check_project_guardrails.py
```

脚本按学习优先级分三类：

| 优先级 | 脚本 | 使用场景 |
| --- | --- | --- |
| 必须掌握 | `rebuild_kb_version.py`、`evaluate_core_chain.py`、`check_evaluation_gate.py`、`api_e2e_smoke.py`、`acceptance_smoke.py` | 入库、评测、接口验收 |
| 验收掌握 | `check_project_guardrails.py`、`check_docs_consistency.py`、`api_e2e_smoke.py`、`acceptance_smoke.py` | 质量自检和交付证明 |
| 了解即可 | `analyze_*`、`build_enterprise_overlay_dataset.py`、`run_enterprise_overlay_activation.py`、LangSmith annotation/dataset | 高级治理和资料增强 |

演示前跑核心验收：

```powershell
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000 --scenario enterprise_knowledge
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000 --query "超过 5 万元的采购付款能走普通报销吗？" --scenario enterprise_knowledge
```

## 7. 文档快速索引

`docs/` 下的文档按用途分类如下。不知道该看哪个时，从这里开始：

| 你想要... | 看这个 |
|----------|--------|
| 系统学习 RAG 开发 | [`course-outline.md`](course-outline.md) → 18 讲课程 |
| 快速浏览全貌（有经验） | [`lecture_notes.md`](lecture_notes.md)（教师备课速查手册） |
| 理解项目架构 | [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) |
| 理解主链路代码流程 | [`current_architecture_flow.md`](current_architecture_flow.md) |
| 准备面试 | [`interview_playbook.md`](interview_playbook.md) + [`interview_faq.md`](interview_faq.md) |
| 写简历 | [`resume_project_pack.md`](resume_project_pack.md) |
| 标准化演示项目 | [`standard_demo_runbook.md`](standard_demo_runbook.md) |
| 理解 RAG 回归验收 | [`evaluation_and_quality.md`](evaluation_and_quality.md) |
| 搭建开发环境 | [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) |
| 理解编码规范 | [`coding_style_cn.md`](coding_style_cn.md) |
| 理解数据隔离设计 | [`data_scope_isolation.md`](data_scope_isolation.md) |
| 理解知识库版本管理 | [`kb_version_management.md`](kb_version_management.md) |
| 了解一期交付成果 | [`phase1_quick_review.md`](phase1_quick_review.md) |
| 了解二期设计规划 | [`phase2_agent_boundary.md`](phase2_agent_boundary.md) |
| 了解企业仿真数据 | [`enterprise_data_realism.md`](enterprise_data_realism.md) |
| 理解旧架构退役原因 | [`legacy_retirement.md`](legacy_retirement.md) |
