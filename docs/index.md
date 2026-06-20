# KnowFoundry-RAG-Console — 系统讲义

欢迎！这是一套基于 **LangChain + Milvus Hybrid Search** 构建的 **KnowFoundry-RAG-Console** 19 讲系统课程。

> 🎬 **快速入门**：先看 [RAG Pipeline 执行流程动画演示](animation/pipeline-demo.html)，5 分钟建立对整个系统执行流程的直观认识。

---

## 前言：如何高效学习这个项目

这个项目已经接近企业级 RAG 教学项目的完整度。如果第一次学习时把所有模块都当成同等重要，会很容易迷失。**正确方式是先抓住"能跑通一次高质量问答"的主链路，再逐步补治理、评测、多场景和复杂资料能力。**

### 核心原则

第一遍学习只回答一个问题：

> 用户问一个业务问题，系统如何找到正确资料，并流式生成一个有引用、可追溯的答案？

能把这条链路讲清楚，就已经掌握了项目的主干。其他内容是为了让系统更像企业项目、更适合面试表达，但不是第一遍必须全部吃透。

### 四层学习优先级

| 层级 | 定位 | 是否必须 | 学习目标 |
|------|------|----------|----------|
| **P0 主链路** | 在线问答闭环 | 必须掌握 | 能解释一次请求从页面到 RAG Pipeline 再到流式返回的全过程 |
| **P1 核心工程能力** | 检索质量、Prompt、入库、版本、评测 | 建议掌握 | 能说明为什么答案可靠、资料如何更新、如何防止质量退化 |
| **P2 企业化增强** | 多租户隔离、多场景、LangSmith Trace/Evaluation、领域指标 | 面试加分 | 能把项目讲成企业可落地方案 |
| **P3 扩展能力** | OCR/VLM、表格复杂治理、GraphRAG、Agent 规划 | 了解即可 | 知道边界和升级路线，不要求第一期完全掌握 |

### P0 主链路一览

| 模块 | 对应代码 | 核心概念 |
|------|----------|----------|
| 意图识别 | `qa_core/intent/classifier.py` | 决定是否检索、如何检索、用什么 Prompt |
| 检索策略 | `qa_core/retrieval/strategy.py` | 不同问题动态调整 top_k 和阈值 |
| 查询改写 | `qa_core/pipeline/rewrite.py` | 补全追问、生成多路检索变体 |
| Milvus 混合检索 | `qa_core/retrieval/store.py` | Dense + BM25 Sparse 混合召回 |
| 上下文构建 | `qa_core/pipeline/context.py` | 控制哪些证据进入 LLM，避免噪声 |
| Prompt Profile | `qa_core/prompts/` | 不同问题类别用不同回答口径 |
| 流式生成 + 引用 | `qa_core/pipeline/rag.py` | WebSocket 流式输出 + 来源标注 |

### P0 最低验收标准

1. 能画出"页面 → API → QAService → Pipeline → 检索 → LLM → 流式返回"的流程
2. 能解释为什么先做意图识别，再做检索计划
3. 能解释 Dense、Sparse、Reranker 在一次检索中的作用
4. 能解释为什么答案必须带来源引用

### 推荐学习节奏

| 阶段 | 时长 | 内容 | 对应讲次 |
|------|------|------|----------|
| **快速体验** | 1 天 | 启动项目，提问看流式输出，理解 RAG vs ChatGPT 区别 | 第 1-2 讲 |
| **主链路** | 3-5 天 | P0 全部内容：LangChain→意图→检索→改写→Milvus→Pipeline→Prompt | 第 3-10 讲 |
| **工程化** | 2-3 天 | 框架原理、入库、版本管理、数据隔离、评测、测试 | 第 11-16 讲 + 第 18 讲 |
| **面试拔高** | 1-2 天 | 多场景复用、LangSmith 诊断、Bad Case 沉淀、二期规划 | 第 17 讲 + 面试材料 |

### 第一遍可以暂时跳过的内容

- 复杂 OCR / VLM 细节
- 复杂 Excel 语义还原
- 所有 scripts 脚本的逐行解释
- 所有 8 个业务场景的完整资料细节
- Mermaid 架构图里的每个辅助节点
- 容量评估和历史报告复盘的实现细节
- 二期 Agent、LangGraph、A2A、GraphRAG 设计细节
- 状态页前端样式和非核心 UI 交互

第一遍只需要知道它们存在，以及它们服务于哪个工程目标。

### 最小闭环作业

完成下面 **5 件事**，就说明已经掌握一期主线：

1. 启动项目并完成一次流式问答
2. 解释该问题的意图分类结果
3. 解释该问题的检索计划为什么这样设置
4. 找到至少一条引用来源，并说明它是怎么从 Milvus 召回来的
5. 修改一份文档或 FAQ，重新入库并让新版本生效

### 代码阅读路线

如果要读代码，按这条线走——不要跳，不要从中间开始：

```
app.py
  → qa_core/api/chat.py          （HTTP 预检 + WebSocket 流式事件）
  → qa_core/application/service.py （QAService 编排层）
  → qa_core/pipeline/rag.py        （RAG 主流程）
  → qa_core/pipeline/steps.py      （意图识别、边界判断、上下文构建）
  → qa_core/pipeline/retrieval_steps.py （FAQ 检索、文档检索）
  → qa_core/retrieval/             （Milvus Hybrid Search、过滤、重排）
  → qa_core/prompts/               （Prompt Profile 路由）
  → qa_core/indexing/              （文档加载、切分、入库）
```

### 面试表达主线

面试时不要从"项目有很多功能"开始讲，而应按主次讲：

> 这个项目的核心是一套企业级 RAG 问答主链路。
> 在线侧通过 HTTP 预检和 WebSocket 实现流式问答；
> 检索侧通过意图识别、动态检索计划、查询改写、Milvus Dense+Sparse 混合检索和 Reranker 提高召回质量；
> 生成侧通过 Prompt Profile、引用增强和上下文筛选保证答案可控；
> 工程侧通过知识库版本、数据隔离、入库质量检查、领域评测指标和 LangSmith Trace/Evaluation 保证系统可维护、可回滚、可诊断。
> OCR/VLM、GraphRAG、LangGraph Agent 属于后续扩展方向，不混入一期主链路。

---

## 快速导航

- [学习优先级与主次拆分](learning_priority.md) — P0-P3 完整优先级说明、18 讲拆分、教学节奏建议
- [课堂代码地图](teaching_code_map.md) — 每讲只读哪些文件，避免从目录树硬啃
- [学习与演示总入口](learning_demo_guide.md) — 演示路径、代码阅读顺序、面试主线
- [核心术语速查表](core_terms.md) — 快速理解 DataScope、Bad Case、OCR/VLM、table_row 等高频概念
- [课程大纲与学习路线](course-outline.md) — 18 讲整体结构和推荐学习顺序
- [第 1 讲：项目概述](01-project-overview.md) — 从这里开始
- [技术附录](appendix/appendix-a-pydantic.md) — 专题深度解析
- [复习笔记卡片](notes/index.md) — 快速回顾每讲要点
- [教师备课速查手册](lecture_notes.md) — 13 章凝练版，含教学要点提示

## 课程结构

| 阶段 | 讲次 | 内容 | 优先级 |
|------|------|------|--------|
| 第一阶段 | 01-02 | 基础概念：RAG 原理、向量检索、Embedding | P0 |
| 第二阶段 | 03-10 | 核心 RAG 链路：LangChain→意图→检索→改写→Milvus→Pipeline→Prompt | **P0（核心）** |
| 第三阶段 | 11-12 | Web 服务基础设施：FastAPI、Preflight | P1 |
| 第四阶段 | 13-18 | 治理与运维：版本管理、数据隔离、入库、评测、测试、追踪 | P1-P2 |

### 18 讲详细优先级

| 讲次 | 主题 | 优先级 | 第一遍学习要求 |
|------|------|--------|----------------|
| 01 | 项目概述 | P0 | 必学，先建立项目全局视角 |
| 02 | RAG 核心概念 | P0 | 必学，理解 Dense/Sparse/Reranker |
| 03 | LangChain 生态 | P0 | 必学，RAG 的"语言基础" — Runnable/LCEL/ChatOpenAI |
| 04 | 意图分类 | P0 | 必学，后续策略都依赖它 |
| 05 | 检索策略 | P0 | 必学，理解动态参数 |
| 06 | 查询改写与变体 | P0 | 必学，追问补全和多路检索 |
| 07 | Milvus 混合检索 | P0 | 必学，RAG 召回核心 |
| 08 | QAService 编排 | P0 | 必学，理解服务层如何串联 |
| 09 | RAG Pipeline | P0 | 必学，全项目主流程 |
| 10 | Prompt Profile | P0 | 必学，控制答案结构和业务边界 |
| 11 | FastAPI 与异步 | P1 | 理解 async/await、WebSocket，RAG 的"骨架" |
| 12 | 应用入口与前置校验 | P1 | 理解为什么环境必须完整 |
| 13 | 知识库版本 | P1 | 建议掌握，体现工程可靠性 |
| 14 | 数据隔离 | P1 | 建议掌握，企业项目必问 |
| 15 | 文档入库 | P1 | 建议掌握，解释知识如何进入系统 |
| 16 | RAG 回归验收与入库质量 | P1 | 建议掌握，证明效果不是拍脑袋 |
| 17 | 测试与接口验收 | P1 | 先掌握核心测试思路 |
| 18 | LangSmith 观测与 Trace | P2 | 第二遍学习，面试加分 |

## 技术栈

| 层级 | 技术 |
|------|------|
| API 框架 | FastAPI + WebSocket |
| RAG 编排 | LangChain |
| 向量数据库 | Milvus 2.6 Hybrid Search |
| Embedding | BGE-M3 (本地部署) |
| Reranker | BGE Reranker Large (CrossEncoder) |
| LLM | DashScope (OpenAI 兼容) |
| 会话存储 | MySQL |

## 如何使用

- **零基础学生**：先读完本页的前言部分，理解 P0→P1→P2→P3 优先级，再按 01 → 18 顺序学习，每学一讲对照优先级表格确认自己在哪个层级
- **有经验的开发者**：先看 [教师备课速查手册](lecture_notes.md)，再针对薄弱讲次精读对应章节
- **赶时间的面试准备**：P0 必学（01 → 02 → 03(LangChain) → 04 → 05 → 06 → 07 → 08 → 09 → 10），P1 选学（13 → 14 → 16），剩下的用 [面试讲解手册](interview_playbook.md) 快速补
- **只做演示**：看 [学习与演示总入口](learning_demo_guide.md)，10-15 分钟完成演示流程
