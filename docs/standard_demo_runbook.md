# KnowForge RAG Platform 标准演示 Runbook

本文用于立项、授课和面试演示。目标是在 10 到 15 分钟内讲清楚：KnowForge 不是普通聊天 Demo，而是一套企业级多场景 RAG 知识平台。

## 1. 演示目标

演示只围绕一条主线：

```text
业务问题
  -> source 推断 / 数据权限过滤
  -> FAQ 或文档混合检索
  -> Prompt Profile 路由
  -> 流式答案 + 来源引用
  -> LangSmith Trace
  -> RAG 回归验收
```

不要在标准演示里展开所有脚本、所有场景和所有报告。状态页只作为辅助视图，用来证明当前 active 版本、入库质量摘要和 LangSmith 状态。

## 2. 演示前检查

正式演示前先确认模型服务、Milvus、MySQL、本地模型和 active 知识库版本可用：

```powershell
python scripts/check_langchain_stack.py
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000 --scenario enterprise_knowledge
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000 --query "发票什么时候可以开？" --scenario enterprise_knowledge
```

如果要展示企业观测，把 `.env` 中的 `LANGSMITH_TRACING=true`、`LANGSMITH_API_KEY` 和 `LANGSMITH_PROJECT=knowforge-rag-platform` 配好。

## 3. 标准演示路径

1. 打开问答页：`http://127.0.0.1:8000/`
2. 选择 `engineering_project_qa`，提问：`施工图纸和强制性规范冲突时怎么办？`
3. 展示流式输出、来源引用、命中路径和 `compliance_guard` 类 Prompt Profile。
4. 选择 `insurance_claims`，提问：`收款账户和被保险人不一致可以打款吗？`
5. 说明赔付类问题不能让模型自由承诺，必须走业务边界和来源引用。
6. 选择 `enterprise_knowledge`，提问：`发票什么时候可以开？`
7. 展示 FAQ 直出和文档 RAG 的不同命中路径。
8. 打开状态页：`http://127.0.0.1:8000/admin`
9. 只看 LangSmith 状态、active 知识库版本、入库质量摘要和回归报告入口。
10. 打开 LangSmith 项目，查看 trace metadata：`scenario_id`、`source_filter`、`kb_version`、`intent`、`hit_type`、`prompt_profile`、`effective_source`。

## 4. 讲解口径

开场可以这样说：

> KnowForge RAG Platform 是一套企业级多场景 RAG 知识平台。项目自研的是业务 RAG 主链路：source 推断、权限过滤、FAQ 策略、Prompt Profile、知识库版本、表格/文档解析和领域评测指标；通用观测、Dataset、Evaluation 和 Annotation 交给 LangSmith。

重点讲 5 件事：

| 重点 | 讲什么 |
| --- | --- |
| 主链路 | 一次请求如何从页面进入 QAService，再到检索、Prompt 和流式返回。 |
| 检索设计 | FAQ 直出和文档 RAG 分层，Milvus dense + BM25 sparse 混合召回。 |
| 业务边界 | source 推断、场景隔离、租户/数据集/角色过滤。 |
| 风险控制 | 费用、赔付、合同、合规、安全类问题走更严格 Prompt Profile。 |
| 企业路线 | 本地只保留业务能力，LangSmith 承担 trace、dataset、evaluation 和 annotation。 |

## 5. 边界案例

演示一个边界问题即可，不要展开成专项测试：

```text
在保险理赔场景里问：施工图纸和强制性规范冲突时怎么办？
```

预期讲解：系统应识别场景边界，不应拿保险资料回答工程规范问题。这个例子用来说明 KnowForge 的目标不是“有问必答”，而是“在正确知识边界内回答”。

## 6. 收尾话术

最后强调：

- 一期已经定版为稳定 RAG 主链路，不把 Agent、GraphRAG、VLM 等二期能力提前塞进代码。
- 课堂重点是企业 RAG 的主链路和业务工程边界，不是自研 LLMOps 平台。
- 后续二期可以基于 LangGraph 增加 Agent 工作流，但 Agent 必须调用一期 RAG 证据，不能绕过知识库版本和权限过滤。
