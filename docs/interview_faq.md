# RAG 项目高频面试问答

本文档用于面试前快速复盘。回答时建议先讲结论，再讲项目中的实现位置和取舍原因。

## 1. 这个项目一句话怎么介绍？

这是一个基于 LangChain + Milvus Hybrid Search 的 KnowForge RAG Platform。它不仅能问答，还包含 FAQ + 文档混合检索、知识库多版本、数据隔离、Prompt 路由、流式输出、入库质量检查、评测回归和状态页可观测。

## 2. 项目里哪里用到了 LangChain？

主要使用在四类地方：

- 文档加载和切分：通过 LangChain loader/splitter 接入本地资料；
- Milvus 集成：通过 `langchain-milvus` 接入 Milvus Hybrid Search；
- LLM 调用：通过 `langchain-openai` 的 OpenAI-compatible ChatModel 调用 DashScope；
- 历史记忆：通过 `SQLChatMessageHistory` 把聊天历史存入 MySQL。

项目没有把 LangChain 当成黑盒大链条使用，而是把它作为成熟组件，主业务编排仍由 `QAService` 控制，这样更容易调试和评测。

## 3. 为什么不用自研 BM25？

当前项目使用 Milvus 2.6.x 内置 BM25，同时完成 dense 向量召回、sparse BM25 召回、版本过滤和数据隔离。

如果再自研 BM25，会带来：

- 第二套索引；
- 第二套入库流程；
- FAQ/文档更新一致性问题；
- 版本过滤和权限过滤重复实现；
- 排查 bad case 时不知道是哪套召回出错。

所以当前方案选择 Milvus Hybrid Search，而不是手写 BM25。

## 4. RedisSearch 在这里有没有必要？

当前主链路没有必要引入 RedisSearch。原因是 Milvus 已经承担向量和 BM25 混合检索，RedisSearch 会增加第二套检索索引。

Redis 更适合后续做：

- 热点 query 结果缓存；
- 短 TTL 的语义缓存；
- 会话级临时状态；
- 限流计数。

但它不应该替代当前 Milvus 检索主链路。

## 5. MySQL 还有存在意义吗？

有，但职责不是知识检索。

当前 MySQL 用于：

- 聊天历史；
- 会话摘要；
- 用户反馈；
- 后续管理元数据。

知识库检索走 Milvus。这样职责边界是：

- Milvus：知识召回；
- MySQL：会话状态和反馈记录。

## 6. 为什么 FAQ 和文档要分开？

FAQ 是标准口径，适合高置信直出。文档是依据和解释，适合复杂问题生成。

如果混在一起，可能出现两个问题：

- 标准答案被长文档稀释；
- 复杂问题被相似 FAQ 误直出。

当前项目先检索 FAQ，再检索文档。FAQ 高置信时直接返回标准答案；低置信或复杂问题进入文档 RAG。

## 7. 当前 RAG 检索策略是什么？

主策略是：

1. 根据场景和 source 构建检索过滤；
2. 做意图识别；
3. 必要时做追问改写；
4. 对知识类问题生成 query variants；
5. FAQ collection 做 Hybrid Search；
6. 文档 collection 做 Hybrid Search；
7. 使用 BGE reranker 重排；
8. 根据阈值和上下文预算构建最终上下文；
9. 根据问题类别选择 Prompt Profile。

不同问题不会共用一套固定 top_k。FAQ、知识咨询、追问、费用、合规、排障问题都会影响检索参数。

## 8. 置信机制是怎么做的？

当前置信机制不是单一分数，而是多因素组合：

- FAQ top score；
- 是否命中标准问题；
- FAQ 直出阈值；
- 问题类型；
- 是否短问题；
- 是否高风险类别；
- 是否有足够文档上下文；
- 最终来源数量和来源分数。

费用、合规类问题会提高 FAQ 直出阈值并扩大文档召回，避免模型过早给出确定结论。

## 9. 为什么要做 Prompt Profile？

因为不同业务问题的风险不同。

例如：

- “如何重置密码”可以走普通 FAQ；
- “客户要求退款能不能答应”必须走费用强口径；
- “HS 编码有争议能不能申报”必须走合规强口径；
- “API 限流怎么排查”应该走排障步骤模板。

当前项目的 Prompt Profile 包括：

- `faq_answer`
- `knowledge_answer`
- `pricing_guard`
- `compliance_guard`
- `troubleshooting_steps`
- `source_bound_summary`
- `follow_up`

这样可以让“怎么检索”和“怎么回答”保持一致。

## 10. 多版本知识库怎么实现？

入库时，每条 FAQ 和每个 chunk 都写入：

- `kb_version`
- `embedding_model_version`
- `reranker_model_version`
- `chunk_schema_version`

每个场景有自己的版本清单。在线检索默认只查 active 版本。新资料上线流程是：

1. 新建版本；
2. 入库；
3. 生成质量报告；
4. 跑评测；
5. 通过后激活；
6. 出问题可回滚 previous 版本。

这样不用覆盖旧数据，也不用批量更新 Milvus 记录。

## 11. 数据隔离怎么做？

每条数据写入 metadata：

- `tenant_id`
- `dataset_id`
- `visibility`
- `allowed_roles`

检索时生成 Milvus 表达式过滤：

- 限制当前租户；
- 限制当前数据集；
- 限制公开/内部可见性；
- 限制用户角色。

这解决的是多租户、多业务数据不能互相串库的问题。

## 12. 聊天历史为什么存 MySQL？

聊天历史是结构化会话数据，适合放 MySQL：

- 可按 session 查询；
- 可做摘要；
- 可审计；
- 可统计反馈；
- 方便和业务系统集成。

它不适合放 Milvus，因为聊天历史不是知识库召回主数据。

## 13. 流式输出是怎么实现的？

前端通过 WebSocket 调用 `/api/stream`。后端返回事件：

- `start`：请求已接收；
- `status`：意图识别、检索、生成等阶段状态；
- `token`：模型流式 token；
- `end`：最终答案、来源、检索诊断和 trace。

这样页面不是等完整答案返回，而是边生成边展示。

## 14. 如何证明效果不是靠手动调出来的？

项目内置评测集：

- `multi_scenario_smoke.json`：多场景基础召回；
- `multi_scenario_interview_regression.json`：source 自动推断和 Prompt 路由；
- `business_depth_regression.json`：业务深度问题；
- `multi_turn_followup_regression.json`：多轮追问。

评测指标包括：

- Recall@K；
- MRR；
- 关键词覆盖；
- hit_type 准确率；
- source 推断准确率；
- Prompt 模板命中率；
- FAQ 直出准确率；
- 场景隔离率；
- 错误率；
- 平均耗时。

最近业务深度回归结果是：

```text
total = 32
errors = 0
recall_at_k = 1.0
mrr = 1.0
prompt_profile_accuracy = 1.0
scenario_isolation_accuracy = 1.0
avg_keyword_coverage = 0.9922
```

## 15. 入库质量怎么保证？

入库质量报告会检查：

- 文件解析失败；
- 不支持文件；
- 空文件；
- 低质量 chunk；
- 重复 chunk；
- FAQ 空问题；
- FAQ 空答案；
- FAQ 重复问题；
- FAQ source 是否非法；
- FAQ 和正文是否存在口径冲突；
- 是否记录知识库版本和模型版本。

入库质量检查不通过时，不应该激活新版本。

## 16. 为什么复杂 OCR 没有默认进入主链路？

复杂 OCR 成本高、依赖重、失败率高。扫描件、图片 PDF、复杂表格如果默认进入主链路，会拖慢入库并引入不稳定结果。

当前一期主链路优先保证普通文本资料、FAQ、Markdown、PDF、Office 文档的稳定入库。复杂 OCR 更适合作为离线增强插件，不作为默认必需能力。

表格资料已经作为一期增强接入：CSV/Excel 会按行转换成带表头、工作表、行号和单元格键值的 Document，并写入 `content_type=table_row`。表格类问题会启用 `prefer_table` 检索计划，优先把表格行放入 prompt，同时来源会展示文件、工作表和行号。这能覆盖清单、金额、状态、材料字段、验收项等真实企业资料常见问题；但扫描件 OCR 仍需要先人工复核或独立清洗，不能默认写进 active 知识库。

当前已经提供离线 OCR 脚本：`scripts/ocr/run_offline_ocr.py`。它使用 PaddleOCR + PyMuPDF 把扫描件转换为待复核 Markdown，并输出 OCR 置信度报告。这个结果不会自动入库，必须人工复核后再通过 `scripts/ocr/promote_ocr_candidates.py` 复制到场景资料目录，最后重新执行知识库版本重建、入库质量检查和RAG 回归验收。

## 17. 图文混排资料算不算多模态？应该怎么设计入库？

算，但在当前项目里应该定位为**多模态入库治理**，不是多模态在线问答。

我们的设计边界是：用户在线提问时不实时解析图片、不执行 OCR、不让模型现场看图。图片、扫描件、图文 PDF、截图、流程图这类资料先走离线处理，生成可复核文本或图文块；复核通过后再进入知识库版本。

推荐流程是：

1. 有文本层的 PDF / Word / PPT：正文先走普通入库，图片进入风险报告；
2. 扫描件 / 图片 PDF：走离线 OCR，生成待复核 Markdown；
3. 图片和正文强相关的资料：生成 `image_text_block`；
4. 图文块必须包含 OCR/VLM 文本、附近正文、页码、图片编号、置信度和复核状态；
5. 只有 `review_status=reviewed` 且入库质量检查通过后，才能进入 active 知识库。

可以这样向面试官解释：

> 我们一期没有做实时多模态对话，而是把多模态能力收敛在知识库入库治理侧。图片、扫描件、图文 PDF 会先通过离线 OCR 或 VLM 生成可复核文本，再绑定页码、图片编号、附近正文和置信度。只有人工复核通过的图文块才会以 `image_text_block` 形式进入知识库，并继续经过入库质量检查、版本激活和回归评测。这样既能处理企业资料中的多模态信息，又不会让在线问答链路变重、变慢、变不稳定。

这比简单说“我接了 OCR”更专业，因为它强调了企业级 RAG 的关键点：资料可信度、来源溯源、人工复核、入库质量检查和版本化上线。

## 18. 为什么一期不做 Agent？

一期目标是把 RAG 做扎实：

- 知识库；
- 检索；
- 版本；
- 隔离；
- 质量；
- 评测；
- 流式问答。

如果一期就混入 Agent，学习者会分不清 RAG 主链路和工具调用流程。二期再用 LangGraph 做 Agent 更合理，可以基于稳定 RAG 结果做工单、审批、核查、风险处置等流程。

## 19. 如果线上答错了，怎么排查？

排查顺序：

1. 看 trace_id；
2. 看 scenario_id 是否正确；
3. 看 source_filter 是否推断正确；
4. 看 intent 是否正确；
5. 看 query_variants 是否合理；
6. 看 FAQ top score；
7. 看 doc source 是否召回正确；
8. 看 prompt_profile 是否正确；
9. 看上下文是否被截断；
10. 看 LLM 是否没有遵循资料边界。

状态页 `/admin` 只查看服务状态、active 版本、入库质量和回归报告入口；trace 细节在 LangSmith 查看。

## 20. 当前项目和普通 RAG Demo 的区别是什么？

普通 RAG Demo 通常只做：

- 文档切分；
- 向量入库；
- 相似度检索；
- 拼 prompt；
- 调模型。

当前项目多了：

- 多场景配置；
- FAQ + 文档双链路；
- Milvus Hybrid Search；
- source 推断；
- Prompt Profile；
- 知识库多版本；
- 数据隔离；
- 质量报告；
- RAG 回归验收；
- 流式输出；
- trace 可观测；
- 状态页；
- 业务深度样本。

## 21. 后续怎么升级？

下一阶段建议升级为 Agent 工作流，但不要破坏一期 RAG 主链路。

可做方向：

- LangGraph 风险处置流程；
- 工单草稿生成；
- 理赔材料缺失检查；
- 工程资料核查任务；
- 合同变更审批模拟；
- 多工具调用和人工确认节点。

面试时可以这样说：

> 一期先把可检索、可评测、可回滚、可观测的 RAG 主链路做稳；二期再在稳定 RAG 结果上叠加 Agent 工作流，避免一开始就把检索问题和工具调用问题混在一起。

