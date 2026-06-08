# 多场景 RAG 项目面试讲解手册

## 一句话介绍

这是一个基于 LangChain + Milvus 2.6 Hybrid Search 的多场景 RAG 教学项目。项目不是只做“能问答”的 Demo，而是补齐了入库、切分、向量化、BM25 稀疏检索、重排、意图识别、Prompt Profile、流式返回、历史记忆、知识库版本、数据隔离、评测、质量报告和可观测状态页。

## 面试时要先划清架构边界

优化前架构以旧目录 `mysql_qa`、`rag_qa` 为主，典型问题是链路分散、MySQL/Redis/手写检索职责混杂、BM25 方案自研倾向明显、历史与问答主链路耦合、缺少知识库版本、缺少入库质量报告、缺少可观测追踪。

优化后架构以 `app.py + qa_core/api + qa_core` 为主，主链路统一进入 `QAService`：

1. `app.py` 只负责创建 FastAPI 应用、静态资源、CORS、startup preflight 和路由注册。
2. `qa_core/api` 负责页面、问答、管理诊断、知识库版本等 HTTP/WebSocket 路由。
3. `qa_core.application.service.QAService` 负责编排意图、改写、检索、生成和历史。
4. `qa_core.indexing` 子包负责本地文档加载、切分和 Milvus 入库。
5. `qa_core.retrieval` 子包负责 Milvus dense + sparse 混合检索。
6. `qa_core.retrieval.strategy` 负责不同问题类型的检索计划。
7. `qa_core.prompts` 子包负责按问题类别选择提示词模板。
8. `qa_core.governance.kb_versions` 负责知识库多版本激活、回滚和灰度。
9. `qa_core.governance.data_scope` 负责租户、数据集、可见级别和角色过滤。
10. `qa_core.observability` 负责把每次问答的业务 metadata 写入 LangSmith Trace。
11. `qa_core.quality.ingestion` 负责入库质量报告。

## 优化前问题与场景例子

### 1. 检索能力分散，答案质量不稳定

优化前如果用户问“Webhook 调用失败如何排查”，系统可能依赖单一路径检索，FAQ、文档和关键词规则之间没有清晰优先级。问题一旦换一种说法，例如“回调失败看哪里”，就可能召回不到标准 FAQ。

优化后使用查询变体、FAQ 优先、文档补充、重排和动态阈值。FAQ 高置信时直接返回标准答案；低置信时进入文档 RAG，避免误直出。

### 2. 手写 BM25 和 RedisSearch 不是当前项目最佳选择

当前项目使用 Milvus 2.6.x 的 BM25BuiltInFunction，主链路没有必要再维护自研 BM25 或额外引入 RedisSearch。否则会出现两个检索索引、两套入库链路、两份一致性问题。

优化后把 dense embedding 和 sparse BM25 都收敛到 Milvus 集合里。入库一次，检索一次，版本过滤和数据隔离也在同一个表达式里完成。

### 3. MySQL 不适合作为知识检索主库

MySQL 适合保存聊天历史、摘要、反馈和审计类结构化数据，不适合承担向量检索、语义召回和多路重排。优化前如果把 FAQ 或文档检索继续压在 MySQL，会导致语义召回弱、排序能力弱、扩展复杂。

优化后 MySQL 保留在历史和反馈链路，知识召回统一走 Milvus。这样 MySQL 的意义变成“会话状态和用户反馈”，而不是“RAG 检索引擎”。

### 4. 缺少知识库版本，无法安全回滚

优化前重新入库可能覆盖旧数据。新切分策略或新 embedding 模型一旦效果变差，很难快速回退。

优化后每条 FAQ/chunk 写入 `kb_version`、`embedding_model_version`、`chunk_schema_version`。在线检索只查 active 版本。新版本可以先入库、评测、再激活；失败时回滚到 previous 版本。

### 5. 缺少数据隔离，容易串库

如果项目扩展成多租户，优化前只按 source 过滤不足以防止 A 客户查到 B 客户资料。

优化后每条数据写入 `tenant_id`、`dataset_id`、`visibility`、`allowed_roles`，检索时追加 Milvus 表达式过滤。这样同一套 collection 可以演示租户隔离、数据集隔离和角色隔离。

### 6. 缺少入库质量报告，无法证明知识库质量

优化前只能说“文件已经入库”，但说不清哪些文件失败、哪些 chunk 过短、哪些 FAQ 重复。

优化后通过 `scripts/rebuild_kb_version.py --quality-gate` 生成报告，覆盖文件解析、FAQ 质量、低质量 chunk、重复 chunk、模型版本和切分版本。

### 7. 缺少可观测追踪，线上问题不好定位

优化前用户说“答得不对”，只能重新猜测问题出在哪。

优化后每次问答都会写入 `LangSmith Trace`，记录 trace_id、问题、场景、数据域、命中路径、意图、检索计划、来源数量、top score、耗时和错误。状态页 `/admin` 只展示 LangSmith 状态，trace 细节在 LangSmith 查看。

## 当前架构优势

### 1. 主链路闭环

用户在页面提问后，流程是：

1. `/api/query` 先判断是否是问候、越界、人工客服等直接答案。
2. 复杂问题走 `/api/stream`。
3. `QAService.stream_query` 输出 start/status/token/end 事件。
4. 意图识别决定是否改写追问、是否推断 source。
5. 检索计划决定 FAQ/doc 是否执行、top_k、阈值、是否 rerank。
6. Milvus 执行 dense + sparse 混合检索。
7. FAQ 高置信直接回答。
8. 否则选择上下文，按 Prompt Profile 调用 LangChain ChatModel 流式生成。
9. 完成后写历史、写追踪、返回 sources 和 retrieval 诊断。

### 2. 面向多个业务场景

当前已有八个场景包：

- 企业知识库：HR、IT、财务制度问答。
- SaaS 客服：账号、计费、集成问题。
- 设备运维：点检、告警、安全 SOP。
- 合规问答：合同、隐私、审计材料。
- 跨境贸易风控：海关申报、制裁筛查、信用证、贸易术语和单证一致性。
- 招投标履约风控：投标文件、合同条款、交付延期、验收材料和履约风险。
- 保险理赔审核：保单条款、理赔材料、责任认定、除外责任和赔付结算。
- 工程项目资料问答：设计图纸版本、施工规范冲突、进度延期、隐蔽验收和安全资料。

学生可以学习同一套 RAG 能力，再把项目包装成不同简历背景，而不是只绑定教育行业。

跨境贸易风控、招投标履约风控、保险理赔审核和工程项目资料问答是当前更建议主推的差异化包装。普通企业知识库、客服问答和设备 SOP 在市场上比较常见，而这四类场景能把 RAG 和业务风险、资料版本、标准规范结合起来：模型不能乱承诺“可以出口”“一定中标”“一定赔付”，工程场景也不能在图纸旧版本、规范冲突或验收资料不足时给出确定施工依据，必须基于资料说明已确认和未确认边界，这更容易体现工程化 RAG 的价值。

当前场景扩展已经冻结为这 8 个。面试时可以强调：项目没有继续堆业务外壳，而是把场景包治理写进 `scripts/check_project_guardrails.py`，后续新增场景会被守护检查拦截，优化重心回到资料质量、版本治理和二期 Agent。

### 3. 面试可讲的技术关键词

- LangChain loader / splitter / ChatModel
- Milvus 2.6 Hybrid Search
- BM25BuiltInFunction
- BGE-M3 embedding
- CrossEncoder rerank
- parent-child chunk
- query variants
- intent routing
- prompt profile
- streaming WebSocket
- SQLChatMessageHistory
- knowledge base versioning
- tenant/dataset/role isolation
- Recall@K / MRR
- ingestion quality report
- trace observability

## 常见面试问题回答

### 为什么不用 RedisSearch？

当前主链路的 dense、sparse、版本过滤和数据隔离都在 Milvus 内完成。RedisSearch 再加入会带来第二套索引和一致性成本。除非后续要做独立语义缓存、热点 query 缓存或非 Milvus 场景的快速全文检索，否则不是必要组件。

### 为什么 MySQL 还保留？

MySQL 保留用于聊天历史、摘要记忆、用户反馈和可能的管理元数据。它不再承担知识检索主库职责。这样职责更清晰：Milvus 负责召回，MySQL 负责会话与反馈。

### 为什么要根据问题类别切 Prompt？

不同问题风险不同。费用、承诺、合规类问题需要更严格模板；业务资料说明、设备 SOP、Webhook 排查需要结构化步骤；问候和越界问题不应该调用知识库。Prompt Profile 和检索计划共用同一份 intent，保证“怎么查”和“怎么答”一致。

### 多版本检索如何实现？

入库时每条记录带 `kb_version`。版本清单记录 active_version。检索时如果有 active 版本，就在 Milvus expr 追加 `kb_version == active`。激活或回滚只改本地版本清单，不需要批量更新 Milvus 数据。

### 数据隔离如何实现？

入库 metadata 写入 `tenant_id`、`dataset_id`、`visibility`、`allowed_roles`。检索时根据请求的数据域生成过滤表达式，限制只能召回当前租户、当前数据集、当前可见级别和当前角色允许的数据。

### 如何证明 RAG 效果？

项目内置 `scripts/evaluate_core_chain.py`。基础评测集是 `eval_sets/multi_scenario_smoke.json`，用于验证 8 个场景的 FAQ、文档召回和场景隔离。增强评测集是 `eval_sets/multi_scenario_interview_regression.json`，刻意不传 `source_filter`，用于验证 source 自动推断和 Prompt Profile 路由。评测输出 Recall@K、MRR、关键词覆盖、hit_type 分布、source 推断准确率、Prompt 模板准确率、错误率和平均耗时。它用于工程回归，不替代人工验收。

面试时可以补一句：我没有只评测“固定分类下能不能查到资料”，还增加了无分类输入的回归样本。例如用户问“员工报销需要准备哪些材料”，系统必须自动推断到 `finance`；用户问“施工图纸和强制性规范冲突时项目部能自行判断吗”，系统必须推断到 `specification`，并进入 `compliance_guard` 模板。

### 如何证明入库质量？

项目内置入库质量检查，报告记录文件扫描数、解析失败文件、不支持文件、FAQ 空问题/空答案/重复问题、低质量 chunk、重复 chunk、模型版本和切分版本。

### 如何定位线上 bad case？

每次问答写入 LangSmith Trace，并可通过 LangSmith 项目页查看。定位顺序是：看 hit_type，看 intent，看 query_variants，看 top source score，看 sources_count，看是否信息不足，看 prompt_profile。

## 简历项目包装示例

### 企业内部知识库智能助手

基于 LangChain + Milvus Hybrid Search 构建企业制度和 IT 支持问答系统，实现 FAQ 直出、文档 RAG、会话记忆、知识库版本管理、租户/角色隔离、流式输出和可观测追踪。

### SaaS 客服智能问答系统

面向账号、计费和集成场景构建客服知识库，使用 FAQ 高置信直出和文档补充生成，支持 Webhook 排查、发票咨询、账号重置等问题，并通过 Recall@K/MRR 做回归评测。

### 设备运维 SOP 智能助手

基于设备点检、温度告警和安全 SOP 构建运维问答系统，支持按告警类别检索、结构化步骤输出、知识库版本回滚和入库质量检测。

### 合规审查知识助手

面向合同、隐私和审计材料构建合规问答系统，对敏感个人信息、合同审批材料和审计缺失处理使用更严格的提示词模板，避免模型做未确认承诺。

### 跨境贸易风控 RAG 知识问答平台

面向海关申报、制裁筛查、信用证不符点、贸易术语和单证一致性构建风控知识库，使用 Milvus Hybrid Search 完成 FAQ 直出和文档召回，结合知识库版本、数据隔离、来源引用和合规类 Prompt Profile 控制回答边界。

### 招投标合规与合同履约 RAG 风控平台

面向投标文件、合同付款条款、项目交付、验收材料和履约风险构建知识库，支持按业务分类检索标准问答和履约文档，重点控制“付款承诺、延期责任、验收通过”等高风险口径。

### 保险理赔材料审核与 RAG 知识问答助手

面向保单条款、理赔材料、责任认定、除外责任和赔付结算构建知识库，支持材料清单查询、除外责任解释和赔付口径约束，强调资料未核定前不能承诺最终赔付金额。

### 工程项目资料与施工规范 RAG 问答助手

面向设计图纸、施工规范、进度计划、质量验收和安全资料构建知识库，支持图纸版本追溯、规范与图纸冲突处理、隐蔽工程验收资料查询和安全交底资料核查，重点体现多文档、多版本和标准规范检索能力。

## 推荐演示顺序

1. 打开 `/`，切换不同场景提问。
2. 打开 `/admin`，展示最近 trace、命中路径、知识库版本和报告。
3. 运行入库质量报告脚本，展示低质量 chunk 和 FAQ 检测能力。
4. 运行基础评测脚本，展示 Recall@K、MRR、关键词覆盖和场景隔离。
5. 运行增强回归脚本，展示 source 自动推断和 Prompt Profile 路由没有退化。
6. 讲解从旧架构到 `qa_core` 的职责收敛。

## 后续可继续强化但不应扩散主业务

- 复杂 OCR 作为可选离线插件，不默认进入主入库链路。
- 表格专用切分可作为 source-specific splitter。
- 语义缓存可用于热点问答，但不能替代知识库版本和数据隔离。
- 完整权限系统可以接入用户中心，但当前项目只做检索层轻量隔离。
