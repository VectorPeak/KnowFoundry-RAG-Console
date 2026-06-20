# KnowFoundry-RAG-Console - 当前架构文档

## 1. 项目定位

当前项目已经从旧版教育问答实现，收敛为基于 `qa_core` 的 KnowFoundry-RAG-Console。

运行时默认不包含课程咨询场景，当前场景包包括：

- `enterprise_knowledge`：企业内部知识助手。
- `saas_support`：SaaS 客服知识助手。
- `equipment_ops`：设备运维知识助手。
- `compliance_qa`：合规制度知识助手。
- `cross_border_risk`：跨境贸易风控助手。
- `tender_contract_risk`：招投标与合同履约风控助手。
- `insurance_claims`：保险理赔审核助手。
- `engineering_project_qa`：工程项目资料问答助手。

旧版 `mysql_qa` 和 `rag_qa` 代码已经从工程中移除，只在文档中保留迁移前实现对照。

业务场景扩展已经冻结为以上 8 个，并由 `scripts/check_project_guardrails.py` 做可执行约束。后续一期优化不再新增场景包，重点转向资料质量、评测覆盖、多版本治理和二期 Agent。

当前补强策略是“冻结数量、加深业务”。每个场景不再只保留一条 FAQ 和一份说明文档，而是增加风险判断、流程状态、冲突处理和资料闭环类样本。例如：

- 企业知识库补充试用期转正、权限回收和预算审批；
- SaaS 客服补充离职账号、退款赠送额度和 API 限流；
- 设备运维补充日检异常升级、反复告警和受限空间作业；
- 合规风控补充供应商尽调、审计整改闭环和数据导出审批；
- 跨境贸易补充 HS 归类争议、最终用途尽调、信用证软条款、DDP 和单证版本；
- 招投标合同补充保证金、合同变更、范围蔓延、口头验收和需求变更；
- 保险理赔补充复效审核、影像质量、责任争议、既往症和账户一致性；
- 工程项目补充图纸会审、强制性条文、关键线路、检验批不一致和安全交底。

## 2. 当前主链路

```text
浏览器页面 static/index.html
  -> FastAPI app.py
  -> qa_core/api 路由层
  -> QAService
  -> 意图识别 classify_intent
  -> 追问改写 rewrite_query_if_needed
  -> 检索计划 build_retrieval_plan
  -> FAQ Milvus Hybrid 检索
  -> 文档 Milvus Hybrid 检索
  -> rerank / 去重 / 上下文预算 / 上下文构建
  -> Prompt Profile
  -> LangChain ChatOpenAI 流式生成
  -> SQLChatMessageHistory 写入历史
  -> 追踪日志与前端 end 事件
```

## 3. 技术栈

| 层级 | 当前方案 | 说明 |
| --- | --- | --- |
| API | FastAPI + WebSocket + `qa_core/api` 路由拆分 | `app.py` 只创建应用和注册路由；HTTP 只做轻量预检，复杂回答统一走 `/api/stream` 流式返回。 |
| RAG 编排 | `qa_core/application/service.py` | 收敛 FAQ、文档、历史、追踪和反馈入口。 |
| LangChain | ChatOpenAI、SQLChatMessageHistory、Document Loader、Text Splitter、Milvus VectorStore | 尽量使用开源生态能力，避免自研完整框架。 |
| 检索 | Milvus Hybrid Search | 使用 dense + Milvus BM25 sparse，不再引入 RedisSearch 或本地 rank_bm25。 |
| Milvus 适配 | `langchain-milvus` + `qa_core/retrieval/milvus_compat.py` | 业务检索统一走 LangChain VectorStore；PyMilvus 只保留在兼容层，用于 database 检查、BM25 函数和 ORM alias 注册。 |
| 模型 | BGE-M3 embedding、BGE reranker、OpenAI-compatible LLM | 支持本地模型路径和 DashScope/OpenAI 兼容接口。 |
| 历史 | MySQL + LangChain SQLChatMessageHistory | MySQL 只承担聊天历史、摘要、反馈，不再做知识检索主引擎。 |
| 治理 | kb_version、data_scope、trace、quality report | 支持版本回滚、租户/数据集隔离、入库质量和评测。 |
| 启动校验 | `qa_core.config.preflight` + `warmup_retrieval_stack` | Milvus、MySQL、模型目录、管理令牌、active 知识库版本和 LLM 真实连通性缺失时直接启动失败；启动预热覆盖全部 8 个冻结场景的 FAQ/doc collection。 |
| 质量检查 | `scripts/check_ingestion_quality_gate.py`、`scripts/check_evaluation_gate.py`、`scripts/check_followup_gate.py`、`scripts/api_e2e_smoke.py` | 把质量报告、主链路评测、多轮追问和接口冒烟转成可失败的检查脚本。 |
| 环境一键化 | `docker-compose.yml`、`Dockerfile`、`.env.local.example`、`.env.compose.example` | 标准化 MySQL、Milvus、API 和环境变量，不提供低配降级；本机 API 与 compose API 两种模式分开配置。 |

## 4. 核心模块

| 模块 | 职责 |
| --- | --- |
| `qa_core/api` | FastAPI 路由层：页面、问答、管理诊断、知识库版本、限流和管理令牌。 |
| `qa_core/application` | 应用服务编排、服务工厂。 |
| `qa_core/intent` | 意图识别和问题风险类别识别。 |
| `qa_core/retrieval` | Milvus store、检索过滤、重排、检索计划。 |
| `qa_core/pipeline` | RAG 流程、事件、上下文构建、查询改写和查询扩展。 |
| `qa_core/prompts` | 不同问题类别和意图的提示词模板。 |
| `qa_core/indexing` | 文档加载、切分、FAQ 入库、manifest 增量清单。 |
| `qa_core/governance` | 知识库版本和数据隔离。 |
| `qa_core/memory` | 聊天历史、摘要和反馈。 |
| `qa_core/quality` | 入库质量、低质量 chunk、FAQ 与正文冲突检测。 |
| `qa_core/scenarios` | 多业务场景注册和 source 白名单配置。 |

前端页面保持轻量静态方案，没有引入新的前端构建工具。问答页入口仍是
`static/index.html`，但 JS 已按状态、接口、渲染、场景、会话和聊天拆分到
`static/js/`，CSS 已按基础变量、布局、表单、侧栏、聊天、输入区和响应式拆分到
`static/css/`。这样能降低阅读成本，同时不增加额外工程化依赖。

## 5. 检索策略

当前检索不是单一路径，而是分层策略：

1. 问候、越界、人工客服短句直接返回。
2. 追问问题先结合历史改写成独立问题。
3. FAQ 集合先检索，高置信命中时直接返回标准答案。
4. FAQ 低置信时进入文档 RAG，FAQ 结果可作为补充上下文。
5. 文档检索使用 dense + sparse hybrid，并经过 reranker 重排。
6. Prompt Profile 根据问题类别选择更严格的输出模板。
7. 上下文进入 Prompt 前会按 FAQ/chunk/parent 去重，并受 `MAX_PROMPT_CONTEXT_CHARS`
   和 `MAX_CONTEXT_DOC_CHARS` 控制，避免低价值长片段挤占模型窗口。

这样设计的原因是：FAQ 适合确定答案，文档 RAG 适合整合资料；两者混在一个粗暴流程里，会同时损失准确率和可解释性。

当前没有改成纯 PyMilvus 自研适配器。教学版的取舍是：用 `langchain-milvus`
承接 RAG VectorStore、Document 和混合检索封装，用 `milvus_compat.py` 集中处理
底层连接现实。这样课堂重点仍然放在 RAG 主链路，而不是把大量时间花在 Milvus schema、
ORM alias 和客户端兼容细节上。

FAQ 性能上，当前意图识别采用“场景 source 推断 + 短问题问法规则”提前识别标准问答形态。例如“交易对手命中制裁名单怎么办？”可以通过 `cross_border_risk` 的 `source_patterns` 推断为 `sanction`，并因“怎么办”命中标准问答形态，直接进入 FAQ 优先检索，减少一次不必要的 LLM 意图分类调用。这个优化不引入缓存，也不牺牲知识库版本和数据隔离。

## 6. 多版本和隔离

每条 FAQ 和文档 chunk 都会写入：

- `scenario_id`
- `tenant_id`
- `dataset_id`
- `visibility`
- `allowed_roles`
- `kb_version`
- `embedding_model_version`
- `reranker_model_version`
- `chunk_schema_version`

在线检索会把这些字段拼入 Milvus 表达式，保证不同场景、版本和数据域不会混查。

## 7. 当前边界

当前一期只完善 RAG 能力，不默认引入二期 Agent 动作执行。

二期 Agent 只保留设计文档，不提前放进一期源码。这样做的目的是：

- 一期继续保持“问答、检索、引用、评测”的 RAG 闭环；
- 二期真正开工时再基于 LangGraph 新建 Agent 模块，并通过 QAService 复用一期 RAG；
- Agent 不允许直接连接 Milvus，也不能绕过 `kb_version`、tenant、dataset、role 隔离；
- 发现 `scenario_boundary`、`source_boundary` 或 `insufficient_context` 时，Agent 必须停止自动处置，转为提示切换场景、切换分类或人工确认。

不进入当前核心链路的内容：

- 旧版 `mysql_qa` / `rag_qa` 在线流程。
- RedisSearch 第二套检索索引。
- 复杂 OCR 默认入库主链路。
- 无版本约束的答案缓存。
- 非核心业务扩散功能。
- 依赖缺失时的低配技术降级路径。
- 二期 Agent 工具调用默认在线启用。

当前 API 边界：

- `app.py` 只做应用启动、静态资源、CORS、startup preflight 和 router 注册；
- `qa_core/api/chat.py` 承接 `/api/stream`、`/api/query`、历史、反馈和检索诊断；
- `qa_core/api/admin.py` 承接 LangSmith 状态、入库报告、回归报告等只读接口；
- `qa_core/api/kb_versions.py` 承接知识库版本查看、激活和归档；
- `qa_core/application/service.py` 仍然是 RAG 业务编排入口，API 层不直接拼 prompt 或查 Milvus。

## 8. 质量与回归保障

当前项目已经补齐基础回归与真实链路验收：

- 纯逻辑测试：问题类别、source 过滤、FAQ 直出、上下文构建、检索计划、场景注册。
- API 保护测试：验证管理令牌和轻量限流逻辑可控。
- 项目守护检查：`scripts/check_project_guardrails.py` 验证导入位置、旧链路、fallback 导入、直接依赖版本、完整依赖锁和场景包结构。
- 入库质量检查：`scripts/check_ingestion_quality_gate.py` 检查解析失败、unsupported 文件、空文件、低质量 chunk、FAQ 空值/重复、FAQ/正文冲突和版本字段。
- 主链路 RAG 回归验收：`scripts/check_evaluation_gate.py` 检查 Recall@K、MRR、关键词覆盖、hit_type 准确率、source 自动推断、Prompt Profile 路由、FAQ 直出准确率、场景隔离和平均耗时，并按场景、source、hit_type 做分组验收，防止局部退化被全局均值掩盖。
- 业务深度回归：`eval_sets/business_depth_regression.json` 覆盖每个场景新增的风险判断、流程状态和冲突处理样本，防止资料补强后召回或模板路由退化。
- 多轮追问评测：`scripts/evaluate_followup_chain.py` 在同一个 session 中连续提问，验证历史读取、追问改写、source 推断和后续召回。
- 多轮追问回归验收：`scripts/check_followup_gate.py` 检查追问召回、追问 source 准确率、Prompt Profile 和场景隔离。
- 性能基线采集：`scripts/collect_performance_baseline.py` 真实调用 `QAService.stream_query`，记录首 token 耗时、总耗时、阶段耗时、token 数、来源数和 hit_type 分布。
- 固定性能基线：`eval_sets/phase1_performance_baseline.json` 覆盖 8 个场景的 FAQ、文档 RAG 和表格 RAG，用于后续优化对比同一批稳定样本。
- 性能回归验收：`scripts/check_performance_gate.py` 检查错误率、首 token、总耗时和阶段耗时是否退化。
- API 合同验收：`scripts/api_e2e_smoke.py` 检查 `/health`、场景、版本、LangSmith 状态、入库质量、回归报告和性能报告接口字段。
- 场景包守卫：检查 `scenarios/*/scenario.toml`、FAQ、source、collection、文档目录和 source pattern，防止新增场景时出现空分类、集合重名或 FAQ source 写错。
- Bad Case 沉淀：线上问题通过 LangSmith Trace 标注，沉淀到 LangSmith Dataset 后进入 Evaluation；项目只保留 source、hit_type、Prompt Profile、表格行召回等领域指标。
- 评测趋势对比：企业路线下优先在 LangSmith Experiments 中查看 Recall、MRR、关键词覆盖、source 推断、Prompt Profile、场景隔离和耗时变化；本地状态页只展示回归报告入口。
- 反馈回归闭环：用户反馈进入 LangSmith annotation/dataset 流程，人工确认后进入回归评测。
- 缺失文档清理：`scripts/cleanup_missing_docs.py --all-scenarios` 对比 manifest 和本地文件，生成旧 chunk 清理差异报告，显式 `--apply` 才删除 Milvus chunk。
- 全场景版本对比：`scripts/kb/compare_all_kb_versions.py` 复用单场景版本对比逻辑，按场景比较 previous 和 active/candidate 版本的召回结果，提前发现局部召回退化。
- 容量评估：`scripts/tools/capacity_estimate.py` 基于当前 loader/splitter 估算不同 chunk 规模下的存储、rerank 和 prompt 压力。
- 真实链路验收：`scripts/acceptance_smoke.py` 通过真实 HTTP/WebSocket 服务检查页面、管理接口和流式问答事件。

全量重建知识库时，推荐把入库质量检查放在激活前：

```powershell
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
```

执行顺序是：

```text
创建 STAGED 版本
  -> FAQ 入库
  -> 文档入库
  -> 生成入库质量报告
  -> 执行入库质量检查
  -> 检查通过后激活版本
```

这样即使 staged 版本已经写入 Milvus，只要报告发现 FAQ/正文冲突、低质量 chunk 或解析
失败，线上 active 版本也不会切换。

当前一期真实链路验收拆成独立脚本：项目守护检查、编译/单测、入库质量、主链路评测、WebSocket 流式验收和 API 合同验收分别执行，便于课堂逐段讲解。

新增工程项目资料问答场景后，多场景评测集扩展到 40 条。最新报告路径为：

```text
reports/evaluation/multi_scenario_smoke_live_40.json
reports/evaluation/multi_scenario_interview_regression_live_16_final.json
reports/performance/multi_scenario_smoke_live_40.json
```

核心结果：

- `errors = 0`
- `recall_at_k = 1.0`
- `mrr = 0.9000`
- `avg_keyword_coverage = 0.9333`
- `faq_direct_accuracy = 1.0`
- `scenario_isolation_accuracy = 1.0`

面试增强回归结果：

- `errors = 0`
- `recall_at_k = 1.0`
- `hit_type_accuracy = 1.0`
- `source_inference_accuracy = 1.0`
- `prompt_profile_accuracy = 1.0`
- `faq_direct_accuracy = 1.0`
- `scenario_isolation_accuracy = 1.0`

性能基线：

- `avg_total_ms = 3444.13`
- `p95_total_ms = 12810.12`
- `avg_first_token_ms = 2478.67`
- `p95_first_token_ms = 5950.4`
- `hit_type_counts = {"faq_direct": 35, "rag": 5}`

新增的在线 trace 字段：

- `first_token_ms`：首 token 耗时，用于衡量页面真实体感；
- `stage_timings_ms`：主链路阶段耗时，覆盖意图识别、历史读取、查询扩展、FAQ/doc 检索、上下文构建、LLM 生成和历史写入；
- `slowest_stage`：本次请求最慢阶段，状态页可直接按慢点排查；
- `prompt_profile_name`、`question_category`：用于解释为什么某个问题进入严格模板或普通模板。

FAQ/正文冲突检测使用 `jieba.cut_for_search` 做中文搜索分词。这样“管理员密码重置”和
“忘记密码、绑定邮箱”这类中文近似表达可以被质量检查识别为同一知识依据，避免整段正则
关键词匹配造成误报。这里不是新增检索引擎，只是入库质量检查的轻量文本依据判断。

删除本地资料文件后，不会在在线问答中自动清理 Milvus。应使用离线脚本先预览再执行：

```powershell
python scripts/cleanup_missing_docs.py --scenario enterprise_knowledge
python scripts/cleanup_missing_docs.py --scenario enterprise_knowledge --apply
```

## 9. 常用命令

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
python scripts/check_project_guardrails.py
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --activate
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
python scripts/check_ingestion_quality_gate.py --scenario enterprise_knowledge
python scripts/check_ingestion_quality_gate.py --report reports/ingestion/enterprise_knowledge_phase1_gate_check.json
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json
python scripts/evaluate_core_chain.py --dataset eval_sets/business_depth_regression.json --limit 32
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_smoke_test.json
python scripts/collect_performance_baseline.py --dataset eval_sets/multi_scenario_smoke.json
python scripts/check_performance_gate.py --dataset eval_sets/multi_scenario_smoke.json --limit 40
python scripts/check_performance_gate.py --dataset eval_sets/phase1_performance_baseline.json --limit 8 --no-warmup --output reports/verification/phase1_performance_latest.json --gate-output reports/verification/phase1_performance_gate_latest.json
python scripts/kb/compare_all_kb_versions.py --dataset eval_sets/business_depth_regression.json --per-scenario-limit 2 --output reports/verification/kb_version_compare_all_latest.json
python scripts/evaluate_followup_chain.py --dataset eval_sets/multi_turn_followup_regression.json
python scripts/rebuild_kb_version.py --scenario cross_border_risk --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario tender_contract_risk --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario insurance_claims --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario engineering_project_qa --new-version --force --quality-gate --activate
在 LangSmith 中筛选低分、信息不足或人工反馈样本，标注后加入 Dataset。
python scripts/tools/capacity_estimate.py --scenario enterprise_knowledge
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000
python -m pytest tests -q
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

## 10. 配置与保护

当前主线配置来源是 `.env + scenario.toml`，不再引入额外配置入口，避免学生在环境启动时面对多套配置来源。

管理令牌是必需配置：

```text
ADMIN_API_TOKEN=<随机长令牌>
API_RATE_LIMIT_PER_MINUTE=120
```

管理接口需要传 `X-Admin-Token`，状态页页面也提供了令牌输入框。命令行验收脚本默认
读取 `.env` 中的 `ADMIN_API_TOKEN`，避免把真实令牌写入终端历史和报告。

## 11. 页面入口

- 问答页：http://127.0.0.1:8000/
- 状态页：http://127.0.0.1:8000/admin

本文档只描述当前优化后架构。历史方案分析见 `docs/archive/optimization_implementation.md` 和 `docs/legacy_retirement.md`。

