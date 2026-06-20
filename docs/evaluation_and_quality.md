# RAG 回归验收、入库质量和 LangSmith 观测说明

## 企业路线调整

本项目后续采用 **LangSmith 优先** 的企业路线。通用 LLMOps 平台能力不继续作为授课主线自研：

- tracing / span / LLM 调用观测；
- dataset 管理；
- evaluation experiment；
- annotation / Bad Case 人审；
- prompt experiment；
- 评测趋势和实验对比。

项目自研只保留业务 RAG 核心能力：

- RAG 主链路编排；
- 业务 source 推断；
- 数据权限过滤；
- 知识库版本规则；
- FAQ 命中策略；
- Prompt Profile 路由；
- 检索后处理；
- 表格/文档解析策略；
- 领域评测指标。

详细迁移边界见 [LangSmith 企业路线与自研边界](langsmith_enterprise_route.md)。

## 已实现内容

当前项目保留三类轻量质量能力，其中 Trace、评测实验和 Bad Case 平台能力由 LangSmith 承接：

1. RAG 回归验收：`scripts/evaluate_core_chain.py` + `scripts/check_evaluation_gate.py`
2. 入库质量报告：`scripts/rebuild_kb_version.py` 或 `scripts/check_ingestion_quality_gate.py`
3. LangSmith 观测：`qa_core.observability.langsmith_adapter`、`/api/admin/langsmith`、`/admin` 状态卡片。

这三类能力解决的问题不同：

- RAG 回归验收回答“检索和回答效果有没有退化”。
- 入库质量报告回答“知识库资料是否适合被检索”。
- LangSmith Trace 回答“某一次线上回答为什么会这样”。

## RAG 回归验收

### 数据集

默认评测集位于：

```bash
eval_sets/multi_scenario_smoke.json
```

面试增强回归集位于：

```bash
eval_sets/multi_scenario_interview_regression.json
```

多轮追问回归集位于：

```bash
eval_sets/multi_turn_followup_regression.json
```

业务深度回归集位于：

```bash
eval_sets/business_depth_regression.json
```

两套评测集的定位不同：

- `multi_scenario_smoke.json` 覆盖 8 个场景的基本 FAQ、文档召回和场景隔离，用来证明主链路没有整体退化。
- `multi_scenario_interview_regression.json` 刻意不传 `source_filter`，要求系统自己根据问题推断 source，并验证高风险问题是否进入正确的 Prompt Profile。
- `multi_turn_followup_regression.json` 使用同一个 session 连续提问，验证历史写入、追问改写、source 推断和后续召回是否闭环。
- `business_depth_regression.json` 覆盖 8 个场景新增的业务深度问题，例如权限回收、退款承诺、受限空间、供应商尽调、HS 归类、合同变更、既往症、强制性条文等。

每条样本可以包含：

- `scenario_id`：业务场景。
- `source_filter`：业务分类过滤。
- `query`：用户问题。
- `expected_keywords`：答案应覆盖的关键词。
- `expected_source_contains`：预期召回来源，可以是文件名、FAQ 标准问题或正文片段。
- `expected_hit_type`：预期命中路径，例如 `faq_direct`。
- `expected_effective_source`：不传 source_filter 时，系统应自动推断出的业务分类。
- `expected_prompt_profile`：高风险或排障类问题应命中的提示词模板，例如 `compliance_guard`、`pricing_guard`、`troubleshooting_steps`。
- `tenant_id`、`dataset_id`、`visibility`、`user_role`：数据隔离参数。

### 执行方式

```bash
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 40 --output reports/evaluation/multi_scenario_smoke_live_40.json
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_smoke_live_40.json
```

增强回归：

```bash
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_interview_regression.json --limit 16 --output reports/evaluation/multi_scenario_interview_regression_live_16_final.json
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_interview_regression_live_16_final.json --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 1.0
```

多轮追问：

```bash
python scripts/evaluate_followup_chain.py --dataset eval_sets/multi_turn_followup_regression.json --output reports/evaluation/multi_turn_followup_live.json
python scripts/check_followup_gate.py --report reports/evaluation/multi_turn_followup_live.json
```

业务深度回归：

```bash
python scripts/evaluate_core_chain.py --dataset eval_sets/business_depth_regression.json --limit 32 --output reports/evaluation/business_depth_regression_live_32.json
python scripts/check_evaluation_gate.py --report reports/evaluation/business_depth_regression_live_32.json --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 0.85
```

默认输出到：

```bash
reports/evaluation/
```

### 指标含义

- `recall_at_k`：预期来源是否出现在 FAQ/doc 检索结果中。
- `mrr`：预期来源越靠前分数越高。
- `avg_keyword_coverage`：最终答案覆盖预期关键词的比例。
- `hit_type_accuracy`：命中路径是否符合预期。
- `source_inference_accuracy`：没有显式选择分类时，系统推断出的 source 是否正确。
- `prompt_profile_accuracy`：不同问题类别是否进入正确提示词模板。
- `followup_rewrite_rate`：追问轮次中有多少问题被改写成独立检索问题。
- `followup_source_accuracy`：追问轮次的 source 推断或继承是否正确。
- `scenario_isolation_accuracy`：多轮追问是否仍停留在当前业务场景，没有跨场景污染。
- `hit_type_counts`：FAQ 直出、RAG、信息不足的分布。
- `avg_elapsed_ms`：端到端平均耗时。

### 分组验收

`check_evaluation_gate.py` 不只检查全局指标，还会从 `rows` 中派生三类分组指标：

- `scenario_metrics`：按业务场景分别检查 Recall@K、MRR、关键词覆盖率和错误率。
- `source_metrics`：按资料分类分别检查召回、排序和答案覆盖，避免某个 source 退化被总体均值掩盖。
- `hit_type_metrics`：按 `faq_direct`、`rag`、`source_boundary`、`scenario_boundary` 等命中路径检查路由准确率。

真实企业 RAG 往往不是整体坏掉，而是某个场景、某类资料或某条路由先退化。分组验收可以把这种局部问题提前暴露出来。

### 质量检查脚本

日常授课和演示按主题分别执行：

脚本会自动读取 `.env` 中的 `ADMIN_API_TOKEN`。只有验收远端临时服务时，才需要显式传
`--admin-token` 覆盖。

```bash
python scripts/check_project_guardrails.py
python scripts/check_evaluation_gate.py --dataset eval_sets/business_depth_regression.json --limit 40
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

课程里不再保留总编排脚本，避免把学习重点从 RAG 主链路转移到脚本调度。需要更完整验收时，可以按需追加：

- 项目守护检查；
- Python 编译；
- 单元测试；
- 全场景缺失文档清理预览；
- 业务深度主链路回归验收；
- 企业 clean overlay 激活后的专用回归验收；
- 多轮追问回归验收；
- 性能回归验收；
- HTTP/WebSocket 接口冒烟。

这样面试或交付时可以直接说明：项目不是靠手工点页面判断可用，而是有可重复执行的质量检查。

### 为什么先 debug 再 stream

评测脚本会先调用 `debug_retrieval`，再调用 `stream_query`。这样可以把问题拆开：

- debug 没召回预期来源：优先查入库、切分、query variants、source_filter、数据隔离、kb_version。
- debug 召回正确但答案不对：优先查 Prompt Profile、上下文选择、模型生成。

## 入库质量报告

### 执行方式

```bash
python scripts/check_ingestion_quality_gate.py --scenario saas_support
```

也可以在全量重建时自动生成：

```bash
python scripts/rebuild_kb_version.py --scenario saas_support --new-version --force --activate
```

全量重建脚本会自动把报告写入：

```bash
reports/ingestion/<scenario_id>/
```

### 报告覆盖项

- 文件扫描数量。
- 成功加载文件数量。
- 不支持文件列表。
- 解析失败文件列表。
- 空文件列表。
- source 到 chunk 数量分布。
- chunk 最小、最大、平均长度。
- 低质量 chunk 列表。
- 重复 chunk 数量。
- FAQ 空问题、空答案、重复问题。
- FAQ source 是否在场景白名单内。
- `embedding_model_version`
- `reranker_model_version`
- `chunk_schema_version`
- `kb_version`
- `tenant_id` / `dataset_id` / `visibility`

### 低质量 chunk 检测规则

当前规则是可解释的工程规则：

- `empty`：chunk 为空。
- `too_short`：chunk 小于 30 字，通常语义不足。
- `too_long`：chunk 过长，召回粒度粗且 prompt 成本高。
- `low_unique_ratio`：字符重复率异常，可能是 OCR 噪声或表格线。
- `duplicate_content`：正文重复，可能造成重复召回。

这些规则不替代人工审核，但能把明显低质资料提前暴露出来。

## LangSmith 观测

### 写入位置

每次 `QAService.preview_query` 直接回答或 `QAService.stream_query` 成功/失败都会写入：

```bash
LangSmith Trace
```

### Trace 字段

主要字段包括：

- `trace_id`
- `session_id`
- `question`
- `answer_preview`
- `hit_type`
- `scenario_id`
- `tenant_id`
- `dataset_id`
- `visibility`
- `source_filter`
- `kb_version`
- `rewritten_query`
- `intent`
- `retrieval`
- `first_token_ms`
- `stage_timings_ms`
- `slowest_stage`
- `prompt_profile_name`
- `question_category`
- `sources_count`
- `top_source_score`
- `elapsed_ms`
- `error`

### 管理接口

```bash
GET /api/admin/langsmith
GET /api/admin/ingestion_reports
GET /api/admin/gate_reports
GET /api/admin/performance_reports
```

企业路线下，Trace 明细、评测实验、人工标注和坏例沉淀都在 LangSmith 中完成。项目本地管理接口只保留：

- LangSmith 配置状态；
- 入库质量报告；
- 质量检查报告；
- 性能回归报告；
- 知识库版本和治理摘要。

## 性能回归验收

性能采集脚本：

```bash
python scripts/collect_performance_baseline.py --dataset eval_sets/multi_scenario_smoke.json --limit 40 --output reports/performance/multi_scenario_smoke_live_40_stage.json
```

一期固定性能基线使用独立评测集，后续每次优化都用同一批样本对比：

```bash
python scripts/check_performance_gate.py --dataset eval_sets/phase1_performance_baseline.json --limit 8 --no-warmup --output reports/verification/phase1_performance_latest.json --gate-output reports/verification/phase1_performance_gate_latest.json
```

这组样本覆盖 8 个冻结场景，并刻意混合 FAQ 直出、文档 RAG 和表格 RAG，避免性能报告只反映某一种命中路径。

脚本默认会先用第一条样本做一次 warmup，不纳入正式统计。原因是本地首次请求可能包含
模型、Milvus collection 和连接池加载成本；报告仍会记录 warmup_row，便于区分冷启动
成本和稳定运行阶段性能。如果需要把冷启动也纳入统计，可以添加 `--no-warmup`。

性能回归脚本：

```bash
python scripts/check_performance_gate.py --report reports/performance/multi_scenario_smoke_live_40_stage.json --gate-output reports/verification/performance_gate_latest.json
```

脚本会检查：

- `error_rate`
- `avg_total_ms`
- `p95_total_ms`
- `avg_first_token_ms`
- `p95_first_token_ms`
- `avg_stage_timings_ms`

`avg_stage_timings_ms` 是必须项。原因是性能问题不能只知道“慢”，还要知道慢在意图识别、
FAQ 检索、文档检索、上下文构建、LLM 生成还是历史写入。

### 状态页面

浏览器访问：

```bash
http://127.0.0.1:8000/admin
```

页面展示：

- 当前知识库版本。
- LangSmith 配置状态。
- 知识库版本清单。
- 入库质量报告。
- 回归状态。
- 性能基线。

请求级 Trace、耗时趋势、Prompt Profile 分布和 Bad Case 复核不再由本地状态页承载，统一在 LangSmith 中查看。

## Bad Case 排查顺序

### 0. 先看场景边界和 source 边界

如果 hit_type 是 `scenario_boundary`，说明用户在当前业务场景中问了明显属于另一个
场景的问题。系统不会自动跨场景检索，只会建议切换场景。

如果 hit_type 是 `source_boundary`，说明用户显式选择的 source 和问题本身明显不匹配。
系统不会自动覆盖用户选择，而是提示切换分类后再问。

这样做是为了避免两个问题：

- 多场景串库：比如在企业知识库里问工程安全交底，不能偷偷查工程项目资料；
- 错误 source 低分检索：比如把“员工报销材料”强行按 HR 检索，只会产生低质量上下文。

### 1. 看 hit_type

- `faq_direct`：说明 FAQ 高置信直出，若答案错，优先查 FAQ 标准答案。
- `rag`：说明进入文档上下文生成，优先查 sources 和 context_count。
- `insufficient_context`：说明没有足够上下文，优先查召回和入库。
- `scenario_boundary`：说明问题明显属于其他业务场景，系统已阻断跨场景检索。
- `source_boundary`：说明用户选择的 source 与问题不匹配，系统已阻断错误分类检索。
- `greeting` / `out_of_scope` / `human_support`：说明意图识别提前截断。

### 2. 看 rewritten_query

追问改写错误会直接影响召回。例如用户问“那费用呢”，如果历史摘要丢失，可能无法改写成完整问题。

### 3. 看 query_variants

如果用户表达和文档表达不一致，query variants 应该生成等价入口。例如“回调失败”和“Webhook 调用失败”。

### 4. 看 source_filter 和 data_scope

如果 source_filter、tenant_id、dataset_id、visibility 或 user_role 过窄，可能导致本来存在的资料被过滤掉。

### 5. 看 top_source_score 和 sources_count

低分或来源为空说明召回质量不足。需要检查入库质量报告、chunk 策略和 embedding 版本。

### 6. 看 prompt_profile

召回正确但答案表达不符合预期，优先检查不同问题类别使用的 Prompt Profile。

### 7. 看 source_inference

如果用户没有选择业务分类，系统会根据当前场景的 `source_patterns` 自动推断 source。
这一步错了，后续 Milvus 表达式就会过滤到错误资料。例如“员工报销需要准备哪些材料”
应该推断到 `finance`，不能因为出现“员工”就落到 `hr`。

### 8. 看 Prompt Profile 回归结果

费用、合同、合规、除外责任、强制性规范这类问题不能只看“召回到了资料”，还要看是否
进入更严格的模板。增强回归集中会断言：

- 客户数据上传外部平台：应进入 `compliance_guard`；
- 制裁名单、强制性规范冲突：应进入 `compliance_guard`；
- 发票数量不一致、验收影响回款：应进入 `pricing_guard`；
- 设备异常升级处理：应进入 `troubleshooting_steps`。

## Bad Case 反馈闭环

用户点击“无用”的反馈会进入人工复核流程，在线链路不会立即根据这条反馈改变答案。
原因是一次反馈只能说明“这次体验不好”，不能自动证明正确答案是什么。

企业路线下，Bad Case 不再通过本地导出/提升脚本维护。
统一流程改为：

```text
LangSmith Trace
  -> Annotation 标注问题原因
  -> Dataset 沉淀为回归样本
  -> Evaluation 运行领域指标
  -> 变更前后对比是否退化
```

人工复核时仍然要保留项目领域字段：

- `expected_hit_type`：预期命中路径，例如 `faq_direct`、`rag`、`source_boundary`、`scenario_boundary`。
- `expected_effective_source`：期望资料分类，用来验证 source 推断或边界提示。
- `expected_prompt_profile`：期望 Prompt 模板，用来验证不同问题类型是否路由到正确模板。
- `expected_keywords`：答案必须包含的关键事实。
- `expected_source_contains`：普通 RAG 样本必须命中的来源片段。

这些字段可以作为 LangSmith dataset example metadata，也可以由项目自定义 evaluator 读取。
项目代码不再复刻 Trace 存储、Bad Case 队列和复核页面。

## 评测趋势对比

状态页 `回归报告` 区域会读取两类报告：

- `reports/evaluation/*.json`：历史评测报告；
- `reports/verification/*evaluation*.json`：本地质量检查生成的最新评测报告。

评测趋势优先在 LangSmith Experiments 中查看。本地只保留关键质量报告，用于记录本次变更是否满足领域指标。

为什么要做趋势：单次评测只能证明“这次达标”，不能证明“比上次没有变差”。企业项目交付时，
更需要能回答“新增资料、改 Prompt、调检索参数之后有没有局部退化”。趋势对比就是为这个问题服务的。

`scripts/api_e2e_smoke.py` 只验证本地服务、场景、知识库版本、入库报告、质量报告和 LangSmith 状态接口是否可用；它不强制本地环境开启 LangSmith tracing。`langsmith_enabled=false` 只表示当前环境没有接入企业观测平台，不代表页面或 RAG 主链路不可用。质量闭环本身由 LangSmith 承担，正式企业演示前再确认 tracing、dataset 和 evaluation 已启用。

## 本地环境诊断

当页面没有流式输出、评测全量报错或 API 验收失败时，先运行：

```bash
python scripts/tools/check_local_runtime.py --require-api --output reports/verification/local_runtime_latest.json
```

它会检查：

- WSL 和虚拟机平台是否真正启用；
- Docker 服务端是否可用；
- Milvus、MySQL、FastAPI 端口是否可连接；
- 本地 embedding/reranker 模型目录是否存在；
- 8 个冻结场景是否都有 active 知识库版本。

这个脚本只诊断，不修复，也不提供降级路径。原因是当前项目就是要展示 Milvus Hybrid、
MySQL 历史、LangChain 主链路和多版本知识库，如果缺少这些前置条件，应该明确失败。

## 交付摘要

当前项目不再保留单独的报告生成脚本。交付摘要直接查看这些独立报告：

- 本地通电诊断；
- 业务深度评测；
- 多轮追问评测；
- 性能基线；
- 缺失文档清理；
- LangSmith Trace/Evaluation 接入状态；
- 面试讲法和下一阶段边界。

## 缺失文档清理闭环

增量入库能处理新增和修改文件，但不能自动感知“某个本地资料已经被删除”。如果不清理，
Milvus 中旧 chunk 仍可能被召回。

全场景预览：

```bash
python scripts/cleanup_missing_docs.py --all-scenarios
```

确认报告后执行：

```bash
python scripts/cleanup_missing_docs.py --all-scenarios --apply
```

报告默认写入：

```bash
reports/ingestion/cleanup_missing_docs_latest.json
```

报告会展示：

- 每个场景检查了多少 manifest 记录；
- 本地已删除但 Milvus 仍保留的文件；
- 受影响 chunk 数量；
- 实际删除 chunk 数量；
- 删除失败记录；
- 下一步建议。

## 面试讲法

可以这样回答：

> 我没有只做一个能聊天的 RAG Demo，而是把 RAG 生产化最容易被问到的治理能力补齐了。入库前后有质量报告，能发现解析失败、低质量 chunk、FAQ 冲突；上线后有 trace_id，能定位一次回答的意图、检索、重排、Prompt 和来源；版本上通过 kb_version 做灰度和回滚；效果上用 Recall@K、MRR 和关键词覆盖做小样本回归。

当前多场景评测集已经扩展到 40 条八场景样本。通过回归验收后应重点观察：

```text
errors = 0
recall_at_k = 1.0
mrr = 0.9000
avg_keyword_coverage = 0.9333
faq_direct_accuracy = 1.0
scenario_isolation_accuracy = 1.0
```

其中 `cross_border_risk`、`tender_contract_risk`、`insurance_claims` 和 `engineering_project_qa` 用于验证更复杂业务背景下的 source 推断、FAQ 直出、文档召回、场景隔离和多版本资料检索。工程项目场景特别适合演示图纸版本、施工规范、验收资料这类多文档、多版本知识库的质量治理。

面试增强回归集当前 16 条样本通过后应重点观察：

```text
errors = 0
recall_at_k = 1.0
hit_type_accuracy = 1.0
source_inference_accuracy = 1.0
prompt_profile_accuracy = 1.0
faq_direct_accuracy = 1.0
scenario_isolation_accuracy = 1.0
```

这组指标更适合回答“系统是否真的能自动判断问题类别、选择检索分类、选择不同提示词模板”，
而不是只证明某个固定 source_filter 下能查到资料。

## 后续扩展建议

- 引入 RAGAS 或 DeepEval 做 LLM-as-judge，但保留当前轻量指标作为快速回归。
- 表格行召回专项回归已经独立为 `eval_sets/table_regression.json`。它用于验证清单、台账、
  字段、状态、金额等问题是否能命中 CSV/Excel 的行级证据，并检查答案是否能引用到表格
  文件、工作表和行号。
- 负样本边界回归已经独立为 `eval_sets/negative_boundary_regression.json`。它用于验证选错
  source、跨场景提问、相似但不应回答的问题不会被系统强行 RAG。
- OCR 作为离线链路接入，不默认进入核心入库。`scripts/ocr/run_offline_ocr.py` 会生成待复核
  Markdown 和 JSON 报告；入库质量报告会统计 `ocr_risk_files`，状态页展示疑似扫描件/OCR
  噪声样例，active 版本必须在人工复核或独立 OCR 清洗后再激活。
- OCR 结果进入知识库前必须走 `scripts/ocr/promote_ocr_candidates.py`。该脚本只接受带有“复核状态：已复核”等人工标记的 Markdown，默认 dry-run，显式 `--apply` 才复制到场景资料目录。
- 知识库版本对比由 `scripts/kb/compare_kb_versions.py` 完成。它复用 debug 检索链路，对比旧版
  和候选版的 top source、预期来源排名和 Recall@K，用于版本激活前发现召回退化。
- 全场景版本对比由 `scripts/kb/compare_all_kb_versions.py` 完成。它按场景分组执行 previous 与 active/candidate 版本对比，适合发布前检查 8 个场景是否存在局部退化。
- 如需接入 OpenTelemetry 或 ELK，应在企业观测平台侧完成，不作为本教学项目主线。
- 将报告和版本清单迁移到 MySQL 管理表，支持多人协作。

