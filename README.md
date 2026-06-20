<div align="center">

# KnowFoundry-RAG-Console | 企业级多场景 RAG 知识平台

**面向企业知识问答、风控审核、设备运维与工程资料检索的 RAG 工程化方案**

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688)
![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00A1EA)
![Hybrid Search](https://img.shields.io/badge/hybrid-search-purple)
![BM25](https://img.shields.io/badge/BM25-sparse-orange)
![RAG](https://img.shields.io/badge/RAG-knowledge--platform-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![MkDocs](https://img.shields.io/badge/docs-MkDocs-blueviolet)

简体中文 | [English](README.md)

</div>

## 项目概述

### 这个项目是什么

本项目是一个面向企业级知识问答与业务资料检索的 RAG 工程化系统。它把 RAG 面试和真实项目里最常见的核心环节串成一条完整链路：资料入库、混合检索、FAQ 直出、重排、Prompt 路由、流式生成、知识库版本治理、质量评测、接口验收和部署文档。

技术主线是 `LangChain + Milvus Hybrid Search + FastAPI`。

项目采用 `LangChain` 组织 RAG 主链路，使用 `Milvus 2.5.x` 承载 dense vector + BM25 sparse 混合检索，通过 `FastAPI / WebSocket` 提供接口与流式问答，并内置多场景切换、数据隔离、入库质量检查、RAG 回归评测和 LangSmith 观测能力。当前一期冻结 8 个业务场景，重点验证企业级 RAG 从“能问答”到“可交付”的工程闭环。

当前业务场景已经冻结为 8 个，一期不再继续新增场景包；后续重点放在资料质量、评测回归、版本治理和二期 Agent 能力。

这个项目的一大亮点是很容易迁移到自己的业务中。得益于可插拔的场景包、检索链路和治理脚本，你可以把它快速改造成企业知识库、客服知识助手、设备运维问答、合同风控审核、保险理赔审核或工程资料检索系统。具体使用策略会在后文 [业务场景](#2-业务场景) 与 [核心功能](#3-核心功能) 中展开。

### 不只是项目，更是一整套思路

比这个项目本身更有价值的，是它背后沉淀的一整套工程化思路：

- 如何把 RAG 从聊天 Demo 拆成可维护的检索、重排、生成、引用和评测模块；
- 如何用 `DEV_SPEC` 式的开发规格约束功能边界、目录结构和验收口径；
- 如何用脚本、测试、guardrail 和 smoke check 保证每次修改后系统仍然可交付；
- 如何把知识库版本、场景隔离、权限 metadata 和评测报告串成上线前的治理闭环；
- 如何在一期 RAG 主链路稳定后，继续向 Agent、GraphRAG、SQL 查询和业务工作流扩展。

学会这套思路后，你可以自己做全新的项目，也可以把它拆成模块迁移到已有业务系统里。上手建议从 [第 1 讲：项目概述](docs/01-project-overview.md) 开始，再按 [最小学习路径](docs/TEACHING_MINIMAL_PATH.md) 和 [标准演示 Runbook](docs/standard_demo_runbook.md) 逐步跑通。

| 模块 | 当前能力 |
| --- | --- |
| 业务场景 | 企业知识库、SaaS 客服、设备运维、合规问答、跨境贸易风控、招投标合同履约、保险理赔审核、工程项目资料问答 |
| 检索链路 | FAQ 高置信直出 + 文档 RAG，Milvus dense/sparse hybrid search，rerank 后构建引用上下文 |
| 知识治理 | 多格式资料入库、OCR 待复核流、知识库版本、active 版本切换、数据隔离 metadata |
| 质量验收 | Recall@K、MRR、关键词覆盖、Prompt 命中、场景隔离、接口 smoke 和交付报告 |
| 部署形态 | Docker Compose 部署，FastAPI API 服务，MkDocs 项目文档，部署态配置与代码仓库隔离 |

建议先从 [最小学习路径](docs/TEACHING_MINIMAL_PATH.md)、[学习与演示总入口](docs/learning_demo_guide.md)、[标准演示 Runbook](docs/standard_demo_runbook.md) 和 [当前项目状态](docs/current_project_status.md) 进入。它们分别回答“先学什么”“怎么演示”“怎么验收”和“当前边界在哪里”。

如果需要从第 05 章开始按章节跟敲项目代码，进入 [codealong/](codealong/README.md)。该目录和主项目源码分开，按章节提供可运行、可测试的小闭环。

## 学习路径减法

第一次学习不要从所有脚本、所有业务场景和所有状态页卡片开始。建议按三层看：

| 层级 | 范围 | 目标 |
| --- | --- | --- |
| 必须掌握 | `app.py`、`qa_core/api`、`qa_core/application`、`qa_core/pipeline`、`qa_core/retrieval`、`qa_core/prompts`、`qa_core/indexing` | 跑通并讲清楚 RAG 主链路 |
| 验收掌握 | `scripts/rebuild_kb_version.py`、`scripts/check_project_guardrails.py`、`scripts/evaluate_core_chain.py`、`scripts/check_evaluation_gate.py`、`scripts/api_e2e_smoke.py`、`scripts/acceptance_smoke.py` | 证明系统可交付 |
| 了解即可 | 企业资料治理、LangSmith Bad Case 沉淀、overlay 激活、OCR 提升、性能检查等专题 | 面试追问或二次扩展时再讲 |

状态页也按这个原则做了减法：只展示 LangSmith 状态、入库质量、知识库版本、回归报告和治理摘要。

## 1. 项目定位

本项目适合用来讲清楚以下能力：

- 如何用 LangChain 组织文档加载、切分、向量化、LLM 调用和聊天历史；
- 如何用 Milvus 2.5.x 内置 BM25 做 dense + sparse 混合检索；
- 如何把 FAQ 标准问答和文档 RAG 组合在一条主链路里；
- 如何做知识库版本、embedding 版本、chunk schema 版本和 active 版本切换；
- 如何做多场景、多 source、多租户数据隔离；
- 如何用 Prompt Profile 控制费用、合规、安全、排障等高风险问题的回答边界；
- 如何用 Recall@K、MRR、关键词覆盖、Prompt 命中率和场景隔离率做回归评测；
- 如何通过 LangSmith Trace 定位 RAG bad case。

一句话介绍：

> 基于 LangChain 和 Milvus Hybrid Search 构建的 KnowFoundry-RAG-Console，支持 FAQ 直出、文档问答、知识库多版本、数据隔离、流式输出、入库质量检查和 RAG 回归验收。

## 2. 业务场景

| 场景 ID | 业务背景 | source 数 | FAQ | 文档 | 简历包装 |
| --- | --- | ---: | ---: | ---: | --- |
| `enterprise_knowledge` | HR、IT、财务制度 | 3 | 8 | 11 | 企业内部知识库智能问答平台 |
| `saas_support` | 账号、计费、开放集成 | 3 | 6 | 11 | SaaS 客服知识库智能助手 |
| `equipment_ops` | 巡检、告警、安全规范 | 3 | 6 | 11 | 制造业设备运维知识助手 |
| `compliance_qa` | 合同、审计、隐私保护 | 3 | 6 | 11 | 企业合规制度智能问答系统 |
| `cross_border_risk` | 海关、制裁、信用证、物流、单证 | 5 | 11 | 15 | 跨境贸易风控 RAG 知识问答平台 |
| `tender_contract_risk` | 招投标、合同、交付、验收、履约风险 | 5 | 11 | 15 | 招投标合规与合同履约 RAG 风控平台 |
| `insurance_claims` | 保单、理赔材料、责任、除外、赔付 | 5 | 10 | 15 | 保险理赔材料审核与 RAG 知识问答助手 |
| `engineering_project_qa` | 图纸、规范、进度、质量、安全资料 | 5 | 11 | 15 | 工程项目资料与施工规范 RAG 问答助手 |

更推荐在简历和面试中主推后四个差异化场景：

- 跨境贸易风控：适合讲海关申报、制裁筛查、信用证和单证一致性；
- 招投标合同履约：适合讲合同风险、交付验收和付款边界；
- 保险理赔审核：适合讲材料审核、责任认定和赔付口径控制；
- 工程项目资料问答：适合讲多文档、多版本、图纸/规范冲突和标准规范检索。

## 3. 核心功能

| 能力 | 当前实现 |
| --- | --- |
| 多场景切换 | `scenarios/<scenario_id>/scenario.toml + faq.csv + data/` 配置化切换 |
| 混合检索 | Milvus dense vector + Milvus 内置 BM25 sparse |
| FAQ 直出 | 高置信 FAQ 直接返回标准答案，低置信进入文档 RAG |
| 文档 RAG | LangChain loader/splitter + parent-child chunk + rerank |
| 表格资料 | CSV/Excel 按表头、工作表、行号和单元格键值转换为行级 Document；表格类问题会优先保留表格行上下文 |
| 多格式样例 | 8 个冻结场景都包含 Markdown、CSV、XLSX、DOCX、PPTX、PDF，便于直接验证多格式入库 |
| 离线 OCR | PaddleOCR + PyMuPDF 生成待复核 Markdown 和 OCR 报告；已复核 Markdown 通过提升脚本进入资料目录，再走版本重建 |
| 意图识别 | FAQ、知识咨询、追问、越界、客服等意图识别 |
| source 推断 | 不手选分类时，根据问题自动推断 source |
| 场景边界 | 问题明显属于其他场景时阻断检索，只提示切换场景 |
| source 边界 | 用户选错分类时阻断错误分类检索，避免低分上下文污染答案 |
| 查询扩展 | 针对知识咨询和追问生成 query variants |
| Prompt 路由 | 费用、合规、排障、总结等问题使用不同 Prompt Profile |
| 多版本知识库 | active 版本检索，支持新版本入库、评测、激活和回滚 |
| 版本对比 | 支持单场景和全场景 base/candidate 召回对比，激活前发现召回退化 |
| 数据隔离 | tenant、dataset、visibility、allowed_roles 写入 metadata 并参与检索过滤 |
| 聊天历史 | MySQL + LangChain `SQLChatMessageHistory` |
| 流式输出 | WebSocket 返回 `start/status/token/end` 事件 |
| LangSmith 观测 | Trace、阶段耗时、首 token、检索诊断、来源引用，RAG 答案缺引用时自动补参考来源 |
| 入库质量检查 | 入库质量报告、FAQ/正文冲突检测、低质量 chunk 检测、表格/OCR 风险统计 |
| 评测回归 | Recall@K、MRR、关键词覆盖、模板命中率、场景隔离率、表格行召回专项回归、负样本边界回归 |
| 性能基线 | 固定 `phase1_performance_baseline.json`，覆盖 8 个场景的 FAQ、文档 RAG 和表格 RAG |
| 就绪总报告 | 汇总表格、负样本、版本对比、性能和接口冒烟，生成一期交付视图 |
| Bad Case 闭环 | LangSmith Trace + Annotation + Dataset，人工确认后进入回归评测 |
| 企业仿真数据包 | `data_packs/enterprise_realistic_pack/` 提供 clean overlay 和 dirty samples，用于拉近教学数据与真实企业资料现场的距离 |

## 4. 技术架构

| 层级 | 方案 |
| --- | --- |
| Web/API | FastAPI + WebSocket |
| RAG 编排 | `qa_core.application.QAService` |
| LangChain | loader、splitter、ChatOpenAI、SQLChatMessageHistory、Milvus 集成 |
| LLM | DashScope OpenAI-compatible API |
| Embedding | 本地 BGE-M3 |
| Rerank | 本地 BGE reranker |
| 向量库 | Milvus 2.5.x，支持内置 BM25 Function / Hybrid Search |
| 稀疏检索 | Milvus `BM25BuiltInFunction` |
| 历史/反馈 | MySQL |
| 前端 | 原生静态页面 + 状态页 |
| RAG 回归验收 | 项目内置脚本 + JSON 报告 |
| 二期边界 | 一期源码不提前放 Agent 预留实现；二期用 LangGraph 新建 Agent 模块，将 RAG Pipeline、GraphRAG、SQL 查询和业务工作流封装为专用 Agent，并可选增加 A2A 协议适配 |

主链路：

```text
浏览器页面
  -> FastAPI / WebSocket
  -> QAService
  -> 场景解析 / 数据域解析
  -> 查询路由 direct_answer / faq_exact / retrieval
  -> 意图识别 / source 推断 / 追问改写
  -> 检索计划生成
  -> FAQ Hybrid 检索
  -> 文档 Hybrid 检索
  -> rerank / 上下文构建
  -> Prompt Profile 路由
  -> LLM 流式生成
  -> MySQL 历史 / LangSmith Trace / feedback
```

## 5. 为什么这样设计

### 为什么不用自研 BM25

当前项目使用 Milvus 2.5.x 的内置 BM25 能力。dense、sparse、版本过滤和数据隔离统一放在 Milvus 中完成，避免维护本地 BM25、RedisSearch 或第二套索引。

### 为什么保留 Milvus 适配层

在线业务检索统一通过 `langchain-milvus` 的 VectorStore 接入 Milvus，`QAService`
和 pipeline 不直接调用 PyMilvus。项目仍保留 `qa_core/retrieval/milvus_compat.py`
作为底层适配层，用来处理 Milvus database 检查、BM25 内置函数和 PyMilvus ORM
连接别名注册。当前稳定组合只保留显式、可读的连接初始化，不在业务链路里加入额外运行时改写。

推荐解释口径：

```text
LangChain / langchain-milvus 负责 RAG VectorStore 抽象；
PyMilvus 负责底层连接现实；
适配层只把 BM25 Function、database 和连接别名接好，不进入业务编排。
```

### 为什么 MySQL 仍然保留

MySQL 不再承担知识检索职责，只保存聊天历史、摘要、反馈和后续可能的管理元数据。知识召回由 Milvus 负责，会话状态由 MySQL 负责，职责更清楚。

### 为什么 FAQ 和文档分开

FAQ 是标准口径，适合高置信直出；文档是解释依据，适合复杂问题补充上下文。二者混在一个检索策略里容易导致标准答案被长文档稀释，或者复杂问题被单条 FAQ 误答。

### 为什么要 Prompt Profile

不同问题风险不同。退款、预算、信用证、保证金、赔付等问题需要 `pricing_guard`；合同、隐私、制裁、安全交底、检验批等问题需要 `compliance_guard`；API 限流、设备告警等问题需要 `troubleshooting_steps`。这类路由必须稳定可测，不能完全交给模型自由发挥。

## 6. 快速启动

先选择运行模式，再准备环境变量。两种模式不要混用。

| 文件 | 是否提交 | 用途 |
|---|---:|---|
| `.env.compose.example` | 是 | Docker Compose 模板，地址使用 `mysql`、`milvus`、`/app/models/...` |
| `.env.compose` | 否 | Docker Compose 实际运行配置，由 `.env.compose.example` 复制后填写 |
| `.env.local.example` | 是 | 本机 API 调试模板，地址使用 `localhost` 和 `models/...` |
| `.env` | 否 | 本机 API 实际运行配置，由 `.env.local.example` 复制后填写 |

项目不再保留 `.env.example`。这个名字无法表达运行模式，容易把容器地址和本机地址混用。

### 6.1 全 Docker 测试

当前如果只是为了验收项目，推荐让 MySQL、Milvus 和 API 都由 Docker Compose 管理。这样
API 容器访问依赖时统一使用 `mysql`、`milvus` 这些 Compose 服务名，不会和宿主机
`localhost` 视角混在一起。

```powershell
if (!(Test-Path .env.compose)) { Copy-Item .env.compose.example .env.compose }
notepad .env.compose
```

必须配置真实可用值：

```text
DASHSCOPE_API_KEY=真实可用的模型服务 Key
ADMIN_API_TOKEN=随机长令牌
```

一键部署当前 active 场景：

```powershell
.\scripts\deploy_docker.ps1
```

如果要一次性初始化 8 个冻结业务场景：

```powershell
.\scripts\deploy_docker.ps1 -AllScenarios
```

脚本执行顺序是：启动 MySQL/Milvus → 构建 API 镜像 → 在 API 容器里重建并激活知识库 →
启动 API。这个顺序不能反过来，因为 API 启动前会检查 active KB 版本；空库直接启动 API
会被 preflight 拒绝。

脚本也会提前创建 `logs/`、`reports/` 两个运行时目录。手动执行
`docker compose` 时也要保证这些目录存在，否则 Windows Docker 可能把缺失的宿主机目录挂成不可用路径。

手动执行等价命令：

```powershell
docker compose --env-file .env.compose up -d mysql etcd minio milvus
docker compose --env-file .env.compose build api
docker compose --env-file .env.compose run --rm api python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
docker compose --env-file .env.compose up -d api
docker compose --env-file .env.compose ps
```

访问：

- 问答页：http://127.0.0.1:8000/
- 状态页：http://127.0.0.1:8000/admin
- 讲义页：http://127.0.0.1:8000/docs

讲义和流程动画由宿主机 `./site` 挂载到容器 `/app/site`。修改 `docs/` 或
`docs/animation/` 后，先执行 `python -m mkdocs build`，刷新 `/docs/...` 即可看到更新；
不需要为了讲义内容重建 API 镜像。

### 6.2 本机 API 调试

本机启动 API，Docker 只跑 MySQL/Milvus：

```powershell
if (!(Test-Path .env)) { Copy-Item .env.local.example .env }
notepad .env
```

本机模式同样必须配置真实可用值：

```text
DASHSCOPE_API_KEY=真实可用的模型服务 Key
ADMIN_API_TOKEN=随机长令牌
```

`DASHSCOPE_API_KEY` 不是形式校验。服务启动前会实际调用一次 OpenAI-compatible
LLM 接口，Key 欠费、无权限、模型名错误或服务地址不可用都会直接启动失败。正式学习和演示前
请先运行 `python scripts/check_langchain_stack.py`，确认模型服务可用。

LangSmith 是企业观测和评测主入口，但本地教学 smoke 不强制开启。未配置
`LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 时，状态页会显示未启用，接口验收仍会通过；
正式企业化演示时再打开 LangSmith tracing。

本地模型必须存在：

```text
models/bge-m3
models/bge-reranker-large
```

启动基础设施：

```powershell
docker compose --env-file .env.compose up -d mysql etcd minio milvus
```

在宿主机启动 API：

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

检查 LangChain、Milvus、MySQL、模型和 LLM 配置：

```powershell
python scripts/check_langchain_stack.py
```

访问：

- 问答页：http://127.0.0.1:8000/
- 状态页：http://127.0.0.1:8000/admin

## 7. 初始化知识库

Docker Compose 模式下，推荐在 API 容器内执行入库脚本。所有
`docker compose --env-file .env.compose ...` 命令都要求项目根目录已经存在
`.env.compose`，仓库只提交 `.env.compose.example`，首次使用前先生成本地配置文件：

```powershell
if (!(Test-Path .env.compose)) { Copy-Item .env.compose.example .env.compose }
notepad .env.compose
```

新环境首次部署，或者 Milvus collection schema 变更后需要重建全部 8 个冻结场景，使用：

```powershell
docker compose --env-file .env.compose up -d mysql etcd minio milvus
docker compose --env-file .env.compose build api
docker compose --env-file .env.compose run --rm api python scripts/rebuild_scenarios.py --reset-collections
```

如果之前已经存在知识库，只是资料内容变化，重建全部 8 个场景时不要删除 collection：

```powershell
docker compose --env-file .env.compose run --rm api python scripts/rebuild_scenarios.py
```

如果只重建某一个已有场景，例如企业知识场景：

```powershell
docker compose --env-file .env.compose run --rm api python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
```

只有在旧 collection schema 不兼容、切换了 hybrid 字段结构，或者明确要清空重建单场景 collection 时，才给单场景命令追加 `--reset-collections`：

```powershell
docker compose --env-file .env.compose run --rm api python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --reset-collections --quality-gate --activate
```

本机 API 调试模式下，首次运行或资料变更后，重建并激活 8 个场景：

```powershell
$scenarios = 'enterprise_knowledge','saas_support','equipment_ops','compliance_qa','cross_border_risk','tender_contract_risk','insurance_claims','engineering_project_qa'
foreach ($s in $scenarios) {
    python scripts/rebuild_kb_version.py --scenario $s --new-version --force --quality-gate --activate
}
```

单独重建某个场景：

```powershell
python scripts/rebuild_kb_version.py --scenario engineering_project_qa --new-version --force --quality-gate --activate
```

## 8. 验收命令

代码与守护检查：

```powershell
python -m compileall app.py qa_core scripts tests
python -m pytest tests -q
python scripts/check_project_guardrails.py
```

最小闭环命令只需要记住这几条：

```powershell
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20 --output reports/evaluation/multi_scenario_smoke_live.json
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_smoke_live.json
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8001
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8001
```

多场景核心评测：

```powershell
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 40 --output reports/evaluation/multi_scenario_smoke_live_40.json
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_smoke_live_40.json
```

面试增强评测：

```powershell
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_interview_regression.json --limit 16 --output reports/evaluation/multi_scenario_interview_regression_live_16_final.json
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_interview_regression_live_16_final.json --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 1.0
```

业务深度评测：

```powershell
python scripts/evaluate_core_chain.py --dataset eval_sets/business_depth_regression.json --limit 32 --output reports/evaluation/business_depth_regression_live_32.json
python scripts/check_evaluation_gate.py --report reports/evaluation/business_depth_regression_live_32.json --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 0.85
```

多轮追问评测：

```powershell
python scripts/evaluate_followup_chain.py --dataset eval_sets/multi_turn_followup_regression.json --output reports/evaluation/multi_turn_followup_smoke.json
python scripts/check_followup_gate.py --report reports/evaluation/multi_turn_followup_smoke.json
```

真实链路验收：

如果 API 在 Docker Compose 中运行，推荐在 API 容器内执行验收脚本，脚本会读取
`.env.compose` 注入的 `ADMIN_API_TOKEN`，不需要把管理令牌写在命令行里。

真实 API 合同验收：

```powershell
docker compose --env-file .env.compose exec api python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000
```

该脚本只检查 LangSmith 状态接口是否可用，不要求本地环境必须开启 LangSmith。是否开启会写入
`details.langsmith_enabled`，方便正式演示前确认。

页面和 WebSocket 验收：

```powershell
docker compose --env-file .env.compose exec api python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

项目里不再保留“一键发布检查”概念，避免把质量证明和脚本编排混在一起。需要证明质量时，按上面的入库质量检查、RAG 回归验收、接口验收脚本分别执行即可。

OCR 复核资料提升：

```powershell
python scripts/ocr/run_offline_ocr.py --input-dir incoming_scans --output-dir reports/ocr/batch_001
python scripts/ocr/promote_ocr_candidates.py --input-dir reports/ocr/batch_001 --scenario engineering_project_qa --source quality
python scripts/ocr/promote_ocr_candidates.py --input-dir reports/ocr/batch_001 --scenario engineering_project_qa --source quality --apply
python scripts/rebuild_kb_version.py --scenario engineering_project_qa --new-version --force --quality-gate --activate
```

固定性能基线：

```powershell
python scripts/check_performance_gate.py --dataset eval_sets/phase1_performance_baseline.json --limit 8 --no-warmup --output reports/verification/phase1_performance_latest.json --gate-output reports/verification/phase1_performance_gate_latest.json
```

全场景版本召回对比：

```powershell
python scripts/kb/compare_all_kb_versions.py --dataset eval_sets/business_depth_regression.json --per-scenario-limit 2 --output reports/verification/kb_version_compare_all_latest.json
```

缺失文档清理预览：

```powershell
python scripts/cleanup_missing_docs.py --all-scenarios
```

如果报告确认无误，再显式执行：

```powershell
python scripts/cleanup_missing_docs.py --all-scenarios --apply
```

Bad Case 反馈闭环统一交给 LangSmith：

```text
Trace 过滤异常样本
  -> Annotation 标注 expected_hit_type / expected_source / expected_keywords
  -> Dataset 沉淀真实线上回归样本
  -> Evaluation 运行领域指标
  -> 变更前后对比关键结果
```

本项目不再维护本地 Bad Case 导出、复核队列和提升脚本。代码重点放在 RAG 主链路、source 推断、权限过滤、知识库版本、FAQ 命中、Prompt Profile、检索后处理和领域评测指标。

本地通电诊断：

```powershell
python scripts/tools/check_local_runtime.py --require-api --output reports/verification/local_runtime_latest.json
```

这份诊断报告现在不只会告诉你“端口不通”，还会额外说明三件事：

- 当前 `.env` 是否符合“本机 API + localhost 端口”模式；
- `docker-compose.yml` 和 `docker-compose.milvus.yml` 分别暴露了哪些宿主机端口；
- `docker ps` 里是否已经有相关容器存在，但没有把 MySQL / Milvus / API 端口映射到宿主机。

如果 `.env` 使用 `mysql`、`milvus` 和 `/app/models/...`，通常说明把 Compose 配置复制到了本机
API 调试配置里。宿主机直接跑诊断时，脚本会把这些容器内视角的端口和路径降级为提示项，避免把
“运行模式不同”误判成项目不可用。全 Docker 模式请检查 `.env.compose`，并优先在 API 容器内执行验收命令。

例如当前仓库里，标准 `docker-compose.yml` 默认把 MySQL 暴露到 `3306`，而 `docker-compose.milvus.yml`
把 MySQL 暴露到 `3307`。如果本机 `.env` 写的是 `MYSQL_PORT=3307`，但你实际启动的是标准
`docker-compose.yml` 那套服务，诊断报告会明确指出是本机配置与容器端口不一致，而不是只给出
一个抽象的“连接失败”。

企业资料真实度分析：

```powershell
python scripts/enterprise_overlay/analyze_enterprise_data_realism.py --output reports/verification/enterprise_data_realism_latest.json
```

当前 `scenarios/` 是可控教学样本。企业仿真数据包位于 `data_packs/enterprise_realistic_pack/`，
其中 `clean_overlay/` 可用于后续入库增强，`dirty_samples/` 只用于资料治理演示，不默认进入
active 知识库版本。详细说明见 [enterprise_data_realism.md](docs/enterprise_data_realism.md)。

clean overlay 预检：

```powershell
python scripts/enterprise_overlay/build_enterprise_overlay_dataset.py --all-scenarios --output reports/verification/enterprise_overlay_build_latest.json
```

这会在 `reports/enterprise_overlay_build/` 下生成临时增强数据集，并复用入库质量报告和入库质量检查。它只验证“增强资料是否具备进入知识库版本重建的资格”，不会修改当前 active 知识库。

dirty samples 治理分析：

```powershell
python scripts/enterprise_overlay/analyze_dirty_enterprise_samples.py --output reports/verification/dirty_enterprise_samples_latest.json
```

它会识别过期口径、FAQ/正文冲突风险、OCR 噪声和表格专用切分需求，并明确这些样本默认不允许 active 入库。

企业 overlay 就绪检查：

```powershell
python scripts/enterprise_overlay/check_enterprise_overlay_readiness.py --output reports/verification/enterprise_overlay_readiness_latest.json
```

它会确认 clean overlay 预检、dirty samples 阻断、资料真实度提升和 overlay 回归评测集覆盖都满足要求。

生成 overlay 上线计划：

```powershell
python scripts/enterprise_overlay/plan_enterprise_overlay_activation.py --output reports/verification/enterprise_overlay_activation_plan_latest.json
```

计划文件只生成 `rebuild_kb_version.py` 命令。需要正式执行计划时，运行：

```powershell
python scripts/enterprise_overlay/run_enterprise_overlay_activation.py --plan reports/verification/enterprise_overlay_activation_plan_latest.json --output reports/verification/enterprise_overlay_activation_run_latest.json
```

增强资料真正激活后，再跑：

```powershell
python scripts/check_evaluation_gate.py --dataset eval_sets/enterprise_overlay_regression.json --limit 24 --output reports/verification/enterprise_overlay_evaluation_latest.json --gate-output reports/verification/enterprise_overlay_evaluation_gate_latest.json --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 0.85
```

这条 overlay 回归作为独立质量检查保留，适合临时排查或资料增强后快速复测。

该报告现在会优先读取最新 live 验收产物，而不是旧的静态报告快照，重点汇总：

- 最新业务深度评测；
- 最新多轮追问评测；
- 最新性能回归结果；
- 最新企业资料真实度、clean overlay 预检和 dirty samples 治理摘要；
- 当前 bad case 复核分层；
- 当前本地通电状态。

这轮又补了两类状态页筛选能力：

- 时间窗口：最近 `30m / 6h / 24h / 7d / 全部时间`
- 定位维度：按 `closure_bucket / session_id / trace_id` 聚焦 bad case 和 trace

现在状态页不只是“看报表”，而是可以直接回答：

- 这次新增问题是最近 24 小时出现的，还是旧历史遗留；
- 某次验收会话里到底是哪条 trace 开始回答变差；
- 这是边界拦截本来就对，还是知识覆盖真的缺了。

## 9. 当前验证结果

最近一次业务深度回归：

```text
reports/evaluation/business_depth_regression_live_32.json
```

关键指标：

```text
total = 32
errors = 0
recall_at_k = 1.0
mrr = 1.0
hit_type_accuracy = 1.0
source_inference_accuracy = 1.0
prompt_profile_accuracy = 1.0
faq_direct_accuracy = 1.0
scenario_isolation_accuracy = 1.0
avg_keyword_coverage = 0.9922
```

项目守护检查已覆盖：

- 禁止恢复旧版 `mysql_qa` / `rag_qa` 主链路；
- 禁止新增第 9 个场景包；
- 禁止 fallback 导入和技术降级路径；
- 检查依赖锁、导入位置和密钥卫生。

## 10. 推荐演示问题

| 场景 | 问题 |
| --- | --- |
| 企业知识库 | 没有预算审批可以先采购再报销吗？ |
| SaaS 客服 | 客户要求退款或赠送额度时能直接答应吗？ |
| 设备运维 | 同一设备反复温度告警要怎么处理？ |
| 合规风控 | 批量导出客户数据需要哪些审批？ |
| 跨境贸易 | HS 编码归类存在争议时能先按客户说法申报吗？ |
| 招投标合同 | 客户只口头确认验收可以申请回款吗？ |
| 保险理赔 | 收款账户和被保险人不一致可以打款吗？ |
| 工程项目 | 安全技术交底只有口头说明可以吗？ |

演示时建议打开 `/admin`，观察命中的 `scenario_id`、`source_filter`、`kb_version`、`prompt_profile`、来源引用和阶段耗时。

## 11. 文档导航

| 文档 | 用途 |
| --- | --- |
| `docs/index.md` | 19 讲系统课程首页、学习优先级和二期规划 |
| `docs/course-outline.md` | 课程大纲、学习路线和 P3 扩展方向 |
| `docs/01-project-overview.md` | 项目概述、环境搭建和生产部署说明 |
| `docs/16-ingestion-pipeline.md` | 文档入库、FAQ 入库和 IndexManifest 增量机制 |
| `docs/17-quality-evaluation.md` | RAG 回归验收、入库质量检查和 Bad Case 沉淀 |
| `docs/19-observability-tracing.md` | LangSmith Trace、阶段耗时诊断和观测闭环 |
| `docs/appendix/` | 8 个技术附录 |

## 12. 二期方向

一期只做 RAG，并且已经闭环：

- 知识库构建；
- 多场景检索；
- FAQ + 文档混合；
- 多版本；
- 数据隔离；
- 流式问答；
- 入库质量检查；
- 评测回归。

二期再引入 Agent，不把 Agent 能力混入一期主链路。二期适合做：

- LangGraph 工作流；
- Router/Planner 统一调度；
- GraphRAG Agent 关系推理；
- SQL Agent 结构化查询；
- Workflow Agent 审批/工单流程；
- Skill Registry；
- Agent 任务状态机；
- 工具权限分级；
- 人工确认与审计；
- Agent Trace 和 Agent 回归验收；
- 模型路由与成本治理；
- A2A 跨 Agent 协作协议适配；
- MCP 外部工具/资源接入适配；
- 工具调用；
- 审批/工单/核查流程模拟；
- 风险处置步骤编排；
- 多角色协同。

后续增强层不抢二期首版主线：

- 多模态入库：复杂 PDF、扫描件、图纸、票据、验收照片，先解析、复核、质检，再版本化入库；
- GraphRAG：只用于合同、理赔、工程规范这类关系链明显的问题，不替代当前 Milvus Hybrid RAG；
- 模型升级实验：在 `bge-m3` 基线稳定后，再评估 Qwen3-Embedding 等新模型。

二期工程治理重点：

- Skill、Prompt、Tool Policy、模型路由和知识库版本都要能灰度和回滚；
- 高风险任务必须进入人工确认；
- Trace、A2A 消息、Agent Card 和报告都要做敏感信息脱敏；
- 长任务要支持状态查询、取消和失败恢复。

## 13. 安全说明

- `.env`、`.env.compose` 不提交真实 Key；
- `.env.local.example`、`.env.compose.example` 只保留占位符；
- `logs/`、`reports/` 默认不作为线上公开材料；
- 当前项目不提供技术降级方案，Milvus、MySQL、本地模型、LLM Key 和 active 知识库版本都是启动前置条件。
