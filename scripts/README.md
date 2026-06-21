# 脚本目录说明

`scripts` 目录已经按学习成本做过收敛：根目录只放一期 RAG 日常开发、入库、评测和发布会高频使用的脚本；低频专项能力放到子目录。

## 根目录主脚本

这些是需要重点掌握的脚本。

| 场景 | 脚本 |
|---|---|
| 环境与项目约束 | `check_langchain_stack.py`、`check_project_guardrails.py`、`check_docs_consistency.py`、`check_chapter_maps.py` |
| 文档和 FAQ 入库 | `rebuild_kb_version.py`、`rebuild_scenarios.py`、`manage_kb_versions.py`、`cleanup_missing_docs.py` |
| 入库质量 | `check_ingestion_quality_gate.py` |
| 主链路评测 | `evaluate_core_chain.py`、`evaluate_followup_chain.py`、`check_evaluation_gate.py`、`check_followup_gate.py` |
| 性能检查 | `collect_performance_baseline.py`、`check_performance_gate.py` |
| 接口验收 | `acceptance_smoke.py`、`api_e2e_smoke.py` |
| Bad Case 沉淀 | LangSmith Annotation/Dataset |
| 公共模块 | `common.py`、`eval_common.py`、`gate_utils.py` |

## Docker 测试部署

本机验收推荐使用全 Docker 模式，避免宿主机和容器两套网络视角混用：

```powershell
if (!(Test-Path .env.compose)) { Copy-Item .env.compose.example .env.compose }
notepad .env.compose
.\scripts\deploy_docker.ps1
```

`deploy_docker.ps1` 会按顺序启动 MySQL/Milvus、构建 API 镜像、初始化 active 场景知识库，
最后启动 API。新环境不能先启动 API 再入库，因为 API 的 preflight 会检查 active KB 版本。
脚本会提前创建 `logs/` 和 `reports/` 目录，避免 Docker 把缺失的宿主机目录挂成不可用路径。
知识库版本与入库 manifest 状态写入 MySQL，不再维护本地 manifest 目录。

## 讲义站点

讲义只保留一条发布链路：`docs/` Markdown 通过 MkDocs 构建到 `site/`，FastAPI 的 `/project-docs`
只读取 `site/`；`/docs` 保留给 Swagger API 文档。不要再恢复 `static/docs/`、`scripts/build_docs.py` 或根目录独立 Markdown 转 HTML
脚本，避免同一份讲义出现两套 HTML 输出。

## 子目录专项脚本

这些脚本不要求第一遍学习时掌握，只在对应专题使用。

| 目录 | 用途 | 代表脚本 |
|---|---|---|
| `tools/` | 本地诊断、容量估算、依赖锁生成 | `check_local_runtime.py`、`capacity_estimate.py`、`generate_requirements_lock.py` |
| `kb/` | 知识库多版本召回对比 | `compare_kb_versions.py`、`compare_all_kb_versions.py` |
| `ocr/` | 扫描件 OCR 离线处理和复核后提升 | `run_offline_ocr.py`、`promote_ocr_candidates.py` |
| `enterprise_overlay/` | 企业仿真资料增强包治理 | `analyze_enterprise_data_realism.py`、`build_enterprise_overlay_dataset.py`、`run_enterprise_overlay_activation.py` |

## 常用命令

```powershell
python scripts/check_project_guardrails.py
python scripts/check_chapter_maps.py
python -m mkdocs build
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
python scripts/rebuild_scenarios.py --reset-collections  # 新环境或 schema 变化：重置 collection 并初始化 8 个场景
python scripts/rebuild_scenarios.py                      # 已有知识库：保留 collection，只刷新 8 个场景的新版本
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20
python scripts/check_evaluation_gate.py --dataset eval_sets/multi_scenario_smoke.json --limit 20
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

章节跟敲代码验收使用目标 Conda 环境运行。第 08 章之后会加载真实
`langchain-milvus`、Embedding/Reranker 和 Milvus 相关链路，一次性跑完整套测试耗时较长；
需要定位问题时优先按章节运行：

```powershell
C:\ProgramData\anaconda3\envs\knowforge-rag\python.exe -m unittest discover -s codealong\chapters\ch08_milvus_hybrid_search\tests
```

Docker Compose 模式下，先保证 `.env.compose` 已经从 `.env.compose.example` 生成并填写真实配置。新环境首次初始化 8 个场景：

```powershell
docker compose --env-file .env.compose up -d mysql etcd minio milvus
docker compose --env-file .env.compose build api
docker compose --env-file .env.compose run --rm api python scripts/rebuild_scenarios.py --reset-collections
```

如果之前已经存在知识库，只是资料内容变化，重建时不要删除 collection：

```powershell
docker compose --env-file .env.compose run --rm api python scripts/rebuild_scenarios.py
```

专项命令示例：

```powershell
python scripts/tools/capacity_estimate.py --scenario enterprise_knowledge
python scripts/kb/compare_all_kb_versions.py --dataset eval_sets/business_depth_regression.json --per-scenario-limit 2
python scripts/ocr/run_offline_ocr.py --input-dir incoming_scans --output-dir reports/ocr/batch_001
python scripts/enterprise_overlay/check_enterprise_overlay_readiness.py
```

## 学习建议

第一遍只看根目录主脚本，理解“入库 -> 入库质量检查 -> RAG 回归验收 -> 接口验收”闭环。

第二遍再看 `kb/` 和 `tools/`，用于解释版本对比、容量评估和本地诊断。

`ocr/` 与 `enterprise_overlay/` 属于增强专题，不影响一期 RAG 主链路闭环。
