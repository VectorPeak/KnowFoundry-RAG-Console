# Changelog

本文件记录仓库阶段版本，不属于学生讲义正文。

## v1.1.0-codealong-complete

发布日期：2026-06-12

### 阶段定位

完成 `codealong/` 跟敲型项目闭环。当前版本适合用于从第 05 章开始，带学生按章节逐步实现企业级 RAG 项目的核心链路。

### 新增内容

- 新增 `codealong/chapters/ch05_intent_classification`
- 新增 `codealong/chapters/ch06_retrieval_strategy`
- 新增 `codealong/chapters/ch07_query_rewrite_variants`
- 新增 `codealong/chapters/ch08_milvus_hybrid_search`
- 新增 `codealong/chapters/ch09_qaservice_orchestration`
- 新增 `codealong/chapters/ch10_rag_pipeline`
- 新增 `codealong/chapters/ch11_prompt_engineering`
- 新增 `codealong/chapters/ch12_fastapi_service`
- 新增 `codealong/chapters/ch13_preflight_checks`
- 新增 `codealong/chapters/ch14_kb_versioning`
- 新增 `codealong/chapters/ch15_data_isolation`
- 新增 `codealong/chapters/ch16_ingestion_pipeline`
- 新增 `codealong/chapters/ch17_quality_evaluation`
- 新增 `codealong/chapters/ch18_test_system`
- 新增 `codealong/chapters/ch19_observability_tracing`
- 新增 `codealong/run_all_tests.py`
- 新增 GitHub Actions：`.github/workflows/codealong-ci.yml`
- 新增 `pytest.ini`，保证本机直接运行 `pytest tests -q` 时能稳定导入 `qa_core`

### 验证结果

- `python codealong\run_all_tests.py` 通过
- `pytest tests -q` 通过：66 passed，13 subtests passed
- `python scripts\check_project_guardrails.py` 通过
- `python scripts\check_no_polyfill_io.py` 通过
- `docker compose --env-file .env.compose config --quiet` 通过
- `python -m compileall -q codealong qa_core scripts app.py` 通过

### 维护说明

`codealong/` 是课堂跟敲目录，由 Git 管理，但不会进入主项目 Docker 镜像。它保留课堂需要学生亲手实现的主链路，不直接复制主项目完整生产级结构。

## v1.0.0-phase1-baseline

发布日期：2026-06-12

### 阶段定位

一期完整项目基线。包含多场景企业级 RAG 项目的主项目代码、讲义、动画、测试和 Docker 部署结构。

### 核心能力

- 多场景 RAG 问答
- FAQ 直出与文档 RAG 链路
- Milvus dense + sparse 混合检索
- source 推断与数据隔离
- 知识库版本管理
- 入库与质量检查
- 回归测试与评测脚本
- FastAPI 服务与前端页面
- Docker Compose 部署

### 维护说明

该标签保留为一期项目基线。后续一期修复走 `phase1-maintenance`，二期能力走 `phase2-graphrag`。
