# 本地与 Docker 环境启动说明

当前项目不提供技术降级路径。Milvus、MySQL、本地 embedding/reranker 模型、真实可用的
LLM Key 和 active 知识库版本都是运行前置条件。

## 1. 准备环境变量

先选择运行模式，再复制样例文件。两种模式不要混用。

本机启动 API，Docker 只跑 MySQL/Milvus：

```powershell
Copy-Item .env.local.example .env
```

API 也放在 Docker Compose 中运行：

```powershell
Copy-Item .env.compose.example .env
```

必须修改：

```text
DASHSCOPE_API_KEY=真实可用的模型服务 Key
ADMIN_API_TOKEN=随机长令牌
```

`DASHSCOPE_API_KEY` 必须能完成真实调用。项目启动前会做一次极小的 LangChain
ChatOpenAI 预检，账号欠费、Key 无权限、模型名错误或 `DASHSCOPE_BASE_URL` 配错都会导致启动失败。

LangSmith 是企业观测与评测主入口，但本地教学 smoke 可以先关闭。未配置
`LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 时，状态页只显示 LangSmith 未启用；
这不影响本地页面、检索和 WebSocket 验收。

本机直接运行时，`MYSQL_HOST` 和 `MILVUS_URI` 可以使用：

```text
MYSQL_HOST=localhost
MYSQL_PORT=3306
MILVUS_URI=http://localhost:19530
```

Docker Compose 中运行 API 时，使用：

```text
MYSQL_HOST=mysql
MYSQL_PORT=3306
MILVUS_URI=http://milvus:19530
```

如果你改用仓库里的 `docker-compose.milvus.yml`，它把 MySQL 映射到宿主机 `3307`，这时本机运行 API
才应该把 `.env` 改成：

```text
MYSQL_HOST=localhost
MYSQL_PORT=3307
MILVUS_URI=http://localhost:19530
```

不要把这两套模式混着写。`MYSQL_HOST=localhost` / `MYSQL_PORT=3307` / `MILVUS_URI=http://localhost:19530`
说明你准备走“本机 API + 宿主机端口”模式；`MYSQL_HOST=mysql` / `MYSQL_PORT=3306` /
`MILVUS_URI=http://milvus:19530` 说明你准备走“compose 网络”模式。混用以后，最常见现象
就是 Docker 明明启动了，但 API 还是连不上。

## 2. 准备本地模型

必须存在：

```text
models/bge-m3
models/bge-reranker-large
```

这两个目录缺失时服务启动会失败。

## 3. 启动基础设施

只启动 MySQL + Milvus：

```powershell
docker compose up -d mysql etcd minio milvus
```

启动 API：

```powershell
docker compose up -d api
```

本机启动 API：

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## 4. 初始化知识库

首次运行或资料变更后，执行全量入库：

```powershell
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario saas_support --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario equipment_ops --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario compliance_qa --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario cross_border_risk --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario tender_contract_risk --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario insurance_claims --new-version --force --quality-gate --activate
python scripts/rebuild_kb_version.py --scenario engineering_project_qa --new-version --force --quality-gate --activate
```

## 5. 验收

静态验收：

```powershell
python scripts/check_project_guardrails.py
python -m unittest discover -s tests -p "test_*.py"
```

真实链路验收：

```powershell
python scripts/check_langchain_stack.py
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

`api_e2e_smoke.py` 只要求 LangSmith 状态接口结构正确，不强制 `enabled=true`。正式企业化演示前，
再通过状态页或脚本输出里的 `langsmith_enabled` 确认是否已经启用。

多场景评测：

```powershell
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 40 --output reports/evaluation/multi_scenario_smoke_live_40.json
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_smoke_live_40.json
```

## 6. 常见问题

`Milvus 不可连接：localhost:19530`

说明 Milvus 没启动，或者 `.env` 中 `MILVUS_URI` 写错。

`MySQL 不可连接：localhost:3307`

先不要只盯着“端口拒绝连接”，先确认自己到底启动的是哪套 compose：

- 标准 `docker-compose.yml`：MySQL 对外是 `3306`
- `docker-compose.milvus.yml`：MySQL 对外是 `3307`

如果 `.env` 写的是 `3307`，但你启动的是标准 `docker-compose.yml`，诊断脚本会把它标记为
“Compose / .env 对齐失败”。

`ACTIVE_KB_VERSION 不存在于版本清单`

说明 `.env` 写死了不存在的版本，或当前场景还没有执行入库激活。

`LLM 服务不可用：请检查 DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL 和 LLM_MODEL`

说明 Key、模型名或模型服务地址无法完成真实调用。当前项目会在启动前执行一次极小的
LangChain ChatOpenAI 调用，不再只检查 Key 是否“有值”。先更换可用 Key，再重新执行：

```powershell
python scripts/check_langchain_stack.py
```

`模型目录不存在`

说明 `EMBEDDING_MODEL_PATH` 或 `RERANKER_MODEL_PATH` 配置不对，或者本地模型没有挂载到容器。
