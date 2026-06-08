# RAG Platform WSL Docker Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start KnowForge RAG Platform under WSL/Docker with a dedicated Docker Compose project named `rag-platform`.

**Architecture:** Use the existing `docker-compose.yml` as the source of truth for MySQL, etcd, MinIO, Milvus, and FastAPI. Add a small optional override file only if container names must also use the `rag-platform-*` prefix, because the base compose file currently hardcodes `knowforge-*` container names.

**Tech Stack:** WSL 2, Docker Desktop Linux engine, Docker Compose, FastAPI, MySQL 8.4, Milvus 2.6.8, MinIO, etcd, local BGE models.

---

### Current Findings

**Files:**
- Read: `docker-compose.yml`
- Read: `Dockerfile`
- Read: `.env.compose.example`
- Read: `.env.local.example`
- Read: `README.md`

- Docker CLI is installed on Windows, but Docker Desktop Linux engine is not running.
- `wsl -l -v` currently shows only `docker-desktop`, state `Stopped`, version `2`.
- `docker-compose.yml` already defines `mysql`, `etcd`, `minio`, `milvus`, and `api`.
- The base compose file hardcodes container names as `knowforge-mysql`, `knowforge-etcd`, `knowforge-minio`, `knowforge-milvus`, and `knowforge-api`.
- Running `docker compose -p rag-platform up` will create a Compose project named `rag-platform`, but the hardcoded `container_name` values will still remain `knowforge-*`.

### Task 1: Start WSL/Docker Runtime

**Files:**
- Modify: none

- [ ] **Step 1: Start Docker Desktop or its WSL backend**

Run from PowerShell:

```powershell
wsl -d docker-desktop
```

Expected: Docker Desktop WSL backend starts, or opens an interactive shell.

- [ ] **Step 2: Verify WSL state**

Run:

```powershell
wsl -l -v
```

Expected: `docker-desktop` state becomes `Running`.

- [ ] **Step 3: Verify Docker engine**

Run:

```powershell
docker version
```

Expected: both `Client` and `Server` sections appear.

### Task 2: Decide Naming Mode

**Files:**
- Optional create: `docker-compose.rag-platform.yml`

- [ ] **Step 1: Use Compose project name only if container names do not matter**

Run:

```powershell
docker compose -p rag-platform config
```

Expected: Compose validates successfully, but containers remain named `knowforge-*` because `container_name` is fixed in `docker-compose.yml`.

- [ ] **Step 2: Create override if containers must be named `rag-platform-*`**

Create `docker-compose.rag-platform.yml` with:

```yaml
services:
  mysql:
    container_name: rag-platform-mysql
  etcd:
    container_name: rag-platform-etcd
  minio:
    container_name: rag-platform-minio
  milvus:
    container_name: rag-platform-milvus
  api:
    container_name: rag-platform-api
```

Then use:

```powershell
docker compose -p rag-platform -f docker-compose.yml -f docker-compose.rag-platform.yml config
```

Expected: Compose validates and final container names use `rag-platform-*`.

### Task 3: Prepare Compose Environment

**Files:**
- Create or modify: `.env`

- [ ] **Step 1: Copy Compose environment template**

Run:

```powershell
Copy-Item .env.compose.example .env
```

Expected: `.env` exists and uses container network values: `MYSQL_HOST=mysql`, `MILVUS_URI=http://milvus:19530`, model paths under `/app/models`.

- [ ] **Step 2: Replace required secrets**

Edit `.env`:

```text
DASHSCOPE_API_KEY=<real usable DashScope or OpenAI-compatible key>
ADMIN_API_TOKEN=<random long token>
```

Expected: Startup preflight will not reject placeholder values.

### Task 4: Start Infrastructure First

**Files:**
- Modify: none

- [ ] **Step 1: Pull and start dependencies**

Run without container-name override:

```powershell
docker compose -p rag-platform up -d mysql etcd minio milvus
```

Or with override:

```powershell
docker compose -p rag-platform -f docker-compose.yml -f docker-compose.rag-platform.yml up -d mysql etcd minio milvus
```

Expected: MySQL, etcd, MinIO, and Milvus start.

- [ ] **Step 2: Check health**

Run:

```powershell
docker compose -p rag-platform ps
```

Expected: `mysql` healthy, `minio` healthy, and `milvus` healthy after its start period.

### Task 5: Build Knowledge Base Before API Startup

**Files:**
- Read/write runtime data: `.index_manifest/`, `reports/`, Milvus volumes

- [ ] **Step 1: Rebuild at least the default scenario**

Run:

```powershell
docker compose -p rag-platform run --rm api python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
```

Expected: FAQ/doc chunks are written to Milvus, quality gate passes, and an active KB version is created.

- [ ] **Step 2: Rebuild all scenarios for full demo**

Run:

```powershell
$scenarios = 'enterprise_knowledge','saas_support','equipment_ops','compliance_qa','cross_border_risk','tender_contract_risk','insurance_claims','engineering_project_qa'
foreach ($s in $scenarios) {
    docker compose -p rag-platform run --rm api python scripts/rebuild_kb_version.py --scenario $s --new-version --force --quality-gate --activate
}
```

Expected: all 8 scenario manifests have active versions.

### Task 6: Start API

**Files:**
- Modify: none

- [ ] **Step 1: Start API container**

Run:

```powershell
docker compose -p rag-platform up -d api
```

Expected: API container starts after MySQL and Milvus are healthy.

- [ ] **Step 2: Check API logs**

Run:

```powershell
docker compose -p rag-platform logs -f api
```

Expected: startup preflight passes, retrieval stack warms up, and Uvicorn listens on port `8000`.

### Task 7: Verify Runtime

**Files:**
- Writes reports under: `reports/verification/`

- [ ] **Step 1: Check health endpoint**

Run:

```powershell
curl http://127.0.0.1:8000/health
```

Expected: HTTP 200 health response.

- [ ] **Step 2: Run API smoke**

Run:

```powershell
docker compose -p rag-platform run --rm api python scripts/api_e2e_smoke.py --base-url http://api:8000
```

Expected: API contract smoke passes.

- [ ] **Step 3: Run page/WebSocket smoke from host**

Run:

```powershell
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

Expected: page and WebSocket stream smoke pass.

### Task 8: Alternative Local API Mode

**Files:**
- Create or modify: `.env`

- [ ] **Step 1: Use local template instead of compose API template**

Run:

```powershell
Copy-Item .env.local.example .env
```

Expected: `.env` uses `MYSQL_HOST=localhost`, `MILVUS_URI=http://localhost:19530`, and model paths under `models/...`.

- [ ] **Step 2: Start only dependency containers**

Run:

```powershell
docker compose -p rag-platform up -d mysql etcd minio milvus
```

Expected: dependencies run in Docker, API runs on Windows host.

- [ ] **Step 3: Start API on host**

Run:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Expected: API listens at `http://127.0.0.1:8000`.

### Recommendation

Use full Compose mode first:

```powershell
Copy-Item .env.compose.example .env
docker compose -p rag-platform up -d mysql etcd minio milvus
docker compose -p rag-platform run --rm api python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
docker compose -p rag-platform up -d api
```

If the requirement is that visible container names must start with `rag-platform`, add `docker-compose.rag-platform.yml` before starting services.
