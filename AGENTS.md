# KnowForge RAG VP Agent Guide

This file is the entry point for coding agents working on this deployment.
Keep it short, factual, and close to the current server state.

## Project Root

- Remote root: `/root/RAG/knowforge-rag-vp`
- Public path: `http://117.72.244.79/projects/knowforge-rag-platform-vp/`
- FastAPI service: `knowforge-rag-vp-api`
- Nginx is used as the public reverse proxy.
- Docker services provide Milvus, MySQL, MinIO, and etcd.

## Current Frontend Location

The frontend is not a separate React/Vue/Vite app right now.
It is a static frontend served by FastAPI from the `static/` directory.

- Main page: `static/index.html`
- Admin page: `static/admin.html`
- Architecture page: `static/architecture.html`
- Main JS boot entry: `static/js/app.js`
- API/WebSocket client: `static/js/api.js`
- Chat flow: `static/js/chat.js`
- Session/history UI: `static/js/session.js`
- Shared state and base path: `static/js/state.js`
- Admin dashboard JS: `static/js/admin.js`
- Main styles: `static/css/*.css`
- Course/docs pages: `static/docs/`

Important current frontend constant:

```js
const API_BASE_URL = '/projects/knowforge-rag-platform-vp';
```

Keep this prefix aligned with the Nginx location when changing deployment paths.

## Backend Entry Points

- App entry: `app.py`
- Page routes: `qa_core/api/pages.py`
- Chat/history/WebSocket routes: `qa_core/api/chat.py`
- Admin routes: `qa_core/api/admin.py`
- KB version routes: `qa_core/api/kb_versions.py`
- Settings: `qa_core/config/settings.py`
- Runtime preflight: `qa_core/config/preflight.py`
- Chat history storage: `qa_core/memory/history.py`

The current chat stream uses:

- `POST /api/create_session`
- `GET /api/history/{session_id}`
- `DELETE /api/history/{session_id}`
- `WS /api/stream`

The admin page uses `X-Admin-Token` for protected admin APIs.

## Next Version Goals

The next product iteration should focus on these four items:

1. Multi-turn conversation display
   - Show complete multi-turn messages in the main chat timeline, not only the compact history list.
   - Preserve session switching and make old turns reload visibly.
   - Likely files: `static/js/session.js`, `static/js/chat.js`, `static/js/render.js`, `qa_core/api/chat.py`, `qa_core/memory/history.py`.

2. Frontend page refactor
   - Keep the current static frontend unless there is a clear need to introduce a build system.
   - Split large UI behavior by feature before redesigning visuals.
   - Likely files: `static/index.html`, `static/js/*.js`, `static/css/*.css`.

3. Add create/upload document capability in admin
   - Resource upload is an admin/status-page responsibility, not a main chat-page responsibility.
   - The main page should consume the active knowledge base and focus on question answering, classification, source tracing, and diagnostics.
   - Add the upload flow to `static/admin.html` and `static/js/admin.js`, then expose a protected backend ingestion endpoint.
   - Decide where uploaded files live before implementation: local disk, MinIO, or scenario data packs.
   - Reuse existing ingestion/reporting code where possible.
   - Likely areas: `qa_core/api/`, `qa_core/indexing/`, `scripts/kb/`, `reports/ingestion/`, `static/admin.html`, `static/js/admin.js`.

4. Refine admin status page
   - Expand health/status detail beyond the current lightweight dashboard.
   - Include service status, vector DB status, active scenario, active KB version, ingestion status, API key readiness without exposing secrets, and recent errors.
   - Likely files: `static/admin.html`, `static/js/admin.js`, `static/css/admin.css`, `qa_core/api/admin.py`.

## Development Commands

Check remote status:

```bash
systemctl is-active nginx knowforge-rag-vp-api docker
docker ps
free -h
df -h /
```

Restart the API after backend changes:

```bash
systemctl restart knowforge-rag-vp-api
```

Reload Nginx after proxy changes:

```bash
nginx -t && systemctl reload nginx
```

Follow API logs:

```bash
journalctl -u knowforge-rag-vp-api -f
```

## Working Rules

- Do not commit secrets, API keys, admin tokens, or model files.
- Do not upload the local embedding model directory to GitHub.
- Preserve the `/projects/knowforge-rag-platform-vp` base path unless the Nginx route and frontend constant are changed together.
- Prefer small, verifiable changes and restart only the service affected by the change.
- Before changing ingestion or KB version behavior, inspect existing reports under `reports/ingestion/` and active manifests under `.index_manifest/`.
- For frontend work, verify both the public IP URL and the server-local URL if possible.
