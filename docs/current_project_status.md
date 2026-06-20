# 当前项目状态

本文用于说明当前仓库的交付边界，避免把教学/演示型 RAG 平台误读为完整生产 SaaS。

## 定位

KnowForge RAG Platform 当前定位是企业级多场景 RAG 教学与作品集项目。核心目标是展示：

- LangChain + Milvus Hybrid Search 的 RAG 主链路；
- 多场景、多 source、知识库版本和数据隔离；
- FAQ 直出、文档 RAG、表格证据、Prompt Profile 和引用补强；
- 入库质量、评测回归、性能基线、接口 smoke 和验收报告。

## 已验证能力

当前本地快照已通过：

```powershell
python -m compileall app.py qa_core scripts tests
python -m pytest tests -q
python scripts/check_project_guardrails.py
python scripts/check_docs_consistency.py
```

其中单元测试覆盖检索过滤、上下文构建、Prompt Profile、API 保护和场景/意图等纯逻辑能力。

## 运行模式

项目支持两种运行方式：

- 本机 API 模式：复制 `.env.local.example`，FastAPI 在宿主机运行，MySQL/Milvus 通过 localhost 端口访问。
- Compose API 模式：复制 `.env.compose.example`，FastAPI 在 compose 网络内运行，通过 `mysql`、`milvus` 等 service 名访问依赖。

不要混用两套配置。`scripts/tools/check_local_runtime.py` 会识别当前 `.env` 属于哪种模式，并输出端口、容器和路径诊断。

## 未覆盖边界

当前项目没有把以下能力做成生产级闭环：

- 用户登录、JWT、租户身份可信注入；
- 生产级 CI/CD 和多环境发布；
- 高并发压测、容量自动扩缩和故障演练；
- 企业权限系统、审计系统和密钥托管系统集成。

代码中已有 tenant、dataset、visibility、allowed_roles 的检索过滤能力，但请求身份来源仍需由可信网关或认证层提供。直接把前端传入的 tenant/role 当作生产安全边界是不充分的。

## 展示建议

对外展示时，优先演示：

1. 多场景切换和 source 边界；
2. FAQ 直出与文档 RAG 的分流；
3. 表格行证据和引用补强；
4. 知识库版本、入库质量和评测 gate；
5. Bad Case 如何进入 LangSmith/评测闭环。

不建议把本项目包装成已经具备完整生产安全体系的平台。
