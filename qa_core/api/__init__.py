"""FastAPI 路由层。

`app.py` 只负责创建 FastAPI 应用、挂载静态资源和注册路由。具体 HTTP/WebSocket
接口放在本包中，避免入口文件继续膨胀。

当前拆分边界：
- `pages.py`：页面入口、健康检查、会话创建；
- `chat.py`：问答、历史、反馈、检索调试、WebSocket 流式输出；
- `admin.py`：追踪、入库报告、评测报告等只读管理诊断；
- `kb_versions.py`：知识库版本查看、激活、归档；
- `dependencies.py`：管理令牌、限流等协议层横切能力。

为什么采用这种拆法：
- 页面、聊天、管理、版本这些接口变化频率不同，放在一起会让 `app.py` 重新变成
  难维护的大文件；
- 二期如果增加 Agent 路由，可以独立新增 `qa_core/api/agent.py`，不污染一期 RAG
  主链路；
- QAService 和 pipeline 仍然是业务核心，API 包只做参数转换和协议适配。
"""
