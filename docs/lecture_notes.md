# KnowForge RAG Platform — 授课讲义

> **本文档定位**：教师备课速查手册（13 章凝练版），以代码片段 + 教学要点为主，适合有经验的开发者快速浏览全貌。
>
> **学生自学请用**：[18 讲系统课程](course-outline.md)，每讲包含前置知识讲解、类比说明、Mermaid 图解和代码详解，适合零基础循序渐进学习。
>
> **面试准备**：学完 18 讲的 01/06/09/11/16 后，可阅读 [`interview_playbook.md`](interview_playbook.md) 和 [`interview_faq.md`](interview_faq.md)。

## 目录

1. 项目概述与架构总览
2. 应用入口与路由层
3. 核心服务编排：QAService
4. 意图识别：规则优先 + LLM 补充
5. 检索计划：把意图变成参数
6. 查询改写与变体生成
7. Milvus 混合检索系统
8. 上下文构建与 Prompt 工程
9. 知识库治理：多版本与数据隔离
10. 入库与索引链路
11. RAG 回归验收与入库质量
12. 多场景配置化设计
13. 前后端交互与流式事件

---

## 1. 项目概述与架构总览

### 1.1 项目定位

这是一个**基于 LangChain + Milvus 2.6 Hybrid Search 的多场景 RAG 教学平台**。它不是一个简单的"文档问答 Demo"，而是一个补齐了企业级 RAG 完整工程闭环的项目。

**一句话概括：**
> 用户提问 → 意图识别 → 查询改写 → 检索计划 → FAQ/Doc Milvus Hybrid 检索 → Rerank/去重 → Prompt Profile → LLM 流式生成 → 保存历史 → 追踪日志

### 1.2 技术栈速览

| 层级 | 技术选型 | 作用 |
|------|----------|------|
| API 框架 | FastAPI + WebSocket | HTTP 预检 + 流式问答 |
| RAG 编排 | LangChain (ChatOpenAI, Milvus, SQLChatMessageHistory) | 复用开源生态，避免自研完整框架 |
| 向量数据库 | Milvus 2.6.x (dense + BM25 sparse) | 语义召回 + 关键词召回，一次入库两种检索 |
| Embedding | BGE-M3 (本地部署) | 中文语义向量化 |
| Reranker | BGE Reranker Large (CrossEncoder) | 对召回结果精排 |
| LLM | DashScope / OpenAI-compatible API | 意图分类 + 答案生成 |
| 会话存储 | MySQL (LangChain SQLChatMessageHistory) | 聊天历史、摘要、反馈，不做检索 |
| 配置管理 | .env + scenario.toml | 环境变量 + 场景配置 |

### 1.3 八大数据场景

项目不是八套代码，而是**一套核心引擎 + 八套业务配置**：

| 场景 ID | 业务背景 | 简历项目名 |
|---------|---------|-----------|
| `enterprise_knowledge` | 企业制度、HR、IT 流程 | 企业内部知识库智能问答平台 |
| `saas_support` | 账号、计费、权限、集成 | SaaS 客服智能问答与工单辅助平台 |
| `equipment_ops` | 巡检、告警、维修、备件 | 工业设备运维知识库与故障诊断助手 |
| `compliance_qa` | 隐私、合同、审计、政策 | 企业合规风控知识问答平台 |
| `cross_border_risk` | 海关、制裁、信用证 | 跨境贸易风控 RAG 问答平台 |
| `tender_contract_risk` | 招标、合同、履约 | 招投标合规与合同履约 RAG 风控平台 |
| `insurance_claims` | 保单、理赔、责任 | 保险理赔材料审核与 RAG 问答助手 |
| `engineering_project_qa` | 图纸、规范、验收 | 工程项目资料与施工规范 RAG 问答助手 |

### 1.4 核心模块一览

```
qa_core/
├── api/              # FastAPI 路由层（页面、问答、管理、版本）
├── application/      # 服务编排（QAService）与服务工厂
├── intent/           # 意图识别与问题类别推断
├── retrieval/        # Milvus Store、过滤、重排、检索计划
├── pipeline/         # RAG 主流程、事件、上下文、改写、变体
├── prompts/          # 提示词模板与选择器
├── indexing/         # 文档加载、切分、FAQ 入库、Manifest
├── governance/       # 知识库版本与数据隔离
├── memory/           # 聊天历史、摘要、反馈
├── quality/          # 入库质量、低质量 Chunk、FAQ/正文冲突检测
├── scenarios/        # 多业务场景注册与解析
├── config/           # 配置管理、日志、启动前校验
└── observability/    # 旧版追踪日志；企业路线迁移到 LangSmith
```

### 1.5 设计原则

- **主链路唯一**：复杂问题统一走 WebSocket 流式问答，不提供 HTTP 和 WebSocket 两套实现
- **不降级**：Milvus、MySQL、本地模型、LLM Key、active 知识库版本缺失时直接启动失败
- **规则优先 + LLM 补充**：高频确定场景用规则（快且稳定），模糊场景再用 LLM
- **FAQ 优先 + 文档补充**：FAQ 适合确定答案，文档 RAG 适合整合资料，两者分层不混用
- **可证明可排查**：每次问答都写入 Trace，入库有质量报告，召回有RAG 回归验收

---

## 2. 应用入口与路由层 (app.py + API)

### 2.1 app.py：极薄的入口文件

```python
# app.py — 只做四件事：
# 1. 创建 FastAPI 应用
# 2. 配置 CORS 和静态资源
# 3. 启动时执行环境校验和检索栈预热
# 4. 注册路由

app = FastAPI(title="多场景知识问答教学平台 API")

@app.on_event("startup")
async def warmup_runtime() -> None:
    summary = validate_runtime_environment()  # 校验全部前置条件
    await asyncio.to_thread(warmup_retrieval_stack)  # 预热检索模型

app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(kb_versions.router)
```

**教学要点：**
- 入口文件为什么要保持很薄？因为它是服务启动点，不应堆砌 HTTP、RAG、数据库细节。
- `validate_runtime_environment()` 在启动时一次性检查 Milvus、MySQL、模型目录、LLM Key、场景配置、active 知识库版本等全部前置条件。任一缺失直接启动失败，避免页面看似能打开但提问时才发现核心链路没通电。
- `warmup_retrieval_stack` 预热全部 8 个场景的 FAQ/doc collection，避免第一个用户首次提问时等待模型加载。

### 2.2 路由拆分

| 路由模块 | 职责 | 关键端点 |
|---------|------|---------|
| `qa_core/api/pages.py` | 页面渲染、健康检查、会话创建 | `GET /`, `GET /health`, `POST /api/create_session` |
| `qa_core/api/chat.py` | 问答、历史、反馈、检索诊断 | `POST /api/query`, `WS /api/stream`, `GET /api/history/{id}`, `POST /api/feedback` |
| `qa_core/api/admin.py` | 管理只读接口 | `GET /api/admin/langsmith`, `GET /api/admin/ingestion_reports` |
| `qa_core/api/kb_versions.py` | 知识库版本管理 | `GET /api/kb_versions`, `POST /api/kb_versions/activate` |

### 2.3 轻量预检 vs 流式主链路

```python
# /api/query — 轻量预检：问候、越界、人工客服直接返回，复杂问题返回 None
result = service.preview_query(query, source_filter, session_id, ...)
if result is not None:
    return result  # 直接答案，不消耗 LLM
# 返回 None 时，前端自动走 /api/stream

# /api/stream — 流式主链路：唯一完整问答路径
async for event in stream_events:
    await websocket.send_json(event)
```

**教学要点：**
- 为什么需要两条路径？问候"你好"、越界拦截、人工客服电话这类问题不需要访问 Milvus 或 LLM，`preview_query` 直接返回可以减少延迟和成本。
- 所有复杂业务知识问答统一走 WebSocket 流式主链路，避免 HTTP 和 WebSocket 两套实现产生不一致。

---

## 3. 核心服务编排：QAService

### 3.1 设计理念

`QAService` 是项目最核心的业务编排层。它不关心 HTTP、WebSocket 或页面渲染，只负责把"用户提问"变成"稳定的问答事件"。

```python
class QAService:
    def __init__(self) -> None:
        self.settings = get_settings()   # 只读配置
        self.history = get_history_store()  # 历史存储适配器

    def preview_query(self, ...) -> QAResult | None:
        # 处理问候等可直接返回的答案
        ...

    def stream_query(self, ...) -> Generator[dict[str, Any], None, None]:
        # 完整 RAG 流式问答
        yield from rag_stream_query(...)

    def debug_retrieval(self, ...) -> dict[str, Any]:
        # 检索调试（不调用 LLM 生成）
        return rag_debug_retrieval(...)
```

**教学要点：**
- QAService 保存在应用工厂中做**进程级缓存**（单例），但**不保存任何请求级状态**。所有随用户提问变化的数据都在方法局部变量中。
- 这样做的好处：多用户并发不会互相覆盖；评测脚本可以绕过 HTTP 直接调用 QAService 复现主链路。

### 3.2 完整问答主流程

```
用户提问
  │
  ├─ 第零阶段：FAQ 快速路径
  │    └─ 短问题 + 像标准问答 → 直接查 FAQ Milvus → 精确命中则直出
  │
  ├─ 第一阶段：意图识别
  │    ├─ 问候/越界/人工客服 → 直接答案（规则，不走 LLM）
  │    └─ 需要 RAG → 继续
  │
  ├─ 第二阶段：查询改写（仅 FOLLOW_UP 时）
  │    └─ "那审批呢" → "入职流程中的审批步骤是什么"
  │
  ├─ 第三阶段：检索计划
  │    └─ 把意图转成具体参数（top_k、阈值、是否 rerank）
  │
  ├─ 第四阶段：FAQ 混合检索
  │    └─ 高置信直出 → 返回标准答案
  │    └─ 低置信 → 进入文档 RAG
  │
  ├─ 第五阶段：文档混合检索
  │    └─ Dense + Sparse Hybrid Search → Rerank → 上下文构建
  │
  ├─ 第六阶段：LLM 流式生成
  │    └─ Prompt Profile + 上下文 → ChatOpenAI stream
  │
  └─ 第七阶段：保存历史 + 写 Trace
```

### 3.3 事件协议

主流程通过 Generator 持续产出事件，前端按类型处理：

```python
# 事件类型
{"type": "start", "session_id": "..."}
{"type": "status", "message": "正在识别问题意图...", "session_id": "..."}
{"type": "token", "content": "入职", "session_id": "..."}
{"type": "token", "content": "流程", "session_id": "..."}
{"type": "end", "sources": [...], "intent": {...}, "retrieval": {...}}
{"type": "error", "message": "..."}
```

---

## 4. 意图识别：规则优先 + LLM 补充

### 4.1 设计思路

意图识别不是简单贴标签，它会直接影响后续所有决策：是否直接返回、是否改写、FAQ 和文档各召回多少、FAQ 直出阈值高低、是否推断业务分类。

**核心策略：规则优先 + LLM 补充**

- 高频、确定、低风险问题用规则（速度快、稳定）
- 模糊问题用 LangChain 结构化输出
- LLM 失败直接暴露依赖问题，不静默降级

### 4.2 意图类型

```python
Intent = Literal[
    "GREETING",        # 问候：你好、在吗
    "FOLLOW_UP",       # 追问：那审批呢
    "KNOWLEDGE_QUERY", # 知识咨询：入职流程有哪些步骤
    "FAQ_QUERY",       # 标准问答：API 限流怎么办
    "HUMAN_SERVICE",   # 人工客服：客服电话
    "OUT_OF_SCOPE",    # 越界：股票内幕
]
```

### 4.3 判断顺序（非常重要）

```python
def classify_intent(query, history, scenario):
    normalized = query.strip().lower()

    # 1. 问候和身份问题最先处理 → 避免无意义检索
    if re.match(r"^(你好|您好|hi|hello|...)", normalized):
        return IntentResult(intent="GREETING", direct_answer=answer, confidence=1.0)

    # 2. 明显越界问题 → 直接拒答
    if OFF_TOPIC_HINTS.search(normalized):
        return IntentResult(intent="OUT_OF_SCOPE", direct_answer=拒答, confidence=0.95)

    # 3. 人工客服短句 → 返回电话（长句可能是问流程，不截断）
    if HUMAN_SERVICE_HINTS.search(normalized) and len(normalized) <= 18:
        return IntentResult(intent="HUMAN_SERVICE", direct_answer=电话, confidence=0.9)

    # 4. 有历史且像追问 → 标记改写
    if history and (FOLLOW_UP_HINTS.search(query) or len(query) <= 8):
        return IntentResult(intent="FOLLOW_UP", requires_rewrite=True, confidence=0.8)

    # 5. 强规则覆盖高频业务问题（不调用 LLM）
    strong = _strong_rule_domain_intent(query, suggested_source)
    if strong:
        return strong

    # 6. 仍不确定 → 调用 LLM 结构化输出
    return _classify_with_llm(query, history, suggested_source, scenario)
```

### 4.4 强规则设计

```python
def _strong_rule_domain_intent(query, suggested_source):
    # FAQ 关键词 + 标准问法形体 → 直接归为 FAQ_QUERY
    if FAQ_HINTS.search(normalized):
        return IntentResult(intent="FAQ_QUERY", confidence=0.82)

    # source 已推断 + 短问题 + 标准问法 → FAQ 优先
    if suggested_source and len(normalized) <= 32 and FAQ_QUESTION_SHAPE_HINTS.search(normalized):
        return IntentResult(intent="FAQ_QUERY", confidence=0.83)

    # 更口语化的短问法（"资料呢""怎么排查"）→ 同样走 FAQ
    if suggested_source and len(normalized) <= 36 and DIRECT_FAQ_SHAPE_HINTS.search(normalized):
        return IntentResult(intent="FAQ_QUERY", confidence=0.84)

    # 知识库关键词 + source 已推断 → 知识咨询
    if KNOWLEDGE_HINTS.search(normalized) and (suggested_source or len(normalized) <= 24):
        return IntentResult(intent="KNOWLEDGE_QUERY", confidence=0.84)

    return None  # 规则无法确定，交给 LLM
```

**教学要点：**
- 规则的关键词列表来自对八个业务场景的长期观察，不是随意编写的。
- 强规则可以减少不必要的 LLM 调用。例如"制裁名单命中怎么办"可以通过 `cross_border_risk` 的 `source_patterns` 推断为 `sanction`，并因"怎么办"命中标准问答形态，直接走 FAQ 优先策略，省掉一次 LLM 意图分类。

### 4.5 LLM 结构化输出

```python
class IntentLLMDecision(BaseModel):
    intent: Intent
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    reason: str = Field(default="")
    requires_rewrite: bool = Field(default=False)
    suggested_source: str | None = Field(default=None)

def _classify_with_llm(query, history, suggested_source, scenario):
    model = get_chat_model(streaming=False).with_structured_output(IntentLLMDecision)
    decision = model.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"对话历史：\n{history_text}\n\n当前问题：{query}"),
    ])
    # 校验 LLM 返回的 source 是否在场景白名单中
    if source and source not in valid_sources:
        source = suggested_source
    return IntentResult(...)
```

**教学要点：**
- 使用 Pydantic 结构化输出限制 LLM 只返回枚举字段，避免模型输出解释性废话。
- LLM 返回的 `suggested_source` 必须经过 `valid_sources` 白名单校验，防止构造无效 Milvus filter。

### 4.6 Source 自动推断

```python
def infer_source(query, scenario):
    # source 必须来自当前场景 TOML，避免主链路代码里散落行业硬编码
    if scenario is None:
        return None
    best_source, _ = score_source_matches(query, scenario)
    return best_source
```

---

## 5. 检索计划：把意图变成参数

### 5.1 为什么需要检索计划

不同类型问题不能共用一套固定的 top_k 和阈值。例如：
- FAQ 问题适合先查标准问答，可以降低直出阈值
- 知识咨询需要更多文档上下文
- 追问信息不完整，应提高 FAQ 直出阈值防止误命中
- 费用/合规类问题宁可多召回资料也不能误答

### 5.2 RetrievalPlan 结构

```python
@dataclass(frozen=True)
class RetrievalPlan:
    run_faq: bool                 # 是否执行 FAQ 检索
    run_doc: bool                 # 是否执行文档检索
    faq_top_k: int                # FAQ 召回数量
    doc_top_k: int                # 文档召回数量
    rerank: bool                  # 是否重排
    faq_direct_threshold: float   # FAQ 直出分数阈值
    final_context_top_n: int      # 最终进入 prompt 的上下文条数
    min_context_score: float      # 最低上下文分数
    max_context_chars: int        # 上下文总字符数上限
    max_context_doc_chars: int    # 单条文档字符数上限
    use_query_variants: bool      # 是否启用查询变体
    question_category: str        # 问题类别（用于 Prompt 选择）
    prefer_table: bool            # 是否优先保留表格行
    faq_direct_exact_only: bool   # 是否只允许精确 FAQ 直出
    reason: str                   # 策略原因（可解释）
```

### 5.3 策略分支逻辑

```python
def build_retrieval_plan(query, intent):
    # 基础参数
    faq_top_k = settings.faq_top_k
    doc_top_k = settings.doc_top_k

    # 直接答案类 → 不检索
    if intent.intent in {"GREETING", "HUMAN_SERVICE", "OUT_OF_SCOPE"}:
        run_faq = False
        run_doc = False
        reason = "direct_answer_no_retrieval"

    # FAQ 查询类 → 降低直出阈值，减少文档候选
    elif intent.intent == "FAQ_QUERY":
        doc_top_k = max(8, settings.doc_top_k // 2)
        direct_threshold = max(0.62, settings.faq_direct_score_threshold - 0.08)
        reason = "faq_first"

    # 知识咨询类 → 增加文档上下文
    elif intent.intent == "KNOWLEDGE_QUERY":
        doc_top_k = max(settings.doc_top_k, settings.doc_complex_query_top_k)
        reason = "knowledge_doc_enriched"

    # 追问类 → 提高阈值，保留更多候选
    elif intent.intent == "FOLLOW_UP":
        faq_top_k = max(settings.faq_top_k, 24)
        doc_top_k = max(settings.doc_top_k, settings.doc_complex_query_top_k)
        direct_threshold = max(settings.faq_direct_score_threshold, 0.78)
        reason = "history_aware_follow_up"

    # 短问题保护（非追问）→ 收缩文档检索，提高 FAQ 阈值
    if is_short and intent.intent != "FOLLOW_UP":
        doc_top_k = min(doc_top_k, max(12, settings.final_context_top_n * 2))
        direct_threshold = max(0.78, direct_threshold)

    # 问题类别特殊保护
    if question_category == "pricing":
        direct_threshold = max(direct_threshold, 0.84)  # 费用类不能轻率直出
    elif question_category == "compliance":
        direct_threshold = max(direct_threshold, 0.86)  # 合规类最严格

    # 表格类问题 → 扩大文档候选，禁止相似 FAQ 直出
    if prefer_table:
        doc_top_k = max(doc_top_k, settings.doc_complex_query_top_k)
        faq_direct_exact_only = True  # 表格行问题泛化 FAQ 答不准

    return RetrievalPlan(...)
```

**教学要点：**
- 检索计划是可解释、可测试、可回归的。每次问答的 trace 都记录 `reason`，方便排查为什么某个问题走了某套参数。
- 费用和合规类问题的保护策略是工程安全性的体现：宁可多召回资料让模型基于上下文回答，也不能让 FAQ 凭相似分数直接承诺"退款金额"或"合规判断"。
- 表格类问题禁止相似 FAQ 直出：用户问"验收清单里测试报告是什么状态"，泛化 FAQ 可能只回答"验收需要哪些资料"，却答不到具体某一行。

---

## 6. 查询改写与变体生成

### 6.1 查询改写（仅 FOLLOW_UP 时执行）

```python
def rewrite_query_if_needed(query, history_messages, should_rewrite):
    if not should_rewrite or not history_messages:
        return query  # 不需要改写时直接返回原问题

    history_text = format_messages(history_messages[-8:])  # 只取最近 8 条
    llm = get_chat_model(streaming=False)
    response = llm.invoke([
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=f"对话历史：\n{history_text}\n\n当前问题：{query}"),
    ])
    return str(response.content).strip()
```

**改写示例：**
- 用户先问："入职流程有哪些步骤"
- 再问："那审批呢"
- 改写后："入职流程中的审批步骤是什么"

**教学要点：**
- 只有 `intent.requires_rewrite=True` 时才调用模型，避免所有请求都增加一次 LLM 调用。
- 只取最近 8 条历史是为了让改写聚焦当前追问主题，不被更早的话题干扰。

### 6.2 查询变体生成

当检索计划启用 `use_query_variants` 时，生成原问题的多个等价表达，提高召回覆盖率：

```python
# 规则命中时本地生成（不用 LLM）
variants = ["入职流程", "入职 SOP", "入职处理步骤"]

# 复杂问题时用 LLM 结构化生成
variants = llm.generate_variants(query)  # 返回多个等价检索入口
```

---

## 7. Milvus 混合检索系统

### 7.1 为什么选择 Milvus Hybrid Search

**Dense + Sparse 混合检索 = 语义理解 + 关键词匹配**

- **Dense 向量（BGE-M3 embedding）**：捕获语义相似度。用户问"怎么重置密码"，能召回"忘记密码的处理方法"，即使用词完全不同。
- **Sparse 向量（Milvus BM25BuiltInFunction）**：捕获关键词精确匹配。用户问"Webhook 回调失败"，能确保包含"Webhook"和"回调"的文档被优先召回。

为什么不用 RedisSearch 或手写 BM25？
- 当前部署使用 Milvus 2.6.x 的内置 BM25 函数，不需要维护第二套检索引擎。
- 入库一次，dense 和 sparse 向量同时写入，检索时在同一个表达式中完成过滤。

### 7.2 MilvusHybridStore 封装

```python
class MilvusHybridStore:
    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self._store: Milvus | None = None  # 懒加载

    @property
    def store(self):
        if self._store is None:
            self._store = Milvus(
                embedding_function=get_embeddings(),   # BGE-M3
                builtin_function=bm25_function(),       # Milvus BM25
                collection_name=self.collection_name,
                vector_field=["dense", "sparse"],       # 双向量字段
                text_field="text",
                primary_field="pk",
                auto_id=False,  # 使用入库时生成的稳定 chunk_id
            )
        return self._store

    def search_many(self, queries, plan, filter_expr):
        """对多个查询变体分别检索，合并结果后 rerank"""
        all_hits = []
        for q in queries:
            results = self.store.similarity_search_with_score(
                q, k=plan.top_k, expr=filter_expr
            )
            all_hits.extend(results)
        # 去重 → Rerank → 排序
        merged = merge_hits_by_document(all_hits)
        ranked = rerank_hits(merged, get_reranker())
        return sort_hits_by_score(ranked)
```

### 7.3 Milvus 过滤表达式

每条 FAQ 和文档 chunk 在入库时写入丰富的 metadata，在线检索时拼接成 Milvus 表达式：

```python
# 入库时写入的 metadata
{
    "scenario_id": "enterprise_knowledge",
    "tenant_id": "tenant_001",
    "dataset_id": "default",
    "visibility": "internal",
    "allowed_roles": ["employee", "hr_admin"],
    "kb_version": "kb_enterprise_knowledge_20260506_103000_9f2a1b3c",
    "source": "hr",
    "source_type": "faq",  # 或 "doc"
    "embedding_model_version": "bge-m3-local-v1",
    "chunk_schema_version": "parent_child_v1",
}

# 在线检索时的过滤表达式
expr = (
    f'scenario_id == "{scenario_id}"'
    f' && kb_version == "{active_version}"'
    f' && tenant_id == "{tenant_id}"'
    f' && visibility in ["internal", "public"]'
    f' && source in ["hr", "it"]'
)
```

### 7.4 FAQ 与文档分层检索

```python
# FAQ 优先
faq_result = faq_store.search_many(query_variants, plan, faq_expr)

# 判断是否高置信直出
if is_exact_match or faq_score >= plan.faq_direct_threshold:
    return faq_answer  # 直接返回标准答案

# FAQ 低置信 → 文档 RAG
doc_result = doc_store.search_many(query_variants, plan, doc_expr)

# 最终上下文可能包含 FAQ Top-2 + 文档 Top-N
```

**教学要点：**
- FAQ 和文档为什么要分集合存储？因为它们的用途不同：FAQ 适合精确回答，文档适合整合分析。混在一个集合里会同时损失准确率和可解释性。
- FAQ 直出不是简单的正则匹配，而是**基于向量检索 + 阈值判断**。用户问"入职当天要准备什么"和 FAQ 标准问题"新人入职需要准备哪些材料"语义相似但不完全相同，是否直出取决于相似分数是否超过动态阈值。
- 阈值是**动态的**，不是写死的。费用类问题阈值 0.84，合规类 0.86，追问类 0.78，短问题更高。这体现了"宁可多检索也不误答"的安全原则。

### 7.5 Rerank 重排

```python
def rerank_hits(hits, reranker):
    """用 CrossEncoder 对合并去重后的候选中做精排"""
    pairs = [[query, hit.document.page_content] for hit in hits]
    scores = reranker.compute_score(pairs)
    for hit, score in zip(hits, scores):
        hit.rerank_score = score
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
```

---

## 8. 上下文构建与 Prompt 工程

### 8.1 上下文构建

```python
def select_context_docs(faq_hits, doc_hits, plan):
    """筛选进入 Prompt 的文档片段"""
    # 1. FAQ 最多保留前 2 条（作为补充参考）
    # 2. 文档按 rerank_score 排序，过滤低于 min_context_score 的
    # 3. 去重：按 chunk/parent_id 去重
    # 4. 优先保留表格行（如果 prefer_table=True）
    # 5. 按 max_context_chars 截断总长度
    # 6. 单条文档按 max_context_doc_chars 截断
    ...

def build_context(docs, scenario):
    """构建最终上下文文本"""
    lines = []
    for i, doc in enumerate(docs):
        source_name = scenario.label_for_source(doc.metadata.get("source", ""))
        lines.append(f"[{i+1}] 来源：{source_name}\n{doc.page_content}")
    return "\n\n".join(lines)
```

### 8.2 Prompt Profile 选择

```python
def build_answer_prompt_profile(intent, scenario, query):
    """根据意图和问题类别选择最终回答模板"""
    question_category = infer_question_category(query)  # pricing/compliance/troubleshooting...

    # 优先级：问题类别 > 意图 > 默认
    profile = (
        CATEGORY_PROMPT_PROFILES.get(question_category)  # 高风险类别优先
        or PROMPT_PROFILES.get(intent)                    # 再按意图
        or DEFAULT_PROMPT_PROFILE                         # 兜底
    )

    # 注入场景上下文
    context = {
        "assistant_name": scenario.assistant_name,  # "企业知识助手"
        "business_domain": scenario.business_domain,  # "企业内部制度与流程"
        "industry": scenario.industry,                # "企业服务"
        "support_contact": scenario.support_contact,  # "400-xxx-xxxx"
    }

    return PromptProfile(
        name=profile.name,
        system_template=profile.system_template.format(**context),
        user_template=profile.user_template,
        reason=profile.reason,
    )
```

**Prompt Profile 示例（合规类）：**

```
System: 你是{assistant_name}，专门解答{business_domain}相关问题。
你的回答必须：
1. 严格基于提供的参考资料，不得超出资料范围
2. 涉及合规、金额、法律责任时，必须注明依据的具体条款
3. 如果资料不足以支持确定结论，必须明确说明信息不足
4. 不得自行做出合规判断或法律承诺
如有不确定的问题，引导用户联系{support_contact}。

User: 参考资料：
{context}

用户问题：{query}
```

**教学要点：**
- Prompt Profile 选择是确定性的，不走 LLM 二次判断。因为检索策略已经确定了问题类型，生成模板必须与检索策略一致。
- 费用类问题使用严格模板，要求模型只能基于可引用资料回答，不能自行承诺金额、退费条件等。
- 同一套模板通过场景配置注入不同的 `assistant_name`、`business_domain`、`industry`，实现多场景复用。

---

## 9. 知识库治理：多版本与数据隔离

### 9.1 知识库多版本管理

**核心问题：** 重新入库时如何避免覆盖旧数据？新切分策略或新 embedding 模型上线后效果变差，如何快速回滚？

**解决方案：** 每条 chunk 写入 `kb_version` 字段，在线检索只查 active 版本。版本切换只修改 JSON 清单，不改 Milvus 数据。

```python
# 版本号生成（包含时间戳 + 配置 hash）
# kb_enterprise_knowledge_20260506_103000_9f2a1b3c
def generate_kb_version(prefix, scenario_id):
    stamp = utc_file_stamp()
    config_hash = stable_hash(
        scenario_id, embedding_model_version,
        reranker_model_version, chunk_schema_version,
        doc_collection, faq_collection,
    )[:8]
    return f"{prefix}_{scenario_id}_{stamp}_{config_hash}"
```

**版本状态机：**

```
STAGED → ACTIVE → ARCHIVED
  ↑        │
  └────────┘ (回滚)
```

```python
# 激活版本（轻量操作，只改 JSON 文件）
def activate_version(self, kb_version):
    record = self.get(kb_version)
    # 旧 active → STAGED
    # 新版本 → ACTIVE
    self.data["previous_version"] = previous
    self.data["active_version"] = kb_version
    self.save()

# 在线检索解析版本
def resolve_active_version(requested=None):
    """优先级：请求显式传入 > 环境变量 > 版本清单"""
    if requested and self.exists(requested):
        return requested
    active = self.active_version_candidate()
    if not active or not self.exists(active):
        raise ValueError("没有 active 知识库版本")
    return active
```

**教学要点：**
- 版本切换不批量更新 Milvus chunk，只修改本地 JSON 中的 `active_version` 字段。这样版本切换是 O(1) 操作。
- 旧的 ACTIVE 版本不会删除，只是改为 STAGED/ARCHIVED，方便回滚和评测对比。
- 评测脚本可以显式传入历史版本号，对比新旧版本对同一批问题的召回效果。

### 9.2 数据隔离

```python
@dataclass(frozen=True)
class DataScope:
    tenant_id: str      # 租户隔离
    dataset_id: str     # 数据集隔离
    visibility: str     # 可见级别（public/internal/restricted）
    allowed_roles: list[str]  # 允许访问的角色
    user_role: str      # 当前用户角色
```

每条 FAQ 和 chunk 入库时写入隔离字段，在线检索时拼入 Milvus 表达式：

```python
# 入库时
metadata = {
    "tenant_id": "company_a",
    "dataset_id": "hr_docs_v2",
    "visibility": "internal",
    "allowed_roles": ["employee", "manager", "hr_admin"],
}

# 检索时
expr += f' && tenant_id == "{data_scope.tenant_id}"'
expr += f' && visibility in ["{data_scope.visibility}", "public"]'
```

---

## 10. 入库与索引链路

### 10.1 离线入库 vs 在线问答

**关键边界：在线问答不解析文件、不执行 OCR、不写入知识库。入库链路负责所有离线处理。**

### 10.2 文档入库流程

```python
def ingest_directory(directory_path, source, scenario_id, tenant_id, ...):
    """把目录中的业务文档增量写入 Milvus"""

    # 1. 创建/确认知识库版本
    version_store = get_kb_version_store(scenario_id)
    version = version_store.ensure_version(kb_version, create_new=create_new_version)

    # 2. 扫描目录，按后缀选择 Loader
    for file_path in directory.glob("*"):
        spec = get_document_loader_spec(file_path.suffix)  # .pdf→PyPDFLoader, .md→TextLoader
        if spec is None:
            continue  # 不支持的文件类型跳过

        # 3. 计算文件指纹 → 未变化则跳过（增量入库）
        fingerprint = file_fingerprint(file_path)
        if not force and manifest.is_unchanged(file_path, fingerprint):
            continue

        # 4. 加载文档 → 标准化 metadata → 切分
        docs = load_file(file_path, spec)
        docs = normalize_documents(docs, source, data_scope, kb_version, ...)
        chunks = split_documents(docs, CHINESE_SEPARATORS)
        # 切分策略：Markdown 标题增强 + 父子块，保留层级关系

        # 5. 删除旧 chunk → 写入新 chunk
        doc_store.delete(ids=manifest.get_old_chunk_ids(file_path))
        doc_store.store.add_documents(chunks)

        # 6. 更新 manifest
        manifest.mark_indexed(file_path, fingerprint, chunk_ids)

    # 7. 记录入库统计
    version_store.record_ingest_result(kb_version, content_type="doc", count=total_written)

    return total_written
```

### 10.3 Chunk 切分策略

```python
CHINESE_SEPARATORS = [
    "\n\n", "\n", "。", "！", "？", "；", "：",
    "，", "、", " ", ""
]

def split_documents(docs):
    """Markdown 标题增强 + 父子块策略"""
    # 1. 识别 Markdown 标题层级 → 作为 chunk 的层级 metadata
    # 2. 按中文分隔符递归切分 → 保证语义完整性
    # 3. 生成父子块关系 → parent_id, parent_content
    # 4. 每个 chunk 保留：chunk_id, parent_id, source, file, doc_id, page_index
```

### 10.4 FAQ 入库

```python
def faq_documents_from_csv(csv_path, scenario):
    """从 CSV 生成 LangChain Document 列表"""
    # CSV 格式：source, question, answer
    for row in csv_reader:
        yield Document(
            page_content=row["question"],  # 标准问题作为检索内容
            metadata={
                "answer": row["answer"],    # 标准答案直接存在 metadata 中
                "source": row["source"],
                "source_type": "faq",
                "kb_version": kb_version,
                "scenario_id": scenario.scenario_id,
                ...
            }
        )
```

### 10.5 增量入库与 Manifest

```python
class IndexManifest:
    """记录每个文件的入库状态，实现增量更新"""

    def is_unchanged(self, file_path, fingerprint):
        """对比文件指纹，判断是否需要重新入库"""
        ...

    def mark_indexed(self, file_path, fingerprint, chunk_ids):
        """记录文件已入库，保存 chunk ID 列表"""
        ...

    def get_old_chunk_ids(self, file_path):
        """获取文件对应的旧 chunk ID，用于删除后重写"""
        ...
```

### 10.6 全量重建与入库质量检查

```bash
# 完整的版本重建 + 入库质量检查 + 激活流程
python scripts/rebuild_kb_version.py \
    --scenario enterprise_knowledge \
    --new-version --force \
    --quality-gate --activate
```

执行顺序：

```
创建 STAGED 版本
  → FAQ 入库
  → 文档入库
  → 生成入库质量报告
  → 执行入库质量检查
  → 检查通过后才激活版本
```

---

## 11. RAG 回归验收与入库质量

### 11.1 RAG 回归与入库质量全景

项目的RAG 回归与入库质量不只是"跑几个测试"，而是一个完整的**可证明、可排查**体系：

```
入库质量 → LangSmith Evaluation → 领域指标回归验收 → Bad Case 沉淀
```

### 11.2 入库质量报告

```bash
python scripts/check_ingestion_quality_gate.py --scenario enterprise_knowledge
```

检查项：
- 文件解析失败（PDF 损坏、编码错误）
- unsupported 文件类型
- 空文件（没有任何有效文本）
- 低质量 chunk（字符数过少、噪声占比过高）
- FAQ 空值/重复（question 或 answer 为空，或完全相同的 FAQ 对）
- FAQ/正文冲突检测（FAQ 的 answer 与文档 chunk 内容矛盾）

```python
# FAQ/正文冲突检测使用 jieba 搜索分词
# 不是新增检索引擎，只是入库质量检查的轻量文本依据判断
words = jieba.cut_for_search(faq_question)
# 在文档 chunk 中搜索这些分词，判断是否有矛盾
```

### 11.3 评测指标体系

```python
# 主链路评测
evaluation_metrics = {
    "recall_at_k": 1.0,           # Top-K 召回率
    "mrr": 0.9000,                # 平均倒数排名
    "avg_keyword_coverage": 0.93, # 关键词覆盖率
    "hit_type_accuracy": 1.0,     # 命中类型准确率
    "source_inference_accuracy": 1.0,  # Source 推断准确率
    "prompt_profile_accuracy": 1.0,    # Prompt Profile 路由准确率
    "faq_direct_accuracy": 1.0,   # FAQ 直出准确率
    "scenario_isolation_accuracy": 1.0,  # 场景隔离准确率
    "errors": 0,                  # 错误数
}
```

### 11.4 RAG 回归验收（按场景/来源/命中路径分组）

```bash
python scripts/check_evaluation_gate.py --report reports/evaluation/multi_scenario_smoke_test.json
```

**分组验收是关键设计**：如果只看全局平均值，某个场景或 source 的退化可能被掩盖。分组验收确保每个 scenario、每个 source、每种 hit_type 都达到最低标准。

### 11.5 Bad Case 沉淀

```
LangSmith Trace → Annotation → Dataset → Evaluation
```

```bash
# 从 trace 中导出异常、低分、超时、信息不足样本
在 LangSmith 中筛选异常 Trace，使用 Annotation 标注原因，确认后加入 Dataset，并在 Evaluation 中作为回归样本。
```

### 11.6 回归验收体系一览

| 验收脚本 | 检查内容 |
|---------|---------|
| `check_project_guardrails.py` | 导入位置、旧链路、fallback 导入、依赖版本、场景包结构 |
| `check_ingestion_quality_gate.py` | 解析失败、空文件、低质量 chunk、FAQ 空值/重复/冲突 |
| `check_evaluation_gate.py` | Recall@K、MRR、关键词覆盖、hit_type、source 推断、场景隔离 |
| `check_followup_gate.py` | 追问召回、追问 source 准确率、Prompt Profile、场景隔离 |
| `check_performance_gate.py` | 错误率、首 token 耗时、总耗时、阶段耗时 |
| `api_e2e_smoke.py` / `acceptance_smoke.py` | 验证管理接口、页面和 WebSocket 流式链路 |

---

## 12. 多场景配置化设计

### 12.1 设计理念

**核心能力一套，业务配置可变。**

核心 RAG 能力（入库、检索、提示词、流式返回）完全复用，变化的只是 TOML 配置、FAQ CSV 和文档目录。

### 12.2 Scenario TOML 文件

```toml
# scenarios/enterprise_knowledge/scenario.toml

scenario_id = "enterprise_knowledge"
display_name = "企业内部知识助手"
industry = "企业服务"
assistant_name = "企业知识助手"
business_domain = "企业内部制度、流程与 IT 支持"
support_contact = "400-xxx-xxxx"

valid_sources = ["hr", "it", "admin", "finance", "legal"]

faq_collection = "enterprise_knowledge_faq"
doc_collection = "enterprise_knowledge_doc"

[source_labels]
hr = "人事制度"
it = "IT 支持"
admin = "行政管理"
finance = "财务制度"
legal = "法务合规"

[source_patterns]
hr = "入职|离职|转正|考勤|请假|加班|薪酬|绩效|试用期|劳动合同|社保"
it = "vpn|邮箱|打印机|wifi|密码|账号|权限|系统|软件|硬件|网络|安全"
admin = "会议室|办公用品|报销|差旅|接待|工位|门禁|停车"
finance = "预算|发票|付款|报销|采购|合同|结算|税率"
legal = "合同|隐私|数据保护|合规|知识产权|保密|竞业"
```

### 12.3 ScenarioDefinition 解析

```python
@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    display_name: str
    industry: str
    assistant_name: str
    business_domain: str
    support_contact: str
    valid_sources: list[str]          # 业务分类白名单
    faq_collection: str               # FAQ Milvus 集合名
    doc_collection: str               # 文档 Milvus 集合名
    data_root: str                    # 文档目录
    faq_csv_path: str                 # FAQ CSV 路径
    source_labels: dict[str, str]     # source 中文名
    source_patterns: dict[str, str]   # source 关键词推断规则
    sample_questions: list[str]       # 示例问题
    resume_project_name: str          # 简历用项目名
    resume_keywords: list[str]        # 简历用关键词

    def compiled_source_patterns(self) -> dict[str, re.Pattern]:
        """编译 source_patterns 中的正则，按 valid_sources 顺序匹配"""
        # 顺序很重要！例如保险理赔中"除外责任"同时包含"责任"，
        # 如果先匹配泛化的 liability 就会抢走更具体的 exclusion
```

### 12.4 如何维护既有场景配置

```bash
# 1. 选择一个已冻结场景做配置维护演示
cd scenarios/enterprise_knowledge

# 2. 调整已有 source 的中文标签、关键词规则或示例问题
vim scenario.toml

# 3. 补充已有 source 下的数据
#    - faq.csv（标准问答对）
#    - data/<source>_data/（业务文档）

# 4. 入库 + 入库质量检查 + 激活
python scripts/rebuild_kb_version.py \
    --scenario enterprise_knowledge \
    --new-version --force \
    --quality-gate --activate
```

**教学要点：**
- 场景包是项目的业务边界，当前已经冻结为 8 个，不再通过新增第 9 个场景来扩散业务外壳。
- 如果要讲扩展能力，应讲清楚“配置化可扩展”的机制，但课堂实操只维护既有场景的 source、FAQ 和资料。
- `valid_sources` 必须是显式的白名单，不允许为空或从全局配置推断。
- `source_patterns` 的顺序很重要，因为 `valid_sources` 同时承担匹配优先级。把更具体的 pattern 放在前面。
- `scenario_id` 会进入 Milvus 表达式，必须保证不为空且不与其他场景冲突。

---

## 13. 前后端交互与流式事件

### 13.1 分离前端的理由

前端 JS 和 CSS 从原来的一两个大文件拆分为多个小文件：

```
static/
├── index.html          # 问答页
├── admin.html          # 状态页
├── css/
│   ├── base.css        # 基础变量
│   ├── layout.css      # 布局
│   ├── sidebar.css     # 侧栏
│   ├── chat.css        # 聊天区
│   ├── input.css       # 输入区
│   └── responsive.css  # 响应式
└── js/
    ├── state.js         # 状态管理
    ├── api.js           # API 调用
    ├── renderer.js      # 渲染器
    ├── scenario.js      # 场景切换
    ├── session.js       # 会话管理
    └── chat.js          # 聊天交互
```

这样降低阅读成本，同时不引入 Vite/Webpack 等工程化依赖。

### 13.2 流式问答交互

```javascript
// 前端 WebSocket 流式问答
const ws = new WebSocket(`ws://127.0.0.1:8000/api/stream`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch (data.type) {
        case "start":
            // 显示加载状态
            showLoading();
            break;
        case "status":
            // 更新进度提示："正在识别问题意图..."
            updateStatus(data.message);
            break;
        case "token":
            // 逐字追加到答案区域
            appendToken(data.content);
            break;
        case "end":
            // 隐藏加载状态，显示来源引用和诊断信息
            hideLoading();
            showSources(data.sources);
            showRetrievalInfo(data.retrieval);
            break;
        case "error":
            // 显示错误，允许继续提问
            showError(data.message);
            break;
    }
};

// 发送问题
ws.send(JSON.stringify({
    query: "入职流程有哪些步骤？",
    session_id: currentSessionId,
    scenario_id: "enterprise_knowledge",
    source_filter: "hr",
    tenant_id: "default",
    dataset_id: "default",
}));
```

### 13.3 轻量状态页

`/admin` 页面只保留本地状态摘要，Trace、Dataset、Evaluation 和 Annotation 进入 LangSmith：

- **一期就绪总览**：项目守护、编译、单测、入库质量、领域指标回归验收、API 合同状态
- **LangSmith 链接**：跳转查看 Trace、Dataset、Evaluation 和 Annotation
- **全场景版本摘要**：新旧版本对同一批问题的召回差异
- **性能摘要**：首 token 耗时、总耗时、阶段耗时、慢点分布
- **入库质量报告**：每个场景的解析/切分/FAQ 质量

---

## 配套附录速查

18 讲配套了 9 个附录，深入讲解各个技术专题。备课时可根据学生水平选用：

| 附录 | 主题 | 一句话 | 关联讲次 |
|------|------|--------|---------|
| A | Pydantic 数据校验 | 类型注解如何自动校验配置/请求/LLM输出 | 3、6、10 |
| B | SHA256 稳定指纹 | 文件内容哈希 + 增量检测原理 | 15 |
| C | HNSW 图索引算法 | 分层可导航小世界图如何加速向量检索 | 2、9 |
| D | CrossEncoder 重排器 | Bi-Encoder vs CrossEncoder 架构对比 + 精排流程 | 2、7、9 |
| E | RecursiveCharacterTextSplitter | 递归降级切分算法的完整执行过程 | 5、15 |
| F | Milvus 索引机制与基本操作 | 五种索引图解 + pymilvus 实操代码 + LangChain 底层揭秘 | 2、9 |
| G | Embedding 模型深入 | Tokenization→Transformer→Pooling 四步详解 + 维度选择 + MTEB | 2 |
| H | 文档切分策略 | Parent-Child Chunking 设计原理 + chunk size/overlap 选择依据 | 5、15 |

> 附录文件位于 `docs/appendix/`，学生自学时可穿插在对应讲次前后阅读。

---

## 附录：关键文件索引

### 代码阅读路线

```
app.py                              # 第 1 步：看入口有多薄
  → qa_core/api/chat.py             # 第 2 步：看 HTTP/WebSocket 路由
  → qa_core/application/service.py  # 第 3 步：看 QAService 编排
  → qa_core/pipeline/rag.py         # 第 4 步：看 RAG 主流程事件
  → qa_core/pipeline/steps.py       # 第 5 步：看意图、改写、Prompt 准备
  → qa_core/pipeline/retrieval_steps.py  # 第 6 步：看 FAQ/doc 检索执行
  → qa_core/intent/classifier.py    # 第 7 步：看意图识别策略
  → qa_core/retrieval/strategy.py   # 第 8 步：看检索计划
  → qa_core/retrieval/store.py      # 第 9 步：看 Milvus 混合检索
  → qa_core/prompts/selector.py     # 第 10 步：看 Prompt 选择
  → qa_core/governance/kb_versions.py  # 第 11 步：看版本管理
  → qa_core/indexing/service.py     # 第 12 步：看入库链路
  → qa_core/scenarios/registry.py   # 第 13 步：看场景配置
```

### 关键配置项

```bash
# .env 必需配置
DASHSCOPE_API_KEY=sk-xxx              # LLM API Key（必需）
ADMIN_API_TOKEN=random-long-token      # 管理令牌（必需）
MILVUS_URI=http://127.0.0.1:19530     # Milvus 地址
MYSQL_HOST=127.0.0.1                  # MySQL 地址
MYSQL_PORT=3306
ACTIVE_KB_VERSION=                     # 可选，指定 active 版本
ACTIVE_SCENARIO_ID=enterprise_knowledge  # 默认场景
EMBEDDING_MODEL_PATH=models/bge-m3
RERANKER_MODEL_PATH=models/bge-reranker-large
```

---

## 小结：这个项目教会你什么

1. **企业级 RAG 不是"文档向量化 + 问答"那么简单**。它需要 FAQ/文档分层检索、混合召回、重排、意图识别、上下文预算、Prompt Profile、流式返回、知识库版本、数据隔离、RAG 回归验收等一系列工程能力的闭环。

2. **规则优先 + LLM 补充** 是工业界 RAG 的常见模式。高频确定场景用规则，快速稳定；模糊场景用 LLM，灵活扩展。

3. **Milvus 2.6 Hybrid Search** 把 dense 和 sparse 检索收敛到一个系统里，不需要维护两套索引和两套入库。

4. **知识库版本管理** 是 RAG 系统从 Demo 到生产的关键一步。没有版本管理，就无法安全回滚和灰度验证。

5. **入库质量检查 > 页面演示**。入库质量报告、LangSmith Evaluation、领域指标回归验收和 Bad Case 沉淀构成了可证明的工程可信度，而不是"看起来能跑"。

6. **多场景 != 多套代码**。通过配置化设计，同一套核心引擎可以包装成多种业务背景，这在简历和面试中非常有价值。
