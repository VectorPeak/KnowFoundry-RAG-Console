# KnowForge RAG Platform 当前架构详细流程图

本文档只描述优化后架构。边界是：`app.py + qa_core/api + qa_core` 是当前主链路；
旧版 `mysql_qa`、`rag_qa` 代码已经从工程中移除，只在文档中保留迁移前实现对照，
不进入在线请求主流程。

## 1. 总体架构

```mermaid
flowchart TD
    User["浏览器用户"] --> Page["static/index.html"]
    Page --> ScenariosApi["GET /api/scenarios"]
    Page --> CreateSession["POST /api/create_session"]
    Page --> Stream["WebSocket /api/stream"]
    Page --> HistoryApi["GET /api/history/{session_id}"]
    Page --> FeedbackApi["POST /api/feedback"]
    Page --> DebugApi["POST /api/retrieval/debug"]
    AdminPage["static/admin.html"] --> AdminApi["GET /api/admin/*"]

    Stream --> ApiChat["qa_core.api.chat<br/>问答 / 历史 / 反馈 / debug"]
    DebugApi --> ApiChat
    HistoryApi --> ApiChat
    FeedbackApi --> ApiChat
    ScenariosApi --> ApiChat
    CreateSession --> ApiPages["qa_core.api.pages<br/>页面 / 健康检查 / 会话"]
    AdminApi --> ApiAdmin["qa_core.api.admin<br/>trace / 报告"]
    KbApi["GET/POST /api/kb_versions*"] --> ApiVersions["qa_core.api.kb_versions<br/>版本查看 / 激活 / 归档"]

    App["app.py 应用入口<br/>CORS / 静态资源 / 路由注册"] --> ApiPages
    App --> ApiChat
    App --> ApiAdmin
    App --> ApiVersions
    ApiChat --> Factory["qa_core.application.factory.get_qa_service"]
    ApiChat --> ScenarioRegistry["qa_core.scenarios.registry.ScenarioRegistry"]
    ApiAdmin --> AdminReports["qa_core.observability / ingestion_quality"]
    Factory --> Service["qa_core.application.service.QAService"]

    Service --> Scenario["qa_core.scenarios.registry.resolve_scenario"]
    Service --> History["qa_core.memory.history.HistoryStore"]
    Service --> Intent["qa_core.intent.classifier.classify_intent"]
    Service --> Rewrite["qa_core.pipeline.rewrite.rewrite_query_if_needed"]
    Service --> Plan["qa_core.retrieval.strategy.build_retrieval_plan"]
    Service --> Variants["qa_core.pipeline.query_variants.generate_query_variants"]
    Service --> PromptProfile["qa_core.prompts.selector.build_answer_prompt_profile"]
    Service --> LLM["qa_core.llm.client.get_chat_model"]
    Service --> DataScope["qa_core.governance.data_scope.DataScope"]
    Service --> Retrieval["qa_core.retrieval.store.MilvusHybridStore"]
    Service --> Trace["qa_core.observability.langsmith_adapter.record_query_trace"]

    Retrieval --> MilvusFAQ["场景 FAQ Hybrid Collection"]
    Retrieval --> MilvusDoc["场景 Doc Hybrid Collection"]
    Retrieval --> Embedding["BGE-M3 Embedding"]
    Retrieval --> Reranker["CrossEncoder Reranker"]

    History --> MySQL["edu-mysql / MySQL<br/>chat_messages<br/>chat_session_summaries"]
    FeedbackApi --> Feedback["qa_core.memory.feedback.FeedbackStore"]
    Feedback --> MySQL
    Trace --> TraceLog["LangSmith Trace"]
    AdminReports --> TraceLog
    AdminReports --> Reports["reports/ingestion<br/>reports/evaluation"]

    LLM --> DashScope["DashScope OpenAI-compatible API"]
    MilvusFAQ --> Milvus["edu-milvus 2.6.4"]
    MilvusDoc --> Milvus
```

## 2. 在线问答主流程

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户页面
    participant WS as qa_core/api/chat.py /api/stream
    participant S as QAService.stream_query
    participant H as ChatHistoryStore
    participant I as Intent
    participant R as Retrieval
    participant M as Milvus
    participant L as ChatOpenAI
    participant DB as MySQL
    participant T as TraceLog

    U->>WS: WebSocket 发送 query/session_id/scenario_id/source_filter/tenant_id/dataset_id
    WS->>S: 创建同步事件生成器
    S->>S: resolve_scenario(scenario_id)
    S->>S: resolve_active_kb_version(scenario_id)
    S->>S: resolve_data_scope(tenant_id, dataset_id, visibility, role)
        S-->>U: start
    S-->>U: status 正在识别问题意图
    S->>H: 读取历史摘要 + 最近消息
    H->>DB: SQLChatMessageHistory 查询
    S->>I: classify_intent(query, history, scenario)

    alt 问候 / 越界 / 人工客服
        S-->>U: token 直接答案
        S->>H: add_turn(question, answer)
        H->>DB: 写入 HumanMessage + AIMessage
        S->>T: record_qa_trace
        S-->>U: end
    else 需要 RAG
        S->>S: rewrite_query_if_needed
        S->>S: build_retrieval_plan
        S->>S: generate_query_variants
        S->>S: build_answer_prompt_profile

    S-->>U: status 正在检索业务知识库
        S->>R: FAQ search_many(query_variants, plan)
        R->>M: FAQ dense + sparse hybrid search<br/>expr 包含 source/kb_version/data_scope
        R->>R: 去重 + rerank

        alt FAQ 高置信直出
            S-->>U: token FAQ 标准答案
            S->>H: add_turn(question, answer)
            H->>DB: 写入历史
            S->>T: record_qa_trace
            S-->>U: end sources/retrieval/prompt_profile
        else 文档 RAG
            S-->>U: status 正在匹配相关业务资料
            S->>R: Doc search_many(query_variants, plan)
            R->>M: Doc dense + sparse hybrid search<br/>expr 包含 source/kb_version/data_scope
            R->>R: 去重 + rerank
            S->>S: _select_context_docs + _build_context
            S-->>U: status 正在生成回答
            S->>L: llm.stream(SystemMessage, HumanMessage)
            loop 每个非空 token
                L-->>S: token chunk
                S-->>U: token
            end
            S->>H: add_turn(question, full_answer)
            H->>DB: 写入历史
            S->>T: record_qa_trace
            S-->>U: end sources/intent/retrieval/prompt_profile
        end
    end

    WS->>H: 后台 refresh_summary_if_needed
    H->>DB: 写入或更新会话摘要
```

## 3. 检索与生成细节

```mermaid
flowchart TD
    Q["用户问题"] --> Intent["意图识别<br/>GREETING / FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP"]
    Intent --> Direct{"是否直接回答"}
    Direct -- 是 --> DirectAnswer["规则直答<br/>问候 / 越界 / 人工客服"]
    Direct -- 否 --> Rewrite["追问改写<br/>只在 FOLLOW_UP 或 requires_rewrite 时执行"]

    Rewrite --> Plan["动态检索计划 RetrievalPlan"]
    Plan --> QueryVariants["原问题 + 等价查询变体<br/>规则命中则本地生成，否则用 LLM 结构化生成"]

    QueryVariants --> FAQSearch["FAQ Hybrid Search"]
    FAQSearch --> FAQRank["FAQ 去重 + rerank"]
    FAQRank --> FAQDirect{"FAQ 是否高置信直出"}
    FAQDirect -- 是 --> FAQAnswer["返回 metadata.answer"]

    FAQDirect -- 否 --> DocSearch["Doc Hybrid Search"]
    DocSearch --> DocRank["Doc 去重 + rerank"]
    DocRank --> ContextSelect["上下文筛选<br/>FAQ 前 2 条 + 文档 min_context_score"]
    ContextSelect --> ContextBuild["_build_context<br/>[1] 来源 + 内容"]

    Intent --> Prompt["Prompt Profile<br/>faq_answer / knowledge_answer / follow_up / default"]
    ContextBuild --> Generate["LLM 流式生成"]
    Prompt --> Generate
    Generate --> Save["写入 SQLChatMessageHistory"]
    Save --> End["end 事件<br/>sources / intent / retrieval / prompt_profile"]
```

## 4. 检索诊断流程

```mermaid
flowchart TD
    DebugReq["POST /api/retrieval/debug"] --> Thread["asyncio.to_thread<br/>避免阻塞事件循环"]
    Thread --> Debug["QAService.debug_retrieval"]
    Debug --> History["读取历史上下文"]
    Debug --> Intent["意图识别"]
    Intent --> Rewrite["追问改写"]
    Rewrite --> Plan["检索计划"]
    Plan --> Variants["查询变体"]
    Variants --> FAQ["FAQ 检索"]
    Variants --> Doc["Doc 检索"]
    Intent --> Profile["Prompt Profile 选择"]
    FAQ --> Response["返回诊断 JSON"]
    Doc --> Response
    Profile --> Response

    Response --> Fields["query / rewritten_query / source_filter<br/>intent / retrieval_plan<br/>prompt_profile / faq_sources / doc_sources"]
```

## 5. 文档与 FAQ 入库流程

```mermaid
flowchart TD
    Docs["场景文档目录<br/>scenarios/*/data"] --> Rebuild["scripts/rebuild_kb_version.py"]
    FAQCSV["场景 FAQ CSV<br/>scenarios/*/faq.csv"] --> Rebuild

    Rebuild --> Indexing["qa_core.indexing.service.ingest_directory"]
    Indexing --> Source["source 推断或显式传入<br/>当前场景 valid_sources"]
    Indexing --> Scope["DataScope<br/>tenant_id / dataset_id / visibility / allowed_roles"]
    Source --> Registry["DOCUMENT_LOADER_REGISTRY<br/>按后缀选择 loader"]
    Registry --> Load["load_file"]
    Load --> Normalize["normalize_documents<br/>补 source/file/doc_id/page_index/data_scope"]
    Normalize --> Split["split_documents<br/>Markdown 标题增强 + 父子块"]
    Split --> ChunkMeta["chunk metadata<br/>parent_id / parent_content / chunk_id"]
    ChunkMeta --> Manifest{"文件 fingerprint 是否变化"}
    Manifest -- 未变化 --> Skip["跳过，不重复入库"]
    Manifest -- 已变化或 force --> DeleteOld["删除旧 chunk_ids"]
    DeleteOld --> DocStore["Milvus Doc add_documents"]
    DocStore --> ManifestSave["更新 .index_manifest/documents.json"]
    Cleanup["scripts/cleanup_missing_docs.py"] --> ManifestScan["扫描 manifest 中本地路径不存在的记录"]
    ManifestScan --> CleanupDryRun{"是否传入 --apply"}
    CleanupDryRun -- 否 --> CleanupPreview["只输出将清理的文件和 chunk 数"]
    CleanupDryRun -- 是 --> CleanupDelete["按 chunk_ids 删除 Milvus 旧 chunk<br/>并移除 manifest 记录"]

    IngestFAQ --> FAQDocs["faq_documents_from_csv"]
    FAQDocs --> FAQMeta["page_content=标准问题<br/>metadata.answer=标准答案<br/>metadata.data_scope=检索隔离字段"]
    FAQMeta --> FAQStore["Milvus FAQ delete_ids + add_documents"]
```

## 6. 质量治理与管理流程

```mermaid
flowchart TD
    Rebuild["scripts/rebuild_kb_version.py"] --> Ingest["FAQ + 文档入库"]
    Rebuild --> Quality["qa_core.quality.ingestion.build_ingestion_quality_report"]
    Quality --> IngestionReport["reports/ingestion/<scenario>/*.json"]

    Eval["scripts/evaluate_core_chain.py"] --> Debug["QAService.debug_retrieval"]
    Eval --> StreamEval["QAService.stream_query"]
    Debug --> EvalReport["Recall@K / MRR / keyword coverage"]
    StreamEval --> EvalReport
    EvalReport --> EvaluationFiles["reports/evaluation/*.json"]

    Online["QAService.preview_query / stream_query"] --> Trace["LangSmith Trace"]
    Trace --> LangSmith["LangSmith Dataset / Evaluation / Annotation"]

    Admin["/admin 状态页"] --> LangSmithApi["/api/admin/langsmith"]
    Admin --> IngestionApi["/api/admin/ingestion_reports"]
    Admin --> GateApi["/api/admin/gate_reports<br/>/api/admin/performance_reports"]
    LangSmithApi --> LangSmith
    IngestionApi --> IngestionReport
    GateApi --> EvaluationFiles
```

这部分能力补齐的是“可证明”和“可排查”：

- 入库质量报告证明知识库资料是否解析成功、切分是否健康；
- 评测报告证明检索策略和回答效果是否退化；
- LangSmith Trace 证明一次线上回答的意图、检索计划、来源和耗时；
- 状态页只展示版本、入库、回归报告和 LangSmith 状态，便于面试演示和本地排查。

## 7. 运行依赖关系

```mermaid
flowchart LR
    API["edu-rag-api<br/>FastAPI + qa_core"] --> MySQL["edu-rag-mysql<br/>history / summary / feedback"]
    API --> Milvus["edu-rag-milvus<br/>FAQ / Doc hybrid search"]
    Milvus --> Etcd["edu-rag-etcd"]
    Milvus --> Minio["edu-rag-minio"]
    API --> LocalModels["本地模型<br/>bge-m3<br/>bge-reranker-large"]
    API --> LLMProvider["DashScope<br/>OpenAI-compatible LLM"]

    Host["宿主机"] --> API8000["127.0.0.1:8000"]
    Host --> Milvus19530["127.0.0.1:19530"]
    Host --> MySQL3306["127.0.0.1:3306（标准 docker-compose.yml）"]
    Host --> MySQL3307["127.0.0.1:3307（仅 docker-compose.milvus.yml）"]
```

## 8. 关键边界

- 在线问答不解析文件、不执行 OCR、不写入知识库。
- 入库链路负责 loader、切分、metadata、manifest 和 Milvus 写入。
- 本地文件删除后，通过 `scripts/cleanup_missing_docs.py` 离线清理旧 chunk；默认 dry-run，不在在线请求中触发。
- MySQL 只负责聊天历史、摘要、反馈，不承担 FAQ/文档检索。
- RedisSearch 不进入当前核心链路。
- 旧 `mysql_qa` / `rag_qa` 代码已移除，不进入 `app.py -> qa_core/api -> qa_core` 主请求链路。
- 配置主链路统一读取 `.env + scenario.toml`，不再引入额外配置入口。
- `ADMIN_API_TOKEN` 是必需配置；问答/调试/反馈接口有轻量进程内限流。
- Milvus、MySQL、本地模型、LLM Key、场景配置和 active 知识库版本都是启动前置条件。
- 业务场景已经冻结为 8 个，`scripts/check_project_guardrails.py` 会阻止继续新增未评审场景包。
- 表格行解析和表格类检索已经进入一期主入库与检索闭环；复杂 OCR 仍是离线治理能力，必须人工复核后再通过资料提升、版本重建和入库质量检查进入 active 知识库。
- 入库质量报告、召回评测和LangSmith 观测已经进入当前工程闭环，但都不在用户在线提问时触发高成本入库或评测。
