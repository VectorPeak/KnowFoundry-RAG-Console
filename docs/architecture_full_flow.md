# QA Core 全链路架构

---

## 一、在线问答主流程

```mermaid
flowchart TD
    Q["用户问题<br/>'入职流程有哪些步骤'"] --> API["/api/stream WebSocket"]

    API --> Ctx["create_query_context()<br/>解析场景 + 数据域 + kb_version"]

    Ctx --> FastPath{"try_fast_faq_direct_answer()<br/>短问题 + FAQ 精确匹配?"}
    FastPath -->|"命中"| Direct1["返回标准答案<br/>跳过全部 RAG"]

    FastPath -->|"未命中"| Prepare["prepare_retrieval()"]
    Prepare --> Intent["classify_intent()<br/>规则优先: GREETING→越界→客服→追问→LLM"]
    Prepare --> Rewrite["rewrite_query_if_needed()<br/>追问→补全为独立检索问题"]
    Prepare --> Plan["build_retrieval_plan()<br/>4层决策 → RetrievalPlan"]
    Prepare --> Variants["generate_query_variants()<br/>同义表达: 'Webhook'→'回调'"]

    Intent -->|"问候/越界/客服"| Direct2["直接答案<br/>不查 Milvus"]

    Plan --> FAQ["search_faq()"]
    FAQ --> FAQCheck{"get_faq_direct_answer()<br/>精确匹配 或 分数≥阈值?"}
    FAQCheck -->|"直出"| Direct3["FAQ 标准答案<br/>跳过文档检索+LLM"]

    FAQCheck -->|"不直出"| Doc["search_doc()"]
    Doc --> Select["select_context_docs()<br/>分数过滤 → 去重 → prefer_table 排序<br/>→ 三重截断(top_n/chars/doc_chars)"]

    Select -->|"上下文为空"| Insufficient["build_insufficient_context_answer()<br/>'信息不足，请联系人工支持'"]

    Select -->|"有上下文"| LLM["stream_llm_answer()<br/>ChatOpenAI 流式生成"]
    LLM --> Cite["enforce_answer_citations()<br/>补充来源编号"]
    Cite --> Save["history.add_turn() + record_trace()"]
    Save --> End["yield end 事件 → 浏览器"]

    style FastPath fill:#ECFDF5,stroke:#059669
    style FAQCheck fill:#ECFDF5,stroke:#059669
    style Intent fill:#FEF2F2,stroke:#DC2626
    style Plan fill:#FEF3C7,stroke:#F59E0B
    style Select fill:#E0F2FE,stroke:#0284C7
```

---

## 二、意图识别：6 级规则优先级

```mermaid
flowchart TD
    Input["query + history"] --> R1{"规则1: GREETING?<br/>你好/在吗/hi"}
    R1 -->|"命中"| G["Intent=GREETING<br/>direct_answer=问候语"]
    R1 -->|"否"| R2{"规则2: OUT_OF_SCOPE?<br/>彩票/赌博/违法"}
    R2 -->|"命中"| O["Intent=OUT_OF_SCOPE<br/>direct_answer=拒答"]
    R2 -->|"否"| R3{"规则3: HUMAN_SERVICE?<br/>人工客服/电话 + 短句"}
    R3 -->|"命中"| H["Intent=HUMAN_SERVICE<br/>direct_answer=客服电话"]
    R3 -->|"否"| R4{"规则4: FOLLOW_UP?<br/>代词开头/短句 + 有历史"}
    R4 -->|"命中"| F["Intent=FOLLOW_UP<br/>requires_rewrite=True"]
    R4 -->|"否"| R5{"规则5: 强规则 FAQ/KNOWLEDGE?<br/>关键词 + source推断 + 问法匹配"}
    R5 -->|"命中"| S["Intent=FAQ_QUERY 或 KNOWLEDGE_QUERY<br/>confidence ≥ 0.82"]
    R5 -->|"否"| LLM["_classify_with_llm()<br/>Pydantic 结构化输出<br/>IntentLLMDecision"]
    LLM --> Result["Intent + confidence +<br/>requires_rewrite + suggested_source"]

    style R1 fill:#FEF2F2,stroke:#DC2626
    style R2 fill:#FEF2F2,stroke:#DC2626
    style R3 fill:#FEF2F2,stroke:#DC2626
    style R4 fill:#FEF2F2,stroke:#DC2626
    style R5 fill:#FEF3C7,stroke:#F59E0B
    style LLM fill:#E0F2FE,stroke:#0284C7
```

---

## 三、检索计划：4 层决策

```mermaid
flowchart TD
    Start["query + IntentResult + Settings"] --> L1

    L1["第1层: intent 分岔"] -->|"直接答案"| L1A["run_faq=false<br/>run_doc=false"]
    L1 -->|"FAQ_QUERY"| L1B["threshold-0.08<br/>doc_top_k 减半"]
    L1 -->|"KNOWLEDGE_QUERY"| L1C["doc_top_k=complex<br/>final_context_top_n≥5"]
    L1 -->|"FOLLOW_UP"| L1D["faq_top_k≥24<br/>threshold≥0.78"]

    L1B --> L2
    L1C --> L2
    L1D --> L2

    L2{"第2层: is_short<br/>且非 FOLLOW_UP?"} -->|"是"| L2A["doc 收缩<br/>threshold≥0.78"]
    L2 -->|"否"| L3

    L2A --> L3

    L3{"第3层: question_category"} -->|"pricing"| L3A["threshold≥0.84<br/>context≥6"]
    L3 -->|"compliance"| L3B["threshold≥0.86<br/>context≥6"]
    L3 -->|"troubleshooting"| L3C["doc 扩大"]
    L3 -->|"summary"| L3D["doc 扩大"]
    L3 -->|"default"| L4

    L3A --> L4
    L3B --> L4
    L3C --> L4
    L3D --> L4

    L4{"第4层: prefer_table?"} -->|"是"| L4A["doc 扩大<br/>faq_direct_exact_only=true<br/>表格行优先排序"]
    L4 -->|"否"| Output

    L4A --> Output["RetrievalPlan (frozen)<br/>run_faq/doc, top_k, threshold,<br/>context 参数, reason"]

    style L1 fill:#FEF2F2,stroke:#DC2626
    style L2 fill:#FFFBEB,stroke:#D97706
    style L3 fill:#FEF3C7,stroke:#F59E0B
    style L4 fill:#E0F2FE,stroke:#0284C7
    style Output fill:#ECFDF5,stroke:#059669
```

---

## 四、FAQ 三条路径对比

```mermaid
flowchart TD
    Q["用户问题"] --> FP{"should_try_faq_fast_path()<br/>短 + 像标准问答?"}

    FP -->|"是 (路径A)"| Fast["FAQ 快速路径<br/>意图识别之前"]
    Fast --> FastSearch["get_faq_store().search_many()<br/>rerank=False"]
    FastSearch --> FastMatch{"_exact_faq_answer()<br/>threshold=∞"}
    FastMatch -->|"精确匹配"| FastHit["直出标准答案<br/>hit_type=faq_direct<br/>首 token < 500ms"]
    FastMatch -->|"未命中"| Continue["继续主流程"]

    FP -->|"否 (路径B/C)"| Normal["正常流程"]
    Normal --> Plan["build_retrieval_plan()<br/>→ RetrievalPlan"]
    Plan --> FAQSearch["search_faq()<br/>rerank=True"]
    FAQSearch --> FAQCheck{"get_faq_direct_answer()"}
    FAQCheck -->|"精确匹配 或 分数≥threshold<br/>(路径B)"| FAQHit["直出标准答案<br/>hit_type=faq_direct<br/>首 token < 1s"]
    FAQCheck -->|"不满足 (路径C)"| DocRAG["继续 search_doc()<br/>→ 完整 RAG"]

    style Fast fill:#ECFDF5,stroke:#059669
    style FastHit fill:#059669,stroke:#059669,color:#fff
    style FAQHit fill:#059669,stroke:#059669,color:#fff
    style DocRAG fill:#E0F2FE,stroke:#0284C7
```

---

## 五、Milvus 检索执行

```mermaid
flowchart TD
    Queries["query_variants<br/>['入职流程步骤','新人入职流程','入职SOP']"] --> Norm["normalize_queries()<br/>清洗去重"]

    Norm --> V1["变体1 检索"]
    Norm --> V2["变体2 检索"]
    Norm --> V3["变体3 检索"]

    V1 --> S1["Dense(BGE-M3) + Sparse(BM25)<br/>weighted fusion 0.55:0.45"]
    V2 --> S2["Dense(BGE-M3) + Sparse(BM25)<br/>weighted fusion 0.55:0.45"]
    V3 --> S3["Dense(BGE-M3) + Sparse(BM25)<br/>weighted fusion 0.55:0.45"]

    S1 --> Filter1["build_source_expr()<br/>source='hr' AND kb_version='...'<br/>AND tenant_id='default'"]
    S2 --> Filter2["build_source_expr()<br/>source='hr' AND kb_version='...'<br/>AND tenant_id='default'"]
    S3 --> Filter3["build_source_expr()<br/>source='hr' AND kb_version='...'<br/>AND tenant_id='default'"]

    Filter1 --> Merge["merge_hits_by_document()<br/>按 chunk_id/faq_id 去重<br/>保留最高分"]
    Filter2 --> Merge
    Filter3 --> Merge

    Merge --> Sort["sort_hits_by_score()<br/>分数降序"]
    Sort --> Rerank["rerank_hits()<br/>CrossEncoder 精排<br/>top_n 截断"]
    Rerank --> Result["RetrievalResult<br/>hits + top_score + elapsed_ms"]

    style Merge fill:#FFFBEB,stroke:#D97706
    style Rerank fill:#ECFDF5,stroke:#059669
```

---

## 六、上下文构建：select_context_docs() 过滤链

```mermaid
flowchart TD
    FAQ_In["FAQ hits"] --> F1{"score ≥<br/>min_context_score?"}
    F1 -->|"是"| F2["取前 2 条"]
    F2 --> F3["转为 '常见问题 + 标准答案' 格式"]
    F3 --> Append

    Doc_In["Doc hits"] --> D1{"score ≥<br/>min_context_score?"}
    D1 -->|"是"| D2{"prefer_table?"}
    D2 -->|"是"| D3["表格行排前面<br/>is_table_document=0<br/>普通正文=1"]
    D2 -->|"否"| D4["保持 rerank 顺序"]
    D3 --> D5["优先用 parent_content"]
    D4 --> D5
    D5 --> Append

    Append["append_doc() 逐条追加"] --> C1{"约束1:<br/>final_context_top_n?"}
    C1 -->|"未达上限"| C2{"约束2:<br/>单条 > max_context_doc_chars?"}
    C2 -->|"超过"| C2A["截断 + 标记 truncated"]
    C2 -->|"未超"| C3{"约束3:<br/>总量 > max_context_chars?"}
    C2A --> C3
    C3 -->|"超过"| Stop["停止追加"]
    C3 -->|"未超"| Add["加入 selected"]
    Add --> C1

    C1 -->|"达上限"| Output["返回 selected<br/>空列表 = insufficient_context"]

    style C1 fill:#FEF2F2,stroke:#DC2626
    style C2 fill:#FFFBEB,stroke:#D97706
    style C3 fill:#FFFBEB,stroke:#D97706
    style Output fill:#ECFDF5,stroke:#059669
```

---

## 七、数据隔离模型

```mermaid
flowchart TD
    subgraph Request["API 请求参数"]
        T["tenant_id='acme'"]
        D["dataset_id='products'"]
        V["visibility='internal'"]
        R["user_roles=['employee','manager']"]
    end

    Request --> Resolve["resolve_data_scope()<br/>→ DataScope.from_request()"]

    Resolve --> DS["DataScope<br/>tenant_id='acme'<br/>dataset_id='products'<br/>visibility='internal'<br/>user_roles=['employee','manager']"]

    DS --> Expr["expr_clauses()"]

    Expr --> C1["tenant_id == 'acme'"]
    Expr --> C2["dataset_id == 'products'"]
    Expr --> C3["visibility == 'public' OR visibility == 'internal'"]
    Expr --> C4["array_contains(allowed_roles,'employee')<br/>OR array_contains(allowed_roles,'manager')"]

    C1 --> Final["Milvus expr:<br/>tenant_id=='acme' AND dataset_id=='products'<br/>AND (visibility=='public' OR visibility=='internal')<br/>AND (array_contains(...) OR array_contains(...))"]
    C2 --> Final
    C3 --> Final
    C4 --> Final

    style DS fill:#F3E8FF,stroke:#9333EA
    style Final fill:#E0F2FE,stroke:#0284C7
```

---

## 八、知识库版本生命周期

```mermaid
flowchart TD
    Create["ensure_version()<br/>create_new=True"] --> Staged["状态: STAGED<br/>version = kb_hr_20250101_120000_a1b2c3d4"]

    Staged --> Ingest["入库写入<br/>ingest_directory() / faq_documents_from_csv()"]
    Ingest --> Report["生成质量报告<br/>build_ingestion_quality_report()"]

    Report --> Gate{"质量门控检查"}
    Gate -->|"通过"| Activate["activate_version()"]
    Gate -->|"不通过"| Fix["修复问题 → 重新入库"]

    Activate --> Active["状态: ACTIVE<br/>在线检索按此版本过滤<br/>kb_version == '...'"]
    Fix --> Ingest

    Active --> NewVer["新版本创建<br/>旧 ACTIVE → STAGED"]
    NewVer --> Staged

    Active --> Archive["archive_version()"]
    Archive --> Archived["状态: ARCHIVED<br/>保留 Milvus 数据<br/>不再用于在线检索"]

    style Staged fill:#FFFBEB,stroke:#D97706
    style Active fill:#ECFDF5,stroke:#059669
    style Archived fill:#F1F5F9,stroke:#64748B
```

---

## 九、入库流程

```mermaid
flowchart TD
    Start["ingest_directory('data/hr_data/')"] --> Res["resolve_scenario()<br/>→ ScenarioDefinition"]
    Res --> Scope["resolve_data_scope()<br/>→ DataScope"]
    Scope --> Ver["get_kb_version_store()<br/>.ensure_version()"]

    Ver --> Walk["遍历目录文件"]

    Walk --> PerFile["_ingest_single_file()"]

    PerFile --> FP{"file_fingerprint()<br/>vs manifest 记录"}
    FP -->|"未变更"| Skip["跳过"]
    FP -->|"已变更/新文件"| Del["delete_ids(old_chunks)"]
    Del --> Load["load_file(path)<br/>→ list[Document]"]
    Load --> Norm["normalize_documents()<br/>写 metadata(tenant,dataset,visibility,kb_version)"]
    Norm --> Split["split_documents()<br/>父子块策略"]
    Split --> Write["add_documents(chunks, ids)<br/>→ Milvus Dense+Sparse"]
    Write --> ManUp["manifest.update()<br/>记录 fingerprint + chunk_ids"]

    Skip --> Next
    ManUp --> Next{"还有文件?"}
    Next -->|"是"| PerFile
    Next -->|"否"| Save["manifest.save()<br/>version_store.record_ingest_result()"]

    Save --> Act{"activate=True?"}
    Act -->|"是"| ActVer["version_store.activate_version()<br/>立即切换在线检索版本"]
    Act -->|"否"| Done["完成<br/>版本保持 STAGED"]

    style Write fill:#ECFDF5,stroke:#059669
    style ActVer fill:#059669,stroke:#059669,color:#fff
```

---

## 十、场景边界检测

```mermaid
flowchart TD
    Q["用户在企业知识助手场景问:<br/>'安全技术交底只有口头说明可以吗'"] --> Current["score_source_matches()<br/>对当前场景 enterprise_knowledge 评分"]

    Current -->|"score < 8"| Cross["遍历所有其他场景<br/>score_source_matches()"]

    Cross --> Other["工程资料场景 engineering_project_qa<br/>命中 safety source_pattern<br/>score ≥ 12"]

    Other --> Decision["detect_scenario_boundary()<br/>返回 crossed=True<br/>matched_scenario='工程资料场景'<br/>matched_source='安全资料'"]

    Decision --> Hint["返回引导提示:<br/>'当前场景是 企业知识助手，<br/>这个问题更像 工程资料 中的 安全资料 分类。<br/>请切换到对应场景后再查询。'"]

    Current -->|"score ≥ 8"| Pass["当前场景已匹配<br/>crossed=False<br/>正常检索"]

    style Decision fill:#FEF2F2,stroke:#DC2626
    style Hint fill:#FEF2F2,stroke:#DC2626
```

---

## 十一、历史压缩策略

```mermaid
flowchart LR
    subgraph Store["会话历史管理"]
        R1["第 1-14 轮<br/>全部保留"]
        R2["第 15+ 轮<br/>refresh_summary_if_needed()<br/>LLM 生成增量摘要<br/>save_summary() → MySQL"]
    end

    subgraph Context["每次请求发给 LLM 的上下文"]
        S["SystemMessage<br/>历史摘要: ..."]
        M["最近 8 条完整消息"]
        Q2["当前问题"]
    end

    R1 --> M
    R2 --> S
    R2 --> M

    S --> Final["[System Prompt]<br/>↓<br/>[摘要 200-1200 字符]<br/>↓<br/>[最近 8 条原文]<br/>↓<br/>[当前问题]"]

    M --> Final
    Q2 --> Final

    style Store fill:#EFF6FF,stroke:#3B82F6
    style Context fill:#ECFDF5,stroke:#059669
```

---

## 模块文件索引

| 层级 | 文件 | 核心职责 |
|------|------|---------|
| 入口 | `api/chat.py` | WebSocket + HTTP 路由，事件转发 |
| 应用 | `application/service.py` | QAService 编排门面 |
| 编排 | `pipeline/rag.py` | stream_query 7 阶段主流程 |
| 步骤 | `pipeline/steps.py` | prepare_retrieval / prepare_answer / stream_llm_answer |
| 检索步骤 | `pipeline/retrieval_steps.py` | search_faq / get_faq_direct_answer / search_doc |
| 上下文 | `pipeline/context.py` | select_context_docs / direct_faq_answer / build_context |
| 运行时 | `pipeline/runtime.py` | RAGQueryContext / create_query_context / record_trace |
| 改写 | `pipeline/rewrite.py` | rewrite_query_if_needed (追问→独立问题) |
| 变体 | `pipeline/query_variants.py` | generate_query_variants (同义检索表达) |
| 引用 | `pipeline/citations.py` | enforce_answer_citations / enforce_table_row_details |
| 事件 | `pipeline/events.py` | start / status / token / end / error 事件构造 |
| 意图 | `intent/classifier.py` | classify_intent (6级规则 + LLM) |
| 类别 | `intent/question_category.py` | infer_question_category / is_table_query |
| 检索 | `retrieval/store.py` | MilvusHybridStore (Dense+Sparse) |
| 策略 | `retrieval/strategy.py` | build_retrieval_plan (4层决策) |
| 排序 | `retrieval/ranking.py` | merge / sort / rerank |
| 过滤 | `retrieval/filters.py` | build_source_expr / validate_source_filter |
| 模型 | `retrieval/models.py` | get_embeddings / get_reranker |
| 工厂 | `retrieval/factory.py` | get_faq_store / get_doc_store / warmup |
| 数据域 | `governance/data_scope.py` | DataScope / resolve_data_scope / expr_clauses |
| 版本 | `governance/kb_versions.py` | KnowledgeBaseVersionStore / activate / archive / ensure |
| 场景 | `scenarios/registry.py` | ScenarioRegistry / ScenarioDefinition |
| 边界 | `scenarios/boundary.py` | detect_scenario_boundary / detect_source_boundary |
| LLM | `llm/client.py` | get_chat_model (ChatOpenAI) |
| 提示词 | `prompts/selector.py` | build_answer_prompt_profile (类别>意图>默认) |
| 历史 | `memory/history.py` | ChatHistoryStore (摘要+最近消息+写入) |
| 反馈 | `memory/feedback.py` | FeedbackStore |
| 入库 | `indexing/service.py` | ingest_directory (文档→Milvus) |
| 质量 | `quality/ingestion.py` | build_ingestion_quality_report |
| 追踪 | `observability/langsmith_adapter.py` | record_query_trace → LangSmith |
