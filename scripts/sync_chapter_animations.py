"""Generate unified chapter animation pages and add lecture links.

The chapter animation pages are standalone HTML files under docs/animation/.
This script keeps their visual style consistent and makes sure each 05-19
lecture links to its own chapter animation and the business flow diagram.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
ANIMATION_DIR = DOCS_DIR / "animation"


CHAPTERS: list[dict[str, object]] = [
    {
        "no": "05",
        "title": "意图分类与路由入口",
        "doc": "05-intent-classification.md",
        "html": "05-intent-flow.html",
        "path": "codealong/chapters/ch05_intent_classification",
        "desc": "从用户问题进入系统开始，先用低成本规则完成直答、边界、FAQ 快路径前置判断和检索意图分类。",
        "nodes": [
            ("DEMO", "demo_intent.py", "scripts/demo_intent.py", "main()", "课堂命令行入口，构造 query、history 和 source_filter。"),
            ("RULES", "规则配置", "qa_core/config/rules.py", "get_rule_config()", "读取 config/rules.toml 中的 FAQ fast path 触发词和查询变体规则。"),
            ("SCENARIO", "场景解析", "qa_core/scenarios/registry.py", "resolve_scenario()", "加载当前业务场景、source 白名单、业务域和联系方式。"),
            ("SOURCE", "source 推断", "qa_core/scenarios/boundary.py", "score_source_matches() / detect_source_boundary()", "根据场景 source_patterns 推断问题所属资料分类。"),
            ("DIRECT", "直答意图", "qa_core/intent/classifier.py", "classify_direct_intent()", "识别空问题、问候、感谢、越界和短句转人工。"),
            ("ROUTE", "入口路由", "qa_core/pipeline/steps.py", "decide_route()", "决定 direct_answer、source_boundary 或 retrieval。"),
            ("FAST", "FAQ 快路径前置判断", "qa_core/pipeline/steps.py", "should_try_faq_fast_path()", "短标准问答只标记为适合优先查 FAQ；真实 FAQ 命中从第 09/10 章的 Pipeline 上下文中调用 Milvus FAQ collection。"),
            ("CLASSIFY", "检索意图", "qa_core/intent/classifier.py", "classify_intent()", "进入检索前输出 FOLLOW_UP、FAQ_QUERY 或 KNOWLEDGE_QUERY。"),
            ("OUT", "章节输出", "qa_core/pipeline/route.py", "RouteDecision", "给第 06 章的检索计划提供稳定输入。"),
        ],
    },
    {
        "no": "06",
        "title": "检索策略与动态计划",
        "doc": "06-retrieval-strategy.md",
        "html": "06-retrieval-flow.html",
        "path": "codealong/chapters/ch06_retrieval_strategy",
        "desc": "接住第 05 章 route=retrieval 的问题，把 IntentResult 转成可执行 RetrievalPlan。",
        "nodes": [
            ("INPUT", "检索入口", "qa_core/pipeline/steps.py", "decide_route()", "只有 route=retrieval 才继续进入本章逻辑。"),
            ("INTENT", "意图结果", "qa_core/intent/classifier.py", "classify_intent()", "沿用第 05 章的 FAQ_QUERY、KNOWLEDGE_QUERY、FOLLOW_UP。"),
            ("CATEGORY", "问题类别", "qa_core/intent/question_category.py", "infer_question_category()", "识别 pricing、compliance、troubleshooting、summary 等风险类别。"),
            ("TABLE", "表格偏好", "qa_core/intent/question_category.py", "is_table_query()", "表格/清单/字段类问题禁用模糊 FAQ 直出。"),
            ("BASE", "参数基线", "qa_core/config/settings.py", "Settings", "定义 top_k、阈值、上下文长度和短问题阈值。"),
            ("PLAN", "计划生成", "qa_core/retrieval/strategy.py", "build_retrieval_plan()", "按意图、短问题、风险类别、表格偏好逐层调整参数。"),
            ("OUT", "章节输出", "qa_core/retrieval/strategy.py", "RetrievalPlan", "输出 run_faq、run_doc、top_k、阈值和 use_query_variants。"),
        ],
    },
    {
        "no": "07",
        "title": "查询改写与变体生成",
        "doc": "07-query-rewrite-variants.md",
        "html": "07-query-flow.html",
        "path": "codealong/chapters/ch07_query_rewrite_variants",
        "desc": "补齐 Stage 2：把依赖历史的追问改写成独立问题，并生成多路查询变体。",
        "nodes": [
            ("PREP", "检索准备", "scripts/demo_query_prepare.py", "main()", "串联 route、intent、rewrite、plan 和 query_variants。"),
            ("HISTORY", "历史格式化", "qa_core/memory/history.py", "format_messages()", "把最近对话压缩为追问改写的上下文。"),
            ("REWRITE", "追问改写", "qa_core/pipeline/rewrite.py", "rewrite_query_if_needed()", "有历史且 requires_rewrite=True 时调用模型改写。"),
            ("MODEL", "真实 LLM 客户端", "qa_core/llm/client.py", "get_chat_model()", "调用完整项目同款 ChatOpenAI 兼容客户端，环境缺失时直接暴露配置错误。"),
            ("STRUCT", "结构化输出", "qa_core/pipeline/query_variants.py", "QueryVariants", "保留 with_structured_output(QueryVariants) 的主项目同向接口。"),
            ("HEURISTIC", "配置化变体", "qa_core/pipeline/query_variants.py + config/rules.toml", "_heuristic_variants()", "读取 query_variants 替换表，用配置化同义词补充稳定变体。"),
            ("DEDUP", "去重限量", "qa_core/pipeline/query_variants.py", "generate_query_variants()", "保留原问题在第一位，并控制变体数量。"),
            ("OUT", "章节输出", "qa_core/pipeline/query_variants.py", "generate_query_variants()", "把 rewritten_query 和 query_variants 交给第 08 章 search_many()。"),
        ],
    },
    {
        "no": "08",
        "title": "Milvus 混合检索",
        "doc": "08-milvus-hybrid-search.md",
        "html": "08-milvus-hybrid-search.html",
        "path": "codealong/chapters/ch08_milvus_hybrid_search",
        "desc": "接入真实 MilvusHybridStore：按 RetrievalPlan 执行 FAQ/Doc 分层检索，每一路内部使用 Dense + BM25 Hybrid Search。",
        "nodes": [
            ("FACTORY", "Store 工厂", "qa_core/retrieval/factory.py", "get_faq_store() / get_doc_store()", "按 collection 名取得 FAQ 或文档检索 store。"),
            ("INDEX", "写入文档", "qa_core/retrieval/store.py", "add_documents()", "把 Document 写入真实 Milvus collection。"),
            ("FILTER", "过滤表达式", "qa_core/retrieval/filters.py", "validate_source_filter() / build_source_expr()", "校验 source 并构建过滤表达式。"),
            ("SEARCH", "多查询检索", "qa_core/retrieval/store.py", "search_many()", "接收 query_variants、top_k、source、kb_version 和 data_scope。"),
            ("VISIBLE", "可见性过滤", "qa_core/retrieval/filters.py", "build_source_expr()", "把 source、知识库版本和数据域转换为 Milvus boolean expr。"),
            ("SCORE", "混合打分", "qa_core/retrieval/store.py", "Collection.hybrid_search()", "由 Milvus dense vector + BM25 sparse function 执行混合召回。"),
            ("MERGE", "合并排序", "qa_core/retrieval/ranking.py", "merge_hits_by_document() / sort_hits_by_score()", "多变体结果按文档去重并排序。"),
            ("RERANK", "重排", "qa_core/retrieval/ranking.py", "rerank_hits()", "保留 CrossEncoder reranker 的工程位置。"),
            ("OUT", "章节输出", "qa_core/retrieval/results.py", "RetrievalResult", "输出 hits、top_score 和 source_payloads。"),
        ],
    },
    {
        "no": "09",
        "title": "QAService 核心编排",
        "doc": "09-qaservice-orchestration.md",
        "html": "09-qaservice-orchestration.html",
        "path": "codealong/chapters/ch09_qaservice_orchestration",
        "desc": "新增应用服务门面，把入口路由、历史、改写、变体和检索能力封装给上层调用。",
        "nodes": [
            ("FACTORY", "服务工厂", "qa_core/application/factory.py", "get_qa_service()", "集中创建并缓存 QAService。"),
            ("HISTORY", "历史存储", "qa_core/memory/history.py", "ChatHistoryStore", "提供 recent_queries、get_context_messages 和 add_turn。"),
            ("SERVICE", "服务门面", "qa_core/application/service.py", "QAService", "API 层只调用服务方法，不关心 RAG 内部细节。"),
            ("VALIDATE", "source 校验", "qa_core/application/service.py", "validate_source()", "委托 retrieval.filters 校验 source_filter。"),
            ("STREAM", "流式问答", "qa_core/application/service.py", "stream_query()", "委托 rag_stream_query() 产出事件流。"),
            ("DEBUG", "检索诊断", "qa_core/application/service.py", "debug_retrieval()", "复用主链路的检索准备与召回，但不生成最终答案。"),
            ("OUT", "章节输出", "qa_core/application/service.py", "QAService.stream_query() / debug_retrieval()", "为第 10 章 Pipeline 下沉和第 12 章 API 入口提供稳定服务接口。"),
        ],
    },
    {
        "no": "10",
        "title": "RAG Pipeline 主流程",
        "doc": "10-rag-pipeline.md",
        "html": "10-rag-pipeline.html",
        "path": "codealong/chapters/ch10_rag_pipeline",
        "desc": "把 QAService 内部编排下沉为可观察的 Stage 0-7 Pipeline。",
        "nodes": [
            ("STREAM", "主流程", "qa_core/pipeline/rag.py", "stream_query()", "串联 route、prepare、FAQ、doc、answer、end。"),
            ("CTX", "请求上下文", "qa_core/pipeline/runtime.py", "create_query_context()", "创建 scenario、DataScope、session_id、trace_id 和 kb_version。"),
            ("EVENT_START", "开始事件", "qa_core/pipeline/runtime.py", "start_event()", "向前端发送 start 事件。"),
            ("ROUTE", "查询路由", "qa_core/pipeline/steps.py", "decide_route()", "处理 direct_answer、faq_exact 或进入 retrieval。"),
            ("PREP", "检索准备", "qa_core/pipeline/steps.py", "prepare_retrieval()", "生成 RetrievalPreparation。"),
            ("FAQ", "FAQ 检索", "qa_core/pipeline/retrieval_steps.py", "search_faq() / get_faq_direct_answer()", "FAQ 高置信可提前结束。"),
            ("DOC", "文档检索", "qa_core/pipeline/retrieval_steps.py", "search_doc()", "FAQ 未直出时继续检索文档。"),
            ("CONTEXT", "上下文构建", "qa_core/pipeline/context.py", "build_context()", "筛选证据并填入 PromptProfile.user_template。"),
            ("ANSWER", "生成与结束", "qa_core/pipeline/rag.py", "_finish_with_single_answer()", "发送 token、保存历史并 finish_success。"),
        ],
    },
    {
        "no": "11",
        "title": "Prompt 工程与 Profile 系统",
        "doc": "11-prompt-engineering.md",
        "html": "11-prompt-engineering.html",
        "path": "codealong/chapters/ch11_prompt_engineering",
        "desc": "把回答模板、风险类别和业务场景口径纳入 Pipeline 生成阶段。",
        "nodes": [
            ("PROFILE", "Profile 结构", "qa_core/prompts/profiles.py", "PromptProfile", "承载 system_template、user_template 和 profile_name。"),
            ("TEMPLATES", "模板常量", "qa_core/prompts/templates.py", "DEFAULT_*", "定义默认回答模板和上下文格式。"),
            ("SELECT", "Profile 选择", "qa_core/prompts/selector.py", "build_answer_prompt_profile()", "按问题类别、意图和场景选择回答口径。"),
            ("SCENARIO", "场景变量", "qa_core/prompts/selector.py", "_scenario_prompt_context()", "把 assistant_name、business_domain、support_contact 注入模板。"),
            ("PROMPT", "用户 Prompt", "qa_core/pipeline/steps.py", "prepare_answer()", "把历史、问题和证据填入选定 Profile。"),
            ("ANSWER", "可靠回答", "qa_core/pipeline/rag.py", "_finish_with_single_answer()", "有证据则流式生成回答并带来源，无证据则明确信息不足。"),
            ("DIAG", "诊断信息", "qa_core/pipeline/rag.py", "retrieval_info['prompt_profile']", "把 Prompt 档位和预览写入诊断信息。"),
        ],
    },
    {
        "no": "12",
        "title": "FastAPI 与异步 Web 框架",
        "doc": "12-fastapi-async.md",
        "html": "12-fastapi-service.html",
        "path": "codealong/chapters/ch12_fastapi_service",
        "desc": "把 QAService 能力暴露为 HTTP 诊断接口和 WebSocket 流式问答接口。",
        "nodes": [
            ("APP", "应用入口", "app.py", "create_app()", "创建 FastAPI 应用并注册路由。"),
            ("ROUTER", "API 路由", "qa_core/api/chat.py", "router", "定义 /api/retrieval/debug 和 /api/stream。"),
            ("SCHEMA", "请求模型", "qa_core/schemas.py", "RetrievalDebugRequest", "用 Pydantic 校验检索诊断请求。"),
            ("CONTEXT", "请求解析", "qa_core/api/service_context.py", "QueryServiceContext.from_ws_payload() / from_debug_request()", "把 WebSocket JSON 或 HTTP debug request 转成服务调用参数。"),
            ("COLLECT", "WebSocket 流式问答", "qa_core/api/chat.py", "websocket_endpoint() / _send_stream_events()", "调用 QAService.stream_query() 并逐条发送事件。"),
            ("DEBUG", "诊断接口", "qa_core/api/chat.py", "debug_retrieval()", "调用 QAService.debug_retrieval()。"),
            ("FEEDBACK", "用户反馈", "qa_core/memory/feedback.py", "FeedbackStore.add_feedback()", "记录 useful/not_useful 和来源快照。"),
            ("WS", "WebSocket 循环", "qa_core/api/chat.py", "websocket_endpoint() / _send_stream_events()", "接收 JSON、发送 start/status/token/end/error。"),
            ("ERROR", "错误处理", "qa_core/api/error_handlers.py", "register_api_exception_handlers()", "统一 Bad Request 和 API 异常响应。"),
        ],
    },
    {
        "no": "13",
        "title": "应用入口与环境前置校验",
        "doc": "13-app-entry-preflight.md",
        "html": "13-preflight-checks.html",
        "path": "codealong/chapters/ch13_preflight_checks",
        "desc": "启动前检查必需配置和运行依赖，避免服务打开后问答链路才失败。",
        "nodes": [
            ("SETTINGS", "配置对象", "qa_core/config/settings.py", "Settings / get_settings()", "集中读取课堂运行配置。"),
            ("CHECK_VALUE", "值校验", "qa_core/config/preflight.py", "_is_placeholder()", "检查 API Key、管理令牌等必需值不是占位符。"),
            ("CHECK_PATH", "路径校验", "qa_core/config/preflight.py", "_require_path()", "检查本地模型、场景目录和 FAQ 文件存在。"),
            ("CHECK_SCENARIO", "场景校验", "qa_core/scenarios/registry.py", "resolve_scenario()", "确认 active_scenario_id 可解析。"),
            ("PREFLIGHT", "前置校验", "qa_core/config/preflight.py", "validate_runtime_environment()", "按固定顺序汇总运行环境检查。"),
            ("VALIDATE", "启动守卫", "qa_core/config/preflight.py", "validate_runtime_environment()", "任一失败直接抛错，阻止启动。"),
            ("APP", "应用接入", "app.py", "create_app()", "创建应用时执行前置校验。"),
            ("LOG", "日志", "qa_core/config/logging_config.py", "get_logger()", "统一课堂运行日志输出。"),
        ],
    },
    {
        "no": "14",
        "title": "知识库多版本管理",
        "doc": "14-kb-versioning.md",
        "html": "14-kb-versioning.html",
        "path": "codealong/chapters/ch14_kb_versioning",
        "desc": "用 MySQL 控制面实现 STAGED、ACTIVE、ARCHIVED 状态机，并把 active 版本接入在线检索。",
        "nodes": [
            ("GEN", "版本号", "qa_core/governance/kb_versions.py", "generate_kb_version()", "生成带场景和时间信息的知识库版本号。"),
            ("COMMON", "公共时间工具", "qa_core/common.py", "utc_now() / utc_file_stamp()", "统一版本时间和文件时间戳格式。"),
            ("MODEL", "版本对象", "qa_core/governance/kb_versions.py", "KnowledgeBaseVersion", "保存版本状态、描述、时间和入库统计。"),
            ("MYSQL", "MySQL 存储基类", "qa_core/memory/base.py", "_MySqlStore", "提供 SQLAlchemy 引擎和安全表名校验。"),
            ("STORE", "版本仓库", "qa_core/governance/kb_versions.py", "KnowledgeBaseVersionStore", "读取和写入 MySQL 版本表与 active 指针。"),
            ("ENSURE", "确保版本", "qa_core/governance/kb_versions.py", "ensure_version()", "入库时创建或复用 STAGED 版本。"),
            ("RESULT", "入库记录", "qa_core/governance/kb_versions.py", "record_ingest_result()", "记录 doc/faq 数量和 source。"),
            ("ACTIVE", "激活归档", "qa_core/governance/kb_versions.py", "activate_version() / archive_version()", "控制当前在线检索可见版本。"),
            ("API", "版本 API", "qa_core/api/kb_versions.py", "list/activate/archive payload", "提供轻量版本管理接口。"),
            ("QUERY", "在线接入", "qa_core/pipeline/runtime.py", "active_kb_version", "查询时把版本写入 RAGQueryContext。"),
        ],
    },
    {
        "no": "15",
        "title": "数据隔离与多租户",
        "doc": "15-data-isolation.md",
        "html": "15-data-isolation.html",
        "path": "codealong/chapters/ch15_data_isolation",
        "desc": "让 tenant、dataset、visibility、roles 与 source、kb_version 一起参与检索过滤。",
        "nodes": [
            ("CLEAN", "参数清洗", "qa_core/governance/data_scope.py", "_clean_token() / _clean_list()", "规范化租户、数据集和角色字段。"),
            ("SCOPE", "数据域", "qa_core/governance/data_scope.py", "DataScope", "表达 tenant_id、dataset_id、visibility 和 user_roles。"),
            ("RESOLVE", "解析数据域", "qa_core/governance/data_scope.py", "resolve_data_scope()", "从 API 参数构造 DataScope。"),
            ("ESCAPE", "表达式转义", "qa_core/governance/data_scope.py", "escape_expr_value()", "防止过滤表达式注入。"),
            ("SOURCE", "source 过滤", "qa_core/retrieval/filters.py", "build_source_expr()", "把 source_filter、kb_version、DataScope 拼成过滤表达式。"),
            ("CONTEXT", "上下文接入", "qa_core/pipeline/runtime.py", "create_query_context()", "每次请求都携带 data_scope。"),
            ("STORE", "结果过滤", "qa_core/retrieval/store.py", "search_many(..., data_scope=...)", "真实 Milvus 检索表达式同时携带 source、kb_version 和 DataScope。"),
            ("TEST", "隔离测试", "tests/test_data_scope.py", "DataScopeChapter15Test", "用测试锁住 DataScope 解析、过滤表达式和检索入参。"),
        ],
    },
    {
        "no": "16",
        "title": "文档入库与索引链路",
        "doc": "16-ingestion-pipeline.md",
        "html": "16-ingestion-pipeline.html",
        "path": "codealong/chapters/ch16_ingestion_pipeline",
        "desc": "把文档和 FAQ 处理成带版本、source 和数据域 metadata 的可检索资料。",
        "nodes": [
            ("INGEST", "入库编排", "qa_core/indexing/service.py", "ingest_directory()", "串联场景、数据域、版本、加载、切分和写入。"),
            ("FILES", "文件发现", "qa_core/indexing/service.py", "os.walk()", "遍历目录文件，并在单文件处理阶段校验 LoaderSpec。"),
            ("LOAD", "多格式加载", "qa_core/indexing/document_loaders.py", "load_file()", "按后缀选择文本、PDF、Word、PPT 或表格 loader。"),
            ("TABLE", "表格行文档", "qa_core/indexing/table_documents.py", "load_table_file()", "把 CSV/Excel 每一行转换为保留表头、sheet 和行号的 Document。"),
            ("UTILS", "稳定 ID 工具", "qa_core/utils.py", "stable_hash() / file_fingerprint()", "统一生成 doc_id、chunk_id、faq_id 和文件指纹。"),
            ("FINGER", "文件指纹", "qa_core/utils.py", "file_fingerprint()", "计算 SHA256 文件指纹。"),
            ("MANIFEST", "增量清单", "qa_core/indexing/manifest.py", "IndexManifest", "用 MySQL 记录文件指纹和 chunk_id，支持跳过未变化文件和删除旧 chunk。"),
            ("META", "来源 metadata", "qa_core/document_metadata.py", "format_source_label() / is_table_metadata()", "统一普通文档和表格行的来源标签。"),
            ("NORMALIZE", "元数据标准化", "qa_core/indexing/document_normalizer.py", "normalize_documents()", "补 source、kb_version、scenario_id 和 data_scope。"),
            ("CHUNK", "文档切分", "qa_core/indexing/chunking.py", "split_documents()", "普通文本做 parent-child 切分，表格行保持完整语义单元。"),
            ("CITE", "引用兜底", "qa_core/pipeline/citations.py", "enforce_answer_citations()", "模型漏写来源时补充参考来源和表格行要点。"),
            ("WRITE", "写入 store", "qa_core/indexing/service.py", "store.delete_ids() / add_documents()", "先清理旧 chunk，再写入新 chunk。"),
            ("FAQ", "FAQ 入库", "qa_core/indexing/faq_ingestion.py", "faq_documents_from_csv() / ingest_faq_csv()", "把 FAQ CSV 写入 FAQ store。"),
            ("VERSION", "版本记录", "qa_core/governance/kb_versions.py", "record_ingest_result() / activate_version()", "记录入库统计并可激活版本。"),
        ],
    },
    {
        "no": "17",
        "title": "RAG 回归验收与入库质量",
        "doc": "17-quality-evaluation.md",
        "html": "17-quality-evaluation.html",
        "path": "codealong/chapters/ch17_quality_evaluation",
        "desc": "用质量报告和回归思维证明知识资料不是随便入库，回答效果也不是凭感觉判断。",
        "nodes": [
            ("REPORT", "质量报告", "qa_core/quality/ingestion.py", "build_ingestion_quality_report()", "扫描候选文件并汇总入库质量问题。"),
            ("FAQ_READ", "FAQ 读取", "qa_core/quality/faq.py", "read_faq_records()", "读取 FAQ CSV 为检查记录。"),
            ("FAQ_ANALYZE", "FAQ 检查", "qa_core/quality/faq.py", "analyze_faq_csv()", "检查必填项、重复问题和非法 source。"),
            ("CONFLICT", "冲突检测", "qa_core/quality/conflicts.py", "detect_faq_document_conflicts()", "识别 FAQ 与文档中的数字、极性和关键词冲突。"),
            ("META", "资料类型判断", "qa_core/document_metadata.py", "is_table_metadata()", "区分普通 chunk 与表格行，避免误报。"),
            ("CHUNK", "Chunk 质量", "qa_core/quality/chunk.py", "analyze_chunk_quality()", "检查 chunk 长度、空白和 metadata。"),
            ("DEMO", "课堂演示", "scripts/demo_quality_report.py", "main()", "输出一份可阅读的质量报告。"),
            ("TEST", "回归测试", "tests/test_quality_report.py", "QualityReportChapter17Test", "锁住质量报告结构和关键告警。"),
        ],
    },
    {
        "no": "18",
        "title": "测试与接口验收",
        "doc": "18-testing-system.md",
        "html": "18-testing-system.html",
        "path": "codealong/chapters/ch18_test_system",
        "desc": "把章节结构、单元测试、QAService/API 冒烟、检索诊断和质量报告收束为自动化验收入口。",
        "nodes": [
            ("UNIT", "单元测试", "tests/test_test_system.py", "TestSystemChapter18Test", "覆盖章节测试系统本身。"),
            ("INTENT", "守护检查测试", "tests/test_test_system.py", "test_guardrails_pass_for_current_chapter()", "通过守护检查锁住章节结构、核心文件和真实链路规则。"),
            ("RETRIEVAL", "冒烟链路测试", "tests/test_test_system.py", "test_acceptance_smoke_exercises_core_chain()", "通过验收冒烟锁住 QAService、追问改写、检索诊断和质量 gate。"),
            ("GUARD", "项目守护检查", "scripts/check_project_guardrails.py", "run_guardrails()", "检查章节结构、测试入口和关键文件。"),
            ("SMOKE", "验收冒烟", "scripts/acceptance_smoke.py", "run_acceptance_smoke()", "串联 QAService、API、质量 gate 和诊断入口。"),
            ("REPORT", "失败报告", "scripts/acceptance_smoke.py", "main()", "失败时返回非零退出码，方便作为门禁。"),
            ("OUT", "章节输出", "scripts + tests", "python -m unittest / acceptance_smoke", "上线前确认代码没有破坏主链路。"),
        ],
    },
    {
        "no": "19",
        "title": "LangSmith 观测、Trace 与生产化部署",
        "doc": "19-observability-tracing.md",
        "html": "19-observability-tracing.html",
        "path": "codealong/chapters/ch19_observability_tracing",
        "desc": "在测试门禁基础上补上运行时观测，让每次问答都能追踪阶段耗时、命中路径和错误信息。",
        "nodes": [
            ("CONFIG", "环境配置", "qa_core/observability/langsmith_adapter.py", "configure_langsmith_environment()", "把 Settings 中的 LangSmith 配置写入环境变量。"),
            ("STATUS", "观测状态", "qa_core/observability/langsmith_adapter.py", "langsmith_enabled() / langsmith_status()", "判断 trace 是否启用并返回状态信息。"),
            ("CTX", "Trace 上下文", "qa_core/pipeline/runtime.py", "RAGQueryContext", "新增 trace_id、stage_timings_ms 和 first_token_ms。"),
            ("STAGE", "阶段计时", "qa_core/pipeline/runtime.py", "run_stage() / stage()", "记录每个阶段耗时。"),
            ("TOKEN", "首 token", "qa_core/pipeline/runtime.py", "mark_first_token()", "记录首 token 延迟。"),
            ("CITE", "引用补强", "qa_core/pipeline/citations.py", "enforce_answer_citations()", "回答补充来源后仍进入同一条 trace。"),
            ("FINALIZE", "汇总耗时", "qa_core/pipeline/runtime.py", "finalize_timings()", "写入 total_elapsed_ms、slowest_stage 等诊断字段。"),
            ("TRACE", "Trace 写入", "qa_core/pipeline/runtime.py", "record_trace()", "统一调用 record_query_trace()。"),
            ("FINISH", "成功/失败收口", "qa_core/pipeline/runtime.py", "finish_success() / finish_error()", "end/error 事件和 trace metadata 一起落地。"),
            ("DEMO", "运行演示", "scripts/demo_observability.py", "main()", "执行真实问答事件流，并通过 LangSmith adapter 暴露 trace 启用状态。"),
        ],
    },
]


BUSINESS_NODES_BY_CHAPTER: dict[str, list[tuple[str, str, str, str]]] = {
    "05": [
        ("B_INPUT", "用户问题进入", "业务输入", "接收一个企业知识问题，以及可选的 source_filter。"),
        ("B_SOURCE", "识别业务分类", "分类边界", "根据场景词表判断问题更像 finance、hr、it 等哪个资料域。"),
        ("B_DIRECT", "低成本直答", "直答闭环", "问候、感谢、空问题、越界和转人工不进入检索，直接给出确定回应。"),
        ("B_BOUNDARY", "防止分类选错", "边界保护", "用户选了错误分类时提示切换，避免按错误资料回答。"),
        ("B_FAST", "FAQ 快路径前置判断", "标准问答入口", "短标准问题先被标记为适合优先查 FAQ，真实命中由后续 Milvus FAQ 检索完成。"),
        ("B_RETRIEVAL", "进入检索分支", "下章输入", "无法确定性回答时输出 route=retrieval。"),
        ("B_OUT", "稳定路由结果", "本章交付", "交付 RouteDecision 和 IntentResult，作为第 06 章检索计划输入。"),
    ],
    "06": [
        ("B_INPUT", "接收检索类问题", "上章输出", "只处理第 05 章判定为 route=retrieval 的问题。"),
        ("B_INTENT", "理解问题意图", "业务意图", "区分 FAQ、知识库问题、追问等不同检索诉求。"),
        ("B_RISK", "判断问题类别", "风险分层", "费用、合规、排障、总结等问题使用不同检索策略。"),
        ("B_TABLE", "识别表格诉求", "检索偏好", "表格、清单、字段类问题避免被模糊 FAQ 提前截断。"),
        ("B_PLAN", "生成动态计划", "检索参数", "把业务特征转成 top_k、阈值、FAQ/doc 开关和 rerank 选项。"),
        ("B_OUT", "交付检索计划", "本章交付", "输出 RetrievalPlan，给第 07 章改写和查询变体使用。"),
    ],
    "07": [
        ("B_INPUT", "接收检索计划", "上章输出", "读取 RetrievalPlan.use_query_variants 和意图里的 requires_rewrite。"),
        ("B_HISTORY", "理解追问上下文", "会话语义", "从最近对话中提取当前追问依赖的主题。"),
        ("B_REWRITE", "改写独立问题", "追问闭环", "把“那流程呢”这类问题改写成可独立检索的问题。"),
        ("B_VARIANTS", "生成多种说法", "召回增强", "为同一个问题生成多路检索表达，提升召回覆盖。"),
        ("B_RULES", "配置化规则补充", "稳定召回", "规则覆盖的高频表达直接从 rules.toml 生成变体；规则未覆盖时调用真实 LLM structured output。"),
        ("B_OUT", "交付查询包", "本章交付", "输出 rewritten_query 和 query_variants，交给第 08 章 search_many()。"),
    ],
    "08": [
        ("B_INPUT", "接收多路查询", "上章输出", "拿到 rewritten_query 和 query_variants。"),
        ("B_FILTER", "限定可见资料", "业务过滤", "按 source、知识库版本和数据域排除不可见资料。"),
        ("B_FAQ", "检索 FAQ", "标准问答召回", "优先召回标准问题和标准答案。"),
        ("B_DOC", "检索文档", "知识片段召回", "从文档 chunk 中召回可作为证据的内容。"),
        ("B_SCORE", "混合打分与重排", "排序闭环", "结合 dense、sparse、去重和 rerank 形成最终候选。"),
        ("B_OUT", "交付召回结果", "本章交付", "输出 RetrievalResult、top_score 和 source_payloads。"),
    ],
    "09": [
        ("B_INPUT", "上层需要问答服务", "服务入口", "API 或脚本不应直接理解 Pipeline 内部细节。"),
        ("B_FACADE", "统一服务门面", "应用服务闭环", "QAService 统一承接问答、诊断、source 校验和历史。"),
        ("B_HISTORY", "维护会话历史", "上下文能力", "保存最近问题，为追问改写和连续问答提供上下文。"),
        ("B_STREAM", "对外提供流式问答", "问答接口", "把 RAG 事件流包装成稳定服务方法。"),
        ("B_DEBUG", "对外提供检索诊断", "排障接口", "复用检索准备和召回，不生成最终答案。"),
        ("B_OUT", "交付服务层 API", "本章交付", "为第 10 章 Pipeline 下沉和第 12 章 Web API 提供稳定入口。"),
    ],
    "10": [
        ("B_INPUT", "接收一次问答请求", "在线问答入口", "携带问题、分类、会话、场景、版本和数据域。"),
        ("B_CONTEXT", "建立请求上下文", "运行态", "统一保存 session、trace、scenario、DataScope 和 KB 版本。"),
        ("B_ROUTE", "先路由再检索", "成本控制", "直答和 FAQ 精确命中可以提前结束。"),
        ("B_PREPARE", "准备检索参数", "Stage 2", "加载历史、分类、改写、计划、变体和 PromptProfile。"),
        ("B_RETRIEVE", "FAQ 与文档召回", "证据获取", "先 FAQ 后文档，按计划检索候选。"),
        ("B_ANSWER", "组织证据并回答", "生成闭环", "构造 Prompt，生成回答或明确说明信息不足。"),
        ("B_EVENTS", "输出事件流", "本章交付", "交付 start/status/token/end/error 的完整在线问答闭环。"),
    ],
    "11": [
        ("B_INPUT", "接收问题和证据", "生成入口", "第 10 章已经召回候选证据，本章决定如何表达。"),
        ("B_PROFILE", "选择回答口径", "Prompt Profile", "按意图、问题类别和场景选择不同模板。"),
        ("B_SCENARIO", "注入业务语境", "场景变量", "把业务域、助手名称和联系方式写入回答约束。"),
        ("B_PROMPT", "构造用户 Prompt", "生成输入", "把历史、问题和上下文填入模板。"),
        ("B_GROUNDED", "生成有依据回答", "可靠回答", "有证据时基于知识库回答，无证据时明确说信息不足。"),
        ("B_OUT", "交付 Prompt 诊断", "本章交付", "把 PromptProfile 和 prompt preview 写入 retrieval_info。"),
    ],
    "12": [
        ("B_CLIENT", "客户端接入", "Web 入口", "浏览器或调用方通过 HTTP/WebSocket 使用问答能力。"),
        ("B_SCHEMA", "校验请求模型", "接口契约", "检索诊断请求先进入 RetrievalDebugRequest，避免 API 层散落字段判断。"),
        ("B_PAYLOAD", "解析请求载荷", "参数标准化", "把 query、source、session、scope、kb_version 转成服务参数。"),
        ("B_STREAM", "WebSocket 流式问答", "实时体验", "持续发送 start/status/token/end/error。"),
        ("B_DEBUG", "HTTP 检索诊断", "排障体验", "直接查看意图、计划、变体、来源和耗时。"),
        ("B_FEEDBACK", "记录用户反馈", "质量资产", "用户对答案的赞踩和来源快照进入 FeedbackStore。"),
        ("B_ERROR", "统一错误响应", "API 稳定性", "空问题、非法 JSON 和业务异常都有明确返回。"),
        ("B_OUT", "交付 Web 服务闭环", "本章交付", "QAService 能力正式暴露给 Web/API 调用方。"),
    ],
    "13": [
        ("B_START", "服务准备启动", "启动入口", "应用启动前先确认运行环境是否满足要求。"),
        ("B_SETTINGS", "读取集中配置", "配置闭环", "统一读取应用名、模型、路径、场景和观测配置。"),
        ("B_CHECK", "逐项前置校验", "环境检查", "检查占位值、必需路径和 active scenario。"),
        ("B_FAIL", "失败即阻止启动", "防故障外溢", "依赖不完整时在启动阶段失败，而不是等用户提问才失败。"),
        ("B_HEALTH", "健康检查可见", "运维入口", "健康接口返回服务状态和前置校验摘要。"),
        ("B_OUT", "交付启动守卫", "本章交付", "服务具备可解释、可排查的启动前置校验闭环。"),
    ],
    "14": [
        ("B_NEED", "一次入库需要版本", "版本入口", "知识资料不能直接覆盖线上版本，需要先进入可管理版本。"),
        ("B_STAGE", "创建或复用 STAGED", "暂存状态", "新资料先写入暂存版本。"),
        ("B_STORE", "MySQL 控制面", "状态存储", "版本记录和 active 指针由 MySQL 事务统一负责。"),
        ("B_RECORD", "记录入库结果", "版本统计", "记录 doc/faq 数量、source 和描述。"),
        ("B_ACTIVATE", "激活或归档版本", "状态机", "控制哪个版本成为线上检索版本。"),
        ("B_QUERY", "查询使用指定版本", "在线接入", "请求可指定 kb_version，也可使用 active/default 版本。"),
        ("B_OUT", "交付版本治理", "本章交付", "知识库具备 STAGED、ACTIVE、ARCHIVED 的版本闭环。"),
    ],
    "15": [
        ("B_REQUEST", "请求携带数据域", "租户入口", "用户请求带 tenant、dataset、visibility 和 roles。"),
        ("B_SCOPE", "解析 DataScope", "隔离上下文", "形成统一的数据可见性对象。"),
        ("B_WRITE", "入库写入隔离元数据", "数据治理", "每个文档 chunk 带上租户、数据集、可见性和角色。"),
        ("B_FILTER", "检索构建过滤条件", "查询治理", "source、kb_version 和 DataScope 一起进入过滤表达式。"),
        ("B_VISIBLE", "只返回可见资料", "隔离验证", "Milvus 过滤表达式保证不可见资料不进入候选集。"),
        ("B_OUT", "交付多租户隔离", "本章交付", "同一系统可按租户、数据集和角色隔离知识。"),
    ],
    "16": [
        ("B_FILES", "准备知识资料", "入库输入", "接收 PDF、DOCX、MD、TXT、CSV、Excel 和 FAQ CSV。"),
        ("B_LOAD", "发现并加载文件", "文档读取", "只处理有 loader 的受支持文件。"),
        ("B_TABLE", "表格按行入库", "结构化资料", "CSV/Excel 每行保留表头、工作表、行号和单元格键值。"),
        ("B_ID", "生成稳定 ID", "可重复入库", "统一使用 stable_hash 和 file_fingerprint 生成 doc/chunk/faq 标识。"),
        ("B_META", "形成可引用来源", "可追溯性", "普通文档、FAQ 和表格行都生成统一 citation 标签。"),
        ("B_NORMALIZE", "补齐业务元数据", "标准化", "写入 source、scenario、kb_version、fingerprint 和 DataScope。"),
        ("B_CHUNK", "切分可检索片段", "索引粒度", "清洗文本并生成稳定 chunk id。"),
        ("B_CITE", "回答补引用", "答案可复核", "模型漏写来源编号时，后处理追加参考来源或表格行要点。"),
        ("B_FAQ", "标准化 FAQ", "问答资料", "把 FAQ CSV 转成标准问题、答案和 source。"),
        ("B_WRITE", "写入检索 Store", "入库闭环", "先删除旧 id，再写入新 chunk/FAQ。"),
        ("B_OUT", "上线可检索", "本章交付", "新资料进入在线 FAQ/doc 检索链路。"),
    ],
    "17": [
        ("B_INPUT", "入库前后需要质检", "质量入口", "知识资料不能只看能否写入，还要看质量是否可靠。"),
        ("B_FAQ", "检查 FAQ 数据", "FAQ 质量", "发现必填缺失、重复问题和非法 source。"),
        ("B_CONFLICT", "检查知识冲突", "一致性", "识别 FAQ 与文档间数字、极性和关键词冲突。"),
        ("B_META", "识别资料类型", "质量边界", "表格行使用不同质量判断，避免把短表格单元格误判为低质量。"),
        ("B_CHUNK", "检查 Chunk 质量", "召回质量", "发现过短、空白或 metadata 不完整的 chunk。"),
        ("B_REPORT", "生成质量报告", "可解释报告", "把问题汇总成课堂可读报告。"),
        ("B_OUT", "交付质量门禁", "本章交付", "为入库和回归验收提供质量依据。"),
    ],
    "18": [
        ("B_CHANGE", "代码或资料发生变化", "验收入口", "每次改动都需要证明主链路没有被破坏。"),
        ("B_UNIT", "运行单元测试", "局部验证", "锁住意图、检索、Prompt、质量等关键模块。"),
        ("B_GUARD", "执行项目守护检查", "结构门禁", "检查章节结构、入口脚本和关键文件。"),
        ("B_SMOKE", "执行验收冒烟", "端到端验证", "串联 QAService、API、质量 gate 和诊断入口。"),
        ("B_REPORT", "失败可定位", "反馈闭环", "失败时返回非零退出码和明确报告。"),
        ("B_OUT", "交付测试系统", "本章交付", "形成上线前可重复执行的自动化验收入口。"),
    ],
    "19": [
        ("B_QUERY", "一次问答开始运行", "观测入口", "需要知道它走了什么路径、花了多久、是否失败。"),
        ("B_TRACE", "生成 trace_id", "追踪标识", "每次请求都有可串联的追踪 id。"),
        ("B_TIMING", "记录阶段耗时", "性能定位", "记录 route、prepare、retrieval、prompt、generate 等阶段耗时。"),
        ("B_TOKEN", "记录首 token 延迟", "体验指标", "衡量用户首次看到回答的等待时间。"),
        ("B_CITE", "记录引用补强", "可追溯答案", "回答追加参考来源后仍作为最终答案进入 trace。"),
        ("B_FINISH", "成功或失败统一收口", "运行闭环", "end/error 事件都带上 trace metadata。"),
        ("B_STATUS", "查看观测状态", "观测配置", "通过 langsmith_status() 明确当前 trace 是否启用，环境未配置时直接说明未启用。"),
        ("B_OUT", "交付可观测闭环", "本章交付", "系统具备 trace、耗时、来源、错误和诊断复盘能力。"),
    ],
}


def escape_text(value: str) -> str:
    return html.escape(value, quote=True)


def code_flow_nodes(chapter: dict[str, object]) -> list[dict[str, str]]:
    no = str(chapter["no"])
    return [
        {
            "id": node[0],
            "title": node[1],
            "file": node[2],
            "symbol": node[3],
            "body": node[4],
            "chapter": f"第 {no} 章",
        }
        for node in chapter["nodes"]  # type: ignore[index]
    ]


def business_flow_nodes(chapter: dict[str, object]) -> list[dict[str, str]]:
    no = str(chapter["no"])
    business_nodes = BUSINESS_NODES_BY_CHAPTER.get(no)
    if not business_nodes:
        return [
            {
                "id": f"B_{node[0]}",
                "title": node[1],
                "file": "业务执行流程",
                "symbol": "章节闭环",
                "body": node[4],
                "chapter": f"第 {no} 章",
            }
            for node in chapter["nodes"]  # type: ignore[index]
        ]
    return [
        {
            "id": node[0],
            "title": node[1],
            "file": "业务执行流程",
            "symbol": node[2],
            "body": node[3],
            "chapter": f"第 {no} 章",
        }
        for node in business_nodes
    ]


def node_label(node: dict[str, str]) -> str:
    return (
        f"{escape_text(node['title'])}<br/>"
        f"<span>{escape_text(node['file'])}</span><br/>"
        f"<em>{escape_text(node['symbol'])}</em>"
    )


def build_diagram(nodes: list[dict[str, str]], *, direction: str) -> str:
    lines = [f"flowchart {direction}"]
    for node in nodes:
        node_id = node["id"]
        lines.append(f'    {node_id}["{node_label(node)}"]')
    for left, right in zip(nodes, nodes[1:]):
        lines.append(f"    {left['id']} --> {right['id']}")
    for index, node in enumerate(nodes):
        title = node["title"]
        file_path = node["file"]
        symbol = node["symbol"]
        if index == 0:
            color = "#1e3a5f"
            stroke = "#60a5fa"
        elif index == len(nodes) - 1:
            color = "#064e3b"
            stroke = "#34d399"
        elif any(word in title or word in file_path or word in symbol for word in ("检索", "Store", "入库", "写入", "召回", "索引")):
            color = "#0f766e"
            stroke = "#22d3ee"
        elif any(word in title or word in symbol for word in ("Prompt", "模型", "生成", "回答", "改写", "变体")):
            color = "#4c1d95"
            stroke = "#a78bfa"
        elif any(word in title or word in symbol for word in ("测试", "质量", "Trace", "观测", "门禁", "治理", "隔离", "版本")):
            color = "#064e3b"
            stroke = "#34d399"
        else:
            color = "#713f12"
            stroke = "#fbbf24"
        lines.append(f"    style {node['id']} fill:{color},stroke:{stroke},stroke-width:2px")
    return "\n".join(lines)


def build_html(chapter: dict[str, object]) -> str:
    no = str(chapter["no"])
    title = str(chapter["title"])
    doc = str(chapter["doc"]).replace(".md", ".html")
    business_nodes = business_flow_nodes(chapter)
    code_nodes = code_flow_nodes(chapter)
    business_diagram = build_diagram(business_nodes, direction="LR")
    code_diagram = build_diagram(code_nodes, direction="TD")
    views_json = json.dumps(
        {
            "business": {
                "title": f"第 {no} 章业务执行流程图",
                "badge": "业务闭环",
                "desc": f"从业务视角说明第 {no} 章解决什么问题、如何形成可运行闭环。",
                "nodes": business_nodes,
            },
            "code": {
                "title": f"第 {no} 章代码执行流程图",
                "badge": "代码调用",
                "desc": f"从代码视角串联第 {no} 章核心文件、函数和执行顺序。",
                "nodes": code_nodes,
            },
        },
        ensure_ascii=False,
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<!-- chapter-animation-template: unified-v2 -->
<title>第 {no} 章 - {escape_text(title)}业务与代码执行动画</title>
<script src="../assets/vendor/mermaid.min.js"></script>
<style>
:root{{color-scheme:dark;--bg:#090f1d;--panel:#101827;--panel-2:#131d30;--line:#26364f;--text:#eef5ff;--muted:#a6b4c8;--cyan:#38bdf8;--emerald:#34d399;--amber:#fbbf24;--violet:#a78bfa}}
*{{box-sizing:border-box}}html,body{{height:100%}}body{{margin:0;background:var(--bg);color:var(--text);font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif;font-size:16px;overflow:hidden;letter-spacing:0}}button{{font:inherit}}
.topbar{{min-height:86px;display:grid;grid-template-columns:minmax(300px,1fr) auto;gap:16px;align-items:center;padding:12px 22px;background:#0c1424;border-bottom:1px solid var(--line)}}.brand h1{{margin:0;font-size:23px;line-height:1.25;color:#e8f8ff}}.brand p{{margin:5px 0 0;color:var(--muted);font-size:14px;line-height:1.45}}.tabs{{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;margin-bottom:8px}}.tab-btn,.tool-btn{{min-height:38px;border:1px solid var(--line);border-radius:8px;background:#101b2d;color:#dbeafe;cursor:pointer;padding:0 13px;transition:background .18s ease,border-color .18s ease,color .18s ease}}.tab-btn:hover,.tool-btn:hover{{background:#162641;border-color:#46658f}}.tab-btn.active{{background:var(--cyan);border-color:var(--cyan);color:#05111f;font-weight:800}}.tools{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}}.tool-btn.primary{{background:#164e63;border-color:#22d3ee;color:#ecfeff;font-weight:800}}.tool-btn.icon{{width:40px;padding:0;display:inline-grid;place-items:center;font-weight:800}}.zoom-label{{min-width:54px;text-align:center;color:#b8c7dc;font-size:14px}}
.workspace{{height:calc(100vh - 86px);display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;padding:16px}}.stage{{min-width:0;border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden;display:flex;flex-direction:column}}.stage-head{{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 18px;border-bottom:1px solid var(--line);background:#0f1829}}.stage-title{{font-size:18px;font-weight:800;color:#f3f8ff;display:flex;align-items:center;gap:10px}}.badge{{display:inline-flex;align-items:center;min-height:23px;padding:0 8px;border:1px solid #36506f;border-radius:999px;background:#142238;color:#b8d7ff;font-size:12px;font-weight:800}}.stage-desc{{font-size:14px;color:var(--muted);line-height:1.5;margin-top:4px}}.progress-wrap{{width:250px;display:grid;gap:7px}}.progress-meta{{display:flex;justify-content:space-between;color:#b6c5d8;font-size:14px}}.progress{{height:8px;border-radius:999px;background:#17243a;overflow:hidden}}.progress span{{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--emerald));transition:width .25s ease}}
.viewport{{position:relative;flex:1;overflow:hidden;cursor:grab;background:linear-gradient(#162136 1px,transparent 1px),linear-gradient(90deg,#162136 1px,transparent 1px);background-size:32px 32px}}.viewport.dragging{{cursor:grabbing}}.canvas{{position:absolute;left:0;top:0;transform-origin:0 0;transition:transform .22s ease}}.canvas.no-animate{{transition:none}}.canvas svg{{max-width:none!important;overflow:visible;font-size:15px!important}}.canvas svg .nodeLabel,.canvas svg .edgeLabel,.canvas svg foreignObject{{font-size:15px!important;line-height:1.42!important}}.loading,.empty-state{{position:absolute;inset:0;display:grid;place-items:center;color:#c8d8eb;font-size:18px;background:rgba(9,15,29,.78)}}.loading[hidden],.empty-state[hidden]{{display:none}}
.side{{min-width:0;display:flex;flex-direction:column;gap:16px}}.panel{{border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden}}.panel h2{{margin:0;padding:14px 16px;border-bottom:1px solid var(--line);font-size:17px}}.detail{{padding:16px;background:#101a2c}}.step-no{{color:var(--cyan);font-size:14px;font-weight:800}}.detail-title{{margin-top:8px;font-size:20px;line-height:1.35;font-weight:800}}.detail-body{{margin-top:10px;color:#c1cede;line-height:1.65;font-size:15px}}.meta{{display:grid;gap:8px;padding:14px 16px}}.meta-row{{display:grid;grid-template-columns:64px minmax(0,1fr);gap:8px;align-items:start;font-size:14px}}.meta-row span:first-child{{color:#91a6c2}}.meta-row code{{color:#dbeafe;background:#17243a;border:1px solid #263954;border-radius:6px;padding:2px 6px;white-space:normal;word-break:break-word}}.timeline{{display:grid;gap:8px;padding:12px;max-height:290px;overflow:auto}}.timeline-row{{display:grid;grid-template-columns:34px 1fr;gap:8px;padding:8px;border-radius:8px;color:#aebed5;background:#0c1626;border:1px solid transparent;font-size:13px;line-height:1.45;cursor:pointer}}.timeline-row.current{{color:#e0f2fe;border-color:var(--cyan);background:#0d2539;box-shadow:0 0 16px rgba(56,189,248,.18)}}.timeline-row.done{{color:#dcfce7;border-color:#0f766e;background:#0b2b27}}.hint{{padding:12px 16px;border-top:1px solid var(--line);color:#9fb0c7;font-size:14px;line-height:1.6}}
@media (max-width:1220px){{body{{overflow:auto}}.topbar{{height:auto;grid-template-columns:1fr;padding:14px 16px}}.tabs,.tools{{justify-content:flex-start}}.workspace{{height:auto;grid-template-columns:1fr}}.viewport{{height:680px;flex:auto}}}}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <h1>第 {no} 章：{escape_text(title)}业务与代码执行动画</h1>
    <p>{escape_text(str(chapter["desc"]))}</p>
  </div>
  <div>
    <nav class="tabs" aria-label="章节流程视角">
      <button class="tab-btn active" data-view="business">业务执行流程图</button>
      <button class="tab-btn" data-view="code">代码执行流程图</button>
    </nav>
    <div class="tools">
      <a class="tool-btn" href="../{doc}">返回本章讲义</a>
      <a class="tool-btn" href="business-flow.html">业务总图</a>
      <button class="tool-btn primary" id="playBtn">▶ 播放</button>
      <button class="tool-btn icon" id="prevBtn" title="上一步">‹</button>
      <button class="tool-btn icon" id="nextBtn" title="下一步">›</button>
      <button class="tool-btn icon" id="zoomOutBtn" title="缩小">−</button>
      <span class="zoom-label" id="zoomLabel">100%</span>
      <button class="tool-btn icon" id="zoomInBtn" title="放大">＋</button>
      <button class="tool-btn" id="fitBtn">适配</button>
    </div>
  </div>
</header>
<main class="workspace">
  <section class="stage">
    <div class="stage-head">
      <div><div class="stage-title" id="stageTitle">第 {no} 章业务执行流程图 <span class="badge">业务闭环</span></div><div class="stage-desc" id="stageDesc">第 {no} 章功能闭环</div></div>
      <div class="progress-wrap"><div class="progress-meta"><span id="progressName">节点 1</span><span id="progressCount">1/{len(business_nodes)}</span></div><div class="progress"><span id="progressBar"></span></div></div>
    </div>
    <div class="viewport" id="viewport"><div class="canvas" id="canvas"></div><div class="loading" id="loading">正在绘制流程图...</div><div class="empty-state" id="emptyState" hidden>Mermaid 加载失败，请刷新页面。</div></div>
  </section>
  <aside class="side">
    <section class="panel"><h2>当前节点</h2><div class="detail"><div class="step-no" id="stepNo">NODE 1</div><div class="detail-title" id="stepTitle">-</div><div class="detail-body" id="stepBody">-</div></div><div class="meta"><div class="meta-row"><span>章节</span><code id="nodeChapter">第 {no} 章</code></div><div class="meta-row"><span>文件</span><code id="nodeFile">-</code></div><div class="meta-row"><span>函数</span><code id="nodeSymbol">-</code></div></div></section>
    <section class="panel"><h2>节点列表</h2><div class="timeline" id="timeline"></div><div class="hint">滚轮缩放，按住图表拖拽。点击节点列表可跳到对应代码节点。</div></section>
  </aside>
</main>
<script type="text/plain" id="diagram-business">
{business_diagram}
</script>
<script type="text/plain" id="diagram-code">
{code_diagram}
</script>
<script>
const views = {views_json};
const state = {{view:"business",step:0,scale:1,panX:0,panY:0,playing:false,timer:null,cache:{{}}}};
const el = {{
  canvas:document.getElementById("canvas"),viewport:document.getElementById("viewport"),loading:document.getElementById("loading"),empty:document.getElementById("emptyState"),stageTitle:document.getElementById("stageTitle"),stageDesc:document.getElementById("stageDesc"),progressName:document.getElementById("progressName"),progressCount:document.getElementById("progressCount"),progressBar:document.getElementById("progressBar"),stepNo:document.getElementById("stepNo"),stepTitle:document.getElementById("stepTitle"),stepBody:document.getElementById("stepBody"),nodeChapter:document.getElementById("nodeChapter"),nodeFile:document.getElementById("nodeFile"),nodeSymbol:document.getElementById("nodeSymbol"),timeline:document.getElementById("timeline"),zoomLabel:document.getElementById("zoomLabel"),playBtn:document.getElementById("playBtn")
}};
function boot(){{
  if(!window.mermaid){{el.loading.hidden=true;el.empty.hidden=false;return;}}
  mermaid.initialize({{startOnLoad:false,securityLevel:"loose",theme:"base",flowchart:{{curve:"basis",htmlLabels:true,nodeSpacing:52,rankSpacing:68,padding:18}},themeVariables:{{darkMode:true,background:"#101827",mainBkg:"#17243a",primaryColor:"#17243a",primaryTextColor:"#edf6ff",primaryBorderColor:"#38bdf8",lineColor:"#93c5fd",secondaryColor:"#132136",tertiaryColor:"#0f172a",fontFamily:"Microsoft YaHei, Segoe UI, sans-serif",fontSize:"15px",clusterBkg:"#0d1829",clusterBorder:"#334155",edgeLabelBackground:"#111b2e"}}}});
  renderView("business");
}}
function diagramSource(viewId){{return document.getElementById(`diagram-${{viewId}}`).textContent.trim();}}
async function renderView(viewId){{
  stopPlayback();
  state.view=viewId;
  state.step=0;
  const view=views[viewId];
  el.stageTitle.innerHTML=`${{view.title}} <span class="badge">${{view.badge}}</span>`;
  el.stageDesc.textContent=view.desc;
  document.querySelectorAll(".tab-btn").forEach(btn=>btn.classList.toggle("active",btn.dataset.view===viewId));
  renderTimeline();
  el.loading.hidden=false;
  el.empty.hidden=true;
  el.canvas.innerHTML="";
  try{{
    if(!state.cache[viewId]){{
      const result = await mermaid.render(`chapter-flow-diagram-${{viewId}}`, diagramSource(viewId));
      state.cache[viewId]=result.svg;
    }}
    el.canvas.innerHTML = state.cache[viewId];
    decorateSvg();
    requestAnimationFrame(() => {{fitToView();activateStep(0,false);el.loading.hidden=true;}});
  }}catch(err){{console.error(err);el.loading.hidden=true;el.empty.hidden=false;}}
}}
function decorateSvg(){{
  const svg = el.canvas.querySelector("svg"); if(!svg) return;
  const vb = svg.viewBox && svg.viewBox.baseVal; if(vb && vb.width && vb.height){{svg.setAttribute("width", String(Math.ceil(vb.width)));svg.setAttribute("height", String(Math.ceil(vb.height)));}}
  svg.style.maxWidth="none"; svg.style.overflow="visible";
  const style = document.createElementNS("http://www.w3.org/2000/svg","style");
  style.textContent = `g.node > rect,g.node > polygon,g.node > path,g.node > circle {{filter:drop-shadow(0 6px 14px rgba(0,0,0,.28));}} g.node.is-active > rect,g.node.is-active > polygon,g.node.is-active > path,g.node.is-active > circle {{stroke:#fbbf24!important;stroke-width:4px!important;filter:drop-shadow(0 0 18px rgba(251,191,36,.62));}} .edgePath path {{stroke-width:2.1px!important;}}`;
  svg.insertBefore(style, svg.firstChild);
}}
function renderTimeline(){{
  const flowNodes = views[state.view].nodes;
  el.timeline.innerHTML = flowNodes.map((node,index) => `<button class="timeline-row" data-index="${{index}}"><span class="num">${{String(index+1).padStart(2,"0")}}</span><span>${{node.title}}<br><small>${{node.symbol}}</small></span></button>`).join("");
  el.timeline.querySelectorAll(".timeline-row").forEach(row => row.addEventListener("click", () => activateStep(Number(row.dataset.index))));
}}
function getSvgBox(){{const svg=el.canvas.querySelector("svg"); if(!svg) return {{width:1000,height:700,x:0,y:0}}; const vb=svg.viewBox&&svg.viewBox.baseVal; if(vb&&vb.width&&vb.height) return {{width:vb.width,height:vb.height,x:vb.x,y:vb.y}}; const box=svg.getBBox(); return {{width:box.width,height:box.height,x:box.x,y:box.y}};}}
function fitToView(){{const box=getSvgBox(); const vw=el.viewport.clientWidth; const vh=el.viewport.clientHeight; state.scale=Math.max(.28,Math.min(1.18,Math.min((vw-64)/box.width,(vh-64)/box.height))); state.panX=(vw-box.width*state.scale)/2-box.x*state.scale; state.panY=Math.max(28,(vh-box.height*state.scale)/2)-box.y*state.scale; applyTransform();}}
function applyTransform(animate=true){{el.canvas.classList.toggle("no-animate",!animate); el.canvas.style.transform=`translate(${{state.panX}}px, ${{state.panY}}px) scale(${{state.scale}})`; el.zoomLabel.textContent=`${{Math.round(state.scale*100)}}%`;}}
function findNode(id){{const svg=el.canvas.querySelector("svg"); if(!svg) return null; const escaped=id.replace(/[.*+?^${{}}()|[\\]\\\\]/g,"\\\\$&"); const generated=new RegExp(`^flowchart-${{escaped}}-\\\\d+$`); return Array.from(svg.querySelectorAll("g.node")).find(node => node.id===id || generated.test(node.id));}}
function activateStep(index, center=true){{const flowNodes=views[state.view].nodes; state.step=(index+flowNodes.length)%flowNodes.length; const node=flowNodes[state.step]; el.canvas.querySelectorAll("g.node").forEach(item=>item.classList.remove("is-active")); const svgNode=findNode(node.id); if(svgNode){{svgNode.classList.add("is-active"); if(center) focusNode(svgNode);}} el.stepNo.textContent=`${{state.view==="business"?"BUS":"CODE"}} ${{state.step+1}}`; el.stepTitle.textContent=node.title; el.stepBody.textContent=node.body; el.nodeChapter.textContent=node.chapter; el.nodeFile.textContent=node.file; el.nodeSymbol.textContent=node.symbol; el.progressName.textContent=`节点 ${{state.step+1}}`; el.progressCount.textContent=`${{state.step+1}}/${{flowNodes.length}}`; el.progressBar.style.width=`${{((state.step+1)/flowNodes.length)*100}}%`; el.timeline.querySelectorAll(".timeline-row").forEach((row,i)=>{{row.classList.toggle("current",i===state.step); row.classList.toggle("done",i<state.step);}});}}
function getNodeSvgBox(node){{const raw=node.getBBox(); const matrix=node.getCTM(); if(!matrix) return raw; const points=[{{x:raw.x,y:raw.y}},{{x:raw.x+raw.width,y:raw.y}},{{x:raw.x,y:raw.y+raw.height}},{{x:raw.x+raw.width,y:raw.y+raw.height}}].map(point=>({{x:matrix.a*point.x+matrix.c*point.y+matrix.e,y:matrix.b*point.x+matrix.d*point.y+matrix.f}})); const xs=points.map(p=>p.x); const ys=points.map(p=>p.y); const x=Math.min(...xs); const y=Math.min(...ys); return {{x,y,width:Math.max(...xs)-x,height:Math.max(...ys)-y}};}}
function focusNode(node){{const box=getNodeSvgBox(node); const svgBox=getSvgBox(); const vw=el.viewport.clientWidth; const vh=el.viewport.clientHeight; const padding=48; const desired=Math.min(1.05,Math.max(.34,Math.min((vw-padding*2)/Math.max(box.width,1),(vh-padding*2)/Math.max(box.height,1)))); if(box.width*state.scale>vw-padding*2||box.height*state.scale>vh-padding*2||state.scale<.42) state.scale=desired; const cx=box.x+box.width/2; const cy=box.y+box.height/2; state.panX=vw/2-(cx-svgBox.x)*state.scale; state.panY=vh/2-(cy-svgBox.y)*state.scale; applyTransform();}}
function setScale(next, anchorX=el.viewport.clientWidth/2, anchorY=el.viewport.clientHeight/2){{const old=state.scale; const bounded=Math.max(.25,Math.min(2.4,next)); const worldX=(anchorX-state.panX)/old; const worldY=(anchorY-state.panY)/old; state.scale=bounded; state.panX=anchorX-worldX*bounded; state.panY=anchorY-worldY*bounded; applyTransform();}}
function startPlayback(){{state.playing=true; el.playBtn.textContent="Ⅱ 暂停"; state.timer=window.setInterval(()=>activateStep(state.step+1),1350);}}
function stopPlayback(){{state.playing=false; el.playBtn.textContent="▶ 播放"; if(state.timer) window.clearInterval(state.timer); state.timer=null;}}
document.querySelectorAll(".tab-btn").forEach(btn=>btn.addEventListener("click",()=>renderView(btn.dataset.view)));
document.getElementById("playBtn").addEventListener("click",()=>state.playing?stopPlayback():startPlayback());
document.getElementById("prevBtn").addEventListener("click",()=>activateStep(state.step-1));
document.getElementById("nextBtn").addEventListener("click",()=>activateStep(state.step+1));
document.getElementById("zoomInBtn").addEventListener("click",()=>setScale(state.scale*1.15));
document.getElementById("zoomOutBtn").addEventListener("click",()=>setScale(state.scale/1.15));
document.getElementById("fitBtn").addEventListener("click",fitToView);
window.addEventListener("resize",fitToView);
let dragging=false; let dragStart={{x:0,y:0,panX:0,panY:0}};
el.viewport.addEventListener("pointerdown",event=>{{dragging=true;dragStart={{x:event.clientX,y:event.clientY,panX:state.panX,panY:state.panY}};el.viewport.classList.add("dragging");el.viewport.setPointerCapture(event.pointerId);}});
el.viewport.addEventListener("pointermove",event=>{{if(!dragging)return;state.panX=dragStart.panX+event.clientX-dragStart.x;state.panY=dragStart.panY+event.clientY-dragStart.y;applyTransform(false);}});
el.viewport.addEventListener("pointerup",event=>{{dragging=false;el.viewport.classList.remove("dragging");el.viewport.releasePointerCapture(event.pointerId);}});
el.viewport.addEventListener("wheel",event=>{{event.preventDefault();const rect=el.viewport.getBoundingClientRect();setScale(state.scale*(event.deltaY<0?1.1:.9),event.clientX-rect.left,event.clientY-rect.top);}},{{passive:false}});
if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",boot,{{once:true}}); else boot();
</script>
</body>
</html>
"""


def animation_block(chapter: dict[str, object]) -> str:
    no = str(chapter["no"])
    title = str(chapter["title"])
    filename = str(chapter["html"])
    return (
        "## 本章动画\n\n"
        f"- [打开第 {no} 章：{title}业务与代码执行动画](./animation/{filename})\n"
        "- [打开业务流程总图](./animation/business-flow.html)\n"
    )


def sync_doc_link(chapter: dict[str, object]) -> bool:
    doc_path = DOCS_DIR / str(chapter["doc"])
    text = doc_path.read_text(encoding="utf-8")
    filename = str(chapter["html"])
    has_block = "## 本章动画" in text
    has_chapter_link = f"./animation/{filename}" in text
    has_business_flow = "./animation/business-flow.html" in text
    block = animation_block(chapter)
    expected_link_text = f"打开第 {str(chapter['no'])} 章：{str(chapter['title'])}业务与代码执行动画"
    if has_block and has_chapter_link and has_business_flow:
        if expected_link_text in text:
            return False
        text = re.sub(
            rf"打开第 {re.escape(str(chapter['no']))} 章：{re.escape(str(chapter['title']))}代码执行动画",
            expected_link_text,
            text,
            count=1,
        )
        doc_path.write_text(text, encoding="utf-8")
        return True

    target_match = re.search(r"\n## 本[讲章]目标\b", text)
    if target_match:
        index = target_match.start()
        text = text[:index] + "\n" + block + text[index:]
    else:
        first_heading = text.find("\n## ")
        index = first_heading if first_heading != -1 else len(text)
        text = text[:index] + "\n" + block + text[index:]
    doc_path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    linked = 0
    for chapter in CHAPTERS:
        out_path = ANIMATION_DIR / str(chapter["html"])
        out_path.write_text(build_html(chapter), encoding="utf-8")
        written += 1
        if sync_doc_link(chapter):
            linked += 1
    print(f"Generated {written} chapter animations; added {linked} lecture links.")


if __name__ == "__main__":
    main()
