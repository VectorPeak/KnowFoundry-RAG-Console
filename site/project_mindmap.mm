<map version="1.0.1">
<node CREATED="1716681600000" ID="root" MODIFIED="1716681600000" TEXT="企业级多场景 RAG 知识平台">
<richcontent TYPE="NOTE"><html><body><p>立项名称：Integrated QA System</p><p>定位：面向教学与企业落地验证的 RAG 全链路工程闭环项目</p><p>代码规模：~18,000 行 Python，85 个核心模块</p></body></html></richcontent>
<font NAME="Microsoft YaHei" SIZE="16" BOLD="true"/>
<node CREATED="1716681600001" ID="position" MODIFIED="1716681600001" TEXT="一、项目定位">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600002" ID="pos_target" MODIFIED="1716681600002" TEXT="目标用户">
<node CREATED="1716681600003" ID="pos_senior" MODIFIED="1716681600003" TEXT="有 1-2 年 Python 经验的进阶学习者"/>
<node CREATED="1716681600004" ID="pos_interview" MODIFIED="1716681600004" TEXT="准备 AI/LLM 方向面试的候选人"/>
<node CREATED="1716681600005" ID="pos_teacher" MODIFIED="1716681600005" TEXT="RAG 技术培训讲师/企业内训"/>
<node CREATED="1716681600006" ID="pos_enterprise" MODIFIED="1716681600006" TEXT="企业 RAG 系统选型 POC 参考"/>
</node>
<node CREATED="1716681600007" ID="pos_diff" MODIFIED="1716681600007" TEXT="差异化价值">
<node CREATED="1716681600008" ID="diff_not_demo" MODIFIED="1716681600008" TEXT="不是 Demo：是完整工程闭环"/>
<node CREATED="1716681600009" ID="diff_not_toy" MODIFIED="1716681600009" TEXT="不是玩具：8 个企业级业务场景"/>
<node CREATED="1716681600010" ID="diff_not_blackbox" MODIFIED="1716681600010" TEXT="不是黑盒：18 讲系统课程全覆盖"/>
<node CREATED="1716681600011" ID="diff_real_enterprise" MODIFIED="1716681600011" TEXT="企业真实感：版本管理+数据隔离+入库质量检查"/>
</node>
<node CREATED="1716681600012" ID="pos_comparison" MODIFIED="1716681600012" TEXT="与常见 RAG Demo 的区别">
<node CREATED="1716681600013" ID="comp_version" MODIFIED="1716681600013" TEXT="Demo: 无版本管理 → 本项目: STAGED→ACTIVE→ARCHIVED 版本状态机"/>
<node CREATED="1716681600014" ID="comp_isolation" MODIFIED="1716681600014" TEXT="Demo: 无数据隔离 → 本项目: 四维隔离 tenant/dataset/visibility/role"/>
<node CREATED="1716681600015" ID="comp_quality" MODIFIED="1716681600015" TEXT="Demo: 无质量保障 → 本项目: 入库质量+检索评测+Bad Case 闭环"/>
<node CREATED="1716681600016" ID="comp_eval" MODIFIED="1716681600016" TEXT="Demo: 凭感觉调优 → 本项目: Recall@K/MRR/关键词覆盖 量化评测"/>
<node CREATED="1716681600017" ID="comp_trace" MODIFIED="1716681600017" TEXT="Demo: 黑盒问答 → 本项目: LangSmith Trace 全链路可观测"/>
</node>
</node>
<node CREATED="1716681600018" ID="architecture" MODIFIED="1716681600018" TEXT="二、技术架构">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600019" ID="arch_overview" MODIFIED="1716681600019" TEXT="分层架构（四层）">
<node CREATED="1716681600020" ID="arch_api" MODIFIED="1716681600020" TEXT="API 层 (8 文件, 718 行)">
<node CREATED="1716681600021" ID="api_http" MODIFIED="1716681600021" TEXT="HTTP POST /api/query — 同步预检"/>
<node CREATED="1716681600022" ID="api_ws" MODIFIED="1716681600022" TEXT="WS /api/stream — 流式问答"/>
<node CREATED="1716681600023" ID="api_debug" MODIFIED="1716681600023" TEXT="POST /api/retrieval/debug — 检索诊断"/>
<node CREATED="1716681600024" ID="api_feedback" MODIFIED="1716681600024" TEXT="POST /api/feedback — 用户反馈"/>
<node CREATED="1716681600025" ID="api_scenarios" MODIFIED="1716681600025" TEXT="GET /api/scenarios — 场景列表"/>
</node>
<node CREATED="1716681600026" ID="arch_app" MODIFIED="1716681600026" TEXT="应用层 (4 文件, 305 行)">
<node CREATED="1716681600027" ID="app_service" MODIFIED="1716681600027" TEXT="QAService 双路径编排">
<node CREATED="1716681600028" ID="app_path_a" MODIFIED="1716681600028" TEXT="路径 A: preview_query() 同步直答（问候/越界）"/>
<node CREATED="1716681600029" ID="app_path_b" MODIFIED="1716681600029" TEXT="路径 B: stream_query() 流式 RAG（复杂问题）"/>
</node>
</node>
<node CREATED="1716681600030" ID="arch_pipeline" MODIFIED="1716681600030" TEXT="管线层 (10 文件, 1,477 行)">
<node CREATED="1716681600031" ID="pipe_rag" MODIFIED="1716681600031" TEXT="rag.py — 七阶段主流程编排"/>
<node CREATED="1716681600032" ID="pipe_steps" MODIFIED="1716681600032" TEXT="steps.py — 意图/改写/答案组装"/>
<node CREATED="1716681600033" ID="pipe_retrieval" MODIFIED="1716681600033" TEXT="retrieval_steps.py — FAQ/文档检索执行"/>
<node CREATED="1716681600034" ID="pipe_context" MODIFIED="1716681600034" TEXT="context.py — 上下文筛选/去重/截断"/>
<node CREATED="1716681600035" ID="pipe_cite" MODIFIED="1716681600035" TEXT="citations.py — 答案引用增强"/>
</node>
<node CREATED="1716681600036" ID="arch_infra" MODIFIED="1716681600036" TEXT="基础设施层">
<node CREATED="1716681600037" ID="infra_retrieval" MODIFIED="1716681600037" TEXT="检索: Milvus 2.6 Hybrid Search (Dense + BM25 Sparse)"/>
<node CREATED="1716681600038" ID="infra_embedding" MODIFIED="1716681600038" TEXT="嵌入: BGE-M3 (本地 GPU/CPU)"/>
<node CREATED="1716681600039" ID="infra_reranker" MODIFIED="1716681600039" TEXT="重排: BGE Reranker Large (CrossEncoder)"/>
<node CREATED="1716681600040" ID="infra_llm" MODIFIED="1716681600040" TEXT="LLM: 通义千问 qwen-plus (DashScope)"/>
<node CREATED="1716681600041" ID="infra_mysql" MODIFIED="1716681600041" TEXT="存储: MySQL 8.4 (历史+反馈+摘要)"/>
<node CREATED="1716681600042" ID="infra_etcd" MODIFIED="1716681600042" TEXT="元数据: etcd (Milvus 分布式协调)"/>
<node CREATED="1716681600043" ID="infra_minio" MODIFIED="1716681600043" TEXT="对象: MinIO (Milvus 数据持久化)"/>
</node>
</node>
<node CREATED="1716681600044" ID="arch_flow" MODIFIED="1716681600044" TEXT="一次完整问答数据流">
<node CREATED="1716681600045" ID="flow_1" MODIFIED="1716681600045" TEXT="① 浏览器提问 → HTTP POST /api/query"/>
<node CREATED="1716681600046" ID="flow_2" MODIFIED="1716681600046" TEXT="② QAService.preview_query() 预检"/>
<node CREATED="1716681600047" ID="flow_3" MODIFIED="1716681600047" TEXT="③ 意图识别: 规则优先(0ms) + LLM 兜底"/>
<node CREATED="1716681600048" ID="flow_4" MODIFIED="1716681600048" TEXT="④ 简单问题(问候/越界) → 直接返回"/>
<node CREATED="1716681600049" ID="flow_5" MODIFIED="1716681600049" TEXT="⑤ 复杂问题 → 前端开 WebSocket"/>
<node CREATED="1716681600050" ID="flow_6" MODIFIED="1716681600050" TEXT="⑥ WS /api/stream → QAService.stream_query()"/>
<node CREATED="1716681600051" ID="flow_7" MODIFIED="1716681600051" TEXT="⑦ 查询改写 → FAQ 检索 → 置信度判断"/>
<node CREATED="1716681600052" ID="flow_8" MODIFIED="1716681600052" TEXT="⑧ 文档检索(Milvus Hybrid) → CrossEncoder 重排"/>
<node CREATED="1716681600053" ID="flow_9" MODIFIED="1716681600053" TEXT="⑨ Prompt Profile 选择 → LLM 流式生成"/>
<node CREATED="1716681600054" ID="flow_10" MODIFIED="1716681600054" TEXT="⑩ 逐 token WebSocket 推送 + 引用来源"/>
<node CREATED="1716681600055" ID="flow_11" MODIFIED="1716681600055" TEXT="⑪ 写历史(MySQL) + 写 Trace(LangSmith)"/>
</node>
</node>
<node CREATED="1716681600056" ID="p0_main" MODIFIED="1716681600056" TEXT="三、P0 核心 RAG 主链路（必学）">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600057" ID="p0_intent" MODIFIED="1716681600057" TEXT="意图识别 (qa_core/intent/)">
<node CREATED="1716681600058" ID="intent_6steps" MODIFIED="1716681600058" TEXT="6 步决策: 问候→越界→转人工→追问→强规则→LLM兜底"/>
<node CREATED="1716681600059" ID="intent_why" MODIFIED="1716681600059" TEXT="核心决策: 规则优先(0ms零成本) + LLM 补充(覆盖长尾)"/>
</node>
<node CREATED="1716681600060" ID="p0_strategy" MODIFIED="1716681600060" TEXT="检索策略 (qa_core/retrieval/strategy.py)">
<node CREATED="1716681600061" ID="strat_5layers" MODIFIED="1716681600061" TEXT="5 层决策链: 意图→短问题保护→风险类别→表格偏好"/>
<node CREATED="1716681600062" ID="strat_dynamic" MODIFIED="1716681600062" TEXT="动态参数: top_k/阈值/rerank 按问题类型自适应"/>
</node>
<node CREATED="1716681600063" ID="p0_rewrite" MODIFIED="1716681600063" TEXT="查询改写与变体 (qa_core/pipeline/)">
<node CREATED="1716681600064" ID="rewrite_coref" MODIFIED="1716681600064" TEXT="追问改写: 「审批呢」 → 「入职审批流程需要多长时间」"/>
<node CREATED="1716681600065" ID="rewrite_variants" MODIFIED="1716681600065" TEXT="多路变体: 启发式+LLM 生成同义检索表达"/>
</node>
<node CREATED="1716681600066" ID="p0_milvus" MODIFIED="1716681600066" TEXT="Milvus 混合检索 (qa_core/retrieval/store.py)">
<node CREATED="1716681600067" ID="milvus_dense" MODIFIED="1716681600067" TEXT="Dense: BGE-M3 语义向量 (权重 0.55)"/>
<node CREATED="1716681600068" ID="milvus_sparse" MODIFIED="1716681600068" TEXT="Sparse: Milvus 内置 BM25 关键词 (权重 0.45)"/>
<node CREATED="1716681600069" ID="milvus_rerank" MODIFIED="1716681600069" TEXT="二阶段重排: CrossEncoder 精细打分（精度 +10~15%）"/>
<node CREATED="1716681600070" ID="milvus_filter" MODIFIED="1716681600070" TEXT="过滤: source/kb_version/tenant/dataset/visibility/role"/>
</node>
<node CREATED="1716681600071" ID="p0_pipeline" MODIFIED="1716681600071" TEXT="七阶段 RAG 管线 (qa_core/pipeline/rag.py)">
<node CREATED="1716681600072" ID="pipe_s0" MODIFIED="1716681600072" TEXT="Stage 0: 创建运行时上下文（场景/数据域/会话/trace）"/>
<node CREATED="1716681600073" ID="pipe_s1" MODIFIED="1716681600073" TEXT="Stage 1: FAQ 精确匹配快速直出（跳过后续步骤）"/>
<node CREATED="1716681600074" ID="pipe_s2" MODIFIED="1716681600074" TEXT="Stage 2: 意图识别 + 改写 + 检索计划 + 查询变体"/>
<node CREATED="1716681600075" ID="pipe_s3" MODIFIED="1716681600075" TEXT="Stage 3: 非 RAG 类回答直接返回（问候/越界/转人工）"/>
<node CREATED="1716681600076" ID="pipe_s4" MODIFIED="1716681600076" TEXT="Stage 4: FAQ 检索 + 置信度判断（分高直出跳过 LLM）"/>
<node CREATED="1716681600077" ID="pipe_s5" MODIFIED="1716681600077" TEXT="Stage 5: 文档检索 + 上下文构建（筛选/去重/截断）"/>
<node CREATED="1716681600078" ID="pipe_s6" MODIFIED="1716681600078" TEXT="Stage 6: LLM 流式生成（逐 token WebSocket 推送）"/>
<node CREATED="1716681600079" ID="pipe_s7" MODIFIED="1716681600079" TEXT="Stage 7: 写历史(MySQL) + 写 Trace(LangSmith) + 发 end"/>
</node>
<node CREATED="1716681600080" ID="p0_prompt" MODIFIED="1716681600080" TEXT="Prompt Profile 系统 (qa_core/prompts/)">
<node CREATED="1716681600081" ID="prompt_8profiles" MODIFIED="1716681600081" TEXT="8 种 Profile: FAQ/知识/追问 + 费用/合规/排障/总结 + 默认"/>
<node CREATED="1716681600082" ID="prompt_risk" MODIFIED="1716681600082" TEXT="高风险问题(费用/合规): 更严格的确认边界和风险提示"/>
</node>
</node>
<node CREATED="1716681600083" ID="p1_eng" MODIFIED="1716681600083" TEXT="四、P1 企业工程能力（核心）">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600084" ID="p1_version" MODIFIED="1716681600084" TEXT="知识库版本管理">
<node CREATED="1716681600085" ID="ver_fsm" MODIFIED="1716681600085" TEXT="版本状态机: STAGED → ACTIVE → ARCHIVED"/>
<node CREATED="1716681600086" ID="ver_ops" MODIFIED="1716681600086" TEXT="操作: 创建/激活/回滚/对比/归档"/>
<node CREATED="1716681600087" ID="ver_why" MODIFIED="1716681600087" TEXT="核心价值: 更新知识库不影响线上 ACTIVE 版本"/>
</node>
<node CREATED="1716681600088" ID="p1_isolation" MODIFIED="1716681600088" TEXT="多租户数据隔离">
<node CREATED="1716681600089" ID="iso_4dim" MODIFIED="1716681600089" TEXT="四维隔离: tenant_id / dataset_id / visibility / allowed_roles"/>
<node CREATED="1716681600090" ID="iso_levels" MODIFIED="1716681600090" TEXT="可见级别: public（公开）/ internal（内部）/ private（私密）"/>
<node CREATED="1716681600091" ID="iso_expr" MODIFIED="1716681600091" TEXT="落地方式: 写入 Milvus metadata → 检索时拼接 expr 过滤"/>
</node>
<node CREATED="1716681600092" ID="p1_ingestion" MODIFIED="1716681600092" TEXT="文档入库链路">
<node CREATED="1716681600093" ID="ingest_loaders" MODIFIED="1716681600093" TEXT="Loader 注册表: 按扩展名路由（.md/.txt/.csv/.xlsx/.pdf）"/>
<node CREATED="1716681600094" ID="ingest_chunking" MODIFIED="1716681600094" TEXT="Parent-Child Chunking: 父块 1000 字符 + 子块 350 字符"/>
<node CREATED="1716681600095" ID="ingest_manifest" MODIFIED="1716681600095" TEXT="IndexManifest: SHA256 指纹增量入库（只更新变化文件）"/>
<node CREATED="1716681600096" ID="ingest_table" MODIFIED="1716681600096" TEXT="表格入库: CSV/Excel 每行一个 chunk（含 table_row 标记）"/>
</node>
<node CREATED="1716681600097" ID="p1_quality" MODIFIED="1716681600097" TEXT="RAG 回归与入库质量体系">
<node CREATED="1716681600098" ID="qual_layers" MODIFIED="1716681600098" TEXT="三层保障: 入库质量 → 检索评测 → 性能基线"/>
<node CREATED="1716681600099" ID="qual_gates" MODIFIED="1716681600099" TEXT="验收机制: 低质量chunk/FAQ冲突/正文冲突 自动检测"/>
<node CREATED="1716681600100" ID="qual_metrics" MODIFIED="1716681600100" TEXT="评测指标: Recall@K / MRR / 关键词覆盖 / 场景隔离率"/>
<node CREATED="1716681600101" ID="qual_badcase" MODIFIED="1716681600101" TEXT="Bad Case 闭环: 自动识别→人工复核→晋级至评测集→回回验"/>
</node>
<node CREATED="1716681600102" ID="p1_test" MODIFIED="1716681600102" TEXT="测试与接口验收">
<node CREATED="1716681600103" ID="test_pyramid" MODIFIED="1716681600103" TEXT="测试金字塔: 纯逻辑测试 → API 保护测试 → E2E 测试"/>
<node CREATED="1716681600104" ID="test_pytest" MODIFIED="1716681600104" TEXT="pytest 用例: 意图/检索/过滤/Prompt/验收逻辑"/>
</node>
</node>
<node CREATED="1716681600105" ID="p2_enterprise" MODIFIED="1716681600105" TEXT="五、P2 企业增强（面试加分）">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600106" ID="p2_scenarios" MODIFIED="1716681600106" TEXT="8 大业务场景">
<node CREATED="1716681600107" ID="s1" MODIFIED="1716681600107" TEXT="enterprise_knowledge — HR/IT/财务内部知识"/>
<node CREATED="1716681600108" ID="s2" MODIFIED="1716681600108" TEXT="saas_support — 账号/计费/集成客服支持"/>
<node CREATED="1716681600109" ID="s3" MODIFIED="1716681600109" TEXT="equipment_ops — 设备巡检/告警/安全操作"/>
<node CREATED="1716681600110" ID="s4" MODIFIED="1716681600110" TEXT="compliance_qa — 合同/审计/隐私合规咨询"/>
<node CREATED="1716681600111" ID="s5" MODIFIED="1716681600111" TEXT="cross_border_risk — 跨境贸易风险管理 ⭐面试亮点"/>
<node CREATED="1716681600112" ID="s6" MODIFIED="1716681600112" TEXT="tender_contract_risk — 招投标合同风险管理 ⭐面试亮点"/>
<node CREATED="1716681600113" ID="s7" MODIFIED="1716681600113" TEXT="insurance_claims — 保险理赔核保 ⭐面试亮点"/>
<node CREATED="1716681600114" ID="s8" MODIFIED="1716681600114" TEXT="engineering_project_qa — 工程施工规范问答 ⭐面试亮点"/>
</node>
<node CREATED="1716681600115" ID="p2_trace" MODIFIED="1716681600115" TEXT="全链路可观测（LangSmith）">
<node CREATED="1716681600116" ID="trace_trace" MODIFIED="1716681600116" TEXT="Trace: 每次问答完整调用链（意图/检索/生成各阶段耗时）"/>
<node CREATED="1716681600117" ID="trace_dataset" MODIFIED="1716681600117" TEXT="Dataset: 评测数据集管理 + 回归测试"/>
<node CREATED="1716681600118" ID="trace_eval" MODIFIED="1716681600118" TEXT="Evaluation: 在线评估 + 离线批量评估"/>
</node>
<node CREATED="1716681600119" ID="p2_boundary" MODIFIED="1716681600119" TEXT="场景边界检测">
<node CREATED="1716681600120" ID="bound_scenario" MODIFIED="1716681600120" TEXT="场景边界: 防止跨场景问答（如选保险问工程问题）"/>
<node CREATED="1716681600121" ID="bound_source" MODIFIED="1716681600121" TEXT="分类边界: 前端 source 与问题实际领域不匹配时提示切换"/>
</node>
</node>
<node CREATED="1716681600122" ID="tech_stack" MODIFIED="1716681600122" TEXT="六、技术栈全景">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600123" ID="ts_lang" MODIFIED="1716681600123" TEXT="语言: Python 3.12"/>
<node CREATED="1716681600124" ID="ts_web" MODIFIED="1716681600124" TEXT="Web 框架: FastAPI + WebSocket + uvicorn"/>
<node CREATED="1716681600125" ID="ts_rag" MODIFIED="1716681600125" TEXT="RAG 编排: LangChain (ChatOpenAI + VectorStore + Loader + Splitter)"/>
<node CREATED="1716681600126" ID="ts_vectordb" MODIFIED="1716681600126" TEXT="向量数据库: Milvus 2.6 (Hybrid Search + 服务端 BM25)"/>
<node CREATED="1716681600127" ID="ts_ml" MODIFIED="1716681600127" TEXT="模型部署: BGE-M3(Embedding) + BGE Reranker Large(CrossEncoder) + BERT(意图分类)"/>
<node CREATED="1716681600128" ID="ts_llm" MODIFIED="1716681600128" TEXT="LLM 服务: 通义千问 qwen-plus (DashScope OpenAI 兼容 API)"/>
<node CREATED="1716681600129" ID="ts_db" MODIFIED="1716681600129" TEXT="关系数据库: MySQL 8.4 (聊天历史 + 用户反馈 + 会话摘要)"/>
<node CREATED="1716681600130" ID="ts_infra" MODIFIED="1716681600130" TEXT="基础设施: Docker + Docker Compose (MySQL + etcd + MinIO + Milvus)"/>
<node CREATED="1716681600131" ID="ts_ocr" MODIFIED="1716681600131" TEXT="文档处理: PaddleOCR + PyMuPDF (PDF/图片 OCR)"/>
<node CREATED="1716681600132" ID="ts_eval" MODIFIED="1716681600132" TEXT="评测框架: RAGAS 0.3.6 (Retrieval + Generation 指标)"/>
<node CREATED="1716681600133" ID="ts_observ" MODIFIED="1716681600133" TEXT="可观测性: LangSmith (Tracing + Dataset + Evaluation)"/>
<node CREATED="1716681600134" ID="ts_doc" MODIFIED="1716681600134" TEXT="文档站点: MkDocs Material + Mermaid + MathJax"/>
<node CREATED="1716681600135" ID="ts_config" MODIFIED="1716681600135" TEXT="配置管理: Pydantic Settings + .env + TOML 场景配置"/>
</node>
<node CREATED="1716681600136" ID="code_structure" MODIFIED="1716681600136" TEXT="七、代码结构">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600137" ID="cs_root" MODIFIED="1716681600137" TEXT="app.py (74行) — FastAPI 应用入口"/>
<node CREATED="1716681600138" ID="cs_core" MODIFIED="1716681600138" TEXT="qa_core/ (8,841行, 85文件)">
<node CREATED="1716681600139" ID="cs_pipeline" MODIFIED="1716681600139" TEXT="pipeline/ (1,477行) — RAG 管线编排"/>
<node CREATED="1716681600140" ID="cs_observ" MODIFIED="1716681600140" TEXT="observability/ (1,211行) — LangSmith 追踪+评测+Bad Case"/>
<node CREATED="1716681600141" ID="cs_indexing" MODIFIED="1716681600141" TEXT="indexing/ (1,021行) — 文档加载+切分+入库"/>
<node CREATED="1716681600142" ID="cs_retrieval" MODIFIED="1716681600142" TEXT="retrieval/ (923行) — Milvus 混合检索+重排"/>
<node CREATED="1716681600143" ID="cs_api" MODIFIED="1716681600143" TEXT="api/ (718行) — FastAPI 路由+WebSocket"/>
<node CREATED="1716681600144" ID="cs_gov" MODIFIED="1716681600144" TEXT="governance/ (629行) — 版本管理+数据隔离"/>
<node CREATED="1716681600145" ID="cs_quality" MODIFIED="1716681600145" TEXT="quality/ (575行) — 入库质量检查"/>
<node CREATED="1716681600146" ID="cs_scenarios" MODIFIED="1716681600146" TEXT="scenarios/ (475行) — 场景定义+TOML加载+边界检测"/>
<node CREATED="1716681600147" ID="cs_memory" MODIFIED="1716681600147" TEXT="memory/ (410行) — MySQL 历史+摘要+反馈"/>
<node CREATED="1716681600148" ID="cs_prompts" MODIFIED="1716681600148" TEXT="prompts/ (346行) — 8种 Prompt Profile"/>
<node CREATED="1716681600149" ID="cs_other" MODIFIED="1716681600149" TEXT="application/ + intent/ + config/ + llm/ (~1,000行)"/>
</node>
<node CREATED="1716681600150" ID="cs_scripts" MODIFIED="1716681600150" TEXT="scripts/ (7,769行, 49文件) — 运维脚本"/>
<node CREATED="1716681600151" ID="cs_tests" MODIFIED="1716681600151" TEXT="tests/ (1,749行, 4文件) — 104 个测试用例"/>
<node CREATED="1716681600152" ID="cs_docs" MODIFIED="1716681600152" TEXT="docs/ — 18讲课程 + 22篇文档 + 9个技术附录"/>
<node CREATED="1716681600153" ID="cs_scenarios_data" MODIFIED="1716681600153" TEXT="scenarios/ — 8 个场景 TOML + CSV + Markdown 数据"/>
</node>
<node CREATED="1716681600154" ID="teaching" MODIFIED="1716681600154" TEXT="八、教学体系（18讲 P0→P1→P2）">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600155" ID="teach_phase1" MODIFIED="1716681600155" TEXT="第一阶段: 基础概念（P0, 2讲）">
<node CREATED="1716681600156" ID="t01" MODIFIED="1716681600156" TEXT="01 项目概述与环境搭建"/>
<node CREATED="1716681600157" ID="t02" MODIFIED="1716681600157" TEXT="02 RAG 核心概念深入"/>
</node>
<node CREATED="1716681600158" ID="teach_phase2" MODIFIED="1716681600158" TEXT="第二阶段: 核心 RAG 链路（P0, 7讲）★核心">
<node CREATED="1716681600159" ID="t03" MODIFIED="1716681600159" TEXT="03 意图分类 — 规则优先+LLM 补充"/>
<node CREATED="1716681600160" ID="t04" MODIFIED="1716681600160" TEXT="04 检索策略与动态计划 — 5层决策链"/>
<node CREATED="1716681600161" ID="t05" MODIFIED="1716681600161" TEXT="05 查询改写与变体生成"/>
<node CREATED="1716681600162" ID="t06" MODIFIED="1716681600162" TEXT="06 Milvus 混合检索 — Dense+BM25 Sparse"/>
<node CREATED="1716681600163" ID="t07" MODIFIED="1716681600163" TEXT="07 QAService 核心编排 — 双路径（预检+流式）"/>
<node CREATED="1716681600164" ID="t08" MODIFIED="1716681600164" TEXT="08 RAG Pipeline 主流程 — 七阶段管线"/>
<node CREATED="1716681600165" ID="t09" MODIFIED="1716681600165" TEXT="09 Prompt 工程与 Profile 系统 — 8种Profile"/>
</node>
<node CREATED="1716681600166" ID="teach_phase3" MODIFIED="1716681600166" TEXT="第三阶段: 入口与框架生态（P1, 2讲）">
<node CREATED="1716681600168" ID="t10" MODIFIED="1716681600168" TEXT="10 应用入口与环境前置校验"/>
<node CREATED="1716681600169" ID="t11" MODIFIED="1716681600169" TEXT="11 LangChain 生态系统"/>
</node>
<node CREATED="1716681600170" ID="teach_phase4" MODIFIED="1716681600170" TEXT="第四阶段: 治理运维（P1-P2, 6讲）">
<node CREATED="1716681600171" ID="t12" MODIFIED="1716681600171" TEXT="12 知识库版本管理"/>
<node CREATED="1716681600172" ID="t13" MODIFIED="1716681600172" TEXT="13 数据隔离与多租户"/>
<node CREATED="1716681600173" ID="t14" MODIFIED="1716681600173" TEXT="14 文档入库与索引链路"/>
<node CREATED="1716681600174" ID="t15" MODIFIED="1716681600174" TEXT="15 RAG 回归验收与入库质量"/>
<node CREATED="1716681600175" ID="t16" MODIFIED="1716681600175" TEXT="16 测试与接口验收"/>
<node CREATED="1716681600176" ID="t17" MODIFIED="1716681600176" TEXT="17 LangSmith 观测与 Trace"/>
</node>
<node CREATED="1716681600167" ID="teach_phase5" MODIFIED="1716681600167" TEXT="第五阶段: Web 框架收束（P1, 1讲）">
<node CREATED="1716681600167" ID="t18" MODIFIED="1716681600167" TEXT="18 FastAPI 与异步 Web 框架"/>
</node>
</node>
<node CREATED="1716681600177" ID="highlights" MODIFIED="1716681600177" TEXT="九、项目亮点与核心竞争力">
<font NAME="Microsoft YaHei" SIZE="14" BOLD="true"/>
<node CREATED="1716681600178" ID="hl_eng" MODIFIED="1716681600178" TEXT="工程完整度">
<node CREATED="1716681600179" ID="hl_closed_loop" MODIFIED="1716681600179" TEXT="从文档入库到在线问答到RAG 回归验收的完整闭环"/>
<node CREATED="1716681600180" ID="hl_not_demo" MODIFIED="1716681600180" TEXT="不是「调个 API 就完」的 Demo，包含版本/隔离/回归验收/评测"/>
</node>
<node CREATED="1716681600181" ID="hl_retrieval" MODIFIED="1716681600181" TEXT="检索先进性">
<node CREATED="1716681600182" ID="hl_hybrid" MODIFIED="1716681600182" TEXT="Dense + BM25 Sparse 混合检索 + CrossEncoder 二阶段重排"/>
<node CREATED="1716681600183" ID="hl_strategy" MODIFIED="1716681600183" TEXT="5 层动态检索策略，不同问题自适应参数"/>
</node>
<node CREATED="1716681600184" ID="hl_risk" MODIFIED="1716681600184" TEXT="风险控制">
<node CREATED="1716681600185" ID="hl_prompt_route" MODIFIED="1716681600185" TEXT="费用/合规类问题自动切换严格 Prompt Profile"/>
<node CREATED="1716681600186" ID="hl_scenario_boundary" MODIFIED="1716681600186" TEXT="场景/Source 边界检测，防止跨域知识污染"/>
</node>
<node CREATED="1716681600187" ID="hl_teaching" MODIFIED="1716681600187" TEXT="教学体系">
<node CREATED="1716681600188" ID="hl_18lectures" MODIFIED="1716681600188" TEXT="18 讲系统课程 + 9 个技术附录 + 18 张复习笔记卡"/>
<node CREATED="1716681600189" ID="hl_p0p1p2" MODIFIED="1716681600189" TEXT="P0→P1→P2 递进设计，4 条学习路径适应不同人群"/>
</node>
<node CREATED="1716681600190" ID="hl_interview" MODIFIED="1716681600190" TEXT="面试表达">
<node CREATED="1716681600191" ID="hl_8scenarios" MODIFIED="1716681600191" TEXT="8 个业务场景可灵活包装为不同行业背景的简历项目"/>
<node CREATED="1716681600192" ID="hl_interview_material" MODIFIED="1716681600192" TEXT="配套面试讲解手册 + 20 个高频面试问答 + 简历包装指南"/>
</node>
</node>
</node>
</map>
