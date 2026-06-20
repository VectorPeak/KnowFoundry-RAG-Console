# 第3讲 笔记：LangChain 生态系统

## 一、学习目标（可验证）

- [ ] 能说出 LangChain 的三大核心抽象（Model/Chain/Agent）及其在本项目中的使用策略
- [ ] 能解释 Runnable 接口的三个核心方法（invoke/stream/batch）及各自适用场景
- [ ] 能写出 LCEL 管道符 `|` 串联 prompt → model → parser 的代码，并解释本项目为何不用
- [ ] 能区分 ChatOpenAI 的三种调用模式：invoke（非流式）、stream（流式）、with_structured_output（Pydantic 约束）
- [ ] 能说出 SystemMessage / HumanMessage / AIMessage 三种消息类型的作用及与 OpenAI API role 的映射关系
- [ ] 能解释 ChatPromptTemplate + MessagesPlaceholder 如何构建带动态对话历史的 Prompt
- [ ] 能说明 Document Loader 注册表模式的设计思路和扩展方式
- [ ] 能解释 RecursiveCharacterTextSplitter 递归降级切分算法和父子块策略的价值
- [ ] 能回答面试题：LangChain 的 Runnable 统一接口解决了什么问题？with_structured_output 相比手写正则有什么优势？

## 二、知识点详解 + 可执行代码

### 2.1 LangChain 核心定位与三大抽象

**是什么**：LangChain 不是模型，而是一个 LLM 应用开发框架。它将对话历史、文档加载、文本切分、向量存储、提示模板、输出解析等环节做标准化封装。三大核心抽象按灵活性从低到高：

```
Model（模型） → Chain（链） → Agent（智能体）
统一调用接口    管道串联组件    LLM 自主决策下一步
```

**为什么需要**：直接用 OpenAI SDK 等于自己买食材做菜——灵活但每道菜都要从头做起。用 LangChain 等于用半成品料理包——大部分工序已做好，你只需组合和调参。

**本项目使用策略**：
- ✅ 全面使用 Model 层：ChatOpenAI 封装 DashScope
- ⚠️ 部分使用 Chain 底层组件：PromptTemplate、OutputParser 各自独立调用，不用管道符串联
- 🔮 二期规划 Agent 层：LangGraph 动态工具调用

### 2.2 Runnable 统一接口

**是什么**：LangChain 所有组件（ChatOpenAI、VectorStore、PromptTemplate、OutputParser）都实现相同的 Runnable 接口：

```python
# 三大核心方法
class Runnable:
    def invoke(self, input) -> Output:
        """单个输入 → 单个输出（同步，等完整结果）"""
        ...

    def stream(self, input) -> Iterator[Output]:
        """单个输入 → 流式输出（逐个 token）"""
        ...

    def batch(self, inputs: list) -> list[Output]:
        """多个输入 → 多个输出（自动并行处理）"""
        ...
```

**为什么需要**：LangChain 早期不同类型的组件有不同的调用方式——`llm.predict()`、`chain.run()`、`retriever.get_relevant_documents()`。换组件就要换 API，无法串成链，测试困难。Runnable 统一了这一切。

**怎么用**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="qwen-plus")

# invoke：等完整结果（用于意图识别、查询改写等需要完整结果继续下一步的场景）
response = model.invoke([HumanMessage("入职流程有哪些步骤")])
print(response.content)  # "入职流程包括：1. 提交材料..."

# stream：逐个 token 返回（用于前端实时渲染）
for chunk in model.stream([HumanMessage("入职流程有哪些步骤")]):
    print(chunk.content, end="", flush=True)  # 50ms 后 "入", 100ms 后 "职", ...

# batch：并行处理多个问题（用于批量评估、测试）
responses = model.batch([
    [HumanMessage("入职流程")],
    [HumanMessage("报销规则")],
    [HumanMessage("VPN 设置")],
])  # → [AIMessage, AIMessage, AIMessage]
```

### 2.3 LCEL：管道符 `|` 串联 Runnable

**是什么**：LCEL（LangChain Expression Language）用 `|` 管道符将多个 Runnable 串联成链，数据自动从上一个流入下一个。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话回答：{question}")
model = ChatOpenAI(model="qwen-plus")
parser = StrOutputParser()

# 管道符串联：三个 Runnable 变成一条链
chain = prompt | model | parser
# 数据流：prompt.invoke(input) → model.invoke(prompt_output) → parser.invoke(model_output)

result = chain.invoke({"question": "入职流程有哪些步骤"})
# result: "入职流程包括提交材料、签订合同、部门审批等步骤。"
```

**为什么本项目不用**：本项目刻意不用 `|` 管道符串联 RAG 流程，四个原因：
1. **分支逻辑复杂**：RAG 有 FAQ 直出、文档 RAG、信息不足、直接答案等多种分支，不适合线性管道
2. **可解释性**：自己编排的每一步都有 `reason` 字段，管道符是黑盒
3. **调试友好**：每个阶段的输入输出可单独检查和记录
4. **控制需求**：RAG 需要在阶段之间做动态决策（如 FAQ 命中直接返回、检索不足追加搜索）

```python
# ✅ 本项目自己编排流程（支持多路分支和提前退出）
def stream_query(...):
    intent = classify_intent(...)      # 可能直接结束（问候/越界）
    plan = build_retrieval_plan(...)   # 动态参数
    faq_result = search_faq(...)       # 可能提前结束（FAQ 直出）
    doc_result = search_doc(...)       # 可能信息不足
    for chunk in model.stream(...):    # 逐 token 流式推送
        yield chunk
```

### 2.4 ChatOpenAI 三种调用模式

**是什么**：本项目使用 LangChain 的 `ChatOpenAI` 封装阿里云 DashScope（OpenAI 兼容接口）。三种模式覆盖不同场景：

| 模式 | 方法 | 适用场景 | 本项目用途 |
|------|------|---------|-----------|
| 非流式 | `invoke()` | 需要完整结果才能继续下一步 | 意图识别、查询改写、历史摘要 |
| 流式 | `stream()` | 前端实时渲染 | 最终答案生成（WebSocket 逐 token 推送） |
| 结构化输出 | `with_structured_output()` | 需要 LLM 严格按字段返回 | 意图分类（Pydantic 模型校验） |

**为什么需要 OpenAI 兼容接口**：切换 LLM 只需改 `.env` 三个变量，代码零修改。DashScope → DeepSeek → Claude，换模型不动代码。

**怎么用**：

```python
# qa_core/llm/client.py — 工厂函数（进程级缓存流式/非流式各一个实例）
from functools import lru_cache
from langchain_openai import ChatOpenAI

@lru_cache(maxsize=2)  # streaming=True 和 streaming=False 各缓一份
def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,              # "qwen-plus"
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        streaming=streaming,
    )
```

> 完整源码：`qa_core/llm/client.py` → `get_chat_model()`

**结构化输出（重点）**：

```python
from pydantic import BaseModel, Field
from typing import Literal

# 1. 定义输出结构
class IntentLLMDecision(BaseModel):
    intent: Literal["GREETING", "FAQ_QUERY", "KNOWLEDGE_QUERY",
                    "FOLLOW_UP", "HUMAN_SERVICE", "OUT_OF_SCOPE"]
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    reason: str = Field(default="")

# 2. 创建带约束的模型
structured_model = get_chat_model(streaming=False).with_structured_output(IntentLLMDecision)

# 3. 调用 → 返回 Pydantic 对象，不是字符串！
decision = structured_model.invoke([
    SystemMessage(content="你是意图识别助手..."),
    HumanMessage(content="用户问：入职流程有哪些步骤？"),
])
print(decision.intent)      # "KNOWLEDGE_QUERY"
print(decision.confidence)  # 0.85
```

**工作原理**：LangChain 将 Pydantic 模型的 JSON Schema 注入 System Prompt → LLM 按 Schema 返回 JSON → Pydantic 自动校验（枚举值不对直接报错，不需要手写正则）。

### 2.5 Message 类型系统

**是什么**：三种消息类型直接映射 OpenAI API 的 role 字段：

| LangChain 类 | OpenAI API role | 用途 |
|-------------|----------------|------|
| `SystemMessage` | `system` | 定义 AI 角色和行为边界 |
| `HumanMessage` | `user` | 用户说的话 |
| `AIMessage` | `assistant` | AI 的回答（多轮对话历史） |

**为什么需要**：LLM 是无状态的——它不会"记住"之前的对话。每次请求必须完整发送历史消息列表。多轮对话的关键：

```python
# 第一轮
messages = [SystemMessage("你是企业助手"), HumanMessage("入职流程")]
response1 = model.invoke(messages)  # "入职流程包括..."

# 第二轮（追问）—— 必须带上历史！
messages.append(response1)                          # 追加 AI 的回答
messages.append(HumanMessage("那审批需要多久"))      # 追加新问题
response2 = model.invoke(messages)
# LLM 看到完整上下文：[System, User("入职流程"), AI("入职包括..."), User("那审批需要多久")]
# → 正确理解"审批"指的是入职审批
```

### 2.6 Prompt Templates

**是什么**：用模板变量管理 System/User Prompt，避免硬编码字符串。

```python
from langchain_core.prompts import ChatPromptTemplate

# 多角色模板
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是{business_domain}的知识助手，名叫{assistant_name}。"),
    ("system", "你只能基于提供的资料回答，不得超出资料范围。"),
    ("human", "参考资料：\n{context}\n\n用户问题：{query}"),
])

# 填充变量
prompt_value = chat_template.invoke({
    "business_domain": "企业内部制度与流程",
    "assistant_name": "小知",
    "context": "[1] 来源：人事制度\n入职流程包括...",
    "query": "入职需要哪些材料",
})
```

**MessagesPlaceholder** — 当消息数量不固定时（如多轮对话历史）：

```python
from langchain_core.prompts import MessagesPlaceholder

chat_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是知识助手。"),
    MessagesPlaceholder(variable_name="history"),  # ← 动态消息列表
    ("human", "{query}"),
])

prompt_value = chat_with_history.invoke({
    "history": [HumanMessage("入职流程"), AIMessage("入职包括...")],
    "query": "那审批呢",
})
# → [System, Human, AI, Human] 共 4 条消息
```

**本项目实现** — `qa_core/prompts/profiles.py` 用 `PromptProfile` 数据类管理多档位模板：

```python
@dataclass(frozen=True)
class PromptProfile:
    name: str
    system_template: str
    user_template: str
    reason: str

# 按意图类型选择不同 Prompt Profile
PROMPT_PROFILES = {
    "FAQ_QUERY": PromptProfile(
        name="faq_answer",
        system_template=FAQ_ANSWER_SYSTEM_PROMPT,
        user_template=FAQ_ANSWER_USER_TEMPLATE,
        reason="FAQ 类问题优先复用标准答案，控制回答长度和业务口径。",
    ),
    "KNOWLEDGE_QUERY": PromptProfile(
        name="knowledge_answer",
        ...
        reason="业务知识咨询需要整合文档资料。",
    ),
    "FOLLOW_UP": PromptProfile(
        name="follow_up",
        ...
        reason="追问需要结合历史理解指代。",
    ),
}
```

> 完整源码：`qa_core/prompts/profiles.py`

### 2.7 Output Parsers

**是什么**：将 LLM 的原始输出（AIMessage 对象）转换成可用的格式。

| Parser | 输入 | 输出 | 用途 |
|--------|------|------|------|
| `StrOutputParser` | AIMessage | str | 提取纯文本 |
| `PydanticOutputParser` | AIMessage（含 JSON） | Pydantic 对象 | 旧版结构化输出 |

**为什么本项目用 `with_structured_output()` 而不是 `PydanticOutputParser`**：新版 API 封装得更好——自动注入 JSON Schema 到 System Prompt，省去手动注入格式指令的步骤。底层仍用 Pydantic Schema，但调用更简洁。

### 2.8 SQLChatMessageHistory — 对话历史持久化

**是什么**：LangChain 的 MySQL 对话历史适配器，一行代码替代手写 CRUD。

**为什么需要**：自己手写需要设计表结构、写 INSERT/SELECT、处理序列化/反序列化、管理连接池和事务。LangChain 全部自动化。

**怎么用**：

```python
# qa_core/memory/history.py — 封装后的使用方式
from langchain_community.chat_message_histories import SQLChatMessageHistory

class ChatHistoryStore(_MySqlStore):
    def for_session(self, session_id: str) -> SQLChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.engine,
            table_name="chat_messages",
        )

    def add_turn(self, session_id: str, question: str, answer: str):
        history = self.for_session(session_id)
        history.add_messages([HumanMessage(content=question), AIMessage(content=answer)])

    def get_context_messages(self, session_id: str) -> list[BaseMessage]:
        """构建传给 RAG 链路的压缩上下文：历史摘要 + 最近对话"""
        recent = self.get_messages(session_id, limit=self.settings.history_recent_messages)
        summary = self.get_summary(session_id)
        if summary:
            return [SystemMessage(content=f"历史摘要：{summary}")] + recent
        return recent
```

> 完整源码：`qa_core/memory/history.py` → `ChatHistoryStore`

**MySQL 的角色边界**：聊天历史 ✅ | 会话摘要 ✅ | 用户反馈 ✅ — FAQ 检索 ❌ | 文档检索 ❌ | 向量相似度 ❌（这些走 Milvus）

### 2.9 Milvus VectorStore — LangChain 封装

**是什么**：LangChain 对 Milvus 的包装层，自动管理向量化和检索，不用手写 pymilvus schema。

**对比直连 pymilvus**：

| 直连 pymilvus | LangChain Milvus |
|-------------|-----------------|
| 手写 schema 定义 | schema 自动推断 |
| 手写向量化逻辑 | add_documents 自动调用 embedding_function |
| 手写搜索的向量化 | similarity_search 自动向量化 query |
| 手动管理连接 | 连接管理内置 |

```python
from langchain_milvus import Milvus

store = Milvus(
    embedding_function=get_embeddings(),     # BGE-M3 → 自动生成 Dense 向量
    builtin_function=bm25_function(),         # Milvus 内置 BM25 → 自动生成 Sparse
    collection_name="enterprise_knowledge_faq",
    vector_field=["dense", "sparse"],         # 双向量字段
    text_field="text",
    primary_field="pk",
    auto_id=False,                            # 手动 ID，支持更新/删除
)
```

> 完整源码：`qa_core/retrieval/store.py` → `MilvusHybridStore`

### 2.10 Document Loaders — 注册表模式

**是什么**：按文件扩展名自动匹配加载器，通过注册表模式管理，新增格式只需加一条注册项，不修改入库主流程。

**为什么不用 if/elif**：扩展新格式时如果散落在 if/elif 分支中，需要找到所有分支并修改。注册表模式把每种格式的"后缀→工厂函数"映射集中管理，新增只要加一行注册。

```python
# qa_core/indexing/document_loaders.py — 注册表模式
DOCUMENT_LOADER_SPECS = (
    DocumentLoaderSpec(suffixes=(".txt", ".md"), factory=_utf8_text_loader, description="UTF-8 文本/Markdown"),
    DocumentLoaderSpec(suffixes=(".pdf",),       factory=_pdf_loader,       description="PDF 文档"),
    DocumentLoaderSpec(suffixes=(".docx", ".doc"), factory=_word_loader,    description="Word 文档"),
    DocumentLoaderSpec(suffixes=(".ppt", ".pptx"), factory=_powerpoint_loader, description="PowerPoint"),
    DocumentLoaderSpec(suffixes=(".csv", ".xlsx", ".xls"), factory=_table_loader, description="CSV/Excel 表格"),
)

# 使用时
def load_file(path: Path) -> list[Document]:
    spec = get_document_loader_spec(path)   # 按后缀查找注册项
    loader = spec.create_loader(path)        # 工厂函数创建 loader
    return loader.load()                     # 统一 .load() 返回 Document 列表
```

> 完整源码：`qa_core/indexing/document_loaders.py`

### 2.11 Text Splitters — 文档切分

**是什么**：将长文档切成适合检索的 chunk，核心是两种策略配合。

**RecursiveCharacterTextSplitter — 递归降级切分**：

```
切分优先级：\n\n（段落）→ \n（换行）→ 。！？（句子）→ ，（短语）→ 空格 → 字符（最后手段）
如果某段 > 500 字符 → 降级到下一个分隔符再切 → 直到能切开为止
```

优先在语义边界（段落、句子）切分，避免把一个完整职位的标题劈开。字符截断是最后的应急手段。

```python
# qa_core/indexing/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHINESE_SEPARATORS = [
    "\n\n", "\n",           # 段落 → 换行
    "。", "！", "？", "；",  # 句子
    "，",                    # 短语
    " ",                     # 词
    "",                      # 字符（最后手段）
]

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 每个 chunk 最多 500 字符
    chunk_overlap=50,        # 相邻 chunk 重叠 50 字符
    separators=CHINESE_SEPARATORS,
)
```

**父子块策略**：

```
原始文档（2000 字符）
  ├── 父块（2000 字符）→ 存于子块 metadata.parent_content，LLM 看到的完整上下文
  └── 子块（500 字符）→ 存入 Milvus，精确召回
```

设计目标：子块精确匹配检索，父块提供完整上下文。避免只检索到半句话而丢失前后文。

> 完整源码：`qa_core/indexing/chunking.py` → `split_documents()`

### 2.12 自研 vs 拥抱生态

| 如果自研 | 采用 LangChain |
|---------|---------------|
| 手写 Milvus schema + pymilvus | LangChain Milvus 自动管理 |
| 手写消息表 + SQL CRUD | SQLChatMessageHistory 自动化 |
| 手写每种文件格式的加载器 | LangChain 100+ Document Loaders |
| 手写文本切分逻辑 | RecursiveCharacterTextSplitter |
| 手写 LLM API 调用 + 重试 | ChatOpenAI 自带重试 |

**但不完全依赖 LangChain 高层 Chain**：本项目不使用 `RetrievalQA`、`ConversationalRetrievalChain` 等黑盒 Chain，而是自己编排 RAG 流程——因为分支逻辑（FAQ 直出/知识查询/追问）、动态阈值、可解释性需求不适合线性管道。

## 三、本讲核心代码串联

```python
# LangChain 组件在本项目中的完整协作流程
# 以下代码提炼自 qa_core/llm/client.py + memory/history.py + prompts/profiles.py

from qa_core.llm.client import get_chat_model
from qa_core.memory.history import get_history_store

# 1. 获取模型客户端（进程级单例，@lru_cache 缓存）
model = get_chat_model(streaming=False)    # 非流式，用于意图识别
stream_model = get_chat_model(streaming=True)  # 流式，用于最终答案

# 2. 加载历史对话（SQLChatMessageHistory 自动从 MySQL 读取）
store = get_history_store()
context_messages = store.get_context_messages("session_abc123")
# → [SystemMessage("历史摘要：上次问了入职流程..."), HumanMessage("入职流程"), AIMessage("...")]

# 3. 意图识别 — with_structured_output 返回 Pydantic 对象
structured_model = model.with_structured_output(IntentLLMDecision)
decision = structured_model.invoke([
    SystemMessage(content="你是意图识别助手..."),
    HumanMessage(content="入职流程有哪些步骤？"),
])
# → IntentLLMDecision(intent="KNOWLEDGE_QUERY", confidence=0.85)

# 4. 根据意图选择 Prompt Profile
profile = PROMPT_PROFILES.get(decision.intent, DEFAULT_PROMPT_PROFILE)
system_prompt = profile.system_template.format(
    assistant_name="小知",
    business_domain="企业制度",
)

# 5. 流式生成最终答案
for chunk in stream_model.stream([
    SystemMessage(content=system_prompt),
    HumanMessage(content=f"参考资料：{retrieved_context}\n\n问题：{query}"),
]):
    yield chunk.content  # 通过 WebSocket 逐个 token 推给前端
```

## 四、常见面试题

**Q1：LangChain 的 Runnable 统一接口解决了什么问题？**
> LangChain 早期不同组件有不同的调用方式——`llm.predict()`、`chain.run()`、`retriever.get_relevant_documents()`。换组件就要换 API、无法串成链、测试困难。Runnable 接口统一为 `invoke()`（完整返回）、`stream()`（流式输出）、`batch()`（并行处理），不管是 ChatOpenAI、VectorStore、PromptTemplate 还是 OutputParser，调用方式完全一样。这使得组件可以灵活替换和组合。

**Q2：with_structured_output 相比手写正则解析有什么优势？**
> 手写正则解析 LLM 的自由文本输出有三大问题：① LLM 返回格式不稳定（有时中文、有时英文、有时有额外解释），正则容易漏；② 字段约束无法强制执行（confidence 范围 0-1、intent 枚举值），正则无法校验；③ 每次新增字段或改枚举值都要重写正则。
>
> `with_structured_output` 将 Pydantic 模型的 JSON Schema 注入 System Prompt，LLM 按 Schema 返回 JSON，Pydantic 自动校验——枚举值超范围直接报错，confidence 超出 [0,1] 直接报错。任何 schema 变更只需改 Pydantic 模型，解析逻辑零修改。

**Q3：ChatPromptTemplate 的 MessagesPlaceholder 解决了什么问题？**
> 普通模板的变量只能是字符串（如 `{query}`），但多轮对话历史的长度不固定——可能是 2 条消息也可能是 20 条。MessagesPlaceholder 允许在模板中留一个"消息列表"插槽，调用时传入 `[HumanMessage(...), AIMessage(...), ...]` 列表，模板自动展开。这样同一个模板既能处理首轮提问（history=[]），也能处理多轮追问。

**Q4：本项目为什么不用 LCEL 管道符串联 RAG 流程？**
> 四个原因：① **分支逻辑复杂**——RAG 有多种分支（FAQ 直出、文档 RAG、信息不足、直接答案），不是线性管道能表达的；② **可解释性**——自编流程每一步都有 `reason` 字段，管道符是黑盒；③ **调试友好**——每阶段的输入输出可单独检查；④ **控制需求**——RAG 需要在阶段之间做动态决策（FAQ 命中直接返回、检索不足追加搜索）。LCEL 适合简单线性链（如 prompt → model → parser），不适合有决策逻辑的工程链路。

**Q5：注册表模式和 if/elif 分支有什么区别？为什么 Document Loader 用注册表？**
> if/elif 分支把每种文件格式的处理逻辑写死在主流程中，新增格式需要找到并修改所有分支。注册表模式把"后缀→工厂函数"的映射集中在一个表中，主流程只查表不关心具体类型。新增格式只需加一条注册项 `DocumentLoaderSpec(suffixes=(".xyz",), factory=_xyz_loader, ...)`，主流程零修改。对扩展开放，对修改关闭——符合开闭原则。

**Q6：父子块切分策略的价值是什么？**
> 单一切片策略有一个矛盾：chunk 太大（如 2000 字符）→ 精确检索难命中具体片段；chunk 太小（如 200 字符）→ LLM 看到的上下文太窄，无法理解完整语义。父子块策略解决这个矛盾：子块（500 字符）存入 Milvus 做精确召回——命中具体句子；父块（2000 字符）存于子块 metadata.parent_content——检索命中子块后，LLM 看到的是父块的完整上下文。精确 + 完整兼得。

## 五、本讲速查卡

| 术语 | 一句话定义 |
|------|-----------|
| LangChain | LLM 应用开发框架，标准化封装对话历史、文档加载、向量存储、提示模板、输出解析 |
| Runnable | LangChain 统一调用接口：invoke(完整)/stream(流式)/batch(并行)，所有组件实现相同方法 |
| LCEL | LangChain 表达式语言，用 `\|` 管道符串联多个 Runnable 成链 |
| ChatOpenAI | LangChain 对 OpenAI 兼容接口的封装类，本项目用它统一接入 DashScope 等模型 |
| SystemMessage | LangChain 消息类型，映射 OpenAI `role:system`，定义 AI 角色和行为边界 |
| HumanMessage | LangChain 消息类型，映射 OpenAI `role:user`，用户说的话 |
| AIMessage | LangChain 消息类型，映射 OpenAI `role:assistant`，AI 的回复（多轮对话历史） |
| ChatPromptTemplate | 多角色 Prompt 模板，支持 System/User/AI 等多条消息模板化 |
| MessagesPlaceholder | 模板中的"消息列表"插槽，用于动态插入不固定数量的多轮对话历史 |
| with_structured_output | ChatOpenAI 方法，用 Pydantic 模型强制约束 LLM 输出 JSON Schema，自动校验 |
| StrOutputParser | 最简单的输出解析器，把 AIMessage 转成纯字符串 |
| SQLChatMessageHistory | LangChain 的 MySQL 对话历史适配器，按 session_id 隔离，自动 CRUD |
| Document Loader | 将文件（PDF/Word/MD/CSV 等）加载为 LangChain Document 对象 |
| 注册表模式 | 后缀→工厂函数的映射表，新增文件类型只加一条注册，不改主流程 |
| RecursiveCharacterTextSplitter | 递归降级切分器：段落→句子→短语→字符，优先在语义边界切分 |
| 父子块策略 | 子块（500 字）精确检索 + 父块（2000 字）提供 LLM 完整上下文 |
| MarkdownHeaderTextSplitter | 按 #/##/### 标题层级切分 Markdown，切分后保留 h1/h2/h3 元数据 |
| Prompt Profile | 本项目按意图类型预设的 System+User Prompt 模板组合，frozen dataclass 防运行时篡改 |
