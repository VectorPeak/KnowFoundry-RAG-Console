# 第3讲：LangChain 生态系统

**上一讲**：[RAG 核心概念深入](./02-rag-fundamentals.md)  
**下一讲**：[Milvus 索引机制与基本操作](./04-milvus-index-and-operations.md)



## 本讲目标

- 理解 LangChain 的 Runnable 统一接口和 LCEL 管道语法
- 掌握 ChatOpenAI 的 invoke/stream/with_structured_output 三种调用模式
- 理解 Message 类型系统和 Prompt Template 的用法
- 掌握 Output Parser 与 Pydantic 结构化输出的绑定
- 理解 SQLChatMessageHistory、Milvus VectorStore、Document Loader、Text Splitter 在本项目中的应用

---

## 第一部分：LangChain 是什么

### 1.1 核心定位

**LangChain 不是一个模型，而是一个 LLM 应用开发框架**。它解决的问题是：

> 当你需要构建一个"不只是调用一次 API"的 LLM 应用时，你需要管理对话历史、文档加载、文本切分、向量存储、提示模板、输出解析……LangChain 把这些环节做了标准化封装。

打个比方：
- **直接用 OpenAI SDK**：等于自己买食材做菜，灵活但每道菜都要从头做起
- **用 LangChain**：等于用半成品料理包，大部分工序已经做好了，你只需要组合和调参

### 1.2 三大核心抽象

```mermaid
flowchart TD
    subgraph LangChain["LangChain 核心抽象"]
        direction TB
        Model["🔌 Model（模型）<br/>ChatOpenAI, HuggingFace, Anthropic<br/>统一的 LLM 调用接口"]
        Chain["🔗 Chain（链）<br/>prompt | model | parser<br/>用管道符串联多个组件"]
        Agent["🤖 Agent（智能体）<br/>LLM 自主决定下一步<br/>调用哪个工具"]
    end

    Model --> Chain --> Agent

    subgraph Project["本项目使用策略"]
        P1["✅ 全面使用 Model 层<br/>ChatOpenAI 封装 DashScope"]
        P2["⚠️ 部分使用 Chain 层<br/>只用底层组件，不用高层 Chain"]
        P3["🔮 二期规划 Agent 层<br/>LangGraph 动态工具调用"]
    end

    style LangChain fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style Project fill:#ECFDF5,stroke:#059669,stroke-width:2px
```

**Model（模型）**：对 LLM 的统一封装。不管是 OpenAI、DashScope 还是本地模型，都用相同的接口调用。

**Chain（链）**：把多个步骤串成一个流程。LangChain 的 LCEL（LangChain Expression Language）用管道符 `|` 串联组件。本项目没有使用 LangChain 的高层 Chain 抽象（如 `RetrievalQA`），而是自己编排 RAG 流程。

**Agent（智能体）**：比 Chain 更灵活 — LLM 自己决定下一步做什么、调用哪个工具。本项目的二期规划中会用到 LangGraph Agent。

### 1.3 本项目使用的 LangChain 组件全景

```mermaid
flowchart TD
    subgraph LangChain["LangChain 组件"]
        CO["ChatOpenAI<br/>LLM 调用抽象"]
        MSG["Message Types<br/>System/Human/AI"]
        PT["PromptTemplate<br/>+ ChatPromptTemplate"]
        OP["Output Parser<br/>StrOutputParser<br/>PydanticOutputParser"]
        MV["Milvus VectorStore<br/>向量存储封装"]
        DL["Document Loaders<br/>PDF/Word/MD/Excel"]
        TS["Text Splitters<br/>RecursiveCharacterTextSplitter<br/>MarkdownHeaderTextSplitter"]
        SH["SQLChatMessageHistory<br/>MySQL 历史管理"]
    end

    subgraph App["本项目 qa_core"]
        Client["llm/client.py"]
        History["memory/history.py"]
        Store["retrieval/store.py"]
        Loaders["indexing/document_loaders.py"]
        Chunk["indexing/chunking.py"]
        Intent["intent/classifier.py"]
        Prompts["prompts/profiles.py"]
    end

    CO --> Client
    MSG --> Client
    MSG --> History
    PT --> Prompts
    OP --> Intent
    MV --> Store
    DL --> Loaders
    TS --> Chunk
    SH --> History

    style LangChain fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style App fill:#ECFDF5,stroke:#059669,stroke-width:2px
```

上面的映射图展示了各组件在项目中的落点。下面看 LangChain 统一这些组件的核心抽象——Runnable 接口。

---

## 第二部分：Runnable 接口 — LangChain 的统一调用协议

### 2.1 为什么需要 Runnable

在 LangChain 早期，不同类型的组件有不同的调用方式：

```python
llm.predict("hello")                   # LLM 用 .predict()
chain.run(input="hello")               # Chain 用 .run()
retriever.get_relevant_documents("x")  # Retriever 用 .get_relevant_documents()
```

这导致：换一个组件类型就要换一套 API；无法把不同类型的组件串成一条链；测试和调试困难。

### 2.2 Runnable 的三个核心方法

LangChain 引入了 **Runnable** 接口，所有组件都实现相同的三个方法：

```python
from langchain_core.runnables import Runnable

# 所有 LangChain 组件都实现了这个接口
class Runnable:
    def invoke(self, input, config=None) -> Output:
        """单个输入 → 单个输出（同步，等完整结果）"""
        ...

    def stream(self, input, config=None) -> Iterator[Output]:
        """单个输入 → 流式输出（逐个 token）"""
        ...

    def batch(self, inputs: list, config=None) -> list[Output]:
        """多个输入 → 多个输出（并行处理）"""
        ...
```

**这意味着**：不管是 ChatOpenAI、Milvus VectorStore、还是自定义 Chain，调用方式完全一样：

```python
# ChatOpenAI 支持 invoke
model = ChatOpenAI(model="qwen-plus")
response = model.invoke([HumanMessage("你好")])     # → AIMessage

# ChatOpenAI 也支持 stream
for chunk in model.stream([HumanMessage("你好")]):  # → 逐个 token
    print(chunk.content, end="")

# ChatOpenAI 还支持 batch（并行处理多个问题）
responses = model.batch([
    [HumanMessage("入职流程")],
    [HumanMessage("报销规则")],
    [HumanMessage("VPN设置")],
])
# → [AIMessage, AIMessage, AIMessage]
```

### 2.3 invoke vs stream 的区别

```python
# invoke: 等全部完成后返回完整结果
model = ChatOpenAI(streaming=False, model="qwen-plus")
response = model.invoke([HumanMessage("入职流程有哪些步骤")])
print(response.content)
# 等了 5 秒 → "入职流程包括以下步骤：1. 提交材料 2. 签订合同 ..."

# stream: 逐个 token 返回，前端可以实时渲染
model = ChatOpenAI(streaming=True, model="qwen-plus")
for chunk in model.stream([HumanMessage("入职流程有哪些步骤")]):
    print(chunk.content, end="", flush=True)
# 50ms 后 → "入"
# 100ms 后 → "职"
# 150ms 后 → "流"
# ...
# 5000ms 后 → 全部完成
```

```mermaid
sequenceDiagram
    participant C as 调用方
    participant M as ChatOpenAI

    Note over C,M: invoke 非流式模式
    C->>M: invoke(messages)
    Note over C,M: 等待 5 秒...
    M-->>C: AIMessage返回完整结果

    Note over C,M: stream 流式模式
    C->>M: stream(messages)
    M-->>C: chunk 入 (50ms)
    M-->>C: chunk 职 (100ms)
    M-->>C: chunk 流 (150ms)
    M-->>C: chunk 程 (200ms)
    M-->>C: ... 持续推送
```

### 2.4 LCEL：用管道符串联 Runnable

**LCEL（LangChain Expression Language）** 使用 `|` 管道符将多个 Runnable 串联，数据自动从前一个流入后一个：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 定义三个组件
prompt = ChatPromptTemplate.from_template("用一句话回答：{question}")
model = ChatOpenAI(model="qwen-plus")
parser = StrOutputParser()  # 把 AIMessage 转成纯字符串

# LCEL 方式：用管道符串联
chain = prompt | model | parser

# 数据流：prompt.invoke(input) → model.invoke(prompt_output) → parser.invoke(model_output)
result = chain.invoke({"question": "入职流程有哪些步骤"})
# result: "入职流程包括提交材料、签订合同、部门审批等步骤。"
```

```mermaid
flowchart LR
    Input["输入 dict<br/>{question: '...'}"] --> A["prompt.invoke()"]
    A -->|"ChatPromptValue"| B["model.invoke()"]
    B -->|"AIMessage"| C["parser.invoke()"]
    C --> Output["输出 str<br/>'入职流程包括...'"]

    style A fill:#EFF6FF,stroke:#2563EB
    style B fill:#ECFDF5,stroke:#059669
    style C fill:#FFFBEB,stroke:#D97706
```

**管道符的本质**：

```python
chain = prompt | model | parser

# 等价于：
chain = RunnableSequence(first=prompt, middle=[model], last=parser)

# 也等价于手动嵌套：
def manual_chain(input_dict):
    prompt_value = prompt.invoke(input_dict)      # 步骤1
    ai_message = model.invoke(prompt_value)        # 步骤2
    result = parser.invoke(ai_message)             # 步骤3
    return result
```

### 2.5 本项目为什么不用 LCEL 管道符

你可能会注意到本项目没有使用 `|` 管道符来串联 RAG 流程。这是**刻意为之**：

```python
# ❌ 本项目不这样做（因为是线性管道，无法表达分支逻辑）：
rag_chain = retrieve | format_context | prompt | model | parser

# ✅ 本项目自己编排流程（支持多路分支和提前退出）：
def stream_query(...):
    intent = classify_intent(...)      # 可能直接结束（问候/越界）
    plan = build_retrieval_plan(...)   # 动态参数
    faq_result = search_faq(...)       # 可能提前结束（FAQ 直出）
    doc_result = search_doc(...)       # 可能信息不足
    for chunk in model.stream(...):    # 逐 token 流式推送
        yield chunk
```

四个原因：
1. **分支逻辑复杂**：RAG 有 FAQ 直出、文档 RAG、信息不足、直接答案等多种分支，不适合线性管道
2. **可解释性**：自己编排的每一步都有 `reason` 字段
3. **调试友好**：每个阶段的输入输出可以单独检查和记录
4. **管道干扰**：LCEL 会自动处理数据流转，但 RAG 需要在阶段之间做决策和控制

---

## 第三部分：Chat Models — ChatOpenAI 详解

### 3.1 为什么使用 OpenAI 兼容接口

本项目的 LLM 实际是**阿里云 DashScope**（通义千问），但代码统一使用 LangChain 的 `ChatOpenAI`：

1. **DashScope 提供 OpenAI 兼容接口**：URL 不同，但 API 格式完全兼容
2. **统一抽象**：切换 LLM（如换成 DeepSeek、Claude）只需改 `.env` 中的三个变量
3. **LangChain 原生支持**：ChatOpenAI 是 LangChain 支持最完善的 ChatModel

```bash
# 切换 LLM 只需改 .env，代码零修改
# 用 DashScope
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=qwen-plus

# 换成 DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-yyy
LLM_MODEL=deepseek-chat
```

### 3.2 get_chat_model() 工厂函数

```python
# qa_core/llm/client.py
from functools import lru_cache
from langchain_openai import ChatOpenAI

@lru_cache(maxsize=2)
def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,              # "qwen-plus"
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,   # 0.0-1.0，越低越确定
        timeout=settings.llm_timeout,           # 请求超时秒数
        streaming=streaming,
    )
```

**`@lru_cache(maxsize=2)`**：缓存 `streaming=True` 和 `streaming=False` 各一个实例。避免每次请求重复创建客户端。但**不能缓存模型输出** — 答案依赖会话历史和检索上下文，缓存输出会导致不同用户看到相同回复。

### 3.3 非流式调用 — invoke()

```python
# 用于需要完整结果的后台任务
model = get_chat_model(streaming=False)

# 简单问答
response = model.invoke([HumanMessage(content="入职流程有哪些步骤")])
print(response.content)  # AIMessage 对象，.content 是字符串

# 带系统指令的问答
response = model.invoke([
    SystemMessage(content="你是企业知识助手，只能基于资料回答。"),
    HumanMessage(content="入职流程有哪些步骤"),
])

# 多轮对话
response = model.invoke([
    SystemMessage(content="你是企业知识助手。"),
    HumanMessage(content="入职需要哪些材料"),
    AIMessage(content="需要身份证复印件、学历证书..."),
    HumanMessage(content="还需要体检报告吗"),
])
```

**本项目中的使用**：意图识别、查询改写、历史摘要 — 这些都需要完整结果才能继续下一步。

### 3.4 流式调用 — stream()

```python
# 用于最终答案生成，逐个 token 推送给前端
model = get_chat_model(streaming=True)

for chunk in model.stream([
    SystemMessage(content="你是企业知识助手。"),
    HumanMessage(content="入职流程有哪些步骤"),
]):
    token = str(getattr(chunk, "content", "") or "")
    if not token:
        continue
    # 通过 WebSocket 推送给浏览器
    yield {"type": "token", "content": token}
```

**LangChain 的 stream() 做了什么**：ChatOpenAI 在底层开启了 OpenAI API 的 `stream=True` 参数。API 不再一次性返回完整 JSON，而是通过 SSE（Server-Sent Events）逐个推送 chunk。LangChain 封装了 SSE 解析，把每个 chunk 包装成 `AIMessageChunk` 对象。

### 3.5 结构化输出 — with_structured_output()

这是 LangChain 最强大的功能之一：**强制 LLM 按 Pydantic 模型的结构返回 JSON**。

```python
from pydantic import BaseModel, Field
from typing import Literal

# 定义输出结构
class IntentLLMDecision(BaseModel):
    intent: Literal["GREETING", "FAQ_QUERY", "KNOWLEDGE_QUERY",
                    "FOLLOW_UP", "HUMAN_SERVICE", "OUT_OF_SCOPE"]
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    reason: str = Field(default="")

# 创建带结构化输出的模型
model = get_chat_model(streaming=False)
structured_model = model.with_structured_output(IntentLLMDecision)

# 调用 — 返回值是 Pydantic 对象，不是字符串
decision = structured_model.invoke([
    SystemMessage(content="你是意图识别助手..."),
    HumanMessage(content="用户问：入职流程有哪些步骤？"),
])

print(decision.intent)      # "KNOWLEDGE_QUERY"
print(decision.confidence)  # 0.85
print(decision.reason)      # "用户询问企业流程制度类问题"
```

**工作原理**：

```
1. LangChain 将 Pydantic 模型的 JSON Schema 注入 System Prompt
2. LLM 按 JSON Schema 的字段和约束返回 JSON
3. Pydantic 自动校验 JSON → 如果 LLM 返回了不在枚举中的值，直接报错
```

**不使用 structured output 的风险**：

```python
# 让 LLM 返回自由文本
response = model.invoke("判断意图：入职流程有哪些步骤")
# LLM 可能返回：
# "这是一个关于企业内部流程的知识咨询问题"           ← 中文解释
# "INTENT: KNOWLEDGE_QUERY"                          ← 另一种格式
# "我认为用户想了解入职流程，意图是知识咨询类型"      ← 又一种格式
# 每种格式都需要手写不同的解析正则 → 脆弱且难维护
```

### 3.6 LLM 连通性验证

```python
def validate_llm_connectivity() -> None:
    """启动时真实调用一次 LLM，确认 Key、URL、模型名都有效。"""
    try:
        response = get_chat_model(streaming=False).invoke(
            [HumanMessage(content="回复 OK")]
        )
    except Exception as exc:
        raise RuntimeError(
            "LLM 服务不可用：请检查 DASHSCOPE_API_KEY、"
            "DASHSCOPE_BASE_URL 和 LLM_MODEL。"
        ) from exc
```

不是只检查配置是空还是占位符 — 是真正调用一次。这样能暴露 API Key 过期、余额不足、网络不通、模型名无效等所有问题。

---

## 第四部分：Message 类型系统

### 4.1 三种消息类型

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# SystemMessage：系统指令，定义 AI 的角色和行为边界
SystemMessage(content="""
你是企业知识助手，专门解答企业内部制度与流程相关问题。
你必须严格基于提供的参考资料回答，不得超出资料范围。
""")

# HumanMessage：用户说的话
HumanMessage(content="入职流程有哪些步骤？")

# AIMessage：AI 的回答（用于记录多轮对话历史）
AIMessage(content="入职流程包括：1. 提交材料 2. 签订合同 ...")
```

### 4.2 OpenAI API 的消息格式

理解 Message 对象的最直接方式 — 看它们如何映射到 OpenAI API 的请求 JSON：

```json
{
  "model": "qwen-plus",
  "messages": [
    {"role": "system",    "content": "你是企业知识助手，严格基于资料回答。"},
    {"role": "user",      "content": "入职流程有哪些步骤？"},
    {"role": "assistant", "content": "入职流程包括提交材料、签订合同..."},
    {"role": "user",      "content": "还需要体检报告吗？"}
  ]
}
```

LangChain 的 Message 对象：
- `SystemMessage` → `role: "system"`
- `HumanMessage` → `role: "user"`
- `AIMessage` → `role: "assistant"`

### 4.3 消息序列的意义

多轮对话的关键：LLM 是无状态的，它不会"记住"之前的对话。每次请求都必须完整发送历史消息。

```python
# 第一轮
messages = [SystemMessage("你是企业助手"), HumanMessage("入职流程")]
response1 = model.invoke(messages)
print(response1.content)  # "入职流程包括..."

# 第二轮（追问）—— 必须带上历史！
messages.append(response1)                         # 追加 AI 的回答
messages.append(HumanMessage("那审批需要多久"))     # 追加新问题
response2 = model.invoke(messages)
# LLM 看到完整上下文：[System, User("入职流程"), AI("入职包括..."), User("那审批需要多久")]
# 所以能正确理解"审批"指的是入职审批
```

```mermaid
sequenceDiagram
    participant U as 用户
    participant S as 系统

    rect rgb(239, 246, 255)
        Note over U,S: 第一轮
        S->>LLM: [System, User:"入职流程"]
        LLM-->>S: "入职流程包括..."
    end

    rect rgb(236, 253, 245)
        Note over U,S: 第二轮（追问）
        S->>LLM: [System, User:"入职流程", AI:"入职包括...", User:"审批多久"]
        LLM-->>S: "入职审批通常需要..."
    end
```

---

## 第五部分：Prompt Templates — 提示词模板

### 5.1 最简单的 PromptTemplate

```python
from langchain_core.prompts import PromptTemplate

# 定义一个模板
template = PromptTemplate.from_template("用一句话介绍{company}的{product}")
# 填充变量
prompt_value = template.invoke({"company": "阿里云", "product": "通义千问"})
print(prompt_value.text)
# "用一句话介绍阿里云的通义千问"
```

### 5.2 ChatPromptTemplate — 多角色消息模板

对于需要 SystemMessage + HumanMessage 的场景：

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义多角色模板
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

# prompt_value.messages 是三个 Message 对象：
# [SystemMessage("你是企业内部制度与流程的知识助手..."),
#  SystemMessage("你只能基于提供的资料回答..."),
#  HumanMessage("参考资料：\n[1] 来源：人事制度\n...")]
```

### 5.3 本项目中的 Prompt 模板

本项目的 `qa_core/prompts/profiles.py` 使用 `ChatPromptTemplate` 的简化版 — 直接使用字符串模板 + 场景变量注入：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptProfile:
    name: str
    system_template: str    # System Prompt 模板
    user_template: str      # User Prompt 模板
    reason: str

# 所有档位集中定义在 PROMPT_PROFILES 字典中，不再单独声明模块级常量
PROMPT_PROFILES: dict[str, PromptProfile] = {
    "FAQ_QUERY": PromptProfile(
        name="faq_answer",
        system_template=FAQ_ANSWER_SYSTEM_PROMPT,
        user_template=FAQ_ANSWER_USER_TEMPLATE,
        reason="FAQ 类问题优先复用标准答案，控制回答长度和业务口径。",
    ),
    "KNOWLEDGE_QUERY": PromptProfile(
        name="knowledge_answer",
        system_template=KNOWLEDGE_ANSWER_SYSTEM_PROMPT,
        user_template=KNOWLEDGE_ANSWER_USER_TEMPLATE,
        reason="业务知识咨询需要整合文档资料，允许按流程、规则、步骤或说明结构化回答。",
    ),
    "FOLLOW_UP": PromptProfile(
        name="follow_up",
        system_template=FOLLOW_UP_ANSWER_SYSTEM_PROMPT,
        user_template=FOLLOW_UP_ANSWER_USER_TEMPLATE,
        reason="追问需要结合历史理解指代，但回答焦点仍限定在当前问题。",
    ),
}
```

模板中的 `{assistant_name}`、`{business_domain}` 等变量在运行时从场景 TOML 配置注入，使得同一套模板可用于全部 8 个行业场景。

### 5.4 MessagesPlaceholder — 动态消息列表

当消息数量不固定时（如多轮对话历史），使用 `MessagesPlaceholder`：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 支持动态插入历史消息
chat_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是知识助手。"),
    MessagesPlaceholder(variable_name="history"),  # ← 动态消息列表
    ("human", "{query}"),
])

# 调用时传入历史消息列表
prompt_value = chat_with_history.invoke({
    "history": [
        HumanMessage("入职流程"),
        AIMessage("入职包括..."),
    ],
    "query": "那审批呢",
})
# 结果：5 条消息 [System, Human, AI, Human]
```

---

## 第六部分：Output Parsers — 输出解析器

### 6.1 StrOutputParser — 最简单的解析器

把 `AIMessage` 对象转成纯字符串：

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
model = ChatOpenAI(model="qwen-plus")

chain = model | parser
result = chain.invoke([HumanMessage("你好")])
# result: "你好！有什么可以帮助你的吗？"
# 如果没有 parser, result 是 AIMessage(content="你好！...")
```

### 6.2 PydanticOutputParser — 结构化输出（旧方式）

在 `with_structured_output` 出现之前，LangChain 使用 `PydanticOutputParser` 来约束 LLM 输出：

```python
from langchain_core.output_parsers import PydanticOutputParser

class IntentResult(BaseModel):
    intent: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=IntentResult)

# parser 自动生成格式指令，注入到 prompt 中
format_instructions = parser.get_format_instructions()
# 输出：
# "The output should be formatted as a JSON instance that conforms
#  to the JSON schema below. Here is the output schema:
#  {"properties": {"intent": {"type": "string"}, ...}}"
```

本项目使用 `with_structured_output()`（新版 API），它内部仍使用 Pydantic Schema，但封装得更好，不需要手动注入格式指令。

---

## 第七部分：SQLChatMessageHistory — 对话历史持久化

### 7.1 为什么用 LangChain 的历史管理

自己手写聊天历史的数据库 CRUD 需要：
- 设计表结构（session_id, role, content, created_at...）
- 手写 INSERT / SELECT 语句
- 手写 Message 对象的序列化/反序列化
- 处理连接池、事务、字符编码

LangChain 的 `SQLChatMessageHistory` 把这些全部自动化：

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

# 只需指定 session_id 和数据库连接
history = SQLChatMessageHistory(
    session_id="session_abc123",
    connection="mysql+pymysql://user:pass@127.0.0.1:3306/subjects_kg",
    table_name="chat_messages",
)

# 添加消息
history.add_message(HumanMessage(content="入职流程"))
history.add_message(AIMessage(content="入职流程包括..."))

# 读取消息（返回 LangChain Message 对象列表）
messages = history.messages
# [HumanMessage("入职流程"), AIMessage("入职流程包括...")]

# 清空会话
history.clear()
```

### 7.2 本项目的 ChatHistoryStore 封装

```python
class ChatHistoryStore(_MySqlStore):
    """_MySqlStore provides the lazy engine property and _execute_ddl() method for MySQL operations."""

    def for_session(self, session_id: str) -> SQLChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.settings.mysql_sync_uri,
            table_name="chat_messages",
        )

    def add_turn(self, session_id: str, question: str, answer: str):
        history = self.for_session(session_id)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=answer))
```

### 7.3 MySQL 的角色边界

```
MySQL 负责：                   MySQL 不负责：
  ✅ 聊天历史                   ❌ FAQ 检索（Milvus）
  ✅ 会话摘要                   ❌ 文档检索（Milvus）
  ✅ 用户反馈                   ❌ 向量相似度（Milvus）
```

---

## 第八部分：Milvus VectorStore — 向量存储

### 8.1 LangChain 的 Milvus 封装

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

# 写入：add_documents 自动调用 embedding_function + builtin_function
documents = [Document(page_content="入职需要提交身份证...", metadata={...})]
store.add_documents(documents)

# 检索：自动将 query 向量化，执行 Dense + Sparse 混合检索
results = store.similarity_search_with_score("入职需要什么材料", k=10)
# 返回 [(Document, score), ...]
```

**对比直接 pymilvus**：

| 直接 pymilvus | LangChain Milvus |
|-------------|-----------------|
| 手写 schema 定义 | schema 自动推断 |
| 手写向量化逻辑 | add_documents 自动调用 embedding_function |
| 手写搜索的向量化 | similarity_search 自动向量化 query |
| 手动管理连接 | 连接管理内置 |

> 📖 **深入学习**：第3讲展示了 LangChain Milvus 的 API 用法（"怎么用"），[第4讲：Milvus 索引机制与基本操作](./04-milvus-index-and-operations.md) 将深入这些 API 底层实际执行的 pymilvus 操作（"底层做了什么"）——包括自动建 Schema、创建 HNSW 索引、生成 Dense+Sparse 双向量、以及 langchain-milvus 与 PyMilvus 的兼容边界。

---

## 第九部分：Document Loaders — 文档加载器

### 9.1 LangChain 内置 Loader

下面的示例展示了 LangChain 内置的几种常见文档加载器。项目中通过注册表模式（见下节）统一管理它们，实现按文件扩展名自动匹配。

LangChain 提供了 100+ 种文档加载器，覆盖主流文件格式：

```python
# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/hr_policy.pdf")
docs = loader.load()  # 每页一个 Document

# Markdown / 纯文本
from langchain_community.document_loaders import TextLoader
loader = TextLoader("data/onboarding.md", encoding="utf-8")
docs = loader.load()

# Word
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("data/contract.docx")
docs = loader.load()

# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data/employees.csv")
docs = loader.load()

# 每个 Document 结构
# Document(
#     page_content="入职流程包括以下步骤...",
#     metadata={"source": "data/hr_policy.pdf", "page": 1}
# )
```

### 9.2 本项目的注册表模式

本项目不是用 `if/elif` 判断文件后缀，而是用**注册表模式**：

```python
# 每种文件格式定义一个注册项
DOCUMENT_LOADER_SPECS: tuple[DocumentLoaderSpec, ...] = (
    DocumentLoaderSpec(
        suffixes=(".txt", ".md"),
        factory=_utf8_text_loader,       # 工厂函数，接收 Path 返回 Loader
        description="UTF-8 文本/Markdown"
    ),
    DocumentLoaderSpec(
        suffixes=(".pdf",),
        factory=_pdf_loader,
        description="PDF 文档"
    ),
    DocumentLoaderSpec(
        suffixes=(".docx",),
        factory=_word_loader,
        description="Word 文档"
    ),
    DocumentLoaderSpec(
        suffixes=(".csv", ".xlsx", ".xls"),
        factory=_table_loader,           # 本项目自定义的表格加载器
        description="表格文件 — 按行解析"
    ),
)

# 使用时
spec = get_document_loader_spec(file_path)  # 接收 Path 对象，内部通过 path.suffix.lower() 查找
if spec:
    documents = load_file(file_path)                # 接收 Path 对象，内部自行查找 spec 并创建 Loader
```

扩展新格式只需添加一个注册项，不修改入库主流程。

---

## 第十部分：Text Splitters — 文档切分

### 10.1 RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHINESE_SEPARATORS = [
    "\n\n", "\n",           # 段落 → 换行
    "。", "！", "？", "；",  # 句子
    "，",                    # 短语
    " ",                     # 词
    "",                      # 字符（最后手段）
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 每个 chunk 最多 500 字符
    chunk_overlap=50,        # 相邻 chunk 重叠 50 字符
    separators=CHINESE_SEPARATORS,
)

chunks = splitter.split_documents(documents)
```

**递归降级切分算法**：先用 `\n\n`（段落）切 → 某段超过 500 字符 → 对该段用 `\n` 切 → 某行超过 500 字符 → 对该行用 `。` 切 → ...最终手段用字符强制截断。优先在语义边界切分。

### 10.2 父子块策略

```mermaid
flowchart TD
    Doc["📄 原始文档（2000 字符）"] --> Parent["📋 父块<br/>2000 字符, overlap 200<br/>→ 给 LLM 的完整上下文"]
    Doc --> Child["📌 子块<br/>500 字符, overlap 50<br/>→ 存入 Milvus 精确检索"]

    Child --> Milvus[("💾 Milvus<br/>page_content = 子块")]
    Parent --> Meta["子块 metadata 中<br/>parent_content = 父块全文"]

    Milvus --> Search["🔍 检索命中子块"]
    Meta --> Context["📝 LLM 看到<br/>子块 + 父块完整上下文"]

    style Doc fill:#EFF6FF,stroke:#2563EB,stroke-width:2px
    style Milvus fill:#ECFDF5,stroke:#059669,stroke-width:2px
```

> 📖 **深入学习**：如果你想全面理解 Parent-Child Chunking 的设计原理、chunk size/overlap 的选择依据、不同文档类型的切分策略，请阅读 [附录H：文档切分策略](./appendix/appendix-h-chunking-strategy.md)。

注意：此处展示的是 MarkdownHeaderTextSplitter（按标题拆分），与上方流程图的父子块策略不同。父子块的具体实现见附录 H。

### 10.3 MarkdownHeaderTextSplitter

```python
# 注意：此处展示的是 MarkdownHeaderTextSplitter（按标题拆分），
# 与上方流程图的父子块策略不同。父子块的具体实现见附录 H。
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

# 输入：
# # 入职管理
# ## 入职流程
# 入职需要提交...
# ## 转正流程
# 试用期结束后...

# 输出：
# [
#   Document("入职需要提交...", metadata={"h1": "入职管理", "h2": "入职流程"}),
#   Document("试用期结束后...", metadata={"h1": "入职管理", "h2": "转正流程"}),
# ]
```

LLM 生成答案时可以引用层级结构："来源：入职管理 > 入职流程"。

---

## 第十一部分：自研 vs 拥抱生态

### 11.1 全面采用 LangChain 的理由

| 如果自研 | 采用 LangChain |
|---------|---------------|
| 手写 Milvus schema + pymilvus | LangChain Milvus 自动管理 |
| 手写消息表 + SQL CRUD | SQLChatMessageHistory 自动化 |
| 手写每种文件格式的加载器 | LangChain Document Loaders（100+ 格式） |
| 手写文本切分逻辑 | RecursiveCharacterTextSplitter |
| 手写 LLM API 调用 + 重试 | ChatOpenAI 自带重试 |

### 11.2 不完全依赖 LangChain 的理由

本项目没有使用 LangChain 的高层 Chain（如 `RetrievalQA`、`ConversationalRetrievalChain`），而是自己编排 RAG 流程：

1. **高层 Chain 是黑盒**：排查问题时很难知道内部发生了什么
2. **分支逻辑**：FAQ 分层检索、动态阈值、Prompt Profile 不适合线性管道
3. **可解释性**：自编流程的每个步骤都有 `reason` 字段

---

## 第十二部分：后续讲次中的 LangChain 实战位置

学完本讲后，以下组件将在后续讲次中反复出现。带着本讲的体系认知去读代码，效果更好：

| 组件 | 本讲覆盖 | 后续讲次实战 |
|------|---------|------------|
| ChatOpenAI + 结构化输出 | ✅ §3.5 | 第5讲（意图分类）、第11讲（Prompt） |
| Message 类型 | ✅ §4 | 第7讲（查询改写）、第9讲（QAService） |
| ChatPromptTemplate | ✅ §5 | 第11讲（Prompt Profile） |
| SQLChatMessageHistory | ✅ §7 | 第7讲（历史压缩）、第9讲 |
| Milvus VectorStore | ✅ §8 | 第4讲（索引机制）、第8讲（Hybrid Search 详解） |
| Document Loaders | ✅ §9 | 第16讲（入库链路） |
| RecursiveCharacterTextSplitter | ✅ §10 | 第16讲、附录E |

---

## 重点掌握

| 优先级 | 内容 | 原因 |
|--------|------|------|
| ★★★ 必会 | Runnable 统一接口：invoke（完整返回）/ stream（流式）/ batch（并行） | LangChain 所有组件的统一调用协议 |
| ★★★ 必会 | ChatOpenAI 的三种调用模式：invoke（非流式）、stream（流式）、with_structured_output（Pydantic 约束输出） | 项目中使用最多的核心能力 |
| ★★★ 必会 | Message 类型系统：SystemMessage（系统指令）、HumanMessage（用户）、AIMessage（AI）及消息序列对多轮对话的意义 | 对话历史和 Prompt 构建的基础 |
| ★★ 理解 | LCEL 管道符（`|`）串联 Runnable 及本项目不用的原因（分支逻辑复杂） | 理解为什么本项目自己编排流程 |
| ★★ 理解 | ChatPromptTemplate + MessagesPlaceholder 构建多角色模板 | Prompt Profile 的实现基础 |
| ★★ 理解 | SQLChatMessageHistory 封装 MySQL 对话历史（自动 CRUD，按 session_id 隔离） | 本项目历史管理的实现方式 |
| ★★ 理解 | Milvus VectorStore：自动向量化 + BM25 内置函数 + auto_id 更新机制 | 检索模块的 LangChain 封装 |
| ★★ 理解 | Document Loaders 注册表模式（后缀→工厂函数的映射） | 文件格式扩展性的设计模式 |
| ★ 了解 | Text Splitters（RecursiveCharacterTextSplitter 递归降级切分、MarkdownHeaderTextSplitter 按标题切分） | 了解切分策略即可，第 16 讲深入 |
| ★ 了解 | Output Parsers（StrOutputParser、PydanticOutputParser） | with_structured_output 的底层机制 |

## 本讲小结

- **Runnable 统一接口**：`invoke / stream / batch`，所有组件调用方式一致
- **LCEL 管道符** `|`：串联组件的语法糖，本项目因分支逻辑复杂选择不用
- **ChatOpenAI** 三种调用模式：invoke（完整）、stream（流式）、with_structured_output（Pydantic 约束）
- **Message 类型**映射 OpenAI API 的角色格式，多轮对话靠完整历史消息列表
- **ChatPromptTemplate** 用模板变量管理 System/User Prompt，支持 MessagesPlaceholder 动态历史
- **Output Parser** 将 AIMessage 转为纯文本或 Pydantic 对象
- **SQLChatMessageHistory** 一行代码替代手写 CRUD
- **Milvus VectorStore** 自动向量化 + BM25 + auto_id 更新机制
- **Document Loaders** 注册表模式管理文件格式，扩展只需一条注册
- **Text Splitters** 递归降级切分 + 父子块策略，优先在语义边界切分

**下一讲**：[Milvus 索引机制与基本操作](./04-milvus-index-and-operations.md) — 四种索引类型、pymilvus 基本操作、LangChain 底层隐藏逻辑
