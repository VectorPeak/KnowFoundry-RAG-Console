# 第1讲 笔记：项目概述与环境搭建

## 一、学习目标（可验证）

- [ ] 能用自己的话解释 RAG（检索增强生成）是什么、解决什么问题
- [ ] 能画出 RAG 系统的离线链路和在线链路，说出每个环节的作用
- [ ] 能手工计算两个向量的余弦相似度，解释"语义相近 = 向量接近"
- [ ] 能跑通 `2.2_verify_env.py`，确认 LLM 连通
- [ ] 能回答面试题：RAG 和 ChatGPT 的区别是什么

## 二、知识点详解 + 可执行代码

### 2.1 向量和向量检索

**是什么**：Embedding 模型把文本转成固定长度的浮点数列表（向量）。语义相近的文本，向量在空间中距离更近。RAG 用这个特性做"语义搜索"——不是按关键词精确匹配，而是按"意思相近"来检索。

**为什么需要**：用户问"入职需要什么材料"，知识库里写的是"入职流程包括提交身份证和学历证明"，关键词不完全匹配但意思一样。关键词搜索会漏掉，向量搜索能命中。

**怎么用**：

```python
# 运行：python notes/01/2.1_cosine_similarity.py
import math

def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b)

# 猫 ↔ 小猫  → 0.98（近义词，相似度接近 1）
# 猫 ↔ 汽车  → 0.32（无关词，相似度低）
```

> 完整可运行代码：`notes/01/2.1_cosine_similarity.py`

### 2.2 环境验证

**是什么**：检查 Python 版本、依赖包、LLM 连通性的脚本。每次搭建新环境后第一件事就是跑这个。

**为什么需要**：避免在环境坏了的情况下盲目调试业务代码。本项目在启动时会自动执行更严格的前置校验（preflight），这里做一个简化版帮助理解。

**怎么用**：

```python
# 运行：python notes/01/2.2_verify_env.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",
)
response = model.invoke([HumanMessage(content="回复 OK")])
# 期望：成功返回非空内容（如 "OK"）
```

> 完整可运行代码：`notes/01/2.2_verify_env.py`

### 2.3 RAG 系统组成（离线 + 在线双链路）

**是什么**：RAG 系统由两条链路组成：

- **离线链路**（入库）：文档加载 → 切分 → 向量化 → 存入向量数据库
- **在线链路**（问答）：用户提问 → 意图识别 → 语义检索 → 重排序 → 上下文构建 → LLM 生成

**为什么需要**：离线链路让知识"可检索"，在线链路让检索结果"可回答"。二者配合实现"基于最新文档的准确回答"。

**怎么用**：`notes/01/section3_mini_rag.py` 用 3 篇模拟文档演示完整的"检索→增强→生成"流程。即使没搭完 Milvus 环境，这段代码也能帮你建立 RAG 的直觉。

### 2.4 项目技术栈

| 层级 | 技术 | 作用 |
|------|------|------|
| LLM | DashScope (qwen-plus) | 大语言模型，OpenAI 兼容接口 |
| 向量数据库 | Milvus 2.6 | 存储向量，支持 Dense+Sparse 混合检索 |
| Embedding | BGE-M3 | 文本转向量，1024 维 |
| Reranker | BGE Reranker | 对检索结果精排 |
| Web 框架 | FastAPI | HTTP + WebSocket 双通道 |
| 编排 | LangChain | ChatOpenAI、VectorStore、MessageHistory 封装 |
| 存储 | MySQL | 聊天历史、会话摘要、用户反馈 |

### 2.5 八大业务场景

项目模拟 8 个企业场景：HR 制度、IT 支持、行政报销、采购流程、法务合规、财务流程、销售管理、项目管理。每个场景独立的知识库和 FAQ，通过 `scenario.toml` 配置切换——**不是 8 套代码，是 1 套引擎 × 8 套数据**。

## 三、本讲核心代码串联

```python
# 运行：python notes/01/section3_mini_rag.py
# 完整演示：知识库 → 提问 → 向量检索 → LLM 增强回答

knowledge_base = [...]     # 3 篇模拟文档，每篇有 content + vector
query = "入职需要什么材料"

# 检索：计算相似度，找最匹配的文档
scored = [(cosine_similarity(query_vector, doc["vector"]), doc) for doc in knowledge_base]
top_doc = max(scored, key=lambda x: x[0])[1]

# 增强生成：检索到的文档 + 问题 → LLM
messages = [
    SystemMessage(content="你是企业知识助手。严格根据参考资料回答。"),
    HumanMessage(content=f"参考资料：{top_doc['content']}\n问题：{query}"),
]
response = model.invoke(messages)
```

## 四、常见面试题

**Q1：RAG 和 ChatGPT 的核心区别？**
> ChatGPT 只靠模型参数中记住的知识回答——知识截止日期固定、无法覆盖私有数据、可能产生幻觉。RAG 回答前先从外部知识库检索相关文档，让 LLM "开卷作答"。答案可溯源、知识可实时更新、幻觉大幅减少。

**Q2：向量检索和传统关键词检索的区别？**
> 关键词检索（如 Elasticsearch、BM25）按词的精确匹配或词频统计。"入职材料"和"提交身份证"用词不同，关键词匹配会漏掉。向量检索把文本转成向量按语义匹配——意思相近的词向量也相近，能命中。

**Q3：RAG 系统的两条链路分别做什么？**
> 离线链路（入库）：文档加载→切分→向量化→存储。在线链路（问答）：提问→意图识别→语义检索→重排序→上下文构建→LLM生成。二者通过向量数据库连接。

## 五、本讲速查卡

| 术语 | 一句话定义 |
|------|-----------|
| RAG | 检索增强生成：先检索知识库，再用 LLM 基于结果生成答案 |
| Embedding | 文本转固定长度向量，语义相近的文本向量也相近 |
| 余弦相似度 | 衡量两个向量方向接近程度，范围 [-1, 1] |
| Milvus | 开源向量数据库，支持 Dense+Sparse 混合检索 |
| 离线链路 | 入库流程：加载→切分→向量化→存储 |
| 在线链路 | 问答流程：提问→意图→检索→重排→上下文→生成 |
| DashScope | 阿里云百炼平台，通过 OpenAI 兼容接口调用 qwen-plus |
