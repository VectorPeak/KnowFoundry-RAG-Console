# 第2讲 笔记：RAG 核心概念深入

## 一、学习目标（可验证）

- [ ] 能用自己的话解释 Embedding 是什么，为什么"语义相近 = 向量距离近"
- [ ] 能手算余弦相似度，说出余弦相似度、欧几里得距离、内积三种计算方式的区别
- [ ] 能解释 Dense 检索（语义）和 Sparse 检索（关键词 BM25）各自的优势和局限
- [ ] 能说明 Hybrid Search（混合检索）为什么要把 Dense + Sparse 结合起来
- [ ] 能画出 Bi-Encoder（粗排）→ CrossEncoder（精排）的两阶段检索链路
- [ ] 能解释 Reranker 为什么比向量相似度更准，以及为什么不能对所有文档都用 Reranker
- [ ] 能回答面试题：Dense 检索和 Sparse 检索的区别是什么？为什么要重排？

## 二、知识点详解 + 可执行代码

### 2.1 Embedding：文本 → 向量

**是什么**：Embedding（嵌入）是将非结构化文本转换成固定长度的浮点数向量的技术。本项目使用 BGE-M3 模型，将任意长度的中文或英文文本转成 1024 维向量。这个向量在数学空间中代表了文本的"语义位置"——语义相近的文本对应的向量在空间中距离更近。

**为什么需要**：计算机只能做数值计算，必须先让文字变成数字才能比较相似度。传统关键词匹配（如 TF-IDF）只统计词频，无法区分"我很开心"和"我非常高兴"是同一意思，也无法区分"银行利率很高"和"河边的银行很美"中"银行"的不同含义。Embedding 基于 Transformer 架构做上下文语义理解，解决了这个问题。

**怎么用**：

```python
# 运行：python notes/01/2.1_cosine_similarity.py（复用第 1 讲代码）
import numpy as np

# 两个语义相近的句子的向量（模拟 BGE-M3 输出）
A = np.array([0.5, 0.3, 0.8, 0.1, 0.6])  # "入职需要什么材料"
B = np.array([0.48, 0.32, 0.79, 0.09, 0.58])  # "入职要准备哪些文件"

# 余弦相似度
cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
# 结果 ≈ 0.95（很高 → 语义相近）

# 两个语义不同的句子
C = np.array([-0.3, 0.7, -0.2, 0.5, -0.1])  # "今天天气怎么样"
cosine_sim_ac = np.dot(A, C) / (np.linalg.norm(A) * np.linalg.norm(C))
# 结果 ≈ 0.1（很低 → 语义无关）
```

### 2.2 三种向量相似度计算方式

**是什么**：衡量两个向量"有多近"的三种常用方法：

| 方法 | 公式 | 取值范围 | 特点 |
|------|------|---------|------|
| **余弦相似度** | dot(A,B) / (\|\|A\|\|×\|\|B\|\|) | [-1, 1] | 只看方向不看长度，最常用于文本 |
| **欧几里得距离** | √Σ(Aᵢ-Bᵢ)² | [0, +∞) | L2 归一化后等价于余弦 |
| **内积（点积）** | Σ(Aᵢ×Bᵢ) | (-∞, +∞) | Milvus 默认，计算最快 |

**为什么需要**：不同场景选不同方式。文本语义比较用余弦（不考虑文档长度），经 L2 归一化后的向量用内积（等价于余弦但计算更快）。本项目 BGE-M3 默认开启 `normalize_embeddings=True`（L2 归一化），所以 Milvus 用内积即可。

**怎么用**：

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def inner_product(a, b):
    return np.dot(a, b)
```

### 2.3 BGE-M3 模型

**是什么**：由北京智源研究院（BAAI）开发的多语言 Embedding 模型，部署在 `models/bge-m3/` 目录下，通过 LangChain 的 HuggingFaceEmbeddings 接口调用。

**为什么需要**：三大特性直接支撑本项目需求：

| 特性 | 说明 | 对本项目的价值 |
|------|------|---------------|
| **多语言** | 支持中英双语及 100+ 语言 | 企业场景中英文混合文档都能处理 |
| **多粒度** | 短句到长文档（最多 8192 token） | FAQ 短句和长篇制度文档用同一模型 |
| **多功能** | 同时支持 Dense + Sparse 向量 | 一个模型产出稠密和稀疏两种向量，不用部署两套 |

**怎么用**：

```python
# qa_core/retrieval/models.py — 进程级单例加载（lru_cache 避免重复加载数百 MB 权重）
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

@lru_cache(maxsize=1)  # 整个进程只加载一次
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="models/bge-m3",
        model_kwargs={"device": "cpu"},       # 或 "cuda"
        encode_kwargs={"normalize_embeddings": True},  # L2 归一化，后续直接用内积
    )
```

> 完整源码：`qa_core/retrieval/models.py` → `get_embeddings()`

### 2.4 向量数据库：为什么是 Milvus

**是什么**：专门为十亿级向量相似度搜索设计的数据库。区别于传统数据库（如 PostgreSQL + pgvector 插件），Milvus 在向量检索性能和索引类型上有数量级的优势。

**为什么需要**：

| 对比维度 | pgvector | Milvus 2.6 |
|---------|----------|------------|
| 百万级向量检索 | 秒级 | 毫秒级 |
| 索引类型 | IVFFlat, HNSW | HNSW, IVF, DiskANN 等 11 种 |
| 混合检索 (Dense+Sparse) | 不支持 | 原生支持 |
| 分布式 | 需额外配置 | 原生支持 |

本项目每个业务场景数千条 FAQ + 数万文档 chunk，要求 Dense 向量检索 + BM25 关键词检索在一次查询中同时完成，Milvus 的 Hybrid Search 是最合适的选择。如需了解 Milvus 索引类型的图解和 pymilvus 实操，请阅读 [第4讲：Milvus 索引机制与基本操作](../04-milvus-index-and-operations.md)。

**Milvus 核心概念**：

| 概念 | 类比 | 说明 |
|------|------|------|
| Collection | 数据库表 | 一组相同结构的数据 |
| Field | 列 | 主键(pk)、向量字段(dense/sparse)、文本字段(text)、标量字段(source/kb_version) |
| Index | 索引 | 加速向量检索的数据结构，本项目使用 HNSW |
| Partition | 分区 | Collection 的物理分片，提升查询效率 |

### 2.5 Dense 检索 vs Sparse 检索（核心）

**是什么**：两种互补的检索方式。

- **Dense 检索**（稠密/语义检索）：用 Embedding 模型把问题和文档都转成向量，算余弦相似度。理解语义，能识别同义词和改写。
- **Sparse 检索**（稀疏/关键词检索）：基于 BM25 算法的词频-逆文档频率匹配，对关键词做精确匹配。不理解语义，但精确术语命中率极高。

**为什么需要**：举例说明——

```
知识库中有：
  D1："忘记密码时，可以通过绑定的邮箱或手机号自助重置"
  D2："管理员可以在后台重置任何用户的密码"
  D3："Webhook 回调地址配置在系统设置-集成管理页面"
  D4："API 密钥在个人设置-安全页面中生成和管理"

用户问："我怎么修改自己的登录密码"

Dense 排序：D1(0.91) > D2(0.72) > D4(0.22) > D3(0.15) ✅ 语义理解正确
Sparse 排序：可能因"密码"词频把 D2 排第一                     ⚠️ 不理解"修改"≈"重置"≈"忘记"

但另一个场景——
用户问："API 密钥在哪里生成"

Dense：可能把所有和 API 相关的都召回        ⚠️ 难以精确定位
Sparse：精确匹配"API""密钥""生成"→ D4命中  ✅ 精确匹配
```

**结论**：Dense 擅长"意思相近"，Sparse 擅长"词精确匹配"。二者互补。

**Hybrid Search = Dense + Sparse**：

```python
# qa_core/retrieval/store.py — 混合检索核心（★★★）
# Milvus 内部自动融合稠密和稀疏两路召回结果
kwargs = {
    "ranker_type": "weighted",
    "ranker_params": {"weights": [0.55, 0.45]},  # Dense 权重略高于 Sparse
}
raw_hits = self.store.similarity_search_with_score(
    query, k=top_k, expr=filter_expression, **kwargs
)
# 结果：既有语义覆盖（同义词、改写），又有精确匹配（编号、术语）
```

> 完整源码：`qa_core/retrieval/store.py` → `MilvusHybridStore.search()`

### 2.6 Reranker（重排器）：从粗排到精排

**是什么**：检索分两阶段——第一阶段用 Embedding 做"粗排"（从海量文档中快速圈定候选，Top 20~50），第二阶段用 Reranker 做"精排"（对候选逐一精细打分并重新排序）。Reranker 使用 CrossEncoder 架构（交叉编码器），将问题和文档拼接后一起编码，通过交叉注意力获得比向量相似度更精确的相关性判断。

**为什么需要**：Embedding 检索是**双塔架构**（Bi-Encoder）——问题和文档分别编码再算相似度，速度快但精度有限（问题和文档之间没有交互）。对 Top 20~50 候选全部用 CrossEncoder 显然太慢，但只对 Top K 候选（20~50 条）做就很快。这就是"先粗排后精排"的两阶段策略。

```
两阶段架构：
  阶段一（Bi-Encoder 粗排）:
    问题 → Encoder → Q_vec ─┐
                              ├→ 余弦相似度 → 取 Top 30（快但粗）
    文档 → Encoder → D_vec ─┘

  阶段二（CrossEncoder 精排）:
    [问题 + 文档₁] → CrossEncoder → 分数₁
    [问题 + 文档₂] → CrossEncoder → 分数₂    （慢但准，只对 30 条做）
    ...
    按新分数重排 → 取 Top 5 进入 LLM 上下文

实际案例：
  用户问："入职时需要提交哪些材料"
  Milvus 召回："离职需要提交离职申请表..." (向量相似度 0.78，排第二)
  重排后：     "离职..." → rerank 分数 0.15，被排到最后 ✅
              "入职当天携带身份证复印件..." → 0.94，排第一    ✅
```

**怎么用**：

```python
# qa_core/retrieval/ranking.py — CrossEncoder 重排
def rerank_hits(query, hits, *, reranker, top_n):
    """对 Milvus 召回的候选用 CrossEncoder 逐一重新打分"""
    if not hits:
        return []
    # 构建 [query, document] 对
    pairs = [(query, hit.document.page_content) for hit in hits]
    # CrossEncoder 交叉注意力打分 — 比向量余弦距离更精确
    scores = reranker.predict(pairs)
    # 按新分数排序并截断
    reranked = [
        RetrievalHit(document=hit.document, score=float(score))
        for hit, score in sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
    ]
    return reranked[:top_n]
```

> 完整源码：`qa_core/retrieval/ranking.py` → `rerank_hits()`；模型加载：`qa_core/retrieval/models.py` → `get_reranker()`

### 2.7 Milvus 部署架构

**是什么**：Milvus Standalone 由三个容器组成——etcd（元数据存储）、MinIO（索引文件存储）、Milvus（核心引擎）。

```
docker-compose up
  ├── etcd   → 分布式 KV 存储：Collection 定义、索引状态、节点信息
  ├── MinIO  → S3 兼容对象存储：向量索引文件、binlog
  └── Milvus → 核心引擎：向量写入、索引构建、相似度搜索
```

**为什么需要**：etcd 和 MinIO 是 Milvus 的存储层，将元数据和索引文件外置，这样 Milvus 本身可以无状态水平扩展。开发环境三个容器一起启动，生产环境可以替换为高可用的 etcd 集群和 MinIO 集群。

## 三、本讲核心代码串联

```python
# 完整检索链路演示：问题 → Embedding → 混合检索 → Reranker → 结果
# 以下代码提炼自 qa_core/retrieval/store.py + ranking.py + models.py

from qa_core.retrieval.models import get_embeddings, get_reranker

# 1. 加载模型（进程级单例，只加载一次）
embedding_model = get_embeddings()   # BGE-M3，1024 维 Dense 向量
reranker = get_reranker()            # BGE Reranker Large，CrossEncoder

# 2. 用户问题 → 向量
query = "入职需要提交哪些材料"
query_vector = embedding_model.embed_query(query)
# → [0.023, -0.451, 0.782, ..., 0.134]（1024 维）

# 3. Milvus 混合检索（Dense + Sparse 两路召回，加权融合）
# store.search() 内部：
#   - Dense: query_vector 在 Milvus 中做 ANN 搜索（语义匹配）
#   - Sparse: Milvus BM25 做关键词匹配（精确术语）
#   - ranker_type="weighted", weights=[0.55, 0.45]
#   → 返回 Top 30 候选文档

# 4. Reranker 精排
# ranking.rerank_hits() 对 Top 30 构造 [query, doc] 对
# → CrossEncoder 逐一打精细分 → 按新分数重排 → 取 Top 5

# 5. Top 5 文档 + 原始问题 → LLM 生成最终答案
```

> 完整源码：`qa_core/retrieval/store.py` → `MilvusHybridStore.search()`（约 70 行，含完整流程和异常处理）

## 四、常见面试题

**Q1：Dense 检索和 Sparse 检索的核心区别？各自适合什么场景？**
> Dense（稠密/语义检索）：用 Embedding 模型将文本转成向量，算余弦相似度。**理解语义**，能识别同义词和改写（"修改密码"能命中"忘记密码怎么办"）。适合自然语言问答、长文本语义匹配。局限是对精确术语（编号、代码）区分能力弱。
>
> Sparse（稀疏/关键词检索）：基于 BM25 词频统计，做关键词精确匹配。**不理解语义**但精确术语命中率极高（"HS编码 8471.30"精确命中）。适合编号查询、法规条文搜索。局限是无法识别同义词。
>
> 实际项目中二者互补，混合检索取长补短。

**Q2：为什么要用 Reranker？为什么不直接用 Milvus 的向量分数？**
> Embedding 检索是 Bi-Encoder（双塔架构）：问题和文档分别编码，没有交叉注意力，精度有限。典型问题：语义相近但答非所问的文档被排到前面（"离职需要什么材料"在检索"入职需要什么材料"时向量相似度可能也很高）。
>
> Reranker 使用 CrossEncoder（交叉编码器）：将问题和文档**拼接后一起编码**，通过交叉注意力机制获得精确的相关性判断，能区分"入职"和"离职"。代价是计算量大，所以只对 Milvus 召回的 Top 20~50 条候选做，而不是对所有文档做。

**Q3：BGE-M3 的核心特性是什么？为什么选它？**
> 三个核心特性：① **多语言**——支持中英双语及 100+ 语言，适合企业混合文档场景；② **多粒度**——短句到长文档（8192 token）都支持，FAQ 和制度文件用同一模型；③ **多功能**——同时输出 Dense 向量（用于语义检索）和 Sparse 向量（用于关键词检索），一个模型覆盖两种检索方式。此外，BGE-M3 可在本地 GPU/CPU 运行，不需要调用外部 API，数据安全有保障。

**Q4：余弦相似度和内积有什么区别？为什么 Milvus 默认用内积？**
> 余弦相似度 = 内积 / (||A|| × ||B||)，即对向量长度做了归一化，只看方向。内积 = Σ(Aᵢ×Bᵢ)，同时受方向和长度影响。
>
> 但 BGE-M3 输出时已经做了 L2 归一化（`normalize_embeddings=True`），使所有向量长度都是 1。L2 归一化后，`||A|| = ||B|| = 1`，内积在数学上等价于余弦相似度，而内积计算少一次开方运算，性能更好。

**Q5：Hybrid Search 中 Dense 和 Sparse 的权重怎么定？**
> 本项目 Dense 权重 0.55，Sparse 权重 0.45。Dense 略高是因为大多数企业场景的问答偏向语义理解（"入职流程"和"新员工报到"是同义词），但 Sparse 的 0.45 权重确保精确术语（如编号、法律条文编号）不会被完全淹没。这个比例是可调的，不同场景可以有不同的权重配置。

## 五、本讲速查卡

| 术语 | 一句话定义 |
|------|-----------|
| Embedding | 文本→固定长度浮点数向量，语义相近的文本向量距离也近 |
| BGE-M3 | BAAI 开发的多语言 Embedding 模型，支持 Dense + Sparse 双向量输出，1024 维 |
| L2 归一化 | 将向量长度缩放到 1，使内积等价于余弦相似度 |
| 余弦相似度 | 衡量两个向量方向接近程度，文本语义比较最常用的指标 |
| 内积 | Milvus 默认相似度计算方式，L2 归一化后等价于余弦相似度 |
| Milvus | 开源向量数据库，原生支持 Dense + Sparse 混合检索，毫秒级十亿向量搜索 |
| Collection | Milvus 中的"表"，存储一组相同结构的数据 |
| HNSW | 基于图的向量索引结构，Milvus 默认索引，适合高召回率场景 |
| Dense 检索 | 语义检索：向量相似度匹配，理解语义和同义词 ← Embedding 模型 |
| Sparse 检索 | 关键词检索：BM25 词频匹配，精确术语命中率极高 ← Milvus BM25 |
| Hybrid Search | Dense + Sparse 加权融合，取长补短：既有语义覆盖又有精确匹配 |
| Bi-Encoder | 双塔架构：问题/文档分别编码再算相似度 → 快但粗 ← 检索阶段 |
| CrossEncoder | 交叉编码器：[问题+文档]拼接编码 → 慢但准 ← 重排阶段 |
| Reranker | 二阶段精排模型（BGE Reranker Large），对粗排候选重新打分排序 |
| 粗排→精排 | 检索标准架构：Bi-Encoder 快速圈定候选（20~50 条）→ CrossEncoder 精细打分取 Top 5 |
| etcd | Milvus 的元数据存储引擎（分布式 KV） |
| MinIO | Milvus 的索引文件存储引擎（S3 兼容对象存储） |
