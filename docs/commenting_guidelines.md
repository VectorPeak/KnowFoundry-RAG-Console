# 后端代码注释规范

本项目后端注释必须使用中文，并且不能只写“这段代码做什么”。核心代码注释要尽量说明：

- 使用场景：什么时候会调用，服务于哪条链路。
- 采用原因：为什么选这个方案，而不是另一个方案。
- 边界限制：什么情况下不能这么用。
- 失效处理：依赖失败、配置变化、知识库更新时如何暴露问题。
- 与主链路关系：当前是否属于 `app.py + qa_core/api + qa_core` 主链路。

## 1. 推荐注释结构

函数或类的 docstring 推荐按下面结构写：

```python
def example():
    """一句话说明该函数的职责。

    使用场景：
    - 什么时候被调用；
    - 属于哪个链路；
    - 输入输出服务于哪个业务动作。

    为什么采用：
    - 说明设计取舍；
    - 说明为什么不用其他方案；
    - 说明成本、性能或一致性考虑。

    不适合的用法：
    - 哪些场景不要复用；
    - 哪些参数不能随便扩展；
    - 哪些状态不能缓存或共享。
    """
```

不是每个小函数都必须写满三段，但主链路、缓存、数据库、Milvus、LLM、入库脚本必须尽量写清楚。

## 2. `lru_cache` 注释要求

使用 `lru_cache` 时必须解释：

- 缓存的是什么对象。
- 为什么该对象适合进程内缓存。
- 缓存是否和用户问题、会话、历史、知识库版本有关。
- 如何清理或失效。
- 为什么不能用它缓存最终 RAG 答案。

当前允许使用 `lru_cache` 的对象类型：

- 配置对象。
- 服务单例。
- 模型客户端。
- embedding 模型。
- reranker 模型。
- Milvus store 封装。
- 历史和反馈存储适配器。

当前不允许直接使用 `lru_cache` 缓存：

- 用户最终答案。
- 流式 token。
- 某个 session 的历史上下文。
- Milvus 检索结果。
- rerank 分数。
- LLM invoke/stream 输出。

如果后续需要结果缓存，优先设计带 TTL、知识库版本、模型版本和 source_filter 的外部缓存，不要直接套 `lru_cache`。

## 3. 旧链路注释要求

旧版代码已经从工程目录中移除。新增代码不允许再恢复 `mysql_qa`、`rag_qa`、
`old_main.py`、`new_main.py` 这类旧入口；如果需要解释历史问题，只写入文档，不新增
可运行旧链路代码。

## 4. RAG 主链路注释要求

涉及以下模块时，注释必须说明业务原因：

- `qa_core/application/service.py`：为什么这个分支直接返回、为什么先 FAQ 后文档、为什么保存历史。
- `qa_core/retrieval/`：为什么使用 Milvus BM25、dense/sparse 权重、rerank、去重。
- `qa_core/indexing/`：为什么使用父子块、为什么增量入库、为什么 FAQ answer 放 metadata。
- `qa_core/intent/classifier.py`：为什么规则优先、什么时候用 LLM 结构化识别、为什么需要 source 推断。
- `qa_core/memory/history.py`：为什么使用摘要 + 最近消息、为什么 MySQL 只做历史和反馈。

## 5. 不推荐的注释

避免只写语法级注释：

```python
# 遍历列表
for item in items:
    ...
```

更推荐写业务目的：

```python
# 同一个 chunk 可能被多个查询变体命中，这里保留最高分，避免重复上下文污染生成。
for item in items:
    ...
```

## 6. 导入规范

后端 Python 文件的导入必须放在文件头部，不能在函数中临时导入。

这样做的原因：

- 文件依赖一眼可见，阅读代码时不用跳进函数内部才发现依赖；
- 缺失依赖会在启动、测试或脚本导入阶段立即暴露，不会拖到用户提问时才失败；
- 避免为了“先跑起来”在函数内部写 fallback 导入，破坏当前项目“环境是前置条件”的约束。

例外情况只允许非常明确的工程原因，例如类型检查专用导入、真正无法避免的循环导入；
如果出现这种情况，必须在代码注释里解释为什么不能放到头部。

当前要求：

- 不在函数内部写 `import xxx` 或 `from xxx import yyy`；
- 不写 `try import A except ImportError import B` 这类隐藏兼容分支；
- 依赖缺失时直接修正 `requirements.txt` 和运行环境。

## 7. 验收方式

补充注释后至少执行：

```powershell
python -m compileall app.py qa_core scripts
```

如果修改了主链路，还需要启动服务并检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

## 8. 本轮已覆盖的注释治理范围

本轮注释治理不是只补 `lru_cache`，已经扩展到主链路和关键脚本。

已补充“使用场景 + 为什么采用 + 不适合场景”的代码位置：

| 文件 | 覆盖点 | 说明 |
| --- | --- | --- |
| `app.py` | 应用创建、CORS、静态资源、startup preflight、router 注册 | 明确入口只负责启动和注册，不承载业务路由细节。 |
| `qa_core/api/` | 页面、WebSocket、debug、HTTP preview、管理接口、知识库版本接口 | 明确协议层职责、为什么复杂问题走 WebSocket、为什么 debug 不生成答案、为什么管理接口需要令牌。 |
| `qa_core/application/service.py` | QAService、stream_query、FAQ 直出、上下文构造 | 解释主 RAG 编排顺序、直接答案、FAQ 直出风险、父子块上下文选择。 |
| `qa_core/intent/classifier.py` | classify_intent、LLM 结构化意图 | 解释规则优先、LLM 结构化识别、source 推断边界。 |
| `qa_core/retrieval/strategy.py` | RetrievalPlan、build_retrieval_plan | 解释为什么采用确定性计划，而不是旧版 LLM 策略选择器。 |
| `qa_core/prompts/profiles.py` | PromptProfile、build_answer_prompt_profile | 解释为什么 FAQ、业务知识咨询、追问使用不同模板，以及为什么模板选择必须确定性。 |
| `qa_core/retrieval/` | embedding、reranker、Milvus store、BM25 | 解释运行时缓存、Milvus hybrid、rerank、为什么不缓存最终答案。 |
| `qa_core/indexing/` | DocumentLoaderSpec、loader 注册表、splitter、增量入库、FAQ CSV | 解释为什么用注册表替代 if/elif、为什么离线入库、为什么父子块、为什么 answer 放 metadata。 |
| `qa_core/memory/history.py` | HistoryStore 缓存和上下文 | 解释为什么历史存 MySQL、为什么摘要 + 最近消息，不缓存 session 上下文。 |
| `qa_core/memory/feedback.py` | FeedbackStore | 解释反馈只写 MySQL、为什么不直接影响在线答案。 |
| `qa_core/config/settings.py` | Settings 缓存 | 解释配置读取优先级和 cache_clear 场景。 |
| `qa_core/llm/client.py` | ChatOpenAI 客户端 | 解释流式/非流式客户端区分，禁止缓存 LLM 输出。 |
| `scripts/rebuild_kb_version.py` | 文档和 FAQ 入库脚本 | 解释为什么离线入库、source 白名单、force 场景，以及为什么 FAQ/文档必须进入同一个 kb_version。 |
| `scripts/evaluate_core_chain.py` | 轻量评测脚本 | 解释为什么直接调用 QAService、如何看工程回归信号。 |
| `scripts/check_langchain_stack.py` | 配置检查脚本 | 解释为什么做必需环境硬校验，但不做写入。 |

后续新增或修改这些文件时，应保持同等注释密度，特别是所有“会影响主链路行为”的判断分支，都要写清楚采用原因和不适用边界。
