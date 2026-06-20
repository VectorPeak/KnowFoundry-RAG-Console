# 附录A：Pydantic 数据校验

## 为什么需要这讲

Pydantic 在本项目中被**五处关键位置**使用，但主讲义从未系统讲解它是什么、怎么工作：

| 使用位置 | 作用 | 代码 |
|---------|------|------|
| 全局配置 | 从 `.env` 加载并校验配置 | `Settings(BaseSettings)` |
| API 请求校验 | 自动校验 JSON 请求体的字段和类型 | `QueryRequest(BaseModel)` |
| LLM 结构化输出 | 限制 LLM 只返回枚举字段 | `IntentLLMDecision(BaseModel)` |
| 检索计划 | 序列化检索参数供诊断使用 | `RetrievalPlan` dataclass |
| 数据隔离 | 封装多租户参数 | `DataScope` dataclass |

## 一、Pydantic 是什么

**Pydantic** 是 Python 最流行的数据校验库。它的核心思想：**用 Python 类型注解定义数据结构，运行时自动校验类型**。

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# 正确数据
user = User(name="张三", age=28, email="zhangsan@example.com")

# 错误的类型 → 自动报错
user = User(name="张三", age="二十八", email="zhangsan@example.com")
# ValidationError: age: Input should be a valid integer
```

对比手写校验：

```python
# 手写校验（容易遗漏、重复代码多）
def create_user(data):
    if "name" not in data:
        raise ValueError("name is required")
    if not isinstance(data["name"], str):
        raise ValueError("name must be string")
    if "age" not in data:
        raise ValueError("age is required")
    # ... 20 行校验代码 ...
    return {"name": data["name"], "age": int(data["age"])}

# Pydantic（一行声明，自动校验）
class User(BaseModel):
    name: str
    age: int
```

## 二、BaseModel 核心能力

### 2.1 基础类型校验

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str                      # 必填，字符串
    session_id: str | None = None   # 可选，默认 None
    scenario_id: str | None = None
    source_filter: str | None = None
    tenant_id: str = "default"      # 可选，默认值 "default"
```

当 FastAPI 收到一个 JSON 请求体时：
```json
{"query": "入职流程", "tenant_id": 123}
```
Pydantic 会自动校验：
- `query` 是 `str` ✅
- `tenant_id` 是 `int` 但期望 `str` ❌ → 返回 422 错误，附带清晰描述

### 2.2 Field 约束

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(
        ...,                      # ... 表示必填
        min_length=1,             # 至少 1 个字符
        max_length=2000,          # 最多 2000 字符
        description="用户问题"     # 用于生成 OpenAPI 文档
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
    )
    tenant_id: str = Field(
        default="default",
        pattern=r"^[a-z0-9_]+$",  # 只允许小写字母+数字+下划线
    )
```

Field 支持的常用约束：
| 约束 | 说明 |
|------|------|
| `min_length` / `max_length` | 字符串长度范围 |
| `ge` / `le` / `gt` / `lt` | 数值范围（≥/≤/>/<） |
| `pattern` | 正则表达式匹配 |
| `description` | OpenAPI 文档描述 |
| `examples` | OpenAPI 示例值 |

### 2.3 嵌套模型

```python
class DataScope(BaseModel):
    tenant_id: str = "default"
    dataset_id: str = "default"

class QueryRequest(BaseModel):
    query: str
    data_scope: DataScope  # 嵌套模型

# JSON 请求
{
    "query": "入职流程",
    "data_scope": {
        "tenant_id": "company_a",
        "dataset_id": "production"
    }
}
```

## 三、BaseSettings — 环境变量管理

这是 Pydantic 的一个特殊子类，专门用于管理配置：

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 自动从环境变量或 .env 文件读取
    llm_api_key: str = ""
    milvus_uri: str = "http://127.0.0.1:19530"
    mysql_port: int = 3306
    api_rate_limit_per_minute: int = 120

    model_config = {
        "env_file": ".env",        # 读取 .env 文件
        "env_file_encoding": "utf-8",
    }

# 使用
settings = Settings()
print(settings.milvus_uri)  # 优先环境变量，其次 .env，最后默认值

# 优先级
# 1. 环境变量 export MILVUS_URI=http://prod:19530
# 2. .env 文件中的 MILVUS_URI=http://prod:19530
# 3. 代码中的默认值 "http://127.0.0.1:19530"
```

本项目中的 Settings 实例被 `@lru_cache` 缓存为全局单例：

```python
# qa_core/config/settings.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# 任何模块导入即可获取同一份配置
from qa_core.config.settings import get_settings
settings = get_settings()  # 整个进程只加载一次 .env
```

## 四、with_structured_output — LLM 输出约束

这是本项目中 Pydantic 最高级的用法：让 LLM 按指定结构返回结果。

```python
from pydantic import BaseModel, Field

class IntentLLMDecision(BaseModel):
    intent: str = Field(description="用户问题意图")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    reason: str = Field(default="")

# 使用 LangChain 的 with_structured_output
model = ChatOpenAI(...).with_structured_output(IntentLLMDecision)
decision = model.invoke([...])  # 返回 IntentLLMDecision 对象

# 不是返回 "用户的意图是FAQ_QUERY，置信度0.82"
# 而是返回 IntentLLMDecision(intent="FAQ_QUERY", confidence=0.82, reason="...")
```

**工作原理**：LangChain 将 Pydantic 模型的 JSON Schema 嵌入 System Prompt，告诉 LLM 只能用这些字段和枚举值来回答。LLM 返回的 JSON 会被 Pydantic 自动校验——如果 LLM 返回了 `{"intent": "INVALID_TYPE"}`，Pydantic 会报错。

## 五、在本项目中的使用速查

| Pydantic 类 | 定义位置 | 作用 |
|-------------|---------|------|
| `Settings(BaseSettings)` | `qa_core/config/settings.py` | 读取 .env 配置 |
| `QueryRequest(BaseModel)` | `qa_core/schemas.py` | API 请求体校验 |
| `FeedbackRequest(BaseModel)` | `qa_core/schemas.py` | 反馈请求校验 |
| `IntentLLMDecision(BaseModel)` | `qa_core/intent/classifier.py` | LLM 意图输出约束 |

## 小结

- **BaseModel** = 类型注解 + 自动校验，替代手写 if/else 校验
- **Field** = 为字段附加约束（长度、范围、正则）
- **BaseSettings** = 自动从环境变量和 .env 加载配置
- **with_structured_output** = 用 Pydantic Schema 约束 LLM 输出格式
