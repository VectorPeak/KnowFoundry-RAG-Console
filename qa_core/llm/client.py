"""LLM 客户端工厂：通过 OpenAI 兼容接口创建 ChatOpenAI 实例。

切换模型时只改配置，不改业务代码。当前项目优先使用 LangChain 的 ChatOpenAI
兼容层接入 DashScope 等模型服务，避免在主链路里绑定某个厂商 SDK。

示例：
    get_chat_model(streaming=False) -> 非流式 ChatOpenAI 客户端
    get_chat_model(streaming=True)  -> 流式 ChatOpenAI 客户端
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from qa_core.config.settings import get_settings

@lru_cache(maxsize=2)  # 缓存流式/非流式两个客户端实例，避免每次重建 TCP 连接
def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    """返回带缓存的 ChatOpenAI 客户端。

    `streaming` 是缓存 key 的一部分，所以流式和非流式客户端会分开缓存。
    `maxsize=2` 正好覆盖这两种模式，避免重复创建连接，也避免缓存无限增长。

    参数：
        streaming: 是否返回支持流式 token 输出的客户端。

    返回：
        已按全局配置初始化好的 ChatOpenAI 实例。
    """
    # 加载全局设置
    settings = get_settings()
    # 构造 ChatOpenAI 客户端
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        streaming=streaming,
    )

def validate_llm_connectivity() -> None:
    """启动时真实调用一次模型，验证 LLM 服务可用。

    执行流程：
      1. 通过 get_chat_model() 获取非流式 ChatOpenAI 客户端。
      2. 发送最小化的“回复 OK”测试消息。
      3. 检查模型返回内容是否非空。
      4. 如果调用失败或返回空内容，抛出 RuntimeError，让启动阶段暴露配置问题。

    异常：
        RuntimeError: LLM 服务不可达、配置错误或返回空内容。
    """
    try:
        # 启动时发送最小测试请求验证 LLM 服务可用，防止上线后才发现配置错误
        response = get_chat_model(streaming=False).invoke([HumanMessage(content="回复 OK")])
    except Exception as exc:
        raise RuntimeError("LLM 服务不可用：请检查 DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL 和 LLM_MODEL。") from exc
    if not str(getattr(response, "content", "") or "").strip():
        raise RuntimeError("LLM 服务返回空内容：请检查模型服务状态。")

