"""结合历史的查询改写：将依赖上下文的追问改写成独立检索问题（如"那审批呢" → "入职流程中的审批步骤是什么"）。
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from qa_core.memory.history import format_messages
from qa_core.llm.client import get_chat_model
from qa_core.prompts.constants import REWRITE_SYSTEM_PROMPT

def rewrite_query_if_needed(query: str, history_messages, should_rewrite: bool) -> str:
    """将依赖上下文的追问（如"那审批呢"）改写成独立检索问题，确保检索不丢失对话意图。
    """
    # 无需改写或无历史上下文时跳过 LLM 调用，直接使用原问题检索，避免语义漂移
    if not should_rewrite or not history_messages:
        return query
    # 只取最近 8 条：过长历史会稀释当前提问焦点，改写只需最近一轮对话的前置信息
    history_text = format_messages(history_messages[-8:])
    # 非流式保证获得完整改写结果，流式片段拼接易导致句式断裂
    llm = get_chat_model(streaming=False)
    # 调用 LLM 将追问改写为独立检索问题（如"那审批呢"→"入职流程中的审批步骤"）
    response = llm.invoke(
        [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"对话历史：\n{history_text}\n\n当前问题：{query}\n\n改写后的检索问题："),
        ]
    )
    rewritten = str(response.content).strip()
    # 改写为空说明 LLM 未能理解上下文，硬失败暴露问题而非静默回退原始查询
    if not rewritten:
        raise RuntimeError("查询改写返回空结果，无法生成独立检索问题。")
    return rewritten

