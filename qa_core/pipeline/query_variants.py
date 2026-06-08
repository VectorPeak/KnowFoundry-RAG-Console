"""检索查询扩展工具：为同一检索意图生成少量同义检索表达（如"Webhook" → "回调"），不改变问题含义。
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from qa_core.config.logging_config import get_logger
from qa_core.prompts.constants import QUERY_VARIANT_SYSTEM_PROMPT
from qa_core.config.settings import get_settings
from qa_core.llm.client import get_chat_model
logger = get_logger(__name__)

class QueryVariants(BaseModel):
    """LLM 输出检索表达时使用的 Pydantic 结构化模型，避免模型输出解释性文本。
    """

    queries: list[str] = Field(default_factory=list, description="等价检索表达")


def generate_query_variants(query: str, *, enabled: bool) -> list[str]:
    """为同一检索意图生成少量同义表达（如"流程"→"SOP"），提升召回而不改变问题含义。
    """
    # 加载应用全局设置（retrieval_variant_max 等检索配置）
    settings = get_settings()
    cleaned = query.strip()
    # 功能禁用或无可变体空间时仅用原问题检索，避免无关变体稀释召回精度
    if not enabled or not cleaned or settings.retrieval_variant_max <= 0:
        return [cleaned]

    # 短结构化问题的常见同义说法已被启发式规则枚举，规则成本为零而 LLM 调用成本高
    if _looks_like_short_structured_question(cleaned):
        return [cleaned]

    # 第一层：先用确定性本地规则（零成本）为高频业务术语生成同义变体
    # 规则生成足够变体时直接返回，跳过第二层 LLM 调用，兼顾延迟与成本
    heuristic_variants = _heuristic_variants(cleaned, settings.retrieval_variant_max)
    if len(heuristic_variants) > 1:
        return heuristic_variants

    variants = [cleaned]
    # 第二层：规则未覆盖的新领域词或罕见表达回退 LLM 扩展，避免召回覆盖率因规则缺失而下降
    model = get_chat_model(streaming=False).with_structured_output(QueryVariants)
    # 调用 LLM 生成等价检索表达（如同义词、不同说法），不改变用户问题含义
    result = model.invoke(
        [
            SystemMessage(content=QUERY_VARIANT_SYSTEM_PROMPT),
            HumanMessage(content=f"原问题：{cleaned}\n最多生成 {settings.retrieval_variant_max} 条检索表达。"),
        ]
    )
    for item in result.queries:
        candidate = str(item).strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
        if len(variants) >= settings.retrieval_variant_max + 1:
            break
    return variants


def _heuristic_variants(query: str, max_extra: int) -> list[str]:
    """用本地确定性规则为高频业务知识说法生成同义变体（流程→SOP、告警→报警等）。
    """
    variants = [query]
    normalized = query.lower()

    def add(candidate: str) -> None:
        """在保持顺序和上限的前提下，追加非空不重复变体。"""
        candidate = candidate.strip()
        if candidate and candidate not in variants and len(variants) < max_extra + 1:
            variants.append(candidate)

    if "安装" in query or "失败" in query or "报错" in query:
        add(query.replace("失败", "报错"))
        add(query.replace("报错", "失败"))
    if "流程" in query:
        add(query.replace("流程", "SOP"))
        add(query.replace("流程", "处理步骤"))
    if "发票" in query:
        add(query.replace("发票", "开票"))
        add(query.replace("发票", "账单"))
    if "开票" in query:
        add(query.replace("开票", "发票"))
    if "告警" in query:
        add(query.replace("告警", "报警"))
        add(query.replace("告警", "异常"))
    if "webhook" in normalized:
        add(query.replace("webhook", "回调").replace("Webhook", "回调"))
        add(query.replace("Webhook", "Webhook 回调").replace("webhook", "Webhook 回调"))
    if "资料" in query:
        add(query.replace("资料", "材料"))
        add(query.replace("资料", "记录"))
    if "材料" in query:
        add(query.replace("材料", "资料"))
    if "流程" in query and "怎么走" in query:
        add(query.replace("怎么走", "有哪些步骤"))
        add(query.replace("流程", "办理流程"))
    if "怎么排查" in query:
        add(query.replace("怎么排查", "如何处理"))
        add(query.replace("怎么排查", "处理步骤"))
    if "能不能" in query:
        add(query.replace("能不能", "是否可以"))
    if "可以吗" in query:
        add(query.replace("可以吗", "是否可以"))
    return variants


def _looks_like_short_structured_question(query: str) -> bool:
    """判断问题的常见同义说法是否已被启发式规则覆盖，无需进一步 LLM 扩展。
    """
    compact = query.strip()
    if not compact or len(compact) > 24:
        return False
    return any(
        marker in compact
        for marker in (
            "怎么走",
            "资料",
            "材料",
            "怎么排查",
            "怎么处理",
            "需要哪些",
            "能不能",
            "可以吗",
            "是什么",
            "要看什么",
        )
    )

