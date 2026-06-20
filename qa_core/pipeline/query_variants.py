"""检索查询扩展工具：为同一检索意图生成少量同义检索表达（如"Webhook" → "回调"），不改变问题含义。
"""

from __future__ import annotations
import re
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from qa_core.config.logging_config import get_logger
from qa_core.config.rules import QueryVariantReplacementRule, get_rule_config
from qa_core.prompts.constants import QUERY_VARIANT_SYSTEM_PROMPT
from qa_core.config.settings import get_settings
from qa_core.llm.client import get_chat_model
logger = get_logger(__name__)

FOLLOW_UP_REWRITE_MARKERS = ("追问：", "追问:")

class QueryVariants(BaseModel):
    """LLM 输出检索表达时使用的 Pydantic 结构化模型，避免模型输出解释性文本。
    """

    queries: list[str] = Field(default_factory=list, description="等价检索表达")


def generate_query_variants(query: str, *, enabled: bool, allow_short_structured: bool = False) -> list[str]:
    """为同一检索意图生成少量同义表达（如"流程"→"SOP"），提升召回而不改变问题含义。
    """
    # 加载应用全局设置（retrieval_variant_max 等检索配置）
    settings = get_settings()
    cleaned = query.strip()
    # 功能禁用或无可变体空间时仅用原问题检索，避免无关变体稀释召回精度
    if not enabled or not cleaned or settings.retrieval_variant_max <= 0:
        return [cleaned]

    # 普通短结构化问题保持克制；已追问改写的问题仍允许规则变体，避免上下文锚点丢失同义召回机会。
    if (
        _looks_like_short_structured_question(cleaned)
        and not allow_short_structured
        and not _is_rewritten_follow_up_query(cleaned)
    ):
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
    """用配置中的确定性规则为高频业务知识说法生成同义变体。"""
    variants = [query]
    rules = get_rule_config().query_variants

    def add(candidate: str) -> None:
        """在保持顺序和上限的前提下，追加非空不重复变体。"""
        candidate = candidate.strip()
        if candidate and candidate not in variants and len(variants) < max_extra + 1:
            variants.append(candidate)

    for rule in rules.replacements:
        if not rule.matches(query):
            continue
        for old, new in rule.replacements:
            add(_replace_term(query, old, new, rule))
    return variants


def _looks_like_short_structured_question(query: str) -> bool:
    """判断问题的常见同义说法是否已被配置规则覆盖，无需进一步 LLM 扩展。"""
    return get_rule_config().query_variants.is_short_structured_question(query)


def _is_rewritten_follow_up_query(query: str) -> bool:
    """判断是否为追问改写产物，例如"报销流程是什么；追问：那审批呢"。"""
    return any(marker in query for marker in FOLLOW_UP_REWRITE_MARKERS)


def _replace_term(query: str, old: str, new: str, rule: QueryVariantReplacementRule) -> str:
    """Apply one configured replacement, optionally case-insensitive."""

    if not rule.ignore_case:
        return query.replace(old, new)
    return re.sub(re.escape(old), new, query, flags=re.IGNORECASE)

