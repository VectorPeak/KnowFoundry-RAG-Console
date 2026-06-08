"""Prompt 模板选择器：根据意图、问题类别和业务场景选择最终回答模板。

这个模块只做配置分发，不查数据库、不调用模型。这样 Prompt 策略可以独立调整，
不会把模板选择逻辑散落到 RAG 主链路中。
"""

from __future__ import annotations
from qa_core.config.settings import get_settings
from qa_core.intent.question_category import infer_question_category
from qa_core.prompts.profiles import CATEGORY_PROMPT_PROFILES, DEFAULT_PROMPT_PROFILE, PROMPT_PROFILES, PromptProfile
from qa_core.scenarios.registry import ScenarioDefinition

def _scenario_prompt_context(scenario: ScenarioDefinition | None = None) -> dict[str, str]:
    """把场景配置转换成 Prompt 模板需要的变量。

    参数：
        scenario: 当前业务场景；为空时使用通用默认值。

    返回：
        包含 assistant_name、business_domain、industry、support_contact、phone 的字典。
    """
    # 没有传入场景时使用全局配置作为兜底展示信息。
    settings = get_settings()
    return {
        "assistant_name": scenario.assistant_name if scenario else "知识助手",
        "business_domain": scenario.business_domain if scenario else "业务知识库",
        "industry": scenario.industry if scenario else "通用知识问答",
        "support_contact": scenario.support_contact if scenario else settings.customer_service_phone,
        "phone": scenario.support_contact if scenario else settings.customer_service_phone,
    }


def build_answer_prompt_profile(
    intent: str,
    scenario: ScenarioDefinition | None = None,
    query: str | None = None,
) -> PromptProfile:
    """根据意图和问题类别选择最终回答模板。

    选择优先级（命中即返回）：
      1. 风险类问题模板：费用、合规、故障、总结等类别优先使用专用模板。
      2. 意图专属模板：FAQ_QUERY、KNOWLEDGE_QUERY、GREETING 等。
      3. 默认模板：前两者都没有命中时使用通用回答模板。

    选中模板后，会把当前场景的助手名称、业务域、行业和联系方式填入 system_template。

    参数：
        intent: 意图识别结果，例如 FAQ_QUERY、GREETING。
        scenario: 当前业务场景，用于注入模板变量。
        query: 用户原始问题，用于判断风险类别。

    返回：
        已完成场景变量填充的 PromptProfile。
    """
    # 判断 RAG 回答风险类别
    question_category = infer_question_category(query or "")
    # 回答模板优先级：风险类别模板 > 意图专属模板 > 默认模板，确保敏感问题口径正确
    profile = CATEGORY_PROMPT_PROFILES.get(question_category) or PROMPT_PROFILES.get(intent, DEFAULT_PROMPT_PROFILE)
    context = _scenario_prompt_context(scenario)
    return PromptProfile(
        name=profile.name,
        system_template=profile.system_template.format(**context),
        user_template=profile.user_template,
        reason=profile.reason,
    )
