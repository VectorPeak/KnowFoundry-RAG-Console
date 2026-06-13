"""意图识别模块：在检索前决定问题应该走哪条处理路径。

它会判断当前问题是问候、追问、FAQ、文档知识查询、人工客服还是越界请求，并把结果
交给检索计划和 Prompt 模板使用。整体策略是“规则优先 + LLM 补充”：高频确定场景用
规则快速返回，模糊问题才调用 LLM，平衡速度、成本和准确性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.llm.client import get_chat_model
from qa_core.memory.history import format_messages
from qa_core.prompts.constants import INTENT_SYSTEM_PROMPT
from qa_core.scenarios.boundary import score_source_matches
from qa_core.scenarios.registry import ScenarioDefinition


logger = get_logger(__name__)
Intent = Literal["GREETING", "FOLLOW_UP", "KNOWLEDGE_QUERY", "FAQ_QUERY", "HUMAN_SERVICE", "OUT_OF_SCOPE"]


GREETING_PATTERNS = [
    r"^(你好|您好|hi|hello|哈喽|在吗|在不在|有人吗)",
    r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
]

FOLLOW_UP_HINTS = re.compile(r"^(那|这个|那个|它|他们|她们|这些|上面|刚才|继续|还有|费用呢|审批呢|权限呢|发票呢|告警呢|步骤呢)")
OFF_TOPIC_HINTS = re.compile(r"(彩票|赌博|股票内幕|色情|违法|攻击|破解|黑客入侵)")
HUMAN_SERVICE_HINTS = re.compile(r"(人工客服|转人工|客服|客服电话|电话|联系方式|联系老师|联系顾问|老师电话)")
FAQ_HINTS = re.compile(r"(费用|价格|安装|环境|失败|报错|地址|时间|退费|优惠|发票|账号|登录|权限|审批|合同|隐私|账单|支付|开票|工单|售后)")
KNOWLEDGE_HINTS = re.compile(r"(知识库|文档|手册|流程|制度|规范|说明|配置|接口|功能|排查|故障|步骤|sop|告警|巡检|设备|合规|条款|入职|审批|合同|隐私|webhook|回调|发票|账单)")
FAQ_QUESTION_SHAPE_HINTS = re.compile(r"(怎么办|如何处理|怎么处理|需要什么|需要哪些|需要准备哪些|有哪些|为什么|什么时候|由谁|能不能|会不会)")
DIRECT_FAQ_SHAPE_HINTS = re.compile(r"(资料呢|材料呢|是什么|如何回收|怎么排查|怎么处理|能不能|可以吗|要看什么)")

@dataclass(frozen=True)
class IntentResult:
    """意图识别的标准输出，供检索计划和下游链路消费。

    这里不是只返回一个标签，而是把“是否可直接回答、是否需要追问改写、建议业务分类、
    判断原因和置信度”一起返回。这样后续检索、Prompt 和前端诊断都能复用同一份决策。
    """

    intent: Intent
    direct_answer: str | None = None
    confidence: float = 0.6
    reason: str = "rule"
    requires_rewrite: bool = False
    suggested_source: str | None = None

    def as_dict(self) -> dict:
        """转换为可 JSON 序列化的字典，供 API 诊断信息返回。

        返回：
            包含 intent、confidence、reason、requires_rewrite、suggested_source 的字典。
        """
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reason": self.reason,
            "requires_rewrite": self.requires_rewrite,
            "suggested_source": self.suggested_source,
        }


class IntentLLMDecision(BaseModel):
    """LLM 意图识别的结构化输出模型。

    使用 Pydantic 约束模型只能返回枚举、布尔值和分数，避免模型输出解释性长文本后
    污染检索计划。
    """

    intent: Intent = Field(description="用户问题意图")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    reason: str = Field(default="")
    requires_rewrite: bool = Field(default=False)
    suggested_source: str | None = Field(default=None)


def classify_direct_intent(query: str, scenario: ScenarioDefinition | None = None) -> IntentResult | None:
    """供查询路由层复用的确定性直答规则：问候、越界、短句转人工。

    这个函数不读取历史、不调用 LLM，只处理必须优先收口的协议/安全类问题；
    普通 FAQ、知识咨询和追问仍交给检索准备阶段的 ``classify_intent()`` 判断。
    """
    normalized = query.strip().lower()
    settings = get_settings()
    suggested_source = infer_source(query, scenario)
    assistant_name = scenario.assistant_name if scenario else "知识助手"
    business_domain = scenario.business_domain if scenario else "业务知识库"
    support_contact = scenario.support_contact if scenario else settings.customer_service_phone
    greeting_answers = [
        f"你好！我是{assistant_name}，可以帮你查询{business_domain}中的制度、流程、FAQ 和文档资料。",
        f"我是{assistant_name}，负责解答{business_domain}相关问题。",
    ]
    # 规则 1 — GREETING：问候类无需检索知识库，直接返回预设友好话术
    for pattern, answer in zip(GREETING_PATTERNS, greeting_answers):
        if re.match(pattern, normalized, re.IGNORECASE):
            return IntentResult(intent="GREETING", direct_answer=answer, confidence=1.0, reason="greeting_rule")
    # 规则 2 — OUT_OF_SCOPE：越界话题必须在任何检索之前拦截
    if OFF_TOPIC_HINTS.search(normalized):
        return IntentResult(intent="OUT_OF_SCOPE", direct_answer=f"这个问题超出了{business_domain}的问答范围，我无法提供帮助。", confidence=0.95, reason="safety_rule")
    # 规则 3 — HUMAN_SERVICE：转人工请求需在短句中识别，避免长文本误触发
    if HUMAN_SERVICE_HINTS.search(normalized) and len(normalized) <= 18:
        return IntentResult(
            intent="HUMAN_SERVICE",
            direct_answer=f"可以联系人工支持，联系方式：{support_contact}。",
            confidence=0.9,
            reason="human_service_rule",
            suggested_source=suggested_source,
        )
    return None


def classify_intent(query: str, history: list[BaseMessage], scenario: ScenarioDefinition | None = None) -> IntentResult:
    """按优先级规则识别用户问题意图，规则无法判断时再调用 LLM。（★★★ 核心）

    执行流程（命中即返回）：
      1. 问候识别：你好、hello、你是谁等，直接返回介绍话术。
      2. 越界识别：违法、攻击、色情等无关/风险问题，直接拒答。
      3. 人工客服：短句中出现转人工、客服电话等请求，直接返回联系方式。
      4. 追问识别：有历史且当前问题是“那这个呢”等省略表达，需要先改写。
      5. 强规则识别：FAQ/知识库关键词和问法足够明确时，不调用 LLM。
      6. LLM 兜底：前面规则都不命中时，使用结构化输出模型判断。

    规则优先于 LLM 的核心业务决策：高频场景（问候/越界拦截/追问/FAQ 关键词）用规则 0ms 返回，避免不必要的 LLM 调用成本和延迟。

    参数：
        query: 用户原始问题。
        history: 历史对话消息列表。
        scenario: 当前业务场景，用于注入助手名称、业务域、联系方式和 source 白名单。

    返回：
        标准化 IntentResult。
    """
    suggested_source = infer_source(query, scenario)
    direct_intent = classify_direct_intent(query, scenario)
    if direct_intent:
        return direct_intent
    # 规则 4 — FOLLOW_UP：代词/省略型短句且在对话中，判定为追问需要重写；缺乏历史时不能走此路避免误判
    if history and (FOLLOW_UP_HINTS.search(query.strip()) or len(query.strip()) <= 8):
        return IntentResult(
            intent="FOLLOW_UP",
            confidence=0.8,
            reason="follow_up_rule",
            requires_rewrite=True,
            suggested_source=suggested_source,
        )
    # 规则 5 — 强领域规则：通过 FAQ/knowledge 关键词和问题句式模式，高置信度判定业务意图而不需要 LLM 开销
    strong_rule_intent = _strong_rule_domain_intent(query, suggested_source)
    if strong_rule_intent is not None:
        return strong_rule_intent
    # 规则 6 — LLM 兜底：前 5 条规则均未命中时，用结构化输出 LLM 做最终意图判断。代价较高但能覆盖长尾。
    return _classify_with_llm(query, history, suggested_source, scenario)

def infer_source(query: str, scenario: ScenarioDefinition | None = None) -> str | None:
    """只根据当前业务场景配置推断问题所属 source。

    source 是数据隔离和检索过滤的重要字段，必须来自当前场景的 `valid_sources`。
    因此这里只读取 `scenario.toml` 中的 `source_patterns`。新增行业或业务分类时，
    改场景配置即可，主链路代码不用变。

    参数：
        query: 用户问题。
        scenario: 当前业务场景；为空时无法判断分类，直接返回 None。

    返回：
        当前场景 valid_sources 中的 source，或 None。
    """
    if scenario is None:
        return None
    best_source, _ = score_source_matches(query, scenario)
    return best_source


def _strong_rule_domain_intent(query: str, suggested_source: str | None) -> IntentResult | None:
    """用领域关键词和问法强度识别高频业务问题。（★★ 理解）

    执行流程（命中即返回）：
      1. 命中 FAQ 高频词：判定为 FAQ_QUERY。
      2. 已推断 source + 短句 + 标准问法：判定为 FAQ_QUERY。
      3. 已推断 source + 短句 + 直接问法：判定为 FAQ_QUERY。
      4. 命中知识库关键词 + source 或短句：判定为 KNOWLEDGE_QUERY。

    规则强度递进设计：宽泛关键词（如"费用"）→ 句式（"怎么处理"）+ 业务域 → 句式 + 业务域 + 短句，置信度依次递增。

    1. confidence=0.82 / 0.83 / 0.84
        这是“规则置信度标签”，主要用于诊断面板、日志和 Trace，不是概率，也不是 Milvus 相似度分数。它表达的是规则强弱排序：
        0.82：只命中 FAQ 高频词，比如“费用/发票/报错”，可靠但偏宽泛
        0.83：命中业务 source + 标准问法，比如“需要哪些/怎么办”，更可靠
        0.84：命中业务 source + 直接 FAQ 问法，比如“是什么/可以吗”，更确定

        这些值不是重点，重点是相对顺序。它们表示“这条规则比上一条稍微更可信”。
    2. len(normalized) <= 32 / 36 / 24
        这是“短句保护阈值”，会影响行为。目的不是精确统计，而是降低误判：
        <=32：标准 FAQ 问法通常比较短，比如“报销需要准备哪些材料？”
        <=36：直接 FAQ 问法稍微放宽，因为可能带具体对象，比如“系统权限回收流程是什么？”
        <=24：知识查询如果没有明确 source，只允许短句命中，避免长文本里偶然出现“文档/流程/制度”就被误判

        规则分类里的数字，一开始通常来自业务样本观察和风险取舍，不是凭空神奇数字。短句阈值用来限制规则的适用范围，置信度用来表达规则强弱。
        上线后应该用真实问答集评测，再把这些数字调优。
    参数：
        query: 用户问题。
        suggested_source: infer_source() 推断出的业务分类，可能为空。

    返回：
        命中规则时返回 IntentResult，否则返回 None。
    """
    normalized = query.strip().lower()
    # FAQ 宽关键词匹配：只要出现"费用/价格/安装/报错"等高频业务词即可判定，覆盖绝大多数业务查询
    if FAQ_HINTS.search(normalized):
        return IntentResult(intent="FAQ_QUERY", confidence=0.82, reason="strong_faq_rule", suggested_source=suggested_source)
    # FAQ 句式匹配：已知业务域 + 问题句式（"怎么办/需要什么"）组合，置信度高于纯关键词
    if suggested_source and len(normalized) <= 32 and FAQ_QUESTION_SHAPE_HINTS.search(normalized):
        return IntentResult(intent="FAQ_QUERY", confidence=0.83, reason="source_question_shape_rule", suggested_source=suggested_source)
    # FAQ 句式精确匹配：已知业务域 + 直接 FAQ 句式（"是什么/可以吗"），短句限制降低误判率
    if suggested_source and len(normalized) <= 36 and DIRECT_FAQ_SHAPE_HINTS.search(normalized):
        return IntentResult(intent="FAQ_QUERY", confidence=0.84, reason="direct_faq_shape_rule", suggested_source=suggested_source)
    # 知识查询判定：知识库相关关键词 + 已知业务域或短句，避免长文本中偶然出现"文档"等词导致误判
    if KNOWLEDGE_HINTS.search(normalized) and (suggested_source or len(normalized) <= 24):
        return IntentResult(intent="KNOWLEDGE_QUERY", confidence=0.84, reason="strong_knowledge_rule", suggested_source=suggested_source)
    return None


def _classify_with_llm(
    query: str,
    history: list[BaseMessage],
    suggested_source: str | None,
    scenario: ScenarioDefinition | None = None,
) -> IntentResult:
    """规则无法判断时，调用 LLM 做结构化意图识别。（★★ 理解）

    业务决策：LLM 兜底的代价——每次调用增加 500ms~2s 延迟和 token 成本。覆盖长尾/模糊场景，但高频场景已在规则层拦截。

    执行流程：
      1. 把最近 6 条历史消息格式化为中文对话文本。
      2. 获取绑定 IntentLLMDecision 的结构化输出模型。
      3. 用系统提示词、历史对话和当前问题调用 LLM。
      4. 校验 LLM 返回的 source 是否属于当前场景白名单。
      5. 组装 IntentResult；如果是 FOLLOW_UP，强制要求后续问题改写。

    参数：
        query: 用户原始问题。
        history: 完整历史消息列表，实际只使用最近 6 条。
        suggested_source: 规则推断出的 source，可能为空。
        scenario: 当前业务场景，用于 source 白名单过滤。

    返回：
        根据 LLM 结构化结果组装出的 IntentResult。
    """
    # 将最近 6 条消息转为中文对话文本
    history_text = format_messages(history[-6:]) if history else "无"
    # 获取结构化输出模型，限制 LLM 只返回枚举字段
    model = get_chat_model(streaming=False).with_structured_output(IntentLLMDecision)
    # 调用 LLM 进行意图识别
    decision = model.invoke(
        [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"对话历史：\n{history_text}\n\n当前问题：{query}"),
        ]
    )
    source = decision.suggested_source or suggested_source
    valid_sources = scenario.valid_sources if scenario else []
    # LLM 建议的 source 必须属于当前场景允许范围，防止跨域数据泄漏
    if source and source not in valid_sources:
        source = suggested_source
    return IntentResult(
        intent=decision.intent,
        confidence=decision.confidence,
        reason=decision.reason or "llm_structured",
        # FOLLOW_UP 必须在检索前结合历史上下文重写问题，否则检索会丢失对话语境
        requires_rewrite=decision.requires_rewrite or decision.intent == "FOLLOW_UP",
        suggested_source=source,
    )

