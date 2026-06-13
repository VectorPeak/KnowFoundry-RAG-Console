"""RAG 链路动态检索计划，将意图结果转为检索参数。

它把意图识别结果转换成具体检索参数，例如 top_k、阈值、是否查询 FAQ/文档集合。
不同问题类型（FAQ、追问、短问题、表格问题）会经过四层决策链得到不同策略。

决策层次：
    1. 意图分支：直接回答、FAQ、知识查询或追问。
    2. 短问题保护：短句歧义大，限制文档检索并提高直出门槛。
    3. 风险类别：费用、合规、排障、总结类问题使用不同参数。
    4. 表格偏好：表格类查询扩大文档候选池。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from qa_core.intent.classifier import IntentResult
from qa_core.intent.question_category import infer_question_category, is_table_query
from qa_core.config.settings import get_settings


@dataclass(frozen=True)
class RetrievalPlan:
    """单个用户问题对应的具体检索参数。QAService 消费此计划而非直接读取 settings，便于策略调整和诊断。

    字段说明：
      - run_faq/run_doc：是否查询 FAQ/文档集合。
      - faq_top_k/doc_top_k：FAQ 和文档初始召回数量。
      - rerank：是否启用 CrossEncoder 重排。
      - faq_direct_threshold：FAQ 直出最低分数阈值。
      - final_context_top_n：最终进入 LLM 上下文的最多片段数。
      - min_context_score：进入上下文的最低相关性分数。
      - max_context_chars/max_context_doc_chars：总上下文和单文档字符上限。
      - use_query_variants：是否生成查询变体。
      - question_category：问题风险类别。
      - prefer_table：是否偏向表格行资料。
      - faq_direct_exact_only：是否只允许精确 FAQ 直出，禁用相似分数直出。
      - reason：检索策略原因标签，用于诊断。
    """

    run_faq: bool
    run_doc: bool
    faq_top_k: int
    doc_top_k: int
    rerank: bool
    faq_direct_threshold: float
    final_context_top_n: int
    min_context_score: float
    max_context_chars: int
    max_context_doc_chars: int
    use_query_variants: bool
    question_category: str
    prefer_table: bool
    faq_direct_exact_only: bool
    reason: str

    def as_dict(self) -> dict:
        """返回可 JSON 序列化的检索计划，供诊断接口使用。

        返回：
            RetrievalPlan 字段字典。
        """
        return asdict(self)


# ── Private helpers ──────────────────────────────────────────────


def _apply_intent_branching(intent: IntentResult, params: dict, settings) -> dict:
    """第 1 层：按意图分岔调整检索参数。（★★ 理解）

    规则说明：
      - direct_answer、GREETING、HUMAN_SERVICE、OUT_OF_SCOPE：关闭 FAQ 和文档检索。
      - FAQ_QUERY：FAQ 优先，压缩 doc_top_k，适当降低直出阈值。
      - KNOWLEDGE_QUERY：扩大文档召回和上下文数量。
      - FOLLOW_UP：扩大 FAQ 和文档召回，提高直出阈值。

    参数：
        intent: 意图识别输出的 IntentResult.
        params: Mutable parameter dictionary, modified in place.
        settings: Application settings object.

    返回：
        根据意图调整后的参数字典.
    """
    # 无需检索：问候/转人工/越界问题直接返回预设答案，不走知识库
    if intent.direct_answer or intent.intent in {"GREETING", "HUMAN_SERVICE", "OUT_OF_SCOPE"}:
        params['run_faq'] = False
        params['run_doc'] = False
        params['direct_threshold'] = 1.0
        params['reason'] = "direct_answer_no_retrieval"
    # FAQ 优先：压缩文档检索预算降低延迟，降低直出门槛让更多 FAQ 命中
    elif intent.intent == "FAQ_QUERY":
        params['doc_top_k'] = max(8, settings.doc_top_k // 2)
        params['direct_threshold'] = max(0.62, settings.faq_direct_score_threshold - 0.08)
        params['reason'] = "faq_first"
    # 知识类问题需要更多文档候选保证深度内容被召回，扩大上下文上限
    elif intent.intent == "KNOWLEDGE_QUERY":
        params['doc_top_k'] = max(settings.doc_top_k, settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(settings.final_context_top_n, 5)
        params['reason'] = "knowledge_doc_enriched"
    # 追问已有对话语境，同步扩大 FAQ 和文档候选，提高直出精度阈值避免噪音
    elif intent.intent == "FOLLOW_UP":
        params['faq_top_k'] = max(settings.faq_top_k, 24)
        params['doc_top_k'] = max(settings.doc_top_k, settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(settings.final_context_top_n, 5)
        params['direct_threshold'] = max(settings.faq_direct_score_threshold, 0.78)
        params['reason'] = "history_aware_follow_up"
    return params


def _apply_short_query_guard(is_short: bool, intent: IntentResult, params: dict, settings) -> dict:
    """第 2 层：短问题保护 — 短句歧义大，收缩文档检索范围、提高 FAQ 直出门槛。（★★ 理解）

    短问题缺少上下文，容易误命中文档或 FAQ。FOLLOW_UP 例外，因为追问会先结合历史改写，
    具备额外语境。

    参数：
        is_short: ``True`` if query length <= ``short_query_max_chars``.
        intent: 意图识别输出的 IntentResult，用于豁免 FOLLOW_UP。
        params: Mutable parameter dictionary, modified in place.
        settings: Application settings object.

    返回：
        应用短问题保护后的参数字典.
    """
    # 短文本歧义大：收缩文档范围避免主题漂移；FOLLOW_UP 有历史语境辅助消歧，豁免此项限制
    if is_short and intent.intent != "FOLLOW_UP":
        params['doc_top_k'] = min(params['doc_top_k'], max(12, settings.final_context_top_n * 2))
        params['direct_threshold'] = max(0.78, params['direct_threshold'])
        params['reason'] = f"{params['reason']}_short_query_guard"
    return params


def _apply_risk_category(question_category: str, params: dict, settings) -> dict:
    """第 3 层：按风险类别收紧检索参数。（★★ 理解）

    规则说明：
      - pricing：费用/金额类问题扩大 FAQ 和文档召回，阈值至少 0.84。
      - compliance：合规类问题更严格，阈值至少 0.86。
      - troubleshooting：排障类问题扩大文档召回，保证步骤完整。
      - summary：总结类问题扩大文档召回，覆盖更多资料。

    参数：
        question_category: infer_question_category() 推断出的问题类别.
        params: Mutable parameter dictionary, modified in place.
        settings: Application settings object.

    返回：
        应用风险类别调整后的参数字典.
    """
    # 费用类属高风险敏感话题，扩大 FAQ+文档召回且门槛≥0.84，确保金额/价格信息可靠
    if question_category == "pricing":
        params['faq_top_k'] = max(params['faq_top_k'], settings.faq_top_k)
        params['doc_top_k'] = max(params['doc_top_k'], settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(params['final_context_top_n'], 6)
        params['direct_threshold'] = max(params['direct_threshold'], 0.84)
        params['reason'] = f"{params['reason']}_pricing_guard"
    # 合规类风险最高（门槛≥0.86），答案错误可能导致监管处罚或合同纠纷
    elif question_category == "compliance":
        params['doc_top_k'] = max(params['doc_top_k'], settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(params['final_context_top_n'], 6)
        params['direct_threshold'] = max(params['direct_threshold'], 0.86)
        params['reason'] = f"{params['reason']}_compliance_guard"
    # 排障类需广召回获取完整排查步骤链，上下文至少 6 条避免遗漏关键环节
    elif question_category == "troubleshooting":
        params['doc_top_k'] = max(params['doc_top_k'], settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(params['final_context_top_n'], 6)
        params['reason'] = f"{params['reason']}_troubleshooting_expanded"
    # 总结类需要多看几篇文档才能覆盖全面，扩大文档召回和上下文窗口
    elif question_category == "summary":
        params['doc_top_k'] = max(params['doc_top_k'], settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(params['final_context_top_n'], 6)
        params['reason'] = f"{params['reason']}_summary_expanded"
    return params


def _apply_table_preference(prefer_table: bool, run_doc: bool, params: dict, settings) -> dict:
    """第 4 层：表格偏好 — 表格类问题扩大候选、关闭相似 FAQ 直出。（★★ 理解）

    表格类问题通常要定位具体行列。相似 FAQ 很容易“看起来相关但不是同一行数据”，
    所以这里扩大文档候选，并关闭相似 FAQ 直出。

    参数：
        prefer_table: ``True`` if the query is classified as a table query.
        run_doc: Whether document search is currently enabled.
        params: Mutable parameter dictionary, modified in place.
        settings: Application settings object.

    返回：
        应用表格偏好调整后的参数字典.
    """
    # 表格查询需扩大候选以匹配正确行列；禁用模糊 FAQ 直出防止表格内容相似但不同的误导
    if prefer_table and run_doc:
        params['doc_top_k'] = max(params['doc_top_k'], settings.doc_complex_query_top_k)
        params['final_context_top_n'] = max(params['final_context_top_n'], 7)
        params['faq_direct_exact_only'] = True
        params['reason'] = f"{params['reason']}_table_row_preferred"
    return params


def build_retrieval_plan(query: str, intent: IntentResult) -> RetrievalPlan:
    """根据问题形态和意图结果构建检索策略。按 5 层决策逐层收紧参数。（★★★ 核心）

    执行流程：
      1. 初始化默认参数：问题清洗、类别识别、表格偏好识别、短句识别。
      2. 应用意图分支：直接回答关闭检索，FAQ/知识/追问分别调整召回量和阈值。
      3. 应用短问题保护：短句提高 FAQ 直出门槛并收缩文档召回。
      4. 应用风险类别：费用、合规、排障、总结问题扩大召回或提高阈值。
      5. 应用表格偏好：表格问题扩大文档候选并禁用模糊 FAQ 直出。
      6. 组装不可变 RetrievalPlan；知识查询和追问启用查询变体。

    参数：
        query: 用户原始问题。
        intent: 意图识别输出的 IntentResult.

    返回：
        QAService 执行检索所需的完整 RetrievalPlan。
    """
    settings = get_settings()
    compact_query = query.strip()
    # 推断问题风险类别（pricing/compliance/troubleshooting/summary/other）—— 风险类别驱动检索阈值和回答模板
    question_category = infer_question_category(compact_query)
    # 判断是否为表格类查询 —— 表格问题需扩大候选集并禁用模糊 FAQ 直出，语义相似但不同列的表格内容误导性极强
    prefer_table = is_table_query(compact_query)
    is_short = len(compact_query) <= settings.short_query_max_chars

    # 以 settings 基线初始化，各层在此基础上增量微调而非覆盖，保证各层决策可叠加
    params = {
        'run_faq': True,
        'run_doc': True,
        'faq_top_k': settings.faq_short_query_top_k if is_short else settings.faq_top_k,
        'doc_top_k': settings.doc_top_k,
        'final_context_top_n': settings.final_context_top_n,
        'direct_threshold': settings.faq_direct_score_threshold,
        'faq_direct_exact_only': False,
        'reason': 'balanced_hybrid',
    }

    # 链式调用 4 个决策层（各层职责正交、解耦便于单独调参）
    params = _apply_intent_branching(intent, params, settings)
    params = _apply_short_query_guard(is_short, intent, params, settings)
    params = _apply_risk_category(question_category, params, settings)
    params = _apply_table_preference(prefer_table, params['run_doc'], params, settings)

    return RetrievalPlan(
        run_faq=params['run_faq'],
        run_doc=params['run_doc'],
        faq_top_k=params['faq_top_k'],
        doc_top_k=params['doc_top_k'],
        rerank=True,
        faq_direct_threshold=params['direct_threshold'],
        final_context_top_n=params['final_context_top_n'],
        min_context_score=settings.rag_min_score_threshold,
        max_context_chars=settings.max_prompt_context_chars,
        max_context_doc_chars=settings.max_context_doc_chars,
        use_query_variants=intent.intent in {"KNOWLEDGE_QUERY", "FOLLOW_UP"},
        question_category=question_category,
        prefer_table=prefer_table,
        faq_direct_exact_only=params['faq_direct_exact_only'],
        reason=params['reason'],
    )

