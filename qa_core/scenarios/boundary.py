"""跨场景和跨 source 边界检测。

当用户当前选择的场景或业务分类与问题内容明显不一致时，这个模块会给出可解释的
拦截/建议结果。例如用户在“企业知识库”场景里问“隐蔽工程验收资料”，系统应能判断
它更像“工程项目资料问答”场景。
"""

from __future__ import annotations

from dataclasses import dataclass

from qa_core.scenarios.registry import ScenarioDefinition, get_scenario_registry


MIN_OTHER_SCENARIO_SCORE = 12
CURRENT_SCENARIO_SAFE_SCORE = 8


@dataclass(frozen=True)
class ScenarioBoundaryDecision:
    """单次跨场景边界检测结果。

    字段说明：
      - crossed：是否判断为跨场景。
      - matched_scenario_id：命中的其他场景 ID。
      - matched_scenario_name：命中场景的展示名称。
      - matched_source：触发命中的 source key。
      - matched_source_label：触发命中的 source 展示名称。
      - reason：判断原因，写入 trace 方便排查。
    """

    crossed: bool
    matched_scenario_id: str = ""
    matched_scenario_name: str = ""
    matched_source: str = ""
    matched_source_label: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, str | bool]:
        """转换为结构化字典，供 trace 和审计日志使用。

        返回：
            包含 crossed、matched_scenario_id、matched_source、reason 等字段的字典。
        """
        return {
            "crossed": self.crossed,
            "matched_scenario_id": self.matched_scenario_id,
            "matched_scenario_name": self.matched_scenario_name,
            "matched_source": self.matched_source,
            "matched_source_label": self.matched_source_label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceBoundaryDecision:
    """单次场景内 source 边界检测结果。

    字段说明：
      - mismatched：用户选择的 source 是否与问题内容明显不匹配。
      - selected_source：用户显式选择的 source。
      - selected_source_label：用户选择的 source 展示名称。
      - matched_source：根据问题内容命中的 source。
      - matched_source_label：命中 source 的展示名称。
      - reason：判断原因。
    """

    mismatched: bool
    selected_source: str = ""
    selected_source_label: str = ""
    matched_source: str = ""
    matched_source_label: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, str | bool]:
        """转换为结构化字典，供 trace 和审计日志使用。

        返回：
            包含 mismatched、selected_source、matched_source、reason 等字段的字典。
        """
        return {
            "mismatched": self.mismatched,
            "selected_source": self.selected_source,
            "selected_source_label": self.selected_source_label,
            "matched_source": self.matched_source,
            "matched_source_label": self.matched_source_label,
            "reason": self.reason,
        }


def score_source_map(query: str, scenario: ScenarioDefinition) -> dict[str, int]:
    """计算问题对当前场景内每个 source pattern 的匹配分数。

    计分公式：命中次数 * 10 + 命中文本长度总和 - source 顺序索引。

    公式设计理念：
      - 命中次数 * 10：给多次命中一个较高的基础分，避免单次短匹配偶然胜出（10 倍放大系数确保
        两次短命的优先于一次长命，因为 pattern 重复出现往往意味着主题相关性更强）。
      - + 匹配文本总长度：同次数时，匹配到的原文越长说明 pattern 与 query 的语义重叠度越高，
        对跨场景判断更有参考价值。
      - - source 索引：在 valid_sources 列表中越靠前的 source 获得轻微的分数优势，让系统在
        得分相近时倾向于优先级更高的分类，避免频繁的边界误报触发切换建议。
    valid_sources 中越靠前的 source，在同分附近会有轻微优先级优势。

    参数：
        query: 用户原始问题。
        scenario: 当前业务场景，内部包含 source_patterns。

    返回：
        {source_key: score} 字典；没有任何命中时返回空字典。
    """
    normalized = query.strip().lower()
    scores: dict[str, int] = {}
    for index, (source, pattern) in enumerate(scenario.compiled_source_patterns().items()):
        matches = list(pattern.finditer(query)) + list(pattern.finditer(normalized))
        if not matches:
            continue
        scores[source] = len(matches) * 10 + sum(len(match.group(0)) for match in matches) - index
    return scores


def score_source_matches(query: str, scenario: ScenarioDefinition) -> tuple[str | None, int]:
    """返回当前场景内最匹配的 source 及其分数。

    参数：
        query: 用户原始问题。
        scenario: 当前业务场景。

    返回：
        (最佳 source 或 None, 分数)。
    """
    scores = score_source_map(query, scenario)
    best_source = None
    best_score = 0
    for source, score in scores.items():
        if score > best_score:
            best_source = source
            best_score = score
    return best_source, best_score


def detect_source_boundary(
    query: str,
    current_scenario: ScenarioDefinition,
    selected_source: str | None,
) -> SourceBoundaryDecision:
    """检查用户显式选择的 source 是否与问题内容明显不匹配。

    执行流程：
      1. 用户没有选择 source 时不做提示。
      2. 根据当前场景 source_patterns 推断问题更像哪个 source。
      3. 如果没有命中或命中结果就是用户选择的 source，则继续正常流程。
      4. 如果命中的是另一个 source，则提示切换分类。

    参数：
        query: 用户原始问题。
        current_scenario: 当前业务场景。
        selected_source: 用户显式选择的 source，可能为空。

    返回：
        source 边界检测结果。
    """
    # 用户未指定分类时不做越界提示，避免空分类场景下的误报干扰
    if not selected_source:
        return SourceBoundaryDecision(mismatched=False, reason="no_selected_source")
    matched_source, score = score_source_matches(query, current_scenario)
    # 无匹配 / 已选分类最佳 → 保持当前流程，减少不必要的前端切换建议
    if not matched_source or matched_source == selected_source:
        return SourceBoundaryDecision(mismatched=False, selected_source=selected_source, reason="source_matched_or_unknown")
    # 命中另一个分类 → 主动建议切换，帮助用户快速定位
    return SourceBoundaryDecision(
        mismatched=True,
        selected_source=selected_source,
        selected_source_label=current_scenario.label_for_source(selected_source),
        matched_source=matched_source,
        matched_source_label=current_scenario.label_for_source(matched_source),
        reason=f"matched_source_score={score}",
    )


def detect_scenario_boundary(query: str, current_scenario: ScenarioDefinition) -> ScenarioBoundaryDecision:
    """检测当前问题是否明显属于另一个业务场景。

    执行流程：
      1. 先对当前场景的 source_patterns 打分。
      2. 当前场景已有可信命中时，不判定跨场景。
      3. 遍历其他场景，分别计算匹配分数。
      4. 没有其他场景达到最低阈值时，不判定跨场景。
      5. 如果其他场景强命中，则返回跨场景建议和命中详情。

    参数：
        query: 用户原始问题。
        current_scenario: 当前业务场景。

    返回：
        场景边界检测结果。
    """
    current_source, current_score = score_source_matches(query, current_scenario)
    # 当前场景有明确匹配证据时直接返回未跨域，避免跨场景误判
    if current_source and current_score >= CURRENT_SCENARIO_SAFE_SCORE:
        return ScenarioBoundaryDecision(crossed=False, reason="current_scenario_matched")

    best_scenario: ScenarioDefinition | None = None
    best_source = None
    best_score = 0
    for scenario in get_scenario_registry().list_scenarios():
        if scenario.scenario_id == current_scenario.scenario_id:
            continue
        source, score = score_source_matches(query, scenario)
        if source and score > best_score:
            best_scenario = scenario
            best_source = source
            best_score = score

    # 无其他场景达到匹配最低分 → 保持当前场景不切换，避免无依据的场景跳转
    if best_scenario is None or best_source is None or best_score < MIN_OTHER_SCENARIO_SCORE:
        return ScenarioBoundaryDecision(crossed=False, reason="no_strong_other_scenario")

    # 其他场景得分远超当前场景 → 判定用户走错了场景，引导切换
    return ScenarioBoundaryDecision(
        crossed=True,
        matched_scenario_id=best_scenario.scenario_id,
        matched_scenario_name=best_scenario.display_name,
        matched_source=best_source,
        matched_source_label=best_scenario.label_for_source(best_source),
        reason=f"other_scenario_score={best_score}, current_score={current_score}",
    )
