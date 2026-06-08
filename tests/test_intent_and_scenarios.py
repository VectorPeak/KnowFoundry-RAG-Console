"""意图识别、问题类别和场景配置的纯逻辑测试。"""

from __future__ import annotations

import unittest

from qa_core.indexing.faq_ingestion import normalize_faq_source
from qa_core.intent.classifier import classify_intent, infer_source
from qa_core.intent.question_category import infer_question_category
from qa_core.scenarios.boundary import detect_scenario_boundary, detect_source_boundary
from qa_core.scenarios.registry import get_scenario_registry


class QuestionCategoryTests(unittest.TestCase):
    """验证问题类别能驱动不同检索策略和提示词模板。"""

    def test_infer_risk_categories(self) -> None:
        self.assertEqual(infer_question_category("发票和退款规则是什么"), "pricing")
        self.assertEqual(infer_question_category("合同隐私条款有什么风险"), "compliance")
        self.assertEqual(infer_question_category("设备出现温度告警怎么排查"), "troubleshooting")
        self.assertEqual(infer_question_category("企业入职流程都包括哪些内容"), "summary")
        self.assertEqual(infer_question_category("请介绍一下产品功能"), "default")


class ScenarioRegistryTests(unittest.TestCase):
    """验证当前项目冻结的 8 个业务场景均已注册。"""

    def test_all_frozen_business_scenarios_are_registered(self) -> None:
        registry = get_scenario_registry()
        scenario_ids = {scenario.scenario_id for scenario in registry.list_scenarios()}
        self.assertEqual(len(scenario_ids), 8)
        self.assertIn("enterprise_knowledge", scenario_ids)
        self.assertIn("saas_support", scenario_ids)
        self.assertIn("equipment_ops", scenario_ids)
        self.assertIn("compliance_qa", scenario_ids)
        self.assertIn("cross_border_risk", scenario_ids)
        self.assertIn("tender_contract_risk", scenario_ids)
        self.assertIn("insurance_claims", scenario_ids)
        self.assertIn("engineering_project_qa", scenario_ids)

    def test_enterprise_source_patterns_are_used_for_source_inference(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        self.assertEqual(infer_source("新人入职流程怎么走", scenario), "hr")
        self.assertEqual(infer_source("VPN 连不上怎么处理", scenario), "it")
        self.assertEqual(infer_source("员工报销需要准备哪些材料", scenario), "finance")

    def test_cross_border_source_patterns_are_used_for_source_inference(self) -> None:
        scenario = get_scenario_registry().resolve("cross_border_risk")
        self.assertEqual(infer_source("交易对手命中制裁名单怎么办", scenario), "sanction")
        self.assertEqual(infer_source("信用证不符点如何处理", scenario), "payment")

    def test_tender_and_insurance_patterns_are_used_for_source_inference(self) -> None:
        tender = get_scenario_registry().resolve("tender_contract_risk")
        insurance = get_scenario_registry().resolve("insurance_claims")
        self.assertEqual(infer_source("投标文件缺少授权书有什么风险", tender), "bidding")
        self.assertEqual(infer_source("非标准付款条款需要谁复核", tender), "contract")
        self.assertEqual(infer_source("理赔申请需要哪些材料", insurance), "claim_material")
        self.assertEqual(infer_source("哪些情况可能属于除外责任", insurance), "exclusion")

    def test_engineering_project_patterns_are_used_for_source_inference(self) -> None:
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        self.assertEqual(infer_source("图纸变更后旧版本还能作为施工依据吗", scenario), "drawing")
        self.assertEqual(infer_source("隐蔽工程验收需要哪些资料", scenario), "quality")
        self.assertEqual(infer_source("高处作业前必须做哪些安全资料", scenario), "safety")
        self.assertEqual(infer_source("施工图纸和强制性规范冲突时怎么办", scenario), "specification")

    def test_scenario_boundary_detects_question_from_other_business_scene(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        decision = detect_scenario_boundary("安全技术交底只有口头说明可以吗？", scenario)
        self.assertTrue(decision.crossed)
        self.assertEqual(decision.matched_scenario_id, "engineering_project_qa")
        self.assertEqual(decision.matched_source, "safety")

    def test_source_boundary_detects_wrong_selected_source(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        decision = detect_source_boundary("员工报销需要准备哪些材料？", scenario, "hr")
        self.assertTrue(decision.mismatched)
        self.assertEqual(decision.matched_source, "finance")

    def test_source_boundary_allows_matching_selected_source(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        decision = detect_source_boundary("员工报销需要准备哪些材料？", scenario, "finance")
        self.assertFalse(decision.mismatched)


class IntentClassifierTests(unittest.TestCase):
    """验证当前意图识别输出通用知识问答意图，不再输出课程咨询意图。"""

    def test_business_knowledge_question_uses_knowledge_intent(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        result = classify_intent("新人入职流程怎么走", [], scenario)
        self.assertEqual(result.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(result.suggested_source, "hr")

    def test_short_direct_faq_shape_prefers_faq_intent(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        result = classify_intent("员工报销需要准备哪些材料？", [], scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.reason, "source_question_shape_rule")
        self.assertEqual(result.suggested_source, "finance")

    def test_saas_faq_question_uses_faq_intent_with_scenario_source(self) -> None:
        scenario = get_scenario_registry().resolve("saas_support")
        result = classify_intent("发票什么时候可以开", [], scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.suggested_source, "billing")

    def test_source_question_shape_uses_faq_rule_without_llm(self) -> None:
        scenario = get_scenario_registry().resolve("cross_border_risk")
        result = classify_intent("交易对手命中制裁名单怎么办", [], scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.reason, "source_question_shape_rule")
        self.assertEqual(result.suggested_source, "sanction")


class FaqIngestionTests(unittest.TestCase):
    """验证 FAQ CSV 入库前的分类归一化规则。"""

    def test_normalize_faq_source_uses_scenario_valid_sources_and_patterns(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        self.assertEqual(normalize_faq_source("finance", scenario=scenario), "finance")
        self.assertEqual(normalize_faq_source("财务报销", scenario=scenario), "finance")
        self.assertEqual(normalize_faq_source("账号权限", scenario=scenario, question="VPN 账号权限怎么申请"), "it")
        self.assertEqual(normalize_faq_source("入职流程", scenario=scenario), "hr")

    def test_normalize_faq_source_rejects_empty_source(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        with self.assertRaises(ValueError):
            normalize_faq_source("", scenario=scenario)
