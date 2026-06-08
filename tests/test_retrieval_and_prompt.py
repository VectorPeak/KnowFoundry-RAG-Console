"""检索过滤、上下文构建、排序和 Prompt Profile 的纯逻辑测试。"""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from qa_core.governance.data_scope import resolve_data_scope
from qa_core.intent.classifier import IntentResult, classify_intent
from qa_core.intent.question_category import infer_question_category, is_table_query
from qa_core.pipeline.citations import enforce_answer_citations, has_source_citation
from qa_core.pipeline.context import build_context, direct_faq_answer, select_context_docs
from qa_core.pipeline.query_variants import generate_query_variants
from qa_core.pipeline.steps import should_try_faq_fast_path
from qa_core.prompts.selector import build_answer_prompt_profile
from qa_core.retrieval.filters import build_source_expr
from qa_core.retrieval.ranking import merge_hits_by_document, normalize_queries, sort_hits_by_score
from qa_core.retrieval.results import RetrievalHit, RetrievalResult
from qa_core.retrieval.store import MilvusHybridStore
from qa_core.retrieval.strategy import build_retrieval_plan
from qa_core.scenarios.registry import get_scenario_registry


class RetrievalFilterTests(unittest.TestCase):
    """验证 source、版本和数据域会进入 Milvus 过滤表达式。"""

    def test_build_source_expr_with_scope_and_version(self) -> None:
        scope = resolve_data_scope(tenant_id="tenant_a", dataset_id="dataset_1", visibility="internal", user_role="admin")
        expr = build_source_expr(
            "billing",
            kb_version="kb_v1",
            valid_sources=["billing", "support"],
            data_scope=scope,
        )
        self.assertIn('source == "billing"', expr)
        self.assertIn('kb_version == "kb_v1"', expr)
        self.assertIn('tenant_id == "tenant_a"', expr)
        self.assertIn('dataset_id == "dataset_1"', expr)
        self.assertIn('array_contains(allowed_roles, "admin")', expr)

    def test_build_source_expr_rejects_invalid_source(self) -> None:
        with self.assertRaises(ValueError):
            build_source_expr("unknown", valid_sources=["billing"])


class ContextBuilderTests(unittest.TestCase):
    """验证 FAQ 直出和上下文构建的保守规则。"""

    def test_direct_faq_answer_requires_exact_match_or_threshold(self) -> None:
        doc = Document(
            page_content="是否支持开发票",
            metadata={"standard_question": "是否支持开发票", "answer": "支持，具体以系统规则为准。"},
        )
        self.assertEqual(direct_faq_answer("是否支持开发票", doc, score=0.1, threshold=0.9), "支持，具体以系统规则为准。")
        self.assertEqual(direct_faq_answer("可以开票吗", doc, score=0.95, threshold=0.9), "支持，具体以系统规则为准。")
        self.assertIsNone(direct_faq_answer("可以开票吗", doc, score=0.3, threshold=0.9))

    def test_faq_fast_path_only_targets_short_complete_questions(self) -> None:
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        self.assertTrue(should_try_faq_fast_path("安全技术交底只有口头说明可以吗？", scenario))
        self.assertFalse(
            should_try_faq_fast_path(
                "请把本项目所有安全资料、质量资料、进度资料、图纸资料和规范资料做一个完整总结，并说明每类资料的风险边界。",
                scenario,
            )
        )

    def test_build_context_deduplicates_and_labels_sources(self) -> None:
        docs = [
            Document(page_content="第一段内容", metadata={"file_name": "a.md"}),
            Document(page_content="第一段内容", metadata={"file_name": "a.md"}),
            Document(page_content="第二段内容", metadata={"standard_question": "标准问题"}),
        ]
        context = build_context(docs)
        self.assertIn("[1] 来源：a.md", context)
        self.assertIn("[2] 来源：标准问题", context)
        self.assertEqual(context.count("第一段内容"), 1)

    def test_build_context_labels_table_row_location(self) -> None:
        context = build_context(
            [
                Document(
                    page_content="施工照片状态：必需",
                    metadata={
                        "file_name": "hidden_acceptance_materials.csv",
                        "content_type": "table_row",
                        "sheet_name": "csv",
                        "row_number": 2,
                    },
                )
            ]
        )
        self.assertIn("hidden_acceptance_materials.csv / 工作表：csv / 第 2 行", context)

    def test_select_context_docs_filters_low_score_faq_and_doc_hits(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("新人入职流程怎么走", [], scenario)
        plan = build_retrieval_plan("新人入职流程怎么走", intent)
        faq_hits = [
            RetrievalHit(
                document=Document(
                    page_content="低分 FAQ",
                    metadata={"standard_question": "低分 FAQ", "answer": "不应进入上下文"},
                ),
                score=plan.min_context_score - 0.01,
            )
        ]
        doc_hits = [
            RetrievalHit(
                document=Document(page_content="低分文档", metadata={"file_name": "low.md"}),
                score=plan.min_context_score - 0.01,
            )
        ]
        self.assertEqual(select_context_docs(faq_hits, doc_hits, plan), [])

    def test_select_context_docs_deduplicates_parent_and_applies_budget(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("新人入职流程怎么走", [], scenario)
        plan = build_retrieval_plan("新人入职流程怎么走", intent)
        doc_hits = [
            RetrievalHit(
                document=Document(
                    page_content="子块一",
                    metadata={
                        "parent_id": "parent_1",
                        "parent_content": "甲" * (plan.max_context_doc_chars + 80),
                        "file_name": "a.md",
                    },
                ),
                score=0.9,
            ),
            RetrievalHit(
                document=Document(
                    page_content="子块二",
                    metadata={
                        "parent_id": "parent_1",
                        "parent_content": "重复父块",
                        "file_name": "a.md",
                    },
                ),
                score=0.8,
            ),
        ]
        docs = select_context_docs([], doc_hits, plan)
        self.assertEqual(len(docs), 1)
        self.assertLessEqual(len(docs[0].page_content), plan.max_context_doc_chars)
        self.assertTrue(docs[0].metadata["context_truncated"])

    def test_table_query_prefers_table_rows_in_context(self) -> None:
        """表格类问题应优先把行级证据放入 prompt。"""
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        intent = classify_intent("隐蔽验收资料清单里施工照片是什么状态？", [], scenario)
        plan = build_retrieval_plan("隐蔽验收资料清单里施工照片是什么状态？", intent)
        doc_hits = [
            RetrievalHit(
                document=Document(page_content="普通正文片段", metadata={"file_name": "hidden_acceptance.md"}),
                score=0.99,
            ),
            RetrievalHit(
                document=Document(
                    page_content="表格文件：hidden_acceptance_materials.csv\n行号：2\n- 资料名称：施工照片\n- 状态：必需",
                    metadata={"file_name": "hidden_acceptance_materials.csv", "content_type": "table_row", "row_number": 2},
                ),
                score=0.72,
            ),
        ]
        docs = select_context_docs([], doc_hits, plan)
        self.assertTrue(plan.prefer_table)
        self.assertTrue(plan.faq_direct_exact_only)
        self.assertEqual(docs[0].metadata["content_type"], "table_row")

    def test_enforce_answer_citations_appends_table_source_when_missing(self) -> None:
        docs = [
            Document(
                page_content="施工照片状态：必需",
                metadata={
                    "file_name": "hidden_acceptance_materials.csv",
                    "content_type": "table_row",
                    "sheet_name": "csv",
                    "row_number": 2,
                },
            )
        ]
        answer = enforce_answer_citations("施工照片状态为必需。", docs)
        self.assertTrue(has_source_citation(answer))
        self.assertIn("hidden_acceptance_materials.csv / 工作表：csv / 第 2 行", answer)

    def test_enforce_answer_citations_keeps_existing_citation(self) -> None:
        answer = enforce_answer_citations("施工照片状态为必需。[1]", [Document(page_content="x")])
        self.assertEqual(answer, "施工照片状态为必需。[1]")

    def test_enforce_answer_citations_appends_missing_table_cells_even_with_citation(self) -> None:
        """表格答案即使已经带引用，也要补齐同一行里被模型漏掉的关键单元格。"""
        docs = [
            Document(
                page_content=(
                    "表格文件：claim_material_review.csv\n"
                    "单元格：\n"
                    "- 材料名称：银行卡信息\n"
                    "- 审核状态：必需\n"
                    "- 核验要点：账户名需与被保险人或授权收款人一致\n"
                    "- 处理动作：不一致时进入人工复核"
                ),
                metadata={
                    "file_name": "claim_material_review.csv",
                    "content_type": "table_row",
                    "sheet_name": "csv",
                    "row_number": 2,
                },
            )
        ]
        answer = enforce_answer_citations("银行卡信息不一致时，需进入人工复核 [1]。", docs)
        self.assertIn("表格行要点", answer)
        self.assertIn("账户名需与被保险人或授权收款人一致", answer)
        self.assertIn("[1]", answer)


class RetrievalRankingTests(unittest.TestCase):
    """验证多路查询召回后的去重和排序规则。"""

    def test_normalize_queries_keeps_order_and_removes_duplicates(self) -> None:
        queries = normalize_queries([" Webhook失败 ", "", "回调失败", "Webhook失败", "  "])
        self.assertEqual(queries, ["Webhook失败", "回调失败"])

    def test_merge_hits_keeps_highest_score_for_same_document(self) -> None:
        first = RetrievalHit(
            document=Document(page_content="同一个 chunk", metadata={"chunk_id": "chunk_1"}),
            score=0.4,
        )
        second = RetrievalHit(
            document=Document(page_content="同一个 chunk 新命中", metadata={"chunk_id": "chunk_1"}),
            score=0.9,
        )
        merged: dict[str, RetrievalHit] = {}
        merge_hits_by_document(merged, [first])
        merge_hits_by_document(merged, [second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["chunk_1"].score, 0.9)

    def test_sort_hits_by_score_descending(self) -> None:
        hits = [
            RetrievalHit(document=Document(page_content="低分"), score=0.2),
            RetrievalHit(document=Document(page_content="高分"), score=0.8),
        ]
        sorted_hits = sort_hits_by_score(hits)
        self.assertEqual([hit.document.page_content for hit in sorted_hits], ["高分", "低分"])

    def test_source_payloads_include_table_citation(self) -> None:
        result = RetrievalResult(
            hits=[
                RetrievalHit(
                    document=Document(
                        page_content="表格文件：acceptance_checklist.csv\n- 验收项：测试报告",
                        metadata={
                            "file_name": "acceptance_checklist.csv",
                            "content_type": "table_row",
                            "sheet_name": "csv",
                            "row_number": 3,
                            "table_headers": "验收项 | 状态",
                        },
                    ),
                    score=0.8,
                )
            ],
            source_type="doc",
        )
        payload = result.source_payloads()[0]
        self.assertEqual(payload["citation"], "acceptance_checklist.csv / 工作表：csv / 第 3 行")
        self.assertEqual(payload["table"]["row_number"], 3)


class MilvusHybridStoreTests(unittest.TestCase):
    """验证 Milvus 检索边界条件不会把底层兼容异常暴露给用户。"""

    def test_empty_query_returns_empty_result_without_touching_store(self) -> None:
        store = MilvusHybridStore("unit_test_collection")
        result = store.search(
            "  ",
            k=5,
            source_filter=None,
            valid_sources=[],
            source_type="faq",
            rerank=False,
        )
        self.assertEqual(result.hits, [])
        self.assertEqual(result.query, "")

    def test_nq_zero_hybrid_error_falls_back_to_dense_search(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.dense_called = False

            def similarity_search_with_score(self, *args, **kwargs):
                from pymilvus.exceptions import MilvusException

                raise MilvusException(
                    code=65535,
                    message=(
                        "nq [0] is invalid, nq (number of search vector per search request) "
                        "should be in range [1, 16384], but got 0"
                    ),
                )

            def similarity_search_with_score_by_vector(self, embedding, *args, **kwargs):
                self.dense_called = True
                self.embedding = embedding
                return [(Document(page_content="dense fallback hit", metadata={"chunk_id": "c1"}), 0.7)]

        class FakeEmbeddings:
            def embed_query(self, query: str):
                return [0.1, 0.2, 0.3]

        fake_store = FakeStore()
        store = MilvusHybridStore("unit_test_collection")
        store._store = fake_store
        import qa_core.retrieval.store as store_module

        original_get_embeddings = store_module.get_embeddings
        store_module.get_embeddings = lambda: FakeEmbeddings()
        try:
            result = store.search(
                "申报要素缺失会有什么风险？",
                k=5,
                source_filter=None,
                valid_sources=[],
                source_type="faq",
                rerank=False,
            )
        finally:
            store_module.get_embeddings = original_get_embeddings

        self.assertTrue(fake_store.dense_called)
        self.assertEqual(fake_store.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(result.hits[0].document.page_content, "dense fallback hit")
        self.assertEqual(result.hits[0].score, 0.7)


class RetrievalPlanTests(unittest.TestCase):
    """验证意图和问题类别会稳定影响检索计划。"""

    def test_knowledge_intent_enables_query_variants_and_expands_docs(self) -> None:
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("新人入职流程怎么走", [], scenario)
        self.assertEqual(intent.intent, "KNOWLEDGE_QUERY")
        plan = build_retrieval_plan("新人入职流程怎么走", intent)
        self.assertTrue(plan.use_query_variants)
        self.assertIn("knowledge_doc_enriched", plan.reason)

    def test_short_structured_questions_do_not_expand_with_llm_variants(self) -> None:
        self.assertEqual(generate_query_variants("新人入职流程怎么走", enabled=True), ["新人入职流程怎么走"])
        self.assertEqual(generate_query_variants("那隐蔽工程验收资料呢？", enabled=True), ["那隐蔽工程验收资料呢？"])

    def test_source_guided_short_question_prefers_faq_intent(self) -> None:
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        intent = classify_intent("那隐蔽工程验收资料呢？", [], scenario)
        self.assertEqual(intent.intent, "FAQ_QUERY")
        self.assertEqual(intent.suggested_source, "quality")

    def test_table_query_signal_does_not_steal_pricing_prompt_category(self) -> None:
        self.assertTrue(is_table_query("付款节点表里质保金金额是多少？"))
        self.assertEqual(infer_question_category("付款节点表里质保金金额是多少？"), "pricing")
        intent = IntentResult(intent="FAQ_QUERY", confidence=0.84, reason="unit_test", suggested_source="contract")
        plan = build_retrieval_plan("付款节点表里质保金金额是多少？", intent)
        self.assertTrue(plan.prefer_table)
        self.assertTrue(plan.faq_direct_exact_only)
        self.assertEqual(plan.question_category, "pricing")


class PromptProfileTests(unittest.TestCase):
    """验证最终回答模板按问题风险和意图确定性选择。"""

    def test_pricing_question_uses_pricing_guard_before_intent_profile(self) -> None:
        profile = build_answer_prompt_profile("FAQ_QUERY", query="发票和退款规则是什么")
        self.assertEqual(profile.name, "pricing_guard")
        self.assertIn("已确认", profile.system_template)

    def test_business_fund_risk_questions_use_pricing_guard(self) -> None:
        """资金、结算和付款承诺类问题即使是 FAQ，也要使用强口径模板。"""
        queries = [
            "没有预算审批可以先采购再报销吗？",
            "信用证软条款有什么风险？",
            "投标保证金异常需要关注什么？",
            "收款账户和被保险人不一致可以打款吗？",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(infer_question_category(query), "pricing")
                profile = build_answer_prompt_profile("FAQ_QUERY", query=query)
                self.assertEqual(profile.name, "pricing_guard")

    def test_business_compliance_questions_use_compliance_guard(self) -> None:
        """监管、安全责任和资料真实性问题要使用合规模板，不按普通知识问答处理。"""
        queries = [
            "受限空间作业前需要哪些安全确认？",
            "HS 编码归类存在争议时能先按客户说法申报吗？",
            "最终用途不清楚可以继续发货吗？",
            "既往症未如实告知会有什么影响？",
            "检验批资料和现场实物不一致怎么办？",
            "安全技术交底只有口头说明可以吗？",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(infer_question_category(query), "compliance")
                profile = build_answer_prompt_profile("KNOWLEDGE_QUERY", query=query)
                self.assertEqual(profile.name, "compliance_guard")

    def test_faq_intent_uses_faq_answer_profile(self) -> None:
        profile = build_answer_prompt_profile("FAQ_QUERY", query="怎么修改账号密码")
        self.assertEqual(profile.name, "faq_answer")

    def test_knowledge_intent_uses_knowledge_profile(self) -> None:
        profile = build_answer_prompt_profile("KNOWLEDGE_QUERY", query="入职流程怎么走")
        self.assertEqual(profile.name, "knowledge_answer")

    def test_unknown_intent_uses_default_profile(self) -> None:
        profile = build_answer_prompt_profile("UNKNOWN", query="请介绍一下")
        self.assertEqual(profile.name, "default_answer")
