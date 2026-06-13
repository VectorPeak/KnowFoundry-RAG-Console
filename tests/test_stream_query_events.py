"""stream_query 的事件级路由契约测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from qa_core.pipeline.rag import stream_query
from qa_core.retrieval.results import RetrievalHit, RetrievalResult


class FakeHistory:
    """最小历史存储假对象，避免测试触达真实 MySQL。"""

    def __init__(self) -> None:
        self.turns: list[tuple[str, str, str]] = []

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        self.turns.append((session_id, query, answer))

    def get_context_messages(self, session_id: str):
        return []


def _noop_validate(source_filter, scenario) -> None:
    return None


class StreamQueryRouteEventTests(unittest.TestCase):
    """验证前端可见事件顺序与查询路由分支保持一致。"""

    def _events(self, query: str, **kwargs):
        history = kwargs.pop("history", FakeHistory())
        with patch("qa_core.pipeline.runtime.resolve_active_kb_version", return_value="kb_test"):
            events = list(
                stream_query(
                    history,
                    _noop_validate,
                    query,
                    kwargs.pop("source_filter", None),
                    kwargs.pop("session_id", "session-test"),
                    scenario_id=kwargs.pop("scenario_id", "enterprise_knowledge"),
                    **kwargs,
                )
            )
        return events, history

    def test_out_of_scope_direct_answer_returns_before_intent_status(self) -> None:
        with patch("qa_core.pipeline.steps.get_faq_store") as get_faq_store:
            events, history = self._events("彩票怎么买")

        self.assertEqual([event["type"] for event in events], ["start", "status", "token", "end"])
        self.assertEqual(events[1]["message"], "正在进行查询路由...")
        self.assertNotIn("正在识别问题意图...", [event.get("message") for event in events])
        self.assertIn("超出了", events[2]["token"])
        self.assertEqual(events[-1]["intent"]["intent"], "OUT_OF_SCOPE")
        self.assertEqual(events[-1]["retrieval"]["route"], "direct_answer")
        self.assertEqual(len(history.turns), 1)
        get_faq_store.assert_not_called()

    def test_faq_exact_route_streams_standard_answer_without_full_intent_status(self) -> None:
        faq_result = RetrievalResult(
            hits=[
                RetrievalHit(
                    document=Document(
                        page_content="员工报销需要准备哪些材料？",
                        metadata={"standard_question": "员工报销需要准备哪些材料？", "answer": "请准备发票、审批单和付款凭证。"},
                    ),
                    score=0.55,
                )
            ],
            query="员工报销需要准备哪些材料？",
            source_type="faq",
        )

        with patch("qa_core.pipeline.steps.get_faq_store") as get_faq_store:
            get_faq_store.return_value.search_many.return_value = faq_result
            events, _history = self._events("员工报销需要准备哪些材料？")

        self.assertEqual([event["type"] for event in events], ["start", "status", "token", "end"])
        self.assertEqual(events[1]["message"], "正在进行查询路由...")
        self.assertNotIn("正在识别问题意图...", [event.get("message") for event in events])
        self.assertEqual(events[2]["token"], "请准备发票、审批单和付款凭证。")
        self.assertEqual(events[-1]["hit_type"], "faq_direct")
        self.assertEqual(events[-1]["intent"]["intent"], "FAQ_QUERY")
        self.assertEqual(events[-1]["retrieval"]["route"], "faq_exact")

    def test_retrieval_route_emits_intent_status_after_query_route(self) -> None:
        history = FakeHistory()

        def fake_search_and_generate(context, prepared, query, history):
            yield {"type": "status", "message": "fake search", "session_id": context.session_id}
            return None

        prepared = SimpleNamespace(intent=SimpleNamespace(direct_answer=None))
        with (
            patch("qa_core.pipeline.runtime.resolve_active_kb_version", return_value="kb_test"),
            patch("qa_core.pipeline.steps.should_try_faq_fast_path", return_value=False),
            patch("qa_core.pipeline.rag.prepare_retrieval", return_value=prepared) as prepare_retrieval,
            patch("qa_core.pipeline.rag._search_and_generate", side_effect=fake_search_and_generate) as search_and_generate,
        ):
            events = list(
                stream_query(
                    history,
                    _noop_validate,
                    "请系统梳理公司新人入职制度、部门协作流程、审批节点、材料归档要求以及常见风险边界。",
                    None,
                    "session-test",
                    scenario_id="enterprise_knowledge",
                )
            )

        self.assertEqual([event["type"] for event in events], ["start", "status", "status", "status"])
        self.assertEqual(events[1]["message"], "正在进行查询路由...")
        self.assertEqual(events[2]["message"], "正在识别问题意图...")
        self.assertEqual(events[3]["message"], "fake search")
        prepare_retrieval.assert_called_once()
        search_and_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
