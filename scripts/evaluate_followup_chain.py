"""多轮追问链路回归评测。

单轮评测能证明“一个完整问题能否答对”，但不能证明历史链路真的可用。真实用户经常
会问“那这个呢”“材料不全呢”“旧版本呢”这类追问，如果追问改写、历史读取或 source
推断有问题，RAG 很容易召回错资料。

本脚本直接调用 QAService.stream_query，并在同一个 session_id 下连续执行多轮问题，
用于验证：

- MySQL 历史是否写入并能被下一轮读取；
- FOLLOW_UP 是否触发追问改写；
- 改写后的问题是否能召回预期来源；
- 高风险追问是否进入正确 Prompt Profile。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.application.factory import get_qa_service
from scripts.common import configure_utf8_stdio
from scripts.evaluate_core_chain import find_expected_source_rank, keyword_coverage, normalize_text


EVALUATION_REPORT_DIR = Path("reports") / "evaluation"


def default_output_path(dataset: str) -> Path:
    """构建默认多轮追问评测报告路径。"""
    EVALUATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = Path(dataset).stem or "followup"
    return EVALUATION_REPORT_DIR / f"{stamp}_{name}.json"


def _case_keywords(turn: dict[str, Any]) -> list[str]:
    """读取追问样本中的预期关键词。"""
    explicit = turn.get("expected_keywords")
    if isinstance(explicit, list):
        return [str(value) for value in explicit]
    ground_truth = str(turn.get("ground_truth") or turn.get("answer") or "").strip()
    return [value for value in re.split(r"[、,，。；;\s]+", ground_truth) if value]


def _prompt_profile_name(retrieval: dict[str, Any], debug_payload: dict[str, Any]) -> str:
    """从正式回答或调试结果中读取 Prompt Profile 名称。"""
    profile = retrieval.get("prompt_profile") or {}
    if profile.get("name"):
        return str(profile["name"])
    debug_profile = ((debug_payload.get("retrieval_plan") or {}).get("prompt_profile") or {})
    return str(debug_profile.get("name") or "")


def _service_kwargs(conversation: dict[str, Any], turn: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """合并会话级和轮次级运行参数。"""
    return {
        "kb_version": turn.get("kb_version") or conversation.get("kb_version") or args.kb_version,
        "scenario_id": turn.get("scenario_id") or conversation.get("scenario_id") or args.scenario,
        "tenant_id": turn.get("tenant_id") or conversation.get("tenant_id") or args.tenant_id,
        "dataset_id": turn.get("dataset_id") or conversation.get("dataset_id") or args.dataset_id,
        "visibility": turn.get("visibility") or conversation.get("visibility") or args.visibility,
        "user_role": turn.get("user_role") or conversation.get("user_role") or args.user_role,
    }


def run_turn(
    *,
    service,
    conversation: dict[str, Any],
    turn: dict[str, Any],
    conversation_index: int,
    turn_index: int,
    session_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """执行一轮追问样本，并返回诊断指标。"""
    question = str(turn.get("question") or turn.get("query") or "").strip()
    source_filter = turn.get("source_filter")
    expected_keywords = _case_keywords(turn)
    expected_sources = [str(value) for value in turn.get("expected_source_contains", [])]
    expected_hit_type = str(turn.get("expected_hit_type") or "").strip()
    expected_effective_source = str(turn.get("expected_effective_source") or "").strip()
    expected_prompt_profile = str(turn.get("expected_prompt_profile") or "").strip()
    service_kwargs = _service_kwargs(conversation, turn, args)

    started = time.perf_counter()
    answer = ""
    hit_type = ""
    rewritten_query = ""
    error = ""
    sources: list[dict[str, Any]] = []
    intent: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}
    debug_payload: dict[str, Any] = {}

    try:
        debug_payload = service.debug_retrieval(question, source_filter, session_id, **service_kwargs)
    except Exception as exc:
        debug_payload = {"error": str(exc)}

    try:
        for event in service.stream_query(question, source_filter, session_id, **service_kwargs):
            if event["type"] == "token":
                answer += event.get("token", "")
            elif event["type"] == "end":
                hit_type = event.get("hit_type", "")
                sources = event.get("sources", [])
                rewritten_query = event.get("rewritten_query") or ""
                intent = event.get("intent") or {}
                retrieval = event.get("retrieval") or {}
            elif event["type"] == "error":
                error = event.get("error", "")
                break
    except Exception as exc:
        error = str(exc)

    debug_sources = list(debug_payload.get("faq_sources") or []) + list(debug_payload.get("doc_sources") or [])
    rank = find_expected_source_rank(debug_sources or sources, expected_sources)
    coverage = keyword_coverage(answer, expected_keywords)
    answer_compact = normalize_text(answer)
    keyword_hits = [keyword for keyword in expected_keywords if normalize_text(keyword) in answer_compact]
    actual_effective_source = str((retrieval.get("source_filter") or debug_payload.get("source_filter") or "")).strip()
    actual_prompt_profile = _prompt_profile_name(retrieval, debug_payload)
    expected_scenario_id = str(service_kwargs["scenario_id"] or "")
    actual_scenario_id = str(retrieval.get("scenario_id") or debug_payload.get("scenario_id") or "")

    return {
        "conversation_id": str(conversation.get("conversation_id") or f"conversation_{conversation_index}"),
        "turn_id": str(turn.get("turn_id") or f"turn_{turn_index}"),
        "turn_index": turn_index,
        "session_id": session_id,
        "question": question,
        "scenario_id": service_kwargs["scenario_id"],
        "actual_scenario_id": actual_scenario_id,
        "scenario_isolation_matched": expected_scenario_id == actual_scenario_id,
        "source_filter": source_filter,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "hit_type": hit_type,
        "expected_hit_type": expected_hit_type,
        "hit_type_matched": None if not expected_hit_type else expected_hit_type == hit_type,
        "effective_source_filter": actual_effective_source,
        "expected_effective_source": expected_effective_source,
        "source_inference_matched": None if not expected_effective_source else expected_effective_source == actual_effective_source,
        "prompt_profile": actual_prompt_profile,
        "expected_prompt_profile": expected_prompt_profile,
        "prompt_profile_matched": None if not expected_prompt_profile else expected_prompt_profile == actual_prompt_profile,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "source_count": len(sources),
        "debug_source_count": len(debug_sources),
        "expected_source_rank": rank,
        "source_recall_hit": rank is not None if expected_sources else None,
        "mrr": round(1 / rank, 4) if rank else 0.0,
        "expected_keywords": expected_keywords,
        "keyword_hits": keyword_hits,
        "keyword_coverage": coverage,
        "answer_preview": answer[:300],
        "retrieval": retrieval,
        "debug_retrieval": debug_payload,
        "error": error or str(debug_payload.get("error") or ""),
    }


def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """执行多轮追问评测集。"""
    conversations = json.loads(Path(args.dataset).read_text(encoding="utf-8"))[: args.limit]
    service = get_qa_service()
    rows: list[dict[str, Any]] = []
    for conversation_index, conversation in enumerate(conversations, start=1):
        session_id = str(conversation.get("session_id") or f"followup-{int(time.time())}-{conversation_index}")
        for turn_index, turn in enumerate(conversation.get("turns") or [], start=1):
            rows.append(
                run_turn(
                    service=service,
                    conversation=conversation,
                    turn=turn,
                    conversation_index=conversation_index,
                    turn_index=turn_index,
                    session_id=session_id,
                    args=args,
                )
            )

    evaluated_source_cases = [row for row in rows if row["source_recall_hit"] is not None]
    hit_type_cases = [row for row in rows if row["expected_hit_type"]]
    source_inference_cases = [row for row in rows if row["expected_effective_source"]]
    prompt_profile_cases = [row for row in rows if row["expected_prompt_profile"]]
    followup_rows = [row for row in rows if row["turn_index"] > 1]
    rewritten_followups = [row for row in followup_rows if row["rewritten_query"] and row["rewritten_query"] != row["question"]]
    followup_source_cases = [row for row in followup_rows if row["expected_effective_source"]]
    hit_type_counts = Counter(row["hit_type"] or "unknown" for row in rows)
    return {
        "report_type": "followup_chain_evaluation",
        "dataset": str(Path(args.dataset)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_conversations": len(conversations),
        "total": len(rows),
        "followup_turns": len(followup_rows),
        "followup_rewrite_rate": round(len(rewritten_followups) / max(len(followup_rows), 1), 4),
        "followup_source_accuracy": (
            round(sum(1 for row in followup_source_cases if row["source_inference_matched"]) / len(followup_source_cases), 4)
            if followup_source_cases
            else 1.0
        ),
        "scenario_isolation_accuracy": round(
            sum(1 for row in rows if row["scenario_isolation_matched"]) / max(len(rows), 1),
            4,
        ),
        "errors": sum(1 for row in rows if row["error"]),
        "avg_keyword_coverage": round(sum(row["keyword_coverage"] for row in rows) / max(len(rows), 1), 4),
        "recall_at_k": round(
            sum(1 for row in evaluated_source_cases if row["source_recall_hit"]) / max(len(evaluated_source_cases), 1),
            4,
        ),
        "mrr": round(sum(row["mrr"] for row in evaluated_source_cases) / max(len(evaluated_source_cases), 1), 4),
        "hit_type_accuracy": round(
            sum(1 for row in hit_type_cases if row["hit_type_matched"]) / max(len(hit_type_cases), 1),
            4,
        ),
        "source_inference_accuracy": (
            round(sum(1 for row in source_inference_cases if row["source_inference_matched"]) / len(source_inference_cases), 4)
            if source_inference_cases
            else 1.0
        ),
        "prompt_profile_accuracy": (
            round(sum(1 for row in prompt_profile_cases if row["prompt_profile_matched"]) / len(prompt_profile_cases), 4)
            if prompt_profile_cases
            else 1.0
        ),
        "avg_elapsed_ms": round(sum(row["elapsed_ms"] for row in rows) / max(len(rows), 1), 2),
        "hit_type_counts": dict(hit_type_counts),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Evaluate multi-turn follow-up QA chain.")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_turn_followup_regression.json"), help="多轮追问评测集 JSON。")
    parser.add_argument("--limit", type=int, default=20, help="最多执行多少段会话。")
    parser.add_argument("--output", default="", help="输出报告路径。默认写入 reports/evaluation。")
    parser.add_argument("--scenario", default=None, help="默认业务场景 ID。")
    parser.add_argument("--tenant-id", default=None, help="默认租户 ID。")
    parser.add_argument("--dataset-id", default=None, help="默认数据集 ID。")
    parser.add_argument("--visibility", default=None, help="默认可见级别。")
    parser.add_argument("--user-role", default=None, help="默认用户角色。")
    parser.add_argument("--kb-version", default=None, help="可选知识库版本。")
    return parser


def main() -> None:
    """执行多轮追问评测并输出 JSON 报告。"""
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    summary = evaluate_dataset(args)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
