"""当前主问答链路的工程回归评测脚本。

该脚本直接调用 QAService，覆盖意图识别、查询改写、检索计划、Milvus 混合检索、
重排、Prompt Profile、流式生成和结束事件。相比只看最终答案，它额外统计：
- Recall@K：预期来源是否被检索到；
- MRR：预期来源排在第几位；
- keyword coverage：答案是否覆盖关键事实；
- hit_type 分布：FAQ 直出、RAG、信息不足的比例；
- 错误率和平均耗时。

它不是学术评测平台，而是工程回归工具：每次改检索策略、切分参数、Prompt 或新增场景
后，用小样本快速判断主链路是否退化。
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
from scripts.eval_common import EvalCaseRuntime, load_eval_items


EVALUATION_REPORT_DIR = Path("reports") / "evaluation"


def normalize_text(value: str) -> str:
    """规范化中文文本，用于近似匹配关键词和来源。"""
    return re.sub(r"[\s，。、“”‘’：:；;,.!?！？（）()【】\[\]「」\-_*]+", "", value or "").lower()


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """计算答案覆盖了多少预期关键词。"""
    keywords = [item for item in expected_keywords if str(item).strip()]
    if not keywords:
        return 0.0
    normalized_answer = normalize_text(answer)
    hits = sum(1 for item in keywords if normalize_text(item) in normalized_answer)
    return round(hits / len(keywords), 4)


def _case_keywords(item: dict[str, Any]) -> list[str]:
    """读取当前评测样本显式维护的预期关键词。

    当前评测集统一使用 `expected_keywords` 描述答案必须覆盖的关键事实。这里不再从
    `ground_truth` 或 `answer` 临时拆词，避免同时维护“标准答案评测”和“关键词评测”
    两套口径。
    """
    explicit = item.get("expected_keywords")
    if isinstance(explicit, list):
        return [str(value) for value in explicit]
    return []


def _source_text(source: dict[str, Any]) -> str:
    """把一次检索命中的可见字段拼成可匹配文本。"""
    metadata = source.get("metadata") or {}
    values = [
        source.get("content"),
        metadata.get("file_name"),
        metadata.get("file_path"),
        metadata.get("standard_question"),
        metadata.get("source"),
        metadata.get("answer"),
    ]
    return " ".join(str(value or "") for value in values)


def find_expected_source_rank(
    sources: list[dict[str, Any]],
    expected_source_contains: list[str],
    *,
    prefer_table: bool = False,
) -> int | None:
    """查找预期来源第一次出现在第几名。

    expected_source_contains 可以写文件名、FAQ 标准问题、source 名或正文关键片段。只要
    任意一个预期片段命中任意来源，即认为召回成功。

    表格专项问题会优先看文档来源。原因是同一个问题可能先召回泛化 FAQ，再召回具体
    表格行；如果按 FAQ+doc 混排计算 MRR，会把“表格行已经第一名命中文档”的情况误算成
    第二名。这里不改变正式问答逻辑，只让评测口径和表格检索目标一致。
    """
    expected = [normalize_text(value) for value in expected_source_contains if str(value).strip()]
    if not expected:
        return None
    ranked_sources = sources
    if prefer_table:
        doc_sources = [source for source in sources if source.get("source_type") == "doc"]
        ranked_sources = doc_sources or sources
    for index, source in enumerate(ranked_sources, start=1):
        text = normalize_text(_source_text(source))
        if any(value in text for value in expected):
            return index
    return None


def _prompt_profile_name(retrieval: dict[str, Any], debug_payload: dict[str, Any]) -> str:
    """从正式回答或调试结果中读取命中的 Prompt Profile 名称。"""
    profile = retrieval.get("prompt_profile") or {}
    if profile.get("name"):
        return str(profile["name"])
    debug_profile = ((debug_payload.get("retrieval_plan") or {}).get("prompt_profile") or {})
    return str(debug_profile.get("name") or "")


def _effective_source_name(
    retrieval: dict[str, Any],
    debug_payload: dict[str, Any],
    hit_type: str,
) -> str:
    """推断评测时应使用的“有效 source”。

    普通 FAQ/RAG 请求直接读取 retrieval/debug 中记录的 source_filter 即可；但边界样本
    的本质是“系统识别出用户当前选错了分类”。如果仍然拿用户原始 source_filter 做
    source_inference_accuracy 评测，就会把正确的边界识别误判成失败。
    """
    if hit_type == "source_boundary":
        boundary = (retrieval.get("source_boundary") or debug_payload.get("source_boundary") or {})
        return str(boundary.get("matched_source") or "")
    if hit_type == "scenario_boundary":
        boundary = (retrieval.get("scenario_boundary") or debug_payload.get("scenario_boundary") or {})
        return str(boundary.get("matched_source") or "")
    return str((retrieval.get("source_filter") or debug_payload.get("source_filter") or "")).strip()


def run_case(
    *,
    service,
    item: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """执行一条评测样本，并同时收集检索诊断和流式答案。"""
    runtime = EvalCaseRuntime.from_item(item, index, args, session_prefix="eval")
    expected_keywords = _case_keywords(item)
    expected_sources = [str(value) for value in item.get("expected_source_contains", [])]
    expected_hit_type = str(item.get("expected_hit_type") or "").strip()
    expected_effective_source = str(item.get("expected_effective_source") or "").strip()
    expected_prompt_profile = str(item.get("expected_prompt_profile") or "").strip()

    started = time.perf_counter()
    answer = ""
    hit_type = ""
    error = ""
    debug_error = ""
    sources: list[dict[str, Any]] = []
    rewritten_query = ""
    intent: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}
    debug_payload: dict[str, Any] = {}

    try:
        # 先跑 debug_retrieval，是为了把“召回有没有命中”与“最终生成有没有说对”分开。
        # 如果 debug 已经没召回预期来源，问题多半在入库、query_variants、过滤或阈值；
        # 如果 debug 命中但最终答案不对，再看 prompt 和模型输出。
        debug_payload = service.debug_retrieval(
            runtime.question,
            runtime.source_filter,
            runtime.session_id,
            **runtime.service_kwargs(),
        )
    except Exception as exc:
        debug_payload = {"error": str(exc)}
        debug_error = str(exc)

    try:
        for event in service.stream_query(
            runtime.question,
            runtime.source_filter,
            runtime.session_id,
            **runtime.service_kwargs(),
        ):
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
    prefer_table = bool(((debug_payload.get("retrieval_plan") or {}).get("prefer_table")) or ((retrieval.get("plan") or {}).get("prefer_table")))
    rank = find_expected_source_rank(debug_sources or sources, expected_sources, prefer_table=prefer_table)
    coverage = keyword_coverage(answer, expected_keywords)
    answer_compact = normalize_text(answer)
    keyword_hits = [keyword for keyword in expected_keywords if normalize_text(keyword) in answer_compact]
    actual_effective_source = _effective_source_name(retrieval, debug_payload, hit_type)
    actual_prompt_profile = _prompt_profile_name(retrieval, debug_payload)

    hit_type_matched = None if not expected_hit_type else expected_hit_type == hit_type
    source_inference_matched = (
        None if not expected_effective_source else expected_effective_source == actual_effective_source
    )
    prompt_profile_matched = (
        None if not expected_prompt_profile else expected_prompt_profile == actual_prompt_profile
    )

    return {
        "case_id": runtime.case_id,
        "question": runtime.question,
        "scenario_id": runtime.scenario_id,
        "source_filter": runtime.source_filter,
        "tenant_id": runtime.tenant_id,
        "dataset_id": runtime.dataset_id,
        "visibility": runtime.visibility,
        "kb_version": runtime.kb_version,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "hit_type": hit_type,
        "expected_hit_type": expected_hit_type,
        "hit_type_matched": hit_type_matched,
        "effective_source_filter": actual_effective_source,
        "expected_effective_source": expected_effective_source,
        "source_inference_matched": source_inference_matched,
        "prompt_profile": actual_prompt_profile,
        "expected_prompt_profile": expected_prompt_profile,
        "prompt_profile_matched": prompt_profile_matched,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "source_count": len(sources),
        "debug_source_count": len(debug_sources),
        "expected_source_rank": rank,
        "source_recall_hit": rank is not None if expected_sources else None,
        "mrr": round(1 / rank, 4) if rank else 0.0,
        "prefer_table": prefer_table,
        "expected_keywords": expected_keywords,
        "keyword_hits": keyword_hits,
        "keyword_coverage": coverage,
        "answer_preview": answer[:300],
        "retrieval": retrieval,
        "debug_retrieval": debug_payload,
        "debug_error": debug_error or str(debug_payload.get("error") or ""),
        "error": error,
    }


def default_output_path(dataset: str) -> Path:
    """构建默认评测报告路径。"""
    EVALUATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = Path(dataset).stem or "eval"
    return EVALUATION_REPORT_DIR / f"{stamp}_{name}.json"


def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """执行评测集并返回完整报告对象。

    该函数把原来写在 main 里的评测主体抽出来，供两类入口复用：
    - `evaluate_core_chain.py`：生成完整评测报告；
    - `check_evaluation_gate.py`：在同一份评测逻辑基础上增加门禁判断。

    为什么不让门禁脚本自己重新实现评测：
    - 召回、MRR、FAQ 直出准确率、场景隔离这些统计口径必须保持一致；
    - 后续新增指标时只改一处，避免报告和门禁看的是两套数字；
    - 评测脚本仍然可以独立运行，不依赖 CI 或门禁工具。
    """
    data = load_eval_items(args.dataset, args.limit)
    service = get_qa_service()
    rows = [run_case(service=service, item=item, index=index, args=args) for index, item in enumerate(data, start=1)]

    evaluated_source_cases = [row for row in rows if row["source_recall_hit"] is not None]
    hit_type_cases = [row for row in rows if row["expected_hit_type"]]
    source_inference_cases = [row for row in rows if row["expected_effective_source"]]
    prompt_profile_cases = [row for row in rows if row["expected_prompt_profile"]]
    hit_type_counts = Counter(row["hit_type"] or "unknown" for row in rows)
    scenario_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario_rows.setdefault(str(row["scenario_id"] or "default"), []).append(row)
    scenario_metrics = {
        scenario: {
            "total": len(items),
            "recall_at_k": round(
                sum(1 for item in items if item["source_recall_hit"]) / max(sum(1 for item in items if item["source_recall_hit"] is not None), 1),
                4,
            ),
            "mrr": round(
                sum(item["mrr"] for item in items if item["source_recall_hit"] is not None)
                / max(sum(1 for item in items if item["source_recall_hit"] is not None), 1),
                4,
            ),
            "avg_keyword_coverage": round(sum(item["keyword_coverage"] for item in items) / max(len(items), 1), 4),
            "errors": sum(1 for item in items if item["error"]),
        }
        for scenario, items in scenario_rows.items()
    }
    faq_cases = [row for row in rows if row["expected_hit_type"] == "faq_direct"]
    expected_scenarios = {str(item.get("scenario_id") or args.scenario or "") for item in data if item.get("scenario_id") or args.scenario}
    returned_scenarios = {str(row["scenario_id"] or "") for row in rows}
    return {
        "report_type": "core_chain_evaluation",
        "dataset": str(Path(args.dataset)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
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
            round(
                sum(1 for row in source_inference_cases if row["source_inference_matched"])
                / len(source_inference_cases),
                4,
            )
            if source_inference_cases
            else 1.0
        ),
        "prompt_profile_accuracy": (
            round(
                sum(1 for row in prompt_profile_cases if row["prompt_profile_matched"])
                / len(prompt_profile_cases),
                4,
            )
            if prompt_profile_cases
            else 1.0
        ),
        "faq_direct_accuracy": round(
            sum(1 for row in faq_cases if row["hit_type"] == "faq_direct") / max(len(faq_cases), 1),
            4,
        ),
        "scenario_isolation_accuracy": round(
            len(expected_scenarios & returned_scenarios) / max(len(expected_scenarios), 1),
            4,
        ),
        "avg_elapsed_ms": round(sum(row["elapsed_ms"] for row in rows) / max(len(rows), 1), 2),
        "hit_type_counts": dict(hit_type_counts),
        "scenario_metrics": scenario_metrics,
        "rows": rows,
    }


def main() -> None:
    """执行数据集评测并输出 JSON 报告。"""
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Evaluate the core LangChain + Milvus QA chain.")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_scenario_smoke.json"), help="JSON list of eval cases.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="", help="Optional JSON output path. Defaults to reports/evaluation.")
    parser.add_argument("--scenario", default=None, help="Default business scenario id.")
    parser.add_argument("--tenant-id", default=None, help="Default tenant/org id used for retrieval.")
    parser.add_argument("--dataset-id", default=None, help="Default dataset id used for retrieval.")
    parser.add_argument("--visibility", default=None, help="Default visibility level used for retrieval.")
    parser.add_argument("--user-role", default=None, help="Default user role used for retrieval.")
    parser.add_argument("--kb-version", default=None, help="Optional knowledge base version for replay.")
    args = parser.parse_args()

    summary = evaluate_dataset(args)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

