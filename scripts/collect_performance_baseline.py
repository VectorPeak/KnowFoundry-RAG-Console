"""采集主问答链路性能基线。

性能基线不是为了追求某一次跑分好看，而是给后续优化留下可对比的事实：
- 首 token 耗时：用户什么时候第一次看到答案；
- 总耗时：一次回答完整结束需要多久；
- hit_type 分布：FAQ 直出、RAG、信息不足分别占多少；
- token 数和来源数：判断回答是否真的走了流式生成和引用返回。

该脚本会真实调用 QAService.stream_query，因此需要 Milvus、MySQL、本地模型和 LLM 全部
可用。没有这些前置环境时应该失败，而不是退回到假数据。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.application.factory import get_qa_service
from qa_core.config.settings import PROJECT_ROOT
from scripts.common import configure_utf8_stdio
from scripts.eval_common import EvalCaseRuntime, load_eval_items


PERFORMANCE_REPORT_DIR = PROJECT_ROOT / "reports" / "performance"


def percentile(values: list[float], ratio: float) -> float:
    """计算简单百分位值。

    数据集通常是 smoke 级别小样本，这里采用 nearest-rank 方法。它比四舍五入下标更
    保守，12 条样本的 P95 会取最大慢请求，避免报告把明显慢样本“抹平”。
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return round(ordered[index], 2)


def warmup_service(service, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """用第一条样本预热主链路，但不纳入统计。

    本地教学环境里，首次请求会加载模型、Milvus collection 或连接池。这个成本对用户
    冷启动有意义，但会扭曲“稳定运行阶段”的性能基线。因此默认先跑一条 warmup，并在
    报告中记录它的耗时，让读者知道冷启动成本没有被隐藏。
    """
    return run_case(service, item, 0, args)


def average_stage_timings(rows: list[dict[str, Any]]) -> dict[str, float]:
    """汇总每个主链路阶段的平均耗时。"""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        timings = row.get("stage_timings_ms") or {}
        if not isinstance(timings, dict):
            continue
        for stage_name, value in timings.items():
            try:
                elapsed = float(value)
            except (TypeError, ValueError):
                continue
            totals[str(stage_name)] = totals.get(str(stage_name), 0.0) + elapsed
            counts[str(stage_name)] = counts.get(str(stage_name), 0) + 1
    return {stage_name: round(total / max(counts[stage_name], 1), 2) for stage_name, total in totals.items()}


def default_output_path(dataset: str) -> Path:
    """构建默认性能报告路径。"""
    PERFORMANCE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = Path(dataset).stem or "baseline"
    return PERFORMANCE_REPORT_DIR / f"{stamp}_{name}.json"


def run_case(service, item: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any]:
    """执行一条性能样本，并记录首 token 和总耗时。

    这里只做性能观测，不判断答案正确性。答案正确性由 `evaluate_core_chain.py` 和
    `check_evaluation_gate.py` 负责，两个脚本关注点不要混在一起。
    """
    runtime = EvalCaseRuntime.from_item(item, index, args, session_prefix="perf")
    started = time.perf_counter()
    first_token_ms: float | None = None
    token_count = 0
    answer_chars = 0
    hit_type = ""
    source_count = 0
    retrieval: dict[str, Any] = {}
    event_types: list[str] = []
    error = ""

    try:
        for event in service.stream_query(
            runtime.question,
            runtime.source_filter,
            runtime.session_id,
            **runtime.service_kwargs(),
        ):
            event_type = str(event.get("type") or "")
            event_types.append(event_type)
            if event_type == "token":
                token = str(event.get("token") or "")
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                token_count += 1
                answer_chars += len(token)
            elif event_type == "end":
                hit_type = str(event.get("hit_type") or "")
                source_count = len(event.get("sources") or [])
                retrieval = event.get("retrieval") or {}
            elif event_type == "error":
                error = str(event.get("error") or "")
                break
    except Exception as exc:
        error = str(exc)

    total_ms = (time.perf_counter() - started) * 1000
    return {
        "case_id": runtime.case_id,
        "question": runtime.question,
        "scenario_id": runtime.scenario_id,
        "source_filter": runtime.source_filter,
        "kb_version": runtime.kb_version,
        "total_ms": round(total_ms, 2),
        "first_token_ms": round(first_token_ms or total_ms, 2),
        "token_count": token_count,
        "answer_chars": answer_chars,
        "hit_type": hit_type,
        "source_count": source_count,
        "stage_timings_ms": retrieval.get("stage_timings_ms") or {},
        "slowest_stage": retrieval.get("slowest_stage") or {},
        "context_count": retrieval.get("context_count"),
        "context_chars": retrieval.get("context_chars"),
        "prompt_profile": (retrieval.get("prompt_profile") or {}).get("name"),
        "event_types": event_types,
        "error": error,
    }


def collect_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """运行性能样本并汇总基线指标。"""
    data = load_eval_items(args.dataset, args.limit)
    service = get_qa_service()
    warmup_row = None
    measure_data = data
    if data and not args.no_warmup:
        warmup_row = warmup_service(service, data[0], args)
        measure_data = data[1:] if len(data) > 1 else data
    rows = [run_case(service, item, index, args) for index, item in enumerate(measure_data, start=1)]
    total_values = [row["total_ms"] for row in rows]
    first_token_values = [row["first_token_ms"] for row in rows]
    hit_type_counts = Counter(str(row.get("hit_type") or "unknown") for row in rows)
    prompt_profile_counts = Counter(str(row.get("prompt_profile") or "unknown") for row in rows if row.get("prompt_profile"))
    slowest_stage_counts = Counter(str((row.get("slowest_stage") or {}).get("name") or "unknown") for row in rows if row.get("slowest_stage"))
    errors = [row for row in rows if row.get("error")]
    summary = {
        "report_type": "performance_baseline",
        "dataset": str(Path(args.dataset)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "warmup_enabled": bool(data and not args.no_warmup),
        "warmup_row": warmup_row,
        "errors": len(errors),
        "error_rate": round(len(errors) / max(len(rows), 1), 4),
        "avg_total_ms": round(statistics.fmean(total_values), 2) if total_values else 0.0,
        "p95_total_ms": percentile(total_values, 0.95),
        "avg_first_token_ms": round(statistics.fmean(first_token_values), 2) if first_token_values else 0.0,
        "p95_first_token_ms": percentile(first_token_values, 0.95),
        "avg_token_count": round(statistics.fmean([row["token_count"] for row in rows]), 2) if rows else 0.0,
        "avg_source_count": round(statistics.fmean([row["source_count"] for row in rows]), 2) if rows else 0.0,
        "avg_stage_timings_ms": average_stage_timings(rows),
        "hit_type_counts": dict(hit_type_counts),
        "prompt_profile_counts": dict(prompt_profile_counts),
        "slowest_stage_counts": dict(slowest_stage_counts),
        "rows": rows,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Collect QA stream performance baseline.")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_scenario_smoke.json"), help="JSON list of baseline cases.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="", help="Optional JSON output path. Defaults to reports/performance.")
    parser.add_argument("--scenario", default=None, help="Default business scenario id.")
    parser.add_argument("--tenant-id", default=None, help="Default tenant/org id used for retrieval.")
    parser.add_argument("--dataset-id", default=None, help="Default dataset id used for retrieval.")
    parser.add_argument("--visibility", default=None, help="Default visibility level used for retrieval.")
    parser.add_argument("--user-role", default=None, help="Default user role used for retrieval.")
    parser.add_argument("--kb-version", default=None, help="Optional knowledge base version for replay.")
    parser.add_argument("--no-warmup", action="store_true", help="不执行预热，冷启动请求也纳入统计。")
    parser.add_argument("--allow-errors", action="store_true", help="采集报告但不因样本错误退出。")
    return parser


def main() -> None:
    """执行性能基线采集并输出 JSON 报告。"""
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    summary = collect_baseline(args)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(payload)
    if summary["errors"] and not args.allow_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
