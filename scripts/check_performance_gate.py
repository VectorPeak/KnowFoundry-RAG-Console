"""主问答链路性能门禁。

性能基线负责采集事实，性能门禁负责把事实变成可执行的通过/失败标准。它关注的是
用户体感和主链路稳定性：

- 首 token 是否过慢；
- 总耗时是否过慢；
- 是否出现错误；
- 是否能采集到阶段耗时，用于定位瓶颈。

该脚本读取 `collect_performance_baseline.py` 生成的报告，也可以现场采集一份小样本报告。
现场采集会真实调用 QAService.stream_query，因此需要 Milvus、MySQL、本地模型和 LLM
全部可用；环境缺失时应该失败，不提供假数据降级。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_performance_baseline import collect_baseline, default_output_path
from scripts.common import print_json, read_json_file, write_json_file, write_optional_json
from scripts.gate_utils import add_max_failure, add_required_failure


@dataclass(frozen=True)
class PerformanceGateThresholds:
    """性能门禁阈值。

    默认值按本地教学环境设置，重点防止明显退化，而不是追求线上 SLA。真实生产环境可在
    CI 中按机器规格和模型服务延迟重新配置。
    """

    max_error_rate: float = 0.0
    max_avg_total_ms: float = 15000.0
    max_p95_total_ms: float = 30000.0
    max_avg_first_token_ms: float = 8000.0
    max_p95_first_token_ms: float = 15000.0
    require_stage_timings: bool = True


def _metric(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全读取性能报告中的数值指标。"""
    try:
        return float(report.get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate_report_against_gate(
    report: dict[str, Any],
    thresholds: PerformanceGateThresholds,
    *,
    report_path: str = "",
) -> dict[str, Any]:
    """根据性能指标判断主链路是否通过门禁。"""
    total = int(report.get("total") or 0)
    errors = int(report.get("errors") or 0)
    error_rate = round(errors / max(total, 1), 4)
    stage_timings = report.get("avg_stage_timings_ms") or {}
    metrics = {
        "total": total,
        "errors": errors,
        "error_rate": error_rate,
        "avg_total_ms": _metric(report, "avg_total_ms"),
        "p95_total_ms": _metric(report, "p95_total_ms"),
        "avg_first_token_ms": _metric(report, "avg_first_token_ms"),
        "p95_first_token_ms": _metric(report, "p95_first_token_ms"),
        "avg_stage_timings_ms": stage_timings,
    }
    failures: list[dict[str, Any]] = []

    add_max_failure(
        failures,
        metric="error_rate",
        actual=metrics["error_rate"],
        maximum=thresholds.max_error_rate,
        message="性能样本出现错误，说明真实主链路不稳定。",
    )
    add_max_failure(
        failures,
        metric="avg_total_ms",
        actual=metrics["avg_total_ms"],
        maximum=thresholds.max_avg_total_ms,
        message="平均总耗时超过门禁，需要检查 Milvus、rerank、模型生成或网络延迟。",
    )
    add_max_failure(
        failures,
        metric="p95_total_ms",
        actual=metrics["p95_total_ms"],
        maximum=thresholds.max_p95_total_ms,
        message="P95 总耗时过高，说明存在明显慢请求。",
    )
    add_max_failure(
        failures,
        metric="avg_first_token_ms",
        actual=metrics["avg_first_token_ms"],
        maximum=thresholds.max_avg_first_token_ms,
        message="平均首 token 过慢，页面流式体验会变差。",
    )
    add_max_failure(
        failures,
        metric="p95_first_token_ms",
        actual=metrics["p95_first_token_ms"],
        maximum=thresholds.max_p95_first_token_ms,
        message="P95 首 token 过慢，需要看阶段耗时定位慢点。",
    )
    add_required_failure(
        failures,
        metric="avg_stage_timings_ms",
        actual=stage_timings,
        enabled=thresholds.require_stage_timings,
        message="性能报告缺少阶段耗时，无法定位瓶颈。",
    )

    return {
        "ok": not failures,
        "report_type": "core_chain_performance_gate",
        "report_path": report_path,
        "dataset": report.get("dataset"),
        "created_at": report.get("created_at"),
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def thresholds_from_args(args: argparse.Namespace) -> PerformanceGateThresholds:
    """把命令行参数转换成性能门禁阈值。"""
    return PerformanceGateThresholds(
        max_error_rate=args.max_error_rate,
        max_avg_total_ms=args.max_avg_total_ms,
        max_p95_total_ms=args.max_p95_total_ms,
        max_avg_first_token_ms=args.max_avg_first_token_ms,
        max_p95_first_token_ms=args.max_p95_first_token_ms,
        require_stage_timings=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Check QA performance report against gate thresholds.")
    parser.add_argument("--report", default="", help="已有性能报告路径。未提供时现场执行性能采集。")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_scenario_smoke.json"), help="性能样本 JSON。")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="", help="现场性能报告输出路径。")
    parser.add_argument("--gate-output", default="", help="门禁判定摘要输出路径。")
    parser.add_argument("--scenario", default=None, help="默认业务场景 ID。")
    parser.add_argument("--tenant-id", default=None, help="默认租户 ID。")
    parser.add_argument("--dataset-id", default=None, help="默认数据集 ID。")
    parser.add_argument("--visibility", default=None, help="默认可见级别。")
    parser.add_argument("--user-role", default=None, help="默认用户角色。")
    parser.add_argument("--kb-version", default=None, help="可选知识库版本。")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="不执行预热，直接把首条请求也纳入性能统计。该参数会原样透传给性能采集脚本。",
    )
    parser.add_argument("--allow-errors", action="store_true", help="现场采集允许样本错误，但门禁仍按错误率判断。")
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-avg-total-ms", type=float, default=15000.0)
    parser.add_argument("--max-p95-total-ms", type=float, default=30000.0)
    parser.add_argument("--max-avg-first-token-ms", type=float, default=8000.0)
    parser.add_argument("--max-p95-first-token-ms", type=float, default=15000.0)
    return parser


def main() -> None:
    """执行性能门禁并按结果设置退出码。"""
    parser = build_parser()
    args = parser.parse_args()
    if args.report:
        report_path = args.report
        report = read_json_file(report_path)
    else:
        output_path = Path(args.output) if args.output else default_output_path(args.dataset)
        report = collect_baseline(args)
        report_path = str(output_path)
        write_json_file(output_path, report)

    result = evaluate_report_against_gate(report, thresholds_from_args(args), report_path=report_path)
    write_optional_json(args.gate_output, result)
    print_json(result)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
