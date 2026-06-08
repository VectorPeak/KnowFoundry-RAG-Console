"""主问答链路评测门禁。

评测报告负责量化 RAG 主链路表现，评测门禁负责把这些指标变成可执行的通过/失败
标准。它可以读取历史评测报告，也可以现场调用 `QAService` 跑一遍评测集。

和入库质量门禁的区别：
- 入库质量门禁关注“资料有没有被正确解析、切分、版本化”；
- 评测门禁关注“真实问题经过意图识别、检索、重排、Prompt 和流式生成后是否稳定”。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import print_json, read_json_file, write_json_file, write_optional_json
from scripts.evaluate_core_chain import default_output_path, evaluate_dataset
from scripts.gate_utils import add_max_failure, add_min_failure


@dataclass(frozen=True)
class EvaluationGateThresholds:
    """主链路评测门禁阈值。

    默认值适合 smoke 级别评测：必须无错误，召回、关键词覆盖和场景隔离不能明显退化。
    如果后续扩展成大规模评测集，可以在 CI 中按环境调整这些阈值。
    """

    min_recall_at_k: float = 0.8
    min_mrr: float = 0.6
    min_keyword_coverage: float = 0.7
    min_hit_type_accuracy: float = 0.7
    min_source_inference_accuracy: float = 0.7
    min_prompt_profile_accuracy: float = 0.7
    min_faq_direct_accuracy: float = 0.7
    min_scenario_isolation_accuracy: float = 1.0
    max_error_rate: float = 0.0
    max_avg_elapsed_ms: float = 60000.0
    min_scenario_recall_at_k: float = 0.8
    min_scenario_mrr: float = 0.6
    min_scenario_keyword_coverage: float = 0.7
    max_scenario_error_rate: float = 0.0
    min_source_recall_at_k: float = 0.8
    min_source_mrr: float = 0.6
    min_source_keyword_coverage: float = 0.7
    max_source_error_rate: float = 0.0
    min_hit_type_group_accuracy: float = 0.7


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    """读取评测报告 JSON。"""
    return read_json_file(path)


def _metric(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全读取数值指标。"""
    value = report.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_key(value: Any) -> str:
    """规范化分组名称，避免空值把分组指标写得不清楚。"""
    return str(value or "").strip() or "unknown"


def _truthy(value: Any) -> bool:
    """把评测行里的布尔字段转成明确 bool。"""
    return value is True or str(value).lower() == "true"


def _matched_value(row: dict[str, Any], match_field: str, expected_field: str, actual_field: str) -> bool:
    """读取评测脚本写入的明细匹配结果。

    这里要求评测报告显式写出 `*_matched` 字段。原因是门禁脚本负责验收“当前标准”
    下的报告质量，不再根据 expected/actual 临时反推结果，避免学生同时学习两套报告格式。
    `expected_field` 和 `actual_field` 只保留在函数签名中，用于让调用处能直接看出该匹配项的业务含义。
    """
    _ = (expected_field, actual_field)
    return _truthy(row.get(match_field))


def _rows_group_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """从评测明细行计算一个分组的质量指标。

    使用场景：
    - 按场景看某个业务包是否退化；
    - 按 source 看某类资料是否召回变差；
    - 按 expected_hit_type 看 FAQ 直出、RAG、边界识别是否各自稳定。

    为什么不只看全局均值：真实项目里某个小场景坏掉时，全局 Recall@K 可能仍然很好。
    分组门禁可以把这种“局部退化被平均值掩盖”的问题直接暴露出来。
    """
    total = len(rows)
    source_rows = [row for row in rows if row.get("source_recall_hit") is not None]
    hit_type_rows = [row for row in rows if row.get("expected_hit_type")]
    source_inference_rows = [row for row in rows if row.get("expected_effective_source")]
    prompt_profile_rows = [row for row in rows if row.get("expected_prompt_profile")]
    keyword_rows = [row for row in rows if "keyword_coverage" in row]
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "total": total,
        "errors": errors,
        "error_rate": round(errors / max(total, 1), 4),
        "recall_at_k": round(
            sum(1 for row in source_rows if _truthy(row.get("source_recall_hit"))) / max(len(source_rows), 1),
            4,
        ) if source_rows else 1.0,
        "mrr": round(
            sum(float(row.get("mrr") or 0.0) for row in source_rows) / max(len(source_rows), 1),
            4,
        ) if source_rows else 1.0,
        "avg_keyword_coverage": round(
            sum(float(row.get("keyword_coverage") or 0.0) for row in keyword_rows) / max(len(keyword_rows), 1),
            4,
        ) if keyword_rows else 1.0,
        "hit_type_accuracy": round(
            sum(1 for row in hit_type_rows if _matched_value(row, "hit_type_matched", "expected_hit_type", "hit_type")) / max(len(hit_type_rows), 1),
            4,
        ) if hit_type_rows else 1.0,
        "source_inference_accuracy": round(
            sum(
                1
                for row in source_inference_rows
                if _matched_value(row, "source_inference_matched", "expected_effective_source", "effective_source_filter")
            ) / max(len(source_inference_rows), 1),
            4,
        ) if source_inference_rows else 1.0,
        "prompt_profile_accuracy": round(
            sum(
                1
                for row in prompt_profile_rows
                if _matched_value(row, "prompt_profile_matched", "expected_prompt_profile", "prompt_profile")
            ) / max(len(prompt_profile_rows), 1),
            4,
        ) if prompt_profile_rows else 1.0,
    }


def _derive_group_metrics(report: dict[str, Any]) -> dict[str, dict[str, dict[str, float | int]]]:
    """从评测报告中派生场景、source 和 hit_type 分组指标。"""
    rows = [dict(row) for row in list(report.get("rows") or [])]
    scenario_metrics: dict[str, dict[str, float | int]] = {}
    source_metrics: dict[str, dict[str, float | int]] = {}
    hit_type_metrics: dict[str, dict[str, float | int]] = {}

    if rows:
        scenario_groups: dict[str, list[dict[str, Any]]] = {}
        source_groups: dict[str, list[dict[str, Any]]] = {}
        hit_type_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            scenario_groups.setdefault(_group_key(row.get("scenario_id")), []).append(row)
            source_name = row.get("expected_effective_source") or row.get("source_filter")
            if source_name:
                source_groups.setdefault(_group_key(source_name), []).append(row)
            expected_hit_type = row.get("expected_hit_type")
            if expected_hit_type:
                hit_type_groups.setdefault(_group_key(expected_hit_type), []).append(row)
        scenario_metrics = {name: _rows_group_metrics(items) for name, items in sorted(scenario_groups.items())}
        source_metrics = {name: _rows_group_metrics(items) for name, items in sorted(source_groups.items())}
        hit_type_metrics = {name: _rows_group_metrics(items) for name, items in sorted(hit_type_groups.items())}
    else:
        for name, metrics in dict(report.get("scenario_metrics") or {}).items():
            total = int(metrics.get("total") or 0)
            errors = int(metrics.get("errors") or 0)
            scenario_metrics[str(name)] = {
                "total": total,
                "errors": errors,
                "error_rate": round(errors / max(total, 1), 4),
                "recall_at_k": _metric(metrics, "recall_at_k", 1.0),
                "mrr": _metric(metrics, "mrr", 1.0),
                "avg_keyword_coverage": _metric(metrics, "avg_keyword_coverage", 1.0),
            }
    return {
        "scenario_metrics": scenario_metrics,
        "source_metrics": source_metrics,
        "hit_type_metrics": hit_type_metrics,
    }


def _add_group_min_failure(
    failures: list[dict[str, Any]],
    *,
    group_type: str,
    group_name: str,
    metric: str,
    actual: float,
    minimum: float,
    message: str,
) -> None:
    """追加分组最小值门禁失败。"""
    add_min_failure(
        failures,
        metric=f"{group_type}.{group_name}.{metric}",
        actual=actual,
        minimum=minimum,
        message=message,
    )


def _add_group_max_failure(
    failures: list[dict[str, Any]],
    *,
    group_type: str,
    group_name: str,
    metric: str,
    actual: float,
    maximum: float,
    message: str,
) -> None:
    """追加分组最大值门禁失败。"""
    add_max_failure(
        failures,
        metric=f"{group_type}.{group_name}.{metric}",
        actual=actual,
        maximum=maximum,
        message=message,
    )


def add_group_failures(
    failures: list[dict[str, Any]],
    group_metrics: dict[str, dict[str, dict[str, float | int]]],
    thresholds: EvaluationGateThresholds,
) -> None:
    """检查场景、source、hit_type 三类分组门禁。"""
    for scenario, metrics in group_metrics["scenario_metrics"].items():
        _add_group_max_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="error_rate",
            actual=float(metrics.get("error_rate") or 0.0),
            maximum=thresholds.max_scenario_error_rate,
            message="该业务场景出现错误，不能被全局错误率掩盖。",
        )
        _add_group_min_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="recall_at_k",
            actual=float(metrics.get("recall_at_k") or 0.0),
            minimum=thresholds.min_scenario_recall_at_k,
            message="该业务场景召回退化，需要检查该场景资料、source 规则或知识库版本。",
        )
        _add_group_min_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="mrr",
            actual=float(metrics.get("mrr") or 0.0),
            minimum=thresholds.min_scenario_mrr,
            message="该业务场景预期来源排名靠后，需要检查重排或 chunk 粒度。",
        )
        _add_group_min_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="avg_keyword_coverage",
            actual=float(metrics.get("avg_keyword_coverage") or 0.0),
            minimum=thresholds.min_scenario_keyword_coverage,
            message="该业务场景答案关键事实覆盖不足。",
        )

    for source, metrics in group_metrics["source_metrics"].items():
        _add_group_max_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="error_rate",
            actual=float(metrics.get("error_rate") or 0.0),
            maximum=thresholds.max_source_error_rate,
            message="该 source 资料链路出现错误，可能是分类过滤或资料版本问题。",
        )
        _add_group_min_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="recall_at_k",
            actual=float(metrics.get("recall_at_k") or 0.0),
            minimum=thresholds.min_source_recall_at_k,
            message="该 source 召回退化，需要检查 source 推断、过滤条件和资料覆盖。",
        )
        _add_group_min_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="mrr",
            actual=float(metrics.get("mrr") or 0.0),
            minimum=thresholds.min_source_mrr,
            message="该 source 的预期来源排名靠后，需要检查 dense/sparse 权重或 rerank。",
        )
        _add_group_min_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="avg_keyword_coverage",
            actual=float(metrics.get("avg_keyword_coverage") or 0.0),
            minimum=thresholds.min_source_keyword_coverage,
            message="该 source 的答案关键事实覆盖不足。",
        )

    for hit_type, metrics in group_metrics["hit_type_metrics"].items():
        _add_group_min_failure(
            failures,
            group_type="hit_type",
            group_name=hit_type,
            metric="hit_type_accuracy",
            actual=float(metrics.get("hit_type_accuracy") or 0.0),
            minimum=thresholds.min_hit_type_group_accuracy,
            message="该命中路径不稳定，FAQ/RAG/边界识别可能被路由到错误链路。",
        )


def evaluate_report_against_gate(
    report: dict[str, Any],
    thresholds: EvaluationGateThresholds,
    *,
    report_path: str = "",
) -> dict[str, Any]:
    """根据评测指标判断主链路是否通过门禁。"""
    total = int(report.get("total") or 0)
    errors = int(report.get("errors") or 0)
    error_rate = round(errors / max(total, 1), 4)
    metrics = {
        "total": total,
        "errors": errors,
        "error_rate": error_rate,
        "recall_at_k": _metric(report, "recall_at_k"),
        "mrr": _metric(report, "mrr"),
        "avg_keyword_coverage": _metric(report, "avg_keyword_coverage"),
        "hit_type_accuracy": _metric(report, "hit_type_accuracy"),
        "source_inference_accuracy": _metric(report, "source_inference_accuracy", 1.0),
        "prompt_profile_accuracy": _metric(report, "prompt_profile_accuracy", 1.0),
        "faq_direct_accuracy": _metric(report, "faq_direct_accuracy"),
        "scenario_isolation_accuracy": _metric(report, "scenario_isolation_accuracy"),
        "avg_elapsed_ms": _metric(report, "avg_elapsed_ms"),
    }
    failures: list[dict[str, Any]] = []
    group_metrics = _derive_group_metrics(report)

    add_max_failure(
        failures,
        metric="error_rate",
        actual=metrics["error_rate"],
        maximum=thresholds.max_error_rate,
        message="评测样本出现错误，说明主链路依赖、检索或生成阶段不稳定。",
    )
    add_min_failure(
        failures,
        metric="recall_at_k",
        actual=metrics["recall_at_k"],
        minimum=thresholds.min_recall_at_k,
        message="预期来源召回不足，需要检查入库、query_variants、过滤条件或 top_k。",
    )
    add_min_failure(
        failures,
        metric="mrr",
        actual=metrics["mrr"],
        minimum=thresholds.min_mrr,
        message="预期来源排名靠后，需要检查 dense/sparse 权重、rerank 或 chunk 粒度。",
    )
    add_min_failure(
        failures,
        metric="avg_keyword_coverage",
        actual=metrics["avg_keyword_coverage"],
        minimum=thresholds.min_keyword_coverage,
        message="答案关键事实覆盖不足，需要检查上下文构建、Prompt 或模型输出。",
    )
    add_min_failure(
        failures,
        metric="hit_type_accuracy",
        actual=metrics["hit_type_accuracy"],
        minimum=thresholds.min_hit_type_accuracy,
        message="FAQ 直出/RAG/信息不足路径判断不稳定。",
    )
    add_min_failure(
        failures,
        metric="source_inference_accuracy",
        actual=metrics["source_inference_accuracy"],
        minimum=thresholds.min_source_inference_accuracy,
        message="source 自动推断不稳定，可能影响无分类条件下的检索过滤。",
    )
    add_min_failure(
        failures,
        metric="prompt_profile_accuracy",
        actual=metrics["prompt_profile_accuracy"],
        minimum=thresholds.min_prompt_profile_accuracy,
        message="Prompt Profile 路由不稳定，高风险问题可能没有进入严格模板。",
    )
    add_min_failure(
        failures,
        metric="faq_direct_accuracy",
        actual=metrics["faq_direct_accuracy"],
        minimum=thresholds.min_faq_direct_accuracy,
        message="FAQ 标准问答直出能力退化。",
    )
    add_min_failure(
        failures,
        metric="scenario_isolation_accuracy",
        actual=metrics["scenario_isolation_accuracy"],
        minimum=thresholds.min_scenario_isolation_accuracy,
        message="多场景隔离评测不达标，可能出现跨场景误检索。",
    )
    add_max_failure(
        failures,
        metric="avg_elapsed_ms",
        actual=metrics["avg_elapsed_ms"],
        maximum=thresholds.max_avg_elapsed_ms,
        message="平均耗时超过门禁，需要检查模型、Milvus、rerank 或网络延迟。",
    )
    add_group_failures(failures, group_metrics, thresholds)

    return {
        "ok": not failures,
        "report_type": "core_chain_evaluation_gate",
        "report_path": report_path,
        "dataset": report.get("dataset"),
        "created_at": report.get("created_at"),
        "metrics": metrics,
        "group_metrics": group_metrics,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def thresholds_from_args(args: argparse.Namespace) -> EvaluationGateThresholds:
    """把命令行参数转换成评测门禁阈值。"""
    return EvaluationGateThresholds(
        min_recall_at_k=args.min_recall_at_k,
        min_mrr=args.min_mrr,
        min_keyword_coverage=args.min_keyword_coverage,
        min_hit_type_accuracy=args.min_hit_type_accuracy,
        min_source_inference_accuracy=args.min_source_inference_accuracy,
        min_prompt_profile_accuracy=args.min_prompt_profile_accuracy,
        min_faq_direct_accuracy=args.min_faq_direct_accuracy,
        min_scenario_isolation_accuracy=args.min_scenario_isolation_accuracy,
        max_error_rate=args.max_error_rate,
        max_avg_elapsed_ms=args.max_avg_elapsed_ms,
        min_scenario_recall_at_k=args.min_scenario_recall_at_k,
        min_scenario_mrr=args.min_scenario_mrr,
        min_scenario_keyword_coverage=args.min_scenario_keyword_coverage,
        max_scenario_error_rate=args.max_scenario_error_rate,
        min_source_recall_at_k=args.min_source_recall_at_k,
        min_source_mrr=args.min_source_mrr,
        min_source_keyword_coverage=args.min_source_keyword_coverage,
        max_source_error_rate=args.max_source_error_rate,
        min_hit_type_group_accuracy=args.min_hit_type_group_accuracy,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Check QA evaluation report against gate thresholds.")
    parser.add_argument("--report", default="", help="已有评测报告路径。未提供时现场执行评测集。")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_scenario_smoke.json"), help="评测集 JSON。")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="", help="现场评测报告输出路径。")
    parser.add_argument("--gate-output", default="", help="门禁判定摘要输出路径。")
    parser.add_argument("--scenario", default=None, help="默认业务场景 ID。")
    parser.add_argument("--tenant-id", default=None, help="默认租户 ID。")
    parser.add_argument("--dataset-id", default=None, help="默认数据集 ID。")
    parser.add_argument("--visibility", default=None, help="默认可见级别。")
    parser.add_argument("--user-role", default=None, help="默认用户角色。")
    parser.add_argument("--kb-version", default=None, help="可选知识库版本。")
    parser.add_argument("--min-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.6)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--min-hit-type-accuracy", type=float, default=0.7)
    parser.add_argument("--min-source-inference-accuracy", type=float, default=0.7)
    parser.add_argument("--min-prompt-profile-accuracy", type=float, default=0.7)
    parser.add_argument("--min-faq-direct-accuracy", type=float, default=0.7)
    parser.add_argument("--min-scenario-isolation-accuracy", type=float, default=1.0)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-avg-elapsed-ms", type=float, default=60000.0)
    parser.add_argument("--min-scenario-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-scenario-mrr", type=float, default=0.6)
    parser.add_argument("--min-scenario-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--max-scenario-error-rate", type=float, default=0.0)
    parser.add_argument("--min-source-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-source-mrr", type=float, default=0.6)
    parser.add_argument("--min-source-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--max-source-error-rate", type=float, default=0.0)
    parser.add_argument("--min-hit-type-group-accuracy", type=float, default=0.7)
    return parser


def main() -> None:
    """执行评测门禁并按结果设置退出码。"""
    parser = build_parser()
    args = parser.parse_args()
    if args.report:
        report_path = args.report
        report = load_evaluation_report(report_path)
    else:
        output_path = Path(args.output) if args.output else default_output_path(args.dataset)
        report = evaluate_dataset(args)
        report_path = str(output_path)
        write_json_file(output_path, report)

    result = evaluate_report_against_gate(report, thresholds_from_args(args), report_path=report_path)
    write_optional_json(args.gate_output, result)
    print_json(result)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
