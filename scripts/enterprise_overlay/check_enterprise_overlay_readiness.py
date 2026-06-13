"""企业资料增强包就绪门禁。

这个脚本把企业资料增强相关的三个离线报告和一个回归评测集串成发布前检查：
- `enterprise_data_realism_latest.json`：资料真实度是否有提升；
- `enterprise_overlay_build_latest.json`：clean overlay 是否能合并并通过入库质量门禁；
- `dirty_enterprise_samples_latest.json`：dirty samples 是否全部保持 active 入库阻断；
- `eval_sets/enterprise_overlay_regression.json`：是否覆盖 clean overlay 的每条 FAQ。

它不调用模型、不访问 Milvus，只负责判断“企业增强资料是否具备进入版本重建和上线演示
的资格”。真正上线仍然要走 `rebuild_kb_version.py --quality-gate --activate`。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.enterprise_overlay.build_enterprise_overlay_dataset import FROZEN_SCENARIOS
from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, read_json_file, write_optional_json


DEFAULT_PACK_ROOT = PROJECT_ROOT / "data_packs" / "enterprise_realistic_pack"
DEFAULT_OVERLAY_REPORT = PROJECT_ROOT / "reports" / "verification" / "enterprise_overlay_build_latest.json"
DEFAULT_DIRTY_REPORT = PROJECT_ROOT / "reports" / "verification" / "dirty_enterprise_samples_latest.json"
DEFAULT_REALISM_REPORT = PROJECT_ROOT / "reports" / "verification" / "enterprise_data_realism_latest.json"
DEFAULT_EVAL_SET = PROJECT_ROOT / "eval_sets" / "enterprise_overlay_regression.json"


def _resolve_path(value: str | Path) -> Path:
    """把命令行路径解析成项目内绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json_or_failure(path: Path, failures: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """读取 JSON；缺失或解析失败时写入门禁失败项。"""
    if not path.exists():
        failures.append({"metric": metric, "message": f"报告不存在：{path}"})
        return {}
    try:
        return read_json_file(path)
    except (json.JSONDecodeError, OSError) as exc:
        failures.append({"metric": metric, "message": f"报告无法读取：{path}，原因：{exc}"})
        return {}


def read_overlay_faq_questions(pack_root: Path) -> list[dict[str, str]]:
    """读取 clean overlay 中所有 FAQ 问题。

    回归评测集必须覆盖这些问题。这样后续新增 overlay FAQ 时，门禁会提醒同步补评测样本，
    避免资料增强只完成了入库预检，却没有可回归的问题。
    """
    questions: list[dict[str, str]] = []
    for scenario_id in FROZEN_SCENARIOS:
        faq_path = pack_root / "clean_overlay" / scenario_id / "faq_overlay.csv"
        if not faq_path.exists():
            continue
        with faq_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                question = str(row.get("question") or "").strip()
                if not question:
                    continue
                questions.append(
                    {
                        "scenario_id": scenario_id,
                        "source": str(row.get("source") or "").strip(),
                        "question": question,
                    }
                )
    return questions


def read_eval_cases(path: Path, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """读取 overlay 回归评测集。"""
    if not path.exists():
        failures.append({"metric": "overlay_eval_set", "message": f"overlay 回归评测集不存在：{path}"})
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        failures.append({"metric": "overlay_eval_set", "message": f"overlay 回归评测集无法读取：{exc}"})
        return []
    if not isinstance(payload, list):
        failures.append({"metric": "overlay_eval_set", "message": "overlay 回归评测集必须是 JSON 数组。"})
        return []
    return [item for item in payload if isinstance(item, dict)]


def validate_overlay_report(report: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    """检查 clean overlay 预检报告。"""
    if not report:
        return
    if report.get("ok") is not True:
        failures.append({"metric": "enterprise_overlay_build", "message": "clean overlay 预检未通过。"})
    if int(report.get("scenario_count") or 0) != len(FROZEN_SCENARIOS):
        failures.append({"metric": "enterprise_overlay_scenario_count", "message": "clean overlay 未覆盖全部 8 个冻结场景。"})
    if int(report.get("failed_scenario_count") or 0) != 0:
        failures.append({"metric": "enterprise_overlay_failed_scenarios", "message": "存在未通过入库质量门禁的 overlay 场景。"})


def validate_dirty_report(report: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    """检查 dirty samples 治理报告。"""
    if not report:
        return
    if int(report.get("sample_count") or 0) <= 0:
        failures.append({"metric": "dirty_sample_count", "message": "dirty samples 治理样本为空，无法演示脏数据治理。"})
    if int(report.get("active_ingestion_allowed_count") or 0) != 0:
        failures.append({"metric": "dirty_active_ingestion", "message": "dirty samples 中存在允许 active 入库的样本。"})
    allowed_items = [item for item in report.get("items") or [] if item.get("allow_active_ingestion")]
    if allowed_items:
        failures.append({"metric": "dirty_items_allowed", "message": "dirty samples 明细中存在 allow_active_ingestion=true。"})


def validate_realism_report(report: dict[str, Any], min_score_delta: float, failures: list[dict[str, Any]]) -> None:
    """检查资料真实度报告。"""
    if not report:
        return
    if int(report.get("scenario_count") or 0) != len(FROZEN_SCENARIOS):
        failures.append({"metric": "realism_scenario_count", "message": "资料真实度报告未覆盖全部 8 个冻结场景。"})
    score_delta = float(report.get("score_delta") or 0.0)
    if score_delta < min_score_delta:
        failures.append(
            {
                "metric": "realism_score_delta",
                "actual": score_delta,
                "threshold": min_score_delta,
                "message": "企业仿真包没有带来足够的资料真实度提升。",
            }
        )


def validate_eval_set(
    overlay_questions: list[dict[str, str]],
    eval_cases: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查 overlay 回归评测集是否覆盖增强 FAQ。"""
    covered = {
        (str(item.get("scenario_id") or ""), str(item.get("question") or item.get("query") or "").strip())
        for item in eval_cases
    }
    required = {(item["scenario_id"], item["question"]) for item in overlay_questions}
    missing = sorted(required - covered)
    field_failures: list[str] = []
    for item in eval_cases:
        case_id = str(item.get("case_id") or item.get("question") or "unknown")
        for field_name in ("scenario_id", "source_filter", "expected_hit_type", "expected_effective_source", "expected_source_contains", "expected_keywords"):
            if not item.get(field_name):
                field_failures.append(f"{case_id}: 缺少 {field_name}")
    if missing:
        failures.append({"metric": "overlay_eval_coverage", "message": f"overlay 回归集缺少 {len(missing)} 条增强 FAQ。", "missing": missing[:20]})
    if field_failures:
        failures.append({"metric": "overlay_eval_fields", "message": "overlay 回归集存在字段不完整样本。", "items": field_failures[:20]})
    return {
        "overlay_question_count": len(overlay_questions),
        "eval_case_count": len(eval_cases),
        "covered_question_count": len(required & covered),
        "missing_question_count": len(missing),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """构建企业 overlay 就绪门禁报告。"""
    failures: list[dict[str, Any]] = []
    pack_root = _resolve_path(args.pack_root)
    overlay_report_path = _resolve_path(args.overlay_report)
    dirty_report_path = _resolve_path(args.dirty_report)
    realism_report_path = _resolve_path(args.realism_report)
    eval_set_path = _resolve_path(args.eval_set)

    overlay_report = _read_json_or_failure(overlay_report_path, failures, "enterprise_overlay_report")
    dirty_report = _read_json_or_failure(dirty_report_path, failures, "dirty_samples_report")
    realism_report = _read_json_or_failure(realism_report_path, failures, "enterprise_realism_report")
    eval_cases = read_eval_cases(eval_set_path, failures)
    overlay_questions = read_overlay_faq_questions(pack_root)

    validate_overlay_report(overlay_report, failures)
    validate_dirty_report(dirty_report, failures)
    validate_realism_report(realism_report, args.min_score_delta, failures)
    eval_summary = validate_eval_set(overlay_questions, eval_cases, failures)

    return {
        "report_type": "enterprise_overlay_readiness_gate",
        "ok": not failures,
        "pack_root": str(pack_root),
        "reports": {
            "overlay": str(overlay_report_path),
            "dirty_samples": str(dirty_report_path),
            "enterprise_realism": str(realism_report_path),
            "overlay_eval_set": str(eval_set_path),
        },
        "scenario_count": len(FROZEN_SCENARIOS),
        "eval_summary": eval_summary,
        "thresholds": {"min_score_delta": args.min_score_delta},
        "failures": failures,
        "recommendation": "通过后可生成 overlay 上线命令；上线后再运行 enterprise_overlay_regression 真实链路评测。",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Check enterprise overlay governance readiness.")
    parser.add_argument("--pack-root", default=str(DEFAULT_PACK_ROOT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--overlay-report", default=str(DEFAULT_OVERLAY_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--dirty-report", default=str(DEFAULT_DIRTY_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--realism-report", default=str(DEFAULT_REALISM_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET.relative_to(PROJECT_ROOT)))
    parser.add_argument("--min-score-delta", type=float, default=1.0)
    parser.add_argument("--output", default="", help="报告输出路径。")
    return parser


def main() -> None:
    """执行企业 overlay 就绪门禁。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = build_report(args)
    write_optional_json(args.output, report)
    print_json(report)
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

