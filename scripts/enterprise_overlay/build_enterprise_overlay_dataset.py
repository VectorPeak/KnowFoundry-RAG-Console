"""构建企业仿真 clean overlay 的预览数据集。

`data_packs/enterprise_realistic_pack/clean_overlay` 里的资料不是直接覆盖 active
知识库，而是先和 `scenarios/` 的基础资料合并成一个临时预览目录，再复用入库质量报告
和质量门禁验证。这样可以讲清楚企业资料增强的完整链路：

1. 基础教学场景保持稳定；
2. 企业增强资料先做离线合并；
3. 合并结果通过入库质量报告检查；
4. 后续确实要上线时，再用知识库版本流程重建、评测、激活。

脚本只写 `reports/enterprise_overlay_build/` 下的生成物，不写 Milvus，也不修改
`scenarios/` 目录。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa_core.quality.ingestion import build_ingestion_quality_report, save_ingestion_quality_report
from qa_core.scenarios.registry import resolve_scenario
from scripts.check_ingestion_quality_gate import IngestionQualityThresholds, evaluate_report_against_gate
from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, write_optional_json


FAQ_FIELDS = ("source", "question", "answer")
DEFAULT_PACK_ROOT = PROJECT_ROOT / "data_packs" / "enterprise_realistic_pack"
DEFAULT_BUILD_ROOT = PROJECT_ROOT / "reports" / "enterprise_overlay_build"
FROZEN_SCENARIOS = (
    "compliance_qa",
    "cross_border_risk",
    "engineering_project_qa",
    "enterprise_knowledge",
    "equipment_ops",
    "insurance_claims",
    "saas_support",
    "tender_contract_risk",
)


def read_faq_rows(path: Path) -> list[dict[str, str]]:
    """读取 FAQ CSV，并只保留主链路需要的 `source/question/answer` 字段。

    clean overlay 使用 `faq_overlay.csv`，基础场景使用 `faq.csv`。两种文件在语义上
    完全一致，区别只是 overlay 还没有进入 active 场景包。这里统一成同一种结构，
    后续质量报告可以像检查普通 FAQ 一样检查增强后的 FAQ。
    """
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            normalized = {field: str(row.get(field) or "").strip() for field in FAQ_FIELDS}
            if any(normalized.values()):
                rows.append(normalized)
    return rows


def merge_faq_rows(base_rows: list[dict[str, str]], overlay_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    """合并基础 FAQ 和 overlay FAQ，并按问题去重。

    去重规则采用“基础场景优先”。原因是 `scenarios/` 代表当前已验收的 active 口径，
    overlay 只是增强候选资料；如果 overlay 中不小心出现同名问题，应该在报告里暴露为
    duplicate，而不是静默覆盖当前标准答案。
    """
    merged: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    duplicate_questions: list[str] = []
    for row in [*base_rows, *overlay_rows]:
        question = row.get("question", "").strip()
        if not question:
            merged.append(row)
            continue
        if question in seen_questions:
            duplicate_questions.append(question)
            continue
        seen_questions.add(question)
        merged.append(row)
    return merged, duplicate_questions


def write_faq_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """写出合并后的 FAQ CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(FAQ_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def ensure_generated_dir(path: Path, build_root: Path) -> None:
    """清理并创建生成目录。

    这里允许删除旧的预览目录，但必须确认目标目录位于 `reports/enterprise_overlay_build`
    之下。这个保护是为了避免路径参数写错时误删项目资料。
    """
    resolved_path = path.resolve()
    resolved_root = build_root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"生成目录不在允许范围内：{resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)


def copy_tree_contents(source: Path, target: Path) -> tuple[int, list[str]]:
    """复制目录内容，并返回复制文件数和同名覆盖列表。

    基础资料先复制，overlay 后复制。同名文件会被覆盖并记录到报告里，方便发现增强包
    是否无意替换了基础文档。当前企业增强包约定使用新增文件名，正常情况下不应出现覆盖。
    """
    copied = 0
    overwritten: list[str] = []
    if not source.exists():
        return copied, overwritten
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative_path = path.relative_to(source)
        output_path = target / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            overwritten.append(str(relative_path))
        shutil.copy2(path, output_path)
        copied += 1
    return copied, overwritten


def build_single_overlay_dataset(scenario_id: str, pack_root: Path, build_root: Path) -> dict[str, Any]:
    """构建单个场景的企业增强预览数据集并运行入库质量门禁。

    输出中的 `gate.ok=True` 表示“增强资料具备进入下一步知识库版本重建的资格”；
    这里仍然不等于已经上线，因为正式上线还需要 `rebuild_kb_version.py --quality-gate`
    和 RAG 回归评测。
    """
    scenario = resolve_scenario(scenario_id)
    overlay_root = pack_root / "clean_overlay" / scenario.scenario_id
    build_dir = build_root / scenario.scenario_id
    data_dir = build_dir / "data"
    faq_path = build_dir / "faq.csv"
    ensure_generated_dir(build_dir, build_root)

    base_doc_count, base_overwrites = copy_tree_contents(Path(scenario.data_root), data_dir)
    overlay_doc_count, overlay_overwrites = copy_tree_contents(overlay_root / "data", data_dir)
    base_rows = read_faq_rows(Path(scenario.faq_csv_path))
    overlay_rows = read_faq_rows(overlay_root / "faq_overlay.csv")
    merged_rows, duplicate_questions = merge_faq_rows(base_rows, overlay_rows)
    write_faq_rows(faq_path, merged_rows)

    kb_version = f"enterprise_overlay_preview_{scenario.scenario_id}"
    quality_report = build_ingestion_quality_report(
        scenario_id=scenario.scenario_id,
        data_dir=str(data_dir),
        faq_csv=str(faq_path),
        kb_version=kb_version,
    )
    quality_report_path = save_ingestion_quality_report(
        quality_report,
        str(build_dir / "ingestion_quality.json"),
    )
    gate = evaluate_report_against_gate(
        quality_report,
        IngestionQualityThresholds(),
        report_path=quality_report_path,
    )
    return {
        "scenario_id": scenario.scenario_id,
        "build_dir": str(build_dir),
        "data_dir": str(data_dir),
        "faq_csv": str(faq_path),
        "base_faq_count": len(base_rows),
        "overlay_faq_count": len(overlay_rows),
        "merged_faq_count": len(merged_rows),
        "base_doc_count": base_doc_count,
        "overlay_doc_count": overlay_doc_count,
        "merged_doc_count": base_doc_count + overlay_doc_count,
        "duplicate_overlay_questions": duplicate_questions,
        "overwritten_files": [*base_overwrites, *overlay_overwrites],
        "quality_report_path": quality_report_path,
        "gate": gate,
    }


def resolve_requested_scenarios(args: argparse.Namespace) -> list[str]:
    """解析命令行要构建的场景列表。"""
    if args.all_scenarios:
        return list(FROZEN_SCENARIOS)
    if args.scenario:
        return [args.scenario]
    return ["enterprise_knowledge"]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """构建一个或多个场景的 overlay 预览报告。"""
    pack_root = (PROJECT_ROOT / args.pack_root).resolve() if not Path(args.pack_root).is_absolute() else Path(args.pack_root)
    build_root = (PROJECT_ROOT / args.build_root).resolve() if not Path(args.build_root).is_absolute() else Path(args.build_root)
    rows = [
        build_single_overlay_dataset(scenario_id, pack_root, build_root)
        for scenario_id in resolve_requested_scenarios(args)
    ]
    failed = [row for row in rows if not row["gate"].get("ok")]
    return {
        "report_type": "enterprise_overlay_build",
        "ok": not failed,
        "pack_root": str(pack_root),
        "build_root": str(build_root),
        "scenario_count": len(rows),
        "failed_scenario_count": len(failed),
        "rows": rows,
        "recommendation": "gate.ok 为 true 的场景可进入知识库版本重建；dirty_samples 仍不得直接并入 active 版本。",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Build and validate enterprise clean overlay preview dataset.")
    parser.add_argument("--scenario", default="", help="只构建一个场景；默认 enterprise_knowledge。")
    parser.add_argument("--all-scenarios", action="store_true", help="构建全部 8 个冻结场景。")
    parser.add_argument("--pack-root", default=str(DEFAULT_PACK_ROOT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD_ROOT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default="", help="汇总报告输出路径。")
    parser.add_argument("--strict", action="store_true", help="有任一场景未通过质量门禁时返回非 0。")
    return parser


def main() -> None:
    """执行企业增强资料预览构建。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = build_report(args)
    write_optional_json(args.output, report)
    print_json(report)
    if args.strict and not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

