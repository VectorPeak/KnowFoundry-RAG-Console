"""分析企业仿真脏样本的治理风险。

`dirty_samples/` 的价值不是“拿来入库”，而是让学生能演示真实企业资料治理：
过期制度、OCR 噪声、表格导出、口径冲突都应该先被识别和分流，再决定是否清洗成
clean overlay。这个脚本只做风险分类和治理建议，不写 Milvus，不修改 active 知识库。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, write_optional_json


DEFAULT_DIRTY_ROOT = PROJECT_ROOT / "data_packs" / "enterprise_realistic_pack" / "dirty_samples"
TEXT_SUFFIXES = {".md", ".txt", ".csv"}
ISSUE_RECOMMENDATIONS = {
    "expired_policy": "标记为过期版本，不允许进入 active 知识库；需要找到当前有效版本后再生成 clean overlay。",
    "policy_conflict": "进入 FAQ/正文冲突复核，确认标准口径后再更新 FAQ 或正文。",
    "ocr_review_required": "先进入 OCR 人工校验或版面还原流程，修正错字、断行和金额字段后再切分。",
    "table_split_required": "按表头、状态列和业务主键做表格专用切分，不要按普通段落直接入库。",
    "active_ingestion_blocked": "作为治理样本保留，默认阻断 active 入库。",
    "manual_review": "需要人工判断是否可清洗为 clean overlay。",
}


def list_dirty_files(root: Path) -> list[Path]:
    """列出脏样本文件，排除说明文档。"""
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name.lower() != "readme.md"
    ]


def read_sample_text(path: Path) -> str:
    """读取可分析的文本内容。"""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def classify_dirty_sample(path: Path, text: str) -> dict[str, Any]:
    """识别单个脏样本的资料治理风险。

    这里采用规则分类而不是调用大模型，是因为脏样本检查属于入库前门禁，应稳定、便宜、
    可解释。规则命中后会给出治理建议，后续真正清洗时再由人工或 Agent 工作流处理。
    """
    content = f"{path.name}\n{text}"
    issues: list[str] = []
    if re.search(r"过期|V20\d{2}|不得入库|旧版|已废止", content, re.IGNORECASE):
        issues.append("expired_policy")
        issues.append("active_ingestion_blocked")
    if re.search(r"冲突|不一致|旧口径|当前口径", content):
        issues.append("policy_conflict")
    if re.search(r"OCR|扫描|错字|识别|O 和 0|断行|噪声", content, re.IGNORECASE):
        issues.append("ocr_review_required")
    if path.suffix.lower() == ".csv" or re.search(r"表格|导出|,.*,", content):
        issues.append("table_split_required")
    if not issues:
        issues.append("manual_review")
    unique_issues = list(dict.fromkeys(issues))
    return {
        "path": str(path),
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "allow_active_ingestion": False,
        "issues": unique_issues,
        "recommendations": [ISSUE_RECOMMENDATIONS[issue] for issue in unique_issues],
        "preview": text[:240].replace("\n", " "),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """构建脏样本治理分析报告。"""
    root = (PROJECT_ROOT / args.dirty_root).resolve() if not Path(args.dirty_root).is_absolute() else Path(args.dirty_root)
    items = [classify_dirty_sample(path, read_sample_text(path)) for path in list_dirty_files(root)]
    issue_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    for item in items:
        issue_counts.update(item["issues"])
        suffix_counts.update([item["suffix"]])
    return {
        "report_type": "dirty_enterprise_samples",
        "ok": True,
        "dirty_root": str(root),
        "sample_count": len(items),
        "active_ingestion_allowed_count": sum(1 for item in items if item["allow_active_ingestion"]),
        "issue_counts": dict(sorted(issue_counts.items())),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "items": items,
        "recommendation": "dirty_samples 只用于治理演示；要进入主链路，必须先清洗成 clean_overlay 并通过入库质量门禁。",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Analyze dirty enterprise samples for governance risks.")
    parser.add_argument("--dirty-root", default=str(DEFAULT_DIRTY_ROOT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default="", help="报告输出路径。")
    return parser


def main() -> None:
    """执行脏样本治理分析。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = build_report(args)
    write_optional_json(args.output, report)
    print_json(report)


if __name__ == "__main__":
    main()

