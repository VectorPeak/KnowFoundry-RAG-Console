"""评估场景资料距离真实企业数据现场的差距。

这个脚本不访问 Milvus，也不修改 active 知识库版本。它只扫描场景目录和可选的
企业仿真数据包，输出一个可解释的资料真实度报告。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import configure_utf8_stdio, print_json, write_optional_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_DOC_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx", ".xlsx", ".csv"}
ENTERPRISE_MARKERS = (
    "版本",
    "适用范围",
    "例外",
    "审批",
    "阈值",
    "区域",
    "角色",
    "法人",
    "复核",
    "留痕",
    "过期",
    "冲突",
)


def count_faq_rows(path: Path) -> int:
    """统计 FAQ CSV 中的有效问题数量。"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for row in csv.DictReader(file) if str(row.get("question") or "").strip())


def list_documents(root: Path) -> list[Path]:
    """列出资料目录下的候选文档。"""
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_DOC_SUFFIXES
        and path.name.lower() != "readme.md"
    ]


def read_text_preview(path: Path) -> str:
    """读取文本类资料内容，用于识别版本、例外、审批等企业化标记。"""
    if path.suffix.lower() not in {".md", ".txt", ".csv"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def summarize_package(root: Path) -> dict[str, Any]:
    """统计一个场景或一个仿真数据包目录的资料情况。"""
    faq_count = count_faq_rows(root / "faq.csv") + count_faq_rows(root / "faq_overlay.csv")
    docs = list_documents(root / "data")
    source_names = {path.parent.name.removesuffix("_data") for path in docs}
    suffix_counts = Counter(path.suffix.lower() for path in docs)
    marker_counts: Counter[str] = Counter()
    for path in docs:
        text = read_text_preview(path)
        for marker in ENTERPRISE_MARKERS:
            if marker in text:
                marker_counts[marker] += 1
    return {
        "root": str(root),
        "faq_count": faq_count,
        "doc_file_count": len(docs),
        "source_count": len(source_names),
        "source_names": sorted(source_names),
        "format_counts": dict(sorted(suffix_counts.items())),
        "enterprise_marker_counts": dict(sorted(marker_counts.items())),
    }


def score_realism(summary: dict[str, Any], *, dirty_sample_count: int = 0) -> dict[str, Any]:
    """根据资料规模、覆盖、格式和企业化标记给出 0-100 的近似分。"""
    faq_count = int(summary.get("faq_count") or 0)
    doc_count = int(summary.get("doc_file_count") or 0)
    source_count = int(summary.get("source_count") or 0)
    format_count = len(summary.get("format_counts") or {})
    marker_total = sum(int(value) for value in (summary.get("enterprise_marker_counts") or {}).values())
    score = 0.0
    score += min(25.0, faq_count / 50 * 25)
    score += min(25.0, doc_count / 80 * 25)
    score += min(15.0, source_count / 5 * 15)
    score += min(15.0, format_count / 4 * 15)
    score += min(10.0, marker_total / 20 * 10)
    score += min(10.0, dirty_sample_count / 8 * 10)
    if score >= 75:
        band = "接近企业仿真数据集"
    elif score >= 55:
        band = "企业教学增强版"
    elif score >= 35:
        band = "高质量教学样本"
    else:
        band = "基础演示样本"
    return {"score": round(score, 2), "band": band}


def merge_summaries(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """合并主场景和仿真 overlay 的统计结果。"""
    format_counts = Counter(base.get("format_counts") or {})
    format_counts.update(overlay.get("format_counts") or {})
    marker_counts = Counter(base.get("enterprise_marker_counts") or {})
    marker_counts.update(overlay.get("enterprise_marker_counts") or {})
    return {
        "root": f"{base.get('root')} + {overlay.get('root')}",
        "faq_count": int(base.get("faq_count") or 0) + int(overlay.get("faq_count") or 0),
        "doc_file_count": int(base.get("doc_file_count") or 0) + int(overlay.get("doc_file_count") or 0),
        "source_count": len(set(base.get("source_names") or []) | set(overlay.get("source_names") or [])),
        "source_names": sorted(set(base.get("source_names") or []) | set(overlay.get("source_names") or [])),
        "format_counts": dict(sorted(format_counts.items())),
        "enterprise_marker_counts": dict(sorted(marker_counts.items())),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """构建资料真实度报告。"""
    scenario_root = PROJECT_ROOT / args.scenarios_dir
    pack_root = PROJECT_ROOT / args.pack_dir
    clean_overlay_root = pack_root / "clean_overlay"
    dirty_sample_count = len(list_documents(pack_root / "dirty_samples"))
    rows: list[dict[str, Any]] = []
    for scenario_dir in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
        base = summarize_package(scenario_dir)
        overlay_dir = clean_overlay_root / scenario_dir.name
        overlay = summarize_package(overlay_dir) if overlay_dir.exists() else {
            "faq_count": 0,
            "doc_file_count": 0,
            "source_count": 0,
            "source_names": [],
            "format_counts": {},
            "enterprise_marker_counts": {},
        }
        merged = merge_summaries(base, overlay)
        rows.append(
            {
                "scenario_id": scenario_dir.name,
                "current": {**base, **score_realism(base)},
                "with_enterprise_pack": {
                    **merged,
                    **score_realism(merged, dirty_sample_count=dirty_sample_count),
                },
                "overlay": overlay,
            }
        )
    current_avg = round(sum(row["current"]["score"] for row in rows) / max(len(rows), 1), 2)
    pack_avg = round(sum(row["with_enterprise_pack"]["score"] for row in rows) / max(len(rows), 1), 2)
    return {
        "report_type": "enterprise_data_realism",
        "scenario_count": len(rows),
        "dirty_sample_count": dirty_sample_count,
        "current_avg_score": current_avg,
        "with_pack_avg_score": pack_avg,
        "score_delta": round(pack_avg - current_avg, 2),
        "rows": rows,
        "recommendation": "当前主链路已企业化，数据侧建议继续按 clean_overlay 扩充可入库资料，并把 dirty_samples 只用于治理演示。",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="Analyze how close scenario data is to enterprise data.")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--pack-dir", default="data_packs/enterprise_realistic_pack")
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    """执行资料真实度分析。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = build_report(args)
    write_optional_json(args.output, report)
    print_json(report)


if __name__ == "__main__":
    main()

