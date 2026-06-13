"""把已人工复核的 OCR Markdown 提升为可入库资料。

离线 OCR 的输出不能直接写入 active 知识库。原因是 OCR 可能识别错金额、日期、合同号、
账号或责任边界。这个脚本只做一件事：把明确标记为“已复核”的 Markdown 复制到指定
场景和 source 的资料目录，后续仍必须通过知识库版本重建、质量门禁和评测门禁才能上线。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.scenarios.registry import resolve_scenario
from scripts.common import configure_utf8_stdio, utc_now, write_json_file


DEFAULT_REVIEW_MARKERS = (
    "复核状态：已复核",
    "人工复核：通过",
    "review_status: approved",
    "reviewed: true",
)


def collect_markdown_files(input_dir: Path) -> list[Path]:
    """收集待提升的 OCR Markdown 文件。

    只接受 Markdown，是为了把 OCR 的图片/临时页渲染文件挡在入库资料目录外。真正复杂
    文件仍应先通过 `scripts/ocr/run_offline_ocr.py` 转成可读文本，再人工复核。
    """
    if not input_dir.exists():
        raise ValueError(f"OCR 输入目录不存在：{input_dir}")
    return sorted(path for path in input_dir.rglob("*.md") if path.is_file())


def is_reviewed(content: str, markers: tuple[str, ...]) -> bool:
    """判断 OCR Markdown 是否带有人工复核通过标记。"""
    return any(marker in content for marker in markers)


def target_path_for(source_dir: Path, source_file: Path, force: bool) -> Path:
    """计算目标文件路径，并阻止默认覆盖已有资料。"""
    target = source_dir / source_file.name
    if target.exists() and not force:
        raise FileExistsError(f"目标文件已存在，需确认后使用 --force 覆盖：{target}")
    return target


def promote_files(args: argparse.Namespace) -> dict[str, Any]:
    """执行 OCR 复核文件提升，默认只生成预览报告。"""
    scenario = resolve_scenario(args.scenario)
    if args.source not in scenario.valid_sources:
        raise ValueError(f"source 不属于场景 {scenario.scenario_id}：{args.source}")

    input_dir = Path(args.input_dir)
    source_dir = Path(scenario.data_root) / f"{args.source}_data" / args.target_subdir
    markers = tuple(item.strip() for item in args.review_marker if item.strip()) or DEFAULT_REVIEW_MARKERS
    markdown_files = collect_markdown_files(input_dir)
    rows: list[dict[str, Any]] = []
    promoted_count = 0

    if args.apply:
        source_dir.mkdir(parents=True, exist_ok=True)

    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        reviewed = is_reviewed(content, markers)
        row: dict[str, Any] = {
            "input_path": str(path),
            "target_path": str(source_dir / path.name),
            "reviewed": reviewed,
            "promoted": False,
            "reason": "",
        }
        if not reviewed:
            row["reason"] = "缺少人工复核通过标记，禁止进入场景资料目录。"
            rows.append(row)
            continue
        try:
            target = target_path_for(source_dir, path, args.force)
            if args.apply:
                shutil.copy2(path, target)
                promoted_count += 1
                row["promoted"] = True
                row["reason"] = "已复制到场景资料目录，后续必须重建知识库版本。"
            else:
                row["reason"] = "dry-run 预览通过，添加 --apply 后才会复制。"
        except OSError as exc:
            row["reason"] = str(exc)
        rows.append(row)

    blocked_count = sum(1 for row in rows if not row["reviewed"])
    failed_count = sum(
        1
        for row in rows
        if row["reviewed"]
        and not row["promoted"]
        and not str(row["reason"]).startswith("dry-run")
    )
    report = {
        "report_type": "ocr_candidate_promotion",
        "created_at": utc_now(),
        "scenario_id": scenario.scenario_id,
        "source": args.source,
        "input_dir": str(input_dir),
        "target_dir": str(source_dir),
        "apply": args.apply,
        "total_markdown_files": len(markdown_files),
        "reviewed_count": sum(1 for row in rows if row["reviewed"]),
        "promoted_count": promoted_count,
        "blocked_count": blocked_count,
        "failed_count": failed_count,
        "ok": bool(markdown_files) and blocked_count == 0 and failed_count == 0,
        "next_commands": [
            f"python scripts/rebuild_kb_version.py --scenario {scenario.scenario_id} --new-version --force --quality-gate --activate",
            f"python scripts/check_evaluation_gate.py --dataset eval_sets/business_depth_regression.json --limit 32 --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 0.85",
        ],
        "rows": rows,
    }
    write_json_file(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Promote reviewed OCR markdown files into scenario data directory.")
    parser.add_argument("--input-dir", required=True, help="离线 OCR 产出的 Markdown 目录。")
    parser.add_argument("--scenario", required=True, help="目标场景 ID。")
    parser.add_argument("--source", required=True, help="目标 source，例如 quality、claim_material。")
    parser.add_argument("--target-subdir", default="ocr_reviewed", help="source_data 下的复核资料子目录。")
    parser.add_argument("--review-marker", action="append", default=list(DEFAULT_REVIEW_MARKERS), help="允许进入资料目录的复核标记。")
    parser.add_argument("--apply", action="store_true", help="真正复制文件；默认只输出预览报告。")
    parser.add_argument("--force", action="store_true", help="允许覆盖目标目录中的同名文件。")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "verification" / "ocr_candidate_promotion_latest.json"))
    return parser


def main() -> None:
    """执行 OCR 候选资料提升。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()
    report = promote_files(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.apply and report["blocked_count"]:
        sys.exit(1)


if __name__ == "__main__":
    main()


