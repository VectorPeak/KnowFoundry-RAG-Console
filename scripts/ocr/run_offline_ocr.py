"""离线 OCR 清洗入口。

该脚本用于处理扫描件、图片 PDF 或图片附件，输出待人工复核的 Markdown 和 JSON 报告。
它不会写 Milvus，不会激活知识库版本，也不会被在线问答调用。

示例：
python scripts/ocr/run_offline_ocr.py --input data_packs/dirty_samples/scan.pdf --output-dir reports/ocr
python scripts/ocr/run_offline_ocr.py --input-dir incoming_scans --output-dir reports/ocr/batch_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa_core.indexing.ocr_documents import create_ocr_engine, is_ocr_supported_file, recognize_file
from scripts.common import configure_utf8_stdio, utc_now, write_json_file


def collect_inputs(input_path: str | None, input_dir: str | None) -> list[Path]:
    """收集本次需要 OCR 的文件。"""
    files: list[Path] = []
    if input_path:
        files.append(Path(input_path))
    if input_dir:
        root = Path(input_dir)
        files.extend(path for path in sorted(root.rglob("*")) if path.is_file() and is_ocr_supported_file(path))
    unique: list[Path] = []
    for path in files:
        if path not in unique:
            unique.append(path)
    return unique


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Run offline OCR and create reviewable markdown files.")
    parser.add_argument("--input", default="", help="单个扫描件、图片 PDF 或图片文件。")
    parser.add_argument("--input-dir", default="", help="批量 OCR 输入目录。")
    parser.add_argument("--output-dir", default=str(Path("reports") / "ocr"), help="OCR Markdown 和报告输出目录。")
    parser.add_argument("--report-output", default="", help="JSON 报告路径，默认写到 output-dir/ocr_report.json。")
    parser.add_argument("--lang", default="ch", help="PaddleOCR 语言配置，中文资料默认 ch。")
    parser.add_argument("--min-confidence", type=float, default=0.78, help="低于该平均置信度时标记为需重新扫描或人工整理。")
    parser.add_argument("--dpi", type=int, default=180, help="PDF 渲染 DPI。")
    parser.add_argument("--max-pages", type=int, default=0, help="每个 PDF 最多处理页数，0 表示不限制。")
    return parser


def main() -> None:
    """执行离线 OCR 并保存报告。"""
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    inputs = collect_inputs(args.input, args.input_dir)
    if not inputs:
        parser.error("必须通过 --input 或 --input-dir 提供至少一个 OCR 文件。")

    output_dir = Path(args.output_dir)
    engine = create_ocr_engine(lang=args.lang)
    results = []
    failures = []
    for path in inputs:
        try:
            result = recognize_file(
                path,
                output_dir=output_dir,
                engine=engine,
                min_confidence=args.min_confidence,
                dpi=args.dpi,
                max_pages=args.max_pages,
            )
            results.append(result.as_dict())
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})

    report = {
        "report_type": "offline_ocr",
        "created_at": utc_now(),
        "input_count": len(inputs),
        "success_count": len(results),
        "failure_count": len(failures),
        "ready_for_review_count": sum(1 for item in results if item.get("ready_for_review")),
        "output_dir": str(output_dir),
        "results": results,
        "failures": failures,
        "next_step": "人工复核 Markdown 后，再放入对应场景 data/<source>_data 目录并重建知识库版本。",
    }
    report_path = Path(args.report_output) if args.report_output else output_dir / "ocr_report.json"
    write_json_file(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()


