"""FAQ CSV 基础质量检测。"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def _resolve_csv_source(row: dict) -> str:
    """从 FAQ CSV 行中解析 source 字段。"""
    return str(
        row.get("source") or row.get("source_filter") or row.get("业务分类")
        or row.get("分类") or row.get("subject_name") or ""
    ).strip()


def read_faq_records(csv_path: str | Path) -> list[dict[str, Any]]:
    """读取 FAQ 记录，兼容中文和英文列名。返回行号、问题、答案和 source。"""
    path = Path(csv_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader, start=2):
            records.append(
                {
                    "row": row_index,
                    "question": str(row.get("问题") or row.get("question") or "").strip(),
                    "answer": str(row.get("答案") or row.get("answer") or "").strip(),
                    "source": _resolve_csv_source(row),
                }
            )
    return records


def analyze_faq_csv(csv_path: str | Path, valid_sources: list[str]) -> dict[str, Any]:
    """检查 FAQ CSV 的基础质量（空答案、重复问题、分类不在白名单等）。"""
    path = Path(csv_path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "record_count": 0,
        "valid_record_count": 0,
        "empty_question_rows": [],
        "empty_answer_rows": [],
        "duplicate_questions": [],
        "invalid_sources": [],
        "source_counts": {},
    }
    if not path.exists():
        result["error"] = "FAQ CSV 不存在。"
        return result

    question_seen: dict[str, int] = {}
    source_counter: Counter[str] = Counter()
    for record in read_faq_records(path):
        question = record["question"]
        answer = record["answer"]
        source = record["source"]
        row_index = record["row"]
        result["record_count"] += 1
        if not question:
            result["empty_question_rows"].append(row_index)
        if not answer:
            result["empty_answer_rows"].append(row_index)
        if question:
            question_seen[question] = question_seen.get(question, 0) + 1
        if source:
            source_counter[source] += 1
            if source not in valid_sources:
                result["invalid_sources"].append({"row": row_index, "source": source})
        if question and answer:
            result["valid_record_count"] += 1
    result["duplicate_questions"] = [question for question, count in question_seen.items() if count > 1]
    result["source_counts"] = dict(source_counter)
    return result
