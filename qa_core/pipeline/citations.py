"""最终答案引用来源强约束：模型漏写来源编号时在末尾补充"参考来源"，确保证据链完整。
"""

from __future__ import annotations
import re
from typing import Any

from langchain_core.documents import Document

from qa_core.document_metadata import format_source_label, is_table_document

CITATION_RE = re.compile(r"\[\d+\]")
TABLE_CELL_RE = re.compile(r"^-\s*(?P<key>[^:：]{1,40})[:：]\s*(?P<value>.+?)\s*$")

def source_reference_label(doc: Document, index: int) -> str:
    """生成简短来源标签（文件名/FAQ 标准问题；表格资料附加 sheet 和行号）。
    """
    metadata: dict[str, Any] = dict(doc.metadata or {})
    # 从元数据中提取可读来源标签（文件名或表格 sheet+行号）
    return f"[{index}] {format_source_label(metadata)}"


def extract_table_cells(doc: Document) -> list[tuple[str, str]]:
    """从表格行文本中提取"列名-单元格值"键值对。
    """
    cells: list[tuple[str, str]] = []
    for line in str(doc.page_content or "").splitlines():
        match = TABLE_CELL_RE.match(line.strip())
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key and value:
            cells.append((key, value))
    return cells


def build_table_row_detail(doc: Document, index: int) -> str:
    """构造一条可直接追加到答案末尾的表格行要点。

    例: "表格行要点：状态：进行中；金额：5000 [1]"
    """
    # 从表格行文本中提取"列名-单元格值"键值对
    cells = extract_table_cells(doc)
    if not cells:
        return ""
    detail = "；".join(f"{key}：{value}" for key, value in cells[:6])
    return f"表格行要点：{detail} [{index}]"


def needs_table_row_detail(answer: str, doc: Document) -> bool:
    """判断模型自由文本是否遗漏了表格行中的键值对，以决定是否需要后处理补全。
    """
    # 从表格行文本中提取"列名-单元格值"键值对
    cells = extract_table_cells(doc)
    if not cells:
        return False
    return any(value not in answer for _, value in cells)


def has_source_citation(answer: str) -> bool:
    """判断答案中是否已经包含 `[数字]` 形式的来源编号。"""
    return bool(CITATION_RE.search(answer or ""))


def enforce_table_row_details(answer: str, context_docs: list[Document]) -> str:
    """LLM 文本生成易丢弃半结构化表格单元格（状态/金额），后处理确定性补全保证信息不遗漏。
    """
    details: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        # 只补全表格文档中模型未覆盖的行，非表格文档或已含单元格值的行无需处理
        if not is_table_document(doc) or not needs_table_row_detail(answer, doc):
            continue
        # 为漏掉的表格行构造要点详情（含来源编号）
        detail = build_table_row_detail(doc, index)
        if detail:
            details.append(detail)
        if len(details) >= 1:
            break
    if not details:
        return answer
    return f"{answer}\n\n" + "\n".join(details)


def enforce_answer_citations(answer: str, context_docs: list[Document]) -> str:
    """后处理保证每条答案都有可追溯的来源编号，不依赖模型在生成时主动遵守引用格式。
    """
    clean_answer = (answer or "").strip()
    # 无答案或无上下文文档时无需补充来源
    if not clean_answer or not context_docs:
        return clean_answer
    # 确保表格类答案不丢失核心单元格信息（状态/金额/责任人等）
    clean_answer = enforce_table_row_details(clean_answer, context_docs)
    # 模型已在答案中嵌入来源编号时保留原样，不做二次追加破坏原文结构
    if has_source_citation(clean_answer):
        return clean_answer
    # 只追加前 3 个文档的来源标签，来源过多反而不利于用户快速定位关键证据
    references = "；".join(source_reference_label(doc, index) for index, doc in enumerate(context_docs[:3], start=1))
    return f"{clean_answer}\n\n参考来源：{references}"
