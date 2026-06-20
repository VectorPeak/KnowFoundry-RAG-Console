"""文档切分策略。把标准化后的 Document 切成适合 Milvus 检索的父子块。"""

from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_table_metadata
from qa_core.utils import stable_hash

CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ";", ".", "!", "?",
    "，", ",",
    " ",
    "",
    # 原因： 中英文混排文档需要同时支持中文句号/感叹号/问号和英文句点/分号作为切分边界，递归切分器按 separator 顺序优先匹配大粒度分隔符
]

def split_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """将文档切成可检索的子块并保留父块上下文。子块用于精确召回，parent_content 保存在 metadata 中。
    Returns (chunks_list, ids_list)."""
    # 原因： parent-child 分别切分使子块保持精确命中而父块提供完整上下文窗口，比单一切片在精确召回率和上下文完整性之间取得更好平衡
    settings = get_settings()
    markdown_headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_overlap,
        separators=CHINESE_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_overlap,
        separators=CHINESE_SEPARATORS,
    )

    chunks: list[Document] = []
    ids: list[str] = []
    for doc in documents:
        file_type = str(doc.metadata.get("file_type", "")).lower()
        parent_docs: list[Document]
        if is_table_metadata(doc.metadata):
            # 表格 loader 已经把一行表格转换成"表头 + 行号 + 单元格键值"的完整语义单元。
            # 这里不再按字符递归切分，否则一行中的列和值可能被拆开，检索到金额却丢失
            # 对应状态或审批人。
            parent_content = str(doc.page_content or "").strip()
            if not parent_content:
                continue
            parent_id = stable_hash(
                doc.metadata.get("scenario_id"),
                doc.metadata.get("kb_version"),
                doc.metadata.get("embedding_model_version"),
                doc.metadata.get("chunk_schema_version"),
                doc.metadata.get("doc_id"),
                doc.metadata.get("table_id"),
                doc.metadata.get("sheet_name"),
                doc.metadata.get("row_number"),
                parent_content,
            )
            chunk_id = stable_hash(parent_id, parent_content)
            metadata = dict(doc.metadata or {})
            metadata.update(
                {
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                    "chunk_id": chunk_id,
                }
            )
            chunks.append(Document(page_content=parent_content, metadata=metadata))
            ids.append(chunk_id)
            continue
        elif file_type == ".md":
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
            # Markdown 标题会先转成结构化元数据，再进入递归切分，能提升来源标签和上下文质量。
            # 如果 Markdown 解析失败，说明资料格式需要修复；入库阶段应该暴露异常并进入
            # 异常文件报告，而不是悄悄按普通文本切分，造成章节 metadata 丢失。
            header_docs = header_splitter.split_text(doc.page_content)
            for header_doc in header_docs:
                # Markdown 标题切分器会生成新的 Document，这里把原始文件 metadata
                # 补回去，避免切分后丢失 source、file_name、doc_id 等关键字段。
                header_doc.metadata.update(doc.metadata)
            parent_docs = parent_splitter.split_documents(header_docs)
        else:
            parent_docs = parent_splitter.split_documents([doc])

        for parent_doc in parent_docs:
            parent_content = parent_doc.page_content
            # parent_id 和 chunk_id 都纳入 kb_version、embedding_model_version 和 chunk_schema_version。
            # 这样同一个文件在两个知识库版本里可以同时存在，不会因为内容相同而主键冲突。
            parent_id = stable_hash(
                parent_doc.metadata.get("scenario_id"),
                parent_doc.metadata.get("kb_version"),
                parent_doc.metadata.get("embedding_model_version"),
                parent_doc.metadata.get("chunk_schema_version"),
                parent_doc.metadata.get("doc_id"),
                parent_content,
            )
            child_docs = child_splitter.split_documents([parent_doc])
            for child_doc in child_docs:
                # chunk_id 由父块和子块内容共同决定。同一文件未变化时 id 稳定；文件变化时
                # id 会变化，配合 manifest 删除旧 chunk 后重建。
                chunk_id = stable_hash(parent_id, child_doc.page_content)
                metadata = dict(child_doc.metadata or {})
                metadata.update(
                    {
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "chunk_id": chunk_id,
                    }
                )
                chunks.append(Document(page_content=child_doc.page_content, metadata=metadata))
                ids.append(chunk_id)
    return chunks, ids


