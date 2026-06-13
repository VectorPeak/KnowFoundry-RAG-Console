"""Milvus 检索过滤表达式构造。

这里把 source_filter、kb_version 和 DataScope 合并成一个 Milvus boolean expr，
供 similarity_search_with_score 调用。所有字符串都会转义，避免把用户输入直接拼进
Milvus 表达式造成过滤条件注入。
"""

from __future__ import annotations

from qa_core.governance.data_scope import DataScope, escape_expr_value


def validate_source_filter(
    source_filter: str | None,
    valid_sources: list[str] | None,
) -> None:
    """校验 source_filter 白名单。source_filter='hr', valid_sources=['hr','legal'] -> pass; source_filter='ai', ... -> ValueError。

    参数：
        source_filter: 要校验的 source 值。
        valid_sources: 当前场景允许的 source 白名单；None 表示不校验。

    异常：
        ValueError: source_filter 不在 valid_sources 中。
    """
    if source_filter:
        # 应用层白名单校验：拒绝用户传入当前场景不支持的 source_filter，防止跨业务分类检索
        if valid_sources is not None and source_filter not in valid_sources:
            raise ValueError(f"无效的业务分类：{source_filter}")


def build_source_expr(
    source_filter: str | None,
    kb_version: str | None = None,
    valid_sources: list[str] | None = None,
    data_scope: DataScope | None = None,
) -> str | None:
    """合并 source/kb_version/data_scope 为 Milvus 布尔表达式。

    示例：
        ``build_source_expr("hr", "v1")`` returns
        ``'source == "hr" and kb_version == "v1"'``.

    执行流程：
      1. 用 valid_sources 校验 source_filter。
      2. source_filter 存在时添加 source == "<value>"。
      3. kb_version 存在时添加 kb_version == "<value>"。
      4. data_scope 存在时追加租户、数据集、可见级别和角色过滤。
      5. 所有子句用 and 连接；没有任何约束时返回 None。

    参数：
        source_filter: 业务分类过滤项。
        kb_version: 知识库版本。
        valid_sources: source_filter 白名单。
        data_scope: 数据隔离范围。

    返回：
        Milvus boolean expr；没有约束时返回 None。

    异常：
        ValueError: source_filter 不在 valid_sources 中。
    """
    validate_source_filter(source_filter, valid_sources)
    clauses: list[str] = []
    # 逐项拼接过滤子句：业务分类 + KB 版本 + 数据域，每项值做转义防止 Milvus 表达式注入
    if source_filter:
        safe_source = escape_expr_value(str(source_filter))
        clauses.append(f'source == "{safe_source}"')
    if kb_version:
        safe_version = escape_expr_value(str(kb_version))
        clauses.append(f'kb_version == "{safe_version}"')
    if data_scope is not None:
        clauses.extend(data_scope.expr_clauses())
    return " and ".join(clauses) if clauses else None

