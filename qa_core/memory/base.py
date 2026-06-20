"""MySQL 存储基类：延迟创建 SQLAlchemy 引擎。"""
from __future__ import annotations

import re

from sqlalchemy import create_engine, text


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_sql_identifier(value: str, *, label: str = "SQL identifier") -> str:
    """校验可拼接到 SQL 中的表名/索引名，避免配置项注入 SQL。"""
    if not _SQL_IDENTIFIER_RE.fullmatch(value or ""):
        raise ValueError(f"{label} 不合法：{value!r}")
    return value

class _MySqlStore:
    """MySQL 存储的轻量基类。

    子类必须先设置 `self.settings`，并且 settings 中要包含 `mysql_sync_uri`。
    这样 `engine` 属性和 `_execute_ddl()` 才能正常创建连接。
    """

    def __init__(self) -> None:
        self._engine = None

    @property
    def engine(self):
        """延迟创建带连接健康检查的 SQLAlchemy 同步引擎。"""
        if self._engine is None:
            # 延迟创建 SQLAlchemy 引擎（带连接健康检查）
            self._engine = create_engine(self.settings.mysql_sync_uri, pool_pre_ping=True)
        return self._engine

    def _execute_ddl(self, sql: str) -> None:
        """在隐式事务中执行 DDL 语句，例如 CREATE TABLE。"""
        with self.engine.begin() as conn:
            # 在隐式事务中执行 DDL
            conn.execute(text(sql))
