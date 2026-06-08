"""MySQL 存储基类：延迟创建 SQLAlchemy 引擎。"""
from sqlalchemy import create_engine, text

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
