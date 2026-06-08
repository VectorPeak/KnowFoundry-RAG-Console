"""问答服务的统一日志配置。qa_core 主链路使用的唯一日志入口。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from qa_core.config.settings import PROJECT_ROOT, get_settings


def get_logger(name: str = "MultiScenarioRAG") -> logging.Logger:
    """返回同时输出到文件和控制台的已配置日志器。
    同一个 logger name 只添加一次 handler，避免热重载或测试重复导入时重复输出。
    """
    # 获取运行时配置中的日志级别
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level.upper())
    if logger.handlers:
        # 避免重复绑定文件处理器
        return logger

    # 日志写入 PROJECT_ROOT/logs/
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件用于复盘，控制台用于 Docker logs 和本地开发
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

