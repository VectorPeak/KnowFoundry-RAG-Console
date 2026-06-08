"""本地 JSON 文件型 Store 基类，封装加载、保存、reload 等通用操作。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qa_core.common import read_json_dict, write_json


class JsonFileStore:
    """对象型 JSON 文件 Store 的最小公共能力，子类需实现 empty_data() 和 normalize_data()。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def empty_data(self) -> dict[str, Any]:
        """返回文件不存在或损坏时的空结构。"""
        return {}

    def normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """补齐历史文件缺失字段，默认不做额外处理。"""
        return data

    def _load(self) -> dict[str, Any]:
        """读取 JSON 文件并补齐结构。"""
        if not self.path.exists():
            return self.empty_data()
        # 从 JSON 文件读取数据并补齐结构
        return self.normalize_data(read_json_dict(self.path, self.empty_data()))

    def reload(self) -> None:
        """重新读取磁盘文件，避免多个脚本步骤之间覆盖彼此写入。"""
        self.data = self._load()

    def save(self) -> None:
        """按项目统一 JSON 格式写回磁盘。"""
        # 按统一 JSON 格式写回磁盘文件
        write_json(self.path, self.data)
