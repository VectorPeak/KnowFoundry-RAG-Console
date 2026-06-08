"""qa_core 内部复用的轻量公共工具：UTC 时间、JSON 文件读写、文件更新时间等。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """返回 UTC ISO 时间字符串，用于版本管理、报告和 trace 的时间统一排序。"""
    return datetime.now(timezone.utc).isoformat()


def utc_file_stamp() -> str:
    """返回适合放进文件名和版本号的 UTC 时间戳（YYYYMMDD_HHMMSS）。"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def path_updated_at(path: str | Path) -> str:
    """返回文件修改时间对应的 UTC ISO 字符串。"""
    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


def read_json(path: str | Path, default: Any = None) -> Any:
    """读取 JSON 文件，读取失败时返回 default。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_json_dict(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取对象型 JSON，非对象或读取失败时返回字典默认值。"""
    # 调用 read_json 获取原始数据
    payload = read_json(path, default=None)
    if isinstance(payload, dict):
        return payload
    return dict(default or {})


def write_json(path: str | Path, payload: Any) -> str:
    """按项目统一格式（ensure_ascii=False, indent=2）写入 JSON 文件，返回文件路径字符串。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def list_json_reports(root: Path, glob_pattern: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """列出目录下最近的 JSON 报告，每个条目包含 path/file_name/updated_at/payload。"""
    if not root.exists():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(root.glob(glob_pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json_dict(path)
        reports.append(
            {
                "path": str(path),
                "file_name": path.name,
                "updated_at": path_updated_at(path),
                "payload": payload,
            }
        )
        if limit and len(reports) >= limit:
            break
    return reports
