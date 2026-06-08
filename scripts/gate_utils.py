"""门禁脚本公共判断工具。

入库质量门禁和评测门禁的业务指标不同，但失败项结构、最大/最小阈值判断、必填项判断
完全一样。抽到这里后，两个门禁脚本只保留“检查哪些指标”的业务含义。
"""

from __future__ import annotations

from typing import Any


def to_count(value: Any) -> int:
    """把报告里的列表、字典或数字统一转成可比较的数量。"""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def add_max_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    actual: float,
    maximum: float,
    message: str,
) -> None:
    """当实际值高于最高允许值时追加失败项。"""
    if actual > maximum:
        failures.append({"metric": metric, "actual": actual, "threshold": maximum, "message": message})


def add_min_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    actual: float,
    minimum: float,
    message: str,
) -> None:
    """当实际值低于最低要求时追加失败项。"""
    if actual < minimum:
        failures.append({"metric": metric, "actual": actual, "threshold": minimum, "message": message})


def add_required_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    actual: Any,
    enabled: bool,
    message: str,
) -> None:
    """当必填项缺失时追加失败项。"""
    if enabled and not actual:
        failures.append({"metric": metric, "actual": actual, "threshold": "required", "message": message})
