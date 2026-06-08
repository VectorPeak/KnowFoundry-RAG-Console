"""评测和性能脚本共享的样本解析工具。

评测脚本、性能基线脚本都要从同一份 eval_set 里读取 question、scenario、source、
tenant、kb_version 等字段。这里统一解析，避免两个脚本字段口径不一致。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCaseRuntime:
    """一条评测样本在运行时需要传给 QAService 的公共参数。"""

    case_id: str
    question: str
    scenario_id: str | None
    source_filter: str | None
    tenant_id: str | None
    dataset_id: str | None
    visibility: str | None
    user_role: str | None
    kb_version: str | None
    session_id: str

    @classmethod
    def from_item(cls, item: dict[str, Any], index: int, args: argparse.Namespace, *, session_prefix: str) -> "EvalCaseRuntime":
        """从评测 JSON 样本和命令行默认值中解析运行参数。"""
        question = str(item.get("question") or item.get("query") or "").strip()
        return cls(
            case_id=str(item.get("case_id") or f"case_{index}"),
            question=question,
            scenario_id=item.get("scenario_id") or args.scenario,
            source_filter=item.get("source_filter"),
            tenant_id=item.get("tenant_id") or args.tenant_id,
            dataset_id=item.get("dataset_id") or args.dataset_id,
            visibility=item.get("visibility") or args.visibility,
            user_role=item.get("user_role") or args.user_role,
            kb_version=item.get("kb_version") or args.kb_version,
            session_id=f"{session_prefix}-{int(time.time())}-{index}",
        )

    def service_kwargs(self) -> dict[str, Any]:
        """返回 QAService stream/debug 共用的关键字参数。"""
        return {
            "kb_version": self.kb_version,
            "scenario_id": self.scenario_id,
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "visibility": self.visibility,
            "user_role": self.user_role,
        }


def load_eval_items(dataset: str | Path, limit: int) -> list[dict[str, Any]]:
    """读取评测集并按 limit 截断。"""
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    return list(data[:limit])
