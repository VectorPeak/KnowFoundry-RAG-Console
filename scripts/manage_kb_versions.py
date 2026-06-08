"""知识库版本管理命令行工具。

该脚本只操作 `.index_manifest/kb_versions.json`，不直接访问 Milvus。它用于本地查看、
创建、激活和归档知识库版本。

常用命令：
- 查看版本：`python scripts/manage_kb_versions.py list`
- 创建版本：`python scripts/manage_kb_versions.py create --description "2026-05-06 全量重建"`
- 激活版本：`python scripts/manage_kb_versions.py activate kb_20260506_103000_xxxxxxxx`
- 归档版本：`python scripts/manage_kb_versions.py archive kb_20260430_090000_xxxxxxxx`

为什么独立成脚本：
- 版本切换是运维动作，不应该混在在线问答请求里；
- 入库脚本可以自动创建和激活版本，但手工回滚时需要一个轻量入口；
- 不直接删除 Milvus 数据，避免误删可回滚版本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.governance.kb_versions import get_kb_version_store


def print_json(payload) -> None:
    """以中文可读的 JSON 格式输出脚本结果。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    """解析命令并执行版本管理操作。"""
    parser = argparse.ArgumentParser(description="Manage multi-scenario RAG knowledge base versions.")
    parser.add_argument("--scenario", default=None, help="Business scenario id. Defaults to ACTIVE_SCENARIO_ID.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List knowledge base versions.")

    create_parser = subparsers.add_parser("create", help="Create a staged knowledge base version.")
    create_parser.add_argument("--kb-version", default=None, help="Optional explicit version id.")
    create_parser.add_argument("--description", default="", help="Human readable description.")

    activate_parser = subparsers.add_parser("activate", help="Activate a knowledge base version.")
    activate_parser.add_argument("kb_version")

    archive_parser = subparsers.add_parser("archive", help="Archive a non-active knowledge base version.")
    archive_parser.add_argument("kb_version")

    args = parser.parse_args()
    store = get_kb_version_store(args.scenario)

    if args.command == "list":
        print_json(store.as_payload())
        return
    if args.command == "create":
        version = store.ensure_version(
            args.kb_version,
            create_new=not bool(args.kb_version),
            description=args.description,
            created_by="manage_kb_versions",
        )
        print_json({"status": "success", "version": version.as_dict()})
        return
    if args.command == "activate":
        version = store.activate_version(args.kb_version)
        print_json({"status": "success", "version": version.as_dict()})
        return
    if args.command == "archive":
        version = store.archive_version(args.kb_version)
        print_json({"status": "success", "version": version.as_dict()})
        return


if __name__ == "__main__":
    main()

