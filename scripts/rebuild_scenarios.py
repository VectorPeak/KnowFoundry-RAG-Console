"""批量重建多个业务场景的知识库版本。

默认用于初始化或重建全部 8 个冻结业务场景：

python scripts/rebuild_scenarios.py --reset-collections

也可以指定场景：

python scripts/rebuild_scenarios.py --scenarios enterprise_knowledge,equipment_ops --reset-collections

脚本通过子进程逐个调用 `scripts/rebuild_kb_version.py`，保证单场景已有的 FAQ/文档入库、
质量门禁、版本激活和 Milvus schema reset 逻辑完全复用。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_kb_version.py"

DEFAULT_ALL_SCENARIOS = (
    "compliance_qa",
    "cross_border_risk",
    "engineering_project_qa",
    "enterprise_knowledge",
    "equipment_ops",
    "insurance_claims",
    "saas_support",
    "tender_contract_risk",
)


@dataclass(frozen=True)
class ScenarioRunResult:
    """单个场景重建结果。"""

    scenario_id: str
    ok: bool
    elapsed_seconds: float
    returncode: int


def parse_scenarios(value: str) -> list[str]:
    """解析逗号分隔场景列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def build_command(args: argparse.Namespace, scenario_id: str) -> list[str]:
    """构造单场景 rebuild 命令。"""
    command = [
        sys.executable,
        str(Path(args.rebuild_script)),
        "--scenario",
        scenario_id,
        "--new-version",
        "--force",
    ]
    if args.reset_collections:
        command.append("--reset-collections")
    if args.quality_gate:
        command.append("--quality-gate")
    if args.activate:
        command.append("--activate")
    if args.description:
        command.extend(["--description", args.description])
    if args.tenant_id:
        command.extend(["--tenant-id", args.tenant_id])
    if args.dataset_id:
        command.extend(["--dataset-id", args.dataset_id])
    if args.visibility:
        command.extend(["--visibility", args.visibility])
    for role in args.allowed_role or []:
        command.extend(["--allowed-role", role])
    return command


def run_one(args: argparse.Namespace, scenario_id: str) -> ScenarioRunResult:
    """执行单个场景重建。"""
    command = build_command(args, scenario_id)
    print("\n" + "=" * 88)
    print(f"Rebuilding scenario: {scenario_id}")
    print("Command:", " ".join(command))
    print("=" * 88)

    started = time.perf_counter()
    if args.dry_run:
        return ScenarioRunResult(
            scenario_id=scenario_id,
            ok=True,
            elapsed_seconds=0.0,
            returncode=0,
        )

    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - started
    return ScenarioRunResult(
        scenario_id=scenario_id,
        ok=completed.returncode == 0,
        elapsed_seconds=elapsed,
        returncode=completed.returncode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rebuild scenario knowledge base versions.")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_ALL_SCENARIOS),
        help=(
            "Comma-separated scenario ids. Defaults to all 8 frozen business scenarios."
        ),
    )
    parser.add_argument(
        "--reset-collections",
        action="store_true",
        help="Drop each scenario FAQ/Doc Milvus collection before rebuild.",
    )
    parser.add_argument(
        "--no-quality-gate",
        dest="quality_gate",
        action="store_false",
        help="Disable ingestion quality gate. Not recommended for production.",
    )
    parser.add_argument(
        "--no-activate",
        dest="activate",
        action="store_false",
        help="Only create staged versions; do not activate them.",
    )
    parser.add_argument("--description", default="batch rebuild scenarios", help="Version description.")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--allowed-role", action="append", default=None)
    parser.add_argument(
        "--rebuild-script",
        default=str(DEFAULT_REBUILD_SCRIPT),
        help=(
            "Path to rebuild_kb_version.py. When running this script through a host volume mounted "
            "at /work inside the api container, use --rebuild-script /app/scripts/rebuild_kb_version.py."
        ),
    )
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue after a scenario fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.set_defaults(quality_gate=True, activate=True)
    args = parser.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    if not scenarios:
        parser.error("--scenarios is empty.")

    results: list[ScenarioRunResult] = []
    for scenario_id in scenarios:
        result = run_one(args, scenario_id)
        results.append(result)
        if not result.ok and not args.continue_on_failure:
            break

    print("\nBatch rebuild summary")
    print("-" * 88)
    for result in results:
        status = "OK" if result.ok else f"FAILED({result.returncode})"
        print(f"{result.scenario_id:28s} {status:12s} {result.elapsed_seconds:8.2f}s")

    failed = [item for item in results if not item.ok]
    if failed:
        print("\nFailed scenarios:", ", ".join(item.scenario_id for item in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
