"""生成项目级依赖锁文件。

不要直接把全局 Anaconda 环境 `pip freeze` 写进项目锁文件。全局环境通常包含大量与本
项目无关的包，会让教学项目显得臃肿，也会让复现环境变得不可控。

本脚本从 `requirements.txt` 的直接依赖出发，读取当前已安装包的 metadata，递归收集
真实依赖闭包，然后生成 `requirements.lock.txt`。这样锁文件既包含传递依赖，又不会把
无关 IDE、Notebook、系统工具包一起带进来。
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Iterable

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIRECT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
DEFAULT_LOCK_FILE = PROJECT_ROOT / "requirements.lock.txt"


def requirement_lines(path: Path) -> list[str]:
    """读取 requirements.txt 中的有效依赖行。"""
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        lines.append(line)
    return lines


def installed_distributions() -> dict[str, metadata.Distribution]:
    """按规范化包名索引当前环境中已安装的发行包。

    Anaconda + 用户 site-packages 容易出现同名包重复安装。`metadata.distributions()` 的
    遍历顺序通常和当前解释器查找顺序一致，因此这里保留第一次出现的发行包，避免后面
    路径中的旧包覆盖实际 import 会使用的版本。
    """
    result: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            normalized = canonicalize_name(name)
            result.setdefault(normalized, dist)
    return result


def marker_allows(requirement: Requirement, extras: Iterable[str]) -> bool:
    """判断某条依赖声明在当前环境和父包 extras 下是否应该纳入闭包。

    Python 包 metadata 里常见写法是：
    `Requires-Dist: pymilvus-model ; extra == "model"`。
    如果父依赖写了 `pymilvus[model]`，这里就应该把 `model` extra 对应的依赖纳入锁文件。
    """
    if requirement.marker is None:
        return True
    if requirement.marker.evaluate({"extra": ""}):
        return True
    return any(requirement.marker.evaluate({"extra": extra}) for extra in extras)


def parse_requirement(line: str) -> Requirement:
    """解析一行依赖声明，去掉环境标记前后的多余空白。"""
    return Requirement(line.strip())


def collect_dependency_closure(requirement_path: Path) -> set[str]:
    """从直接依赖递归收集已安装包依赖闭包。"""
    distributions = installed_distributions()
    queue: deque[tuple[str, set[str]]] = deque()
    seen: set[str] = set()
    for line in requirement_lines(requirement_path):
        req = parse_requirement(line)
        queue.append((canonicalize_name(req.name), set(req.extras)))

    while queue:
        package_name, extras = queue.popleft()
        if package_name in seen:
            continue
        seen.add(package_name)
        dist = distributions.get(package_name)
        if dist is None:
            raise RuntimeError(f"当前环境缺少 requirements.txt 声明的依赖：{package_name}")
        for raw_requirement in dist.requires or []:
            req = parse_requirement(raw_requirement)
            if marker_allows(req, extras):
                queue.append((canonicalize_name(req.name), set(req.extras)))
    return seen


def distribution_lock_line(dist: metadata.Distribution) -> str:
    """把一个已安装包转换成 lock 文件行。"""
    name = dist.metadata.get("Name") or dist.name
    version = dist.version
    normalized_name = re.sub(r"[-_.]+", "-", name).strip()
    return f"{normalized_name}=={version}"


def build_lock(requirement_path: Path) -> str:
    """构建 requirements.lock.txt 的完整文本。"""
    distributions = installed_distributions()
    package_names = collect_dependency_closure(requirement_path)
    lines = [
        "# 项目级完整依赖锁文件，由 scripts/tools/generate_requirements_lock.py 生成。",
        "# 只包含 requirements.txt 的递归依赖闭包，不包含全局环境中的无关包。",
    ]
    for package_name in sorted(package_names):
        lines.append(distribution_lock_line(distributions[package_name]))
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Generate focused requirements lock from installed dependency closure.")
    parser.add_argument("--requirements", default=str(DIRECT_REQUIREMENTS))
    parser.add_argument("--output", default=str(DEFAULT_LOCK_FILE))
    return parser


def main() -> None:
    """生成项目级依赖锁文件。"""
    parser = build_parser()
    args = parser.parse_args()
    requirement_path = Path(args.requirements)
    output_path = Path(args.output)
    lock_text = build_lock(requirement_path)
    output_path.write_text(lock_text, encoding="utf-8")
    print(f"Generated {output_path} with {len(lock_text.splitlines()) - 2} packages.")


if __name__ == "__main__":
    main()


