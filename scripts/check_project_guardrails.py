"""项目结构和代码约束守护检查。

该脚本把当前项目已经明确下来的工程约束做成可执行检查，而不是只写在文档里。
适合在每次重构后运行，也适合后续接入 CI。

检查内容：
1. Python 导入必须在文件头部，禁止函数内部临时导入；
2. 禁止 `try import A except ImportError import B` 这类隐藏兼容分支；
3. 禁止恢复旧版 `mysql_qa` / `rag_qa` / `legacy` 等运行入口；
4. 禁止代码重新引用旧链路模块；
5. `requirements.txt` 必须锁定版本，避免教学环境漂移；
6. 必须存在 `requirements.lock.txt`，记录当前环境的完整依赖快照。
"""

from __future__ import annotations

import ast
import csv
import re
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SCAN_DIRS = ("app.py", "qa_core", "scripts", "tests")
LEGACY_PATHS = (
    "mysql_qa",
    "rag_qa",
    "legacy",
    "api.py",
    "old_main.py",
    "new_main.py",
    "static/old_index.html",
    "tests/test_websocket_stream.py",
)
LEGACY_IMPORT_PREFIXES = ("mysql_qa", "rag_qa", "legacy")
FORBIDDEN_ENV_TOKENS = (
    "EDURAG_USE_LEGACY_CONFIG",
    "RERANK_ENABLED",
    "INTENT_LLM_ENABLED",
    "RETRIEVAL_VARIANT_ENABLED",
)
FORBIDDEN_IMPORT_PREFIXES = (
    *LEGACY_IMPORT_PREFIXES,
    "rank_bm25",
    "RedisSearch",
)
REQUIRED_GITIGNORE_PATTERNS = (".env", "logs/", "reports/", ".index_manifest/", "models/")
SCENARIO_REQUIRED_FIELDS = (
    "scenario_id",
    "display_name",
    "industry",
    "assistant_name",
    "business_domain",
    "support_contact",
    "valid_sources",
    "faq_collection",
    "doc_collection",
)
SUPPORTED_SCENARIO_DOC_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".doc", ".ppt", ".pptx", ".csv", ".xlsx", ".xls"}
FROZEN_SCENARIO_IDS = {
    "enterprise_knowledge",
    "saas_support",
    "equipment_ops",
    "compliance_qa",
    "cross_border_risk",
    "tender_contract_risk",
    "insurance_claims",
    "engineering_project_qa",
}


@dataclass(frozen=True)
class GuardrailIssue:
    """一条守护检查问题。"""

    file: Path
    line: int
    message: str

    def format(self) -> str:
        """返回适合终端输出的中文问题描述。"""
        rel = self.file.relative_to(PROJECT_ROOT)
        return f"{rel}:{self.line}: {self.message}"


def iter_python_files() -> list[Path]:
    """返回需要扫描的 Python 文件列表。"""
    files: list[Path] = []
    for item in PYTHON_SCAN_DIRS:
        path = PROJECT_ROOT / item
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def attach_parents(tree: ast.AST) -> None:
    """给 AST 节点补 parent 属性，方便判断导入是否位于模块头部。"""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)


def import_module_name(node: ast.Import | ast.ImportFrom) -> str:
    """提取导入语句的模块名称。"""
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    if not node.names:
        return ""
    return node.names[0].name


def node_contains_import(node: ast.AST) -> bool:
    """判断某个 AST 子树中是否包含导入语句。"""
    return any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node))


def check_python_file(path: Path) -> list[GuardrailIssue]:
    """检查单个 Python 文件的导入位置、fallback 导入和旧链路引用。"""
    issues: list[GuardrailIssue] = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    attach_parents(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parent = getattr(node, "parent", None)
            module_name = import_module_name(node)
            if not isinstance(parent, ast.Module):
                issues.append(GuardrailIssue(path, node.lineno, "导入必须放在文件头部，不能写在函数、方法或分支内部。"))
            if module_name.split(".")[0] in FORBIDDEN_IMPORT_PREFIXES:
                issues.append(GuardrailIssue(path, node.lineno, f"禁止引用旧链路模块：{module_name}"))
        elif isinstance(node, ast.Try):
            catches_import_error = any(
                isinstance(handler.type, ast.Name) and handler.type.id == "ImportError"
                for handler in node.handlers
                if handler.type is not None
            )
            if catches_import_error and node_contains_import(node):
                issues.append(GuardrailIssue(path, node.lineno, "禁止使用 ImportError fallback 导入；缺依赖应修正 requirements 和环境。"))

    if path.name != "check_project_guardrails.py":
        for token in FORBIDDEN_ENV_TOKENS:
            if token in source:
                issues.append(GuardrailIssue(path, 1, f"禁止在当前主链路代码中恢复旧配置或旧检索开关：{token}"))
    return issues


def check_legacy_paths() -> list[GuardrailIssue]:
    """检查旧链路目录和旧入口是否被重新创建。"""
    issues: list[GuardrailIssue] = []
    for rel_path in LEGACY_PATHS:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            issues.append(GuardrailIssue(path, 1, "旧链路文件或目录不应重新出现在工程中。"))
    return issues


def check_requirements() -> list[GuardrailIssue]:
    """检查直接依赖和完整锁文件是否满足当前项目约束。"""
    path = PROJECT_ROOT / "requirements.txt"
    lock_path = PROJECT_ROOT / "requirements.lock.txt"
    issues: list[GuardrailIssue] = []
    if not path.exists():
        issues.append(GuardrailIssue(path, 1, "requirements.txt 不存在。"))
    if not lock_path.exists():
        issues.append(GuardrailIssue(lock_path, 1, "requirements.lock.txt 不存在，无法追溯完整依赖快照。"))
    if issues:
        return issues
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            issues.append(GuardrailIssue(path, index, "依赖必须使用 == 锁定版本。"))
    if not lock_path.read_text(encoding="utf-8").strip():
        issues.append(GuardrailIssue(lock_path, 1, "requirements.lock.txt 为空。"))
    return issues


def check_secret_hygiene() -> list[GuardrailIssue]:
    """检查本地密钥和运行产物不会被默认提交。

    该检查不读取真实 `.env` 内容，只确认 `.gitignore` 会忽略敏感文件，并确认
    环境模板只保留占位符。这样既能保护本地 Key，又不会把密钥打印到终端。
    """
    issues: list[GuardrailIssue] = []
    gitignore = PROJECT_ROOT / ".gitignore"
    env_examples = [
        PROJECT_ROOT / ".env.local.example",
        PROJECT_ROOT / ".env.compose.example",
    ]
    if not gitignore.exists():
        return [GuardrailIssue(gitignore, 1, ".gitignore 不存在，真实 .env、logs、reports 可能被误提交。")]
    content = gitignore.read_text(encoding="utf-8")
    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        if pattern not in content:
            issues.append(GuardrailIssue(gitignore, 1, f".gitignore 必须包含敏感/运行产物规则：{pattern}"))
    for env_example in env_examples:
        if env_example.exists() and "sk-" in env_example.read_text(encoding="utf-8"):
            issues.append(GuardrailIssue(env_example, 1, f"{env_example.name} 只能写占位符，不能出现真实 Key 形态。"))
    return issues


def _faq_text(row: dict[str, str], *names: str) -> str:
    """按兼容列名读取 FAQ 单元格。"""
    return str(next((row.get(name) for name in names if row.get(name)), "")).strip()


def check_scenario_packages() -> list[GuardrailIssue]:
    """检查场景包配置是否满足多场景项目约束。

    场景包是当前项目扩展业务背景的唯一入口。这里把常见人为错误前置拦住：
    collection 重名、source 顺序不清、FAQ 分类写错、文档目录缺失、正则不可编译。
    这些问题如果等到线上检索时才发现，排查成本会比在守卫阶段高很多。
    """
    issues: list[GuardrailIssue] = []
    scenario_root = PROJECT_ROOT / "scenarios"
    if not scenario_root.exists():
        return [GuardrailIssue(scenario_root, 1, "scenarios 目录不存在，无法加载多业务场景配置。")]

    faq_collections: Counter[str] = Counter()
    doc_collections: Counter[str] = Counter()
    scenario_ids: Counter[str] = Counter()
    for config_path in sorted(scenario_root.glob("*/scenario.toml")):
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            issues.append(GuardrailIssue(config_path, getattr(exc, "lineno", 1) or 1, f"scenario.toml 解析失败：{exc}"))
            continue

        missing = [name for name in SCENARIO_REQUIRED_FIELDS if not payload.get(name)]
        for field_name in missing:
            issues.append(GuardrailIssue(config_path, 1, f"场景配置缺少必填字段：{field_name}"))
        if missing:
            continue

        scenario_id = str(payload["scenario_id"]).strip()
        scenario_ids[scenario_id] += 1
        if scenario_id not in FROZEN_SCENARIO_IDS:
            issues.append(
                GuardrailIssue(
                    config_path,
                    1,
                    f"业务场景已经冻结，不能继续新增未评审场景：{scenario_id}",
                )
            )
        if scenario_id != config_path.parent.name:
            issues.append(GuardrailIssue(config_path, 1, "scenario_id 必须与目录名一致，避免版本清单和资料目录错位。"))

        valid_sources = [str(item).strip() for item in payload.get("valid_sources", []) if str(item).strip()]
        if len(valid_sources) != len(set(valid_sources)):
            issues.append(GuardrailIssue(config_path, 1, "valid_sources 不能重复；它同时决定 source 白名单和匹配优先级。"))
        if not valid_sources:
            issues.append(GuardrailIssue(config_path, 1, "valid_sources 不能为空。"))

        source_labels = {str(key): str(value) for key, value in dict(payload.get("source_labels", {})).items()}
        source_patterns = {str(key): str(value) for key, value in dict(payload.get("source_patterns", {})).items()}
        for source in valid_sources:
            if source not in source_labels:
                issues.append(GuardrailIssue(config_path, 1, f"source_labels 缺少 {source} 的中文标签。"))
            if source not in source_patterns:
                issues.append(GuardrailIssue(config_path, 1, f"source_patterns 缺少 {source} 的推断规则。"))
        for source in set(source_labels) | set(source_patterns):
            if source not in valid_sources:
                issues.append(GuardrailIssue(config_path, 1, f"source 配置包含不在 valid_sources 中的分类：{source}"))
        for source, pattern in source_patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                issues.append(GuardrailIssue(config_path, 1, f"{source} 的 source_patterns 正则不可编译：{exc}"))

        faq_collection = str(payload["faq_collection"]).strip()
        doc_collection = str(payload["doc_collection"]).strip()
        faq_collections[faq_collection] += 1
        doc_collections[doc_collection] += 1
        if faq_collection == doc_collection:
            issues.append(GuardrailIssue(config_path, 1, "FAQ collection 和文档 collection 不能相同。"))

        faq_path = config_path.parent / "faq.csv"
        if not faq_path.exists():
            issues.append(GuardrailIssue(faq_path, 1, "场景 FAQ 文件不存在。"))
        else:
            issues.extend(_check_scenario_faq(faq_path, valid_sources))

        data_root = config_path.parent / "data"
        if not data_root.exists():
            issues.append(GuardrailIssue(data_root, 1, "场景 data 目录不存在。"))
        for source in valid_sources:
            source_dir = data_root / f"{source}_data"
            if not source_dir.exists():
                issues.append(GuardrailIssue(source_dir, 1, f"缺少 source 文档目录：{source}_data"))
                continue
            docs = [path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SCENARIO_DOC_SUFFIXES]
            if not docs:
                issues.append(GuardrailIssue(source_dir, 1, f"{source}_data 中没有可入库文档。"))

    duplicate_scenarios = [item for item, count in scenario_ids.items() if count > 1]
    duplicate_faq_collections = [item for item, count in faq_collections.items() if count > 1]
    duplicate_doc_collections = [item for item, count in doc_collections.items() if count > 1]
    missing_frozen_scenarios = sorted(FROZEN_SCENARIO_IDS - set(scenario_ids))
    if missing_frozen_scenarios:
        issues.append(GuardrailIssue(scenario_root, 1, f"冻结场景缺失：{missing_frozen_scenarios}"))
    if duplicate_scenarios:
        issues.append(GuardrailIssue(scenario_root, 1, f"scenario_id 重复：{duplicate_scenarios}"))
    if duplicate_faq_collections:
        issues.append(GuardrailIssue(scenario_root, 1, f"FAQ collection 重复：{duplicate_faq_collections}"))
    if duplicate_doc_collections:
        issues.append(GuardrailIssue(scenario_root, 1, f"文档 collection 重复：{duplicate_doc_collections}"))
    return issues


def _check_scenario_faq(path: Path, valid_sources: list[str]) -> list[GuardrailIssue]:
    """检查单个场景 FAQ CSV 的字段、空值、重复和 source 合法性。"""
    issues: list[GuardrailIssue] = []
    questions: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return [GuardrailIssue(path, 1, "FAQ CSV 缺少表头。")]
        for row_index, row in enumerate(reader, start=2):
            question = _faq_text(row, "question", "问题")
            answer = _faq_text(row, "answer", "答案")
            source = _faq_text(row, "source", "source_filter", "业务分类", "分类")
            if not question:
                issues.append(GuardrailIssue(path, row_index, "FAQ 问题不能为空。"))
            if not answer:
                issues.append(GuardrailIssue(path, row_index, "FAQ 答案不能为空。"))
            if not source:
                issues.append(GuardrailIssue(path, row_index, "FAQ source 不能为空。"))
            elif source not in valid_sources:
                issues.append(GuardrailIssue(path, row_index, f"FAQ source 不在当前场景 valid_sources 中：{source}"))
            if question:
                questions[question] += 1
            if source:
                source_counter[source] += 1
    for question, count in questions.items():
        if count > 1:
            issues.append(GuardrailIssue(path, 1, f"FAQ 问题重复：{question}"))
    missing_sources = [source for source in valid_sources if source_counter[source] == 0]
    if missing_sources:
        issues.append(GuardrailIssue(path, 1, f"FAQ 未覆盖这些 source：{missing_sources}"))
    return issues


def main() -> None:
    """执行全部守护检查。"""
    issues: list[GuardrailIssue] = []
    issues.extend(check_legacy_paths())
    issues.extend(check_requirements())
    issues.extend(check_secret_hygiene())
    issues.extend(check_scenario_packages())
    for path in iter_python_files():
        issues.extend(check_python_file(path))

    if issues:
        print("项目守护检查失败：")
        for issue in issues:
            print(f"- {issue.format()}")
        sys.exit(1)
    print("项目守护检查通过：导入位置、旧链路、fallback 导入、依赖锁、密钥卫生和冻结场景包均符合当前约束。")


if __name__ == "__main__":
    main()
