"""检查正式讲义、章节代码和动画流程是否保持对齐。

这个检查用于防止后续维护时出现三类回退：
- 正式讲义暴露内部授课实现标记；
- 章节 README 缺少统一的课堂说明结构；
- 章节缺少可运行源码或测试文件。
- 已打磨章节的动画流程仍停留在旧代码口径。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_utf8_stdio, print_json, utc_now, write_optional_json
from scripts.check_chapter_maps import validate_maps


CHAPTERS: tuple[tuple[str, str], ...] = (
    ("05", "ch05_intent_classification"),
    ("06", "ch06_retrieval_strategy"),
    ("07", "ch07_query_rewrite_variants"),
    ("08", "ch08_milvus_hybrid_search"),
    ("09", "ch09_qaservice_orchestration"),
    ("10", "ch10_rag_pipeline"),
    ("11", "ch11_prompt_engineering"),
    ("12", "ch12_fastapi_service"),
    ("13", "ch13_preflight_checks"),
    ("14", "ch14_kb_versioning"),
    ("15", "ch15_data_isolation"),
    ("16", "ch16_ingestion_pipeline"),
    ("17", "ch17_quality_evaluation"),
    ("18", "ch18_test_system"),
    ("19", "ch19_observability_tracing"),
)

CHAPTER_ANIMATION_FILES: dict[str, str] = {
    "05": "05-intent-flow.html",
    "06": "06-retrieval-flow.html",
    "07": "07-query-flow.html",
    "08": "08-milvus-hybrid-search.html",
    "09": "09-qaservice-orchestration.html",
    "10": "10-rag-pipeline.html",
    "11": "11-prompt-engineering.html",
    "12": "12-fastapi-service.html",
    "13": "13-preflight-checks.html",
    "14": "14-kb-versioning.html",
    "15": "15-data-isolation.html",
    "16": "16-ingestion-pipeline.html",
    "17": "17-quality-evaluation.html",
    "18": "18-testing-system.html",
    "19": "19-observability-tracing.html",
}

UNIFIED_CHAPTER_ANIMATION_MARKER = "chapter-animation-template: unified-v2"

MIN_PUBLIC_DOC_LINES = 120

FORBIDDEN_PUBLIC_DOC_FRAGMENTS = (
    "跟敲",
    "codealong/chapters/",
    "codealong-code-flow.html",
    "跟敲代码全链路",
    "全链路图",
)

REQUIRED_ANIMATION_PAGE_FRAGMENTS = (
    UNIFIED_CHAPTER_ANIMATION_MARKER,
    "业务执行流程图",
    "代码执行流程图",
    "返回本章讲义",
    "business-flow.html",
)

GAP_DECISION_DOC = PROJECT_ROOT / "codealong" / "CODEALONG_TO_PROJECT_GAP_DECISION.md"

REQUIRED_README_SECTIONS = (
    "## 本章目标",
    "## 和上一章的关系",
    "## 本章代码",
    "## 运行",
    "## 测试",
    "## 对应主项目源码",
    "## 本章边界",
)

REQUIRED_QA_CORE_FILES: dict[str, tuple[str, ...]] = {
    "05": (
        "qa_core/intent/classifier.py",
        "qa_core/pipeline/steps.py",
        "qa_core/scenarios/registry.py",
        "qa_core/scenarios/boundary.py",
    ),
    "06": (
        "qa_core/intent/question_category.py",
        "qa_core/retrieval/strategy.py",
    ),
    "07": (
        "qa_core/pipeline/rewrite.py",
        "qa_core/pipeline/query_variants.py",
        "qa_core/llm/client.py",
        "qa_core/prompts/constants.py",
    ),
    "08": (
        "qa_core/retrieval/store.py",
        "qa_core/retrieval/results.py",
        "qa_core/retrieval/filters.py",
        "qa_core/retrieval/ranking.py",
        "qa_core/retrieval/factory.py",
    ),
    "09": (
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/application/service.py",
        "qa_core/application/factory.py",
    ),
    "10": (
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/pipeline/rag.py",
        "qa_core/pipeline/runtime.py",
        "qa_core/pipeline/events.py",
        "qa_core/pipeline/context.py",
        "qa_core/pipeline/retrieval_steps.py",
    ),
    "11": (
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/prompts/profiles.py",
        "qa_core/prompts/selector.py",
        "qa_core/prompts/templates.py",
    ),
    "12": (
        "app.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/api/chat.py",
        "qa_core/api/service_context.py",
        "qa_core/api/error_handlers.py",
        "qa_core/schemas.py",
    ),
    "13": (
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/config/settings.py",
        "qa_core/config/preflight.py",
        "qa_core/config/logging_config.py",
        "qa_core/schemas.py",
    ),
    "14": (
        "qa_core/common.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/schemas.py",
        "qa_core/governance/kb_versions.py",
        "qa_core/api/kb_versions.py",
    ),
    "15": (
        "qa_core/common.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/schemas.py",
        "qa_core/governance/data_scope.py",
        "qa_core/retrieval/filters.py",
    ),
    "16": (
        "qa_core/common.py",
        "qa_core/utils.py",
        "qa_core/document_metadata.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/schemas.py",
        "qa_core/pipeline/citations.py",
        "qa_core/indexing/chunking.py",
        "qa_core/indexing/document_loaders.py",
        "qa_core/indexing/table_documents.py",
        "qa_core/indexing/document_normalizer.py",
        "qa_core/indexing/faq_ingestion.py",
        "qa_core/indexing/service.py",
    ),
    "17": (
        "qa_core/common.py",
        "qa_core/utils.py",
        "qa_core/document_metadata.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/schemas.py",
        "qa_core/pipeline/citations.py",
        "qa_core/indexing/table_documents.py",
        "qa_core/quality/ingestion.py",
        "qa_core/quality/faq.py",
        "qa_core/quality/conflicts.py",
        "qa_core/quality/chunk.py",
    ),
    "18": (
        "qa_core/common.py",
        "qa_core/utils.py",
        "qa_core/document_metadata.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/schemas.py",
        "qa_core/pipeline/citations.py",
        "qa_core/indexing/table_documents.py",
        "scripts/check_project_guardrails.py",
        "scripts/acceptance_smoke.py",
    ),
    "19": (
        "qa_core/common.py",
        "qa_core/utils.py",
        "qa_core/document_metadata.py",
        "qa_core/memory/__init__.py",
        "qa_core/memory/base.py",
        "qa_core/memory/history.py",
        "qa_core/memory/feedback.py",
        "qa_core/schemas.py",
        "qa_core/pipeline/citations.py",
        "qa_core/indexing/table_documents.py",
        "qa_core/observability/langsmith_adapter.py",
        "qa_core/pipeline/runtime.py",
    ),
}

STALE_ANIMATION_FRAGMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "05": (
        ("内存 FAQ 字典", "第 05 章动画仍使用旧口径；当前 FAQ fast path 只允许作为后续真实 FAQ 检索的前置判断。"),
    ),
    "07": (
        ("用确定性同义词规则生成少量等价表达", "第 07 章动画缺少 LLM structured-output fallback 路径。"),
        ("用本地同义词补充稳定变体", "第 07 章动画仍使用旧口径；查询变体同义词规则已迁移到 config/rules.toml。"),
    ),
}

STALE_QUERY_VARIANT_CODE_FRAGMENTS = (
    'if "流程" in query',
    'if "webhook" in normalized',
    'query.replace("失败", "报错")',
    "SHORT_STRUCTURED_MAX_CHARS",
)

FOLLOW_UP_TEST_MARKERS = (
    "FOLLOW_UP",
    "那审批呢",
    "审批",
)

FOLLOW_UP_REWRITE_MARKERS = (
    "rewritten_query",
    "rewritten",
)

FOLLOW_UP_VARIANT_MARKERS = (
    "query_variants",
    "variants",
)


def text_of(path: Path) -> str:
    """读取文本文件；不存在时返回空字符串。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_python(path: Path) -> ast.Module | None:
    """解析 Python 文件；文件不存在时返回 None。"""
    if not path.exists():
        return None
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function_arg_names(module: ast.Module, function_name: str) -> set[str]:
    """读取函数的位置参数和 keyword-only 参数名。"""
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)}
    return set()


def calls_named(module: ast.Module, function_name: str) -> list[ast.Call]:
    """返回模块里对指定函数名或同名属性的调用。"""
    calls: list[ast.Call] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == function_name:
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == function_name:
            calls.append(node)
    return calls


def find_doc(chapter_no: str) -> Path | None:
    """按章节号定位正式讲义 Markdown。"""
    matches = sorted((PROJECT_ROOT / "docs").glob(f"{chapter_no}-*.md"))
    return matches[0] if matches else None


def add_failure(
    failures: list[dict[str, Any]],
    *,
    metric: str,
    chapter: str,
    path: str,
    message: str,
) -> None:
    failures.append({"metric": metric, "chapter": chapter, "path": path, "message": message})


def check_chapter(chapter_no: str, chapter_dir_name: str) -> list[dict[str, Any]]:
    """检查单个 codealong 章节和对应讲义入口。"""
    failures: list[dict[str, Any]] = []
    marker = f"codealong/chapters/{chapter_dir_name}"
    chapter_rel = f"codealong/chapters/{chapter_dir_name}"
    chapter_dir = PROJECT_ROOT / chapter_rel

    if not chapter_dir.exists():
        add_failure(failures, metric="chapter_dir", chapter=chapter_no, path=chapter_rel, message="codealong 章节目录不存在")
        return failures

    readme = chapter_dir / "README.md"
    readme_text = text_of(readme)
    normalized_readme_text = readme_text.replace("\\", "/")
    if not readme_text:
        add_failure(failures, metric="chapter_readme", chapter=chapter_no, path=str(readme.relative_to(PROJECT_ROOT)), message="章节 README 不存在或为空")
    else:
        for section in REQUIRED_README_SECTIONS:
            if section not in readme_text:
                add_failure(
                    failures,
                    metric="chapter_readme_section",
                    chapter=chapter_no,
                    path=str(readme.relative_to(PROJECT_ROOT)),
                    message=f"章节 README 缺少固定段落：{section}",
                )
        if marker not in normalized_readme_text:
            add_failure(
                failures,
                metric="chapter_readme_path",
                chapter=chapter_no,
                path=str(readme.relative_to(PROJECT_ROOT)),
                message=f"章节 README 缺少自身目录路径：{marker}",
            )

    src_dir = chapter_dir / "src"
    if src_dir.exists():
        add_failure(
            failures,
            metric="chapter_legacy_src",
            chapter=chapter_no,
            path=str(src_dir.relative_to(PROJECT_ROOT)),
            message="最终跟敲交付禁止使用孤立 src 小样例；请迁移到 qa_core/ + scripts/ + tests/ 的增量工程结构",
        )

    qa_core_dir = chapter_dir / "qa_core"
    if not qa_core_dir.exists():
        add_failure(
            failures,
            metric="chapter_qa_core",
            chapter=chapter_no,
            path=str(qa_core_dir.relative_to(PROJECT_ROOT)),
            message="章节缺少 qa_core 业务源码目录，无法证明它是主项目同向的增量实现",
        )

    for relative in REQUIRED_QA_CORE_FILES.get(chapter_no, ()):
        required_path = chapter_dir / relative
        if not required_path.exists():
            add_failure(
                failures,
                metric="chapter_required_module",
                chapter=chapter_no,
                path=str(required_path.relative_to(PROJECT_ROOT)),
                message=f"章节缺少主项目同向模块：{relative}",
            )

    scripts_dir = chapter_dir / "scripts"
    script_files = sorted(scripts_dir.glob("*.py")) if scripts_dir.exists() else []
    if not script_files:
        add_failure(
            failures,
            metric="chapter_scripts",
            chapter=chapter_no,
            path=str(scripts_dir.relative_to(PROJECT_ROOT)),
            message="章节缺少 scripts/*.py 课堂运行入口",
        )

    tests_dir = chapter_dir / "tests"
    test_files = sorted(tests_dir.glob("test_*.py")) if tests_dir.exists() else []
    if not test_files:
        add_failure(failures, metric="chapter_tests", chapter=chapter_no, path=str(tests_dir.relative_to(PROJECT_ROOT)), message="章节 tests 目录缺少 test_*.py")

    doc = find_doc(chapter_no)
    if doc is None:
        add_failure(failures, metric="doc_markdown", chapter=chapter_no, path=f"docs/{chapter_no}-*.md", message="正式讲义 Markdown 不存在")
    else:
        doc_rel = str(doc.relative_to(PROJECT_ROOT))
        doc_text = text_of(doc)
        doc_lines = doc_text.splitlines()
        first_nonempty_line = next((line.strip() for line in doc_lines if line.strip()), "")
        if not first_nonempty_line.startswith("# "):
            add_failure(
                failures,
                metric="doc_public_h1",
                chapter=chapter_no,
                path=doc_rel,
                message="正式讲义必须以一级标题开头，避免章节页面结构异常。",
            )
        if len(doc_lines) < MIN_PUBLIC_DOC_LINES:
            add_failure(
                failures,
                metric="doc_public_body_length",
                chapter=chapter_no,
                path=doc_rel,
                message=f"正式讲义正文过短，可能被同步脚本误裁剪；当前 {len(doc_lines)} 行，至少应为 {MIN_PUBLIC_DOC_LINES} 行。",
            )
        for fragment in FORBIDDEN_PUBLIC_DOC_FRAGMENTS:
            if fragment in doc_text:
                add_failure(
                    failures,
                    metric="doc_public_internal_marker",
                    chapter=chapter_no,
                    path=doc_rel,
                    message=f"正式讲义不能暴露内部授课实现信息：{fragment}",
                )
        site_doc = PROJECT_ROOT / "site" / f"{doc.stem}.html"
        site_text = text_of(site_doc)
        if not site_text:
            add_failure(failures, metric="site_doc", chapter=chapter_no, path=str(site_doc.relative_to(PROJECT_ROOT)), message="MkDocs 站点 HTML 不存在或为空，请先运行 python -m mkdocs build")
        else:
            for fragment in FORBIDDEN_PUBLIC_DOC_FRAGMENTS:
                if fragment in site_text:
                    add_failure(
                        failures,
                        metric="site_doc_public_internal_marker",
                        chapter=chapter_no,
                        path=str(site_doc.relative_to(PROJECT_ROOT)),
                        message=f"MkDocs 站点讲义不能暴露内部授课实现信息：{fragment}",
                    )

    return failures


def check_animation_alignment() -> list[dict[str, Any]]:
    """检查动画流程是否覆盖已打磨章节，并拦截已知旧口径。"""
    failures: list[dict[str, Any]] = []
    animation_dir = PROJECT_ROOT / "docs" / "animation"
    business_flow = animation_dir / "business-flow.html"
    business_text = text_of(business_flow)
    if not business_text:
        add_failure(
            failures,
            metric="animation_business_flow",
            chapter="ALL",
            path=str(business_flow.relative_to(PROJECT_ROOT)),
            message="业务流程总动画不存在或为空",
        )
    else:
        for chapter_no, _ in CHAPTERS:
            if f"CH{chapter_no}" not in business_text:
                add_failure(
                    failures,
                    metric="animation_business_flow_chapter",
                    chapter=chapter_no,
                    path=str(business_flow.relative_to(PROJECT_ROOT)),
                    message=f"业务流程总动画缺少 CH{chapter_no}",
                )

    for chapter_no, chapter_dir_name in CHAPTERS:
        chapter_dir = PROJECT_ROOT / "codealong" / "chapters" / chapter_dir_name
        is_polished = (chapter_dir / "qa_core").exists() and not (chapter_dir / "src").exists()
        expected_animation_name = CHAPTER_ANIMATION_FILES[chapter_no]
        expected_animation = animation_dir / expected_animation_name
        single_chapter_animations = sorted(animation_dir.glob(f"{chapter_no}-*.html"))
        if is_polished and not expected_animation.exists():
            add_failure(
                failures,
                metric="animation_chapter_flow",
                chapter=chapter_no,
                path=str(expected_animation.relative_to(PROJECT_ROOT)),
                message=f"已打磨章节缺少固定命名的单章代码执行动画：{expected_animation_name}",
            )
        if expected_animation.exists():
            expected_animation_text = text_of(expected_animation)
            for fragment in REQUIRED_ANIMATION_PAGE_FRAGMENTS:
                if fragment not in expected_animation_text:
                    add_failure(
                        failures,
                        metric="animation_chapter_template",
                        chapter=chapter_no,
                        path=str(expected_animation.relative_to(PROJECT_ROOT)),
                        message=f"单章动画未使用统一模板或缺少必要导航：{fragment}",
                    )

        for animation_path in single_chapter_animations:
            animation_text = text_of(animation_path)
            for fragment, message in STALE_ANIMATION_FRAGMENTS.get(chapter_no, ()):
                if fragment in animation_text:
                    add_failure(
                        failures,
                        metric="animation_stale_text",
                        chapter=chapter_no,
                        path=str(animation_path.relative_to(PROJECT_ROOT)),
                        message=message,
                    )
    return failures


def check_rule_config_alignment() -> list[dict[str, Any]]:
    """检查 query variants 规则已经配置化，防止业务词表回写到代码里。"""

    failures: list[dict[str, Any]] = []
    rules_toml = PROJECT_ROOT / "config" / "rules.toml"
    rules_text = text_of(rules_toml)
    for fragment in (
        "[query_variants]",
        "short_structured_markers",
        "[[query_variants.replacements]]",
    ):
        if fragment not in rules_text:
            add_failure(
                failures,
                metric="rules_query_variants_config",
                chapter="07",
                path=str(rules_toml.relative_to(PROJECT_ROOT)),
                message=f"config/rules.toml 缺少查询变体配置：{fragment}",
            )

    query_variant_files = [
        PROJECT_ROOT / "qa_core" / "pipeline" / "query_variants.py",
        *(
            PROJECT_ROOT / "codealong" / "chapters" / chapter_dir_name / "qa_core" / "pipeline" / "query_variants.py"
            for chapter_no, chapter_dir_name in CHAPTERS
            if chapter_no >= "07"
        ),
    ]
    for path in query_variant_files:
        text = text_of(path)
        if not text:
            continue
        if "get_rule_config().query_variants" not in text:
            add_failure(
                failures,
                metric="query_variants_config_usage",
                chapter="07",
                path=str(path.relative_to(PROJECT_ROOT)),
                message="查询变体代码未读取 get_rule_config().query_variants，可能仍在代码中写死业务词表",
            )
        for fragment in STALE_QUERY_VARIANT_CODE_FRAGMENTS:
            if fragment in text:
                add_failure(
                    failures,
                    metric="query_variants_hardcoded_rule",
                    chapter="07",
                    path=str(path.relative_to(PROJECT_ROOT)),
                    message=f"查询变体代码仍包含写死业务规则片段：{fragment}",
                )
        module = parse_python(path)
        if module is None:
            continue
        if "allow_short_structured" not in function_arg_names(module, "generate_query_variants"):
            add_failure(
                failures,
                metric="query_variants_follow_up_signature",
                chapter="07",
                path=str(path.relative_to(PROJECT_ROOT)),
                message="generate_query_variants 必须保留 allow_short_structured，避免追问改写后被短问题规则截断。",
            )
        for fragment in ("FOLLOW_UP_REWRITE_MARKERS",):
            if fragment not in text:
                add_failure(
                    failures,
                    metric="query_variants_follow_up_guard",
                    chapter="07",
                    path=str(path.relative_to(PROJECT_ROOT)),
                    message=f"查询变体缺少追问闭环回归片段：{fragment}",
                )

    runtime_call_paths = [
        PROJECT_ROOT / "qa_core" / "pipeline" / "steps.py",
        *(
            path
            for _, chapter_dir_name in CHAPTERS
            for path in (
                PROJECT_ROOT / "codealong" / "chapters" / chapter_dir_name
            ).glob("scripts/*.py")
        ),
        *(
            path
            for _, chapter_dir_name in CHAPTERS
            for path in (
                PROJECT_ROOT / "codealong" / "chapters" / chapter_dir_name
            ).rglob("qa_core/application/*.py")
        ),
        *(
            path
            for _, chapter_dir_name in CHAPTERS
            for path in (
                PROJECT_ROOT / "codealong" / "chapters" / chapter_dir_name
            ).rglob("qa_core/pipeline/*.py")
        ),
    ]
    for path in sorted({path for path in runtime_call_paths if path.exists()}):
        if path.name == "query_variants.py":
            continue
        text = text_of(path)
        if "generate_query_variants(" not in text:
            continue
        module = parse_python(path)
        if module is None:
            continue
        for call in calls_named(module, "generate_query_variants"):
            keyword_names = {keyword.arg for keyword in call.keywords if keyword.arg}
            if "allow_short_structured" not in keyword_names:
                add_failure(
                    failures,
                    metric="query_variants_follow_up_call",
                    chapter="07",
                    path=str(path.relative_to(PROJECT_ROOT)),
                    message="运行链路调用 generate_query_variants 时必须传 allow_short_structured=intent.intent == 'FOLLOW_UP'。",
                )

    return failures


def check_follow_up_variant_test_coverage() -> list[dict[str, Any]]:
    """确保第 07-19 章测试都锁住追问改写后的查询变体闭环。

    追问改写由真实 LLM 生成，不能再要求某个固定字符串。这里检查的是
    章节测试是否覆盖 FOLLOW_UP、追问输入、改写结果和变体输出这条真实闭环。
    """

    failures: list[dict[str, Any]] = []
    for chapter_no, chapter_dir_name in CHAPTERS:
        if chapter_no < "07":
            continue
        tests_dir = PROJECT_ROOT / "codealong" / "chapters" / chapter_dir_name / "tests"
        test_text = "\n".join(text_of(path) for path in sorted(tests_dir.glob("test_*.py")))
        for fragment in FOLLOW_UP_TEST_MARKERS:
            if fragment not in test_text:
                add_failure(
                    failures,
                    metric="query_variants_follow_up_test",
                    chapter=chapter_no,
                    path=str(tests_dir.relative_to(PROJECT_ROOT)),
                    message=f"章节测试缺少真实追问闭环断言：{fragment}",
                )
        if not any(fragment in test_text for fragment in FOLLOW_UP_REWRITE_MARKERS):
            add_failure(
                failures,
                metric="query_variants_follow_up_test",
                chapter=chapter_no,
                path=str(tests_dir.relative_to(PROJECT_ROOT)),
                message="章节测试缺少追问改写结果断言：rewritten_query/rewritten",
            )
        if not any(fragment in test_text for fragment in FOLLOW_UP_VARIANT_MARKERS):
            add_failure(
                failures,
                metric="query_variants_follow_up_test",
                chapter=chapter_no,
                path=str(tests_dir.relative_to(PROJECT_ROOT)),
                message="章节测试缺少查询变体输出断言：query_variants/variants",
                )
    return failures


def check_ch19_project_gap_decision() -> list[dict[str, Any]]:
    """确保 CH19 与完整项目的文件差异都有明确决策登记。"""

    failures: list[dict[str, Any]] = []
    decision_text = text_of(GAP_DECISION_DOC)
    if not decision_text:
        add_failure(
            failures,
            metric="codealong_gap_decision_doc",
            chapter="19",
            path=str(GAP_DECISION_DOC.relative_to(PROJECT_ROOT)),
            message="缺少 CH19 与完整项目差异决策文档",
        )
        return failures

    required_fragments = (
        "`mainline`",
        "`appendix`",
        "`productization`",
        "## 差异清单",
        "## 主线补齐顺序",
    )
    for fragment in required_fragments:
        if fragment not in decision_text:
            add_failure(
                failures,
                metric="codealong_gap_decision_structure",
                chapter="19",
                path=str(GAP_DECISION_DOC.relative_to(PROJECT_ROOT)),
                message=f"差异决策文档缺少固定内容：{fragment}",
            )

    ch19_dir = PROJECT_ROOT / "codealong" / "chapters" / "ch19_observability_tracing"
    project_files = {path.relative_to(PROJECT_ROOT).as_posix() for path in (PROJECT_ROOT / "qa_core").rglob("*.py")}
    project_files.add("app.py")
    ch19_files = {path.relative_to(ch19_dir).as_posix() for path in (ch19_dir / "qa_core").rglob("*.py")}
    if (ch19_dir / "app.py").exists():
        ch19_files.add("app.py")

    missing_files = sorted(project_files - ch19_files)
    for relative in missing_files:
        if f"`{relative}`" not in decision_text:
            add_failure(
                failures,
                metric="codealong_gap_decision_missing_file",
                chapter="19",
                path=str(GAP_DECISION_DOC.relative_to(PROJECT_ROOT)),
                message=f"CH19 缺少完整项目文件但未登记决策：{relative}",
            )

    mainline_files = [
        line.split("|", maxsplit=4)[1].strip().strip("`")
        for line in decision_text.splitlines()
        if line.startswith("| `") and "| `mainline` |" in line
    ]
    if not mainline_files:
        add_failure(
            failures,
            metric="codealong_gap_decision_mainline",
            chapter="19",
            path=str(GAP_DECISION_DOC.relative_to(PROJECT_ROOT)),
            message="差异决策文档没有登记任何 mainline 文件",
        )

    return failures


def run_check() -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for chapter_no, chapter_dir_name in CHAPTERS:
        failures.extend(check_chapter(chapter_no, chapter_dir_name))
    for issue in validate_maps():
        add_failure(
            failures,
            metric="chapter_map_alignment",
            chapter=issue.chapter,
            path=issue.file_path,
            message=f"{issue.title} / {issue.symbol}: {issue.message}",
        )
    failures.extend(check_animation_alignment())
    failures.extend(check_rule_config_alignment())
    failures.extend(check_follow_up_variant_test_coverage())
    failures.extend(check_ch19_project_gap_decision())

    return {
        "report_type": "codealong_alignment_check",
        "created_at": utc_now(),
        "ok": not failures,
        "checked_chapter_count": len(CHAPTERS),
        "failed_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检查正式讲义与 codealong 跟敲章节是否对齐。")
    parser.add_argument("--json-output", type=Path, default=None, help="可选：把检查报告写入 JSON 文件。")
    args = parser.parse_args()

    configure_utf8_stdio()
    report = run_check()
    write_optional_json(args.json_output, report)
    print_json(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

