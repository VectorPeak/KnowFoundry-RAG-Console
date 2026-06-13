"""本地 LangChain/Milvus 运行依赖冒烟检查脚本。

该脚本用于启动服务前检查当前配置是否符合主链路预期。它不做破坏性写入，但会真实
校验必需环境：LLM Key、管理令牌、模型目录、场景配置、Milvus、MySQL 和 active
知识库版本。

使用场景：
- 修改本机 `.env` 或 Compose 注入配置后确认配置是否生效；
- 启动 API 前确认 Milvus URI、集合名、模型路径；
- 排查为什么本地读取的配置和预期不一致。

为什么要做硬校验：
- 当前架构不提供技术降级方案，依赖缺失时应该在启动前暴露；
- 比页面提问后才失败更容易定位；
- 不写 Milvus、不写 MySQL，仍然保持安全。

不适合的场景：
- 不要用它判断知识库是否有数据；
- 不要用它替代 `/api/retrieval/debug`；
- 不要在这里创建集合或写入测试文档。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.config.preflight import validate_runtime_environment
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario
from qa_core.config.settings import get_settings


def main() -> None:
    """打印当前运行配置，并验证必需环境。

    输出内容重点看 Milvus URI、当前场景集合名、本地模型路径和 valid_sources。若这些值和
    当前运行配置预期不一致，说明配置优先级或环境变量加载存在问题。

    使用场景：
    - 本地开发第一步确认配置；
    - Docker 环境通过 Compose 注入变量后确认路径；
    - 评测或入库脚本运行前确认集合名不会写错。
    """
    settings = get_settings()
    preflight = validate_runtime_environment()
    scenario = resolve_scenario()
    registry = get_scenario_registry()
    print("LangChain stack configuration")
    print(f"Active scenario: {scenario.scenario_id} / {scenario.display_name}")
    print(f"Available scenarios: {[item.scenario_id for item in registry.list_scenarios()]}")
    print(f"Milvus URI: {settings.milvus_uri}")
    print(f"Milvus database: {settings.milvus_database}")
    print(f"FAQ collection: {scenario.faq_collection}")
    print(f"Doc collection: {scenario.doc_collection}")
    print(f"LLM model: {settings.llm_model}")
    print(f"Embedding model: {settings.embedding_model_path}")
    print(f"Reranker model: {settings.reranker_model_path}")
    print(f"Valid sources: {scenario.valid_sources}")
    print(f"Active KB version: {preflight['active_kb_version']}")
    print("Runtime preflight: passed")


if __name__ == "__main__":
    main()

