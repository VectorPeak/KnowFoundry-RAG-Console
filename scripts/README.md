# 脚本目录说明

`scripts` 目录已经按学习成本做过收敛：根目录只放一期 RAG 日常开发、入库、评测和发布会高频使用的脚本；低频专项能力放到子目录。

## 根目录主脚本

这些是学生需要重点掌握的脚本。

| 场景 | 脚本 |
|---|---|
| 环境与项目约束 | `check_langchain_stack.py`、`check_project_guardrails.py`、`check_docs_consistency.py` |
| 文档和 FAQ 入库 | `rebuild_kb_version.py`、`manage_kb_versions.py`、`cleanup_missing_docs.py` |
| 入库质量 | `check_ingestion_quality_gate.py` |
| 主链路评测 | `evaluate_core_chain.py`、`evaluate_followup_chain.py`、`check_evaluation_gate.py`、`check_followup_gate.py` |
| 性能检查 | `collect_performance_baseline.py`、`check_performance_gate.py` |
| 接口验收 | `acceptance_smoke.py`、`api_e2e_smoke.py` |
| Bad Case 沉淀 | LangSmith Annotation/Dataset |
| 公共模块 | `common.py`、`eval_common.py`、`gate_utils.py` |

## 子目录专项脚本

这些脚本不要求第一遍学习时掌握，只在对应专题使用。

| 目录 | 用途 | 代表脚本 |
|---|---|---|
| `tools/` | 本地诊断、容量估算、依赖锁生成 | `check_local_runtime.py`、`capacity_estimate.py`、`generate_requirements_lock.py` |
| `kb/` | 知识库多版本召回对比 | `compare_kb_versions.py`、`compare_all_kb_versions.py` |
| `ocr/` | 扫描件 OCR 离线处理和复核后提升 | `run_offline_ocr.py`、`promote_ocr_candidates.py` |
| `enterprise_overlay/` | 企业仿真资料增强包治理 | `analyze_enterprise_data_realism.py`、`build_enterprise_overlay_dataset.py`、`run_enterprise_overlay_activation.py` |

## 常用命令

```powershell
python scripts/check_project_guardrails.py
python scripts/rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --quality-gate --activate
python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20
python scripts/check_evaluation_gate.py --dataset eval_sets/multi_scenario_smoke.json --limit 20
python scripts/api_e2e_smoke.py --base-url http://127.0.0.1:8000
python scripts/acceptance_smoke.py --base-url http://127.0.0.1:8000
```

专项命令示例：

```powershell
python scripts/tools/capacity_estimate.py --scenario enterprise_knowledge
python scripts/kb/compare_all_kb_versions.py --dataset eval_sets/business_depth_regression.json --per-scenario-limit 2
python scripts/ocr/run_offline_ocr.py --input-dir incoming_scans --output-dir reports/ocr/batch_001
python scripts/enterprise_overlay/check_enterprise_overlay_readiness.py
```

## 学习建议

第一遍只看根目录主脚本，理解“入库 -> 入库质量检查 -> RAG 回归验收 -> 接口验收”闭环。

第二遍再看 `kb/` 和 `tools/`，用于解释版本对比、容量评估和本地诊断。

`ocr/` 与 `enterprise_overlay/` 属于增强专题，不影响一期 RAG 主链路闭环。
