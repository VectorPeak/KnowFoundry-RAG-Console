# 脚本使用主次指南

项目里的 `scripts/` 不是让学生逐个背诵的“脚本大全”。它的正确学习方式是：围绕真实任务找入口脚本，再顺藤摸瓜理解它调用的 `qa_core` 能力。

第一遍学习只需要掌握 P0 和 P1。P2、P3 是企业化和扩展材料，用于面试加分或二期规划，不应该压到主链路学习里。

## 一、脚本分层

| 层级 | 定位 | 是否第一遍必须掌握 | 学习目标 |
|------|------|------------------|----------|
| P0 主入口脚本 | 跑通一期核心闭环 | 必须 | 能完成环境检查、入库、问答验证和接口验收。 |
| P1 回归验收脚本 | 证明系统质量稳定 | 建议掌握 | 能解释评测、性能、入库质量为什么能防止退化。 |
| P2 运维诊断脚本 | 企业交付和问题排查 | 第二遍学习 | 能看懂报告、LangSmith Trace、Bad Case 和版本对比。 |
| P3 扩展专项脚本 | OCR、overlay、容量等扩展能力 | 了解即可 | 知道什么时候用，不要求第一期全部精读。 |

## 二、P0：主入口脚本

这些脚本对应一期最核心的“资料入库 → 激活版本 → 页面问答 → 接口验收”闭环。

| 脚本 | 使用场景 | 学生需要掌握什么 |
|------|----------|------------------|
| `scripts/check_langchain_stack.py` | 检查 LangChain、Milvus、模型依赖是否可用 | 理解项目拥抱 LangChain 生态，不走自研检索栈。 |
| `scripts/rebuild_kb_version.py` | 重建并激活一个知识库版本 | 理解 staged/active 版本切换，不直接覆盖线上知识库。 |
| `scripts/check_docs_consistency.py` | 文档一致性检查 | 理解 README、架构文档和课程边界如何保持一致。 |

建议第一遍按这个顺序跑：

```powershell
python scripts\check_langchain_stack.py
python scripts\rebuild_kb_version.py --scenario enterprise_knowledge
python scripts\check_project_guardrails.py
python scripts\check_docs_consistency.py
```

真实 API、Milvus、MySQL 和模型服务验收放到 `api_e2e_smoke.py` 与 `acceptance_smoke.py`，课堂上按需演示即可。

## 三、P1：回归验收脚本

这些脚本是企业级 RAG 和普通 Demo 的分水岭：它们用指标证明系统没有退化。

| 脚本 | 使用场景 | 关键指标 |
|------|----------|----------|
| `scripts/check_ingestion_quality_gate.py` | 入库质量检查 | 是否允许激活新知识库版本。 |
| `scripts/evaluate_core_chain.py` | 跑主链路评测集 | Recall@K、MRR、关键词覆盖、命中路径准确率。 |
| `scripts/check_evaluation_gate.py` | RAG 回归验收 | 场景隔离、source 推断、Prompt Profile、FAQ 直出。 |
| `scripts/collect_performance_baseline.py` | 采集性能基线 | 首 token、总耗时、阶段耗时。 |
| `scripts/check_performance_gate.py` | 性能回归验收 | 是否存在慢请求和缺失阶段耗时。 |
| `scripts/check_project_guardrails.py` | 项目守护检查 | 禁止旧链路、fallback 导入、密钥泄漏和场景包漂移。 |
| `scripts/check_docs_consistency.py` | 文档一致性检查 | 讲义、README、场景和代码是否对齐。 |

学习重点不是记住每个参数，而是能讲清楚：

1. 质量报告负责描述“发生了什么”。
2. 验收脚本负责判断“是否允许通过”。
3. 评测集负责把主观体验变成可重复验证的样本。
4. 阶段耗时必须存在，否则无法定位性能瓶颈。

## 四、P2：运维诊断脚本

这些脚本适合第二遍学习或面试扩展。

| 脚本 | 使用场景 | 学习重点 |
|------|----------|----------|
| LangSmith Annotation/Dataset | 从 Trace 中标注失败样本并沉淀为回归集 | Bad Case 如何从线上样本进入评测资产。 |
| `scripts/evaluate_followup_chain.py` | 多轮追问评测 | 追问改写、历史读取、source 保持是否稳定。 |
| `scripts/check_followup_gate.py` | 多轮追问回归验收 | 防止多轮问答质量退化。 |
| `scripts/kb/compare_kb_versions.py` | 对比两个知识库版本 | 新版本激活前如何验证召回没有倒退。 |
| `scripts/kb/compare_all_kb_versions.py` | 全场景版本对比 | 多场景封版前的批量检查。 |

P2 的教学方式建议用一个问题串起来：

```text
某次回答错了 → LangSmith Trace 里定位原因 → Annotation 人工确认 → 加入 Dataset → 下次变更前跑评测检查。
```

这样学生能理解“质量闭环”，而不是只看到一堆脚本文件。

## 五、P3：扩展专项脚本

这些脚本不进入第一遍主线，只在企业场景需要时讲。

| 脚本范围 | 对应能力 | 为什么后置 |
|----------|----------|------------|
| `scripts/ocr/` | OCR 候选资料治理 | OCR 成本高、依赖重、失败率高，适合作为复杂资料专题。 |
| `scripts/enterprise_overlay/` | 企业真实度增强 | 用于让样例数据更接近真实企业资料，不影响 RAG 主链路。 |
| `scripts/tools/capacity_estimate.py` | 容量评估 | 适合部署规划，不影响第一遍理解在线问答。 |
| `scripts/tools/check_local_runtime.py` | 本地环境深度诊断 | 排障时使用，不需要逐行讲。 |
| `scripts/tools/generate_requirements_lock.py` | 依赖锁生成 | 用于交付一致性，了解用途即可。 |

## 六、脚本学习建议

学生第一遍不要从 `scripts/` 文件夹开始读代码。推荐顺序是：

1. 先读 `qa_core/application/service.py`，理解服务入口。
2. 再读 `qa_core/pipeline/`，理解问答主流程。
3. 再读 `qa_core/indexing/`，理解资料如何入库。
4. 最后回到 `scripts/`，把脚本看作调用主链路的任务入口。

面试表达可以这样说：

```text
这个项目没有把所有能力塞进一个 main.py，而是把核心业务沉到 qa_core。
scripts 只负责具体任务编排，比如入库、评测和验收。
这样学生学习时先掌握主链路，再按任务理解脚本；企业落地时也更容易接入 CI/CD。
```

## 七、哪些脚本不建议第一遍逐行讲

以下脚本价值明确，但不适合第一遍逐行拆解：

| 脚本 | 原因 |
|------|------|
| `scripts/tools/check_local_runtime.py` | 面向本地排障，分支多，教学收益低。 |
| `scripts/enterprise_overlay/*.py` | 业务真实度增强，容易分散 RAG 主链路。 |
| `scripts/ocr/*.py` | OCR 属于复杂资料治理专题，依赖和异常场景较多。 |
| `scripts/kb/compare_all_kb_versions.py` | 适合封版前批量运行，不需要一开始深入。 |

这些脚本不是“多余”，而是“学习优先级靠后”。代码存在的意义是支撑企业化交付，不是增加学生第一遍的记忆负担。
