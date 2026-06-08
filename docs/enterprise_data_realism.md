# 场景库数据真实性评估与增强方案

当前项目的一期主链路已经企业化，但场景数据本身仍是教学样本。更准确的定位是：

- 系统架构：接近企业级 RAG 工程形态；
- 场景问题：接近真实业务问法；
- 数据现场：仍明显简化，缺少真实企业资料的规模、混乱度、版本冲突和组织差异。

## 当前距离

如果把真实企业知识库数据现场视为 100 分，当前 `scenarios/` 数据大约是 35-45 分。

原因不是业务方向错，而是资料还太干净：

- 每个场景只有少量 FAQ 和文档；
- 大多是结构清晰的 Markdown；
- 缺少真实企业常见的过期制度、冲突通知、表格附件、扫描件 OCR、区域差异和金额阈值；
- 缺少按法人主体、部门、角色、区域、合同类型变化的规则。

## 已新增企业仿真数据包

路径：

```text
data_packs/enterprise_realistic_pack/
```

它分为两部分：

```text
clean_overlay/
dirty_samples/
```

`clean_overlay/` 是可以作为后续入库增强的干净资料，覆盖区域差异、角色权限、金额阈值、例外审批、合同和付款风险、数据处理合规、工程资料补签、跨境单证金额变更、理赔材料金额不一致等企业资料特征。

`dirty_samples/` 不默认入库，只用于资料治理演示，覆盖过期口径、OCR 噪声、表格导出、多状态资料和 FAQ/正文潜在冲突。

## 为什么不默认并入 active 知识库

这是刻意设计。

真实企业资料里经常有脏数据，但不能把脏数据直接塞进线上 active 版本。正确做法是：

1. 干净增强资料先做入库质量报告；
2. 通过检查后再生成新知识库版本；
3. 新版本通过评测后才激活；
4. 脏数据样本只用于演示质量治理、冲突检测和人工复核。

这样既能讲真实企业资料复杂度，又不会破坏当前已经闭环的一期主链路。

## 分析脚本

新增脚本：

```text
scripts/enterprise_overlay/analyze_enterprise_data_realism.py
```

执行：

```powershell
python scripts/enterprise_overlay/analyze_enterprise_data_realism.py --output reports/verification/enterprise_data_realism_latest.json
```

报告会输出当前场景数据真实度、叠加企业仿真数据包后的真实度、FAQ 数量、文档数量、source 覆盖、文件格式分布，以及版本、例外、审批、阈值、区域、角色等企业化标记。

## clean overlay 预览构建

`clean_overlay/` 不是靠复制粘贴进入主场景，而是先生成一个临时预览数据集：

```powershell
python scripts/enterprise_overlay/build_enterprise_overlay_dataset.py --all-scenarios --output reports/verification/enterprise_overlay_build_latest.json
```

这个脚本会做四件事：

- 把 `scenarios/<scenario_id>/data` 和 `clean_overlay/<scenario_id>/data` 合并到 `reports/enterprise_overlay_build/<scenario_id>/data`；
- 把基础 `faq.csv` 和 `faq_overlay.csv` 合并成预览版 `faq.csv`；
- 复用现有入库质量报告检查 loader、source 白名单、chunk 质量、FAQ 质量和 FAQ/正文冲突；
- 输出每个场景的 `gate.ok`，只有通过的增强资料才具备进入知识库版本重建的资格。

这里刻意不自动激活知识库版本。正式上线仍要走：

```powershell
python scripts/rebuild_kb_version.py --scenario <scenario_id> --data-dir reports/enterprise_overlay_build/<scenario_id>/data --faq-csv reports/enterprise_overlay_build/<scenario_id>/faq.csv --new-version --force --quality-gate --activate
```

这样做的原因是：企业增强资料虽然是“干净样本”，但仍可能引入 FAQ 重复、正文依据缺失、source 分类错误或 chunk 质量问题。先预检，再重建，再评测，才是可控的企业资料上线流程。

## dirty samples 治理分析

`dirty_samples/` 只用于资料治理教学，不允许直接并入 active 知识库。执行：

```powershell
python scripts/enterprise_overlay/analyze_dirty_enterprise_samples.py --output reports/verification/dirty_enterprise_samples_latest.json
```

报告会把样本分成几类：

- `expired_policy`：过期制度或旧版本口径；
- `policy_conflict`：与当前 FAQ 或正文可能冲突；
- `ocr_review_required`：扫描件或 OCR 噪声，需要人工校验；
- `table_split_required`：表格导出资料，需要按表头、行号和业务主键保留结构化语义；
- `active_ingestion_blocked`：默认阻断 active 入库。

脏样本的正确流向是：先被治理脚本识别，再人工清洗或进入二期 Agent 审核流程；清洗完成后沉淀为 `clean_overlay/`，再跑 overlay 预检和知识库版本重建。

## 表格与 OCR 处理边界

当前一期已经把 CSV/Excel 表格作为正式资料类型接入入库链路：

- `.csv`、`.xlsx`、`.xls` 会被识别为表格资料；
- 每一行会转换成包含表头、工作表、行号和“列名：值”的 LangChain Document；
- metadata 会写入 `content_type=table_row`、`table_id`、`sheet_name`、`row_number`、`row_count`、`column_count`；
- 表格行不会再被普通字符切分拆散，避免检索到金额却丢失状态、材料名或审批人。

复杂 OCR 仍然不默认进入 active 入库链路。质量报告会识别疑似扫描件、OCR 噪声、断行、
错字等风险，并把它们记入 `ocr_risk_files`。这些文件必须先人工复核或走独立 OCR 清洗
流程，清洗后再沉淀为 clean overlay。这样能避免把扫描件识别错误直接写进线上知识库。

## 质量检查接入

企业资料真实度、clean overlay 预检和 dirty samples 治理摘要可以独立执行：

```powershell
python scripts/enterprise_overlay/analyze_enterprise_data_realism.py --output reports/verification/enterprise_data_realism_latest.json
python scripts/enterprise_overlay/check_enterprise_overlay_readiness.py --output reports/verification/enterprise_overlay_readiness_latest.json
```

这让面试讲法更完整：不是只说“我扩了 8 个业务场景”，而是能证明项目具备企业资料增强、入库质量检查、脏数据分流和版本化上线流程。

## 资料治理脚本

企业资料治理保留为独立脚本，按需执行：

```powershell
python scripts/enterprise_overlay/analyze_enterprise_data_realism.py --output reports/verification/enterprise_data_realism_latest.json
python scripts/enterprise_overlay/build_enterprise_overlay_dataset.py --all-scenarios --output reports/verification/enterprise_overlay_build_latest.json --strict
python scripts/enterprise_overlay/analyze_dirty_enterprise_samples.py --output reports/verification/dirty_enterprise_samples_latest.json
python scripts/enterprise_overlay/check_enterprise_overlay_readiness.py --output reports/verification/enterprise_overlay_readiness_latest.json
python scripts/enterprise_overlay/plan_enterprise_overlay_activation.py --output reports/verification/enterprise_overlay_activation_plan_latest.json
python scripts/enterprise_overlay/run_enterprise_overlay_activation.py --plan reports/verification/enterprise_overlay_activation_plan_latest.json --output reports/verification/enterprise_overlay_activation_run_latest.json
```

其中 `check_enterprise_overlay_readiness.py` 不调用模型，也不访问 Milvus。它检查四件事：

- clean overlay 是否全部通过入库质量检查；
- dirty samples 是否全部阻断 active 入库；
- 资料真实度是否确实提升；
- `eval_sets/enterprise_overlay_regression.json` 是否覆盖 clean overlay 的每条 FAQ。

## overlay 回归评测集

新增评测集：

```text
eval_sets/enterprise_overlay_regression.json
```

它覆盖 8 个场景的 24 条 clean overlay FAQ。增强资料正式激活到 active 知识库后，执行：

```powershell
python scripts/check_evaluation_gate.py --dataset eval_sets/enterprise_overlay_regression.json --limit 24 --output reports/verification/enterprise_overlay_evaluation_latest.json --gate-output reports/verification/enterprise_overlay_evaluation_gate_latest.json --min-source-inference-accuracy 1.0 --min-prompt-profile-accuracy 0.85
```

这一步才是真实问答链路验证：FAQ 直出、source 推断、Prompt Profile、检索来源和答案关键词都会被检查。
clean overlay 激活后建议手动跑这条回归，避免只看入库成功就误判为问答链路成功。

## overlay 上线计划

预检通过后，可以生成正式知识库版本重建命令：

```powershell
python scripts/enterprise_overlay/plan_enterprise_overlay_activation.py --output reports/verification/enterprise_overlay_activation_plan_latest.json
```

计划文件会给出每个场景的标准命令，格式类似：

```powershell
python scripts/rebuild_kb_version.py --scenario <scenario_id> --data-dir reports/enterprise_overlay_build/<scenario_id>/data --faq-csv reports/enterprise_overlay_build/<scenario_id>/faq.csv --new-version --force --quality-gate --description "企业 clean overlay 增强资料版本" --activate
```

这条链路的边界是：

1. `build_enterprise_overlay_dataset.py` 只生成候选数据集；
2. `check_enterprise_overlay_readiness.py` 判断候选资料是否具备上线资格；
3. `plan_enterprise_overlay_activation.py` 只生成上线命令；
4. `run_enterprise_overlay_activation.py` 只执行经过校验的 `rebuild_kb_version.py` 命令；
5. `rebuild_kb_version.py` 负责写 Milvus、生成新 kb_version、入库质量检查和激活；
6. `enterprise_overlay_regression.json` 用于激活后的真实链路回归。

## 状态页展示

状态页新增“企业资料治理”面板，展示：

- 当前资料真实度和增强后真实度；
- clean overlay 每个场景是否通过预检；
- dirty samples 风险分布和 active 入库阻断数量；
- overlay 回归评测覆盖情况；
- overlay 上线计划中的命令数量和阻断场景数。

接口：

```text
GET /api/admin/enterprise_governance
```

## 面试讲法

可以这样讲：

> 一期项目没有直接把真实企业脏数据塞进主链路，而是把企业资料分成 active 知识库和仿真治理样本。active 版本只接收通过入库质量检查的资料；过期、冲突、OCR 噪声和表格导出样本用于演示入库质量报告、FAQ/正文冲突检测、版本治理和人工复核流程。

这比单纯说“我做了 8 个行业场景”更有分量。重点不是行业外壳，而是知识库版本管理、资料入库质量检查、数据隔离、多源格式处理、边界拒答、bad case 回归和真实企业资料治理思路。

