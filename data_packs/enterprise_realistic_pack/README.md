# 企业仿真数据包

这个目录用于补齐“教学样本”和“真实企业资料现场”之间的距离。

它分为两类资料：

- `clean_overlay/`：可以作为后续入库增强的干净资料，强调版本、区域、角色、金额阈值、审批链和例外规则。
- `dirty_samples/`：不默认入库，只用于演示资料治理，包括过期文件、口径冲突、扫描件占位、表格转写和命名混乱样本。

当前主链路仍使用 `scenarios/` 下的冻结场景数据。这个数据包是企业仿真增强层，不会自动影响 active 知识库版本。

## 使用方式

先分析资料真实度：

```powershell
python scripts/enterprise_overlay/analyze_enterprise_data_realism.py --output reports/verification/enterprise_data_realism_latest.json
```

再构建 clean overlay 预览数据集并跑入库质量门禁：

```powershell
python scripts/enterprise_overlay/build_enterprise_overlay_dataset.py --all-scenarios --output reports/verification/enterprise_overlay_build_latest.json
```

最后分析 dirty samples 的治理风险：

```powershell
python scripts/enterprise_overlay/analyze_dirty_enterprise_samples.py --output reports/verification/dirty_enterprise_samples_latest.json
```

只有 `clean_overlay/` 预检通过后，才可以进入知识库版本重建和回归评测。`dirty_samples/`
始终作为治理样本，不直接进入 active 知识库。

