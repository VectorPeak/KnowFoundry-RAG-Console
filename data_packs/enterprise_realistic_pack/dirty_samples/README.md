# 脏数据样本说明

这些文件用于资料治理演示，不默认入库：

- `cross_border_invoice_conflict_v2025.md`：过期且与当前口径冲突。
- `expense_scan_ocr_noise.txt`：模拟扫描件 OCR 错字和格式噪声。
- `engineering_table_export.csv`：模拟表格导出后的多状态资料。

使用这些样本时，应先运行入库质量报告，观察冲突、低质量 chunk 或人工复核提示。

当前推荐先运行独立治理分析：

```powershell
python scripts/enterprise_overlay/analyze_dirty_enterprise_samples.py --output reports/verification/dirty_enterprise_samples_latest.json
```

这些样本的默认结论都是 `allow_active_ingestion=false`。如果后续要用于主链路，需要先人工
清洗成 `clean_overlay/` 中的干净资料，再执行 overlay 预检、知识库版本重建和 RAG 回归。

