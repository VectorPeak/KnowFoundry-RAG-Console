# 数据域隔离检索实施说明

本文档说明当前项目新增的轻量数据域隔离能力。它不是完整 RBAC 权限系统，而是在 RAG 检索阶段增加租户、数据集、可见级别和角色过滤，防止多租户、多数据集场景下出现串库检索。

## 1. 为什么需要

多场景 RAG 项目里，仅有 `scenario_id` 和 `source_filter` 还不够。

示例：

- 同样是 SaaS 客服场景，企业 A 和企业 B 的知识库不能互相检索；
- 同一个企业内部知识库里，公开资料和内部 SOP 不能混在一起；
- 同一场景下可能存在“产品文档库”“工单 FAQ 库”“内部排障库”等多个数据集；
- 面试或教学中，多租户 RAG、数据隔离检索、权限过滤是很重要的工程点。

因此当前实现增加一层数据域：

```text
scenario_id + kb_version + source + tenant_id + dataset_id + visibility + allowed_roles
```

## 2. 已落地代码

核心模块：

```text
qa_core/governance/data_scope.py
```

主链路改造：

- `qa_core/indexing/`：入库 metadata 写入 `tenant_id`、`dataset_id`、`visibility`、`allowed_roles`。
- `qa_core/retrieval/filters.py`：Milvus expr 追加数据域过滤条件。
- `qa_core/application/service.py`：`preview_query`、`stream_query`、`debug_retrieval` 传递 `DataScope`。
- `qa_core/schemas.py`：请求体支持 `tenant_id`、`dataset_id`、`visibility`、`user_role`、`user_roles`。
- `qa_core/api/chat.py`：HTTP、WebSocket、debug、feedback 透传数据域参数。
- `static/index.html`：页面增加租户、数据集、角色的轻量输入。
- `qa_core/memory/feedback.py`：反馈表记录 `tenant_id` 和 `dataset_id`。

## 3. 入库字段

每条 FAQ 和文档 chunk 都会写入：

```json
{
  "tenant_id": "default",
  "dataset_id": "default",
  "visibility": "public",
  "allowed_roles": ["public"]
}
```

默认值表示：

- `tenant_id=default`：默认租户；
- `dataset_id=default`：默认数据集；
- `visibility=public`：公开可检索；
- `allowed_roles=["public"]`：公开角色可检索。

## 4. 检索过滤逻辑

在线检索时会追加类似表达式：

```text
tenant_id == "default"
and dataset_id == "default"
and (visibility == "public")
and (array_contains(allowed_roles, "public"))
```

如果请求角色是 `internal`，可见级别会允许：

```text
visibility == "public" or visibility == "internal"
```

这是一种教学项目可解释、可验证的轻量策略。真实生产系统可以把用户认证、角色授权和组织结构放在业务系统里，RAG 只接收已经计算好的数据域。

## 5. 入库命令

默认公开数据入库：

```powershell
python scripts\rebuild_kb_version.py --scenario saas_support --new-version --force --activate --tenant-id default --dataset-id default --visibility public --allowed-role public
```

内部数据入库：

```powershell
python scripts\rebuild_kb_version.py --scenario enterprise_knowledge --new-version --force --activate --tenant-id company_a --dataset-id internal_sop --visibility internal --allowed-role internal --allowed-role admin
```

只入库 FAQ：

```powershell
python scripts\rebuild_kb_version.py --scenario saas_support --new-version --force --quality-gate --activate --tenant-id company_a --dataset-id faq_v1 --visibility public --allowed-role public
```

只入库文档：

```powershell
python scripts\rebuild_kb_version.py --scenario saas_support --new-version --force --quality-gate --activate --tenant-id company_a --dataset-id product_docs --visibility internal --allowed-role internal
```

## 6. API 请求示例

检索调试：

```json
{
  "query": "发票什么时候可以开",
  "scenario_id": "saas_support",
  "source_filter": "billing",
  "tenant_id": "default",
  "dataset_id": "default",
  "visibility": "public",
  "user_role": "public"
}
```

WebSocket 请求：

```json
{
  "query": "发票什么时候可以开",
  "session_id": "saas_support:xxx",
  "scenario_id": "saas_support",
  "source_filter": "billing",
  "tenant_id": "default",
  "dataset_id": "default",
  "visibility": "public",
  "user_role": "public"
}
```

## 7. 验证结果

已验证：

- `dataset_id=default` 可以命中 SaaS 发票 FAQ 和文档；
- `dataset_id=missing_dataset` 返回 0 条 FAQ、0 条文档；
- 流式主链路正常返回 FAQ 直出答案；
- `end.retrieval.data_scope` 会返回当前数据域，便于调试。

## 8. 当前边界

当前能力只负责检索隔离，不负责：

- 用户登录；
- token 鉴权；
- 用户角色从数据库查询；
- 组织架构继承；
- 复杂 ABAC 条件；
- 审计审批流。

这些属于业务系统或权限系统。RAG 主链路只接收已经确定的数据域，并把它转换成 Milvus 过滤条件。

## 9. 简历表达

可以描述为：

```text
实现多租户 RAG 数据隔离检索，在向量库 metadata 中维护 tenant_id、dataset_id、visibility、allowed_roles，
在线检索阶段统一生成 Milvus 过滤表达式，确保不同租户、数据集和角色之间不会发生串库召回。
```
