# 二期 Agent 边界设计

## 1. 结论

一期继续保持 RAG 主链路纯净：检索、问答、引用、历史、评测和入库质量检查已经闭环。
二期 Agent 不应该绕过 RAG 直接查库，也不应该把工具调用塞进一期问答流程。

推荐边界是：

```text
用户任务
  -> Agent 工作流
  -> 调用 RAG 证据检索工具
  -> 基于证据做计划、核查、草稿、处置建议
  -> 输出结构化结果
```

RAG 负责“证据是否可靠”，Agent 负责“下一步怎么处理”。

## 2. 为什么这样分

- RAG 的核心难点是知识库质量、检索召回、引用来源和回答边界；
- Agent 的核心难点是任务规划、工具调用、状态流转和异常恢复；
- 两者混在一期里，会让学生同时面对两套复杂度，学习成本会变高；
- 先把 RAG 做稳，再用 LangGraph 编排 Agent，面试表达更清晰。

## 3. RAG 对 Agent 暴露的能力

二期 Agent 只消费结构化证据，不直接拼 Prompt 查 Milvus。

当前一期源码不提前放 Agent 预留实现。二期真正开始时，再新建 Agent 模块，用清晰的
工具契约固定“Agent 如何合法调用 RAG”。这样一期代码只服务 RAG 闭环，学生不会在
第一阶段被未启用的 Agent 代码分散注意力。

建议工具：

| 工具 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `rag_search_evidence` | 场景、问题、source、数据域、kb_version | answer、sources、hit_type、confidence、trace_id | 获取证据和引用 |
| `rag_debug_retrieval` | 场景、问题、source、数据域、kb_version | FAQ/doc 候选、分数、query variants | Agent 自检证据是否足够 |
| `kb_version_status` | scenario_id | active_version、历史版本 | 判断是否使用正确知识库 |
| `bad_case_submit` | trace_id、原因、备注 | feedback_id | 把失败任务沉淀回治理闭环 |

二期建议落地的最小契约：

| 代码对象 | 作用 |
| --- | --- |
| `RAGEvidenceRequest` | 明确 query、scenario、source、kb_version、tenant、dataset、role 等入参 |
| `RAGEvidenceResult` | 返回 answer、sources、hit_type、trace_id、retrieval、intent 和 can_continue |
| `AgentRagTool.search_evidence` | 复用 QAService.stream_query，确保 Agent 与页面共用同一条 RAG 主链路 |
| `AgentRagTool.debug_retrieval` | 复用 QAService.debug_retrieval，供 Agent 做证据自检 |

其中 `can_continue=False` 的典型情况包括：

- `scenario_boundary`：问题属于其他业务场景；
- `source_boundary`：用户选择了错误资料分类；
- `insufficient_context`：当前知识库依据不足。

这些情况不能由 Agent 强行补全，只能提示切换场景、切换分类、补充资料或进入人工确认。

## 4. Skill 能力包边界

二期需要引入 Skill Registry，但 Skill 不是随便新增工具函数，而是可复用的业务能力包。

推荐 Skill 结构：

```text
skill_id
适用场景
触发意图
输入 Schema
输出 Schema
可调用工具白名单
Prompt Profile
权限等级
是否需要人工确认
评测样本
```

优先 Skill：

| Skill | 适用场景 | 说明 |
| --- | --- | --- |
| `contract_risk_review` | 招投标合同、合规 | 输出风险点、依据、建议动作 |
| `claim_material_check` | 保险理赔 | 检查材料缺失、责任依据和除外风险 |
| `engineering_document_check` | 工程项目 | 检查资料完整性、规范依据和整改项 |
| `ticket_draft` | SaaS 客服、设备运维 | 生成工单草稿、处理步骤和升级条件 |
| `trade_compliance_check` | 跨境贸易 | 检查制裁、单证、信用证和物流风险 |

Skill 和 Tool 的边界：

```text
Skill = 面向业务任务的能力包
Tool = Skill 可以调用的具体动作
LangGraph = 编排 Skill 和 Tool 的状态流
```

## 5. Agent 不应该做的事

- 不直接连接 Milvus；
- 不绕过 `kb_version`、tenant、dataset、role 数据隔离；
- 不自己决定“低分也能回答”；
- 不把未确认结论包装成确定建议；
- 不自动跨场景检索，只能建议用户切换场景。

## 6. 推荐二期场景

二期不用新增行业包，而是在现有 8 个场景上升级任务形态。

| 场景 | Agent 方向 | 示例 |
| --- | --- | --- |
| 工程项目资料 | 资料核查 Agent | 检查隐蔽工程验收资料是否齐全 |
| 招投标合同 | 合同风险审查 Agent | 识别非标准付款和验收风险 |
| 保险理赔 | 理赔材料审核 Agent | 输出缺失材料清单和人工复核项 |
| 跨境贸易 | 贸易合规核查 Agent | 检查制裁、信用证、单证一致性风险 |
| SaaS 客服 | 工单草稿 Agent | 根据知识库生成客服处理草稿 |

## 7. LangGraph 与 A2A 协议规划

二期建议采用“内部编排用 LangGraph，对外互通用 A2A”的分层方案：

```text
前端 Agent 模式
  -> FastAPI Agent API
  -> LangGraph Agent 工作流
  -> 工具注册器
       -> 一期 QAService / RAG 证据工具
       -> 工单草稿 / 清单生成 / 版本查询 / 人工转接
  -> 可选 A2A Adapter
       -> 对外暴露 Agent Card
       -> 接收或委派跨 Agent 任务
```

LangGraph 适合做二期内部工作流，因为 Agent 任务不是单次问答，而是有状态的多步骤流程：
任务识别、证据检索、证据质量判断、工具调用、人工确认、失败恢复和结构化输出都需要明确节点。

A2A 不替代 LangGraph，也不替代工具注册器。它的定位是“跨 Agent 通信协议”。当项目未来
把工程资料 Agent、合同风险 Agent、理赔审核 Agent 拆成独立服务，或者需要和外部 Agent
互通时，再通过 A2A 暴露能力、接收任务、返回任务状态和流式事件。

边界如下：

| 能力 | 定位 | 本项目规划 |
| --- | --- | --- |
| LangGraph | 单个 Agent 内部的计划、状态、工具调用、人工确认和恢复 | 二期核心编排引擎 |
| A2A | 独立 Agent 之间的发现、通信、任务状态和流式事件协议 | 二期后半增加 Adapter |
| MCP | Agent 连接工具、资源和外部 API 的协议 | 可作为工具接入方向，优先级低于 LangGraph |
| QAService | 一期 RAG 能力入口 | 被 Agent 当作本地工具调用 |

推荐落地顺序：

| 阶段 | 目标 | 是否进入二期首版 |
| --- | --- | --- |
| 2.1 LangGraph 单 Agent | 完成计划、证据检索、工具调用、人工确认、结构化输出 | 必须 |
| 2.2 多 Agent 内部协作 | 在同一服务内拆出合同、理赔、工程等专业 Agent 节点 | 可选 |
| 2.3 A2A Adapter | 暴露 Agent Card，支持 A2A 任务发送、流式状态和结果查询 | 可选增强 |

如果启用 A2A，只开放受控能力：

| A2A 能力 | 本项目实现建议 |
| --- | --- |
| Agent Card | 暴露能力、版本、输入输出格式，不暴露内部供应商和敏感系统名称 |
| message/send | 接收一次 Agent 任务请求 |
| message/stream | 把 LangGraph 事件映射成 A2A 流式任务事件 |
| tasks/get | 查询任务状态和最终结果 |
| tasks/cancel | 取消未完成任务 |
| artifacts | 返回结构化清单、草稿、风险项和引用来源 |

安全约束：

- A2A 接口必须使用服务端鉴权，不复用普通页面会话；
- 不允许 A2A 直接激活知识库版本、删除数据或执行外部黑盒系统动作；
- 涉及合同、赔付、费用、合规、安全的结论必须保留人工确认节点；
- Agent Card 和输出内容不能暴露后台第三方系统名称，用户侧只看到业务结果。

## 8. LangGraph 编排建议

推荐最小工作流：

```text
classify_task
  -> search_evidence
  -> check_evidence_quality
  -> plan_actions
  -> generate_structured_output
  -> human_review_if_needed
```

关键点：

- `check_evidence_quality` 必须检查 hit_type、sources、top_score、kb_version；
- 如果 RAG 返回 `scenario_boundary` 或 `source_boundary`，Agent 只能要求用户切换场景或分类；
- 高风险任务必须进入人工复核节点；
- 所有 Agent 输出保留 trace_id，方便回查证据链。

## 9. 多模态、GraphRAG、MCP 的后续边界

这些能力有价值，但不进入二期首版主线。

| 能力 | 阶段 | 边界 |
| --- | --- | --- |
| 多模态入库 | 二期后半或三期 | OCR/VLM 解析后必须人工复核、入库质量检查、版本化入库 |
| GraphRAG | 三期 | 只增强关系推理，不替代 Milvus Hybrid RAG |
| MCP | 二期后半 | 用于标准化接入外部工具和资源，优先级低于 LangGraph |

推荐顺序：

```text
二期首版：LangGraph + Skill Registry + Tool Registry + 人工确认 + Agent 评测
二期后半：A2A Adapter + MCP Adapter + 内部多 Agent 协作
三期：多模态入库 + GraphRAG + 更复杂跨 Agent 协作
```

多模态优先处理复杂 PDF、扫描件、图纸、票据和验收照片；GraphRAG 优先用于合同、理赔、
工程规范这类关系链明显的场景。

## 10. Agent 工程治理边界

二期 Agent 必须补齐工程治理，不然会退化成“模型会调用工具”的演示。

必须规划：

| 治理项 | 说明 |
| --- | --- |
| 任务状态机 | `created/planning/running/waiting_approval/completed/failed/cancelled` |
| Skill 生命周期 | `skill_version/prompt_version/tool_policy_version/enabled/eval_set` |
| 工具权限分级 | L0 只读、L1 草稿、L2 业务判断、L3 改变系统状态 |
| 人工确认 | 合同、赔付、费用、合规、安全类任务必须确认 |
| Agent 评测 | 评任务分类、Skill 选择、计划完整性、工具参数、人工确认触发 |
| Agent Trace | 记录计划、工具调用、RAG 证据、人工确认、错误和耗时 |
| 敏感信息脱敏 | trace、A2A 消息、Agent Card、报告和前端输出都要脱敏 |
| 模型路由 | 普通 RAG、复杂计划、结构化抽取、多模态解析分开治理 |
| 灰度回滚 | Skill、Prompt、Tool Policy、模型路由、知识库版本都要可回滚 |

二期首版至少要做到：

```text
Agent 任务有 task_id
Skill 有版本
工具有权限等级
高风险动作有人工确认
Agent 有独立评测集
Trace 能回放完整任务
```

## 11. 面试表达

可以这样讲：

> 一期我没有急着做 Agent，而是先把 RAG 的证据链、版本、隔离、入库质量检查和 bad case 闭环做稳。二期内部用 LangGraph 编排任务流，把 RAG 封装成受控工具；如果后续需要跨系统或跨 Agent 协作，再在 AgentService 外围增加 A2A Adapter。这样 LangGraph 负责内部流程，A2A 负责 Agent 间互通，不会破坏一期 RAG 的版本、隔离和引用来源。
