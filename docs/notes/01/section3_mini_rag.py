"""核心代码串联：用纯 Python 模拟一次最简 RAG 问答
学习目标：把"向量相似度 + LLM 调用"串联成一次完整的增强生成
运行方式：python notes/01/section3_mini_rag.py
前置条件：已配置 .env 中的 DASHSCOPE_API_KEY
"""
import os
import math
from dotenv import load_dotenv

# 加载环境变量（先找项目根目录的 .env）
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env")
load_dotenv(env_path)


# ===== 步骤1：准备"知识库"（3篇文档，每篇已手工转成简化向量） =====
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


knowledge_base = [
    {
        "title": "入职流程",
        "content": "入职流程包括：1. 提交身份证和学历证明 2. 签订劳动合同 3. HR开通办公账号 4. 领取办公设备",
        "vector": [0.1, 0.1, 0.9, 0.1, 0.1],
    },
    {
        "title": "请假制度",
        "content": "员工请假需提前1天在OA系统提交申请，3天以上需部门负责人审批。病假需提供医院证明。",
        "vector": [0.9, 0.1, 0.1, 0.1, 0.1],
    },
    {
        "title": "报销规则",
        "content": "差旅报销需保留发票，每月25日前提交报销单，餐饮费每人每天不超过100元。",
        "vector": [0.1, 0.9, 0.1, 0.1, 0.1],
    },
]

# ===== 步骤2：用户提问 → 向量化（用关键词示意） =====
# 实际 RAG 中这一步由 Embedding 模型（如 BGE-M3）完成
query = "入职需要什么材料"
query_vector = [0.1, 0.1, 0.8, 0.2, 0.1]  # 模拟：和"入职流程"文档最接近

print("=" * 60)
print("最简 RAG 演示：", query)
print("=" * 60)

# ===== 步骤3：检索 — 找最相似的文档 =====
print("\n检索结果（余弦相似度排名）：")
scored = [(cosine_similarity(query_vector, doc["vector"]), doc) for doc in knowledge_base]
scored.sort(key=lambda x: x[0], reverse=True)

for score, doc in scored:
    bar = "#" * int(score * 20)
    print(f"  {doc['title']:8s}  {score:.4f}  {bar}")

top_doc = scored[0][1]
print(f"\n  [TOP] 最相关文档：{top_doc['title']}")
print(f"  [DOC] 内容：{top_doc['content']}")

# ===== 步骤4：生成 — 把检索结果 + 问题发给 LLM =====
if os.getenv("DASHSCOPE_API_KEY"):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    model = ChatOpenAI(
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        streaming=False,
        temperature=0,
    )

    messages = [
        SystemMessage(content="你是企业知识助手。严格根据提供的参考资料回答，不得编造。"),
        HumanMessage(content=f"参考资料：\n{top_doc['content']}\n\n用户问题：{query}\n\n请根据参考资料回答："),
    ]
    response = model.invoke(messages)
    print("\n--- LLM 回答 ---：")
    print(f"  {response.content}")
else:
    print("\n[SKIP] 跳过 LLM 调用（未配置 API Key）")
    print("  如果配置了 .env，LLM 会基于检索到的文档生成答案")

print("\n" + "=" * 60)
print("这就是 RAG 的完整流程：检索 → 增强 → 生成")
