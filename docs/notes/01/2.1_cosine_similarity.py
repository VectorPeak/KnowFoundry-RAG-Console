"""知识点 2.1：向量相似度 — 手动计算余弦相似度
学习目标：理解 RAG 检索的数学基础
运行方式：python notes/01/2.1_cosine_similarity.py
"""
import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    公式：cos(θ) = (A·B) / (|A| × |B|)
    结果范围 [-1, 1]，1 表示完全相同，0 表示正交（无关），-1 表示完全相反。
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b)


# ========== 演示：用简化向量理解"语义相近 = 向量接近" ==========
word_vectors = {
    "猫":       [0.8, 0.2, 0.1],
    "小猫":     [0.7, 0.3, 0.1],
    "汽车":     [0.1, 0.9, 0.8],
    "车辆":     [0.1, 0.8, 0.9],
    "入职":     [0.0, 0.1, 0.9],
    "入职流程": [0.0, 0.1, 0.8],
}

print("=" * 50)
print("向量相似度演示 — 余弦相似度")
print("=" * 50)

pairs = [
    ("猫", "小猫"), ("汽车", "车辆"), ("入职", "入职流程"),  # 应相似
    ("猫", "汽车"), ("猫", "入职"),                          # 应不相似
]
for w1, w2 in pairs:
    score = cosine_similarity(word_vectors[w1], word_vectors[w2])
    tag = "← 近义/相关" if score > 0.7 else "← 无关"
    print(f"  {w1:6s} <-> {w2:6s}  = {score:.4f}  {tag}")

print()
print("这就是 RAG 检索的核心：用户问题 → 向量 → 找最相似文档")
