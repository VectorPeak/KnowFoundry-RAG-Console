"""知识点 2.2：验证开发环境 — 检查 Python、依赖包、LLM 连通性
学习目标：掌握环境检查的基本方法
运行方式：python notes/01/2.2_verify_env.py
"""
import sys
import subprocess
import importlib

print("=" * 50)
print("KnowForge RAG Platform — 环境验证")
print("=" * 50)

# ---- 1. Python 版本 ----
print(f"\n1. Python 版本：{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info < (3, 10):
    print("   ⚠ 需要 Python 3.10+")
else:
    print("   ✅ OK")

# ---- 2. 关键依赖包 ----
required_packages = {
    "langchain": "langchain",
    "langchain_openai": "langchain-openai",
    "langchain_community": "langchain-community",
    "pymilvus": "pymilvus",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
}

print("\n2. 依赖包检查：")
for import_name, pip_name in required_packages.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        print(f"   ✅ {pip_name} ({version})")
    except ImportError:
        print(f"   ❌ {pip_name} — 未安装，请运行: pip install {pip_name}")

# ---- 3. 检查 .env 文件 ----
import os

print("\n3. 环境变量检查：")
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env")
if os.path.exists(env_path):
    print(f"   ✅ .env 文件存在：{env_path}")
    from dotenv import load_dotenv
    load_dotenv(env_path)
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    llm_model = os.getenv("LLM_MODEL", "")
    print(f"   DASHSCOPE_API_KEY = {'***' + api_key[-4:] if api_key else '❌ 未设置'}")
    print(f"   DASHSCOPE_BASE_URL = {base_url or '❌ 未设置'}")
    print(f"   LLM_MODEL = {llm_model or '❌ 未设置'}")
else:
    print(f"   ❌ .env 文件未找到：{env_path}")
    print("   请复制 .env.example 为 .env 并填写配置")

# ---- 4. LLM 连通性测试 ----
print("\n4. LLM 连通性测试：")
if os.getenv("DASHSCOPE_API_KEY"):
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        model = ChatOpenAI(
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            streaming=False,
            temperature=0,
        )
        response = model.invoke([HumanMessage(content="回复 OK")])
        answer = str(response.content).strip()
        print(f"   ✅ LLM 连通：发送 '回复 OK' → 收到 '{answer[:50]}'")
    except Exception as e:
        print(f"   ❌ LLM 调用失败：{e}")
else:
    print("   ⏭ 跳过（未配置 API Key）")

print("\n" + "=" * 50)
print("环境验证完成")
