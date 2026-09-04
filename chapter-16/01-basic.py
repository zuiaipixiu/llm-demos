from langchain.agents import Tool, AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from searxng import SearxNGClient

from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatOpenAI(
    model="gpt-5.5",
    openai_api_key=API_KEY,
    openai_api_base="https://techhub.ssss818.com/v1",
    temperature=0  # 降低随机性，使输出更稳定
)

def search_with_searx(query: str) -> str:
    # 配置 SearxNG 客户端，使用本地服务器
    searx = SearxNGClient()
    results = searx.search(query)
    if results and 'results' in results:
        return "\n".join([f"- {r.get('title', '')}: {r.get('content', '')}" for r in results['results']])
    return "未找到相关信息"

def get_current_time() -> str:
    current_time = datetime.now()
    return f"当前时间是：{current_time.strftime('%Y-%m-%d %H:%M:%S')}"

# 定义工具
tools = [
    Tool(
        name="搜索",
        func=search_with_searx,
        description="当你需要查找网络上的信息时使用。输入应该是一个搜索查询字符串。",
    ),
    Tool(
        name="获取日期与时间",
        func=get_current_time,
        description="获取当前日期与时间，不需要输入参数。"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)

try: 
    result = agent.invoke({"input": "2025年google大模型最新的进展是什么？"})
    print(result)
except Exception as e:
    print(f"Error: {e}")