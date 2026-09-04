from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from searxng import SearxNGClient

from datetime import datetime

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
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

react_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个助手，能够回答用户的问题。你有以下的 {tools} 工具可以使用，工具的名称是 {tool_names},
    Thought: 你的思考过程
    Action: 要使用的工具名称（必须是提供的工具之一）
    Action Input: 工具的输入
    Observation: 工具的结果
    ... (可以重复 Thought/Action/Action Input/Observation)
    Thought: 我现在知道最终答案
    Final Answer: 对原始输入的最终回答
    
- 如果用户的问题中包含了具体的年份（如“2025年”），请优先使用用户指定的年份进行搜索。
- 如果用户没有指定年份，但询问的是“最新”或“当前”的资讯，请直接搜索“最新进展”或“最新消息”等相关关键词，不要强行添加当前年份。
- 搜索后，请根据结果内容判断哪些信息是最近一两年内的，并明确告知用户这些信息的时间点。
- 如果搜索结果中没有明确的时间信息，可以结合当前日期，提示用户“以下是截至当前年份的最新公开信息”。

"""),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

agent = create_react_agent(
    llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True,
)

try: 
    result = agent_executor.invoke({"input": "2025年google大模型最新的进展是什么？"})
    print(result)
except Exception as e:
    print(f"Error: {e}")