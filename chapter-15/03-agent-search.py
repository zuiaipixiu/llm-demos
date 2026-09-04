from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import ChatPromptTemplate
from searxng import SearxNGClient


import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

chat_model = ChatOpenAI(
  model="deepseek-chat",
  openai_api_key=API_KEY,
  openai_api_base="https://api.deepseek.com/v1",
)

search_client = SearxNGClient()

search_tool = Tool(
  name="search",
  description="搜索网络信息，用于查询最新数据如天气、新闻等",
  func=search_client.search,
)

react_prompt = ChatPromptTemplate.from_messages([
  ("system", """你是一个有用的助手，能够回答问题。你有以下 {tools} 工具可以使用，工具的名称是 {tool_names}。
  
  你必须使用以下格式回应：
  
  Thought: 你的思考过程
  Action: 要使用的工具名称（必须是提供的工具之一）
  Action Input: 工具的输入
  Observation: 工具的结果
  ... (可以重复 Thought/Action/Action Input/Observation)
  Thought: 我现在知道最终答案
  Final Answer: 对原始输入的最终回答
  """),
  ("user", "{input}"),
  ("assistant", "{agent_scratchpad}"),
])

agent = create_react_agent(
  chat_model,
  tools=[search_tool],
  prompt=react_prompt,
)

agent_executor = AgentExecutor(
  agent=agent,
  tools=[search_tool],
  verbose=True,
  handle_parsing_errors=True
)

response = agent_executor.invoke({"input": "北京的天气怎么样？"})

print(response)
