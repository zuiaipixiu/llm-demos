from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from searxng import SearxNGClient

from datetime import datetime

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

# ===== 1. 基础类和工具定义 =====
class SharedContext:
    """共享上下文管理器，用于Agent之间的状态共享"""
    def __init__(self):
        self.search_results = []
        self.analysis_results = []
        self.report_data = {}
        self.search_agent = None
        self.analysis_agent = None
        self.report_agent = None

    def add_search_results(self, results: List[Dict[str, Any]]):
        self.search_results.extend(results)

    def add_analysis_results(self, results: Dict[str, Any]):
        self.analysis_results.append(results)

    def set_report_data(self, data: Dict[str, Any]):
        self.report_data.update(data)

    def get_search_results(self) -> List[Dict[str, Any]]:
        return self.search_results

    def get_analysis_results(self) -> List[Dict[str, Any]]:
        return self.analysis_results

    def get_report_data(self) -> Dict[str, Any]:
        return self.report_data

# 创建全局共享上下文
shared_context = SharedContext()

# ===== 2. 业务工具函数 =====
def search_with_searx(query: str) -> str:
    """使用SearxNG进行搜索"""
    searx = SearxNGClient()
    results = searx.search(query)
    if results and 'results' in results:
        formatted_results = []
        for r in results['results']:
            formatted_results.append({
                'title': r.get('title', ''),
                'content': r.get('content', ''),
                'url': r.get('url', '')
            })
        shared_context.add_search_results(formatted_results)
        return "\n".join([f"- {r['title']}: {r['content']}" for r in formatted_results])
    return "未找到相关信息"

def analyze_search_results(query: str = "") -> str:
    """分析搜索结果"""
    results = shared_context.get_search_results()
    if not results:
        return "没有可分析的搜索结果"
    
    # 改进的分析逻辑
    analysis = {
        'total_results': len(results),
        'topics': {},
        'key_findings': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 统计主题词和关键发现
    for result in results:
        # 提取标题和内容中的关键词
        text = f"{result['title']} {result['content']}".lower()
        words = text.split()
        
        # 统计词频
        for word in words:
            if len(word) > 3:  # 忽略短词
                analysis['topics'][word] = analysis['topics'].get(word, 0) + 1
        
        # 提取关键发现
        if result['title'] and result['content']:
            analysis['key_findings'].append({
                'title': result['title'],
                'summary': result['content'][:200] + '...' if len(result['content']) > 200 else result['content']
            })
    
    # 获取前10个最常见主题词
    top_topics = sorted(analysis['topics'].items(), key=lambda x: x[1], reverse=True)[:10]
    analysis['top_topics'] = dict(top_topics)
    
    shared_context.add_analysis_results(analysis)
    return f"分析完成：共找到 {analysis['total_results']} 条结果，包含 {len(analysis['topics'])} 个主题词，提取了 {len(analysis['key_findings'])} 个关键发现"

def generate_report(query: str = "") -> str:
    """生成分析报告"""
    analysis_results = shared_context.get_analysis_results()
    if not analysis_results:
        return "没有可用的分析结果"
    
    latest_analysis = analysis_results[-1]
    
    # 生成更详细的报告
    report = {
        'title': '搜索分析报告',
        'timestamp': latest_analysis['timestamp'],
        'summary': f"基于 {latest_analysis['total_results']} 条搜索结果的分析",
        'top_topics': latest_analysis['top_topics'],
        'key_findings': latest_analysis['key_findings'][:5],  # 只显示前5个关键发现
        'recommendations': [
            "建议关注高频主题词相关的领域",
            "深入分析关键发现中的创新点",
            "跟踪最新发展趋势"
        ]
    }
    
    shared_context.set_report_data(report)
    
    # 格式化报告输出
    report_text = f"""
报告标题：{report['title']}
生成时间：{report['timestamp']}
{report['summary']}

主要发现：
{chr(10).join([f"- {finding['title']}: {finding['summary']}" for finding in report['key_findings']])}

热门主题：
{chr(10).join([f"- {topic}: {count}次" for topic, count in report['top_topics'].items()])}

建议：
{chr(10).join([f"- {rec}" for rec in report['recommendations']])}
"""
    return report_text

# ===== 3. 环境配置 =====
load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)

# ===== 4. 工具定义 =====
search_tools = [
    Tool(
        name="search_with_searx",
        description="使用搜索引擎查找信息，输入应该是一个搜索查询字符串",
        func=search_with_searx
    )
]

analysis_tools = [
    Tool(
        name="analyze_search_results",
        description="分析搜索结果，生成统计信息",
        func=analyze_search_results
    )
]

report_tools = [
    Tool(
        name="generate_report",
        description="生成分析报告",
        func=generate_report
    )
]

# ===== 5. Agent提示模板 =====
search_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的搜索专家，负责查找和分析信息。你有以下 {tools} 工具可以使用，工具的名称是 {tool_names}。
    
    请使用以下格式回应：
    Thought: 你的思考过程
    Action: 工具名称
    Action Input: 工具输入
    Observation: 工具结果
    ... (可以重复 Thought/Action/Action Input/Observation)
    Thought: 我现在知道最终答案
    Final Answer: 对用户的最终回答
    """),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

analysis_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个数据分析专家，负责分析搜索结果并提取关键信息。你有以下 {tools} 工具可以使用，工具的名称是 {tool_names}。
    
    请使用以下格式回应：
    Thought: 你的思考过程
    Action: 工具名称
    Action Input: 工具输入
    Observation: 工具结果
    ... (可以重复 Thought/Action/Action Input/Observation)
    Thought: 我现在知道最终答案
    Final Answer: 对用户的最终回答
    """),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

report_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个报告生成专家，负责生成清晰的分析报告。你有以下 {tools} 工具可以使用，工具的名称是 {tool_names}。
    
    请使用以下格式回应：
    Thought: 你的思考过程
    Action: 工具名称
    Action Input: 工具输入
    Observation: 工具结果
    ... (可以重复 Thought/Action/Action Input/Observation)
    Thought: 我现在知道最终答案
    Final Answer: 对用户的最终回答
    """),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

# ===== 6. 创建Agents =====
search_agent = create_react_agent(
    llm,
    tools=search_tools,
    prompt=search_agent_prompt
)

analysis_agent = create_react_agent(
    llm,
    tools=analysis_tools,
    prompt=analysis_agent_prompt
)

report_agent = create_react_agent(
    llm,
    tools=report_tools,
    prompt=report_agent_prompt
)

# ===== 7. 创建Agent执行器 =====
search_executor = AgentExecutor(
    agent=search_agent,
    tools=search_tools,
    verbose=True,
    handle_parsing_errors=True
)

analysis_executor = AgentExecutor(
    agent=analysis_agent,
    tools=analysis_tools,
    verbose=True,
    handle_parsing_errors=True
)

report_executor = AgentExecutor(
    agent=report_agent,
    tools=report_tools,
    verbose=True,
    handle_parsing_errors=True
)

# 将执行器添加到共享上下文
shared_context.search_agent = search_executor
shared_context.analysis_agent = analysis_executor
shared_context.report_agent = report_executor

# ===== 8. 运行测试 =====
def run_agents():
    """
    演示多Agent协作流程：
    1. 搜索专家查找信息
    2. 数据分析师分析结果
    3. 报告生成器创建报告
    """
    # 示例1：搜索信息
    print("=" * 50)
    print("示例1：搜索信息")
    search_response = search_executor.invoke({"input": "获取2025年，人工智能在医疗领域的最新应用"})
    print("搜索专家回答：", search_response["output"])

    # 示例2：分析结果 - 使用搜索结果的输出作为输入
    print("=" * 50)
    print("示例2：分析结果")
    analysis_prompt = f"""基于以下搜索结果进行分析：
{search_response["output"]}

请提取关键信息和主题词，并生成分析报告。"""
    analysis_response = analysis_executor.invoke({"input": analysis_prompt})
    print("数据分析师回答：", analysis_response["output"])

    # 示例3：生成报告 - 使用分析结果的输出作为输入
    print("=" * 50)
    print("示例3：生成报告")
    report_prompt = f"""基于以下分析结果生成详细报告：
{analysis_response["output"]}

请生成一份包含主要发现、热门主题和建议的完整报告。"""
    report_response = report_executor.invoke({"input": report_prompt})
    print("报告生成器回答：", report_response["output"])

    # 返回完整的处理链结果
    return {
        "search_results": search_response["output"],
        "analysis_results": analysis_response["output"],
        "report": report_response["output"]
    }

if __name__ == "__main__":
    run_agents()
