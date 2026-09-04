# 03-multiple-agents.py

import datetime
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import ChatPromptTemplate

from utils import task_timer, parse_relative_time


# —— 1. 定义共享上下文管理器 ——
class SharedContext:
    def __init__(self):
        self.current_time = None
        self.tasks = []
        self.time_agent = None
        self.task_agent = None

    def update_time(self, time_str: str):
        self.current_time = time_str

    def add_task(self, task: str):
        self.tasks.append(task)

    def get_current_time(self) -> str:
        return self.current_time

    def get_tasks(self) -> list:
        return self.tasks


# 创建全局共享上下文
shared_context = SharedContext()


# —— 2. 定义业务函数 —— 
def get_current_time(query: str = "") -> str:
    """获取当前时间的函数"""
    current_time = datetime.datetime.now()
    time_str = f"当前时间是：{current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    shared_context.update_time(time_str)
    return time_str


def calculate_time_difference(start_time: str, end_time: str) -> str:
    """计算两个时间点之间的时间差"""
    try:
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        diff = end - start
        return f"时间差为：{diff}"
    except Exception as e:
        return f"时间格式错误：{str(e)}"


def create_task_reminder(task: str, time_str: str = "1分钟后") -> str:
    """创建任务提醒
    
    Args:
        task: 任务内容
        time_str: 时间字符串，可以是相对时间（如'2分钟后'）或具体时间（如'2024-03-20 15:00:00'）
    
    Returns:
        str: 创建结果信息
    """
    try:
        # 清理输入
        task = task.strip()
        time_str = time_str.strip()

        # 如果任务为空，使用默认任务名
        if not task:
            task = "未命名任务"

        # 解析时间字符串（支持 "X分钟后"/"X小时后"/"X天后" 或 绝对时间 "%Y-%m-%d %H:%M:%S"）
        target_time = parse_relative_time(time_str)

        # 创建定时器
        timer_id = task_timer.create_timer(task, target_time)

        # 添加到共享上下文
        shared_context.add_task(f"{task} - {target_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if timer_id >= 0:
            return (
                f"已创建任务提醒：{task}\n"
                f"提醒时间：{target_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"定时器ID：{timer_id}"
            )
        else:
            return "创建任务提醒失败：目标时间已过去"
    except Exception as e:
        return f"创建任务提醒失败：{str(e)}"


def get_shared_time(query: str = "") -> str:
    """获取共享上下文中的时间"""
    return shared_context.get_current_time() or "未设置时间"


def get_shared_tasks(query: str = "") -> str:
    """获取共享上下文中的任务列表"""
    tasks = shared_context.get_tasks()
    if not tasks:
        return "当前没有任务"
    return "\n".join([f"- {task}" for task in tasks])


# —— 3. 加载环境变量 —— 
load_dotenv()
API_KEY = os.getenv("API_KEY")


# —— 4. 初始化 ChatOpenAI 实例 —— 
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)


# —— 5. 定义工具 —— 
time_tools = [
    Tool(
        name="get_current_time",
        description="获取本地当前时间，格式为 YYYY-MM-DD HH:MM:SS",
        func=get_current_time,
    ),
    Tool(
        name="calculate_time_difference",
        description="计算两个时间点之间的时间差，输入格式为：'开始时间,结束时间'",
        func=calculate_time_difference,
    ),
    Tool(
        name="get_shared_tasks",
        description="获取当前所有任务列表，不需要输入参数",
        func=get_shared_tasks,
    ),
]

task_tools = [
    Tool(
        name="create_task_reminder",
        description="创建任务提醒，输入格式为：'任务内容,时间'，时间可以是相对时间（如'2分钟后'）或具体时间（如'2024-03-20 15:00:00'）",
        func=lambda x: create_task_reminder(*[part.strip() for part in x.split(",", 1)])
        if "," in x
        else create_task_reminder(x.strip(), "1分钟后"),
    ),
    Tool(
        name="get_shared_time",
        description="获取当前共享的时间信息，不需要输入参数",
        func=get_shared_time,
    ),
]


# —— 6. 定义 ReAct 提示模板（改进版） —— 
time_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个时间管理助手，负责处理所有与时间相关的问题。你有以下 {tools} 工具可以使用，工具的名称是 {tool_names}。

你可以：
1. 获取当前时间
2. 计算时间差
3. 查看当前任务列表

请严格按照下面的格式回应（注意每个标签后面都换行，不要合并在一行）：

Thought: 你的思考过程
Action: 工具的名称（必须是提供的工具之一）
Action Input: 工具的输入  （若无输入，留空白即可）
Observation: 工具的返回结果

Thought: 继续思考或进入下一个工具调用
Action: 下一个工具名
Action Input: ...
Observation: ...

Thought: 我现在知道最终答案
Final Answer: 对用户的最终回答

以下是一个示例（仅供格式参考）：
Thought: 用户想知道当前时间，所以我需要调用获取当前时间的工具。
Action: get_current_time
Action Input: 
Observation: 当前时间是：2025-06-05 10:50:00

Thought: 我已获得当前时间，可以直接回复给用户
Final Answer: 当前时间是2025年6月5日 10点50分。

注意：每个标签都必须单独占一行，不要将 Observation 与下一个 Thought 连在一起。
""",
        ),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ]
)

task_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个任务规划助手，负责创建和管理任务提醒。你有以下 {tools} 工具可以使用，工具的名称是 {tool_names}。

你可以：
1. 创建任务提醒
2. 获取当前时间信息

请严格按照下面的格式回应（注意每个标签后面都换行，不要合并在一行）：

Thought: 你的思考过程
Action: 工具的名称（必须是提供的工具之一）
Action Input: 工具的输入  （若无输入，留空白即可）
Observation: 工具的返回结果

Thought: 继续思考或进入下一个工具调用
Action: 下一个工具名
Action Input: ...
Observation: ...

Thought: 我现在知道最终答案
Final Answer: 对用户的最终回答

以下是一个示例（仅供格式参考）：
Thought: 我想创建一个2分钟后的提醒，需要先获取当前时间。
Action: get_current_time
Action Input: 
Observation: 当前时间是：2025-06-05 10:50:00

Thought: 我已知当前时间，2分钟后是10:52:00，接下来创建提醒。
Action: create_task_reminder
Action Input: 会议,2025-06-05 10:52:00
Observation: 已创建任务提醒：会议  提醒时间：2025-06-05 10:52:00  定时器ID：1

Thought: 任务创建成功
Final Answer: 已为您创建“会议”提醒，时间为2025-06-05 10:52:00

注意：每个标签都必须单独占一行，不要将 Observation 与下一个 Thought 连在一起。
""",
        ),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ]
)


# —— 7. 创建 ReAct Agents —— 
time_agent = create_react_agent(
    llm,
    tools=time_tools,
    prompt=time_agent_prompt,
)

task_agent = create_react_agent(
    llm,
    tools=task_tools,
    prompt=task_agent_prompt,
)


# —— 8. 创建 Agent 执行器 —— 
time_agent_executor = AgentExecutor(
    agent=time_agent,
    tools=time_tools,
    verbose=True,
    handle_parsing_errors=True,
)

task_agent_executor = AgentExecutor(
    agent=task_agent,
    tools=task_tools,
    verbose=True,
    handle_parsing_errors=True,
)

# 将执行器添加到共享上下文
shared_context.time_agent = time_agent_executor
shared_context.task_agent = task_agent_executor


def run_agents():
    """
    这个函数演示多 Agent 协作流程：
    1) 时间管理 Agent 处理时间相关查询
    2) 任务规划 Agent 处理任务创建
    3) 两个 Agent 可以协作完成复杂任务
    """
    # 示例1：时间管理
    print("=" * 50)
    print("示例1：时间管理")
    time_response = time_agent_executor.invoke({"input": "现在几点了？"})
    print("时间管理 Agent 回答：", time_response["output"])

    # 示例2：任务规划
    print("=" * 50)
    print("示例2：任务规划")
    task_response = task_agent_executor.invoke({"input": "创建一个2分钟之后的会议提醒"})
    print("任务规划 Agent 回答：", task_response["output"])

    # 示例3：查看任务列表
    print("=" * 50)
    print("示例3：查看任务列表")
    tasks_response = time_agent_executor.invoke({"input": "查看当前所有任务"})
    print("时间管理 Agent 回答：", tasks_response["output"])


if __name__ == "__main__":
    run_agents()