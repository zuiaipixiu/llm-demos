#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import ChatPromptTemplate
from searxng import SearxNGClient

# ========== 新增依赖：爬虫库 & LangChain Embedding/Chroma/RetrievalQA ==========
from crawl4ai import AsyncWebCrawler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA

# ========== 引入我们刚刚创建的 db_utils ==========
from db_utils import create_new_chroma_db, build_vectorstore, persist_report_info

# ===== 1. 基础类和工具定义 =====
class SharedContext:
    """共享上下文管理器，用于 Agent 之间的状态共享"""
    def __init__(self):
        self.search_results: List[Dict[str, Any]] = []
        self.analysis_results: List[Dict[str, Any]] = []
        self.crawled_results: List[Dict[str, Any]] = []
        self.report_data: Dict[str, Any] = {}
        self.search_agent = None
        self.analysis_agent = None
        self.report_agent = None

    def add_search_results(self, results: List[Dict[str, Any]]):
        self.search_results.extend(results)

    def add_analysis_results(self, results: Dict[str, Any]):
        self.analysis_results.append(results)

    def add_crawled_results(self, item: Dict[str, Any]):
        """
        item 结构：{ "ref": int, "url": str, "summary": str }
        """
        self.crawled_results.append(item)

    def set_report_data(self, data: Dict[str, Any]):
        self.report_data.update(data)

    def get_search_results(self) -> List[Dict[str, Any]]:
        return self.search_results

    def get_analysis_results(self) -> List[Dict[str, Any]]:
        return self.analysis_results

    def get_crawled_results(self) -> List[Dict[str, Any]]:
        return self.crawled_results

    def get_report_data(self) -> Dict[str, Any]:
        return self.report_data

# 创建全局共享上下文
shared_context = SharedContext()

# ===== 2. 业务工具函数 =====

def search_with_searx(query: str) -> str:
    """使用 SearxNG 进行搜索，返回带引用序号的结果"""
    searx = SearxNGClient()
    results = searx.search(query)
    if results and 'results' in results:
        formatted_results = []
        for idx, r in enumerate(results['results'], 1):
            formatted_results.append({
                'title': r.get('title', ''),
                'content': r.get('content', ''),
                'url': r.get('url', ''),
                'ref': idx
            })
        shared_context.add_search_results(formatted_results)
        return "\n".join([
            f"[{r['ref']}] {r['title']}: {r['content']} (链接: {r['url']})"
            for r in formatted_results
        ])
    return "未找到相关信息"

def analyze_search_results(query: str = "") -> str:
    """分析搜索结果，提取主题词和关键发现"""
    results = shared_context.get_search_results()
    if not results:
        return "没有可分析的搜索结果"
    
    analysis = {
        'total_results': len(results),
        'topics': {},
        'key_findings': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    for result in results:
        text = f"{result['title']} {result['content']}".lower()
        words = text.split()
        for word in words:
            if len(word) > 3:
                analysis['topics'][word] = analysis['topics'].get(word, 0) + 1
        
        if result['title'] and result['content']:
            summary = result['content'][:200] + '...' if len(result['content']) > 200 else result['content']
            analysis['key_findings'].append({
                'title': result['title'],
                'summary': summary
            })
    
    top_topics = sorted(analysis['topics'].items(), key=lambda x: x[1], reverse=True)[:10]
    analysis['top_topics'] = dict(top_topics)
    
    shared_context.add_analysis_results(analysis)
    return f"分析完成：共找到 {analysis['total_results']} 条结果，包含 {len(analysis['topics'])} 个主题词，提取了 {len(analysis['key_findings'])} 个关键发现"

# ===== 2.5. 修订后的 real_crawl4ai 函数 =====
def real_crawl4ai(url: str) -> str:
    """
    使用 crawl4ai.AsyncWebCrawler 真实爬取指定 URL，然后返回 Markdown 格式的摘要字符串。
    如果出错，会给出具体的异常提示。
    """
    async def _crawl():
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url)

    try:
        result = asyncio.run(_crawl())
        if hasattr(result, "markdown"):
            return result.markdown
        else:
            return f"[爬取失败] 返回结果中不包含 markdown，返回对象: {type(result)}"
    except Exception as e:
        err_msg = str(e)
        if "Executable doesn't exist" in err_msg or "playwright" in err_msg.lower():
            return "[爬取失败] Playwright 浏览器二进制未找到，请执行：\n    python -m playwright install chromium"
        return f"[爬取失败] 无法获取 {url} 的内容，错误原因：{e}"

# 包装成 Tool 对象
crawl4ai_tool = Tool(
    name="crawl4ai",
    description="真实爬取指定链接内容并返回 Markdown 格式的摘要，输入为 URL 字符串",
    func=real_crawl4ai
)

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
        description="使用搜索引擎查找信息，输入应为搜索查询字符串",
        func=search_with_searx
    ),
    crawl4ai_tool  # 真实爬取工具
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
        description="生成分析报告（基于分析结果）",
        func=lambda _: "请改用 generate_report_with_retrieval 函数。"
    )
]

# ===== 5. Agent 提示模板 =====
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

- 如果用户的问题中包含了具体的年份（如“2025年”），请优先使用用户指定的年份进行搜索。
- 如果用户没有指定年份，但询问的是“最新”或“当前”的资讯，请直接搜索“最新进展”或“最新消息”等相关关键词，不要强行添加当前年份。
- 搜索后，请根据结果内容判断哪些信息是最近一两年内的，并明确告知用户这些信息的时间点。
- 如果搜索结果中没有明确的时间信息，可以结合当前日期，提示用户“以下是截至当前年份的最新公开信息”。

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

# ===== 6. 创建 Agents =====
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

# ===== 7. 创建 Agent 执行器 =====
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

# ===== 8. 检索式问答生成报告 =====
def generate_report_with_retrieval(
    llm: ChatOpenAI,
    query: str = "请生成一份多轮搜索与爬取分析报告，包括主要发现与建议。"
) -> (str, Dict[str, Any]):
    """
    改造版：
    1. 在创建完 vectordb 后，先从 shared_context 中取出第一轮分析的关键词
    2. 针对每个关键词分别向 vectordb 发起检索，生成“小报告”
    3. 将所有“小报告”整合，使用 LLM 生成最终报告
    4. 原有的 persist_report_info 逻辑保持不变
    """

    # ————— 1. 创建新的 Chroma 数据库目录，并记录 task_id 与 created_at ————— 
    db_path = create_new_chroma_db(base_dir="chroma_db")
    task_id = os.path.basename(db_path)
    created_at = datetime.now()

    # ————— 2. 构建 Document 列表（同原逻辑） ————— 
    docs_to_add: List[Document] = []
    # 2.1 从搜索结果构造 Document
    for r in shared_context.get_search_results():
        text = r.get("content", "")
        if not text and r.get("title"):
            text = r["title"]
        if text:
            docs_to_add.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": r.get("url", "未知来源"),
                        "type": "search",
                        "ref": r.get("ref")
                    }
                )
            )
    # 2.2 从爬取摘要构造 Document
    for c in shared_context.get_crawled_results():
        text = c.get("summary", "")
        if text:
            docs_to_add.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": c.get("url", "未知来源"),
                        "type": "crawl",
                        "ref": c.get("ref")
                    }
                )
            )

    # ————— 3. 构建 Chroma 向量数据库 ————— 
    vectordb = build_vectorstore(
        docs=docs_to_add,
        db_path=db_path,
        embedding_model_path="./bge-large-zh"
    )

    # ————— 4. 多轮检索 & 多轮汇总 ————— 

    # 4.1 从 shared_context 获取最新一轮分析结果中的 top_topics
    if not shared_context.get_analysis_results():
        # 如果没有进行过第一轮分析，直接用 query 做一次全局检索
        selected_keywords = [query]
    else:
        latest_analysis = shared_context.get_analysis_results()[-1]
        top_topics_dict = latest_analysis.get("top_topics", {})
        # 若 top_topics 太少，则至少用 query 做兜底
        if top_topics_dict:
            selected_keywords = list(top_topics_dict.keys())[:5]
        else:
            selected_keywords = [query]

    # 4.2 针对每个关键词进行向量检索
    keyword_to_docs = {}
    for kw in selected_keywords:
        # 这里我们用 similarity_search，取 top 5 条
        docs_for_kw = vectordb.similarity_search(query=kw, k=5)
        keyword_to_docs[kw] = docs_for_kw

    # 4.3 为每个关键词生成“小报告”
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    small_report_template = PromptTemplate(
        input_variables=["keyword", "documents"],
        template=(
            "你是一个信息整合专家，下面给出与主题“{keyword}”高度相关的文档内容：\n\n"
            "{documents}\n\n"
            "请你基于这些文档，提炼出与“{keyword}”最密切的关键观点，"
            "生成一段约 150~250 字的**小报告**，"
            "用中文输出，包含核心结论与简要解释。"
        )
    )
    small_report_chain = LLMChain(llm=llm, prompt=small_report_template)

    keyword_small_reports = {}
    for kw, docs in keyword_to_docs.items():
        if not docs:
            keyword_small_reports[kw] = f"（关键词“{kw}”未检索到相关文档。）"
            continue

        # 合并文档内容，防止过长，可截断
        docs_text = "\n\n".join(
            [
                f"来源：{d.metadata.get('source', '未知')}\n内容：{d.page_content[:1000]}..."
                for d in docs
            ]
        )
        small_report = small_report_chain.run({"keyword": kw, "documents": docs_text})
        keyword_small_reports[kw] = small_report.strip()

    # 4.4 整合所有“小报告”，让 LLM 生成最终综合报告
    all_small_reports_text = ""
    for kw, rpt in keyword_small_reports.items():
        all_small_reports_text += f"=== 关键词：{kw} ===\n{rpt}\n\n"

    final_report_template = PromptTemplate(
        input_variables=["small_reports"],
        template=("""下面是针对各个关键词的“小报告”汇总，请你基于这些小报告，完成最终的**综合报告**：
{small_reports}
要求：
1. 对上述内容做整体概括，点明各关键词之间的内在联系；
2. 提炼出至少三点核心发现；
3. 给出具体可行的建议（例如，后续研究方向、应用场景、潜在风险等）；
4. 报告控制在 600~800 字左右，用中文输出。"""
        )
    )
    final_report_chain = LLMChain(llm=llm, prompt=final_report_template)
    report_text = final_report_chain.run({"small_reports": all_small_reports_text})

    # ————— 5. 示例占位：构造 token_usage ————— 
    token_usage = {
        "prompt_tokens": -1,
        "completion_tokens": -1,
        "total_tokens": -1
    }
    # （如果你的 LLM 支持真实的 usage 信息，这里可以替换成真实调用）

    # ————— 6. 记录完成时间并持久化 JSON ————— 
    completed_at = datetime.now()
    json_path = persist_report_info(
        task_id=task_id,
        db_path=db_path,
        report_text=report_text,
        token_usage=token_usage,
        created_at=created_at,
        completed_at=completed_at,
        base_dir="reports"
    )
    print(f"[Persist] 已将报告信息保存到：{json_path}")

    return report_text, token_usage

# ===== 9. 运行测试 & 主流程 =====
def run_agents():
    """
    多轮协作流程：
    1. 第1轮搜索
    2. 第1轮分析
    3. 对所有链接真实爬取摘要
    4. 对爬取摘要简单统计分析
    5. 基于向量检索 & 本地嵌入生成最终报告
    """
    created_at = datetime.now()
    db_path = create_new_chroma_db(base_dir="chroma_db")
    task_id = os.path.basename(db_path)
    print(f"任务 {task_id} 开始，Chroma 数据库目录：{db_path}")
    print("=" * 50)

    # 1. 第1轮搜索
    print("示例1：第1轮搜索")
    search_response = shared_context.search_agent.invoke({"input": "2025年人工智能在医疗领域的最新应用"})
    print("搜索专家回答：", search_response["output"])
    print("=" * 50)

    # 2. 第1轮分析
    print("示例2：第1轮分析")
    analysis_prompt = (
        f"基于以下搜索结果进行分析：\n{search_response['output']}\n"
        "请提取关键信息和主题词。"
    )
    analysis_response = shared_context.analysis_agent.invoke({"input": analysis_prompt})
    print("数据分析师回答：", analysis_response["output"])
    print("=" * 50)

    # 3. 对所有链接真实爬取摘要
    print("示例3：对所有链接真实爬取摘要")
    for r in shared_context.get_search_results():
        url = r["url"]
        ref = r["ref"]
        summary = real_crawl4ai(url)
        shared_context.add_crawled_results({
            "ref": ref,
            "url": url,
            "summary": summary
        })
        print(f"[{ref}] 爬取完成：{url}")
    print("=" * 50)

    # 4. 对爬取摘要简单统计分析
    print("示例4：对爬取摘要简单统计分析")
    total_crawled = len(shared_context.get_crawled_results())
    if total_crawled > 0:
        print(f"共爬取 {total_crawled} 个链接，示例摘要（第1条）：\n"
              f"{shared_context.get_crawled_results()[0]['summary'][:200]}...")
    else:
        print("未爬取到任何摘要")
    print("=" * 50)

    # 5. 生成最终报告（基于向量检索 & 本地嵌入）
    print("示例5：生成最终报告（基于向量检索 & 本地嵌入）")
    report_text, token_usage = generate_report_with_retrieval(
        llm=llm,
        query="请基于之前所有的搜索结果与爬取摘要，写一份多轮搜索与爬取分析报告，包括主要发现与建议。"
    )
    print("最终报告内容预览（前500 字）：")
    print(report_text[:500] + "...")
    completed_at = datetime.now()

    # 保存报告 JSON
    json_path = persist_report_info(
        task_id=task_id,
        db_path=db_path,
        report_text=report_text,
        token_usage=token_usage,
        created_at=created_at,
        completed_at=completed_at,
        base_dir="reports"
    )
    print(f"任务完成，报告已保存到：{json_path}")

    return report_text

if __name__ == "__main__":
    run_agents()