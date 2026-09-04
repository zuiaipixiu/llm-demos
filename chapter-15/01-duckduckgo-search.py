from typing import Dict, List, Optional

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


class DDGSAPIWrapper(DuckDuckGoSearchAPIWrapper):
    """LangChain 仍依赖已停更的 duckduckgo_search。

    该包把 auto 后端写死为 Bing，且默认按「最近一年」过滤，容易返回空结果。
    这里改用官方继任包 ddgs，并显式指定较稳定的引擎。
    """

    time: Optional[str] = None
    backend: str = "yahoo,duckduckgo,google"

    def _ddgs_text(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Dict[str, str]]:
        last_error: Exception | None = None
        for backend in (self.backend, "auto"):
            try:
                with DDGS(timeout=20) as ddgs:
                    results = ddgs.text(
                        query,
                        region=self.region or "wt-wt",
                        safesearch=self.safesearch,
                        timelimit=self.time,
                        max_results=max_results or self.max_results,
                        backend=backend,
                    )
                if results:
                    return list(results)
            except DDGSException as exc:
                last_error = exc
        if last_error:
            raise last_error
        return []


search_tool = DuckDuckGoSearchResults(
    num_results=20,
    output_format="list",
    api_wrapper=DDGSAPIWrapper(),
)

search_query = "DeepSeek AI 最新进展"

print(f"搜索查询: {search_query}")
search_result = search_tool.invoke(search_query)

for i, result in enumerate(search_result):
    print(f"{i+1}. {result['title']}")
    print(f"   {result['snippet']}")
    print(f"   {result['link']}")
    print("-" * 100)
