from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from provider.base import BaseSearchProvider

class BaiduSearchProvider(BaseSearchProvider):
    def __init__(self):
        provider_url = "https://www.baidu.com/s?wd=%s"
        super().__init__(provider_url)

    def parse_valid_urls(self, html_content):
        """解析百度搜索结果页面，提取有效的URL"""
        soup = BeautifulSoup(html_content, "html.parser")
        results = []
        seen_urls = set()

        def resolve_baidu_url(raw_url):
            """从百度跳转链接中提取真实URL；无法提取时返回原始URL。"""
            if not raw_url:
                return ""

            raw_url = raw_url.strip()
            parsed = urlparse(raw_url)

            # 百度结果页中部分链接会把真实地址放在 url / wd 参数里。
            if "baidu.com" in parsed.netloc:
                params = parse_qs(parsed.query)
                for key in ("url", "wd"):
                    values = params.get(key)
                    if values and values[0].startswith(("http://", "https://")):
                        return unquote(values[0])

            return raw_url

        # 百度搜索结果通常位于 .result / .c-container 容器中。
        containers = soup.select("div.result, div.c-container")
        if not containers:
            containers = soup.select("h3 a")

        for container in containers:
            link = container if container.name == "a" else container.select_one("h3 a[href], a[href]")
            if not link:
                continue

            url = resolve_baidu_url(link.get("href", ""))
            title = link.get_text(" ", strip=True)

            if not url.startswith(("http://", "https://")):
                continue
            if not title:
                title = url
            if url in seen_urls:
                continue

            seen_urls.add(url)
            results.append(SimpleNamespace(title=title, url=url, content=""))

        return results