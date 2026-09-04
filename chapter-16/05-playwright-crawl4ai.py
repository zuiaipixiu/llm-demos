# test_playwright_and_crawl4ai.py

import asyncio
from playwright.sync_api import sync_playwright
from crawl4ai import AsyncWebCrawler

def test_playwright():
    """
    使用 Playwright 同步 API 验证 Chromium 是否可用。
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://www.djkpai.com/ai/178280.html", timeout=10000)
        title = page.title()
        print("Playwright 测试 — 页面标题：", title)
        print("Playwright 测试 — 页面 URL：", page.url)
        browser.close()

async def test_crawl4ai_single(url: str):
    """
    使用 crawl4ai.AsyncWebCrawler 异步 API 对单个 URL 进行爬取。
    返回 Markdown 摘要文本。
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def test_crawl4ai(url: str):
    """
    同步地调用异步爬取函数 test_crawl4ai_single，
    并打印摘要前 500 字以验证是否成功爬取。
    """
    try:
        markdown = asyncio.run(test_crawl4ai_single(url))
        print(f"Crawl4AI 测试 — URL: {url}")
        print("Crawl4AI 测试 — 摘要前 500 字：")
        print(markdown[:2500] + "...")
    except Exception as e:
        # 如果播放或爬虫初始化失败，这里会捕获并打印原因
        if "Executable doesn't exist" in str(e):
            print("[爬取失败] Playwright 浏览器二进制未找到，请执行：python -m playwright install chromium")
        else:
            print(f"[爬取失败] 无法使用 Crawl4AI 获取 {url} 的内容，错误原因：{e}")

if __name__ == "__main__":
    # 1. 测试 Playwright
    print("=== Playwright 功能测试 ===")
    test_playwright()
    print("\n")

    # 2. 测试 AsyncWebCrawler 爬取
    print("=== Crawl4AI 功能测试 ===")
    test_url = "http://www.djkpai.com/ai/178280.html"
    test_crawl4ai(test_url)