# test_playwright.py

from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        # 以 headless 模式启动 Chromium
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 打开一个示例页面
        page.goto("http://www.djkpai.com/ai/178280.html", timeout=10000)  # 最多等待 10 秒
        
        # 获取并打印页面标题
        title = page.title()
        print("页面标题：", title)

        # 还可以额外打印页面 URL，以确认已经成功加载
        print("页面 URL：", page.url)

        browser.close()

if __name__ == "__main__":
    main()
    
    
  