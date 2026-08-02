
from urllib.parse import quote

import requests
import concurrent.futures
import logging

from bs4 import BeautifulSoup
import html2text

import brotli
import zlib
import chardet

#配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseSearchProvider:
    def __init__(self, provider_url):
        self.provider_url = provider_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Pragma': 'no-cache'
        }

    def search(self, query, max_results=10, content_limit=None):
        try:
            # 1. 校验入参，清空首尾空格
            if not query.strip():
                raise ValueError("Search query cannot be empty")

            # 2. 清洗query：如果换行分隔，只取第一行作为搜索词
            cleaned_query = query.split('\r\n')[1] if '\r\n' in query else query
            # URL拼接：用quote对关键词做URL编码，替换地址里的%s占位符
            url = self.provider_url.replace('%s', quote(cleaned_query))

            # 3. 发送GET网络请求，自定义请求头，超时30秒
            response = requests.get(url, headers=self.headers, timeout=30)
            # 主动抛出HTTP异常（4xx/5xx报错直接终止）
            response.raise_for_status()

            # 4. 响应预处理：解码、解压、乱码修复
            content = self._process_results(response)

            # 5. 调用子类方法解析网页，提取有效链接列表
            search_items = self.parse_valid_urls(content)

            # 6. 截断结果，控制返回条数不超过max_results
            if max_results and max_results > 0:
                search_items = search_items[:max_results]
        
            valid_items = [item for item in search_items if item.url.startswith('http')]

            # 并行获取每个结果的内容
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_item = {
                    executor.submit(self.fetch_content, item.url): item
                    for item in valid_items
                }

            results = []
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    content = future.result()
                    if content_limit and len(content) > content_limit:
                        content = content[:content_limit] + '...'
                    item.content = content
                    results.append(item)
                except Exception as e:
                    logger.error(f"Error fetching {item.url}: {e}")

            filtered_results = [r for r in results if r.content != "No content found"]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"query": query, "results": []}

        return {"query": query, "results": filtered_results}

        
    def _process_results(self, response):
        """处理响应，尝试多种方法解码内容"""
        content = response.text
        # 判断：内容开头不是HTML标签/文档声明，大概率编码错误
        if not content.strip().startswith('<') and not content.strip().startswith('<!DOCTYPE'):
            detected = chardet.detect(response.content)
            encoding = detected.get('encoding')
            # 探测置信度大于0.5才采用新编码
            if encoding and detected['confidence'] > 0.5:
                try:
                    content = response.content.decode(encoding)
                except Exception:
                    pass
            try:
                if response.headers.get('Content-Encoding') == 'gzip':
                    decompressed = zlib.decompress(response.content, 16+zlib.MAX_WBITS)
                    content = decompressed.decode('utf-8', errors='replace')
                elif response.headers.get('Content-Encoding') == 'br':
                    decompressed = brotli.decompress(response.content)
                    content = decompressed.decode('utf-8', errors='replace')
                elif response.headers.get('Content-Encoding') == 'deflate':
                    decompressed = zlib.decompress(response.content)
                    content = decompressed.decode('utf-8', errors='replace')
            except Exception:
                pass
            
        return content
    
    def parse_valid_urls(self, html_content):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def fetch_content(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            # 编码修复
            if response.encoding.lower() == 'iso-8859-1':
                detected = chardet.detect(response.content)
                encoding = detected['encoding']
                if encoding:
                    response.encoding = encoding

            soup = BeautifulSoup(response.text, 'html.parser')
            # 清理标签
            for tag in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                tag.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.unicode_snob = True
                
                content = str(main_content)
                markdown = h.handle(content)
                return markdown
            return "No content found"
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return "No content found"