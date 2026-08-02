
import sys

from provider.baidu import BaiduSearchProvider


def main():
    if len(sys.argv) < 2:
        print("Usage: python search.py <query> <engine>")
        print("Available engines: baidu, google")
        return
    query = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else "baidu"
    
    if engine == "baidu":
        provider = BaiduSearchProvider()
    elif engine == "google":
        provider = GoogleSearchProvider()
    else:
        print(f"Unknown search engine: {engine}")
        return
    
    results = provider.search(query)
    
    for i, result in enumerate(results['results']):
        print(f"\n—— Result {i+1} ——")
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        if result.content:
            # 仅显示前200个字符，并确保它们是可打印的
            preview = result.content[:200].replace('\n', ' ').strip()
            print(f"Content (preview): {preview}...") 

if __name__ == "__main__":
    main()