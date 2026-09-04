from searxng import SearxNGClient

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    searx = SearxNGClient()
    
    # 执行搜索
    query = "北京的天气"
    results = searx.search(query)
    
    # 打印结果
    if results:
        print(f"查询: {results['query']}")
        print(f"结果数: {results['number_of_results']}")
        
        # 打印前3个结果
        for i, result in enumerate(results['results'][:3], 1):
            print(f"\n结果 {i}:")
            print(f"标题: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"内容: {result['content']}")

