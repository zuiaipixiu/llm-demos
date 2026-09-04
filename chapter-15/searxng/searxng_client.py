import requests
import json

class SearxNGClient:
    """
    SearxNG搜索客户端工具类，用于向SearxNG搜索引擎发送请求并处理响应
    """
    
    def __init__(self, base_url="http://localhost:1234"):
        """
        初始化SearxNG客户端
        
        参数:
            base_url: SearxNG实例的基础URL，默认为本地实例
        """
        self.base_url = base_url.rstrip('/')
        self.search_endpoint = "/search"
    
    def search(self, query, **kwargs):
        """
        执行搜索请求
        
        参数:
            query: 搜索查询字符串
            **kwargs: 其他可选参数
        
        返回:
            dict: 包含搜索结果的字典
        """
        # 默认使用JSON格式返回结果
        params = {
            'q': query,
            'format': 'json'
        }
        
        # 添加其他可选参数
        params.update(kwargs)
        
        # 构建完整URL
        url = f"{self.base_url}{self.search_endpoint}"
        
        try:
            # 发送GET请求
            response = requests.get(url, params=params)
            response.raise_for_status()  # 如果响应状态码不是200，则抛出异常
            
            # 解析JSON响应
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    
    def get_results(self, query, limit=None):
        """
        获取搜索结果并可选择限制结果数量
        
        参数:
            query: 搜索查询字符串
            limit: 限制返回的结果数量
            
        返回:
            list: 搜索结果列表
        """
        response = self.search(query)
        
        if not response:
            return []
        
        results = response.get('results', [])
        
        if limit and isinstance(limit, int) and limit > 0:
            return results[:limit]
            
        return results
    
    def get_first_result(self, query):
        """
        获取第一个搜索结果
        
        参数:
            query: 搜索查询字符串
            
        返回:
            dict: 第一个搜索结果或None
        """
        results = self.get_results(query, limit=1)
        return results[0] if results else None

