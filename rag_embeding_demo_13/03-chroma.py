import chromadb
from chromadb.utils import embedding_functions

# 创建持久化客户端，指定存储目录
client = chromadb.PersistentClient(path="./chroma_db")
# 创建客户端
# client = chromadb.Client()

# 创建集合（可以指定嵌入函数）
# 这里使用默认的嵌入函数，实际应用中应选择合适的模型
embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_function
)

# 添加文档
documents = [
    "Python是一种易于学习的编程语言，广泛应用于数据分析和AI领域。",
    "向量数据库专门用于存储和检索向量数据，支持相似性搜索。",
    "大语言模型可以生成文本、回答问题，但存在知识截止日期的限制。",
    "RAG技术结合了检索和生成功能，提高了大语言模型回答的准确性和时效性。"
]

# 添加数据到集合
collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# 查询与"向量数据库"相似的文档
results = collection.query(
    query_texts=["向量数据库的应用"],
    n_results=2
)

print("查询结果：")
for i, doc_id in enumerate(results['ids'][0]):
    doc_text = results['documents'][0][i]
    print(f"文档ID: {doc_id}")
    print(f"文档内容: {doc_text}")
    print("-" * 50)