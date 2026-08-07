import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 创建持久化客户端，指定存储目录（与 05-chroma-read.py 保持一致）
client = chromadb.PersistentClient(path="./chroma_db_bge")

# 使用 BGE 中文模型作为嵌入函数
model_path = "./bge-large-zh"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_path)

# 创建集合（使用 BGE 嵌入函数）
collection = client.get_or_create_collection(
    name="documents_bge",
    embedding_function=embedding_function
)

# 添加文档（与 03-chroma.py 内容一致，便于对比）
documents = [
    "Python是一种易于学习的编程语言，广泛应用于数据分析和AI领域。",
    "向量数据库专门用于存储和检索向量数据，支持相似性搜索。",
    "大语言模型可以生成文本、回答问题，但存在知识截止日期的限制。",
    "RAG技术结合了检索和生成功能，提高了大语言模型回答的准确性和时效性。"
]

# 如果集合已有相同 id 的文档，先删除避免重复插入报错
existing = collection.get()
if existing['ids']:
    collection.delete(ids=existing['ids'])
    print(f"已清除旧数据：{len(existing['ids'])} 条")

# 添加数据到集合
collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

print(f"成功写入 {len(documents)} 条文档到 documents_bge 集合")
print(f"集合当前文档数: {collection.count()}")
