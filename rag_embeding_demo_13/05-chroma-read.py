import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 创建持久化客户端，指定存储目录
client = chromadb.PersistentClient(path="./chroma_db_bge")

# 加载BGE模型以便查看单个文本的嵌入
model_path = "./bge-large-zh"
model = SentenceTransformer(model_path)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_path)

collection = client.get_or_create_collection(
    name="documents_bge",
    embedding_function=embedding_function
)

# 方法一：通过查询获取向量嵌入
print("方法一：从查询结果中获取向量嵌入")
results = collection.query(
    query_texts=["向量数据库的应用"],
    n_results=2,
    include=["embeddings", "documents", "metadatas"]
)

print("查询结果：")
for i, doc_id in enumerate(results['ids'][0]):
    doc_text = results['documents'][0][i]
    # 获取嵌入向量
    embedding = results['embeddings'][0][i]
    
    print(f"文档ID: {doc_id}")
    print(f"文档内容: {doc_text}")
    print(f"向量维度: {len(embedding)}")
    print(f"向量前5个值: {embedding[:5]}")  # 只打印前5个值，完整向量太长
    print("-" * 50)

# 方法二：直接获取单个文本的向量嵌入
print("\n方法二：直接获取文本的向量嵌入")
test_text = "向量数据库的应用"
embedding = model.encode(test_text)
print(f"文本: {test_text}")
print(f"向量维度: {len(embedding)}")
print(f"向量前5个值: {embedding[:5]}")
print("-" * 50)

# 方法三：获取集合中所有文档及其向量
print("\n方法三：获取集合中所有文档的向量")
all_items = collection.get(include=["embeddings", "documents"])
print(f"集合中总文档数: {len(all_items['ids'])}")

for i, doc_id in enumerate(all_items['ids']):
    doc_text = all_items['documents'][i]
    embedding = all_items['embeddings'][i]
    print(f"文档ID: {doc_id}")
    print(f"文档内容: {doc_text}")
    print(f"向量维度: {len(embedding)}")
    print(f"向量前5个值: {embedding[:5]}")
    print("-" * 50)