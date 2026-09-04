from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from .rag.document_loader import load_documents, split_documents

MODEL_DIR = "./bge-large-zh"
DB_DIR="./chroma_db"
DOCS_DIR = "./documents"

def init_embeddings(model_dir=MODEL_DIR):
  """初始化embeddings"""
  if not os.path.exists(model_dir):
    print(f"模型文件夹不存在，请检查路径：{model_dir}")
    return None
  
  try: 
    embeddings = HuggingFaceEmbeddings(
      model_name=model_dir,
      model_kwargs={"device": "cpu"},
      encode_kwargs={"normalize_embeddings": True}
    )
    
    # _ = embeddings.embed_query("你好")
    # print("成功初始化嵌入模型")
    return embeddings
  except Exception as e:
    print(f"初始化embeddings失败，错误信息：{e}")
    return None
  
  
def create_vector_store(documents, embeddings, db_dir=DB_DIR):
  """创建并保存向量数据"""
  try:
    vector_store = Chroma.from_documents(
      documents=documents,
      embedding=embeddings,
      persist_directory=db_dir
    )
    print(f"向量数据保存成功，路径：{db_dir}")
    return vector_store
  except Exception as e:
    print(f"创建向量数据失败，错误信息：{e}")
    return None
  
  
def query_vector_store(vector_store, query_text, k=5):
  """向量数据查询"""
  try:
    results = vector_store.similarity_search_with_score(query_text, k=k)
    return results
  except Exception as e:
    print(f"向量数据查询失败，错误信息：{e}")
    return None
  
def load_vector_store(db_dir=DB_DIR):
  """加载向量数据"""
  try:
    vector_store = Chroma(
      persist_directory=db_dir,
      embedding_function=embeddings
    )
    return vector_store
  except Exception as e:
    print(f"加载向量数据失败，错误信息：{e}")
    return None
    
    
if __name__ == "__main__":
  embeddings = init_embeddings()
  if embeddings:
    print("embeddings初始化成功")
    docs = load_documents(DOCS_DIR)
    if not docs:
      print("加载文档失败")
      exit()
    
    chunks = split_documents(docs)
    if not chunks:
      print("分割文档失败")
      exit()
      
    print("向量数据创建成功")
    # vector_store = create_vector_store(chunks, embeddings, DB_DIR)
    # print("向量数据创建成功")
    vector_store = load_vector_store(DB_DIR)
    
    result = query_vector_store(vector_store, "带记忆的研究助手提示词")
    print(result)
      
  else:
    print("embeddings初始化失败")
    
    