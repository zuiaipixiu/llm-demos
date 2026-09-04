import os
import shutil

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
  from chromadb.telemetry.product.posthog import Posthog
  Posthog.capture = lambda self, *args, **kwargs: None
except Exception:
  pass

CHROMA_SETTINGS = Settings(anonymized_telemetry=False, is_persistent=True)

MODEL_DIR = "../bge-large-zh"
DB_DIR = "./chroma_db"
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
    return embeddings
  except Exception as e:
    print(f"初始化embeddings失败，错误信息：{e}")
    return None


def create_vector_store(documents, embeddings, db_dir=DB_DIR):
  """创建并保存向量数据。写入前清空旧库，避免重复入库。"""
  try:
    if os.path.exists(db_dir):
      shutil.rmtree(db_dir)
    vector_store = Chroma.from_documents(
      documents=documents,
      embedding=embeddings,
      persist_directory=db_dir,
      client_settings=CHROMA_SETTINGS
    )
    print(f"向量数据保存成功，路径：{db_dir}，共 {len(documents)} 条")
    return vector_store
  except Exception as e:
    print(f"创建向量数据失败，错误信息：{e}")
    return None


def query_vector_store(vector_store, query_text, k=5):
  """向量数据查询，按文本去重，避免相同片段被反复返回。"""
  try:
    fetch_k = max(k * 3, k)
    results = vector_store.similarity_search_with_score(query_text, k=fetch_k)
    unique = []
    seen = set()
    for doc, score in results:
      key = doc.page_content.strip()
      if key in seen:
        continue
      seen.add(key)
      unique.append((doc, score))
      if len(unique) >= k:
        break
    return unique
  except Exception as e:
    print(f"向量数据查询失败，错误信息：{e}")
    return None

def load_vector_store(db_dir=DB_DIR, embeddings=None):
  """加载向量数据。目录不存在或集合为空时返回 None。"""
  try:
    if not os.path.exists(db_dir):
      return None
    vector_store = Chroma(
      persist_directory=db_dir,
      embedding_function=embeddings,
      client_settings=CHROMA_SETTINGS
    )
    if vector_store._collection.count() == 0:
      return None
    return vector_store
  except Exception as e:
    print(f"加载向量数据失败，错误信息：{e}")
    return None
