from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DOCS_DIR = "./documents"

def load_documents(docs_dir=DOCS_DIR):
  """加载documents目录下的文档"""
  
  documents = []
  
  print(f"加载{docs_dir}目录下的文档...")
  for file in os.listdir(docs_dir):
    file_path = os.path.join(docs_dir, file)
    if not os.path.isfile(file_path):
      continue
    
    loader = None
    
    try:
      if file.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
      elif file.lower().endswith(".docx") or file.lower().endswith(".doc"):
        loader = Docx2txtLoader(file_path)
      elif file.lower().endswith(".txt"):
        loader = TextLoader(file_path)
      elif file.lower().endswith(".md") or file.lower().endswith(".markdown"):
        loader = TextLoader(file_path, encoding="utf-8")
    except Exception as e:
      print(f"加载{file}失败，错误信息：{e}")
      continue
    
    if loader:
      doc = loader.load()
      print(f"加载{file}成功，文档数量：{len(doc)}")
      documents.extend(doc)
      
  return documents

def split_documents(documents):
  """将文档分割成小块"""
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "##", "#", "。"],
    keep_separator=False
  )
  print("开始分割文档...")
  chunks = splitter.split_documents(documents)
  return chunks