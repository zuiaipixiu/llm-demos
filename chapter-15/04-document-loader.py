# 目标：完成对不同常见文档类型的解析，md, word, pdf, txt
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
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
      elif file.lower().endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    except Exception as e:
      print(f"加载{file}失败，错误信息：{e}")
      continue
    
    if loader:
      doc = loader.load()
      print(f"加载{file}成功，文档数量：{len(doc)}")
      documents.extend(doc)
      
  return documents

if __name__ == "__main__":
  documents = load_documents()
  print(f"加载完成，文档数量：{len(documents)}")
  for i, doc in enumerate(documents):
    print(doc.page_content)
    print("-"*100)
    
  