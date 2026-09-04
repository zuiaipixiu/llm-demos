import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import httpx
from langchain_openai import ChatOpenAI
from vector_embedding import init_embeddings, load_vector_store, create_vector_store, query_vector_store
from document_loader import load_documents, split_documents

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE", "https://techhub.ssss818.com/v1")
MODEL_NAME = os.getenv("MODEL", "gpt-5.5")

DOCS_DIR = "./documents"

class SimpleRAG:
  def __init__(self, force=False):
    self.embeddings = init_embeddings()
    self.vector_store = None if force else load_vector_store(embeddings=self.embeddings)
    if not self.vector_store:
      docs = load_documents(DOCS_DIR)
      if docs:
        chunks = split_documents(docs)
        self.vector_store = create_vector_store(chunks, self.embeddings)
    self.llm = self.init_llm()

  def init_llm(self):
    # 本地 HTTP 代理会打断该网关的 TLS 握手，这里不读取环境代理。
    http_client = httpx.Client(trust_env=False, timeout=60.0)
    return ChatOpenAI(
      model=MODEL_NAME,
      openai_api_key=API_KEY,
      openai_api_base=API_BASE,
      http_client=http_client,
      timeout=60,
      max_retries=2,
    )

  def answer(self, question):
    result = query_vector_store(self.vector_store, question)
    if not result:
      return "没有找到相关内容"

    print(f"检索到 {len(result)} 条相关片段：")
    for i, (doc, score) in enumerate(result, 1):
      source = os.path.basename(doc.metadata.get("source", "unknown"))
      preview = doc.page_content.replace("\n", " ").strip()[:80]
      print(f"  [{i}] {source} score={score:.4f} {preview}")

    context = "\n\n".join(doc.page_content for doc, _ in result)
    prompt = f"根据以下内容回答问题：\n{context}\n问题：{question}\n答案："
    try:
      response = self.llm.invoke(prompt)
      return response.content
    except Exception as e:
      return f"检索成功，但调用模型失败：{e}"


if __name__ == "__main__":
  rag = SimpleRAG(force=False)
  print(rag.answer("什么是向量检索？"))
