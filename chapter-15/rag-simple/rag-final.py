from langchain_openai import ChatOpenAI
from vector_embedding import init_embeddings, load_vector_store, create_vector_store ,query_vector_store
from document_loader import load_documents, split_documents

import os

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOCS_DIR = "./documents"
# It's good practice to define where the vector store is persisted,
# assuming create_vector_store and load_vector_store use a consistent location.
# This path might be implicitly handled within your vector_embedding functions.
# VECTOR_STORE_PERSIST_PATH = "./vector_store_db" 
MAX_HISTORY_TURNS = 5 # Maximum number of conversation turns to keep in history

class SimpleRAG:
  def __init__(self, force_create_db = False): # Renamed 'force' to 'force_create_db' for clarity
    self.embeddings = init_embeddings()
    self.vector_store = None
    
    if not force_create_db:
      print("正在尝试加载已存在的向量数据库...")
      # Assuming load_vector_store knows where to load from or accepts a path
      self.vector_store = load_vector_store(embeddings=self.embeddings)
      if self.vector_store:
          print("向量数据库加载成功。")
    
    if not self.vector_store:
      if force_create_db:
          print(f"接收到强制指令，正在创建/覆盖向量数据库...")
      else:
          print(f"未找到或无法加载向量数据库，将创建新的数据库...")
      
      docs = load_documents(DOCS_DIR)
      if docs:
        chunks = split_documents(docs)
        # Assuming create_vector_store handles creation or overwriting at its designated path
        self.vector_store = create_vector_store(chunks, self.embeddings)
        if self.vector_store:
            print("新的向量数据库创建成功。")
        else:
            print("错误：创建向量数据库失败。RAG 功能可能受限。")
      else:
        print(f"警告：在目录 '{DOCS_DIR}' 中没有找到任何文档，无法创建向量数据库。")
        # The application might not be usable if vector_store remains None.
        # The answer() method should handle this.

    self.llm = self.init_llm()
    self.history = [] # Initialize chat history
    
  def init_llm(self):
    chat_model= ChatOpenAI(
      model="deepseek-chat",
      openai_api_key=API_KEY,
      openai_api_base="https://api.deepseek.com/v1",
    )
    return chat_model
  
  def answer(self, question: str) -> str:
    if not self.vector_store:
        return "抱歉，向量数据库尚未初始化或初始化失败，我无法回答您的问题。"
    
    result = query_vector_store(self.vector_store, question)
    
    context = ""
    if result:
      # Assuming result is a list of (document, score) tuples
      # Print the number of matched documents
      print(f"从向量数据库中匹配到 {len(result)} 条相关数据。")
      context = "\\n".join([doc[0].page_content for doc in result])
      # The following line was removed as per your request:
      # print(context) 
    else:
      print("从向量数据库中未匹配到相关数据。")

    history_formatted = "\\n".join([f"用户: {q}\\nAI: {a}" for q, a in self.history])
    
    prompt = f"""请根据以下对话历史和提供的上下文信息来回答用户的问题。如果上下文中没有相关信息或您不知道答案，请明确告知。

对话历史:
{history_formatted if self.history else "无对话历史"}

上下文信息:
{context if context else "无相关上下文信息"}

用户问题: {question}
AI回答:"""
      
    response = self.llm.invoke(prompt)
    answer_content = response.content
    
    # Add current question and answer to history
    self.history.append((question, answer_content))
    # Limit history length
    if len(self.history) > MAX_HISTORY_TURNS:
      self.history = self.history[-MAX_HISTORY_TURNS:]
      
    return answer_content
    
    
if __name__ == "__main__":
  # 设置 force_create_db=True 可以强制重新创建向量数据库，
  # 例如，当 ./documents 目录中的文档更新后。
  # 正常运行时，设置为 False 会尝试加载已存在的数据库。
  print("正在初始化RAG问答系统...")
  rag = SimpleRAG(force_create_db=False) 
  
  if not rag.vector_store:
      print("\\nRAG系统初始化失败，向量数据库不可用。请检查文档目录和数据库配置。程序将退出。")
  else:
      print("\\nRAG问答系统已就绪。请输入您的问题。输入 'exit' 或 'quit' 结束对话。")
      while True:
        user_question = input("\\n你: ")
        if user_question.strip().lower() in ["exit", "quit"]:
          print("AI: 再见！")
          break
        if not user_question.strip():
            continue # Skip empty input
        
        ai_response = rag.answer(user_question)
        print(f"AI: {ai_response}")
