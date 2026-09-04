# db_utils.py

import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, Any

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# ============ 1. 动态创建 Chroma DB 目录 =============
def create_new_chroma_db(base_dir: str = "chroma_db") -> str:
    """
    每次开始新的分析任务时调用：在 base_dir 下生成一个唯一的子目录来存储向量索引。
    例如：base_dir/20250605_153022_UUID/
    返回：完整的任务目录路径
    """
    # 如果 base_dir 不存在，则创建
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # 用当前时间 + uuid 生成一个唯一 task_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    task_id = f"{timestamp}_{unique_id}"

    task_dir = os.path.join(base_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir  # e.g. "chroma_db/20250605_153022_ab12cd34"

# ============ 2. 构建并返回 Chroma 向量数据库实例 =============
def build_vectorstore(
    docs: "List[Document]",
    db_path: str,
    embedding_model_path: str = "./bge-large-zh"
) -> Chroma:
    """
    使用 HuggingFaceEmbeddings 加载本地 bge-large-zh，并将传入的 Document 列表存入 Chroma 数据库。
    Args:
      - docs: Document 列表，每个 Document 包含 page_content 和 metadata
      - db_path: 上一步 create_new_chroma_db 返回的目录，比如 "chroma_db/20250605_.../"
      - embedding_model_path: 本地嵌入模型路径（默认 ./bge-large-zh）
    Returns:
      - 完成 add_documents 和 persist() 之后的 Chroma 实例
    """
    # 1. 加载本地嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)

    # 2. 创建或加载 Chroma 向量数据库
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # 3. 如果 docs 不为空，则 add_documents 并持久化
    if docs:
        vectordb.add_documents(docs)
        vectordb.persist()

    return vectordb

# ============ 3. 保存最终报告到 JSON 文件 =============
def persist_report_info(
    task_id: str,
    db_path: str,
    report_text: str,
    token_usage: Dict[str, Any],
    created_at: datetime,
    completed_at: datetime,
    base_dir: str = "reports"
) -> str:
    """
    将任务执行情况保存到 JSON 文件里。包括：
      - task_id
      - db_path（Chroma 索引所在目录）
      - report_text（LLM 最终输出的报告字符串）
      - token_usage：{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
      - created_at、completed_at：datetime 对象
    JSON 文件最终存放在 base_dir/{task_id}.json
    返回：JSON 文件的完整路径
    """
    # 1. 如果 base_dir 不存在，则创建
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    # 2. 构造要写入的字典
    data = {
        "task_id": task_id,
        "db_path": db_path,
        "report": report_text,
        "token_usage": token_usage,
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": completed_at.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 3. 写 JSON 文件
    json_path = os.path.join(base_dir, f"{task_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path