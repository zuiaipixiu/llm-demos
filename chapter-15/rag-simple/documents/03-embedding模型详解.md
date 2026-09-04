# Embedding模型详解

## 学习目标

- 深入理解文本嵌入(Embedding)模型的原理和发展历程
- 掌握主流Embedding模型的特点和适用场景
- 学习如何选择和使用合适的Embedding模型
- 了解Embedding模型评估方法和常见指标
- 掌握优化Embedding质量的实用技巧

## 文本嵌入模型简介

文本嵌入(Text Embedding)是将文本转换为固定维度的稠密向量表示的过程，这些向量捕捉了文本的语义信息，是向量检索系统的核心基础。

### 为什么需要文本嵌入？

1. **语义理解**：将文本转换为计算机可处理的数值表示
2. **相似度计算**：通过向量间的距离/相似度度量文本间的语义关系
3. **维度降低**：将高维稀疏表示(如one-hot编码)转换为低维稠密表示
4. **迁移学习**：预训练的嵌入可用于各种下游任务

### 文本嵌入模型的发展历程

文本嵌入技术经历了从静态到动态、从浅层到深层的演进过程：

1. **统计方法时代**：TF-IDF、LSA/LSI、LDA等
2. **浅层神经网络时代**：Word2Vec、GloVe、FastText
3. **预训练语言模型时代**：BERT、RoBERTa、MPNet等
4. **最新进展**：OpenAI的text-embedding系列、BGE等专用嵌入模型

## 主流Embedding模型详解

### 1. 浅层词嵌入模型

#### Word2Vec

- **原理**：基于"相似的词出现在相似的上下文中"的假设，通过预测上下文词(CBOW)或根据上下文预测目标词(Skip-gram)学习词向量
- **优势**：训练速度快，捕捉语义关系
- **局限**：静态表示，无法处理一词多义，不考虑句子级别语境

```python
# 使用Gensim训练Word2Vec模型
from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取词向量
cat_vector = model.wv['cat']
```

#### GloVe

- **原理**：结合全局矩阵分解和局部上下文窗口方法
- **优势**：捕捉全局词共现统计信息
- **局限**：与Word2Vec相似，是静态表示

#### FastText

- **原理**：扩展Word2Vec，将词表示为子词(n-gram)的集合
- **优势**：能处理未登录词，适合形态丰富的语言
- **局限**：仍是静态表示，计算复杂度较高

### 2. 预训练语言模型嵌入

#### BERT系列嵌入

- **原理**：利用BERT等预训练模型的中间层表示作为嵌入
- **优势**：动态表示，捕捉上下文语义
- **代表模型**：BERT、RoBERTa、DistilBERT等

```python
# 使用transformers获取BERT嵌入
from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# 生成嵌入
inputs = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 使用[CLS]标记作为句子表示
embeddings = outputs.last_hidden_state[:, 0, :]
```

#### Sentence-BERT

- **原理**：通过孪生/三元组网络微调BERT，优化句子级别的相似度任务
- **优势**：直接优化句子嵌入的相似度，性能优于原始BERT
- **应用**：语义搜索、文本聚类、推荐系统

```python
# 使用Sentence-BERT生成句子嵌入
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sentences = ["This is an example sentence", "Each sentence is converted"]

embeddings = model.encode(sentences)
print(f"嵌入维度: {embeddings.shape}")  # (2, 384)
```

### 3. 专用嵌入模型

#### OpenAI Embedding模型

- **代表模型**：text-embedding-ada-002、text-embedding-3-small/large
- **特点**：高质量、通用性强、维度高(1536或3072)
- **应用**：广泛用于RAG系统、语义搜索

```python
# 使用OpenAI API生成嵌入
from openai import OpenAI

client = OpenAI(api_key="your-api-key")
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text goes here",
    dimensions=1536  # 可选，默认为1536
)
embedding = response.data[0].embedding
```

#### BGE系列

- **开发者**：北京智源研究院(BAAI)
- **特点**：针对中英文优化，适合中文场景
- **版本**：提供大中小多种规格

```python
# 使用BGE模型生成嵌入
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings = model.encode(["这是一个中文句子示例。"])
```

#### GTE系列

- **开发者**：阿里巴巴达摩院
- **特点**：通用文本嵌入，多语言支持
- **优势**：在多个基准测试中表现优异

## 嵌入模型的选择与使用

### 选择合适的嵌入模型

选择嵌入模型时，需要考虑以下因素：

1. **质量与性能**：
   - 大模型通常质量更高，但计算成本也更高
   - 小模型速度快，适合资源受限场景

2. **语言支持**：
   - 多语言模型 vs 单语言模型
   - 中文场景推荐：BGE、GTE或专为中文优化的模型

3. **长度支持**：
   - 句子级模型：适合短文本
   - 支持长文本的模型：适合文档嵌入

4. **特定领域需求**：
   - 通用域模型 vs 特定领域模型（如医疗、法律等）

5. **部署限制**：
   - API调用：OpenAI嵌入API
   - 本地部署：Sentence-BERT等开源模型

### 嵌入模型性能对比

| 模型名称 | 维度 | 语言支持 | MTEB得分 | 适用场景 |
|---------|-----|---------|---------|----------|
| OpenAI text-embedding-3-small | 1536 | 多语言 | 62.3 | 通用搜索、RAG |
| OpenAI text-embedding-3-large | 3072 | 多语言 | 65.7 | 高精度搜索、困难领域 |
| BAAI/bge-large-zh-v1.5 | 1024 | 中英文 | 63.1(中文) | 中文搜索、RAG |
| BAAI/bge-small-zh-v1.5 | 512 | 中英文 | 56.2(中文) | 轻量级中文应用 |
| sentence-t5-xxl | 768 | 英文 | 61.3 | 英文搜索 |
| GTE-large | 1024 | 多语言 | 64.2 | 通用多语言应用 |

## 优化嵌入模型的实用技巧

### 1. 文本预处理

- **规范化**：大小写转换、去除特殊字符
- **分段**：将长文本分成适当长度的段落
- **去除停用词**：对于某些应用可提高效果
- **格式标准化**：统一日期、数字等格式

```python
def preprocess_text(text):
    # 基本清洗
    text = re.sub(r'[^\w\s]', '', text)
    # 转小写
    text = text.lower()
    # 分词和去停用词（可选）
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)
```

### 2. 嵌入策略

#### 文档嵌入方法

1. **直接嵌入**：整个文档作为一个输入
   - 优点：保留整体语义
   - 缺点：受模型最大长度限制，可能丢失细节

2. **分块嵌入**：文档分成小块，每块单独嵌入
   - 优点：保留详细信息，适合精确检索
   - 缺点：可能丢失全局语义

3. **层次化嵌入**：结合段落和文档级别嵌入
   - 优点：同时保留局部和全局信息
   - 缺点：计算成本高，实现复杂

```python
# 分块嵌入示例
def chunk_document(text, chunk_size=512, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # 避免在单词中间切分
        if end < len(text):
            # 向后查找空格
            while end > start and text[end] != ' ':
                end -= 1
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 对每个块生成嵌入
embeddings = []
for chunk in chunk_document(long_document):
    embedding = model.encode(chunk)
    embeddings.append(embedding)
```

### 3. 提示工程优化嵌入

使用适当的提示（prompt）可以显著提高嵌入质量：

1. **任务前缀**：添加指示嵌入用途的前缀
   - 例如："查询: "、"文档: "、"为检索准备的文本: "

2. **上下文增强**：添加相关背景信息
   - 例如，对于医学文本："以下是关于糖尿病治疗的医学文本: "

3. **结构化提示**：使用模板结构化文本
   - 例如："标题: {title}\n内容: {content}\n关键词: {keywords}"

```python
# 带提示工程的嵌入生成
def enhanced_embedding(text, purpose="search"):
    if purpose == "search":
        prompt = f"查询: {text}"
    elif purpose == "document":
        prompt = f"文档: {text}"
    else:
        prompt = text
    
    return model.encode(prompt)
```

## 嵌入模型的评估方法

### 1. 内在评估

- **向量空间分析**：
  - 余弦相似度分布
  - 聚类分析
  - 降维可视化（t-SNE, UMAP）

- **语义属性测试**：
  - 词类比测试
  - 近义词/反义词测试

```python
# t-SNE可视化嵌入
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
for i, text in enumerate(sentences):
    plt.annotate(text[:20], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.title("t-SNE Visualization of Embeddings")
plt.savefig("embeddings_visualization.png")
```

### 2. 外在评估

- **检索性能指标**：
  - Precision@K
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

- **基准数据集**：
  - MTEB (Massive Text Embedding Benchmark)
  - BEIR (Benchmark for Information Retrieval)
  - GLUE/SuperGLUE

## 小结

本节我们深入学习了文本嵌入模型的原理和应用：

1. 从浅层词嵌入到预训练语言模型嵌入，再到专用嵌入模型的发展历程
2. 各类主流嵌入模型的特点和适用场景
3. 嵌入模型的选择标准和性能对比
4. 通过文本预处理、分块策略和提示工程优化嵌入效果
5. 嵌入模型的评估方法和常用指标

良好的嵌入质量是向量检索系统的关键基础，选择合适的嵌入模型并进行适当优化，能够显著提升检索系统的整体性能。

## 思考题

1. 对于一个特定领域（如医疗或法律）的应用，你会如何选择和优化嵌入模型？
2. 分析不同文档分块策略（大块、小块、重叠块）对检索性能的影响，并思考如何为特定应用选择最佳分块方案。
3. 如何评估提示工程对嵌入质量的影响？设计一个实验方案来量化这种影响。
4. 针对中文文本，现有的嵌入模型存在哪些挑战，如何改进？ 