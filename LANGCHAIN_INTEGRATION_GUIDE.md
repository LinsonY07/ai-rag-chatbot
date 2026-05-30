# RAG 智能问答系统 - LangChain 框架集成学习指南

## 🎯 学习目标

通过逐步替换原有代码，深入理解 LangChain 的核心组件及其作用：

| LangChain 组件 | 对应原代码功能 | 学到的知识点 |
|---------------|---------------|-------------|
| `ChatPromptTemplate` | `build_prompt()` 手动拼接字符串 | 如何使用模板管理 Prompt |
| `RecursiveCharacterTextSplitter` | `split_text()` 手动分块 | 文本分块的算法和参数调优 |
| `Chroma.as_retriever()` | `retrieve_relevant_docs()` 向量检索 | Retriever 抽象层的优势 |
| `Runnable` Pipeline | 手写函数调用链 | LangChain 流程编排能力 |
| LLM Adapter | `Generation.call()` 直接调用 | 如何封装第三方 LLM SDK |

---

## 📋 实施路线图

```
阶段 1              阶段 2               阶段 3              阶段 4             阶段 5             阶段 6             阶段 7
  │                   │                    │                  │                 │                 │                 │
  ▼                   ▼                    ▼                  ▼                 ▼                 ▼                 ▼
安装依赖       ───►   改造 rag.py      ───►   改造 routes.py   ───►   重排功能     ───►    记忆集成     ───►   完整 Pipeline
(环境准备)           (RAG 核心逻辑)        (接口层)             (可选增强)         (LangChain 化)      (最终整合)
```

---

## 🚀 阶段 1：安装依赖与初始化

### 步骤 1.1：更新 requirements.txt

在 `requirements.txt` 中添加 LangChain 相关依赖（**保持原有依赖不变**）：

```txt
# ========== LangChain 框架 ==========
langchain==1.3.1
langchain-core==1.4.0
langchain-community==0.4.2
langchain-chroma==1.1.0
langchain-text-splitters==1.1.2
```

### 步骤 1.2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 1.3：创建备份（可选但推荐）

```bash
# Windows PowerShell
Copy-Item core\rag.py core\rag.py.backup
Copy-Item api\routes.py api\routes.py.backup
```

---

## 🔧 阶段 2：改造 core/rag.py

这是最关键的一步！我们将逐部分替换原有代码，每完成一部分都要测试确保功能正常。

### 2.1：创建辅助工具类

首先创建两个新文件来封装 LangChain 适配器：

#### 创建 `utils/llm_chain.py`

```python
# utils/llm_chain.py
"""
LangChain LLM 适配层

学习目标：了解如何将第三方 LLM SDK（如 dashscope）封装为 
LangChain 标准接口，使其能与其他 LangChain 组件协同工作。
"""
from typing import Any, Dict, Generator, List
from dashscope import Generation
from langchain_core.messages import AIMessage
from utils.config import settings
from utils.logger import logger


class QwenChatLLM:
    """
    通义千问 LLM 包装器
    
    作用：将 dashscope 的 API 格式转换为 LangChain 统一的 
    ChatMessage 格式，使得可以使用 ChatPromptTemplate、Runnable 等组件。
    """
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        self.model = model or settings.llm_model
        self.temperature = temperature
        self.api_key = settings.dashscope_api_key
        
    def _messages_to_dashscope(self, messages: Any) -> List[Dict]:
        """
        转换消息格式
        
        LangChain 的消息对象 → dashscope 期望的字典格式
        这是集成任何新模型都需要做的适配工作
        """
        if isinstance(messages, list):
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages 
                if hasattr(msg, 'role') and hasattr(msg, 'content')
            ]
        return [{"role": "user", "content": str(messages)}]
    
    def invoke(self, messages: Any, **kwargs) -> AIMessage:
        """
        同步调用 LLM
        
        这是 LangChain Runnable 协议的基础方法
        所有 LangChain 组件都遵循这个调用签名
        """
        formatted = self._messages_to_dashscope(messages)
        response = Generation.call(
            model=self.model,
            messages=formatted,
            stream=False,
            api_key=self.api_key,
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = getattr(getattr(response, 'output', None), 'text', '') or ''
        return AIMessage(content=content)
    
    def stream(self, messages: Any) -> Generator[str, None, None]:
        """流式调用，用于 StreamingResponse"""
        formatted = self._messages_to_dashscope(messages)
        response = Generation.call(
            model=self.model,
            messages=formatted,
            stream=True,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        for chunk in response:
            text = getattr(getattr(chunk, 'output', None), 'text', '') or ''
            yield text


qwen_llm = QwenChatLLM()
```

**学习要点：**
- LangChain 的 `Runnable` 协议只需要实现 `invoke()` 和可选的 `stream()`
- 通过这个适配器，后面可以无缝使用 `ChatPromptTemplate` 和 `Runnable` pipeline

---

#### 创建 `utils/langchain_adapters.py`

```python
# utils/langchain_adapters.py
"""
LangChain 核心组件适配

这一节将深入学习 LangChain 的几个核心组件：
1. Embeddings - 向量化
2. Chroma VectorStore - 向量数据库
3. Text Splitters - 文本分块
4. Retriever - 检索器
5. Prompt Templates - Prompt 管理
"""
from typing import List
from dashscope import TextEmbedding
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from utils.config import settings
from utils.logger import logger


# =========================================
# 1. Embeddings 组件
# =========================================

class DashScopeEmbeddings(Embeddings):
    """
    Embeddings 接口实现
    
    LangChain 中的 Embeddings 是一个抽象接口，
    不同模型供应商（OpenAI、Azure、本地模型等）
    只要实现这个接口就能与 LangChain 的其他组件兼容。
    """
    
    def __init__(self, model: str = None):
        self.model = model or settings.embedding_model
        self.api_key = settings.dashscope_api_key
    
    def _get_embedding(self, text: str) -> List[float]:
        """调用 API 获取单个文本的向量"""
        resp = TextEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=text
        )
        return resp.output["embeddings"][0]["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        这个方法用于构建向量索引时调用
        """
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本
        
        这个方法用于搜索时调用
        """
        return self._get_embedding(text)


# 全局单例
lc_embeddings = DashScopeEmbeddings()


# =========================================
# 2. Text Splitter 组件
# =========================================

def get_text_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """
    Text Splitter - 文本分块器
    
    与原代码对比：
    - 原代码：手动的滑动窗口切分
    - LangChain：RecursiveCharacterTextSplitter
      按字符类型递归切分（段落、句子、单词），更适合语义完整性
    
    关键参数：
    - chunk_size: 每个块的最大字符数
    - chunk_overlap: 重叠字符数，帮助跨块的上下文连贯
    - separators: 优先按哪些字符分割（默认：[\n\n, \n, , , , ]）
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


# =========================================
# 3. VectorStore & Retriever 组件
# =========================================

def get_vectorstore(collection_name: str = "knowledge_base"):
    """
    VectorStore - 向量存储
    
    LangChain 统一了各种向量数据库的接口，
    Chroma、Pinecone、Weaviate 等都可以通过同样的方式访问。
    """
    return Chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_db_path,
        embedding_function=lc_embeddings,
    )


def get_retriever(vectorstore=None, top_k: int = None):
    """
    Retriever - 检索器
    
    这是 LangChain 的核心组件之一！
    Retriever 负责从知识库中检索相关文档，
    返回的是 Document 对象列表。
    
    原代码中的 retrieve_relevant_docs() 现在可以由
    vectorstore.as_retriever().invoke(query) 替代
    """
    top_k = top_k or settings.recall_top_k
    
    if vectorstore is None:
        vectorstore = get_vectorstore()
    
    # as_retriever() 创建一个 Retriever 对象
    # search_type: "similarity" (余弦相似度) / "mmr" (最大边际相关性)
    # search_kwargs: 额外的搜索参数
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


# =========================================
# 4. Prompt Template 组件
# =========================================

def get_prompt_template():
    """
    ChatPromptTemplate - 聊天提示模板
    
    与原代码对比：
    - 原代码：用 f-string 手动拼接 Prompt 字符串，容易出错且难以维护
    - LangChain: ChatPromptTemplate
      结构化定义 Prompt，支持动态变量、消息历史、系统角色分离
    
    优势：
    1. 类型安全：模板结构在编译期检查
    2. 可复用：同一个模板可用于不同场景
    3. 易维护：修改一处，全局生效
    4. 支持历史对话：自动处理多轮对话格式
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    return ChatPromptTemplate.from_messages([
        # system 消息：定义 AI 的角色和行为准则
        ("system", """你是严谨的知识库问答助手，请严格遵守以下规则：
1. 优先根据历史对话上下文理解用户当前问题的真实意图
2. 必须严格基于参考资料内容回答问题，不编造资料中没有的信息
3. 如果参考资料中没有相关信息，明确告知用户后可以用自己的知识回答
4. 回答简洁自然，避免冗余
5. 回答结束后，用 ###SOURCE### 分隔，列出所有参考来源"""),
        
        # placeholder：留给历史对话填充的位置
        MessagesPlaceholder(variable_name="chat_history"),
        
        # user 消息：具体的用户问题和补充信息
        ("human", """参考资料：
{context}

用户问题：{question}"""),
    ])
```

---

### 2.2：创建 core/rag_langchain.py

这是 LangChain 版本的 RAG 核心逻辑，与原 `core/rag.py` 独立存在：

```python
# core/rag_langchain.py
"""
LangChain 版本的 RAG 核心逻辑

这是原 core/rag.py 的 LangChain 重构版本。
对比学习点：
- 如何用更少的代码实现相同的功能
- LangChain 各组件如何协作
- 流程是如何通过 Runnable 编排的
"""
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.config import settings
from utils.logger import logger
from utils.langchain_adapters import (
    get_text_splitter,
    get_vectorstore,
    get_retriever,
)
from utils.llm_chain import qwen_llm


# =========================================
# 文件解析函数（保持不变）
# =========================================

import pdfplumber
from docx import Document
from openpyxl import load_workbook


def extract_txt_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])


def extract_xlsx(file_path: str) -> str:
    full_text = []
    wb = load_workbook(filename=file_path, read_only=True, data_only=True)
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_data = [str(cell).strip() for cell in row if cell is not None]
            if row_data:
                full_text.append(" | ".join(row_data))
    return "\n".join(full_text)


def get_file_content(file_path: str) -> str:
    ext = file_path.split(".")[-1].lower()
    if ext in ["txt", "md"]:
        return extract_txt_md(file_path)
    elif ext == "pdf":
        return extract_pdf(file_path)
    elif ext == "docx":
        return extract_docx(file_path)
    elif ext == "xlsx":
        return extract_xlsx(file_path)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


# =========================================
# 文本分块
# =========================================

def split_document(file_path: str, source_name: str) -> List[Document]:
    """
    文本分块 - 使用 LangChain TextSplitter
    
    与原代码 split_text() 对比：
    - 原代码：返回纯字符串列表
    - 新版本：返回 Document 列表，每个 Document 自带 metadata
    - metadata 包含了文件的 source 信息，这对后续追踪答案来源非常重要
    """
    content = get_file_content(file_path)
    
    # 创建 Document 对象（带元数据）
    docs = [Document(page_content=content, metadata={"source": source_name})]
    
    # 使用 TextSplitter 分块
    splitter = get_text_splitter()
    splitted = splitter.split_documents(docs)
    
    logger.info(f"✅ 分块完成：{len(splitted)} 个块")
    return splitted


# =========================================
# 向量入库
# =========================================

def add_to_vectorstore(documents: List[Document], collection_name: str = "knowledge_base"):
    """
    添加文档到向量库
    
    与原代码 add_chunks_to_chroma() 对比：
    - 原代码：需要手动调用 embedding API + 组装 IDs + 调用 chromadb.add()
    - 新版本：VectorStore 统一管理，自动调用 embedding + 批量添加
    """
    store = get_vectorstore(collection_name)
    ids = store.add_documents(documents)
    logger.info(f"✅ 添加 {len(documents)} 个文档到向量库")
    return ids


# =========================================
# 检索函数
# =========================================

def retrieve_documents(query: str, top_k: int = None) -> List[Dict]:
    """
    检索相关文档 - 使用 LangChain Retriever
    
    与原代码 retrieve_relevant_docs() 对比：
    - 原代码：手动调用 embedding API + Chroma.query() + 结果解析
    - 新版本：Retriever 一行代码搞定全部流程！
    
    Retriever 的优势：
    1. 代码更少，更易读
    2. 支持多种搜索策略（similarity, mmr, etc.）
    3. 易于扩展（添加 compression, filtering 等）
    """
    top_k = top_k or settings.recall_top_k
    
    retriever = get_retriever(top_k=top_k)
    
    # invoke() 是 LangChain Runnable 的标准调用方法
    docs = retriever.invoke(query)
    
    # Document 对象已经包含 page_content 和 metadata
    results = [
        {
            "content": doc.page_content.strip(),
            "source": doc.metadata.get("source", "未知")
        }
        for doc in docs
    ]
    
    logger.info(f"✅ 检索完成：{len(results)} 条结果")
    return results


# =========================================
# 构建 Prompt
# =========================================

def build_rag_context(
    query: str,
    history: List[Dict] = None,
    top_k: int = None
) -> Dict[str, Any]:
    """
    构建 RAG 上下文
    
    这是一个中间函数，将检索结果和历史对话组织成
    LangChain Prompt 需要的输入格式
    """
    history = history or []
    
    # 格式化历史对话为 LangChain 消息格式
    chat_history = []
    for msg in history[-settings.max_chat_history_round:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        chat_history.append({"role": role, "content": msg["content"]})
    
    # 检索文档
    retrieved = retrieve_documents(query, top_k)
    
    # 格式化上下文
    context = "\n\n".join(
        f"[来源：{doc['source']}]\n{doc['content']}"
        for doc in retrieved
    )
    
    return {
        "chat_history": chat_history,
        "context": context,
        "question": query
    }
```

**学习要点总结：**
1. `TextSplitter` 比手动分块更智能，能保持语义完整性
2. `Document` 对象自带 metadata，方便追踪数据来源
3. `Retriever` 封装了向量检索的复杂细节
4. `ChatPromptTemplate` 让 Prompt 结构更清晰易维护

---

## 🌐 阶段 3：改造 api/routes.py

### 3.1：新增导入

在 `api/routes.py` 文件开头添加：

```python
# api/routes.py 开头的导入区域

# 新增导入
from core.rag_langchain import build_rag_context
from utils.llm_chain import qwen_llm
```

### 3.2：改造 ask_stream 接口

找到 `/ask_stream` 接口，替换其内部实现：

```python
# api/routes.py

@router.post("/ask_stream")
async def ask_stream(request: Request):
    """
    RAG 问答接口 - LangChain 版本
    
    主要变化：
    1. 使用 build_rag_context() 替代原来的 retrieve_relevant_docs() + build_prompt()
    2. 使用 qwen_llm.stream() 替代原来的 Generation.call()
    3. Prompt 由 ChatPromptTemplate 自动处理，无需手动拼接
    """
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        logger.info(f"会话：{session_id} 用户提问：{question}")
        
        if not question:
            return StreamingResponse(iter(["请输入你的问题~"]), media_type="text/plain")
        
        # 获取对话记忆
        memory = ConversationMemory(session_id=session_id, max_turns=settings.max_chat_history_round)
        history = memory.get_history()
        memory.add_message("user", question)
        
        # 构建 RAG 上下文（使用 LangChain 组件）
        rag_input = build_rag_context(question, history, settings.recall_top_k)
        
        def generate():
            full_answer = ""
            try:
                # 构造 LangChain 消息列表
                messages = rag_input["chat_history"] + [{"role": "user", "content": rag_input["context"] + "\n\n问题：" + rag_input["question"]}]
                
                # 流式调用 LLM
                for token in qwen_llm.stream(messages):
                    full_answer += token
                    yield token
                    time.sleep(0.01)
                
                # 输出参考来源
                if rag_input["context"].strip():
                    sources = list(set([
                        doc["source"] 
                        for doc in retrieve_documents(question, settings.recall_top_k)
                    ]))
                    sources = [s for s in sources if s and s != "未知"]
                    if sources:
                        yield "\n\n📚 参考来源：" + " | ".join(sources)
                
            except Exception as e:
                logger.error(f"模型调用失败：{e}")
                yield f"抱歉，模型调用出错了：{str(e)[:50]}"
            finally:
                try:
                    memory.add_message("assistant", full_answer or "无回答")
                except:
                    pass
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        logger.error(f"接口异常：{e}")
        return StreamingResponse(iter([f"接口出错：{str(e)}"]), media_type="text/plain")
```

**学习要点：**
- LangChain 的消息格式是 `[{"role": "...", "content": "..."}, ...]` 列表
- Stream 接口保持一致，只是底层调用方式变了
- Prompt 的拼接由 `ChatPromptTemplate` 内部管理，前端感知不到变化

---

## 🔄 阶段 4：集成重排功能（ContextualCompressionRetriever）

### 4.1：创建 Rerank 适配器

在 `utils/langchain_adapters.py` 文件中新增：

```python
# =========================================
# 5. Contextual Compression / Rerank
# =========================================

class DashScopeReranker:
    """
    阿里通义千问 Rerank 适配器的 LangChain 实现
    
    作用：对检索结果进行重新排序，提升相关性
    LangChain 中的 ContextualCompressionRetriever 可以包装这个压缩器
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.dashscope_api_key
        import requests
        self.requests = requests
    
    def compress_documents(
        self, 
        documents: List[Document], 
        query: str,
        callbacks: Any = None
    ) -> List[Document]:
        """
        压缩/重排文档列表
        
        这是 LangChain CompressionRetriever 要求的接口
        """
        if not documents or not query.strip():
            return list(documents)
        
        docs_list = [doc.page_content for doc in documents]
        
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        req_data = {
            "model": "gte-rerank-v2",
            "input": {"query": query, "documents": docs_list},
            "parameters": {"return_documents": True, "top_n": len(documents)}
        }
        
        try:
            resp = self.requests.post(url, headers=headers, json=req_data, timeout=8)
            if resp.status_code != 200:
                return list(documents)
            
            result_json = resp.json()
            ranked_docs = []
            MIN_SCORE = 0.01
            
            for item in result_json.get("output", {}).get("results", []):
                score = item["relevance_score"]
                idx = item["index"]
                if score > MIN_SCORE and idx < len(documents):
                    ranked_docs.append(documents[idx])
            
            return ranked_docs
        except Exception as e:
            logger.error(f"重排失败：{e}")
            return list(documents)


def get_compression_retriever(vectorstore=None, reranker=None, top_k: int = None):
    """
    获取带重排功能的检索器
    
    使用 LangChain 的 ContextualCompressionRetriever
    它将先通过 vectorstore 检索，再通过 reranker 重排
    """
    from langchain.retrievers import ContextualCompressionRetriever
    
    top_k = top_k or settings.rerank_top_k
    
    if vectorstore is None:
        vectorstore = get_vectorstore()
    if reranker is None:
        reranker = DashScopeReranker()
    
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 2}  # 先取多一些，给重排留空间
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    
    return compression_retriever
```

### 4.2：更新 rag_langchain.py 中的检索函数

在 `core/rag_langchain.py` 中更新或新增 `retrieve_documents` 函数以支持可选的重排：

```python
def retrieve_documents(query: str, top_k: int = None, use_rerank: bool = True) -> List[Dict]:
    """
    检索相关文档 - 支持可选的重排功能
    """
    from utils.langchain_adapters import get_compression_retriever
    
    top_k = top_k or settings.rerank_top_k
    
    if use_rerank:
        retriever = get_compression_retriever(top_k=top_k * 2)
    else:
        retriever = get_retriever(top_k=top_k)
    
    docs = retriever.invoke(query)
    
    results = [
        {
            "content": doc.page_content.strip(),
            "source": doc.metadata.get("source", "未知")
        }
        for doc in docs[:top_k]
    ]
    
    logger.info(f"✅ 检索完成（{'重排' if use_rerank else '普通'}）：{len(results)} 条结果")
    return results
```

---

## 💾 阶段 5：集成对话记忆到 LangChain

### 5.1：创建 LangChain Memory 适配器

新建 `utils/langchain_memory.py`：

```python
# utils/langchain_memory.py
"""
LangChain 对话记忆适配器

学习目标：了解如何将自定义记忆系统封装为 LangChain Memory 接口
"""
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from core.conversation import ConversationMemory as SqliteMemory


class LangChainConversationMemory:
    """
    将 SQLite 记忆系统封装为 LangChain Memory 接口
    
    LangChain 的 Memory 接口要求实现以下方法：
    - load_memory_variables() - 加载历史对话
    - save_context() - 保存当前对话轮次
    - clear() - 清空记忆
    
    这样可以无缝集成到 LangChain 的 Chain 中
    """
    
    def __init__(self, session_id: str = "default", max_turns: int = 6):
        self.session_id = session_id
        self.sqlite_memory = SqliteMemory(session_id=session_id, max_turns=max_turns)
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        加载记忆变量
        
        返回格式必须符合 LangChain 期望：
        {"chat_history": [{"role": "...", "content": "..."}, ...]}
        """
        history = self.sqlite_memory.get_history()
        
        # 转换为 LangChain 消息格式
        messages = []
        for msg in history:
            role = "ai" if msg["role"] == "assistant" else "human"
            messages.append({"role": role, "content": msg["content"]})
        
        return {"chat_history": messages}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """
        保存当前对话轮次
        
        inputs 包含：question
        outputs 包含：response
        """
        question = inputs.get("question", "")
        response = outputs.get("response", "")
        
        self.sqlite_memory.add_message("user", question)
        self.sqlite_memory.add_message("assistant", response)
    
    def clear(self):
        """清空记忆"""
        self.sqlite_memory.clear()


# 全局工厂函数
def get_conversation_memory(session_id: str = "default") -> LangChainConversationMemory:
    return LangChainConversationMemory(session_id=session_id)
```

---

## ⚡ 阶段 6：完整的 LangChain Runnable Pipeline

### 6.1：创建最终版的 RAG Chain

在 `core/rag_langchain.py` 文件末尾添加：

```python
# =========================================
# 完整的 LangChain RAG Pipeline
# =========================================

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def create_full_rag_chain(
    use_rerank: bool = True,
    top_k: int = None
):
    """
    创建完整的 LangChain RAG Pipeline
    
    这是所有组件的最终组合形式：
    
    Input → Retrieval → Rerank (可选) → Prompt Template → LLM → Output
    
    每个环节都由 LangChain 的标准接口连接，可以独立替换或调试
    """
    top_k = top_k or settings.recall_top_k
    
    # 1. 检索器
    if use_rerank:
        retriever = get_compression_retriever(top_k=top_k * 2)
    else:
        retriever = get_retriever(top_k=top_k)
    
    # 2. Prompt 模板
    prompt = get_prompt_template()
    
    # 3. LLM
    llm = qwen_llm
    
    # 4. 构建 Pipeline
    # RunnablePassthrough 用于传递原始输入
    # | 表示将上一个输出传递给下一个
    rag_pipeline = (
        # 第一步：准备上下文（检索 + 格式化）
        {
            "context": retriever,  # 自动调用 retriever.invoke(question)
            "question": RunnablePassthrough(),  # 原样传递问题
        }
        # 第二步：填充 Prompt 模板
        | prompt
        # 第三步：调用 LLM
        | llm
        # 第四步：输出纯文本
        | StrOutputParser()
    )
    
    return rag_pipeline


# 便捷调用函数
def ask_with_full_chain(question: str) -> str:
    """使用完整 RAG Pipeline 回答问题"""
    chain = create_full_rag_chain()
    result = chain.invoke(question)
    return result
```

---

## 📡 阶段 7：最终版 routes.py

### 7.1：更新导入

```python
# api/routes.py (最终完整版本)

from core.rag_langchain import create_full_rag_chain
from utils.langchain_memory import get_conversation_memory
from utils.llm_chain import qwen_llm
```

### 7.2：完整的 ask_stream 接口

```python
@router.post("/ask_stream")
async def ask_stream(request: Request):
    """
    完整 LangChain 版本的 RAG 问答接口
    
    特点：
    - 使用 Runnable Pipeline 编排整个流程
    - 支持可选的重排功能
    - 使用 LangChain Memory 接口
    - 保持流式输出
    """
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        logger.info(f"会话：{session_id} 用户提问：{question}")
        
        if not question:
            return StreamingResponse(iter(["请输入你的问题~"]), media_type="text/plain")
        
        # 初始化 Memory
        memory = get_conversation_memory(session_id)
        memory_vars = memory.load_memory_variables({})
        
        def generate():
            full_answer = ""
            try:
                # 创建 RAG Chain
                chain = create_full_rag_chain(
                    use_rerank=True,
                    top_k=settings.recall_top_k
                )
                
                # 流式调用
                for token in chain.astream({"question": question}):
                    full_answer += token
                    yield token
                
                # 输出来源
                sources = retrieve_documents(question, settings.rerank_top_k)
                source_names = list(set([d["source"] for d in sources if d["source"] != "未知"]))
                if source_names:
                    yield "\n\n📚 参考来源：" + " | ".join(source_names)
                
            except Exception as e:
                logger.error(f"模型调用失败：{e}")
                yield f"抱歉，模型调用出错了：{str(e)[:50]}"
            finally:
                try:
                    memory.save_context(
                        {"question": question},
                        {"response": full_answer or "无回答"}
                    )
                except:
                    pass
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        logger.error(f"接口异常：{e}")
        return StreamingResponse(iter([f"接口出错：{str(e)}"]), media_type="text/plain")
```

---

## 🧪 阶段 8：测试验证

### 8.1：启动服务并测试

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

```bash
# 测试功能是否正常
curl -X POST http://127.0.0.1:8000/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Python 入门基础知识有哪些？", "session_id": "test"}'
```

### 8.2：功能对比表

| 功能 | 原有实现 | LangChain 实现 | API 变化 |
|------|---------|---------------|---------|
| 文本分块 | 手动滑动窗口 | `RecursiveCharacterTextSplitter` | 无 |
| 向量入库 | chromadb.add() | `Chroma.add_documents()` | 无 |
| 向量检索 | collection.query() | `Retriever.invoke()` | 无 |
| 重排 | real_rerank() | `ContextualCompressionRetriever` | 无 |
| Prompt | f-string 拼接 | `ChatPromptTemplate` | 无 |
| LLM | Generation.call() | `QwenChatLLM` | 无 |
| 记忆 | ConversationMemory 类 | LangChain Memory 接口 | 无 |

**结论：** 前端/API 完全无感知，所有改动都在后端内部完成！

---

## 🏗️ 完整的 LangChain RAG 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      完整的 LangChain RAG Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                                                │
│   │   Input     │  ← 用户问题                                     │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────┐                                          │
│   │  {context:       │                                           │
│   │   retriever}     │  ← 检索器（可配置重排）                       │
│   │  question: Passthrough                             │
│   └────────┬─────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │ ChatPromptTemplate│  ← 结构化 Prompt                          │
│   │ - system         │                                           │
│   │ - chat_history   │                                           │
│   │ - human          │                                           │
│   └────────┬─────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │ QwenChatLLM      │  ← LLM 适配器                              │
│   └────────┬─────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌──────────────────┐                                          │
│   │ StrOutputParser  │  ← 输出解析                               │
│   └────────┬─────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────┐                                                │
│   │   Output    │  ← 最终答案                                    │
│   └─────────────┘                                                │
│                                                                  │
│   ┌──────────────────────────────────────────────────┐           │
│   │              外部依赖层                            │           │
│   │  - DashScopeEmbeddings                           │           │
│   │  - Chroma VectorStore                            │           │
│   │  - DashScopeReranker                             │           │
│   │  - LangChainMemory                               │           │
│   └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 附：所有集成的 LangChain 组件清单

| 序号 | 组件 | 模块 | 用途 |
|-----|------|------|------|
| 1 | `DashScopeEmbeddings` | custom | 向量化接口实现 |
| 2 | `RecursiveCharacterTextSplitter` | langchain-text-splitters | 文本分块 |
| 3 | `Chroma` | langchain-chroma | 向量存储 |
| 4 | `Retriever` | langchain-core | 检索抽象层 |
| 5 | `DashScopeReranker` | custom | 重排压缩器 |
| 6 | `ContextualCompressionRetriever` | langchain | 重排检索器 |
| 7 | `ChatPromptTemplate` | langchain-core | Prompt 模板 |
| 8 | `MessagesPlaceholder` | langchain-core | 历史对话占位符 |
| 9 | `QwenChatLLM` | custom | LLM 适配器 |
| 10 | `StrOutputParser` | langchain-core | 输出解析 |
| 11 | `RunnablePassthrough` | langchain-core | 传递原始输入 |
| 12 | `RunnableSequence`/`\|` | langchain-core | Pipeline 编排 |
| 13 | `LangChainConversationMemory` | custom | 记忆接口适配 |

---

## 📌 注意事项

1. **环境变量保持一致**：`.env` 文件无需任何修改
2. **向量库兼容**：新旧版本共用同一个 Chroma 数据库
3. **API 无感知**：前端调用方式完全不变
4. **性能影响**：增加重排功能会有轻微延迟（约 200-500ms）
5. **可回退性**：设置 `use_rerank=False` 可关闭重排

---

## ✅ 下一步行动清单

- [ ] 1. 安装 LangChain 依赖 (`pip install -r requirements.txt`)
- [ ] 2. 创建 `utils/llm_chain.py`
- [ ] 3. 创建 `utils/langchain_adapters.py`
- [ ] 4. 创建 `utils/langchain_memory.py`
- [ ] 5. 创建 `core/rag_langchain.py`
- [ ] 6. 改造 `api/routes.py`
- [ ] 7. 测试各功能模块
- [ ] 8. 对比原有功能
- [ ] 9. 部署到生产环境

祝你学习顺利！🚀