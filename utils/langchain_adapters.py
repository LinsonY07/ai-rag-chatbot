"""
LangChain 核心组件适配
1.Embeddings - 文本向量化
2.Chroma VectorStore - 向量数据库
3.Text Splitter - 文本分块
4.Retriever - 文档检索
5.Prompt Template - 提示词模板
"""
from typing import List, Any    #类型注解工具
import requests
from dashscope import TextEmbedding     #通问千义向量SDK
from langchain_core.embeddings import Embeddings    #LangChain 向量基类
from langchain_text_splitters import RecursiveCharacterTextSplitter #文本切分依赖
from langchain_chroma import Chroma #向量数据库
from langchain_core.documents import Document
# ChatPromptTemplate:LangChain 专门给对话场景用的提示词模板,
# MessagesPlaceholder:消息占位符，LangChain 专用语法。
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 项目配置 & 日志
from utils.config import settings
from utils.logger import logger

# 该类要对接通义千问的向量模型，且必须符合 LangChian 规范
# LangChain 要求必须继承 Embeddings 抽象类
# =========================================
# 1. Embeddings 组件
# =========================================
class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str = None):
        # 模型名称：外部可传，否则用配置默认值
        self.model = model or settings.embedding_model
        self.api_key = settings.dashscope_api_key
        
    # 私有方法
    def _get_embedding(self, text: str) -> List[float]:
        # 调用通义千问 API
        resp = TextEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=text
        )
        # 从返回结构中取出向量
        return resp.output["embeddings"][0]["embedding"]
    
    # LangChain 强制要求：存入向量库时用
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量生成文档向量
        return [self._get_embedding(text) for text in texts]
    
    # LangChain 强制要求：用户提问时用
    def embed_query(self, text: str) -> List[float]:
        # 生成用户问题向量
        return self._get_embedding(text)
    
lc_embeddings = DashScopeEmbeddings()   #全局复用一个向量实例，避免重复创建

# =========================================
# 2. Text Splitter 组件
# =========================================
"""
# Text Splitter - 文本分块器
    
    # 与原代码对比：
    # - 原代码：手动的滑动窗口切分
    # - LangChain：RecursiveCharacterTextSplitter
    #   按字符类型递归切分（段落、句子、单词），更适合语义完整性
    
    # 关键参数：
    # - chunk_size: 每个块的最大字符数
    # - chunk_overlap: 重叠字符数，帮助跨块的上下文连贯
    # - separators: 优先按哪些字符分割（默认：[\n\n, \n, , , , ]）
"""
def get_text_splitter(chunk_size=None, chunk_overlap = None):
    # 不传递则使用配置默认值
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex=False,
    )
    
# =========================================
# 3. VectorStore & Retriever 组件
# =========================================
"""
    # VectorStore - 向量存储
    
    # LangChain 统一了各种向量数据库的接口，
    # Chroma、Pinecone、Weaviate 等都可以通过同样的方式访问。
"""
def get_vectorstore(collection_name: str = "knowledge_base"):
    return Chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_db_path,
        embedding_function=lc_embeddings,
    )

"""
    Retriever - 检索器
    
    这是 LangChain 的核心组件之一！
    Retriever 负责从知识库中检索相关文档，
    返回的是 Document 对象列表。
    
    原代码中的 retrieve_relevant_docs() 现在可以由
    vectorstore.as_retriever().invoke(query) 替代
"""
def get_retriever(vectorstore=None, top_k=None):
    top_k = top_k or settings.recall_top_k
    
    if vectorstore is None:
        vectorstore = get_vectorstore()
        
    return vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": top_k}
    )
    
# =========================================
# 4. Prompt Template 组件
# =========================================
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
def get_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system","""你是严谨的知识库问答助手，请严格遵守以下规则：
1. 优先根据历史对话上下文理解用户当前问题的真实意图
2. 必须严格基于参考资料内容回答问题，不编造资料中没有的信息
3. 如果参考资料中没有相关信息，明确告知用户后可以用自己的知识回答
4. 回答简洁自然，避免冗余
5. 【重要】只有当你的回答直接引用或参考了文档的具体内容时，才在回答末尾用 ###SOURCE### 列出来源
6. 如果答案完全来自对话历史或你的通用知识，绝对不要输出 ###SOURCE###，也不要列出任何文档"""),
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human","""参考资料：
         {context}
         
         用户问题：{question}
         """),
    ])
    
"""
集成重排功能：

LangChain 的重排功能通过 ContextualCompressionRetriever 实现：
关键概念：
    1.压缩器：负责对检索结果进行重排和过滤
    2.基础检索器：负责初步检索
    3.压缩检索器：组合压缩器和基础检索器
作用：实现 LangChain 的重排功能，使用通义千问的 rerank 模型提升检索质量
关键知识点：
    1.压缩器接口：实现 LangChain 的 BaseDocumentCompressor 接口
    2.API 调用：调用通义千问的 rerank 接口
    3.错误处理：API 调用失败时的降级处理
"""

# =========================================
# 5. Contextual Compression / Rerank
# =========================================
class DashScopeReranker:
    """
    阿里通问千义 Rerank 适配器的 LangChain 实现
    
    作用：对检索结果进行重新排序，提升相关性
    LangChain 中的 ContextualCompressionRetriever 可以包装这个压缩器
    """
    def __init__(self, api_key: str = None, model: str = "gte-rerank-v2", min_score: float = 0.01):
        self.api_key = api_key or settings.dashscope_api_key
        self.model = model
        self.min_score = min_score
    
    # LangChain 规定的必须接口，输入文档列表 + query，输出重排后的文档列表
    def compress_documents(
        self,
        documents,
        query,
        callbacks: Any = None
    ) -> List[Document]:
        # 第一步：防御性判断
        # 没文档/没查询 -> 直接返回原数据（防止程序崩溃，保证流程健壮）
        if not documents or not query.strip():
            return list(documents)
        
        # 第二步：提取文档文本内容
        # 把 LangChain DOcument 对象 ——> 纯文本列表（阿里 API 只接收字符串）
        docs_list = [doc.page_content for doc in documents]
        
        # 第三步：拼接 API 请求信息
        # 阿里官方 Rerank 接口地址
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        
        # 请求头：鉴权 + JSON 格式
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 请求体：严格按照阿里官方格式
        req_data = {
            "model": self.model,
            "input": {"query": query, "documents": docs_list},
            "parameters": {"return_documents": True, "top_n": len(documents)}
        }

        # 第四步：发送请求 + 异常捕获
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=req_data,
                timeout=8
            )

            # 第五步：HTTP 状态判断(API 报错 → 直接返回原始检索结果)
            if resp.status_code != 200:
                return list(documents)

            # 第六步：解析重排结果
            result_json = resp.json()
            ranked_docs = []

            # 遍历重排返回结果
            for item in result_json.get("output", {}).get("results", []):
                score = item.get("relevance_score", 0)
                idx = item.get("index", -1)

                # 分数达标 + 索引不越界
                if score > self.min_score and 0 <= idx < len(documents):
                    ranked_docs.append(documents[idx])

        # 第七步：异常捕获 + 降级返回
        except Exception as e:
            logger.error(f"重排失败：{e}")
            return list(documents)

        # 如果重排后没有结果，降级返回原始检索结果
        return ranked_docs if ranked_docs else list(documents)
         
     # 作用：把向量检索 + 重排 包装成一个检索器
     # 第一步：参数处理
def get_compression_retriever(vectorstore=None, reranker=None, top_k: int = None):
    """
    获取带重排功能的检索器
    流程：向量粗检索 -> API 精排
    使用 LangChain 的 ContextualCompressionRetriever
    它将先通过 vectorstore 检索，再通过 reranker 重排
    """
    # 延迟导入（只在用到时加载）
    # LangChain 1.x 把 retrieval 相关组件拆分到了 langchain_core，
    # 但 ContextualCompressionRetriever 本身没有被保留在核心包里。
    """
    解决的整体策略:
    先尝试用官方基类，
    失败就用自己的简易实现
    """
    try:
        # 能导入官方基类时 → 做一个 “正版壳子类”
        from langchain_core.retrievers import BaseRetriever
        
        # 继承 LangChain 官方基类 BaseRetriever，满足类型规范
        class ContextualCompressionRetriever(BaseRetriever):
            # 显式声明 base_retriever、base_compressor 两个字段，
            # 骗过新版本的 Pydantic 校验
            base_retriever: Any
            base_compressor: Any

            # BaseRetriever 规定的唯一抽象方法:定义检索逻辑
            """
            *：后面参数必须关键字传参（LangChain 规范）
            run_manager：LangChain 回调管理器（日志、追踪、监控）
            """
            # 同步方法
            def _get_relevant_documents(self, query, *, run_manager=None):
                docs = self.base_retriever.invoke(query)
                return self.base_compressor.compress_documents(docs, query)
            
            # 让检索可以异步执行，不阻塞主线程，更快、更适合高并发
            # 异步方法
            async def _aget_relevant_documents(self, query, *, run_manager=None):
                docs = await self.base_retriever.ainvoke(query)
                return self.base_compressor.compress_documents(docs, query)
            
    except ImportError:
        # 导入失败时 → 降级到 "简易兼容类"
        # 不继承任何官方类，纯手写一个 “长得一样、用起来一样” 的类
        class ContextualCompressionRetriever:
            
            """
            init 构造方法:创建实例时，传入两个必须的东西
                1.base_retriever
                基础检索器（向量库检索器）
                负责：粗检索
                2.base_compressor
                重排器（你的 DashScopeReranker）
                负责：精排
            """
            def __init__(self, base_retriever, base_compressor):
                self.base_retriever = base_retriever
                self.base_compressor = base_compressor

            def invoke(self, query):
                retriever = self.base_retriever
                """
                hasattr(对象, '方法名')
                → 判断对象是否拥有某个方法 / 属性
                → 返回 True 或 False
                如果有 invoke → 用retriever.invoke(query)调用
                如果没有 invoke → 用retriever.get_relevant_documents(query)调用
                """
                # 让基础检索器去向量库做粗检索
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
                # 把粗检索的文档 → 丢给重排器 → 返回精排后的文档
                return self.base_compressor.compress_documents(docs, query)

            def get_relevant_documents(self, query):
                retriever = self.base_retriever
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
                return self.base_compressor.compress_documents(docs, query)

    # 最终返回几条结果(读配置)
    top_k = top_k or settings.recall_top_k
        
    # 第二步:自动补全依赖
    # 没传向量库 -> 自动获取默认向量库
    if vectorstore is None:
        vectorstore = get_vectorstore()
            
    # 没传重排器 -> 自动使用阿里重排
    if reranker is None:
        reranker = DashScopeReranker()
        
    # 第三步:创建基础检索器(粗排)
    base_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": top_k * 2}    #多取一点,给重排留空间
    )
    
    # 第四步:包装成带重排的检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor = reranker,     #重排器
        base_retriever = base_retriever #粗重排器  
    )
    return compression_retriever
            
        
# # ============================
# # 工具类测试代码
# # ============================
# if __name__ == "__main__":
#     # 测试1：文本分块器
#     print("=" * 50)
#     print("【测试1】文本分块器")
#     text_splitter = get_text_splitter(chunk_size=200, chunk_overlap=20)
#     test_text = """
#     人工智能（AI）是一门旨在使计算机系统模拟人类智能的技术。
#     它包括机器学习、深度学习、自然语言处理等多个领域。
#     机器学习是AI的一个子集，让系统从数据中学习并改进。
#     深度学习是机器学习的分支，基于神经网络结构。
#     """
#     chunks = text_splitter.split_text(test_text)
#     for i,c in enumerate(chunks):
#         print(f"块{i+1}：{c[:50]}...")
        
#     # 测试2：Embeddings 向量生成
#     print("\n" + "=" * 50)
#     print("【测试2】Embeddings 向量生成")
#     try:
#         vec = lc_embeddings.embed_query("测试文本")
#         print(f"向量长度: {len(vec)}")
#         print(f"向量前5个值: {vec[:5]}")
#     except Exception as e:
#         print("向量测试需要配置API Key，正常")
        
# # 测试3：检索器 & 提示词模板
# print("\n" + "=" * 50)
# print("【测试3】检索器 & 提示词模板")
# retriever = get_retriever()
# prompt = get_prompt_template()
# print("[OK] 检索器创建成功")
# print("[OK] 提示词模板创建成功")
# print("模板变量:", prompt.input_variables)  # 打印模板需要的变量

# # =========================================
# # 【测试4：重排功能测试】
# # =========================================
# print("\n" + "=" * 50)
# print("【测试4】带重排的检索器")
# try:
#     # 1.创建带重排的检索器
#     compression_retriever = get_compression_retriever(top_k=3)
#     print("[OK] 带重排的检索器创建成功")
    
#     # 2.测试查询
#     test_query = "什么是人工智能?"
#     print(f"查询:{test_query}")
    
#     # 3.执行检索 + 重排
#     reranked_docs = compression_retriever.invoke(test_query)
#     print(f"[OK] 重排完成，返回 {len(reranked_docs)} 条结果\n")
    
#     # 4.打印结果
#     for i,doc in enumerate(reranked_docs):
#         print(f"【重排结果 {i+1}】")
#         print(f"内容：{doc.page_content[:80]}...")
#         print(f"来源：{doc.metadata.get('source', '无')}\n")
        
# except Exception as e:
#     logger.error(f"重排测试出错：{e}")
#     print(f"[ERROR] 重排测试出错：{e}")
#     print("[TIP] 提示：先插入文档到向量库，再测重排！")

