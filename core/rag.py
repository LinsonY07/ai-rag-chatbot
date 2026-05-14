# 存放RAG核心逻辑（存放检索、伪重排、prompt拼接所有核心业务）
from utils.config import settings
from utils.logger import logger
from db.chroma_client import get_chroma_collection
from dashscope import TextEmbedding

# 伪重排函数
def pseudo_rerank(query: str, docs: list, top_n : int = 2):
    try:
        # 防御：空列表直接返回
        if not docs or not isinstance(docs, list):
            return []
        
        # 去重
        seen = set()
        unique_docs = []
        for doc in docs:
            if isinstance(doc, dict) and "content" in doc:  #先判断字典和 content 存在
                content = doc["content"].strip()
                if content not in seen:
                    seen.add(content)
                    unique_docs.append(doc)

        # 关键词匹配
        scored = []
        for idx,doc in enumerate(unique_docs):
            score = 0
            for word in query.split():
                if word in doc["content"]:
                    score += 1
            scored.append((-score, idx, doc))  # Python 默认从小到大排序(实现 高分在前！)
        
        scored.sort() #排序
        # 返回时对应取 doc
        return [doc for (_, _, doc) in scored[:top_n]]
    except Exception as e:
        logger.error(f"重排失败：{e}")
        # 异常兜底：返回前几条，不直接崩
        return docs[:top_n] if isinstance(docs, list) else []
    
# 向量检索 + 伪重排
def retrieve_relevant_docs(query: str):
    # 防御：空问题直接返回空
    if not query or not query.strip():
        logger.warning("用户提问为空")
        return []

    try:
        collection = get_chroma_collection()
        # 1.向量检索
        resp = TextEmbedding.call(
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
            input=query
        )
        query_embedding = resp.output["embeddings"][0]["embedding"]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.recall_top_k
        )

        documents = results["documents"][0] if results["documents"] else [] #文本
        metadatas = results["metadatas"][0] if results["metadatas"] else [] # 来源信息
        docs = []
        for doc, meta in zip(documents, metadatas):
        # 过滤掉 None 或空字符串,并判断doc和meta的类型
            if doc and meta and isinstance(doc, str) and isinstance(meta, dict):
                docs.append({
                    "content": doc,
                    "source": meta.get("source", "未知文档")
                })
        logger.info(f"检索到的原始文档数{len(docs)}")
        
        # 2.伪重排
        finally_docs = pseudo_rerank(
            query,
            docs,
            top_n=settings.rerank_top_k
        )
        logger.info(f"重排后的最终文档数：{len(finally_docs)}")
        return finally_docs
    except Exception as e:
        logger.error(f"向量检索异常：{e}")
        # 接口失败兜底返回空
        return []
    

# 拼接Prompt
def build_prompt(query: str, docs: list, history: list):
    # 防御：确保是列表
    if not isinstance(docs, list):
        docs = []
    if not isinstance(history, list):
        history = []
    
    # 拼接参考上下文
    if not docs:
        context = "暂无参考资料"
    else:
        context = ""
        for doc in docs:
            # 防御：必须是字典且包含字段
            if isinstance(doc, dict) and "source" in doc and "content" in doc:
                context += f"来源：{doc['source']}\n"
                context += f"内容：{doc['content']}\n\n"

    # 拼接历史对话（简单容错）
    chat_history = "\n".join(history[-6:])  #只保留最近6轮，避免过长

    prompt = f"""
你是一个智能助手，请根据知识库内容回答用户问题，不要编造信息。

### 参考内容：
{context}

### 历史对话：
{chat_history}

### 用户问题：
{query}

请回答：
"""
    return prompt
