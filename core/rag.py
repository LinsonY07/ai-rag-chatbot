# 存放RAG核心逻辑（存放检索、伪重排、prompt拼接所有核心业务）
from utils.config import settings
from utils.logger import logger
from db.chroma_client import get_chroma_collection
from dashscope import TextEmbedding

# 伪重排函数
def pseudo_rerank(query: str, docs: list, top_n : int = 2):
    try:
        # 去重
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc["content"] not in seen:
                seen.add(doc["content"])
                unique_docs.append(doc)

        # 关键词匹配
        scored = []
        for doc in unique_docs:
            score = 0
            for word in query.split():
                if word in doc["content"]:
                    score += 1
            scored.append((-score, doc))  # Python 默认从小到大排序(实现 高分在前！)
        
        scored.sort() #排序
        return [doc for (_, doc) in scored[:top_n]]
    except Exception as e:
        logger.error(f"重排失败：{e}")
        return docs[:top_n]
    
# 向量检索 + 伪重排
def retrieve_relevant_docs(query: str):
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
    # 过滤掉 None 或空字符串
        if doc and meta:
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
    

# 拼接Prompt
def build_prompt(query: str, docs: list, history: list):
    # 确保 docs 是列表，防止 None 导致 join 报错
    if not docs:
        context = "暂无参考资料"
    else:
        context = ""
        for doc in docs:
            context += f"来源：{doc['source']}\n"
            context += f"内容：{doc['content']}\n\n"

    prompt = f"""
你是一个智能助手，请根据知识库内容回答用户问题，不要编造信息。

### 参考内容：
{context}

### 历史对话：
{history}

### 用户问题：
{query}

请回答：
"""
    return prompt
