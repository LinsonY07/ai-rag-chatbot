# 存放RAG核心逻辑（存放检索、重排、prompt拼接所有核心业务）
from utils.config import settings
from utils.logger import logger
from db.chroma_client import get_chroma_collection
from dashscope import TextEmbedding
import requests     #调用阿里Rerank

# 真实 Rerank 重排
def real_rerank(query: str, docs: list, top_n: int = 2) -> list:  
    """
    阿里DashScope 真实重排接口
    功能：调用阿里gte-rerank-v2真实重排接口
    机制：参数校验→接口请求→异常捕获→失败自动降级伪重排
    """
    # 基础合法性校验
    if not docs or not query.strip():
            logger.info("真实重排：无效问题或文档，直接降级伪重排")
            return pseudo_rerank(query, docs, top_n)
    
    # 清洗有效文档内容，过滤空内容
    contents = []
    for doc in docs:
        content = doc.get("content", "").strip()
        if content:
            contents.append(content)
    # 无效文本，直接降级
    if not contents:
        logger.info("真实重排：清洗后无有效文本内容，降级伪重排")
        return pseudo_rerank(query, docs, top_n)

    # 阿里 Rerank 接口地址
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json"
    }

    req_data = {
        "model": "gte-rerank-v2",
        "input": {
            "query": query,
            "documents": contents
        },
        "parameters": {
            "return_documents": True,
            "top_n": top_n
        }
    }

    try:
        # 打印请求详情，方便排查
        # logger.debug(f"真实重排请求体: {req_data}")
        
        # 发送请求,设置超时防止卡死
        response = requests.post(url, headers=headers, json=req_data, timeout=8)
        
        # 打印响应详情
        # logger.info(f"真实重排响应码: {response.status_code}")
        # logger.info(f"真实重排响应体: {response.text}")
        
        # 状态码非200直接降级
        if response.status_code != 200:
            logger.error(f"真实重排：接口状态码异常{response.status_code},降级伪重排")
            return pseudo_rerank(query, docs, top_n)
        
        res_json = response.json()
        # 校验返回结构是否合法
        if "output" not in res_json or "results" not in res_json["output"]:
            logger.error("真实重排：返回数据结构异常，降级伪重排")
            return pseudo_rerank(query, docs, top_n)
        
        # 按重排后的下标重新整理文档
        ranked_docs = []
        # 增加一个相关性阈值，过滤掉低相关性的噪声文档
        MIN_RELEVANCE_SCORE = 0.01  # 可以根据效果调整，比如 0.05, 0.1
        for item in res_json["output"]["results"]:
            score = item["relevance_score"]
            idx = item["index"]
            # 只保留相关性分数 >  设定阈值的文档
            if score >  MIN_RELEVANCE_SCORE:
                ranked_docs.append(docs[idx])

        logger.info(f"✅ 真实重排完成 | 有效相关文档：{len(ranked_docs)} 个")
        return ranked_docs
    except Exception as e:
        # 捕获所有异常：网络超时、参数错误、密钥错误等
        logger.error(f"真实重排调用异常：{str(e)}，自动降级伪重排")
        return pseudo_rerank(query, docs, top_n)
       

# 伪重排函数
def pseudo_rerank(query: str, docs: list, top_n : int = 2) -> list:
    """
    功能：关键词打分 + 文档去重 + 相关性排序
    规则：只排高低，不强制过滤低分文档，防止误伤知识库已有问题
    """
    try:
        # 防御：空列表直接返回
        if not docs or not isinstance(docs, list):
            return []
        
         # 1. 文档去重：避免重复内容占用召回名额
        seen_content = set()
        unique_docs = []
        for doc in docs:
            content = doc.get("content", "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        # 2. 关键词分词匹配打分
        scored = []
        
        # 对查询词简单清洗，去掉无意义符号
        clean_query = query.replace("？", "?").replace("，", ",").replace("。", ".")
        query_words = [w for w in clean_query.split() if len(w) >= 2]  # 过滤太短的词

        
        for idx, doc in enumerate(unique_docs):
            score = 0
            # 按空格拆分问题关键词
            for word in query_words:
                if word in doc["content"]:
                    score += 1
            # 负分用于升序排序，实现高分在前
            scored.append((-score, idx, doc))

        # 3. 按相关性分数排序
        scored.sort()
        
        # 4. 【关键强过滤】只保留关键词匹配分数 > 0 的文档
        # 不匹配 → 直接不要，彻底杜绝无关文档进入prompt
        valid_docs = []
        for neg_score, idx, doc in scored:
            score = -neg_score
            if score > 0:
                valid_docs.append(doc)

        # 最终返回前 top_n 个有效文档
        final_docs = valid_docs[:top_n]
        logger.info(f"✅ 伪重排完成 | 匹配有效文档：{len(final_docs)} 个")
        return final_docs

    except Exception as e:
        logger.error(f"伪重排异常：{e}")
        return []
    
# 向量检索 + 重排
def retrieve_relevant_docs(query: str):
    """向量召回 -> 进入重排流程"""
    if not query.strip():
        logger.warning("用户提问为空")
        return []

    try:
        # 1. 获取向量数据库连接
        collection = get_chroma_collection()
        # 2. 文本向量化
        emb_resp = TextEmbedding.call(
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
            input=query
        )
        query_emb = emb_resp.output["embeddings"][0]["embedding"]

        # 3. 向量相似度召回
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=settings.recall_top_k
        )

        # 4. 解析召回结果
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []

        # 5. 组装成统一文档结构：content+source
        docs = []
        for doc_text, meta in zip(documents, metadatas):
            if not doc_text or not meta:
                continue
            
            # 确保 source 是干净的字符串
            source = meta.get("source","未知文档")
            
            #防止 meta 里的 source 还是字典类型，再取一层
            if isinstance(source, dict):
                source = source.get("source", "未知文档")
                
            # 去除多余的路径，只保留文件名
            source = source.split("/")[-1].split("\\")[-1]
            
            docs.append({
                "content": doc_text.strip(),
                "source": source.strip()
            })

        # 6. 进入重排（优先真实重排，失败自动走伪重排）
        return real_rerank(query, docs, settings.rerank_top_k)

    except Exception as e:
        logger.error(f"向量检索异常：{str(e)}")
        return []


# 拼接Prompt
def build_prompt(query: str, docs: list, history: list):
    """
    防幻觉核心逻辑：
    1. 无任何召回文档 → 强制固定话术，禁止编造
    2. 有文档 → 严格按参考资料回答，禁止延伸未知内容
    """
    # 1. 完全没有知识库文档，直接锁死回答，杜绝幻觉
    if not docs:
        return f"""
你是专业智能问答助手，必须严格遵守规则：
1. 当前没有任何参考资料可用
2. 一律固定回复：暂无相关信息，无法回答
3. 绝对不能编造、猜测、延伸任何内容

用户问题：{query}
"""

    # 2. 拼接参考资料
    context = ""
    for doc in docs:
        context += f"【参考来源】{doc['source']}\n【参考内容】{doc['content']}\n\n"

    # 3. 拼接历史对话上下文
    chat_history = ""
    for msg in history[-settings.max_chat_history_round:]:
        role_name = "用户" if msg["role"] == "user" else "助手"
        chat_history += f"{role_name}：{msg['content']}\n"

    # 4. 大模型约束提示词
    prompt = f"""
你是严谨的知识库问答助手，严格执行以下规则：
1. 优先遵循历史对话上下文理解用户意图
2. 只能依据提供的参考内容进行回答
3. 如果参考内容和用户问题无关，也必须回复：暂无相关信息，无法回答
4. 参考内容里没有的信息，绝对不能编造、不能瞎猜、不能自行延伸
5. 回答简洁自然，不冗余、不啰嗦

历史对话：
{chat_history}

参考资料：
{context}

用户问题：{query}
请严格按规则回答,回答结束后，用 ###SOURCE### 分隔，列出所有参考来源
"""
    return prompt