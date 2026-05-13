import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import time
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import os
import chromadb
from dashscope import Generation, TextEmbedding
from fastapi.middleware.cors import CORSMiddleware
import asyncio


# 1. 加载配置
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 2. 初始化 FastAPI
app = FastAPI(title="RAG智能体问答接口", version="1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 创建logger对象
logger = logging.getLogger(__name__)

# 对话历史存储（单用户简单实现）
chat_history = {}


# 3. 初始化向量数据库
client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection(name="my_knowledge")

# 4. 定义请求格式
class QuestionRquest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户提问内容")
    session_id :str = Field(default="default",description="会话id，区分不同对话")
    
    @field_validator("question")
    def question_not_blank(cls, v):
        if not v.strip():
            raise ValueError("问题不能全为空格或空字符串")
        return v

# 5. 同步的 RAG 核心逻辑（不用 acall，不会报错）
def rag_ask_sync(question, history_message = None):
    try:
        # 问题转向量（同步调用）
        resp = TextEmbedding.call(
            api_key=api_key,
            model="text-embedding-v1",
            input=question
        )
        if resp["status_code"] != 200:
            return "向量化失败"
        
        q_embedding = resp["output"]["embeddings"][0]["embedding"]

        # 检索知识库
        res = collection.query(
            query_embeddings=[q_embedding],
            n_results=2
        )
        refs = "\n".join(res["documents"][0])

        # 构建消息列表
        messages = []
        # 如果有历史对话，先加进去
        if history_message:
            messages.extend(history_message)

        # 加上当前问题和上下文
        prompt = f"""
请只根据参考内容回答，不要编造。
参考：{refs}
问题：{question}
"""
        messages.append({"role": "user", "content":prompt})
        
        # 调用大模型（同步调用）
        ans = Generation.call(
            api_key=api_key,
            model="qwen-turbo",
            messages=messages,
            result_format="message"
        )
        return ans["output"]["choices"][0]["message"]["content"]
    except Exception as e:

        logging.info(f"RAG处理失败：{str(e)}")
        return "服务器出错"
    

# 流式输出版本函数
def rag_ask_stream(question, history_message = None):
    try:
        # 向量化
        resp = TextEmbedding.call(
            api_key=api_key,
            model="text-embedding-v1",
            input=question
        )
        q_emb = resp["output"]["embeddings"][0]["embedding"]

        # 检索知识库
        res = collection.query(
            query_embeddings = [q_emb],
            n_results = 5   #检索时多拿几条
            )
         
        docs = res["documents"][0]

        # 创建两个变量（用来去重 + 过滤）
        filtered_docs = []  #最终留下的干净文档
        seen = set()        #记录已经出现过的文档（去重用）（集合判断内容是否重复）

        # 将问题拆分为关键词
        keywords = question.replace("?", "").split()
        
        #循环处理每一条文档
        for doc in docs:
            # 去重
            if doc in seen:
                continue
            seen.add(doc) 

            # 关键词匹配（判断文档和问题是否相关）(将相关性高的排在前面)
            if any(keyword in doc for keyword in keywords):
                filtered_docs.append(doc)
            else:
                filtered_docs.append(doc)
        
        # 只保留最相关的两条
        top_docs = filtered_docs[:2]
        refs = "\n".join(top_docs)      #拼接为最终内容

        # 拼接历史对话
        messages = []
        if history_message:
            messages.extend(history_message)

        prompt = f"""
请只根据参考内容回答问题，不要编造。
参考内容：{refs}
用户问题：{question}
"""
        messages.append({"role":"user","content":prompt})
        
        # 检查模型收到的消息是否有重复
        # print("模型收到的消息列表：", messages)
        
        # 4. 调用通义千问真实流式 API
        responses = Generation.call(
            api_key=api_key,
            model="qwen-turbo",
            messages=messages,
            result_format="message",
            stream=True  # 关键参数：开启流式输出
        )

        # 5. 循环接收每一段内容，逐字 yield（关键增量输出）
        prev_content = ""      #记录上一次的内容 
        for response in responses:
            if response.status_code == 200:
                current_content = response.output.choices[0].message.content
                # 只输出新增的部分
                if len(current_content) > len(prev_content):
                    new_part = current_content[len(prev_content):]
                    yield new_part
                    prev_content = current_content
            else:
                yield f"API调用失败：{response.message}"

    except Exception as e:
        yield f"出错了：{str(e)}"


# 全局异常捕获
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 提取第一个错误信息
    first_error = exc.errors()[0]
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": first_error["msg"],
            "data": None
        }
    )

# -------------------
# 接口 1：测试接口
# -------------------
@app.get("/")
def home():
    return {"message": "RAG接口服务已成功启动！"}

# -------------------
# 接口 2：AI问答接口（用 asyncio.to_thread 实现高性能）
# -------------------
@app.post("/ask")
async def ask_ai(req: QuestionRquest):
    
    logger.info(f"用户提问{req.question}")

    # 获取这个用户的历史对话（没有就创建）
    if req.session_id not in chat_history:
        chat_history[req.session_id] = []

    # 把同步函数放到线程池里运行，不阻塞 FastAPI 主线程,把历史传给ai
    answer = await asyncio.to_thread(rag_ask_sync, req.question, chat_history[req.session_id])

    # 保存本轮对话（用户 + AI）
    chat_history[req.session_id].append({"role":"user","content":req.question})
    chat_history[req.session_id].append({"role":"assistant","content":answer})

    # 限制最长记忆（防止爆内存）
    if len(chat_history[req.session_id]) > 20:
        chat_history[req.session_id] = chat_history[req.session_id][-20:]

    logger.info(f"AI 回答完成")

    return {
        "question": req.question,
        "answer": answer
    }

# 流式接口
@app.post("/ask_stream")
async def create_ask_stream(req:QuestionRquest):
    sid = req.session_id
    if sid not in chat_history:
        chat_history[sid] = []

    def generate():
        full_answer = ""
        try:
            # 调用函数时，只传干净的历史（还没加当前问题）
            for part in rag_ask_stream(req.question, chat_history[sid]):
                full_answer += part
                yield part
        finally:
            # 流结束后，一次性保存用户问题 + AI回答
            chat_history[sid].append({"role": "user", "content": req.question})
            chat_history[sid].append({"role": "assistant", "content": full_answer})

            if len(chat_history[sid]) > 20:
                chat_history[sid] = chat_history[sid][-20:]
            
    return StreamingResponse(generate(), media_type="text/plain")