import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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


# 3. 初始化向量数据库
client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection(name="my_knowledge")

# 4. 定义请求格式
class QuestionRquest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户提问内容")
    @field_validator("question")
    def question_not_blank(cls, v):
        if not v.strip():
            raise ValueError("问题不能全为空格或空字符串")
        return v

# 5. 同步的 RAG 核心逻辑（不用 acall，不会报错）
def rag_ask_sync(question):
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
        
        prompt = f"""
请只根据参考内容回答，不要编造。
参考：{refs}
问题：{question}
"""
        
        # 调用大模型（同步调用）
        ans = Generation.call(
            api_key=api_key,
            model="qwen-turbo",
            messages=[{"role":"user", "content":prompt}],
            result_format="message"
        )
        return ans["output"]["choices"][0]["message"]["content"]
    except Exception as e:

        logging.info(f"RAG处理失败：{str(e)}")
        return "服务器出错"

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

    # 把同步函数放到线程池里运行，不阻塞 FastAPI 主线程
    answer = await asyncio.to_thread(rag_ask_sync, req.question)

    logger.info(f"AI 回答完成")

    return {
        "question": req.question,
        "answer": answer
    }