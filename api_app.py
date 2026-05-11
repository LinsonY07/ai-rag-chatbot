# 1.导入需要的库
from fastapi import FastAPI     #用来做接口服务（让你的代码能被网页/app调用）
from pydantic import BaseModel  #定义前端传过来的问题格式
from dotenv import load_dotenv  #加载API key
import os                       #加载环境变量
import chromadb                 #向量数据库
from dashscope import Generation, TextEmbedding #通义大模型 + 嵌入模型
from fastapi.middleware.cors import CORSMiddleware

# 2.加载配置 + 初始化接口服务
# 加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化FastAPI应用
app = FastAPI(title="RAG智能体问答接口",version="1.0")

# 新增：添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3.连接/初始化向量数据库
client = chromadb.PersistentClient("/chroma_db")
collection = client.get_or_create_collection(name="my_knowledge")

# 4.定义请求格式（前端/外部传过来的问题）
class QuestionRquest(BaseModel):
    question: str

# 5.RAG核心代码
def rag_ask(question):
    try:
        # 问题转向量
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
        # 给大模型发提示
        prompt = f"""
    请只根据参考内容回答，不要编造。
    参考：{refs}
    问题：{question}
    """
        
        # 调用大模型
        ans = Generation.call(
            api_key=api_key,
            model="qwen-turbo",
            messages=[{"role":"user", "content":prompt}],
            result_format = "message"
        )
        return ans["output"]["choices"][0]["message"]["content"]
    except Exception as e:
        return f"出错：{str(e)}"
    
# -------------------
# 接口 1：测试接口
# -------------------
@app.get("/")
def home():
    return {"message": "RAG接口服务已成功启动！"}

# -------------------
# 接口 2：AI问答接口
# -------------------
@app.post("/ask")
def ask_ai(req: QuestionRquest):
    answer = rag_ask(req.question)
    return {
        "question": req.question,
        "answer":answer
    }
