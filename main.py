# 主入口文件
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from utils.config import settings
from utils.logger import logger
from core.rag import process_single_file
from db.chroma_client import get_chroma_collection
from core.conversation import ConversationMemory
from contextlib import asynccontextmanager

app = FastAPI(title="RAG智能问答系统")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 【新增】文档上传接口 ====================
# 确保上传文件夹存在
UPLOAD_DIR = "upload_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 定义上传接口
@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    try:
        #1.保存文件到本地
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:    #wb:以二进制写入模式打开
            shutil.copyfileobj(file.file, f)    #将前端上传的文件流复制到本地文件 

        # 2.调用 RAG 处理函数：解析 + 分块 + 入库
        process_single_file(file_path, get_chroma_collection())
        
        # 返回结果
        return {
            "status": "success",
            "message": f"文件 {file.filename} 上传并导入知识库成功！"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败：{str(e)}")
    
    
# ==================== 【新增】获取已上传文件列表 ====================
@app.get("/api/files")
async def get_uploaded_files():
    upload_dir = "upload_files"
    # 如果文件夹不存在，直接返回空列表
    if not os.path.exists(upload_dir):
        return {"files": []}
    files = os.listdir(upload_dir)  # 获取文件夹里的所有内容
    files = [f for f in files if os.path.isfile(os.path.join(upload_dir, f))]  # 只保留文件，过滤掉文件夹、隐藏文件
    return {"files": files} # 返回文件列表给前端

# ==================== 【新增】删除单个文件 ====================
@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    upload_dir = "upload_files"
    file_path = os.path.join(upload_dir, filename)
    
    # 如果文件不存在，返回 404 错误
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    os.remove(file_path)    #删除本地文件（真正删掉硬盘里的文件）
    
    # 同步删除向量库中该文件的数据
    try:
        # 连接数据库
        vector_db = get_chroma_collection()
        # 查询 metadata 里 source 等于文件名的数据
        results = vector_db.get(where={"source": filename})
        # 如果有数据，就删除
        if results["ids"]:
            vector_db.delete(ids=results["ids"])
    except Exception as e:
        logger.error(f"删除向量库数据失败:{e}")
        pass
    
    return {"message":f"文件{filename}已删除"}

# ==================== 【新增】清空所有文件 ====================
@app.delete("/api/files")
async def clear_all_files():
    upload_dir = "upload_files"
    
    # 删除本地所有文件
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
    # 清空向量库所有数据
    try:
        vector_db = get_chroma_collection()   #连接数据库
        all_ids = vector_db.get()["ids"]
        if all_ids:
            vector_db.delete(ids=all_ids)
    except Exception as e:
        logger.error(f"清空向量库失败:{e}")
        pass
    
    return {"message": "所有文件已清空"}


# ==================== 获取对话历史 ====================
@app.get("/api/history")
async def get_api_history(session_id: str = "default"):
     memory = ConversationMemory(session_id=session_id)
     return {"history": memory.get_history()}
     


# ==================== 挂载路由 ==================== 
app.include_router(router)

# ==================== 挂载前端页面 ==================== 
app.mount("/", StaticFiles(directory="./", html=True), name="static")

# ==================== 启动事件 ==================== 

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("RAG 智能问答系统启动成功！")
    logger.info(f"访问地址: http://{settings.host}:{settings.port}/index.html")
    logger.info("=" * 50)