# 主入口文件
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from utils.config import settings
from utils.logger import logger

app = FastAPI(title="RAG智能问答系统")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(router)

# 挂载前端页面
app.mount("/", StaticFiles(directory="./", html=True), name="static")

# 启动事件
@app.on_event("startup")
def startup_event():
    logger.info("=" * 50)
    logger.info("RAG 智能问答系统启动成功！")
    logger.info(f"访问地址: http://{settings.host}:{settings.port}/index.html")
    logger.info("=" * 50)