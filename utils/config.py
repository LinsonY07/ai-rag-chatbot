# 配置读取工具
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # 通义千问配置
    dashscope_api_key : str = os.getenv("DASHSCOPE_API_KEY")
    embedding_model: str = "text-embedding-v1"
    llm_model: str = "qwen-turbo"

    # RAG配置
    chroma_db_path: str = "./chroma_db"
    recall_top_k: int = 5  
    rerank_top_k: int = 2

    # 服务配置
    host: str = "127.0.0.1"
    port: int = 8000

settings = Settings()