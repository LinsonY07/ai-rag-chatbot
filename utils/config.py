# 配置读取工具
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    """
    项目全局配置中心
    所有密钥、模型参数、业务配置统一管理，一处修改，全局生效
    """
    # ===================== 1. 通义千问模型配置 =====================
    dashscope_api_key : str = os.getenv("DASHSCOPE_API_KEY")
    embedding_model: str = "text-embedding-v1"
    llm_model: str = "qwen-turbo"

    # ===================== 2. RAG 系统配置 =====================
    # 向量数据库配置
    chroma_db_path: str = "./chroma_db"
    # 文本分块配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 检索与重排配置
    recall_top_k: int = 3  
    rerank_top_k: int = 2
    # 知识库路径配置
    knowledge_dir: str = "./knowledge"

    # ===================== 3. 多轮对话记忆配置 =====================
    max_chat_history_round: int = 6     # 单会话最大保留历史轮数
    session_expire_minutes: int = 120   # 会话过期时间（分钟）


    # ===================== 4. FastAPI 服务配置 =====================
    host: str = "127.0.0.1"
    port: int = 8000

    # pydantic 配置，增强健壮性
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"    # 忽略 .env 中未定义的变量，避免报错

# 全局唯一配置实例，项目中的所有文件都导入这一个实例
settings = Settings()