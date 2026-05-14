# 存放向量数据库连接（统一管理Chroma连接）
import chromadb
from utils.config import settings
from utils.logger import logger

# 全局唯一的 Chroma 客户端
# 项目运行期间， 只创建一次
client = chromadb.PersistentClient(path=settings.chroma_db_path)

# 获取集合（表）
def get_chroma_collection():
    try:
        collection = client.get_collection(name="knowledge_base")
        logger.info("成功加载向量数据库集合")
        return collection
    except Exception as e:
        logger.error(f"向量数据库加载失败：{e}")
        raise