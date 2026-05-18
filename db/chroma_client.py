# 存放向量数据库连接（统一管理Chroma连接）
import chromadb
from utils.config import settings
from utils.logger import logger

# 全局唯一的 Chroma 客户端
# 项目运行期间， 只创建一次
# 内存模式，重启服务就清空向量库，速度飞快
client = chromadb.PersistentClient(path=settings.chroma_db_path)

# 获取集合（表）
def get_chroma_collection():
    try:
        print("🔥 服务读取的数据库路径：", settings.chroma_db_path)
        collection = client.get_collection(name="knowledge_base")
        logger.info("成功加载向量数据库集合")
        return collection
    except Exception as e:
        logger.error(f"向量数据库加载失败：{e}")
        raise