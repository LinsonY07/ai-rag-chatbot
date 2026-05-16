import sqlite3
import os
from utils.logger import logger

# 数据库文件路径配置
DB_PATH = "conversation.db"

def get_db_connection():
    """
    统一获取数据库连接
    所有地方都用这个函数，不要自己写 connect
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败：{e}")
        return None
    
def init_chat_db():
    """初始化对话历史表"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        conn.commit()
        logger.info("✅ 对话历史表初始化完成")
    except Exception as e:
        logger.error(f"初始化聊天数据库失败：{e}")
    finally:
        conn.close()
