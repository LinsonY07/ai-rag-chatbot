# 单独封装日志功能（统一管理项目日志，方便排查问题）
import logging
from logging.handlers import TimedRotatingFileHandler   #按时间切割日志
import os

def setup_logger(name: str = "rag_project"):
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler(防止日志重复打印)
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s"
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ========================
    # 升级：自动创建 logs 文件夹
    # ========================
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 按天分割日志，保留 7 天
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "rag.log"),
        when="D",        # 按天切割
        interval=1,
        backupCount=7,   # 只保留最近7天日志
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 全局日志对象，所有文件都可以导入使用
logger = setup_logger()