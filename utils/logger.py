# 单独封装日志功能（统一管理项目日志，方便排查问题）
import logging

def setup_logger(name: str = "rag_project"):
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler(防止日志重复打印)
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（保存到 rag.log）
    file_handler = logging.FileHandler("rag.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 全局日志对象，所有文件都可以导入使用
logger = setup_logger()