# 支持多文件遍历
import os
from utils.parsers import FileParser
from utils.logger import logger

def load_all_knowledge_files(folder_path: str = "knowledge"):
    """
    遍历knowledge文件夹，支持所有格式文件
    返回：列表，每个元素是（文件来源名，文件全文）
    """

    file_list = []
    # 如果文件夹不存在，直接返回空
    if not os.path.exists(folder_path):
        return file_list
    
    # 遍历文件夹里所有文件
    for filename in os.listdir(folder_path):
        file_full_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_full_path):  #只处理文件不处理文件夹
            continue
            
        try:
            content = FileParser.parse(file_full_path)

            if content:
                # 把（文件名，文件内容）打包，放进列表里
                file_list.append((filename, content.strip()))

        except Exception as e:
                print(f"读取文件{filename} 失败：{e}")
    return file_list

