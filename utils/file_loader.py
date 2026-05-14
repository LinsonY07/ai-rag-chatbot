# 支持多文件遍历
import os

def load_all_knowledge_files(folder_path: str = "knowledge"):
    """
    遍历knowledge文件夹，返回所有txt/md文件
    返回：列表，每个元素是（文件来源名，文件全文）
    """

    file_list = []
    # 如果文件夹不存在，直接返回空
    if not os.path.exists(folder_path):
        return file_list
    
    # 遍历文件夹里所有文件
    for filename in os.listdir(folder_path):
        # 只处理 txt 和 md
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_full_path = os.path.join(folder_path, filename)    #将文件夹和文件名拼接在一起，形成完整路径
            
            try:
                with open(file_full_path,"r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    # 来源就是文件名
                    file_list.append((filename, content))

            except Exception as e:
                print(f"读取文件{filename} 失败：{e}")
    return file_list


# import os

# def load_all_knowledge_files(folder_path: str = "knowledge"):
#     """
#     遍历knowledge文件夹，读取所有txt/md文件
#     返回：列表，每个元素是 (文件来源名, 文件全文)
#     """
#     file_list = []
#     # 如果文件夹不存在，直接返回空
#     if not os.path.exists(folder_path):
#         print(f"❌ 文件夹不存在：{folder_path}")
#         return file_list

#     print(f"📂 遍历文件夹：{folder_path}")
#     print(f"📋 文件夹内所有文件：{os.listdir(folder_path)}")

#     # 遍历文件夹里所有文件
#     for filename in os.listdir(folder_path):
#         print(f"\n🔍 正在检查文件：{filename}")
        
#         # 只处理 txt 和 md
#         if filename.endswith(".txt") or filename.endswith(".md"):
#             print(f"✅ 后缀符合要求：{filename}")
#             file_full_path = os.path.join(folder_path, filename)
#             print(f"📁 文件完整路径：{file_full_path}")
            
#             try:
#                 with open(file_full_path, "r", encoding="utf-8") as f:
#                     content = f.read().strip()
#                 print(f"📄 读取成功，内容长度：{len(content)}")
                
#                 if content:
#                     # 来源就是文件名
#                     file_list.append((filename, content))
#                     print(f"➕ 已添加到列表：{filename}")
#                 else:
#                     print(f"⚠️ 文件内容为空，跳过：{filename}")
            
#             except Exception as e:
#                 print(f"❌ 读取文件 {filename} 失败：{e}")
#         else:
#             print(f"⚠️ 后缀不符合要求，跳过：{filename}")
    
#     print(f"\n📊 最终读取到的文件列表：{[name for name, _ in file_list]}")
#     return file_list