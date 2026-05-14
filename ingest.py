# 1.导包 + 加载配置
from dotenv import load_dotenv
import os
import chromadb
from dashscope import TextEmbedding
from utils.config import settings
from utils.file_loader import load_all_knowledge_files



# ==========================
# 【修复 1】文本分块函数（正确循环）
# ==========================
def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    total_len = len(text)

    # 循环切分，直到文本末尾
    while start < total_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 移动到下一个位置（带重叠）

    return chunks  # 循环结束才 return！！！


# ==========================
# 【修复 2】正确入库向量库
# ==========================
def save_to_chroma():
    client = chromadb.PersistentClient(path=settings.chroma_db_path)

    # 先删掉旧集合（保证维度正确）
    try:
        client.delete_collection("knowledge_base")
        print("🗑️  旧集合已删除，重新创建")
    except:
        pass

    # 创建新集合
    coll = client.create_collection(name="knowledge_base")

    # =====================
    # 【核心：读取所有文件】
    # =====================
    file_list = load_all_knowledge_files()  #读取 konwledge/ 下所有的txt/md
    if not file_list:
        print("❌ knowledge 文件夹下没有可读取的文件！")
        return
    
    global_idx = 0  #全局唯一ID，防止重复

    # 循环处理每个文件
    for file_name, file_content in file_list:
        print(f"\n📄 正在处理文件：{file_name}")

        # 对当前文件进行分块
        chunks = split_text(file_content)

        # 分块入库
        for chunk in chunks:
            try:
                # 向量化
                resp = TextEmbedding.call(
                    api_key=settings.dashscope_api_key,
                    model = settings.embedding_model,
                    input=chunk
                )
                emb = resp.output["embeddings"][0]["embedding"]

                # 【关键：来源 = 文件名】
                coll.add(
                    embeddings=[emb],
                    documents=[chunk],
                    ids=[f"doc_{global_idx}"],
                    metadatas=[{"source": file_name}]   #自动用文件名当来源
                )

                global_idx += 1
                print(f"✅ 已入库片段 {global_idx}")

            except Exception as e:
                print(f"❌ 片段入库失败: {e}")
            
        print(f"\n🎉 所有文件入库完成！总片段数：{global_idx}")

# ==========================
# 运行入口
# ==========================
if __name__ == "__main__":
    save_to_chroma()  # 直接调用，内部自动读取所有文件