# 1.导包 + 加载配置
from dotenv import load_dotenv
import os
import chromadb
from dashscope import TextEmbedding
from utils.config import settings

# 加载密钥
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

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
# 读取知识库
# ==========================
def load_knowledge(file_path="knowledge.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================
# 【修复 2】正确入库向量库
# ==========================
def save_to_chroma(chunks):
    print("🔥 ingest 写入的路径：", settings.chroma_db_path)
    client = chromadb.PersistentClient(path=settings.chroma_db_path)

    # 先删掉旧集合（保证维度正确）
    try:
        client.delete_collection("knowledge_base")
        print("🗑️  旧集合已删除，重新创建")
    except:
        pass

    # 创建新集合
    coll = client.create_collection(name="knowledge_base")

    # 批量入库
    for idx, chunk in enumerate(chunks):
        try:
            resp = TextEmbedding.call(
                api_key=api_key,
                model="text-embedding-v1",
                input=chunk
            )
            emb = resp.output["embeddings"][0]["embedding"]

            coll.add(
                embeddings=[emb],
                documents=[chunk],
                ids=[f"doc_{idx}"],
                metadatas=[{"source": "knowledge.txt"}]
            )
            print(f"✅ 已入库第 {idx+1} 块")
        except Exception as e:
            print(f"❌ 入库失败: {e}")

    print(f"\n🎉 全部入库完成！共 {len(chunks)} 块")


# ==========================
# 运行入口
# ==========================
if __name__ == "__main__":
    text = load_knowledge()
    chunks = split_text(text)
    save_to_chroma(chunks)