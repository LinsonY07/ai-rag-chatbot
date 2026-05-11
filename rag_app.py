# # 1.导入库 + 加载环境变量
# from dotenv import load_dotenv  #读密钥
# import os  #读环境变量
# import chromadb  #向量数据库
# from dashscope import Generation,TextEmbedding  #Generation：大模型对话；TextEmbedding：文本转向量

# # 2.加载环境变量
# load_dotenv()
# api_key = os.getenv("DASHSCOPE_API_KEY")

# print(api_key)
# # 初始化向量数据库
# client = chromadb.PersistentClient(path="./chroma_db") #PersistentClient:持久化存储，重启不丢数据
# # 创建/获取集合
# collection = client.get_or_create_collection(name="my_knowledge") #collection：就是存放文本向量的“表”

# # 3.读取本地知识库文件
# def load_knowledge(file_path="knowledge.txt"):
#     with open(file_path,"r", encoding="utf-8") as f:
#         return f.read()
    
# text = load_knowledge()
# print("知识库加载完成")

# # 4.文本简单分块 + 转向量存入Chroma
# # 简单换行分块
# chunks = [c.strip() for c in text.split("/n") if c.split()]

# # 向量化并存入
# for idx, chunk in enumerate(chunks):
#     # 文本转向量
#     resp = TextEmbedding.call(
#         api_key=api_key,
#         model="text-embedding-v1",
#         input=chunk
#     )
#     embedding = resp["output"]["embeddings"][0]["embedding"]

#     # 存入向量库
#     collection.add(
#         documents=[chunk],
#         embeddings=[embedding],
#         ids=[str(idx)]
#     )

#     # 5.检索 + 问答函数
#     def rag_ask(question):
#         # 问题转向量
#         resp = TextEmbedding.call(
#             api_key=api_key,
#             model="text-embedding-v1",
#             input=question
#         )
#         q_embedding = resp["output"]["embeddings"][0]["embedding"]
        
#         # 相似度检索
#         res = collection.query(
#             query_embeddings=[q_embedding],
#             n_results=2
#         )
#         refs = "\n".join(res["documents"][0])

#         # 构造提示词
#         prompt = f"""
#     请只根据下面的内容回答问题，不要编造。
#     参考内容：
#     {refs}
#     问题：{question}

# """
#         # 调用大模型生成答案
#         ans = Generation.call(
#             api_key=api_key,
#             model="qwen-turbo",
#             messages=[{"role":"user","content":prompt}],
#             result_format="message"
#         )
#         return ans["output"]["choices"][0]["message"]["content"]
#     # 6.循环测试入口
#     if __name__ == "__main__":
#         while True:
#             q = input("请输入问题:")
#             if q == "exit":
#                 break
#             a = rag_ask(q)
#             print("AI回答：", a)


from dotenv import load_dotenv
import os
import chromadb
from dashscope import Generation, TextEmbedding

# --------------------------
# 1. 加载环境变量
# --------------------------
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
print("✅ API Key 已加载")

# --------------------------
# 2. 初始化向量数据库
# --------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_knowledge")

# --------------------------
# 3. 读取本地知识库文件
# --------------------------
def load_knowledge(file_path="knowledge.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

text = load_knowledge()
print("✅ 知识库加载完成")

# --------------------------
# 4. 文本分块
# --------------------------
chunks = [c.strip() for c in text.split("\n") if c.strip()]
print(f"✅ 共切分 {len(chunks)} 个文本块")

# --------------------------
# 5. 向量化并存入向量库（修复版）
# --------------------------
for idx, chunk in enumerate(chunks):
    print(f"正在向量化第 {idx+1}/{len(chunks)} 块...")
    try:
        # 文本转向量
        resp = TextEmbedding.call(
            api_key=api_key,
            model="text-embedding-v1",
            input=chunk
        )
        
        # 检查API返回状态
        if resp["status_code"] != 200:
            print(f"❌ 向量化失败，状态码：{resp['status_code']}, 响应：{resp}")
            continue
        
        # 【关键修复】直接从字典中获取向量
        embedding = resp["output"]["embeddings"][0]["embedding"]
        
        # 存入向量库
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(idx)]
        )
        print(f"✅ 第 {idx+1} 块存入成功")
        
    except Exception as e:
        print(f"❌ 第 {idx+1} 块向量化异常：{e}")
        print("完整响应：", resp)

print("✅ 全部文本已向量化存入向量数据库")

# --------------------------
# 6. RAG问答函数（修复版）
# --------------------------
def rag_ask(question):
    try:
        # 问题转向量
        resp = TextEmbedding.call(
            api_key=api_key,
            model="text-embedding-v1",
            input=question
        )
        if resp["status_code"] != 200:
            return "❌ 问题向量化失败"
        q_embedding = resp["output"]["embeddings"][0]["embedding"]

        # 相似度检索
        res = collection.query(
            query_embeddings=[q_embedding],
            n_results=2
        )
        refs = "\n".join(res["documents"][0])

        # 构造提示词
        prompt = f"""
请只根据下面的参考内容回答问题，不要编造。
参考内容：
{refs}
问题：{question}
"""

        # 调用大模型
        ans = Generation.call(
            api_key=api_key,
            model="qwen-turbo",
            messages=[{"role":"user","content":prompt}],
            result_format="message"
        )
        if ans["status_code"] != 200:
            return "❌ 大模型调用失败"
        return ans["output"]["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ 问答过程异常：", e)
        return "❌ 问答服务异常"

# --------------------------
# 7. 测试入口
# --------------------------
if __name__ == "__main__":
    while True:
        q = input("请输入问题：")
        if q == "exit":
            break
        a = rag_ask(q)
        print("AI回答：", a)