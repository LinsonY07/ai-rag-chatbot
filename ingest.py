# 1.导包 + 加载配置
from dotenv import load_dotenv       #加载密钥
import os                            #加载环境变量
import chromadb                     #向量数据库
from dashscope import TextEmbedding  #文字转向量

# 2.加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# 3.文本分块
def split_text(text, chunk_size = 300, overloap = 50):  #定义函数（指定切块长度和重叠程度）
    chunks = []  #创建空列表，存放切好的小段文字
    start = 0    #从第0个字段开始切
    while start < len(text):  #一直切到文本末尾
        end = start + chunk_size   #切完的结束位置
        chunk = text[start:end]     #截取开始到结束的一段文本
        chunks.append(chunk)        #将截取好的文本存入列表
        start = end - overloap      #重新确定开始位置（但需要重叠overlap）
        return chunks               #返回所有切好的文本块
    
# 4.读取知识库
def load_knowledge(file_path = "knowledge.txt"):
    with open (file_path, "r", encoding = "utf-8") as f:    #以只读权限打开知识库
        return f.read()     #读取全部内容，返回字符串
    
# 5.存入向量数据库
def save_to_chroma(chunks):
    client = chromadb.PersistentClient("./chroma_db")   #连接本地向量数据库
    try:
        client.delete_collection("my_knowledge")  #先删除旧的集合，避免重复数据，保证每次入库都是最新数据
    except:
        pass
    
    coll = client.get_or_create_collection(name="my_knowledge") #创建新的空集合
    
    for idx, chunk in enumerate(chunks):    #循环每一个分块，并生成对应的唯一id(idx)
        resp = TextEmbedding.call(
            api_key=api_key,
            model="text-embedding-v1",
            input=chunk
        )
        emb = resp["output"]["embeddings"][0]["embedding"]   #提取向量

        coll.add(
            embeddings=[emb],   #存入向量
            documents=[chunk],  #存入文本
            ids=[f"doc_{idx}"]  #存入特定id
        )
        print(f"✅ 成功入库 {len(chunks)}个文本块")

# 6.最后：运行入口
if __name__ == "__main__":
    text = load_knowledge()
    chunks = split_text(text)
    save_to_chroma(chunks)