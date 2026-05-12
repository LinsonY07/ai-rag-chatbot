# 1.导入库
from fastapi import FastAPI     #做接口
from dotenv import load_dotenv  #读密钥
import os                       #读环境变量
import dashscope                #调用通义千问
from dashscope import Generation

# 2.加载环境变量（企业级安全规范）(作用；代码不会暴露在代码里)
load_dotenv()   #加载密钥
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  #获取密钥


# 3.创建FastAPI应用
app = FastAPI(title="AI多轮聊天机器人服务")

#4.定义全局变量(记忆功能) 
# 用户信息
Chat_profile = {
    "name":"",
    "age":"",
    "gender":"",
    "hobby":""
}
# 聊天记录
chat_history = []

# 5.编写接口装饰器 + 接收参数
#聊天接口
@app.get("/chat")
@app.get("/chat")
def chat(msg: str):
    global chat_history

    chat_history.append({"role": "user", "content": msg})

    info_str = ""
    if Chat_profile["name"]:
        info_str += f"用户名字：{Chat_profile['name']}。"
    if Chat_profile["age"]:
        info_str += f"年龄：{Chat_profile['age']}岁。"
    if Chat_profile["gender"]:
        info_str += f"性别：{Chat_profile['gender']}。"
    if Chat_profile["hobby"]:
        info_str += f"爱好：{Chat_profile['hobby']}。"

    system_prompt = f"""
    你是友好的AI助手，记住用户信息和聊天历史，回答简短自然。
    用户信息：{info_str}
    """

    messages = [{"role": "system", "content": system_prompt}] + chat_history

    try:
        response = Generation.call(
            model="qwen-turbo",
            messages=messages,
            result_format='message'
        )

        if response.status_code == 200:
            ai_reply = response.output.choices[0].message.content
            chat_history.append({"role": "assistant", "content": ai_reply})
            return {
                "user_msg": msg,
                "ai_reply": ai_reply
            }
        else:
            return {"error": "AI调用失败"}

    except Exception as e:
        print(e)
        return {"error": "服务器异常"}