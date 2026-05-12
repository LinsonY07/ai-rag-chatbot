import random
import os
import json
import dashscope
from dashscope import Generation
from dotenv import load_dotenv
import os



# 启用Windows终端颜色
os.system("")

# 加载 .env 文件
load_dotenv()

# 从环境变量安全读取 KEY
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


# ======================
#全局变量：保存聊天历史
# ======================
chat_history = []


# ======================
# 1. 用户个人记忆（单用户）
# ======================
profile = {
    "name": None,
    "age": None,
    "hobby": None,
    "gender": None
}

# 保存记忆到本地文件
def save_profile():
    try:
        with open("profile.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 加载本地记忆
def load_profile():
    global profile
    try:
        with open("profile.json", "r", encoding="utf-8") as f:
            profile = json.load(f)
    except FileNotFoundError:
        profile = {
            "name": None,
            "age": None,
            "hobby": None,
            "gender": None
        }
    except Exception:
        profile = {
            "name": None,
            "age": None,
            "hobby": None,
            "gender": None
        }

# 程序启动自动加载
load_profile()

# ======================
# 2. 聊天历史记忆
# ======================
history = []

# 关键词对话库
reply_dict = {
    "你好": ["你好呀！今天也要加油写代码哦~", "嗨！我是你的专属AI助手 😊"],
    "你是谁": ["我是你亲手写的第一个带AI大脑的CLI聊天机器人！", "我是你的Python练手项目，已经学会调用大模型啦~"],
    "再见": ["拜拜！下次再聊 👋", "下次见，祝你学习顺利！"],
    "加油": ["一起加油！你已经迈出第一步了！", "你超棒的，继续冲！"],
    "代码": ["写代码遇到问题可以随时问我~", "别着急，一步步调试就好啦"],
    "python": ["Python超好用的，我们一起慢慢学", "你现在做的就是Python项目，超酷的！"],
    "吃饭": ["你喜欢吃什么","快去吃饭！","吃饱了才有力气写代码~"]
}

# 兜底随机回复（现在几乎用不到了）
fallback_replies = [
    "这个问题我还不会，换个话题聊聊吧~",
    "我还在学习，暂时听不懂哦 😂",
    "你可以试试问我‘你好’或者‘加油’"
]

# ======================
# 3. 新增：调用大模型的核心函数
# ======================
def call_ai_model(text):
    global chat_history  #使用全局聊天历史

    # 1.把用户最新消息加入历史
    chat_history.append({"role":"user", "content":text})
    
    # 2.拼接用户个人信息
    info_str = ""
    if profile["name"]:
        info_str += f"用户的名字叫{profile['name']}。"
    if profile["age"]:
        info_str += f"用户今年{profile['age']}岁。"
    if profile["gender"]:
        info_str += f"用户的性别是{profile['gender']}。"
    if profile["hobby"]:
        info_str += f"用户的爱好是{profile['hobby']}。"

    # 3.构造系统提示词
    system_prompt = f"""
    你是一个友好的、聪明的AI助手。
    记住用户信息{info_str}
    记住聊天历史，根据上下文回答问题。
    请用自然、简短、口语化回答
"""
    
    #4.拼接最终提示词 = 系统提示 + 全部聊天历史
    messages = [{"role":"system", "content":system_prompt}] + chat_history 


    # 5.调用通义千问大模型
    try:
        response = Generation.call(
            model="qwen-turbo",  # 使用免费的通义千问模型
            messages = messages,
            result_format='message'
        )

        if response.status_code == 200:
            ai_reply = response.output.choices[0].message.content
    #将AI的回答也加入历史 
            chat_history.append({"role":"assistant", "content":ai_reply})
            return ai_reply
        
        else:
            print(f"错误码：{response.status_code}, 错误信息：{response.message}")
            return "我暂时没法回答这个问题，换个话题吧~"

    except Exception as e:
        print("调试错误：", e)
        return "网络有点问题，稍后再问我吧~"


# ======================
# 4. 记忆录入 & 查询封装函数
# ======================
def answer_info(text, key_word, field, has_msg, empty_msg):
    if key_word in text:
        if profile[field]:
            print("\033[91mAI：", has_msg.format(profile[field]), "\033[0m", sep="")
        else:
            print("\033[91mAI：", empty_msg, "\033[0m", sep="")
        return True
    return False

def extract_info(text, key_word, field, has_msg):
    if key_word in text:
        profile[field] = text.replace(key_word, "").strip()
        print("\033[91mAI：", has_msg.format(profile[field]), "\033[0m", sep="")
        save_profile()
        return True
    return False

# ======================
# 5. 获取机器人回复（优先规则，其次AI）
# ======================
def get_robot_response(user_text):
    history.append(["用户", user_text])

    # 1. 优先匹配关键词
    for keyword, replies in reply_dict.items():
        if keyword in user_text:
            res = random.choice(replies)
            history.append(["机器人", res])
            return res

    # 2. 没匹配到关键词，交给大模型处理
    res = call_ai_model(user_text)
    history.append(["机器人", res])
    return res

# 展示对话历史
def show_history():
    print("\n📜 当前对话历史：")
    for idx, chat in enumerate(history, 1):
        role, text = chat
        print(f"  {idx}. {role}：{text}")
    print("-" * 40)

# ======================
# 6. 主程序
# ======================
def main():
    print("\033[94m===== AI大模型版聊天机器人（输入 quit 退出）=====\033[0m\n")
    while True:
        # 绿色用户输入
        user_input = input("\033[92m我：\033[0m")
        print()

        # 退出
        if user_input.strip().lower() == "quit":
            print("\033[91mAI：拜拜！下次再聊~ 👋\033[0m")
            show_history()
            break

        # 清空个人记忆
        if user_input == "清空记忆":
            profile["name"] = None
            profile["age"] = None
            profile["hobby"] = None
            profile["gender"] = None
            save_profile()
            print("\033[91mAI：已清空你的所有个人记忆～\033[0m\n")
            show_history()
            continue

        # 查看个人全部信息
        if user_input == "查看我的信息":
            print("\033[91mAI：你的个人信息如下：\033[0m")
            print(f"  姓名：{profile['name'] if profile['name'] else '未设置'}")
            print(f"  年龄：{profile['age'] if profile['age'] else '未设置'}")
            print(f"  性别：{profile['gender'] if profile['gender'] else '未设置'}")
            print(f"  爱好：{profile['hobby'] if profile['hobby'] else '未设置'}\n")
            show_history()
            continue

        # ---------- 查询个人信息 ----------
        if answer_info(user_input, "我叫什么", "name", "你叫{}", "你还没告诉我你叫什么呢~"):
            show_history()
            continue
        elif answer_info(user_input, "我几岁", "age", "你{}岁", "你还没告诉我你的年龄呢~"):
            show_history()
            continue
        elif answer_info(user_input, "我喜欢什么", "hobby", "你喜欢{}", "你还没告诉我你的爱好呢~"):
            show_history()
            continue
        elif answer_info(user_input, "我性别是什么", "gender", "你的性别是{}", "你还没告诉我你的性别呢~"):
            show_history()
            continue

        # ---------- 录入个人信息 ----------
        elif extract_info(user_input, "我叫", "name", "我记住了你的名字是{}"):
            show_history()
            continue
        elif extract_info(user_input, "我今年", "age", "我记住了你的年龄是{}岁"):
            show_history()
            continue
        elif extract_info(user_input, "我喜欢", "hobby", "我知道你喜欢{}了"):
            show_history()
            continue
        elif extract_info(user_input, "我性别是", "gender", "我知道你是{}"):
            show_history()
            continue

        # ---------- 交给大模型聊天 ----------
        else:
            res = get_robot_response(user_input)
            print(f"\033[91mAI：{res}\033[0m\n")
            show_history()

if __name__ == "__main__":
    main()