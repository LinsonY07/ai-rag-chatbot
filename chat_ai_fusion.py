import random
import os
import json

# 启用Windows终端颜色
os.system("")

# ======================
# 1. 用户个人记忆（单用户）
# ======================
profile_fusion = {
    "name": None,
    "age": None,
    "hobby": None,
    "gender": None
}

# 保存记忆到本地文件
def save_profile():
    try:
        with open("profile_fusion.json", "w", encoding="utf-8") as f:
            json.dump(profile_fusion, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 加载本地记忆
def load_profile():
    global profile_fusion
    try:
        with open("profile_fusion.json", "r", encoding="utf-8") as f:
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
    "你是谁": ["我是你亲手写的第一个CLI聊天机器人！", "我是你的Python练手项目，还在学习中~"],
    "再见": ["拜拜！下次再聊 👋", "下次见，祝你学习顺利！"],
    "今天干嘛": ["陪你一起学Python，做AI项目呀", "帮你解决代码问题，陪你聊天！"],
    "加油": ["一起加油！你已经迈出第一步了！", "你超棒的，继续冲！"],
    "代码": ["写代码遇到问题可以随时问我~", "别着急，一步步调试就好啦"],
    "python": ["Python超好用的，我们一起慢慢学", "你现在做的就是Python项目，超酷的！"],
    "吃饭": ["你喜欢吃什么","快去吃饭！","吃饱了才有力气写代码~"]
}

# 兜底随机回复
fallback_replies = [
    "这个问题我还不会，换个话题聊聊吧~",
    "我还在学习，暂时听不懂哦 😂",
    "你可以试试问我‘你好’或者‘加油’"
]

# ======================
# 3. 记忆录入 & 查询封装函数
# ======================
def answer_info(text, key_word, field, has_msg, empty_msg):
    if key_word in text:
        if profile_fusion[field]:
            print("\033[91mAI：", has_msg.format(profile_fusion[field]), "\033[0m", sep="")
        else:
            print("\033[91mAI：", empty_msg, "\033[0m", sep="")
        return True
    return False

def extract_info(text, key_word, field, has_msg):
    if key_word in text:
        profile_fusion[field] = text.replace(key_word, "").strip()
        print("\033[91mAI：", has_msg.format(profile_fusion[field]), "\033[0m", sep="")
        save_profile()
        return True
    return False

# ======================
# 4. 获取机器人普通聊天回复
# ======================
def get_robot_response(user_text):
    history.append(["用户", user_text])

    # 优先关键词匹配
    for keyword, replies in reply_dict.items():
        if keyword in user_text:
            res = random.choice(replies)
            history.append(["机器人", res])
            return res

    # 兜底回复
    default = random.choice(fallback_replies)
    history.append(["机器人", default])
    return default

# 展示对话历史
def show_history():
    print("\n📜 当前对话历史：")
    for idx, chat in enumerate(history, 1):
        role, text = chat
        print(f"  {idx}. {role}：{text}")
    print("-" * 40)

# ======================
# 5. 主程序
# ======================
def main():
    print("\033[94m===== 融合版带个人记忆+聊天历史机器人（输入 quit 退出）=====\033[0m\n")
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
            profile_fusion["name"] = None
            profile_fusion["age"] = None
            profile_fusion["hobby"] = None
            profile_fusion["gender"] = None
            save_profile()
            print("\033[91mAI：已清空你的所有个人记忆～\033[0m\n")
            show_history()
            continue

        # 查看个人全部信息
        if user_input == "查看我的信息":
            print("\033[91mAI：你的个人信息如下：\033[0m")
            print(f"  姓名：{profile_fusion['name'] if profile_fusion['name'] else '未设置'}")
            print(f"  年龄：{profile_fusion['age'] if profile_fusion['age'] else '未设置'}")
            print(f"  性别：{profile_fusion['gender'] if profile_fusion['gender'] else '未设置'}")
            print(f"  爱好：{profile_fusion['hobby'] if profile_fusion['hobby'] else '未设置'}\n")
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

        # ---------- 普通闲聊 ----------
        res = get_robot_response(user_input)
        print(f"\033[91mAI：{res}\033[0m\n")
        show_history()

if __name__ == "__main__":
    main()