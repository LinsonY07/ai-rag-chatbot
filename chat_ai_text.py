import json

"""
编写代码的思路：
1.先搭建容器：（例如列表或字典，存储数据）
2.在搭建整个项目的骨架（方便后续代码的编写）
3.之后先完成退出的逻辑（保证程序能够正常停止）
4.在逐步实现程序中具体的每一个功能
5.最后在加入兜底功能（处理程序本身无法识别或处理的输入）
"""

# 全局记忆容器
profile = {
    "name": None,
    "age": None,
    "hobby": None,
    "gender": None
}

# 保存信息到本地文件
def save_profile():
    try:
        with open("profile.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception:
        # 就算保存失败也不崩溃
        pass

# 从本地文件加载信息
def load_profile():
    global profile
    try:
        with open("profile.json", "r", encoding="utf-8") as f:
            profile = json.load(f)
    except FileNotFoundError:
        # 文件不存在，初始化默认结构
        profile = {
            "name": None,
            "age": None,
            "hobby": None,
            "gender": None
        }
    except Exception:
        # 文件损坏、格式错误，也用默认结构兜底
        profile = {
            "name": None,
            "age": None,
            "hobby": None,
            "gender": None
        }

# 程序启动自动加载
load_profile()

# 查询信息封装
def answer_info(text, key_word, field, has_msg, empty_msg):
    if key_word in text:
        if profile.get(field):
            print("AI:", has_msg.format(profile[field]))
        else:
            print("AI:", empty_msg)
        return True 
    return False

# 录入信息封装
def extract_info(text, key_word, field, has_msg):
    if key_word in text: 
        profile[field] = text.replace(key_word, "").strip()
        print('AI:', has_msg.format(profile[field]))
        save_profile()
        return True
    return False

def main():
    while True:
        user_text = input("我: ")

        # 退出程序
        if user_text == "quit":
            print("AI: 拜拜，下次再聊")
            break
        
        # 清空记忆
        if user_text == "清空记忆":
            profile["name"] = None
            profile["age"] = None
            profile["hobby"] = None
            profile["gender"] = None
            save_profile()
            print("AI: 已清空所有记忆～")
            continue

        # 查询问句
        if answer_info(user_text, "我叫什么", "name", "你叫{}", "你还没告诉我你叫什么呢~"):
            pass
        elif answer_info(user_text, "我几岁", "age", "你{}岁", "你还没告诉我年龄呢~"):
            pass
        elif answer_info(user_text, "我喜欢什么", "hobby", "你喜欢{}", "你还没告诉我喜欢什么呢~"):
            pass
        elif answer_info(user_text, "我性别是什么", "gender", "你的性别是{}", "你还没告诉我性别~"):
            pass
        
        # 录入信息
        elif extract_info(user_text,"我叫", "name", "我记住了你的名字是{}"):
            pass
        elif extract_info(user_text, "我今年", "age", "我记住了你的年龄是{}岁"):
            pass
        elif extract_info(user_text, "我喜欢", "hobby", "我知道你喜欢{}了"):
            pass
        elif extract_info(user_text, "我性别是", "gender", "我知道你是{}"):
            pass

        # 兜底
        else:
            print("AI:抱歉，我暂时还没有其他的功能")

if __name__ == "__main__":
    main()