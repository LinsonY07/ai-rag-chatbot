import random
import os

# 启用Windows终端颜色
os.system("")

# 记忆存储：用列表记住所有的对话
history = []

# 对话库（支持关键词匹配）
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

def get_robot_response(user_text):
    # 把对话封存起来，并将对话添加到历史列表中
    history.append(["用户", user_text])
    
    # 优先匹配关键词
    for keywprd, replies in reply_dict.items():
        if keywprd in user_text:
            history.append(["机器人",replies])
            return random.choice(replies)
    #没匹配到就返回兜底回复
    default = random.choice(fallback_replies)
    history.append(["机器人", default])
    return default

# 输出清楚的对话历史
def show_history():
    print("\n📜 当前对话历史：")
    for idx, chat in enumerate(history, 1):
        role, text = chat
        print(f"  {idx}. {role}：{text}")
    print("-" * 40)


def main():
    print("\033[94m===== 本地版带记忆的CLI聊天机器人（输入 quit 退出）=====\033[0m\n")
    while True:
        # 获取用户输入（绿色字体）
        user_input = input("\033[92m我：\033[0m")
            
        # 退出条件
        if user_input.strip().lower() == "quit":
            print("\033[91mAI：拜拜！下次再聊~ 👋\033[0m")
            # 退出前展示完整历史
            show_history()
            break
            
        # 获取回复并输出（红色字体）
        response = get_robot_response(user_input)
        print(f"\033[91mAI：{response}\033[0m\n")

        # 每轮都展示历史
        show_history()

    

if __name__ == "__main__":
    main()