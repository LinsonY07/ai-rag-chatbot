# 记忆模块：用来专门管理对话历史
from typing import List, Dict

class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        """
        初始化对话记忆
        ：param max_tyrns:最多保留多少轮对话（防止Prompt过长）
        """
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_message(self, role: str, content: str):
        """添加一条消息到历史记录"""
        self.history.append({"role":role, "content":content})
        # 如果超过最大轮数，就删除最旧的对话
        if len(self.history) > self.max_turns * 2:
            self.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        """获取当前所有历史对话"""
        return self.history
    
    def clear(self):
        """清空对话历史"""
        self.history.clear()
        