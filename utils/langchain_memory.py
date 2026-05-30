"""
LangChain 对话记忆适配器

设计目的：
1.复用项目已用的 SQLite 持久化记忆 （core/conversation）
2.适配器 LangChain 的 BaseMemory 标准接口
3.让自定义记忆可直接接入 RunnableWithMessagesHistory / RAG链

遵循规范：
- 继承 langchain_core.memory.BaseMemory
- 实现三个必须方法：memory_variables / load_memory_variables / save_context / clear
"""
from typing import List, Dict, Any
from core.conversation import ConversationMemory as SqliteMemory


class LangChainConversationMemory:
    """
    将 SQLite 记忆系统封装为 LangChain Memory 接口
    
    LangChain 的 Memory 接口要求实现以下方法：
    - langchain_memory_variables() - 加载历史对话
    - save_context() - 保存当前对话轮次
    - clear（）- 清空记忆
    
    这样可以无缝集成到 LangChain 的 Chain 中
    """
    def __init__(self, session_id: str = "default", max_turns: int = 6):
        self.session_id = session_id
        self.sqlite_memory = SqliteMemory(session_id = session_id, max_turns = max_turns)
        
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        加载记忆变量
        
        返回格式必须符合 LangChain 期望：
        {"chat_history":[{"role":"...", "content": "..."},...]}
        """
        history = self.sqlite_memory.get_history()
        
        # 转换为 LangChain 消息格式
        messages = []
        for msg in history:
            role = "ai" if msg["role"] == "assistant" else "human"
            messages.append({"role": role, "content": msg["content"]})
            
        return {"chat_history": messages}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """
        保存当前对话轮次
        
        inputs 包含: question
        outputs 包含: response
        """
        question = inputs.get("question", "")
        response = outputs.get("response", "")
        
        # 将获取到的当前对话，通过调用 SQLite 存储到历史对话 message 中
        self.sqlite_memory.add_message("user", question)
        self.sqlite_memory.add_message("assistant", response)
        
    def clear(self):
        """清空记忆"""
        self.sqlite_memory.clear()
        
# 全局工厂函数:创建并返回对话记忆适配器实例
def get_conversation_memory(session_id: str = "default") -> LangChainConversationMemory:
    return LangChainConversationMemory(session_id=session_id)
        
        
    