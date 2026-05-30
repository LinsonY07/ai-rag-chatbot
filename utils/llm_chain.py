"""
# 创建 LangChain 适配器工具类
## 作用:将 dashscope 的API格式转换为 LangChain 统一的 ChatMessage 格式,
   使其能与其他 LangChain 组件协同工作
## 关键知识点：
        1.适配器模式：将第三方 SDK 包装为标准接口
        2.消息格式转换：LangChain 的消息对象 -> dashscope 期望的字典格式
        3.流式支持：实现 steam（）方法支持流式输出
"""
# LangChain LLM 适配层

# 学习目标：了解如何将第三方 LLM SDK（如 dashscope）封装为 
# LangChain 标准接口，使其能与其他 LangChain 组件协同工作。



from dashscope import Generation
from utils.config import settings
from typing import Any, Generator, List, Dict
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage

class QwenChatLLM(Runnable):
    """
    通问千义 LLM 包装器
    作用：将 dashscope 的 API 格式转换为 LangChain 统一的 ChatMessage 格式，
    使得可以使用 ChatPromptTemplate、 Runnable 等组件。
    """
    # 初始化（构造方法）
    def __init__(self, model = None, temperature: float = 0.0):
        self.model = model or settings.llm_model
        self.temperature = temperature
        self.api_key = settings.dashscope_api_key
        
    # 格式转换方法(将 langchain 中的 messages 格式的消息转换为大模型可以看懂的 列表嵌套字典的数据类型)
    def _messages_to_dashscope(self, messages: Any) -> List[Dict]:
        if isinstance(messages, list):
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages
                ]
        return [{"role": "user", "content": str(messages)}]
    
    # inoke同步调用(调用大模型SDK)
    def invoke(self, messages: Any, config=None) -> AIMessage:
        formatted = self._messages_to_dashscope(messages)
        # 调用大模型SDK
        response = Generation.call(
            model=self.model,
            messages=formatted,
            api_key=self.api_key,
            temperature=self.temperature
        )
        
        content = response.output.text
        return AIMessage(content)   #LangChain 标准返回格式（必须返回 AIMessage ,才可以接入链、提示词、RAG）
    
    # stream 流式调用（Generator生成器：一边生成一边返回）
    def stream(self, messages: Any, config = None) -> Generator[str, None, None]:
        formatted = self._messages_to_dashscope(messages)
        
        response = Generation.call(
            model=self.model,
            messages=formatted,
            stream = True,  #开启流式输出
            api_key=self.api_key,
            temperature = self.temperature
        )
        
        last_text = ""
        for chunk in response:
            text = chunk.output.text
            if text is None:
                continue
            
            # 计算增量：但前文本 - 上次文本
            delta_text = text[len(last_text):]
            
            if delta_text:
                yield delta_text
                last_text = text 
            
# 创建实例对象
qwen_llm = QwenChatLLM()
