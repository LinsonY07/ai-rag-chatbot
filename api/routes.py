# 存放路由接口（只负责接收前端请求，不写业务逻辑）
# 第一步改动：先替换导入区，将旧的rag模块换为新的 LangChain 模块
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from utils.logger import logger
import json
import time
from utils.config import settings
# from core.rag import retrieve_relevant_docs, build_prompt, pseudo_rerank  #旧的检索模块
# from dashscope import Generation  
# from core.conversation import ConversationMemory  #旧的记忆模块
from core.rag_langchain import create_full_rag_chain, retrieve_documents
from utils.langchain_memory import get_conversation_memory



router = APIRouter()

# 封装函数：利用历史对话改写用户提问逻辑
def rewrite_question(question, history):
    rewrite = question
    # 确保有上一轮对话
    if len(history) > 2:
        last_user_question = None
        for msg in reversed(history):   #倒着遍历历史信息
            if msg["role"] == "user" and msg["content"] != question:
                last_user_question = msg["content"]
                break
        if last_user_question:
            # 将模糊问题改写为明确问题
            rewrite = f"基于上一轮问题；'{last_user_question}',回答当前问题{question}"
    return rewrite

# 第二步改动：重写 ask_stream
# LangChain 版本
@router.post("/ask_stream")
async def ask_stream(request: Request):
    try:
        # 1.接收前端 JSON
        data = await request.json()
        question = data.get("question","").strip()
        session_id = data.get("session_id", "default")
        
        # 2.日志记录
        logger.info(f"会话：{session_id}，用户提问：{question}")
        
        # 3.空问题判断
        if not question:
            return StreamingResponse(iter(["请输入你的问题~"]), media_type= "text/plain") 
        
        # 一、加载对话记忆
        # 1.根据 session_id 获取记忆
        memory = get_conversation_memory(session_id)
        
        # 2.加载记忆变量（LangChain 格式）
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # 3.获取原始历史（给问题改写用）
        raw_history = memory.sqlite_memory.get_history()
        
        # 二、问题改写
        # 将上下文依赖的问题 ——> 变成无依赖的检索问题
        rewritten_question = rewrite_question(question, raw_history)
        
        # 三、流式生成函数
        def generate():
            full_answer = ""    #保存完整回答，最后存记忆
            try:
                # 1.创建 RAG 链
                chain = create_full_rag_chain(
                    use_rerank = True,
                    top_k = settings.recall_top_k
                )
                
                # 2.构造输入（必须包含：question + chat_history）
                input_data = {
                    "question": rewritten_question,
                    "chat_history":chat_history
                }
                
                # 3.流式调用
                for token in chain.stream(input_data):
                    full_answer += token
                    yield token #输出给前端
                    time.sleep(0.01)    #模拟打字机效果
                    
                    
                # 异常捕获 + 保存对话记忆
            except Exception as e:
                logger.error(f"模型调用失败：{e}")
                yield f"抱歉，模型调用出错了：{str(e)[:50]}"
                
            finally:
                # 必须保存对话
                try:
                    memory.save_context(
                        {"question": question},
                        {"response": full_answer or "无回答"}
                    )
                except:
                    pass
                
        # 返回流式响应
        return StreamingResponse(generate(), media_type = "text/plain")              
    except Exception as e:
        logger.error(f"接口异常：{e}")
        return StreamingResponse(iter([f"接口出错：{str(e)}"]),media_type = "text/plain")

# 清空历史会话( LangChian 版)
@router.post("/clear_history")
async def clear_history(request: Request):
    try:
        # 1.等待前端传来的 json
        data = await request.json()
        
        # 2.取出 session_id ，没有就用默认值
        session_id = data.get("session_id","default")
        
        # 根据 session_id 获取记忆实例
        memory = get_conversation_memory(session_id)
        
        #清空该会话的所用历史 
        memory.clear()
        
        # 返回结构化结果
        return {"status": "success", "msg":"对话已清空"}
    except Exception as e:
        return {"status": "error", "msg": f"清空失败：{str(e)}"}
    