# 存放路由接口（只负责接收前端请求，不写业务逻辑）
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from utils.logger import logger
from core.rag import retrieve_relevant_docs, build_prompt, pseudo_rerank
from dashscope import Generation
import json
import time
from utils.config import settings
from core.conversation import ConversationMemory  


memory = ConversationMemory(max_turns=5)  

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

# 流式接口
@router.post("/ask_stream")
async def ask_stream(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        # session_id = data.get("session_id", "default")

        logger.info(f"用户提问: {question}")
        # 防御：空问题直接返回友好提示
        if not question:
            def empty_gen():
                yield "请输入你的问题~"
            return StreamingResponse(empty_gen(), media_type="text/plain")

        # 1. 把当前问题加入记忆
        memory.add_message("user", question)
        
        # 2.从记忆类获取历史
        history = memory.get_history()

        # 调用修改对话函数
        rewritten_question = rewrite_question(question, history)

        # 3. 检索
        finally_docs = retrieve_relevant_docs(rewritten_question)
        # 4. 拼接Prompt
        prompt = build_prompt(question, finally_docs, history)

        # 5. 调用通义千问
        def gen():
            # 实现流式增量输出
            full_answer = ""
            last_len = 0   # 记录上一次的长度，用来取增量
            try:
                responses = Generation.call(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    api_key=settings.dashscope_api_key,
                    temperature = 0.1
                )
                for resp in responses:
                    if hasattr(resp, "output") and hasattr(resp.output, "text"):
                        current_text = resp.output.text
                        if current_text:
                            token = current_text[last_len:]
                            full_answer += token
                            yield token
                            last_len = len(current_text)
                            time.sleep(0.01)

                # 流式回答已经发完，开始拼接参考来源
                try:
                    if isinstance(finally_docs, list) and len(finally_docs) > 0:
                        valid_sources = []
                        for doc in finally_docs:
                            if isinstance(doc, dict) and "source" in doc:
                                valid_sources.append(doc["source"])
                        # 去重
                        valid_sources = list(set(valid_sources))
                        if valid_sources:
                            source_str = "###SOURCE###" + "###".join(valid_sources)    
                            yield source_str
                except Exception as se:
                    logger.error(f"来源拼接失败：{se}")

            except Exception as e:
                logger.error(f"模型调用失败：{str(e)}")
                yield f"抱歉，模型调用出错了{str(e)[:50]}"
            finally:
                # 5. 保存记忆
                try:
                    # 支持上下文理解
                    memory.add_message("assistant", full_answer or "无回答")
                   
                except:
                    pass
        return StreamingResponse(gen(), media_type="text/plain")
    
    except Exception as e:
        logger.error(f"接口异常{str(e)}")
        return StreamingResponse(iter([f"接口出错：{str(e)}"]), media_type = "text/plain")


# 清空记忆接口
@router.post("/clear_history")
async def clear_history():
    memory.clear()
    return {"status": "success", "msg": "对话已清空"}# ========================
