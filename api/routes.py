# 存放路由接口（只负责接收前端请求，不写业务逻辑）
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from utils.logger import logger
from core.rag import retrieve_relevant_docs, build_prompt
from dashscope import Generation
import json
import time
from utils.config import settings

router = APIRouter()

# 对话记忆存储
chat_memory = {}

# 流式接口
@router.post("/ask_stream")
async def ask_stream(request: Request):
    data = await request.json()
    question = data.get("question", "")
    session_id = data.get("session_id", "default")

    logger.info(f"用户提问: {question}")

    if not question:
        return StreamingResponse(iter(["问题不能为空"]), media_type="text/plain")

    # 1. 拿历史
    history = chat_memory.get(session_id, [])

    # 2. 检索
    docs = retrieve_relevant_docs(question)

    # 3. 拼接Prompt
    prompt = build_prompt(question, docs, history)

    # 4. 调用通义千问
    def gen():
        responses = Generation.call(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            api_key=settings.dashscope_api_key
        )
        # 实现流式增量输出
        full_answer = ""
        last_len = 0   # 记录上一次的长度，用来取增量
        for resp in responses:
            if hasattr(resp, "output") and hasattr(resp.output, "text"):
                current_text = resp.output.text
                if current_text:
                    token = current_text[last_len:]
                    full_answer += token
                    yield token
                    last_len = len(current_text)
                    time.sleep(0.01)
        
        # 5. 保存记忆
        chat_memory[session_id] = history + [f"用户：{question}", f"助手：{full_answer}"]

    return StreamingResponse(gen(), media_type="text/plain")

