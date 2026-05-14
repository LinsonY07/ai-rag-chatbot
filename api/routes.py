# 存放路由接口（只负责接收前端请求，不写业务逻辑）
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from utils.logger import logger
from core.rag import retrieve_relevant_docs, build_prompt, pseudo_rerank
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
    try:
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
        finally_docs = pseudo_rerank(question, docs, top_n=settings.rerank_top_k)
        # 3. 拼接Prompt
        prompt = build_prompt(question, docs, history)

        # 4. 调用通义千问
        def gen():
            # 实现流式增量输出
            full_answer = ""
            last_len = 0   # 记录上一次的长度，用来取增量
            try:
                responses = Generation.call(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    api_key=settings.dashscope_api_key
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
                # 1. 先过滤掉 None 和无效文档
                valid_docs = [doc for doc in finally_docs if doc and isinstance(doc, dict)]
                # 2. 提取所有来源（已去重）
                source_list = list({doc["source"] for doc in valid_docs})
                # 3. 拼接固定分隔符 + 所有来源
                source_str = "###SOURCE###" + "###".join(source_list)
                # 4. 把来源也通过流发给前端
                yield source_str
            except Exception as e:
                logger.error(f"模型调用失败：{str(e)}")
                yield f"抱歉，模型调用出错了{str(e)}"
            finally:
                # 5. 保存记忆
                chat_memory[session_id] = history + [
                    f"用户：{question}",
                    f"助手：{full_answer if full_answer else '模型无响应'}"
                ]

        return StreamingResponse(gen(), media_type="text/plain")
    except Exception as e:
        logger.error(f"接口异常{str(e)}")
        return StreamingResponse(iter([f"接口出错：{str(e)}"]), media_type = "text/plain")
