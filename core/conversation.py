# 记忆模块：用来专门管理对话历史—— 已支持多会话隔离（电脑/手机互不干扰）
from typing import List, Dict
from utils.logger import logger
from db.db_manager import get_db_connection, init_chat_db

# 数据库初始化
init_chat_db()

class ConversationMemory:
    def __init__(self, session_id: str = "default", max_turns: int = 3):
        """
        初始化对话记忆（支持多会话）
        :param session_id: 会话ID → 每个设备/浏览器唯一
        :param max_turns: 最多保留多少轮对话（防止Prompt过长）
        """
        self.session_id = session_id    #关键：每个会话独立ID
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

        # 程序启动 → 自动加载当前历史对话
        self._load_from_db()
    
    # 从数据库中读取历史对话
    def _load_from_db(self):
        """
        功能：
        1. 打开数据库
        2. 读取所有聊天记录
        3. 按时间顺序放回内存
        """
        conn = get_db_connection()
        if not conn:
            return
        try:
            c = conn.cursor()
            # 查询当前ID所有记录，按 id 从小到大（时间正序）
            c.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                      (self.session_id,)
                      )

             # 获取所有行
            rows = c.fetchall()

            """
            把数据库数据 → 转成内存列表
            列表推导式：把数据库的每一行变成字典
            数据库一行：("user", "你好")
            变成字典：{"role": "user", "content": "你好"}
            """
            self.history = [{"role": r, "content": c} for r,c in rows]
            
            logger.info(f"✅ 加载会话【{self.session_id}】历史：{len(self.history)} 条")
        except Exception as e:
            logger.error(f"❌ 加载对话历史失败：{e}")
        finally:
            conn.close()     # 关闭数据库

    # 保存单条信息到数据库
    def _save_to_db(self, role:str, content:str):
        """
        每次发消息，自动存入数据库
        role: user / assistant
        content: 消息内容
        """
        conn = get_db_connection()
        if not conn:
            return
        try:
            c = conn.cursor()

            # 插入数据(加上会话ID)
            # ? 是占位符，防止SQL注入，安全写法
            c.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                (self.session_id, role, content)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"❌ 保存对话到数据库失败：{e}")

        finally:
            conn.close()

    def _clear_db(self):
        """
        点击清空历史时：
        1. 清空内存
        2. 清空数据库
        """
        conn = get_db_connection()
        if not conn:
            return
        try:
            c = conn.cursor()
            # 只删除删除当前会话的所有数据
            c.execute("DELETE FROM chat_history WHERE session_id = ?", (self.session_id,))
            conn.commit()
            logger.info(f"✅ 已清空会话【{self.session_id}】历史")
        except Exception as e:
            logger.error(f"❌ 清空数据库失败：{e}")
        finally:
            conn.close()

    def add_message(self, role: str, content: str):
        """添加一条消息到历史记录"""
        self.history.append({"role":role, "content":content})

        # 同时加到数据库
        self._save_to_db(role, content)

        # 如果超过最大轮数，就删除最旧的对话
        if len(self.history) > self.max_turns * 2:
            self.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        """获取当前所有历史对话"""
        return self.history
    
    def clear(self):
        """只清空当前会话"""
        self.history.clear()    # 清空内存
        self._clear_db()        # 清空数据库
        