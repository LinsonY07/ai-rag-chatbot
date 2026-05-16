# 📚 RAG 智能问答助手

一个基于 FastAPI、Chroma 向量数据库和通义千问大模型的检索增强生成（RAG）问答系统，支持多格式文档解析、对话记忆与兜底逻辑。

---

## ✨ 项目亮点
- **多格式文档解析**：支持 `.txt`/`.md`/`.pdf`/`.docx` 文件自动解析入库。
- **精准检索与重排**：向量检索 + 关键词伪重排，优先返回最相关的文档片段。
- **对话记忆持久化**：基于 SQLite 存储对话历史，支持上下文理解。
- **完善兜底逻辑**：知识库为空时，直接返回固定提示，杜绝模型编造内容。
- **流式响应输出**：实现打字机效果，提升用户交互体验。
- **优雅异常处理**：对不支持的文件格式、数据库异常等场景做了降级处理。

---

## 🛠️ 技术栈
| 模块 | 技术/工具 |
|------|-----------|
| 后端框架 | FastAPI |
| 向量数据库 | Chroma DB |
| 嵌入模型 | DashScope Text Embedding |
| 大语言模型 | 通义千问 |
| 对话存储 | SQLite |
| 文档解析 | PyPDF2 / python-docx |
| 前端 | HTML + JavaScript |

---

## 📁 项目结构
```
ai_daima/
├── api/
│   └── routes.py          # 问答接口与流式响应实现
├── core/
│   ├── rag.py             # 检索、重排、Prompt 拼接
│   └── conversation.py    # 对话记忆管理
├── db/
│   └── chroma_client.py   # 向量数据库初始化与操作
├── utils/
│   ├── config.py          # 配置文件
│   ├── logger.py          # 日志配置
│   ├── parsers.py         # 多格式文档解析器
│   └── file_loader.py     # 文件加载与预处理
├── knowledge/             # 待入库文档目录
├── chroma_db/             # 向量数据库持久化目录（自动生成）
├── conversation.db         # 对话历史数据库（自动生成）
├── main.py                # 项目入口文件
├── ingest.py              # 文档入库脚本
├── index.html             # 前端页面
└── requirements.txt       # 依赖清单
```

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv .venv
# 激活虚拟环境（Windows）
.venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
在项目根目录创建 `.env` 文件，添加以下配置：
```env
DASHSCOPE_API_KEY=你的通义千问API Key
EMBEDDING_MODEL=text-embedding-v1
CHROMA_DB_PATH=./chroma_db
```

### 3. 文档入库
将需要导入的文档放入 `knowledge` 文件夹，执行入库脚本：
```bash
python ingest.py
```
✅ 预期输出：显示文件处理进度、片段入库情况，最终输出总片段数。

### 4. 启动服务
```bash
uvicorn main:app --reload
```
服务启动后，访问 `http://127.0.0.1:8000/index.html` 即可使用问答功能。

---

## 📖 核心功能说明
1.  **文档解析与入库**：支持 `.txt`/`.md`/`.pdf`/`.docx` 四种格式，自动分块生成向量并存入 Chroma DB。
2.  **检索增强问答流程**：用户提问 → 向量检索 → 伪重排 → Prompt 拼接 → 流式响应输出。
3.  **兜底逻辑设计**：知识库为空时，直接返回 `暂无相关信息，无法回答`，不依赖模型通用知识编造内容。

---

## 🧪 测试场景验证
| 测试场景 | 预期结果 | 验证状态 |
|----------|----------|----------|
| 空知识库提问 | 返回固定提示，不编造内容 | ✅ 通过 |
| 多格式文档入库 | 不同格式文件均正常解析 | ✅ 通过 |
| 超长文档后半部分提问 | 能精准命中分块后的相关内容 | ✅ 通过 |
| 不支持格式文件 | 输出警告，不影响其他文件入库 | ✅ 通过 |
| 对话记忆测试 | 能记住上下文信息 | ✅ 通过 |

---

## 📌 注意事项
- 请确保 `.env` 文件中的 API Key 配置正确，否则无法调用大模型服务。
- 每次更新 `knowledge` 文件夹中的文档后，需重新执行 `python ingest.py` 完成入库。
- `chroma_db` 和 `conversation.db` 为自动生成的持久化文件，无需手动修改。
- 服务默认端口为 `8000`，可通过 `uvicorn main:app --port 自定义端口` 修改。
