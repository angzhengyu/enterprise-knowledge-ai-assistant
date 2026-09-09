# 企业知识库 AI 助手（Enterprise Knowledge AI Assistant）

基于 **RAG（检索增强生成）** 的知识库问答服务：上传文档 → 切块 → 语义向量化 → 检索 → 大模型生成，返回答案并**标注来源**。提供网页界面和 REST API。

> 本项目是个人学习 / 求职作品，用于练习真实企业 RAG 的完整链路：**文档处理、向量化、检索、生成、溯源、API、前端页面、部署**。

---

## 1. 项目简介

一个可运行的知识库问答系统。用户上传 Markdown / 文本文件作为知识库，然后在网页输入问题，系统会：
1. 把问题转成语义向量；
2. 在知识库里检索**最相关的文档块**；
3. 让大模型**基于该文档块**回答；
4. 同时返回**来源文档**和**引用原文**，方便核对、防止编造。

## 2. 项目背景

企业里大量知识散落在文档中（规章制度、产品手册、FAQ），员工查找困难、新人不熟悉。传统方式要么靠人搜，要么靠 LLM 直接回答（但会**编造**）。RAG 通过在回答前**先检索真实文档**，让回答有依据、可溯源，是目前企业知识库的主流解法。

## 3. 核心功能

- 📄 **文档入库**：通过 `POST /upload` 上传 `.md` / `.txt` 文件，自动保存到 `knowledge/` 并**重建索引**
- 🔪 **文档切块**：按句号把长文档切成小块
- 🧬 **语义向量化**：用 embedding 模型（bge-m3）把文字变成语义向量
- 🔍 **语义检索**：问题和文档块用余弦相似度比对，找到最相关块
- 🧠 **生成回答**：把命中的块作为上下文交给 DeepSeek 生成
- 📌 **引用来源**：返回 `source_title` 和 `source_chunk`，答案可溯源
- 🖥️ **网页前端**：输入问题、显示答案与来源

## 4. 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python · FastAPI · Uvicorn |
| 向量化 | 硅基流动 SiliconFlow · `BAAI/bge-m3`（OpenAI 兼容接口） |
| 生成 | DeepSeek · `deepseek-chat` |
| 检索 | scikit-learn · `cosine_similarity` + `np.argmax` + numpy |
| 数据处理 | python-dotenv · python-multipart |
| 前端 | HTML · CSS · 原生 JS（`fetch`） |

> 说明：**生成（DeepSeek）和向量化（硅基流动 bge-m3）是两个不同服务**。因为 DeepSeek 只开放了 `chat`（生成）接口，没有 `embeddings` 接口；embedding 我们用提供免费中文模型的硅基流动（OpenAI 兼容，`api.siliconflow.cn`）。

## 5. 系统架构

```
浏览器（前端页面 static/index.html）
         │  POST /ask  {"question": "..."}
         ▼
     FastAPI 服务（main.py）
         │
         ├─ 1. 切块 chunk（按句号切）
         ├─ 2. 语义向量化 embed_texts()  → 硅基流动 bge-m3
         ├─ 3. 检索 cosine_similarity + np.argmax → 命中 best_chunk、source_title
         ├─ 4. 生成 client.chat.completions → DeepSeek deepseek-chat
         └─ 5. 返回 {answer, source_title, source_chunk}

知识库：knowledge/ 文件夹（.md/.txt），POST /upload 可上传新增并重建索引
```

## 6. 项目目录

```
enterprise-knowledge-ai-assistant/
├── main.py            # FastAPI 服务（切块/向量化/检索/生成/接口）
├── knowledge/         # 知识库（.md/.txt 文档），往里面加文件即新增知识
├── static/
│   └── index.html     # 前端页面
├── requirements.txt   # Python 依赖
├── .env.example       # 配置示例（复制为 .env 填 key）
├── .gitignore         # 忽略 .env / .venv 等
└── README.md
```

## 7. 环境安装

```powershell
# 建议 Python 3.10+
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 8. 配置方法

1. 复制 `.env.example` 为 `.env`；
2. 填入两个 key：
   - `DEEPSEEK_API_KEY`：DeepSeek 平台的 key（生成回答）
   - `SILICONFLOW_API_KEY`：硅基流动的 key（向量化，免费）
3. `knowledge/` 里放你的文档（`.md` / `.txt`）。

## 9. 启动方式

```powershell
cd enterprise-knowledge-ai-assistant
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

- 网页：打开 http://127.0.0.1:8000
- 接口文档：打开 http://127.0.0.1:8000/docs

## 10. API 说明

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 前端页面 |
| `/ask` | POST | 提问。请求 `{"question": "..."}`；返回 `{"question", "answer", "source_title", "source_chunk"}` |
| `/upload` | POST | 上传 `.md/.txt`，保存到 `knowledge/` 并重建索引 |

**示例**

`POST /ask`：
```json
// 请求
{"question": "docker是什么"}
// 响应
{
  "question": "docker是什么",
  "answer": "根据提供的文档，Docker 是一种用于把应用容器化的工具，目的是方便部署。",
  "source_title": "docker",
  "source_chunk": "Docker 用于把应用容器化，方便部署。"
}
```

## 11. Demo 截图

> 运行后截图补充：
> - 网页提问界面（含答案 + 来源）
> - `/docs` 接口文档页

_（此处放实际截图）_

## 12. 技术难点

1. **中文没有空格**，`sklearn` 默认按空格分词会把整串汉字当一个词，导致检索相似度全为 0、检索不准。
2. **字面检索容易被通用字误导**：`TfidfVectorizer` 按字面重叠算相似度，提问「docker是什么」会被「什么是/是什么」这种通用字抢走，命中错误文档。
3. **生成和向量化是不同服务**：DeepSeek 只有 chat 接口，没有 embeddings 接口。
4. **文档上传后索引要重建**：新文档要立刻可检索，不能靠手动重启。

## 13. 解决方案

1. 用 `token_pattern=r"(?u)[\u4e00-\u9fff]|[A-Za-z]+"` 让**每个汉字单独成特征**（后来语义检索取代了它）。
2. **升级到语义向量（embedding）**：用 bge-m3 把整句话变成语义向量，按"意思"而非"字面"匹配，解决被通用字干扰的问题。
3. 用**两个 OpenAI 兼容客户端**：一个指向 `api.deepseek.com`（生成），一个指向 `api.siliconflow.cn`（向量化），复用同一套调用方式。
4. 把「读文件夹→切块→向量化」抽成 `rebuild_index()`，`/upload` 上传后调用它**重建索引**，新文档立即可检索。

## 14. 测试

- 上传一份新文档后，`/ask` 能正确回答并返回其来源；
- 修改/删除 `knowledge/` 里文件后重启，索引随之更新；
- 中文、英文问题均可正常检索与回答。

## 15. Docker（TODO）

计划用 Dockerfile + docker-compose 打包运行（含 FastAPI 服务），便于一键部署。

## 16. Future Work

- [ ] 支持 PDF / Word 等更多格式解析
- [ ] 引入真正的向量数据库（如 Chroma）与重排序（Reranker）
- [ ] 用户会话 / 权限 / 日志
- [ ] 前端更完善（多轮对话、文档管理界面）
- [ ] Docker 化部署
- [ ] 单元测试与 GitHub Actions

---

*个人学习作品。仅用于展示 RAG + FastAPI + 语义检索的完整工程链路。*
