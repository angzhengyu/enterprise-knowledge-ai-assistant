# -*- coding: utf-8 -*-
"""
企业知识库 AI 助手（Enterprise Knowledge AI Assistant）
=========================================================
一个基于 RAG 的知识库问答服务：
  上传文档 → 切块 → 语义向量(bge-m3) → 检索 → DeepSeek 生成 → 返回答案 + 来源

接口：
  GET  /        前端页面
  POST /ask     提问：{"question": "..."} → {question, answer, source_title, source_chunk}
  POST /upload  上传 .md/.txt 入库并重建索引

配置：.env 里需要两个 key（见 .env.example）
  DEEPSEEK_API_KEY    生成回答（DeepSeek）
  SILICONFLOW_API_KEY 向量化（硅基流动 bge-m3）
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 读 API 密钥（优先读本项目 .env；也兼容你机器上的老配置）
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(r"D:\AI-Career\.env")
api_key = os.getenv("DEEPSEEK_API_KEY")            # 生成回答用的 DeepSeek
sf_key = os.getenv("SILICONFLOW_API_KEY")          # 向量化用的硅基流动
if not sf_key:
    raise RuntimeError("请在 .env 里加 SILICONFLOW_API_KEY=sk-xxxx（硅基流动的 key）")

# 知识库文件夹
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
# 前端页面文件夹
STATIC_DIR = Path(__file__).parent / "static"


# ---- 切块 ----
def cut_into_chunks(text, sep="。"):
    parts = text.split(sep)
    chunks = []
    for p in parts:
        p = p.strip()
        if p:
            chunks.append(p + sep)
    return chunks


# 两个"客户端"：一个管生成（DeepSeek），一个管向量化（硅基流动）
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
embed_client = OpenAI(api_key=sf_key, base_url="https://api.siliconflow.cn/v1")


def embed_texts(texts):
    """把一堆文字变成向量矩阵。texts 是字符串列表，返回 (n, 维度) 的数组。"""
    resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return np.array([d.embedding for d in resp.data])   # 每个 d.embedding 是一串数字


# ---- 重建索引：读文件夹 → 切块 → 向量化 ----
def rebuild_index():
    global documents, all_chunks, chunk_sources, chunk_matrix
    documents = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.md")) + sorted(KNOWLEDGE_DIR.glob("*.txt")):
        documents.append({"title": f.stem, "content": f.read_text(encoding="utf-8")})
    all_chunks = []
    chunk_sources = []            # 和 all_chunks 一一对应：每个块来自哪篇文档
    for doc in documents:
        doc["chunks"] = cut_into_chunks(doc["content"])
        for c in doc["chunks"]:
            all_chunks.append(c)
            chunk_sources.append(doc["title"])
    chunk_matrix = embed_texts(all_chunks)   # 每个块 → 语义向量


# 服务启动时先建一次索引
rebuild_index()


# ---- 检索 + 生成 ----
def answer(question: str) -> dict:
    """给定问题，返回 答案 + 来源（来自哪篇文档、哪一块原文）。"""
    q_vec = embed_texts([question])                     # 问题也变成语义向量
    scores = cosine_similarity(q_vec, chunk_matrix)[0]  # 和每个块比"意思"相近度
    best_index = int(np.argmax(scores))
    best_chunk = all_chunks[best_index]                 # 命中的那块文字
    source_title = chunk_sources[best_index]            # 它来自哪篇文档
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是知识库助手，只能根据提供的文档回答，不要自己编。"},
            {"role": "user", "content": f"参考文档：\n{best_chunk}\n\n问题：{question}\n请根据文档回答。"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "source_title": source_title,   # 来源文档名
        "source_chunk": best_chunk,     # 命中的原文块
    }


# ====================== FastAPI 接口 ======================
app = FastAPI(title="企业知识库助手")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    """返回前端页面（企业知识库 AI 助手）。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask")
def ask(req: AskRequest):
    result = answer(req.question)          # 现在返回的是带来源的字典
    return {"question": req.question, **result}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """上传一个文本文件，保存到 knowledge/ 并重建索引。"""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="没有文件名")
    safe_name = Path(file.filename).name          # 只取文件名，防目录穿越
    if not safe_name.endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="目前只支持 .md / .txt")
    dest = KNOWLEDGE_DIR / safe_name
    dest.write_bytes(file.file.read())            # 把上传的文件字节写到 knowledge/
    rebuild_index()                               # 重建索引，让新文档立即能被检索
    return {"filename": safe_name, "indexed_chunks": len(all_chunks)}
