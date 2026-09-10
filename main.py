# -*- coding: utf-8 -*-
"""
企业知识库 AI 助手（Enterprise Knowledge AI Assistant）
=========================================================
一个基于 RAG 的知识库问答服务：
  上传文档 → 切块 → 语义向量(bge-m3) → 检索 → DeepSeek 生成 → 返回答案 + 来源

检索：用向量数据库 Chroma 持久化存储（chroma_db/），相似检索交给 collection.query

接口：
  GET  /        前端页面
  POST /ask     提问：{"question": "..."} → {question, answer, source_title, source_chunk}
  POST /upload  上传 .md/.txt 入库并重建索引
  POST /reindex 从 knowledge/ 全量重建向量库
  GET  /health  健康检查（向量库条数等）

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
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
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

# 向量数据库（Chroma）：数据存在 chroma_db/ 目录，重启不丢
chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
# 距离空间用 cosine：距离 = 1 - 余弦相似度（0 = 一模一样，越大越不相关）
collection = chroma_client.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})

# ---- 混合检索参数：融合分 = 语义权重×语义相似度 + 关键词权重×关键词相似度 ----
# 数值都要用真实数据校准（见 README「测试」一节）
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.8"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.2"))
# 融合分低于此值 → 判定"知识库里没有相关内容"（不给假来源）
# 该值由真实数据校准：该拒的样本融合分 ≈ -0.05，该答的弱相关样本 ≈ 0.20，故取两者之间
MIN_FUSED_SCORE = float(os.getenv("MIN_FUSED_SCORE", "0.12"))
# 取融合分最高的前 K 个块一起交给模型（只给 1 块容易漏掉正确答案）
TOP_K = int(os.getenv("TOP_K", "3"))


def embed_texts(texts):
    """把一堆文字变成向量矩阵。texts 是字符串列表，返回 (n, 维度) 的数组。"""
    resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return np.array([d.embedding for d in resp.data])   # 每个 d.embedding 是一串数字


all_chunks = []           # 当前所有块（按文件顺序）
sources = []              # 和 all_chunks 一一对应：每个块来自哪篇文档
tfidf_vectorizer = None   # 关键词检索用的"词表"
tfidf_matrix = None       # 关键词检索用的 TF-IDF 矩阵


def load_chunks():
    """读 knowledge/ 文件夹 → 切块（纯本地，不调 API，很快）。"""
    global all_chunks, sources
    all_chunks, sources = [], []
    for f in sorted(KNOWLEDGE_DIR.glob("*.md")) + sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        # 去掉 markdown 标题行（# 开头）——标题混进块里是噪音，会干扰检索和展示
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
        for c in cut_into_chunks(text):
            all_chunks.append(c)
            sources.append(f.stem)


def build_lexical_index():
    """建"字块/关键词"索引：字符级 TF-IDF（中文按"每个字一个特征"处理）。"""
    global tfidf_vectorizer, tfidf_matrix
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)[\u4e00-\u9fff]|[A-Za-z]+")
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_chunks) if all_chunks else None


# ---- 重建索引：切块 → 向量化进 Chroma → 建关键词索引 ----
def rebuild_index():
    global collection
    load_chunks()

    # 先删旧集合、再建新的（= 全量重建）
    try:
        chroma_client.delete_collection("knowledge")
    except Exception:
        pass
    collection = chroma_client.create_collection("knowledge", metadata={"hnsw:space": "cosine"})

    if all_chunks:
        vectors = embed_texts(all_chunks)                     # 每个块 → 语义向量
        collection.add(
            ids=[str(i) for i in range(len(all_chunks))],      # 每条一个唯一 id
            embeddings=vectors.tolist(),                       # 向量
            documents=all_chunks,                              # 原文
            metadatas=[{"source": s} for s in sources],        # 元数据（来源文档）
        )
    build_lexical_index()


# ---- 启动 ----
load_chunks()
build_lexical_index()            # 关键词索引很便宜，每次启动都建
if collection.count() == 0:
    rebuild_index()              # 向量库为空才建（贵：要调 embedding API）


# ---- 检索 + 生成 ----
def answer(question: str) -> dict:
    """混合检索（语义 + 关键词）→ 融合排序 → 取 top-K → 过了门槛才生成回答。"""
    if collection.count() == 0:
        return {"answer": "知识库为空，请先上传文档。", "source_title": "",
                "source_chunk": "", "matched": False}

    q_vec = embed_texts([question])          # 问题 → 语义向量

    # ① 语义分：向量库返回"每一个块"的距离（距离越小越相关）→ 相似度 = 1 − 距离
    res = collection.query(query_embeddings=q_vec.tolist(), n_results=collection.count())
    ids = [int(i) for i in res["ids"][0]]
    sem_sims = [1.0 - float(d) for d in res["distances"][0]]

    # ② 关键词分：字符级 TF-IDF 余弦相似度（"字块"检索）
    if tfidf_matrix is not None and tfidf_matrix.shape[0] > 0:
        lex_sims = cosine_similarity(tfidf_vectorizer.transform([question]), tfidf_matrix)[0]
    else:
        lex_sims = np.zeros(len(all_chunks))

    # ③ 融合打分：语义 × 0.8 + 关键词 × 0.2（权重可调），然后按分数排序
    scored = []
    for pos, idx in enumerate(ids):
        if idx >= len(all_chunks):
            continue
        sem = sem_sims[pos]
        lex = float(lex_sims[idx])
        fused = SEMANTIC_WEIGHT * sem + LEXICAL_WEIGHT * lex
        scored.append((fused, idx, sem, lex))
    scored.sort(reverse=True)                # 分数从高到低

    if not scored:
        return {"answer": "知识库中没有找到与这个问题相关的内容，无法回答。",
                "source_title": "", "source_chunk": "", "matched": False}

    # ④ 门槛：用"最高分"（top-1）去卡
    top_fused, top_idx, top_sem, top_lex = scored[0]
    scores = {
        "score": round(top_fused, 4),      # 融合分（用来卡门槛）
        "semantic": round(top_sem, 4),     # 语义分（方便你校准权重）
        "lexical": round(top_lex, 4),      # 关键词分
    }
    if top_fused < MIN_FUSED_SCORE:
        return {
            "answer": "知识库中没有找到与这个问题相关的内容，无法回答。",
            "source_title": "",          # 不给来源，避免误导用户
            "source_chunk": "",
            "matched": False,
            **scores,
        }

    # ⑤ 取 top-K 块一起当参考文档（只给 1 块容易漏掉真正能回答的那块）
    top_hits = scored[:TOP_K]
    context = "\n\n".join(f"[{sources[i]}] {all_chunks[i]}" for _, i, _, _ in top_hits)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是知识库助手，只能根据提供的文档回答，不要自己编。"},
            {"role": "user", "content": f"参考文档：\n{context}\n\n问题：{question}\n请根据文档回答。"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "source_title": sources[top_idx],                         # 最相关的那篇
        "source_chunk": all_chunks[top_idx],                      # 最相关的那块原文
        "sources_used": [sources[i] for _, i, _, _ in top_hits],  # 实际用了哪几篇
        "matched": True,
        **scores,                       # 带上三个分数，方便看数据调权重/门槛
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


@app.post("/reindex")
def reindex():
    """从 knowledge/ 全量重建向量库（手工改了文件夹里的文件后可调它）。"""
    rebuild_index()
    return {"status": "reindexed", "chunks": collection.count()}


@app.get("/health")
def health():
    """健康检查：能看出用的是向量库、以及里面有多少条。"""
    return {"status": "ok", "vector_db": "chroma", "chunks": collection.count()}


# ---- 方便启动：直接 `python main.py` 就行（等价于 uvicorn main:app --reload）----
# __name__ == "__main__" 的意思是"这个文件是被直接运行的"（而不是被别的文件 import 的）。
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
