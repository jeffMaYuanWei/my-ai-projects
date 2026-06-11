#!/usr/bin/env python3
"""Week 2 Day 2: ChromaDB Vector Database — 存入向量並做相似度搜尋"""

import chromadb
from sentence_transformers import SentenceTransformer

# 載入多語言 embedding 模型（昨天驗證過中英文都支援）
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 建立 ChromaDB client（in-memory，程式結束就消失，適合本地練習）
client = chromadb.Client()

# 建立一個 collection（相當於資料庫裡的一張表）
collection = client.create_collection(name="my_docs")

# 準備幾段文件（模擬 PDF 切出來的段落）
documents = [
    "蛋白質的熱穩定性可用 Tm 值衡量",
    "抗體的聚集傾向會影響藥物開發",
    "今天台北下著毛毛雨，氣溫24度",
    "深度學習常使用 GPU 加速訓練",
]

# 把每段文件轉成向量
# .tolist() 把 numpy array 轉成 Python list，ChromaDB 要求這格式
embeddings = model.encode(documents).tolist()

# 存進資料庫
# ids：每筆資料的唯一編號；documents：原文；embeddings：對應向量
collection.add(
    ids=[f"doc{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings,
)

# === 開始搜尋 ===
query = "蛋白質穩定性怎麼測量"
print(f"\n查詢問題：{query}\n")

# 把問題也轉成向量
query_embedding = model.encode([query]).tolist()

# 在資料庫搜尋最相似的 2 筆
# n_results=2：回傳前 2 筆最相關結果
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2,
)

# 印出結果
# distances 是「距離」，越小代表越相似
print("=== 最相關的段落 ===\n")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"距離 {dist:.4f}：{doc}")
