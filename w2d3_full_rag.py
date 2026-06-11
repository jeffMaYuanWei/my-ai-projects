#!/usr/bin/env python3
"""Week 2 Day 3: 完整 RAG — 向量搜尋 + LLM 回答"""

import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

chroma = chromadb.Client()
collection = chroma.create_collection(name="knowledge")

documents = [
    "蛋白質的熱穩定性可用 Tm 值衡量，Tm 越高代表越穩定。",
    "抗體的聚集傾向會影響藥物開發，HIC 是常用的評估方法。",
    "acdc-nn 工具可預測 Tm2，也就是 Fab/CH3 區域的熔點。",
    "A3D 工具用來預測 HIC 聚集傾向，Spearman 相關係數約 0.54。",
    "深度學習常使用 GPU 加速訓練，CUDA 是常見的運算平台。",
]

embeddings = model.encode(documents).tolist()
collection.add(
    ids=[f"doc{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings,
)


def ask(question):
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=2)
    relevant = results["documents"][0]

    print("\n[檢索到的段落]")
    for r in relevant:
        print(f"  - {r}")

    context = "\n".join(relevant)
    prompt = (
        f"請只根據以下資料回答問題，若資料中沒有答案就說不知道。\n\n"
        f"資料：\n{context}\n\n"
        f"問題：{question}"
    )
    response = client_llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        q = input("\n你的問題（exit 結束）：").strip()
        if q.lower() == "exit":
            break
        if q:
            print(f"\nAI：{ask(q)}")
