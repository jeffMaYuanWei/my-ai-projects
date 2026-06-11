#!/usr/bin/env python3
"""Week 2 Day 4: 真實 PDF 的 RAG — 切段 + 向量檢索 + LLM 回答"""

import os
import sys
import chromadb
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def load_pdf(path):
    """讀取 PDF 全文"""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def build_database(text):
    """把全文切段、轉向量、存進 ChromaDB"""

    # chunk_size=500：每段最多 500 字
    # chunk_overlap=50：段落間重疊 50 字，避免句子被切斷失去語意
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)
    print(f"PDF 切成 {len(chunks)} 個段落")

    embeddings = model.encode(chunks).tolist()

    chroma = chromadb.Client()
    collection = chroma.create_collection(name="pdf")
    collection.add(
        ids=[f"chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )
    return collection


def ask(collection, question):
    """檢索最相關段落，交給 LLM 回答"""

    # n_results=3：取最相關的 3 段（比之前多，因為真實文件資訊較分散）
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    relevant = results["documents"][0]

    context = "\n\n".join(relevant)
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
    pdf_path = input("PDF 檔案路徑：").strip()
    if not os.path.isfile(pdf_path):
        print("找不到檔案")
        sys.exit(1)

    print("讀取中...")
    text = load_pdf(pdf_path)
    collection = build_database(text)
    print("準備完成，可以開始提問\n")

    while True:
        q = input("\n你的問題（exit 結束）：").strip()
        if q.lower() == "exit":
            break
        if q:
            print(f"\nAI：{ask(collection, q)}")
