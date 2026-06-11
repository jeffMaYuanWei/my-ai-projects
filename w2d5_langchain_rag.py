#!/usr/bin/env python3
"""Week 2 Day 5: LangChain 版 RAG（新版 LCEL 語法）"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ① Loader：讀 PDF
pdf_path = input("PDF 檔案路徑：").strip()
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# ② Splitter：切段
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"PDF 切成 {len(chunks)} 個段落")

# ③ Embeddings + VectorStore
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ④ LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ⑤ Prompt 模板：定義怎麼把檢索結果塞進提示詞
prompt = ChatPromptTemplate.from_template(
    "請只根據以下資料回答問題，若資料中沒有答案就說不知道。\n\n"
    "資料：\n{context}\n\n"
    "問題：{question}"
)

# 把多個檢索段落合併成一段文字
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ⑥ Chain：用 | 把各步驟串成一條鏈（這就是 LCEL，新版核心語法）
# 流程：問題 → 檢索 → 填入 prompt → LLM → 取出文字
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("準備完成，可以開始提問\n")

while True:
    q = input("\n你的問題（exit 結束）：").strip()
    if q.lower() == "exit":
        break
    if q:
        print(f"\nAI：{chain.invoke(q)}")
