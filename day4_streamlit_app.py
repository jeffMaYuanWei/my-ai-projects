#!/usr/bin/env python3
"""Day 4: Streamlit PDF Q&A — 網頁版 PDF 問答"""

import io
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

MODEL = "llama-3.3-70b-versatile"
MAX_PDF_CHARS = 3000


def get_api_key():
  load_dotenv()
  return os.getenv("GROQ_API_KEY")


def extract_pdf_text(uploaded_file) -> str:
  reader = PdfReader(io.BytesIO(uploaded_file.read()))
  pages = []
  for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
      pages.append(f"--- 第 {i} 頁 ---\n{text}")

  content = "\n\n".join(pages).strip()
  if not content:
    raise ValueError("PDF 中沒有可讀取的文字內容")

  return content


def reset_chat():
  st.session_state.messages = []
  st.session_state.api_messages = []
  st.session_state.pdf_loaded = False
  st.session_state.pdf_name = None


def load_pdf(uploaded_file):
  content = extract_pdf_text(uploaded_file)
  original_len = len(content)
  truncated = original_len > MAX_PDF_CHARS
  if truncated:
    content = content[:MAX_PDF_CHARS]

  st.session_state.messages = []
  st.session_state.api_messages = [
    {
      "role": "system",
      "content": (
        "你是一個 PDF 問答助手。請根據以下 PDF 內容回答使用者的問題。"
        "如果答案不在 PDF 中，請誠實說明。\n\n"
        f"PDF 內容：\n{content}"
      ),
    }
  ]
  st.session_state.pdf_loaded = True
  st.session_state.pdf_name = uploaded_file.name
  st.session_state.pdf_truncated = truncated
  st.session_state.pdf_original_len = original_len


def main():
  st.set_page_config(page_title="PDF 問答助手", page_icon="📄")
  st.title("📄 PDF 問答助手")

  if "messages" not in st.session_state:
    reset_chat()

  api_key = get_api_key()
  if not api_key:
    st.error("請在 .env 檔案中設定 GROQ_API_KEY")
    st.stop()

  uploaded_file = st.file_uploader("上傳 PDF 檔案", type=["pdf"])

  if uploaded_file is not None:
    if st.session_state.pdf_name != uploaded_file.name:
      try:
        load_pdf(uploaded_file)
      except ValueError as e:
        st.error(str(e))
        reset_chat()
        st.stop()

    if st.session_state.pdf_loaded:
      if st.session_state.pdf_truncated:
        st.info(
          f"PDF 內容過長，已截斷為前 {MAX_PDF_CHARS} 字"
          f"（原文共 {st.session_state.pdf_original_len} 字）"
        )
      else:
        st.success(f"已讀取 PDF：{st.session_state.pdf_name}")

  for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
      st.markdown(msg["content"])

  if st.session_state.pdf_loaded:
    if prompt := st.chat_input("輸入你的問題..."):
      st.session_state.messages.append({"role": "user", "content": prompt})
      st.session_state.api_messages.append({"role": "user", "content": prompt})

      with st.chat_message("user"):
        st.markdown(prompt)

      with st.chat_message("assistant"):
        with st.spinner("思考中..."):
          client = Groq(api_key=api_key)
          response = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.api_messages,
          )
          answer = response.choices[0].message.content
          st.markdown(answer)

      st.session_state.messages.append({"role": "assistant", "content": answer})
      st.session_state.api_messages.append({"role": "assistant", "content": answer})
  else:
    st.info("請先上傳 PDF 檔案，再開始提問。")


if __name__ == "__main__":
  main()
