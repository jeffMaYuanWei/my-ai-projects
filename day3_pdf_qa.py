#!/usr/bin/env python3
"""Day 3: PDF Q&A — 讀取 PDF 並連續提問"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

MODEL = "llama-3.3-70b-versatile"
MAX_PDF_CHARS = 3000


def get_api_key():
  load_dotenv()
  return os.getenv("GROQ_API_KEY")


def read_pdf(path: str) -> str:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"找不到檔案：{path}")

  reader = PdfReader(path)
  pages = []
  for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
      pages.append(f"--- 第 {i} 頁 ---\n{text}")

  content = "\n\n".join(pages).strip()
  if not content:
    raise ValueError("PDF 中沒有可讀取的文字內容")

  return content


def main() -> None:
  api_key = get_api_key()
  if not api_key:
    print("錯誤：請在 .env 檔案中設定 GROQ_API_KEY")
    sys.exit(1)

  pdf_path = input("請輸入 PDF 檔案路徑：").strip()
  if not pdf_path:
    print("未輸入路徑，程式結束。")
    return

  try:
    pdf_content = read_pdf(pdf_path)
    original_len = len(pdf_content)
    if original_len > MAX_PDF_CHARS:
      pdf_content = pdf_content[:MAX_PDF_CHARS]
  except (FileNotFoundError, ValueError) as e:
    print(f"錯誤：{e}")
    sys.exit(1)

  client = Groq(api_key=api_key)
  messages = [
    {
      "role": "system",
      "content": (
        "你是一個 PDF 問答助手。請根據以下 PDF 內容回答使用者的問題。"
        "如果答案不在 PDF 中，請誠實說明。\n\n"
        f"PDF 內容：\n{pdf_content}"
      ),
    }
  ]

  print(f"\n已讀取 PDF：{pdf_path}")
  if original_len > MAX_PDF_CHARS:
    print(f"PDF 內容過長，已截斷為前 {MAX_PDF_CHARS} 字（原文共 {original_len} 字）")
  print("可以開始提問（輸入 exit 結束）\n")

  while True:
    question = input("你：").strip()
    if not question:
      continue
    if question.lower() == "exit":
      print("再見！")
      break

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
      model=MODEL,
      messages=messages,
    )

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})

    print(f"\nAI：{answer}\n")


if __name__ == "__main__":
  main()
