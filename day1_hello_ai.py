#!/usr/bin/env python3
"""Day 1: Hello AI — 使用 Groq 回答你的問題"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq


def get_api_key():
  load_dotenv()
  return os.getenv("GROQ_API_KEY")


def main() -> None:
  api_key = get_api_key()
  if not api_key:
    print("錯誤：請在 .env 檔案中設定 GROQ_API_KEY")
    sys.exit(1)

  client = Groq(api_key=api_key)

  question = input("請輸入你的問題：").strip()
  if not question:
    print("未輸入問題，程式結束。")
    return

  response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": question}],
  )

  print("\nAI 回答：")
  print(response.choices[0].message.content)


if __name__ == "__main__":
  main()
