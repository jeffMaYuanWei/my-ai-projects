#!/usr/bin/env python3
"""Day 2: Chatbot — 使用 Groq 進行連續對話"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

MODEL = "llama-3.3-70b-versatile"


def get_api_key():
  load_dotenv()
  return os.getenv("GROQ_API_KEY")


def main() -> None:
  api_key = get_api_key()
  if not api_key:
    print("錯誤：請在 .env 檔案中設定 GROQ_API_KEY")
    sys.exit(1)

  client = Groq(api_key=api_key)
  messages = []

  print("Groq 聊天機器人已啟動（輸入 exit 結束）\n")

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
