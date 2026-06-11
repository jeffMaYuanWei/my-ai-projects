#!/usr/bin/env python3
"""Day 5: Tool Use — 使用 Groq Function Calling 查詢天氣"""

import json
import os
import sys

import requests
from dotenv import load_dotenv
from groq import Groq

MODEL = "llama-3.3-70b-versatile"


def get_api_key():
  load_dotenv()
  return os.getenv("GROQ_API_KEY")


def get_weather(location: str) -> str:
  """查詢指定城市的天氣（使用 wttr.in 免費 API）"""
  url = f"https://wttr.in/{location}?format=j1&lang=zh-tw"
  headers = {"User-Agent": "curl/7.64.1"}

  response = requests.get(url, headers=headers, timeout=10)
  response.raise_for_status()
  data = response.json()

  current = data["current_condition"][0]
  area = data["nearest_area"][0]
  city = area["areaName"][0]["value"]
  country = area["country"][0]["value"]
  weather = current["weatherDesc"][0]["value"]
  temp_c = current["temp_C"]
  humidity = current["humidity"]
  wind_speed = current["windspeedKmph"]

  return (
    f"地點：{city}，{country}\n"
    f"天氣：{weather}\n"
    f"氣溫：{temp_c}°C\n"
    f"濕度：{humidity}%\n"
    f"風速：{wind_speed} km/h"
  )


# 步驟 1：定義工具（Tool Schema）
# 告訴 AI「有哪些工具可以用」，包含名稱、說明、參數格式
# AI 不會真的執行這個函式，只會根據這份說明決定要不要呼叫
TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查詢指定城市或地區的即時天氣資訊",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "城市名稱，例如：台北、Tokyo、London",
          }
        },
        "required": ["location"],
      },
    },
  }
]

# 步驟 2：建立工具名稱與實際函式的對應表
# 當 AI 要求呼叫某個工具時，我們用這張表找到要執行的 Python 函式
AVAILABLE_FUNCTIONS = {
  "get_weather": get_weather,
}


def run_tool_call(tool_call) -> str:
  """執行單一工具呼叫，並回傳字串結果給 AI"""
  function_name = tool_call.function.name
  arguments = json.loads(tool_call.function.arguments)

  if function_name not in AVAILABLE_FUNCTIONS:
    return f"錯誤：未知的工具 {function_name}"

  try:
    return AVAILABLE_FUNCTIONS[function_name](**arguments)
  except Exception as e:
    return f"查詢失敗：{e}"


def chat_with_tools(client: Groq, messages: list) -> str:
  """帶有 Function Calling 的對話流程"""

  # 步驟 3：第一次呼叫 API，把 tools 傳給模型
  # 模型會閱讀使用者問題，自行判斷是否需要呼叫工具
  response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
  )

  assistant_message = response.choices[0].message

  # 步驟 4：檢查模型是否要求呼叫工具
  # 若 message.tool_calls 有內容，代表 AI 想使用工具（例如查天氣）
  # 若沒有 tool_calls，代表 AI 可直接回答，不需查外部資料
  if not assistant_message.tool_calls:
    return assistant_message.content or ""

  # 把模型的「工具呼叫請求」加入對話歷史
  # 這一步很重要：後續 API 需要完整的對話脈絡
  messages.append({
    "role": "assistant",
    "content": assistant_message.content or "",
    "tool_calls": [
      {
        "id": tool_call.id,
        "type": "function",
        "function": {
          "name": tool_call.function.name,
          "arguments": tool_call.function.arguments,
        },
      }
      for tool_call in assistant_message.tool_calls
    ],
  })

  # 步驟 5：執行工具，並把結果以 role="tool" 回傳給模型
  # 每個 tool_call 都要對應一筆工具結果，並用 tool_call_id 配對
  for tool_call in assistant_message.tool_calls:
    print(f"[系統] AI 決定呼叫工具：{tool_call.function.name}")
    print(f"[系統] 參數：{tool_call.function.arguments}")

    result = run_tool_call(tool_call)
    print(f"[系統] 工具回傳結果：\n{result}\n")

    messages.append({
      "role": "tool",
      "tool_call_id": tool_call.id,
      "content": result,
    })

  # 步驟 6：第二次呼叫 API
  # 模型已收到工具查詢結果，會整理成自然語言回答使用者
  final_response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
  )

  return final_response.choices[0].message.content or ""


def main() -> None:
  api_key = get_api_key()
  if not api_key:
    print("錯誤：請在 .env 檔案中設定 GROQ_API_KEY")
    sys.exit(1)

  client = Groq(api_key=api_key)
  messages = []

  print("Groq 工具呼叫示範已啟動（輸入 exit 結束）")
  print("試試問：「台北今天天氣如何？」或「幫我查東京的天氣」\n")

  while True:
    question = input("你：").strip()
    if not question:
      continue
    if question.lower() == "exit":
      print("再見！")
      break

    messages.append({"role": "user", "content": question})

    answer = chat_with_tools(client, messages)
    messages.append({"role": "assistant", "content": answer})

    print(f"\nAI：{answer}\n")


if __name__ == "__main__":
  main()
