# Week Summary — AI Learning (Day 1–5)

本週學習成果整理。Each day builds on the last.

---

## Day 1：AI Chatbot

**檔案：** `day1_hello_ai.py`

### What I Learned

- **API basics** — How to call an LLM with Groq API
- **Environment variables** — Read `GROQ_API_KEY` from `.env` (keep secrets safe)
- **Single-turn chat** — User asks one question → AI gives one answer

### Core Concept

> A chatbot starts with: **input → API call → output**

The model does not remember past messages in Day 1. Each question is independent.

---

## Day 2：Chatbot with Memory

**檔案：** `day2_chatbot.py`

### What I Learned

- **Message history** — Use a `messages` list to store the full conversation
- **Multi-turn chat** — Send all past messages to the API every time
- **Context** — The AI can remember what you said before (e.g. your name)

### Core Concept

> Memory = **keep all messages and send them again** on each API call

```
messages = [
  {"role": "user", "content": "我叫小明"},
  {"role": "assistant", "content": "你好小明！"},
  {"role": "user", "content": "我叫什麼？"},  ← AI knows from history
]
```

---

## Day 3：PDF Q&A (RAG)

**檔案：** `day3_pdf_qa.py`

### What I Learned

- **Read PDF** — Extract text with `pypdf`
- **RAG (Retrieval-Augmented Generation)** — Put document content into the prompt so AI answers from your file
- **System prompt** — Tell the AI: "Answer only from this PDF"
- **Truncate** — Use first 3000 characters to stay within token limits

### Core Concept

> RAG = **give the AI your data first**, then ask questions

Flow:

1. Read PDF → get text
2. Put text in system message
3. User asks questions → AI answers from that content

This is a simple RAG pattern (no vector database yet).

---

## Day 5：Function Calling + Tool Use

**檔案：** `day5_tool_use.py`

### What I Learned

- **Tool / Function schema** — Describe tools for the AI (name, description, parameters)
- **Function calling** — AI decides *when* to call a tool (e.g. weather question → call `get_weather`)
- **Two-step flow** — (1) AI requests tool → (2) we run it → (3) send result back → (4) AI gives final answer
- **External API** — Connect real data (wttr.in weather API)

### Core Concept

> The AI **chooses tools**. Your code **runs them**.

```
User: "台北天氣如何？"
  ↓
AI: tool_calls → get_weather(location="台北")
  ↓
Your code: call wttr.in API
  ↓
AI: "台北目前氣溫 25°C，多雲..."
```

---

## Skills Map

| Day | Skill | English |
|-----|-------|---------|
| 1 | 單次問答 | Single-turn Q&A |
| 2 | 對話記憶 | Conversation memory |
| 3 | 文件問答 | Document Q&A / RAG |
| 5 | 工具呼叫 | Function calling / Tool use |

---

## Tech Stack

- **API:** Groq
- **Model:** `llama-3.3-70b-versatile`
- **Libraries:** `groq`, `python-dotenv`, `pypdf`, `requests`

---

## Next Steps (Ideas)

- Day 4: Streamlit web UI (`day4_streamlit_app.py`)
- Better RAG: chunking, embeddings, vector search
- More tools: calculator, search, database
