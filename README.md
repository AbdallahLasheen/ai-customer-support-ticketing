# Intelligent Customer Support & Ticketing System

**Lab Work 6 — LangChain Agents Applications | Application 1**

---

## Overview

An agent-based customer support demo built with **LangChain**. It combines **Retrieval-Augmented Generation (RAG)** over a local knowledge base, **keyword issue classification**, **conversation history** (passed explicitly on each turn), and **JSON ticket storage**.

```
User Query
    │
    ▼
LangChain tool-calling agent (Groq or Anthropic — see Configuration)
    │
    ├── search_knowledge_base   ──► FAISS + local embeddings ──► KB docs
    ├── classify_customer_issue ──► Keyword classifier
    ├── create_support_ticket   ──► JSON files in tickets/
    └── list_open_tickets       ──► Ticket reader
    │
    ▼
Structured reply (category / priority / solution / ticket ID)
```

---

## Features

| Feature | Implementation |
|--------|----------------|
| Document ingestion | `DirectoryLoader` + `TextLoader` on `knowledge_base/` |
| Chunking | `RecursiveCharacterTextSplitter` (chunk size 500, overlap 80) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, Hugging Face cache) |
| Vector store | In-memory **FAISS** |
| Retrieval | Top **4** chunks per query; answer chain uses **LCEL** (prompt → LLM → string output) |
| Agent | `langchain.agents.create_agent` with four `@tool` functions |
| Chat memory | Turns stored as `{role, content}`; each step is converted to LangChain `HumanMessage` / `AIMessage` via `dict_messages_to_lc()` and passed to the agent. Helpers: `InMemoryChatMessageHistory`, `history_from_dicts()`, `new_conversation_memory()`. |
| Unknown-query fallback | If retrieval returns **no chunks** or an **empty** model answer, `search_knowledge_base` returns a `[KB_FALLBACK]` instruction so the agent answers honestly and can suggest a ticket. |
| Tool error handling | `try/except` on KB search, classification, ticket create, list tickets; corrupt ticket JSON files are skipped when loading. |
| Classification | Rule-based: 8 categories × 4 priority levels |
| Tickets | One JSON file per ticket under `tickets/` |
| LLM backend | **Groq** (`GROQ_API_KEY`) preferred when set; optional **Anthropic** (`ANTHROPIC_API_KEY`) |
| Web UI | **Streamlit** IDE-style layout (`streamlit_app.py`) |

---

## Project structure

```
customer_support_agent/
├── src/
│   ├── __init__.py
│   └── agent.py           # RAG pipeline, tools, agent graph, CLI entrypoint
├── knowledge_base/        # .txt (and optional .pdf) sources for RAG
├── tickets/               # Auto-created JSON tickets
├── streamlit_app.py       # Streamlit GUI
├── requirements.txt
├── .env.example           # Environment template (no secrets)
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- At least one LLM API key:
  - **Groq**: sign up at [console.groq.com](https://console.groq.com) (keys typically start with `gsk_`)
  - **Anthropic**: [console.anthropic.com](https://console.anthropic.com) (keys typically start with `sk-ant-`)

**Never commit real API keys.** Use `.env` locally only (see `.gitignore`).

---

## Setup

### 1. Virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

On first run, the embedding model downloads once (~80 MB) into the Hugging Face cache.

### 3. Environment variables

Copy the template and edit **your** `.env` (do not paste keys into README or chat):

```bash
copy .env.example .env          # Windows
cp .env.example .env            # Linux / macOS
```

**Groq (recommended for quick testing)**

```env
GROQ_API_KEY=your_groq_key_here
```

Optional model override:

```env
GROQ_MODEL=llama-3.3-70b-versatile
```

**Anthropic**

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**If both keys are present**, the app uses **Groq** by default. To force Anthropic:

```env
LLM_PROVIDER=anthropic
```

The app loads `customer_support_agent/.env` with **override** so values in `.env` take precedence over stale shell variables when supported by the entrypoint.

---

## How to run

### Streamlit UI (recommended)

From `customer_support_agent/`:

```bash
python -m streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (e.g. `http://127.0.0.1:8501`).

The sidebar includes **Chat**, **Tickets**, and **Architecture** (renders this README). Use **New conversation** to reset chat and bump the session counter.

If a request fails, expand **Error details** to see the underlying message (without exposing your key).

### CLI (interactive)

```bash
python src/agent.py
```

| Input | Action |
|-------|--------|
| `tickets` | List tickets in the console |
| `quit` / `exit` / `bye` | Exit |

On **Windows**, console output uses UTF-8 where possible so emoji/log text in `agent.py` prints reliably.

---

## Usage (how to use the app)

### Streamlit

1. Start the app (see **How to run** → Streamlit) and open the browser URL.
2. Stay on **Chat** (sidebar). Wait for the first load if embeddings are still downloading.
3. Type a question in the box at the bottom and send it. Examples:
   - *“I was charged twice on my subscription — what should I do?”*
   - *“I can’t log in after enabling 2FA and I lost my phone.”*
   - *“We get Error 500 on the dashboard; the whole team is blocked. Contact: name@company.com”*
4. Read the reply. You should often see a short **issue summary** (category / priority), **next steps** from the knowledge base, and sometimes a **ticket ID** (e.g. `TKT-…`) if the agent escalates.
5. **Tickets** (sidebar): open to see saved tickets and check that **STATS → Tickets** matches.
6. **Architecture** (sidebar): opens this README inside the app for quick reference.
7. **New conversation**: clears chat history and increases the **Sessions** counter (stats).
8. If you see a generic API error, open **Error details** under the message to read the real error (fix keys in `.env`, then rerun).

### CLI (`python src/agent.py`)

1. Run the command from `customer_support_agent/`.
2. Type normal customer messages at the prompt.
3. Special inputs:
   - `tickets` — print all ticket JSON summaries to the console.
   - `quit`, `exit`, or `bye` — stop the session.

### What “working correctly” looks like

- Answers mention content consistent with `knowledge_base/` (billing, account, errors, etc.).
- For serious or unresolved issues, the agent may call **create_support_ticket**; a new file appears under `tickets/` and the reply includes a **TKT-** id.

---

## Behaviour notes

- **Structured answers**: The system prompt asks the model to end with blocks such as **ISSUE SUMMARY**, **SOLUTION / NEXT STEPS**, and **TICKET ID** when relevant.
- **Tickets**: Creating a ticket writes `tickets/TKT-xxxxxxxx.json`. You can confirm in the Streamlit **Tickets** view or on disk.
- **Agent cache (Streamlit)**: The loaded agent is cached; the cache key includes a fingerprint of your LLM-related env vars so changing keys in `.env` picks up a fresh agent after rerun.

---

## RAG pipeline (summary)

- Load all `*.txt` under `knowledge_base/`.
- Split with `RecursiveCharacterTextSplitter`.
- Build FAISS with `HuggingFaceEmbeddings`.
- Retrieval: `k=4`; answer path chains retriever context + user question through the **same** chat model used for the agent.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| No API key error | Set `GROQ_API_KEY` and/or `ANTHROPIC_API_KEY` in `.env` (correct variable names). |
| Wrong provider used | If both keys exist, default is Groq; set `LLM_PROVIDER=anthropic` for Anthropic only. |
| Auth / quota errors | Check the provider dashboard; expand **Error details** in Streamlit. |
| First run slow | One-time embedding model download. |
| `ModuleNotFoundError` | `pip install -r requirements.txt` inside the activated venv. |
| Empty RAG answers | Ensure `.txt` files exist under `knowledge_base/`. |
| Port already in use | Stop the old Streamlit process or run on another port: `streamlit run streamlit_app.py --server.port 8502` |

---

## Extending the project

- Add more `.txt` (or supported) files under `knowledge_base/`.
- Add a new `@tool` in `src/agent.py` and register it in `build_agent()`.
- Adjust models via `GROQ_MODEL`, `ANTHROPIC_MODEL`, or `get_chat_llm()` in `agent.py`.

---

## LangChain concepts (current code)

| Concept | Where |
|---------|--------|
| Tool-calling agent | `create_agent` in `build_agent()` |
| Custom tools | `@tool` on four functions |
| RAG | FAISS retriever + LCEL chain in `build_rag_chain()` |
| Document loading / split | `DirectoryLoader`, `RecursiveCharacterTextSplitter` |
| Embeddings | `HuggingFaceEmbeddings` |

---

*Lab Work 6 — LangChain Agents Applications*  
*Application 1: Intelligent Customer Support & Ticketing System*
