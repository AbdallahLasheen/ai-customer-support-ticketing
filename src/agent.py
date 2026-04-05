"""
=============================================================
  Intelligent Customer Support & Ticketing System
  Lab Work 6 - LangChain Agents Applications
  Application 1 - Project 9
=============================================================

Architecture:
  - RAG pipeline over knowledge base documents
  - LangChain agent with multiple tools
  - Conversation memory
  - Automatic issue classification
  - Ticket creation & storage
"""

import os
import sys
import json
import uuid
import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
from typing import Any, Optional

# ── LangChain core ─────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# ── RAG components ──────────────────────────────────────────
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv

# ── Paths (project root = parent of src/) ───────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
KB_DIR        = BASE_DIR / "knowledge_base"
TICKETS_DIR   = BASE_DIR / "tickets"
TICKETS_DIR.mkdir(exist_ok=True)

# ── Issue categories ─────────────────────────────────────────
CATEGORIES = {
    "billing":        ["charge", "payment", "invoice", "refund", "subscription", "cancel", "price", "billing"],
    "technical":      ["error", "crash", "bug", "slow", "not working", "broken", "fail", "timeout", "loading"],
    "account":        ["password", "login", "account", "locked", "2fa", "email", "reset", "access", "permission"],
    "integration":    ["api", "integration", "slack", "salesforce", "connect", "sync", "webhook", "key"],
    "data":           ["export", "import", "data", "file", "upload", "download", "backup", "csv"],
    "mobile":         ["mobile", "app", "ios", "android", "phone", "notification", "offline"],
    "performance":    ["slow", "speed", "fast", "performance", "lag", "loading", "report"],
    "general":        [],   # fallback
}

PRIORITY_KEYWORDS = {
    "urgent":   ["urgent", "critical", "emergency", "down", "outage", "cannot work", "blocked", "immediately"],
    "high":     ["important", "asap", "soon", "broken", "not working", "failing", "error"],
    "medium":   ["issue", "problem", "help", "question", "how to"],
    "low":      ["wondering", "curious", "suggestion", "feedback", "when", "will"],
}

# Fallback when retrieval returns no chunks or the model returns an empty answer
KB_UNKNOWN_FALLBACK = (
    "[KB_FALLBACK] No matching documentation was found in the knowledge base for this query. "
    "Tell the customer honestly that this topic is not covered in the internal docs, offer general "
    "goodwill, and suggest creating a support ticket or speaking with a human agent if they need a firm answer."
)


# ═══════════════════════════════════════════════════════════
#   CONVERSATION MEMORY (LangChain chat message history)
# ═══════════════════════════════════════════════════════════

def dict_messages_to_lc(messages: list[dict]) -> list[BaseMessage]:
    """Map stored {role, content} turns to LangChain messages for the agent."""
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "")
        text = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=text))
        elif role == "assistant":
            out.append(AIMessage(content=text))
    return out


def new_conversation_memory() -> InMemoryChatMessageHistory:
    """Empty LangChain in-memory history (useful for tests or alternate UIs)."""
    return InMemoryChatMessageHistory()


def history_from_dicts(chat_history: list[dict]) -> InMemoryChatMessageHistory:
    """Hydrate LangChain InMemoryChatMessageHistory from persisted role dicts."""
    h = InMemoryChatMessageHistory()
    for m in dict_messages_to_lc(chat_history):
        h.add_message(m)
    return h


# ═══════════════════════════════════════════════════════════
#   BUILD RAG PIPELINE
# ═══════════════════════════════════════════════════════════

def build_rag_chain(llm):
    """Load knowledge base docs → chunk → embed → FAISS → Q&A chain."""
    print("📚  Loading knowledge base documents …")

    loader = DirectoryLoader(str(KB_DIR), glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"    Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"    Split into {len(chunks)} chunks.")

    print("    Building vector store (HuggingFace embeddings) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create a simple Q&A chain using LangChain's LCEL
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    qa_chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅  RAG pipeline ready.\n")
    return qa_chain, retriever


# ═══════════════════════════════════════════════════════════
#   UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def classify_issue(text: str) -> dict:
    """Rule-based + keyword classifier for category and priority."""
    lower = text.lower()

    # Category
    category = "general"
    best_hits = 0
    for cat, keywords in CATEGORIES.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits = hits
            category = cat

    # Priority
    priority = "medium"
    for prio, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            priority = prio
            break

    return {"category": category, "priority": priority}


def save_ticket(ticket: dict) -> str:
    """Persist ticket as JSON; return the file path."""
    filepath = TICKETS_DIR / f"{ticket['ticket_id']}.json"
    with open(filepath, "w") as f:
        json.dump(ticket, f, indent=2)
    return str(filepath)


def load_all_tickets() -> list[dict]:
    """Return all stored tickets sorted by creation time."""
    tickets: list[dict] = []
    for fp in sorted(TICKETS_DIR.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                tickets.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return tickets


# ═══════════════════════════════════════════════════════════
#   AGENT TOOLS
# ═══════════════════════════════════════════════════════════

# Injected after bootstrap (tools read these globals).
_rag_chain: Optional[Any] = None
_kb_retriever: Optional[Any] = None


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal support knowledge base for answers to customer questions.
    Use this tool FIRST for any technical, billing, account, or product question.
    Input: a concise natural-language question.
    Output: an answer sourced from official documentation.
    """
    if _rag_chain is None:
        return "Knowledge base not initialised."
    try:
        if _kb_retriever is not None:
            docs = _kb_retriever.get_relevant_documents(query)
            if not docs:
                return KB_UNKNOWN_FALLBACK
        result = _rag_chain.invoke(query)
        text = result if isinstance(result, str) else str(result)
        if not text or not str(text).strip():
            return KB_UNKNOWN_FALLBACK
        return text
    except Exception as exc:
        return f"Knowledge base search failed: {exc}"


@tool
def classify_customer_issue(issue_description: str) -> str:
    """
    Classify a customer issue into a category and priority level.
    Use this tool when you need to determine what kind of issue the customer has.
    Input: a description of the customer's problem.
    Output: JSON with 'category' and 'priority' fields.
    Categories: billing, technical, account, integration, data, mobile, performance, general.
    Priority levels: urgent, high, medium, low.
    """
    try:
        result = classify_issue(issue_description)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps(
            {"error": "classification_failed", "detail": str(exc), "category": "general", "priority": "medium"},
            indent=2,
        )


@tool
def resolve_ticket_by_id(ticket_id: str) -> str:
    """
    Use this tool ONLY when the customer confirms their issue is fixed or 
    explicitly asks to close their ticket.
    Input: The Ticket ID (e.g., TKT-A7C633F6)
    """
    try:
        filepath = TICKETS_DIR / f"{ticket_id.upper()}.json"
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                ticket_data = json.load(f)
            
            ticket_data['status'] = 'resolved'
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ticket_data, f, indent=2)
                
            return f"Success: Ticket {ticket_id} has been marked as RESOLVED in the system."
        else:
            return f"Error: Ticket {ticket_id} not found."
    except Exception as e:
        return f"Failed to resolve ticket: {str(e)}"

@tool
def create_support_ticket(
    customer_name: str,
    customer_email: str,
    issue_summary: str,
    issue_description: str,
    suggested_solution: str = "",
) -> str:
    """
    Create and save a support ticket for a customer issue.
    Use this tool when the customer's issue cannot be resolved immediately
    or when the customer explicitly requests escalation.
    Inputs:
      customer_name       - full name of the customer
      customer_email      - customer's email address
      issue_summary       - one-line summary (≤80 characters)
      issue_description   - full description of the issue
      suggested_solution  - any solution already suggested (optional)
    Output: JSON with ticket details including ticket_id and status.
    """
    try:
        classification = classify_issue(issue_description)
        ticket = {
            "ticket_id":          f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "status":             "open",
            "created_at":         datetime.datetime.utcnow().isoformat() + "Z",
            "customer_name":      customer_name,
            "customer_email":     customer_email,
            "issue_summary":      issue_summary,
            "issue_description":  issue_description,
            "category":           classification["category"],
            "priority":           classification["priority"],
            "suggested_solution": suggested_solution,
            "agent_notes":        "",
        }
        filepath = save_ticket(ticket)
        return json.dumps({
            "ticket_id":   ticket["ticket_id"],
            "status":      ticket["status"],
            "category":    ticket["category"],
            "priority":    ticket["priority"],
            "saved_to":    filepath,
            "message":     "Ticket created successfully. A support agent will follow up within the SLA window.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": "ticket_creation_failed", "detail": str(exc)}, indent=2)


@tool
def list_open_tickets(filter_category: str = "") -> str:
    """
    List all support tickets, optionally filtered by category.
    Use this tool when an admin asks to view or review existing tickets.
    Input: optional category filter (billing/technical/account/integration/data/mobile/performance/general).
    Output: JSON list of tickets with key fields.
    """
    try:
        all_tickets = load_all_tickets()
    except Exception as exc:
        return json.dumps({"error": "list_tickets_failed", "detail": str(exc)}, indent=2)
    if filter_category:
        all_tickets = [t for t in all_tickets if t.get("category") == filter_category.lower()]

    summary = [
        {
            "ticket_id": t["ticket_id"],
            "status":    t["status"],
            "priority":  t["priority"],
            "category":  t["category"],
            "summary":   t["issue_summary"],
            "customer":  t["customer_name"],
            "created":   t["created_at"],
        }
        for t in all_tickets
    ]
    if not summary:
        return "No tickets found."
    return json.dumps(summary, indent=2)


# ═══════════════════════════════════════════════════════════
#   AGENT ASSEMBLY
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Alex, an expert AI customer support agent for TechSupport Pro — 
a B2B SaaS productivity platform. You are helpful, professional, and empathetic.

Your responsibilities:
1. Answer customer questions accurately using the knowledge base.
2. Classify issues to understand severity and routing.
3. Create support tickets when issues need escalation or follow-up.
4. Always provide structured, clear responses.

Workflow:
- For every customer issue: FIRST search the knowledge base.
- If the issue is complex or unsolvable via documentation, create a ticket.
- Always classify the issue before creating a ticket.
- If the knowledge-base tool returns a [KB_FALLBACK] message, follow that guidance (be honest, suggest a ticket).
- End every response with a structured summary block (see format below).

Response Format:
─────────────────────────────────────
📋 ISSUE SUMMARY
  Category : <category>
  Priority : <priority>
  Status   : Resolved / Escalated / Pending

💡 SOLUTION / NEXT STEPS
  <your answer or escalation details>

🎫 TICKET ID (if created)
  <TKT-XXXXXXXX or N/A>
─────────────────────────────────────

Be concise but complete. If you cannot find an answer, be honest and create a ticket.
"""

def _make_anthropic_llm(api_key: str) -> BaseChatModel:
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    return ChatAnthropic(
        model=model,
        temperature=0.2,
        anthropic_api_key=api_key,
    )


def _make_groq_llm(api_key: str) -> BaseChatModel:
    from langchain_groq import ChatGroq

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0.2, groq_api_key=api_key)


def get_chat_llm() -> BaseChatModel:
    """
    Resolve chat model from environment (GROQ_API_KEY and/or ANTHROPIC_API_KEY).
    If both are set, Groq is used unless LLM_PROVIDER=anthropic.
    """
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if provider == "groq":
        if not groq_key:
            raise EnvironmentError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return _make_groq_llm(groq_key)
    if provider == "anthropic":
        if not anthropic_key:
            raise EnvironmentError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        return _make_anthropic_llm(anthropic_key)

    if groq_key:
        return _make_groq_llm(groq_key)
    if anthropic_key:
        return _make_anthropic_llm(anthropic_key)

    raise EnvironmentError(
        "No LLM API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY in .env. "
        "If both are set, Groq is used; set LLM_PROVIDER=anthropic for Anthropic."
    )


def build_agent(llm):
    """Assemble the tool-calling agent."""
    tools = [
        search_knowledge_base,
        classify_customer_issue,
        create_support_ticket,
        list_open_tickets,
        resolve_ticket_by_id,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        debug=False,
    )
    return agent


def bootstrap_support_agent():
    """
    Initialize LLM, RAG pipeline, and agent graph.
    Sets global _rag_chain for knowledge-base tools.
    """
    llm = get_chat_llm()

    global _rag_chain, _kb_retriever
    _rag_chain, _kb_retriever = build_rag_chain(llm)
    return build_agent(llm)


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content) if content is not None else ""


def response_to_assistant_text(response: Any) -> str:
    """Extract plain assistant text from create_agent invoke() result."""
    if isinstance(response, dict) and "messages" in response:
        assistant_message = response["messages"][-1]
        if hasattr(assistant_message, "content"):
            return _normalize_message_content(assistant_message.content)
        if isinstance(assistant_message, dict):
            return _normalize_message_content(assistant_message.get("content", ""))
        return str(assistant_message)
    return str(response)


def chat_turn(agent: Any, chat_history: list[dict], user_input: str) -> tuple[str, list[dict]]:
    """
    Run one user turn using LangChain message history (user/assistant only in state).
    Conversation context is rebuilt from chat_history into BaseMessage list each call.
    """
    lc_messages = dict_messages_to_lc(chat_history) + [HumanMessage(content=user_input)]
    response = agent.invoke({"messages": lc_messages})
    assistant_text = response_to_assistant_text(response)
    new_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_text},
    ]
    return assistant_text, new_history

def resolve_ticket_by_id(ticket_id: str) -> bool:
    """Helper function for the UI to mark a ticket as resolved."""
    # TICKETS_DIR متعرفة عندك فوق في الملف
    filepath = TICKETS_DIR / f"{ticket_id.upper()}.json"
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['status'] = 'resolved'
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        return False
    except Exception:
        return False
        
# ═══════════════════════════════════════════════════════════
#   MAIN CHAT LOOP
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TechSupport Pro — Intelligent Support Agent")
    print("  Lab Work 6 | Application 1")
    print("=" * 60)

    agent_executor = bootstrap_support_agent()
    chat_history = []

    print("\nAgent ready. Type 'quit' to exit, 'tickets' to view all tickets.\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\n🤖 Alex: Thank you for contacting TechSupport Pro. Goodbye!")
            break

        if user_input.lower() == "tickets":
            tickets = load_all_tickets()
            if not tickets:
                print("📭  No tickets created yet.")
            else:
                print(f"\n📁  {len(tickets)} ticket(s) on record:")
                for t in tickets:
                    print(f"   [{t['priority'].upper()}] {t['ticket_id']} — {t['issue_summary']} ({t['category']})")
            continue

        print("\n🤖 Alex: ", end="", flush=True)
        try:
            assistant_text, chat_history = chat_turn(
                agent_executor, chat_history, user_input
            )
            print(assistant_text)
        except Exception as exc:
            print(f"⚠️  An error occurred: {exc}")
            print("Please try rephrasing your question or contact support directly.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
