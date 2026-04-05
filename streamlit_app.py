"""
Streamlit UI for TechSupport Pro — IDE-style support desk layout.

Run from the project root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import streamlit as st

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

from src.agent import bootstrap_support_agent, chat_turn, load_all_tickets

PRIORITY_COLORS: dict[str, str] = {
    "urgent": "#f43f5e",
    "high": "#fb923c",
    "medium": "#fbbf24",
    "low": "#4ade80",
}

IDE_TABS = (
    ("chat.py", "chat"),
    ("tickets.json", "tickets"),
    ("architecture.md", "architecture"),
)


def _esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _load_readme() -> str:
    p = ROOT / "README.md"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return "_README.md not found._"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

          :root {
            --ide-bg: #0b0f19;
            --ide-surface: #161b22;
            --ide-border: #30363d;
            --ide-accent: #3b82f6;
            --ide-text: #f8fafc;
            --ide-muted: #94a3b8;
            --ide-green: #10b981;
            --ide-font: 'Inter', system-ui, sans-serif;
            --ide-mono: 'JetBrains Mono', ui-monospace, monospace;
            --ide-base-size: 17px;
            --ide-chat-size: 17px;
          }

          html { font-size: var(--ide-base-size); }

          .stApp {
            font-family: var(--ide-font) !important;
            background: var(--ide-bg) !important;
            color: var(--ide-text) !important;
          }

          .stApp * {
            -webkit-font-smoothing: antialiased;
          }

          section.main {
            position: relative;
            z-index: 1;
          }

          .main .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2rem;
            max-width: 1080px;
          }

          /* Sidebar */
          [data-testid="stSidebar"] {
            background: var(--ide-surface) !important;
            border-right: 1px solid var(--ide-border) !important;
            min-width: 240px !important;
          }
          [data-testid="stSidebar"] * {
            font-size: 1rem !important;
          }
          .ide-nav-label, .ide-stat-label {
            font-size: 0.7rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            color: var(--ide-muted) !important;
            margin: 0 0 0.65rem 0 !important;
          }
          .ide-stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.35rem 0;
            font-size: 1rem !important;
            color: var(--ide-text);
            border-bottom: 1px solid rgba(48,54,61,0.6);
          }
          .ide-stat-row:last-child { border-bottom: none; }
          .ide-stat-val { font-weight: 600; font-variant-numeric: tabular-nums; }
          .ide-stat-val.tickets { color: var(--ide-accent); }
          .ide-stat-val.resolved { color: var(--ide-green); }

          /* Primary / secondary buttons → IDE blue */
          button[kind="primary"] {
            background: linear-gradient(180deg, #2563eb, #1d4ed8) !important;
            color: #ffffff !important;
            border: 1px solid #3b82f6 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
          }
          button[kind="secondary"] {
            background: transparent !important;
            color: var(--ide-muted) !important;
            border: 1px solid transparent !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
          }
          button[kind="secondary"]:hover {
            color: var(--ide-text) !important;
            border-color: var(--ide-border) !important;
            background: rgba(59,130,246,0.08) !important;
          }

          /* IDE top bar */
          .ide-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            padding: 0.35rem 0 0.85rem 0;
            border-bottom: 1px solid var(--ide-border);
            margin-bottom: 1rem;
          }
          .ide-breadcrumb {
            font-family: var(--ide-mono);
            font-size: 0.95rem !important;
            color: var(--ide-muted) !important;
            font-weight: 500;
          }
          .ide-breadcrumb strong {
            color: var(--ide-text);
            font-weight: 600;
          }
          .ide-live {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.8rem !important;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--ide-green);
            border: 1px solid rgba(16,185,129,0.45);
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(16,185,129,0.1);
          }
          .ide-live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--ide-green);
            box-shadow: 0 0 10px var(--ide-green);
          }

          /* File tabs */
          .ide-tabs {
            display: flex;
            gap: 0;
            border-bottom: 1px solid var(--ide-border);
            margin-bottom: 1.25rem;
          }
          .ide-tab {
            font-family: var(--ide-mono);
            font-size: 0.9rem !important;
            padding: 0.55rem 1.1rem;
            color: var(--ide-muted);
            border-bottom: 3px solid transparent;
            margin-bottom: -1px;
          }
          .ide-tab.active {
            color: var(--ide-accent);
            border-bottom-color: var(--ide-accent);
            font-weight: 600;
          }

          /* Status line under agent */
          .ide-agent-foot {
            font-size: 0.85rem !important;
            color: var(--ide-muted) !important;
            margin-top: 0.5rem;
            font-family: var(--ide-mono);
          }

          /* Chat bubbles */
          [data-testid="stChatMessage"] {
            background: var(--ide-surface) !important;
            border: 1px solid var(--ide-border) !important;
            border-radius: 14px !important;
            padding: 1rem 1.15rem !important;
            margin-bottom: 1rem !important;
          }
          [data-testid="stChatMessage"] p,
          [data-testid="stChatMessage"] li {
            font-size: var(--ide-chat-size) !important;
            line-height: 1.65 !important;
            color: var(--ide-text) !important;
            font-weight: 450;
          }
          .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid var(--ide-accent) !important;
          }
          .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #1a2332 !important;
            border-color: rgba(59,130,246,0.35) !important;
            margin-left: auto !important;
            max-width: 92% !important;
          }

          /* Align user messages right */
          [data-testid="stChatMessageContainer"] {
            display: flex !important;
            flex-direction: column !important;
          }
          .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            align-self: flex-end !important;
          }
          .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            align-self: flex-start !important;
            max-width: 92% !important;
          }

          /* White chat input + visible Send */
          [data-testid="stChatInput"] {
            background: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.25);
          }
          [data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            font-size: 1.05rem !important;
            font-family: var(--ide-font) !important;
            min-height: 52px !important;
            font-weight: 500 !important;
          }
          [data-testid="stChatInput"] textarea::placeholder {
            color: #64748b !important;
            font-size: 1.02rem !important;
          }
          [data-testid="stChatInput"] button {
            background: var(--ide-accent) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border-radius: 10px !important;
            border: none !important;
          }
          [data-testid="stChatInput"] button:hover {
            background: #2563eb !important;
          }

          /* General markdown contrast */
          .main .stMarkdown, .main .stMarkdown p {
            color: var(--ide-text) !important;
            font-size: 1rem !important;
          }
          .main h1, .main h2, .main h3 {
            color: var(--ide-text) !important;
          }

          /* Ticket cards (main + sidebar) */
          .ts-ticket {
            font-size: 1rem !important;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            border: 1px solid var(--ide-border);
            background: var(--ide-surface);
            margin-bottom: 0.65rem;
            border-left: 4px solid var(--accent, #64748b);
            color: var(--ide-text);
          }
          .ts-ticket-id { color: var(--ide-accent); font-weight: 700; }
          .ts-ticket-meta { color: var(--ide-muted); font-size: 0.92rem !important; margin-top: 0.4rem; }

          header[data-testid="stHeader"] { background: transparent !important; }

          /* Spinner text */
          .stSpinner > div { font-size: 1rem !important; color: var(--ide-muted) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ide_tabs(active: str) -> None:
    parts: list[str] = []
    for label, key in IDE_TABS:
        cls = "ide-tab active" if active == key else "ide-tab"
        parts.append(f'<span class="{cls}">{html.escape(label)}</span>')
    st.markdown(f'<div class="ide-tabs">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_ide_header() -> None:
    st.markdown(
        '<div class="ide-top">'
        '<div class="ide-breadcrumb">techsupport_pro / <strong>support-agent</strong></div>'
        '<div class="ide-live"><span class="ide-live-dot"></span> LIVE</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_ticket_card(ticket: dict) -> None:
    pid = (ticket.get("priority") or "").lower()
    accent = PRIORITY_COLORS.get(pid, "#64748b")
    bid = _esc(ticket.get("ticket_id"))
    prio = _esc(ticket.get("priority"))
    summary = _esc(ticket.get("issue_summary"))
    cat = _esc(ticket.get("category"))
    status = _esc(ticket.get("status"))
    st.markdown(
        f'<div class="ts-ticket" style="--accent:{accent}">'
        f'<span class="ts-ticket-id">{bid}</span> '
        f'<span style="color:{accent};font-weight:600">({prio})</span><br/>'
        f"{summary}"
        f'<div class="ts-ticket-meta">{cat} · {status}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _ticket_stats(tickets: list[dict]) -> tuple[int, int]:
    total = len(tickets)
    resolved = sum(
        1
        for t in tickets
        if str(t.get("status", "")).lower() in ("resolved", "closed", "done")
    )
    return total, resolved


def _llm_env_signature() -> str:
    """Invalidate cached agent when .env keys or provider change."""
    blob = "|".join(
        [
            os.environ.get("GROQ_API_KEY") or "",
            os.environ.get("ANTHROPIC_API_KEY") or "",
            os.environ.get("LLM_PROVIDER") or "",
            os.environ.get("GROQ_MODEL") or "",
        ]
    )
    return hashlib.sha256(blob.encode()).hexdigest()


@st.cache_resource(show_spinner=False)
def get_agent(_env_signature: str) -> object:
    """`_env_signature` busts cache when .env keys change."""
    return bootstrap_support_agent()


def main() -> None:
    st.set_page_config(
        page_title="techsupport_pro · support-agent",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    if "view" not in st.session_state:
        st.session_state.view = "chat"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_counter" not in st.session_state:
        st.session_state.session_counter = 1

    try:
        tickets = load_all_tickets()
    except Exception:
        tickets = []
    t_total, t_resolved = _ticket_stats(tickets)

    with st.sidebar:
        st.markdown('<p class="ide-nav-label">NAVIGATION</p>', unsafe_allow_html=True)
        for label, key in [("Chat", "chat"), ("Tickets", "tickets"), ("Architecture", "architecture")]:
            is_active = st.session_state.view == key
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.view = key
                st.rerun()

        st.markdown('<p class="ide-stat-label" style="margin-top:1.25rem">STATS</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ide-stat-row"><span>Sessions</span>'
            f'<span class="ide-stat-val">{st.session_state.session_counter}</span></div>'
            f'<div class="ide-stat-row"><span>Tickets</span>'
            f'<span class="ide-stat-val tickets">{t_total}</span></div>'
            f'<div class="ide-stat-row"><span>Resolved</span>'
            f'<span class="ide-stat-val resolved">{t_resolved}</span></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("New conversation", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_counter = st.session_state.session_counter + 1
            st.session_state.pop("seeded_welcome", None)
            st.rerun()

    render_ide_header()
    render_ide_tabs(st.session_state.view)

    if st.session_state.view == "architecture":
        st.markdown(_load_readme())
        return

    if st.session_state.view == "tickets":
        st.subheader("🎟️ Support Ticket Management")
        
        # Load fresh tickets
        try:
            all_tickets = load_all_tickets()
        except Exception:
            all_tickets = []

        if not all_tickets:
            st.info("No tickets yet — open Chat and escalate an issue to create one.")
        else:
            # Separate tickets by status
            open_tickets = [t for t in all_tickets if t.get("status", "").lower() == "open"]
            resolved_tickets = [t for t in all_tickets if t.get("status", "").lower() == "resolved"]

            st.markdown(f"### Pending Tickets ({len(open_tickets)})")
            
            for t in reversed(open_tickets):
                # Using columns to put info on left and button on right
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    render_ticket_card(t) # الدالة دي موجودة عندك أصلاً بتعرض الكارت
                
                with col2:
                    st.write("") # Just for spacing
                    # Add resolve button
                    if st.button("Mark Resolved", key=f"res_{t['ticket_id']}"):
                        from src.agent import resolve_ticket_by_id
                        if resolve_ticket_by_id(t['ticket_id']):
                            st.toast(f"Ticket {t['ticket_id']} resolved!", icon="✅")
                            st.rerun()

            if resolved_tickets:
                st.divider()
                st.markdown(f"### Resolved History ({len(resolved_tickets)})")
                for t in reversed(resolved_tickets):
                    st.markdown(f"✔️ `{t['ticket_id']}` - {t['issue_summary']}")
        
        return # Important to stop here
    # --- Chat view ---
    agent_ok = True
    agent = None
    try:
        agent = get_agent(_llm_env_signature())
    except EnvironmentError as err:
        st.error(str(err))
        agent_ok = False
    except Exception as err:
        st.error(f"Agent initialization failed: {err}")
        agent_ok = False

    if (
        agent_ok
        and not st.session_state.messages
        and not st.session_state.get("seeded_welcome")
    ):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm **Alex**, your AI support engineer for TechSupport Pro. "
                    "Ask about billing, accounts, errors, integrations, or say you need a human — "
                    "I'll search our knowledge base and open tickets when needed.\n\n"
                    "_Agent ready · 4 tools loaded_"
                ),
            }
        ]
        st.session_state.seeded_welcome = True

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not agent_ok:
        st.stop()

    if prompt := st.chat_input("Describe your issue…"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Alex is working…"):
                err_detail: str | None = None
                try:
                    reply, st.session_state.messages = chat_turn(
                        agent, st.session_state.messages, prompt
                    )
                except Exception as exc:
                    err_detail = str(exc)
                    reply = (
                        "Sorry, I encountered an error connecting to the AI service. "
                        "Please check your API key and try again."
                    )
                    st.session_state.messages = st.session_state.messages + [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": reply},
                    ]
            st.markdown(reply)
            if err_detail:
                with st.expander("Error details (for debugging)", expanded=False):
                    st.code(err_detail, language="text")
                st.caption(
                    "Tips: use `GROQ_API_KEY=gsk_...` in `.env` (not inside `ANTHROPIC_API_KEY`). "
                    "Save the file, then stop Streamlit (Ctrl+C) and run it again — or click **Rerun** "
                    "after changing keys (cache now refreshes when keys change)."
                )


if __name__ == "__main__":
    main()
