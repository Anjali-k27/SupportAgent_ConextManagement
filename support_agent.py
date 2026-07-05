"""
Enterprise AI Support Platform
Session 5 of 12 — Context Management & Summarization

Extends Session 3 with SQLite-backed checkpointing.
State survives process restarts. Conversations are isolated
by thread_id. Time travel and HITL are now possible.

Run server: python api.py  → http://localhost:8000
Run CLI:    python support_agent.py
"""

import os
import time
import operator
import json
import uuid
import sqlite3
from typing import TypedDict, Annotated, Literal, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# ── Environment setup ──────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY='your-key-here'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("[System] Gemini 2.5 Flash initialized | temperature=0")


# ── ReAct Constants (Session 3) ─────────────────────────────────────────────

MAX_ITERATIONS    = 5
CONTEXT_THRESHOLD = 12

print(f"[ReAct] MAX_ITERATIONS={MAX_ITERATIONS} | "
      f"CONTEXT_THRESHOLD={CONTEXT_THRESHOLD}")

# ── Summarization Constants (Session 5) ──────────────────────────────────────

SUMMARY_THRESHOLD = 8   # messages before summarization triggers

print(f"[Summarization] SUMMARY_THRESHOLD={SUMMARY_THRESHOLD}")

# ── Checkpointer (Session 4) ─────────────────────────────────────────────────

DB_PATH = 'support.db'

_db_conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_db_conn)

print(f"[Checkpointer] SQLite initialized → {DB_PATH}")

# ── Custom Message Reducer (Session 5) ────────────────────────────────────────

