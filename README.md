# Enterprise AI Support Platform

A production-grade AI customer support agent built session by session across 12 incremental milestones. Each session introduces one architectural concept — from a bare LLM call to a fully hardened, multi-modal, human-in-the-loop enterprise system.

**Current state: Session 5 complete — Context Management & Summarization**

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash (via `langchain-google-genai`) |
| Agent framework | LangGraph `StateGraph` |
| Persistence | SQLite (`langgraph-checkpoint-sqlite`) |
| API | FastAPI + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS (single `index.html`) |
| Runtime | Python 3.12 |

---

## Quick Start

```bash
# 1. Clone and enter the project
cd session5/phase3

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Google API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# 5. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 6. Open the UI
open http://localhost:8000
```

---

## Project Structure

```
phase3/
├── support_agent.py   # LangGraph agent — state, nodes, graph, tools
├── api.py             # FastAPI server — REST + SSE endpoints
├── index.html         # Single-page UI — chat, inspector, timeline
├── requirements.txt   # Pinned Python dependencies
├── .env               # GOOGLE_API_KEY (never commit this)
├── support.db         # SQLite checkpoint store (auto-created)
└── verify_s4.js       # Playwright verification script (Session 4)
```

---

## Session-by-Session Implementation

### Session 1 — Stateful Agent Foundation
**Goal:** Replace a bare LLM call with a LangGraph `StateGraph` that holds typed state and routes through nodes.

**What was built:**
- `SupportState` TypedDict with 17 fields: `messages`, `raw_input`, `category`, `final_response`, `is_safe`, `pii_detected`, `injection_detected`, `session_id`, `thread_id`, `iteration_count`, `tool_calls_made`, `customer_id`, `fraud_score`, `sentiment`, `resolution_status`, `escalation_required`, `system_summary`
- `classify_node` — sends user message to Gemini, extracts category (billing / technical / fraud / general)
- `route_by_category` — conditional edge that fans out to the right handler
- `billing_handler`, `technical_handler`, `fraud_handler`, `general_handler` — stub nodes that set `final_response`
- `respond_node` — final node that returns the response
- `build_graph()` — assembles the `StateGraph`, compiles with no checkpointer
- `add_messages` as the messages reducer

**Key concept:** LangGraph state is immutable per node — each node returns a partial dict that is merged into the running state snapshot.

---

### Session 2 — Tool Use & ReAct Loop
**Goal:** Give the agent tools it can call in a loop (ReAct pattern) with a hard iteration cap.

**What was built:**
- Three `@tool`-decorated functions:
  - `get_customer_details(customer_id)` — returns mock account data
  - `search_knowledge_base(query)` — returns mock KB articles
  - `check_fraud_signals(customer_id)` — returns mock fraud risk score
- `ToolNode` wired as `tool_node` in the graph
- `agent_node` — the ReAct core: builds messages, calls LLM with tools bound, checks for tool calls
- `route_after_agent` — conditional edge: `tool_calls` → `tool_node`, otherwise → `respond_node`
- `MAX_ITERATIONS = 5` guard in `agent_node` (increments `iteration_count`, hard-stops at limit)
- `trim_context()` helper — naive sliding window to keep message list under `CONTEXT_THRESHOLD = 12`
- `ToolNode` registered in `build_graph()`

**Key concept:** ReAct = Reason + Act. The agent decides which tool to call, receives the result, reasons again, and loops until it has enough information or hits the iteration cap.

---

### Session 3 — FastAPI + SSE Streaming UI
**Goal:** Expose the agent over HTTP and stream execution events to a live browser UI.

**What was built:**
- `api.py` — FastAPI application with:
  - `POST /api/run` — synchronous endpoint, returns full result JSON
  - `GET /api/stream` — SSE endpoint, streams one event per node execution
  - `GET /health` — liveness probe
  - `GET /` — serves `index.html`
- `index.html` — single-page UI with:
  - Chat input and conversation history panel
  - Execution timeline (one row per node, with timing)
  - State inspector (live JSON view of `SupportState`)
  - Tool calls panel (expandable per tool invocation)
  - Session progress tracker (S1–S12 status badges)
  - Verification test accordion
- SSE event format: `data: {"node": "...", "state": {...}, ...}\n\n`
- Thread ID generated per conversation for isolation

**Key concept:** SSE (Server-Sent Events) lets the browser receive a stream of node-level events over a single HTTP connection without WebSockets.

---

### Session 4 — SQLite Persistence & Thread Management
**Goal:** Survive process restarts. Conversations must be reloadable; state must outlive the Python process.

**What was built:**
- `SqliteSaver` from `langgraph-checkpoint-sqlite` wired into `build_graph()` as `checkpointer`
- `sqlite3.connect(DB_PATH, check_same_thread=False)` — single shared connection
- Every `graph.invoke()` / `graph.stream()` call passes `config={"configurable": {"thread_id": thread_id}}`
- Conversation history loads automatically from SQLite on re-invoke of the same thread
- `support.db` created automatically on first run
- UI: thread ID displayed; new thread button; session history list

**Key concept:** The checkpointer intercepts every state transition and writes a snapshot. On the next invocation, LangGraph reads the latest snapshot and resumes from that exact state — no application code changes required.

---

### Session 5 — Context Management & Summarization *(current)*
**Goal:** Prevent unbounded message list growth. Compress old context into a rolling summary instead of truncating it.

**What was built:**

**`deduplicate_messages` custom reducer** (replaces `add_messages`):
- Handles `RemoveMessage` deletion objects — removes matched IDs from the list
- Deduplicates additions by message ID to prevent double-writes after checkpointer replay
- Preserves all Session 4 persistence behaviour

**`summarization_node`**:
- Fires when `len(state["messages"]) >= SUMMARY_THRESHOLD` (default: 8)
- Filters messages to Human + AI-without-tool-calls before sending to Gemini (prevents invalid sequence errors)
- Strips Gemini thinking tokens (`<thinking>...</thinking>`) from the response
- Hard caps summary at 1500 characters
- Trims the message list to the last 4+ messages, always starting at a `HumanMessage` boundary
- Emits `RemoveMessage` objects for each deleted message + updated `system_summary`

**`route_after_classify`** (replaces `route_by_category`):
- Combined router from `classify_node`
- Fraud / general categories → direct to their handlers (no summarization check needed)
- Billing / technical categories → checks message count first, routes to `summarization_node` or `agent_node`

**`agent_node` update**:
- Removed `trim_context()` call
- Reads `state.get("system_summary", "")` and prepends `PRIOR CONTEXT SUMMARY:\n...\n\n` to system prompt when non-empty

**UI additions**:
- Context Summary Panel in right sidebar: INACTIVE / FIRING / ACTIVE badge, progress bar, summary text
- Timeline: `summarization_node` row in purple with message-before count
- State inspector: `system_summary` row

**Constants:**
```
SUMMARY_THRESHOLD = 8   # messages before summarization triggers
```

**Key concept:** Instead of discarding old messages (lossy), compress them into a dense text summary that travels forward in state. The LLM always has the full context of the conversation, just in compressed form.

---

## Upcoming Sessions

### Session 6 — Guardrails & Execution Bounding
**Goal:** Intercept unsafe inputs before they reach the LLM. Block PII leakage on outputs.

**What will be added:**
- `presidio-analyzer` + `presidio-anonymizer` for PII detection and masking
- `spacy en_core_web_lg` NLP model
- `ingress_node` — scans `raw_input` for PII and injection patterns; sets `pii_detected`, `injection_detected`, `is_safe`; writes `sanitized_input`
- `egress_node` — scans `final_response` for PII leakage after the agent responds
- `blocked_response_node` — safe refusal node, zero LLM tokens consumed
- `route_after_ingress` — `is_safe == False` → blocked, `True` → classify
- Graph entry point changes from `classify_node` to `ingress_node`

---

### Session 7 — Human-in-the-Loop (HITL)
**Goal:** Pause execution mid-graph for human review on high-risk actions.

**What will be added:**
- `interrupt()` call inside nodes that need human approval (e.g., fraud escalation, refund over threshold)
- `graph.update_state()` to inject the human's decision back into the paused thread
- Approval/rejection UI in the browser
- `escalation_required` field drives the interrupt decision

---

### Session 8 — Multi-Agent Orchestration
**Goal:** Delegate sub-tasks to specialist agents. The supervisor routes, specialists execute.

**What will be added:**
- Supervisor agent with a routing LLM call
- Specialist sub-graphs: `BillingAgent`, `TechnicalAgent`, `FraudAgent`
- `Command` objects for inter-agent communication
- Sub-graph results merged back into parent state
- Timeline shows nested agent execution

---

### Session 9 — Streaming & Real-Time UX
**Goal:** Token-level streaming from LLM to browser. Users see words appear as they are generated.

**What will be added:**
- `astream_events` replacing `astream` for token-level events
- `on_chat_model_stream` events forwarded via SSE
- UI: typing indicator, token-by-token text append
- Backpressure handling for slow clients

---

### Session 10 — Evaluation & Tracing
**Goal:** Measure agent quality. Every run produces a score. Regressions are caught automatically.

**What will be added:**
- LangSmith integration for full trace capture
- Evaluation dataset: 20 golden input/output pairs
- `evaluate()` runner with custom scorers (category accuracy, tool use correctness, response quality)
- CI step that fails if eval score drops below threshold
- Trace viewer linked from UI

---

### Session 11 — Production Hardening
**Goal:** The agent is no longer a prototype. Rate limiting, retries, cost tracking, health checks.

**What will be added:**
- Token usage tracking per invocation (stored in state)
- Cost estimation (Gemini pricing per 1K tokens)
- Retry logic with exponential backoff for transient LLM errors
- Rate limiter middleware in FastAPI
- `/metrics` endpoint (Prometheus-compatible)
- Structured JSON logging with correlation IDs

---

### Session 12 — Multi-Modal & Document Understanding
**Goal:** Accept image and PDF inputs. Extract structured data from uploaded support documents.

**What will be added:**
- File upload endpoint (`POST /api/upload`)
- Gemini vision for image analysis (screenshots, error photos)
- PDF text extraction with `pypdf`
- `document_content` field in `SupportState`
- UI: drag-and-drop file attachment
- Tool: `analyze_attachment(file_id)` — returns extracted text/data

---

## API Reference

### `POST /api/run`
Synchronous agent invocation.

**Request:**
```json
{
  "message": "My invoice is wrong",
  "thread_id": "user-123-session-abc"
}
```

**Response:**
```json
{
  "response": "I can help with that billing issue...",
  "category": "billing",
  "thread_id": "user-123-session-abc",
  "iterations": 2,
  "tool_calls_made": 1,
  "message_count": 6,
  "system_summary": "",
  "summary_active": false,
  "summary_threshold": 8
}
```

### `GET /api/stream?message=...&thread_id=...`
SSE stream of node-level execution events.

**Event format:**
```
data: {"node": "classify_node", "state": {...}, "elapsed_ms": 312}
```

**Summarization event:**
```
data: {"node": "summarization_node", "summary_fired": true, "summary": "...", "msgs_before": 10}
```

### `GET /health`
```json
{
  "status": "ok",
  "session": 5,
  "tools": 3,
  "max_iterations": 5,
  "summary_threshold": 8,
  "persistence": "sqlite",
  "db_path": "support.db"
}
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key for Gemini 2.5 Flash |

Copy `.env.example` (if present) or create `.env` manually:
```
GOOGLE_API_KEY=your-key-here
```

---

## Graph Architecture (Session 5)

```
                    ┌─────────────────┐
                    │   classify_node  │
                    └────────┬────────┘
                             │ route_after_classify
              ┌──────────────┼──────────────────────┐
              │              │                       │
              ▼              ▼                       ▼
      fraud_handler   summarization_node       general_handler
              │              │ (when msgs >= 8)      │
              │              ▼                       │
              │         agent_node ◄─────────────────┘
              │              │
              │         route_after_agent
              │         ┌────┴────┐
              │         ▼         ▼
              │    tool_node   respond_node
              │         │         │
              │         └────┬────┘
              │              │
              └──────► respond_node ──► END
```

---

## Running Tests

```bash
# CLI verification suite (Session 5 — 5 checks)
python support_agent.py

# UI verification (open browser, click "Session 5 — Verification Test")
open http://localhost:8000
```

---

## License

MIT
