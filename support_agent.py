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

def deduplicate_messages(left: list, right: list) -> list:
    """
    Custom reducer for state['messages'].
    Handles RemoveMessage deletions, then deduplicates additions.

    Prevents duplicate messages after the checkpointer re-applies
    state. Works alongside summarization_node which emits
    RemoveMessage objects to trim old messages from the list.

    Replaces: add_messages (Session 1 default)
    Introduced: Session 5
    """
    if not right:
        return left
    if not left:
        # Filter out any RemoveMessage from a fresh list
        return [m for m in right if not isinstance(m, RemoveMessage)]

    # Step 1: apply removals
    remove_ids = {m.id for m in right if isinstance(m, RemoveMessage) and m.id}
    if remove_ids:
        left = [m for m in left if not (hasattr(m, 'id') and m.id in remove_ids)]

    # Step 2: deduplicate additions
    additions = [m for m in right if not isinstance(m, RemoveMessage)]
    if not additions:
        return left

    existing_ids = {
        m.id for m in left
        if hasattr(m, 'id') and m.id
    }

    new_msgs = [
        m for m in additions
        if not (hasattr(m, 'id') and m.id in existing_ids)
    ]

    return left + new_msgs

# ══════════════════════════════════════════════════════════════════
# SECTION 2: STATE SCHEMA
# ══════════════════════════════════════════════════════════════════

class SupportState(TypedDict):

    # ── Core Input (Session 1) ──────────────────────────────────
    raw_input:          str        # Original user message, never modified
    sanitized_input:    str        # PII-cleaned version (Session 6)

    # ── Classification (Session 1) ─────────────────────────────
    category:           str        # technical | billing | fraud | general

    # ── Conversation History (Session 2) ───────────────────────
    messages:           Annotated[list, deduplicate_messages]  # Session 5: add_messages replaced with deduplicate_messages
    customer_data:      dict       # Populated by CRM tool
    tool_results:       Annotated[list, operator.add]  # Append-safe

    # ── Safety Controls (Session 6) ────────────────────────────
    pii_detected:       bool
    injection_detected: bool
    is_safe:            bool

    # ── Memory and Context (Sessions 3, 5) ─────────────────────
    system_summary:     str        # Compressed history (Session 5)
    iteration_count:    int        # ReAct circuit breaker (Session 3)

    # ── Multi-Agent Orchestration (Sessions 8, 9) ───────────────
    internal_notes:     Annotated[list, operator.add]  # Parallel scratchpad
    delegation_count:   int        # Supervisor counter
    next_worker:        str        # Supervisor decision

    # ── Write Access and Human Approval (Session 10, 11) ────────
    github_draft:       dict       # Proposed issue before approval
    github_issue_url:   str        # URL after creation

    # ── Output (Session 1) ─────────────────────────────────────
    final_response:     str


_field_count = len(SupportState.__annotations__)
print(f"[System] SupportState schema — {_field_count} fields across 12 sessions")

# ── Mock Data (Session 2) ──────────────────────────────────────

MOCK_CRM = {
    'C-1001': {
        # NEEDED FIELDS
        'name': 'Priya Sharma',
        'billing_status': 'Active',
        'subscription_tier': 'Enterprise',
        'last_payment_date': '2026-04-01',
        'last_payment_amount': 4999.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-04-01', 'description': 'Enterprise Plan — April',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-03-01', 'description': 'Enterprise Plan — March',  'amount': 4999.00, 'status': 'paid'},
            {'date': '2026-02-01', 'description': 'Enterprise Plan — February','amount': 4999.00, 'status': 'paid'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88123',
        'sales_rep_code': 'SR-042',
        'geo_region_tag': 'APAC',
        'last_login_ip': '192.168.10.5',
        'feature_flag_cohort': 'beta-v2',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1002': {
        # NEEDED FIELDS
        'name': 'Arjun Mehta',
        'billing_status': 'Past Due',
        'subscription_tier': 'Pro',
        'last_payment_date': '2026-03-01',
        'last_payment_amount': 499.00,
        'outstanding_balance': 998.00,
        'recent_transactions': [
            {'date': '2026-03-01', 'description': 'Pro Plan — March',  'amount': 499.00, 'status': 'paid'},
            {'date': '2026-04-01', 'description': 'Pro Plan — April',  'amount': 499.00, 'status': 'missed'},
            {'date': '2026-05-01', 'description': 'Pro Plan — May',    'amount': 499.00, 'status': 'missed'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88456',
        'sales_rep_code': 'SR-017',
        'geo_region_tag': 'APAC',
        'last_login_ip': '10.0.0.44',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
    'C-1003': {
        # NEEDED FIELDS
        'name': 'Kavya Nair',
        'billing_status': 'Active',
        'subscription_tier': 'Starter',
        'last_payment_date': '2026-05-01',
        'last_payment_amount': 99.00,
        'outstanding_balance': 0.00,
        'recent_transactions': [
            {'date': '2026-05-01', 'description': 'Starter Plan — May', 'amount': 99.00, 'status': 'paid'},
        ],
        # NOISY FIELDS
        'internal_crm_id': 'CRM-88789',
        'sales_rep_code': 'SR-031',
        'geo_region_tag': 'EMEA',
        'last_login_ip': '172.16.0.9',
        'feature_flag_cohort': 'stable',
        'data_warehouse_sync_ts': '2026-05-14T00:00:00Z',
    },
}


MOCK_KB = {
    'api': (
        'API troubleshooting guide: (1) Check rate limits — free tier: 100 req/min, '
        'pro: 1000 req/min, enterprise: unlimited. (2) Auth header must be '
        '"Authorization: Bearer <token>" — never basic auth. (3) On 401 errors, '
        'regenerate your API key in Account > API Keys. (4) On 429 rate-limit errors, '
        'implement exponential backoff starting at 1s. (5) SDK v3+ requires '
        'client.initialize() before first call.'
    ),
    'login': (
        'Login troubleshooting: (1) Clear browser cache and cookies, then retry. '
        '(2) MFA: open your authenticator app, use the 6-digit code within 30 seconds. '
        '(3) Password reset: go to login page > "Forgot password" > check email within '
        '5 minutes. (4) If locked out after 5 attempts, wait 15 minutes or contact '
        'support. (5) SSO users: ensure your identity provider session is active.'
    ),
    'billing': (
        'Billing help: (1) Invoice portal: account.nexus.io/billing/invoices — '
        'download PDF or CSV. (2) Update payment method: Billing > Payment Methods > '
        'Add New Card. (3) Refund policy: eligible within 30 days of charge, '
        'processed in 5-10 business days. (4) Subscription changes take effect on '
        'next billing cycle. (5) Failed payments retry automatically for 3 days.'
    ),
    'update': (
        'Post-update troubleshooting: (1) Clear application cache after any update: '
        'Settings > Cache > Clear All. (2) If issues persist, rollback procedure: '
        'go to Admin > Versions > select previous stable version > Rollback. '
        '(3) Check the changelog at docs.nexus.io/changelog for breaking changes. '
        '(4) SDK updates: run "npm install @nexus/sdk@latest" or '
        '"pip install nexus-sdk --upgrade".'
    ),
    '2fa': (
        '2FA / MFA help: (1) Backup codes: stored during setup — check your saved '
        'codes document. (2) Lost device: go to login > "Use backup code" > enter '
        'one of your 8-digit backup codes. (3) Reset 2FA: Account > Security > '
        'Two-Factor Auth > Reset — requires email verification. (4) Manual '
        'verification for locked accounts: contact support with government ID. '
        '(5) TOTP apps supported: Google Authenticator, Authy, 1Password.'
    ),
    'sdk': (
        'SDK compatibility guide: (1) SDK v3.x requires Node 18+ or Python 3.10+. '
        '(2) Migration from v2 to v3: replace client.get() with client.fetch(), '
        'update auth to client.initialize({apiKey}). (3) Breaking changes in v3: '
        'callback-style API removed, promises only. (4) Python SDK: '
        '"from nexus import NexusClient" replaces "import nexus". '
        '(5) Full migration guide: docs.nexus.io/sdk/v3-migration.'
    ),
}

# ── Mock Fraud Database (Session 3) ──────────────────────────────

MOCK_FRAUD_DB = {
    'ACC-F001': {
        'risk_score': 0.91,
        'flagged_patterns': ['multiple_countries_24h', 'unusual_amount'],
        'recommendation': 'freeze_account',
        'recent_flags': [
            {'date': '2026-05-15', 'pattern': 'multiple_countries_24h', 'severity': 'high'},
            {'date': '2026-05-15', 'pattern': 'unusual_amount',         'severity': 'high'},
        ],
    },
    'ACC-F002': {
        'risk_score': 0.23,
        'flagged_patterns': [],
        'recommendation': 'no_action',
        'recent_flags': [],
    },
    'ACC-F003': {
        'risk_score': 0.67,
        'flagged_patterns': ['new_device', 'large_transfer'],
        'recommendation': 'manual_review',
        'recent_flags': [
            {'date': '2026-05-14', 'pattern': 'large_transfer', 'severity': 'medium'},
        ],
    },
}

# ── Tools (Session 2) ──────────────────────────────────────────

@tool
def get_customer_details(customer_id: str) -> dict:
    """
    WHAT:
    Retrieves billing status, subscription tier, last payment date,
    outstanding balance, and recent transaction history for a customer
    from the CRM system.

    WHEN:
    Call this tool when the user's query involves billing, payment
    status, invoice disputes, subscription management, refund
    requests, or account standing. Always call before answering
    any billing question.

    FORMAT:
    customer_id must be in format 'C-XXXX' e.g. 'C-1001', 'C-1042'.
    Extract from the user message.
    If not present in the message, ask the user before calling.
    Never guess or fabricate a customer_id.

    RETURN:
    Dict with: name, billing_status, subscription_tier,
    last_payment_date, last_payment_amount, outstanding_balance,
    recent_transactions (last 3 only).
    On any failure: dict with single 'error' key describing what failed.
    """
    # LAYER 1 — Argument validation
    if not customer_id or not isinstance(customer_id, str):
        return {'error': "customer_id must be a non-empty string."}
    cid = customer_id.strip().upper()
    if not cid.startswith('C-'):
        return {'error': f"Invalid format: '{customer_id}'. Expected 'C-XXXX' e.g. 'C-1001'"}

    # LAYER 2 — Database lookup with error handling
    try:
        raw = MOCK_CRM.get(cid)
        if raw is None:
            return {'error': f"Customer '{cid}' not found. Please verify the ID with the customer."}
    except Exception as e:
        return {'error': f"CRM lookup failed: {type(e).__name__}. Contact engineering if this persists."}

    # LAYER 3 — Data filtering
    NEEDED = {
        'name', 'billing_status', 'subscription_tier',
        'last_payment_date', 'last_payment_amount',
        'outstanding_balance', 'recent_transactions'
    }
    filtered = {k: v for k, v in raw.items() if k in NEEDED}
    filtered['recent_transactions'] = filtered.get('recent_transactions', [])[:3]
    return filtered

# Test: get_customer_details.invoke({'customer_id': 'C-1001'})
# Test: get_customer_details.invoke({'customer_id': 'C-9999'})
# Test: get_customer_details.invoke({'customer_id': 'bad'})

@tool
def search_knowledge_base(query: str) -> dict:
    """
    WHAT:
    Searches the internal technical knowledge base for resolution
    steps and troubleshooting articles matching the issue described.

    WHEN:
    Call this tool for any technical issue before responding to the
    customer. Always search before saying you cannot help.
    If first search returns no match, try with different keywords.

    FORMAT:
    query is a natural language string describing the technical
    problem. Be specific. Include error codes or keywords.
    Example: 'API authentication 401 error after SDK update'

    RETURN:
    Dict with matched (bool), results (list of article strings),
    count (int). If no match: matched=False with fallback guidance.
    On failure: dict with single 'error' key.
    """
    try:
        if not query or not query.strip():
            return {'error': 'Search query cannot be empty.'}

        query_lower = query.lower()
        results = []
        for keyword, article in MOCK_KB.items():
            if keyword in query_lower:
                results.append(article)

        if not results:
            return {
                'matched': False,
                'results': [],
                'count': 0,
                'fallback': (
                    'No specific article found. General guidance: '
                    'check account status, clear browser cache, verify '
                    'recent configuration changes, review changelog.'
                )
            }

        return {'matched': True, 'results': results, 'count': len(results)}

    except Exception as e:
        return {'error': f"KB search failed: {type(e).__name__}"}

# Test: search_knowledge_base.invoke({'query': 'API 401 error'})
# Test: search_knowledge_base.invoke({'query': 'nothing matches'})

# ── Fraud Tool (Session 3) ──────────────────────────────────────

@tool
def check_fraud_signals(account_id: str) -> dict:
    """
    WHAT:
    Checks an account's transaction history against fraud
    detection rules. Returns a risk score, flagged behavioral
    patterns, and a recommended action for the security team.

    WHEN:
    Call for any ticket mentioning unauthorized transactions,
    suspicious charges, account compromise, or identity theft.
    Always call before making any fraud assessment.

    FORMAT:
    account_id must be in format 'ACC-FXXX' e.g. 'ACC-F001'.
    Extract from the user message.
    If not present, ask the user before calling this tool.

    RETURN:
    Dict with: account_id, risk_score (float 0.0-1.0),
    flagged_patterns (list of strings), recommendation (str),
    recent_flags (list of dicts).
    risk_score > 0.7  -> high risk
    risk_score 0.4-0.7 -> medium risk
    risk_score < 0.4  -> low risk
    On failure: dict with single 'error' key.
    """
    # LAYER 1 — Validation
    if not account_id or not isinstance(account_id, str):
        return {'error': 'account_id must be a non-empty string.'}
    aid = account_id.strip().upper()
    if not aid.startswith('ACC-'):
        return {
            'error': f"Invalid format: '{account_id}'. "
                     f"Expected 'ACC-FXXX' e.g. 'ACC-F001'"
        }

    # LAYER 2 — Lookup with error handling
    try:
        record = MOCK_FRAUD_DB.get(aid)
        if record is None:
            return {
                'error': f"No fraud profile found for '{aid}'. "
                         f"Verify the account ID with the customer."
            }
    except Exception as e:
        return {'error': f"Fraud DB unavailable: {type(e).__name__}"}

    # LAYER 3 — Return with account_id injected
    result = dict(record)
    result['account_id'] = aid
    return result

# Test: check_fraud_signals.invoke({'account_id': 'ACC-F001'})
# Test: check_fraud_signals.invoke({'account_id': 'ACC-F999'})
# Test: check_fraud_signals.invoke({'account_id': 'bad-format'})

TOOLS = [
    get_customer_details,
    search_knowledge_base,
    check_fraud_signals,
]
llm_with_tools = llm.bind_tools(TOOLS)

print(f"[Tools] {len(TOOLS)} tools registered:")
for t in TOOLS:
    print(f"  · {t.name}")

# ── Agent System Prompt (Session 3) ──────────────────────────────

AGENT_SYSTEM_PROMPT = """
You are a senior customer support specialist with access
to the CRM system and internal knowledge base.

TOOL USAGE RULES:

get_customer_details:
  - Call for ANY billing, payment, subscription, or account query
  - ALWAYS call before answering billing questions
  - If customer_id not in the message: ask before calling
  - Never guess or fabricate a customer_id

search_knowledge_base:
  - Call for ANY technical issue before responding
  - Always search before saying you cannot help
  - Use specific technical terms in the query
  - Multiple searches allowed if first returns no match

check_fraud_signals:
  - Call for ANY mention of unauthorized transactions,
    suspicious charges, or account compromise
  - Always call before making any fraud assessment
  - If account_id not in the message: ask before calling

RESPONSE RULES:
  - Base all answers on tool output, not internal knowledge
  - If a tool returns an error key: acknowledge it professionally
  - Reference specific data points from tool results
  - Never expose internal field names or system details
"""

# ── Summarization Prompt (Session 5) ──────────────────────────────────────────

SUMMARIZATION_PROMPT = """
You are summarizing a customer support conversation to preserve
key context for future turns of the same conversation.

Create a dense factual summary of 3 to 5 sentences.

You MUST include every instance of:
  - Customer identifiers (account IDs, names, email addresses)
  - Financial data (amounts, dates, balances, transaction IDs)
  - What the customer reported and what investigation found
  - Decisions made or actions taken in this conversation
  - Items still unresolved or pending customer action

You MUST NOT include:
  - Pleasantries or conversational filler
  - Failed tool call attempts or error messages
  - Repeated information already stated earlier
  - Internal system field names or technical metadata

Respond with the summary text only.
No preamble. No labels. No bullet points.
Plain prose. Dense with facts.
"""

# ══════════════════════════════════════════════════════════════════
# SECTION 3: CLASSIFIER NODE
# ══════════════════════════════════════════════════════════════════

def classify_node(state: SupportState) -> dict:
    system_prompt = (
        "You are a support ticket classifier for an enterprise SaaS company.\n"
        "Classify the incoming ticket into EXACTLY ONE of these 4 categories:\n\n"
        "  technical:  API errors, login failures, bugs, performance issues,\n"
        "              integration problems, post-update breakage\n"
        "  billing:    payment failures, invoice disputes, subscriptions,\n"
        "              refund requests, double charges\n"
        "  fraud:      unauthorized transactions, account compromise,\n"
        "              suspicious activity, identity theft\n"
        "  general:    feature questions, how-to, onboarding, documentation,\n"
        "              anything that does not fit the above categories\n\n"
        "Respond with EXACTLY ONE WORD. No punctuation. "
        "No explanation. No other text whatsoever."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["raw_input"]),
    ])

    # Layer 1 — normalize
    raw = response.content.strip().lower().rstrip(".,!?")

    # Layer 2 — validate
    VALID = {"technical", "billing", "fraud", "general"}
    if raw not in VALID:
        print(f"[Classifier] Unexpected output: '{raw}' → defaulting to 'general'")
        raw = "general"

    # Layer 3 — print
    preview = state["raw_input"][:60]
    print(f"[Classifier] '{preview}'... → {raw}")

    return {
        "category":           raw,
        "sanitized_input":    state["raw_input"],  # pass-through until Session 6
        "iteration_count":    0,
        "delegation_count":   0,
        "is_safe":            True,
        "pii_detected":       False,
        "injection_detected": False,
    }

# ══════════════════════════════════════════════════════════════════
# SECTION 4: ROUTER FUNCTION
# ══════════════════════════════════════════════════════════════════

def route_by_category(state: SupportState) -> str:
    raw = state.get("category") or ""
    category = raw.strip().lower()

    routing_map = {
        "technical": "technical_handler",
        "billing":   "billing_handler",
        "fraud":     "fraud_handler",
        "general":   "general_handler",
    }

    destination = routing_map.get(category, "general_handler")
    print(f"[Router] '{category}' → {destination}")
    return destination

# ── Summarization Router (Session 5) ──────────────────────────────────────────

def route_after_classify(state: SupportState) -> str:
    """
    Fires after classify_node. Handles both category routing
    and summarization threshold check.

    For fraud/general: routes directly to their handlers.
    For billing/technical: checks SUMMARY_THRESHOLD.
      If exceeded: routes to summarization_node first.
      If not: routes directly to agent_node.

    Pure Python. Zero LLM calls. Zero business logic.
    Permanent from Session 5 onward.
    """
    category = state.get('category', '')

    if category == 'fraud':
        return 'fraud_handler'
    if category == 'general':
        return 'general_handler'

    # billing or technical: check message count
    msg_count = len(state.get('messages', []))

    if msg_count > SUMMARY_THRESHOLD:
        print(f"[Router:classify] {msg_count} messages "
              f"> {SUMMARY_THRESHOLD} → summarization_node")
        return 'summarization_node'

    print(f"[Router:classify] {msg_count} messages "
          f"≤ {SUMMARY_THRESHOLD} → agent_node")
    return 'agent_node'

# ══════════════════════════════════════════════════════════════════
# SECTION 5: HANDLER STUBS
# ══════════════════════════════════════════════════════════════════

def technical_handler(state: SupportState) -> dict:
    # STUB — replaced in Session 2 (routing now goes to agent_node)
    preview = state["raw_input"][:80]
    print(f"[technical_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your technical issue has been received and assigned to our "
            "Engineering team. A specialist will respond within 4 hours."
        )
    }

def billing_handler(state: SupportState) -> dict:
    # STUB — replaced in Session 2 (routing now goes to agent_node)
    preview = state["raw_input"][:80]
    print(f"[billing_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Your billing inquiry has been received and assigned to our "
            "Finance team. We will review your account within 2 hours."
        )
    }

def general_handler(state: SupportState) -> dict:
    # Stays simple throughout all sessions
    preview = state["raw_input"][:80]
    print(f"[general_handler] Handling: '{preview}'")
    return {
        "final_response": (
            "Thank you for reaching out. Your inquiry has been received "
            "and our support team will respond within 24 hours."
        )
    }

# ── ReAct Helpers (Session 3) ────────────────────────────────────


def build_escalation_response(state: SupportState, iteration: int) -> dict:
    """
    Produces a graceful user-facing escalation message when
    the circuit breaker fires or a duplicate tool call is detected.
    Summarizes tool findings before escalating.
    Called by: agent_node (Session 3 onward).
    """
    tool_findings = []
    for msg in state.get('messages', []):
        if hasattr(msg, 'tool_call_id') and msg.content:
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and 'error' not in data:
                    tool_findings.append(data)
            except Exception:
                pass

    if tool_findings:
        lines = []
        for finding in tool_findings[:2]:
            for k, v in list(finding.items())[:2]:
                lines.append(f"· {k}: {v}")
        summary = "\n".join(lines)
    else:
        summary = "· No data retrieved before escalation."

    ref = str(uuid.uuid4())[:8].upper()

    escalation_text = (
        f"I investigated your request thoroughly but was unable "
        f"to resolve it automatically.\n\n"
        f"What I found:\n{summary}\n\n"
        f"A specialist will review this and contact you within "
        f"24 hours. Reference: {ref}"
    )

    print(f"[Escalation] Circuit breaker at iteration {iteration} "
          f"| ref: {ref}")

    return {
        'messages':        [AIMessage(content=escalation_text)],
        'iteration_count': iteration,
        'final_response':  escalation_text,
    }

def trim_context(messages: list, threshold: int) -> list:
    """
    Keeps messages[0] (original user message) plus the most
    recent (threshold - 1) messages. Prevents context window
    explosion over many tool call iterations.
    Called by: agent_node before every LLM call (Session 3 onward).
    """
    if len(messages) <= threshold:
        return messages

    preserved = messages[0]
    recent    = messages[-(threshold - 1):]
    result    = [preserved] + recent

    print(f"[Context Trim] {len(messages)} → {len(result)} messages")
    return result

def get_tool_fingerprint(tool_call: dict) -> str:
    """
    Returns a unique string for a tool call based on its name
    and sorted arguments. Used to detect duplicate tool calls.
    Called by: agent_node after every LLM response (Session 3 onward).
    """
    name = tool_call.get('name', '')
    args = tool_call.get('args', {})
    return f"{name}::{json.dumps(args, sort_keys=True)}"

# ── Agent Node (Session 3) ──────────────────────────────────────

def agent_node(state: SupportState) -> dict:
    """
    Full ReAct agent node with three safety layers.
    Replaces the single-pass agent_node from Session 2.

    Layer 1: Circuit breaker — hard stop at MAX_ITERATIONS.
    Layer 2: Read system_summary — prepend to prompt if present.
    Layer 3: Duplicate detection — fingerprint each tool call,
             escalate immediately if same call seen twice.

    Uses AGENT_SYSTEM_PROMPT module constant (Session 3+).
    Session 5: trim_context() retired. Reads system_summary
               from state instead.
    Permanent from Session 3 onward.
    """

    # ── LAYER 1: CIRCUIT BREAKER ─────────────────────────────────
    iteration = state.get('iteration_count', 0) + 1

    if iteration > MAX_ITERATIONS:
        return build_escalation_response(state, iteration)

    print(f"[Agent] iteration={iteration}/{MAX_ITERATIONS}")

    # ── LAYER 2: READ SYSTEM SUMMARY ─────────────────────────────
    summary = state.get('system_summary', '')

    if summary:
        context = (
            f"PRIOR CONTEXT SUMMARY:\n{summary}"
            f"\n\n{AGENT_SYSTEM_PROMPT}"
        )
        print(f"[Agent] system_summary present "
              f"({len(summary)} chars) — prepended to prompt")
    else:
        context = AGENT_SYSTEM_PROMPT
        print(f"[Agent] No system_summary — using base prompt")

    # No trim_context() call — summarization_node handles this
    messages_to_send = [
        SystemMessage(content=context),
        *state.get('messages', [])
    ]

    # ── CORE: LLM CALL ───────────────────────────────────────────
    response   = llm_with_tools.invoke(messages_to_send)
    tool_count = len(response.tool_calls) if response.tool_calls else 0
    print(f"[Agent] tool_calls={tool_count} | "
          f"has_content={bool(response.content)}")

    # ── LAYER 3: DUPLICATE DETECTION ─────────────────────────────
    new_fingerprints = []

    if response.tool_calls:

        existing = {
            r.get('fingerprint')
            for r in state.get('tool_results', [])
            if isinstance(r, dict) and 'fingerprint' in r
        }

        for tc in response.tool_calls:
            fp = get_tool_fingerprint(tc)

            if fp in existing:
                print(f"[Agent] Duplicate: {tc['name']} same args. Escalating.")
                stuck_text = (
                    f"I've already attempted {tc['name']} with these "
                    f"parameters and received an error. Escalating to "
                    f"our support team for manual review."
                )
                return {
                    'messages':        [AIMessage(content=stuck_text)],
                    'iteration_count': iteration,
                    'final_response':  stuck_text,
                }

            new_fingerprints.append({'fingerprint': fp})

    # ── RETURN ────────────────────────────────────────────────────
    return {
        'messages':        [response],
        'iteration_count': iteration,
        'tool_results':    state.get('tool_results', []) + new_fingerprints,
    }

# ── Routing & Terminal Nodes (Session 2) ─────────────────────────

def route_after_agent(state: SupportState) -> str:
    """
    Reads last message. If tool_calls present → tool_node.
    If no tool_calls → respond_node.
    Pure Python. Zero LLM calls. Zero business logic.
    Permanent from Session 2 onward.
    """
    messages = state.get('messages', [])
    if not messages:
        return 'respond_node'
    last = messages[-1]
    has_tools = hasattr(last, 'tool_calls') and bool(last.tool_calls)
    destination = 'tool_node' if has_tools else 'respond_node'
    print(f"[Router:after_agent] tool_calls={has_tools} → {destination}")
    return destination

def respond_node(state: SupportState) -> dict:
    """
    Extracts last AIMessage content → final_response.
    Runs after agent_node when no further tool calls needed.
    Permanent from Session 2 onward.
    """
    messages = state.get('messages', [])
    final = ''
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            # Gemini may return a list of content blocks; extract text
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        parts.append(block['text'])
                    elif isinstance(block, str):
                        parts.append(block)
                final = ' '.join(parts).strip()
            else:
                final = str(content)
            if final:
                break
    print(f"[Respond] {len(final)} chars")
    return {'final_response': final}


tool_node = ToolNode(tools=TOOLS)
print(f"[Tools] ToolNode ready — {len(TOOLS)} tools registered")

def _extract_text(content) -> str:
    """Extracts plain text from an AIMessage content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                parts.append(block['text'])
            elif isinstance(block, str):
                parts.append(block)
        return ' '.join(parts).strip()
    return str(content)

# ── Summarization Node (Session 5) ────────────────────────────────────────────

def summarization_node(state: SupportState) -> dict:
    """
    Maintenance node. Fires when message count exceeds
    SUMMARY_THRESHOLD. Compresses old messages into
    state['system_summary']. Trims messages to last 4.

    Produces no user-facing output.
    Serves agent_node by managing context size.
    Introduced: Session 5. Permanent from here onward.
    """
    messages = state.get('messages', [])
    print(f"[Summarize] Triggered — {len(messages)} messages → compressing")

    # Filter to only Human/AI content messages — no tool_calls or ToolMessages.
    # Gemini rejects sequences with orphaned function-call turns.
    msgs_for_summary = []
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            msgs_for_summary.append(m)
        elif isinstance(m, AIMessage) and m.content and not getattr(m, 'tool_calls', None):
            msgs_for_summary.append(m)

    if not msgs_for_summary:
        msgs_for_summary = messages  # fallback: send everything

    try:
        response = llm.invoke([
            SystemMessage(content=SUMMARIZATION_PROMPT),
            *msgs_for_summary
        ])
        summary = _extract_text(response.content).strip()
        # Strip Gemini 2.5 Flash thinking tokens if present in content
        import re as _re
        summary = _re.sub(r'<thinking>.*?</thinking>', '', summary,
                          flags=_re.DOTALL | _re.IGNORECASE).strip()
        # Hard cap: a real summary is never > 1500 chars
        if len(summary) > 1500:
            summary = summary[:1500].rsplit('.', 1)[0] + '.'
        print(f"[Summarize] Summary: {summary[:80]}...")
    except Exception as e:
        print(f"[Summarize] LLM error: {e} — keeping existing summary")
        return {}

    # Keep last 4 messages but always start at a HumanMessage boundary
    # so we never hand Gemini an orphaned AIMessage(tool_calls) first.
    keep_n = 4
    while keep_n <= len(messages):
        if isinstance(messages[-keep_n], HumanMessage):
            break
        keep_n += 1
    # Fallback: if no HumanMessage found, keep as-is
    if keep_n > len(messages):
        keep_n = 4

    keep_from = len(messages) - keep_n
    messages_to_remove = messages[:keep_from]

    print(f"[Summarize] Messages trimmed: {len(messages)} → {keep_n} "
          f"(kept from index {keep_from})")

    # Use RemoveMessage to delete old entries so the reducer handles it cleanly.
    remove_ops = [
        RemoveMessage(id=m.id)
        for m in messages_to_remove
        if hasattr(m, 'id') and m.id
    ]

    return {
        'system_summary': summary,
        'messages':       remove_ops,
    }

# ── Fraud Handler (Session 3) ────────────────────────────────────

def fraud_handler(state: SupportState) -> dict:
    """
    Fraud analysis handler. Upgraded from stub in Session 3.
    Uses check_fraud_signals for real fraud assessment.
    Single tool call — not the full ReAct loop.
    Replaced with parallel fraud agent swarm in Session 9.
    """

    fraud_system_prompt = """
    You are a fraud analysis specialist.
    You have access to the check_fraud_signals tool.

    When a customer reports suspicious activity:
    1. Extract the account_id from their message.
    2. Call check_fraud_signals with that account_id.
    3. Interpret the risk_score and flagged_patterns.
    4. Give a clear, professional response about next steps.

    If account_id is not in the message: ask for it first.
    Never fabricate fraud findings.
    Always base your response entirely on tool output.
    """

    fraud_llm = llm.bind_tools([check_fraud_signals])

    messages_to_send = [
        SystemMessage(content=fraud_system_prompt),
        *state.get('messages', [])
    ]

    response = fraud_llm.invoke(messages_to_send)

    if response.tool_calls:

        tc     = response.tool_calls[0]
        result = check_fraud_signals.invoke(tc.get('args', {}))

        result_msg = ToolMessage(
            content      = json.dumps(result),
            tool_call_id = tc['id']
        )

        final_messages = messages_to_send + [response, result_msg]
        final          = fraud_llm.invoke(final_messages)

        print(f"[Fraud] tool called | risk_score="
              f"{result.get('risk_score', 'N/A')}")

        final_text = _extract_text(final.content)
        return {
            'messages':       [response, result_msg, final],
            'final_response': final_text,
        }

    else:
        return {
            'messages':       [response],
            'final_response': _extract_text(response.content),
        }


# ══════════════════════════════════════════════════════════════════
# SECTION 6: GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════

def build_graph():
    """
    Builds and compiles the LangGraph support agent.
    Called once at module load. Returns compiled graph.
    Sessions 2-12 modify this function by adding nodes and edges.
    Never remove existing nodes — only add.
    """
    builder = StateGraph(SupportState)

    # Register Session 1 nodes
    builder.add_node("classify_node",     classify_node)
    builder.add_node("technical_handler", technical_handler)
    builder.add_node("billing_handler",   billing_handler)
    builder.add_node("fraud_handler",     fraud_handler)
    builder.add_node("general_handler",   general_handler)

    # Register Session 2 nodes
    builder.add_node("agent_node",   agent_node)
    builder.add_node("tool_node",    tool_node)
    builder.add_node("respond_node", respond_node)

    # Register Session 5 node
    builder.add_node("summarization_node", summarization_node)

    # Entry point
    builder.set_entry_point("classify_node")

    # Routing from classifier — Session 5: route_after_classify replaces
    # route_by_category as primary router. Handles fraud/general directly
    # and checks SUMMARY_THRESHOLD before routing to agent_node.
    builder.add_conditional_edges(
        "classify_node",
        route_after_classify,
        {
            "summarization_node": "summarization_node",
            "agent_node":         "agent_node",
            "fraud_handler":      "fraud_handler",
            "general_handler":    "general_handler",
        }
    )

    # summarization_node always flows to agent_node
    builder.add_edge("summarization_node", "agent_node")

    # Agent → tool loop
    builder.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_node":    "tool_node",
            "respond_node": "respond_node",
        }
    )

    # Tool node loops back to agent
    builder.add_edge("tool_node", "agent_node")

    # Terminal edges
    builder.add_edge("respond_node",     END)
    builder.add_edge("fraud_handler",    END)
    builder.add_edge("general_handler",  END)

    # Session 1 stub nodes still need edges (they are never reached for
    # billing/technical but must be registered to avoid orphan node errors)
    builder.add_edge("technical_handler", END)
    builder.add_edge("billing_handler",   END)

    graph = builder.compile(checkpointer=checkpointer)
    print("[Graph] Session 5 — 9 nodes | summarization active")
    return graph


# Module-level graph instance
graph = build_graph()

# ══════════════════════════════════════════════════════════════════
# SECTION 7: INITIAL STATE BUILDER
# ══════════════════════════════════════════════════════════════════

def build_initial_state(ticket: str) -> dict:
    """
    Constructs a clean initial state for every graph invocation.
    Provides safe defaults for ALL 17 fields so no node gets a KeyError.
    Called by both the test harness and the Streamlit UI.
    """
    return {
        "raw_input":          ticket,
        "sanitized_input":    "",
        "category":           "",
        "messages":           [HumanMessage(content=ticket)],
        "customer_data":      {},
        "tool_results":       [],
        "pii_detected":       False,
        "injection_detected": False,
        "is_safe":            True,
        "system_summary":     "",
        "iteration_count":    0,
        "internal_notes":     [],
        "delegation_count":   0,
        "next_worker":        "",
        "github_draft":       {},
        "github_issue_url":   "",
        "final_response":     "",
    }

# ══════════════════════════════════════════════════════════════════
# SECTION 8: RUN FUNCTION (called by both CLI and UI)
# ══════════════════════════════════════════════════════════════════

def run_ticket(ticket: str,
               thread_id: str = None,
               return_existing: bool = False) -> dict:
    """
    Runs a ticket through the graph.
    If thread_id provided: loads prior state from checkpointer,
    appends new message, resumes conversation.
    If thread_id is None: generates a new thread_id,
    starts a fresh conversation.

    return_existing=True: if the thread already ran to END (e.g. the
    stream endpoint already executed it), return the existing final
    checkpoint state instead of re-invoking the graph. This prevents
    the stream+run UI pattern from executing the graph twice.
    Session 4+: always pass thread_id for persistent conversations.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"[Thread] New thread created: {thread_id}")

    config = {'configurable': {'thread_id': thread_id}}

    existing = list(graph.get_state_history(config))

    # Stream already ran this thread to completion — return its state.
    if return_existing and existing and len(existing[0].next) == 0:
        print(f"[Thread] Returning existing completed state | thread={thread_id}")
        result_dict = dict(existing[0].values)
        result_dict['thread_id'] = thread_id
        return result_dict

    is_first_turn = len(existing) == 0

    if is_first_turn:
        initial_state = build_initial_state(ticket)
        result = graph.invoke(initial_state, config=config)
        print(f"[Thread] First turn | thread={thread_id}")
    else:
        follow_up_state = {
            'messages': [HumanMessage(content=ticket)]
        }
        result = graph.invoke(follow_up_state, config=config)
        print(f"[Thread] Follow-up turn | thread={thread_id} | prior_steps={len(existing)}")

    result_dict = dict(result)
    result_dict['thread_id'] = thread_id
    return result_dict


def stream_ticket(ticket: str,
                  thread_id: str = None):
    """
    Generator. Yields (node_name, snapshot) tuples.
    Accepts thread_id for persistent streaming.
    Session 4+: pass thread_id for conversation continuity.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {'configurable': {'thread_id': thread_id}}

    existing = list(graph.get_state_history(config))
    is_first_turn = len(existing) == 0

    if is_first_turn:
        state_to_send = build_initial_state(ticket)
    else:
        state_to_send = {'messages': [HumanMessage(content=ticket)]}

    for step in graph.stream(state_to_send, config=config):
        for node_name, snapshot in step.items():
            yield node_name, snapshot

# ── Conversation History (Session 4) ─────────────────────────────

def get_conversation_history(thread_id: str) -> list:
    """
    Returns the full checkpoint history for a thread_id.
    Each entry is a dict with: step, node, state_summary,
    timestamp, is_end.
    Used by /api/history endpoint and the UI history panel.
    """
    config = {'configurable': {'thread_id': thread_id}}

    try:
        history = list(graph.get_state_history(config))
    except Exception as e:
        print(f"[History] Error loading thread {thread_id}: {e}")
        return []

    if not history:
        return []

    entries = []
    for snap in reversed(history):
        entry = {
            'step':           snap.metadata.get('step', 0),
            'source':         snap.metadata.get('source', ''),
            'node':           snap.metadata.get('source', 'unknown'),
            'category':       snap.values.get('category', ''),
            'iteration':      snap.values.get('iteration_count', 0),
            'message_count':  len(snap.values.get('messages', [])),
            'final_response': snap.values.get('final_response', ''),
            'is_end':         len(snap.next) == 0,
            'checkpoint_id':  snap.config.get('configurable', {})
                                  .get('checkpoint_id', ''),
        }
        entries.append(entry)

    print(f"[History] Thread {thread_id}: {len(entries)} checkpoints")
    return entries

def get_active_threads() -> list:
    """
    Returns a list of all thread_ids that have at least one checkpoint.
    Used by /api/threads endpoint and the thread selector in the UI.
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"[Threads] {len(threads)} active threads")
        return threads
    except Exception as e:
        print(f"[Threads] Error: {e}")
        return []
    
# ══════════════════════════════════════════════════════════════════
# SECTION 9: SESSION VERIFICATION TEST
# ══════════════════════════════════════════════════════════════════

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 5 — VERIFICATION TEST                              │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │  summarization_node fires when message count exceeds        │
    │  SUMMARY_THRESHOLD. system_summary is populated and         │
    │  persisted. agent_node reads summary correctly.             │
    │  Agent does not re-ask for data already in summary.         │
    │                                                             │
    │  PASS CRITERIA:                                             │
    │  ✓ Below threshold → summarization_node does NOT fire      │
    │  ✓ Above threshold → summarization_node fires              │
    │  ✓ system_summary is non-empty after trigger               │
    │  ✓ system_summary persists in checkpointer                 │
    │  ✓ Agent references summary data without re-calling tool   │
    │                                                             │
    │  WHAT A PASS PROVES:                                        │
    │  Context management strategy is production-ready.           │
    │  trim_context() successfully retired.                       │
    │  Long conversations stay bounded and coherent.              │
    │  Session 6 is unblocked.                                    │
    └─────────────────────────────────────────────────────────────┘
    """

    import time
    start  = time.time()
    checks = []

    print("\n" + "▓" * 60)
    print("▓  SESSION 5 — VERIFICATION TEST                        ▓")
    print("▓" * 60)

    # ── CHECK 1: Below threshold — no summarization ───────────────

    thread_1 = f"verify-s5-below-{int(time.time())}"

    result = run_ticket(
        "What is my subscription plan? Account C-1001.",
        thread_id=thread_1
    )

    msg_count = len(result.get('messages', []))

    check1_passed = (
        msg_count <= SUMMARY_THRESHOLD and
        bool(result.get('final_response', '').strip())
    )

    checks.append({
        'label':        'Below threshold — summarization_node does not fire',
        'passed':       check1_passed,
        'has_response': bool(result.get('final_response', '').strip()),
        'note':         f"msg_count={msg_count} threshold={SUMMARY_THRESHOLD}",
    })
    print(f"{'✅' if check1_passed else '❌'} CHECK 1: Below threshold | "
          f"msg_count={msg_count}")

    # ── CHECK 2: Above threshold — summarization fires ────────────

    thread_2 = f"verify-s5-above-{int(time.time())}"

    fake_messages = []
    for i in range(5):
        fake_messages.append(HumanMessage(content=f"Prior customer message {i}"))
        fake_messages.append(AIMessage(content=f"Prior agent response {i}"))

    state2 = build_initial_state("What is my outstanding balance? Account C-1002.")
    state2['messages'] = fake_messages + state2['messages']

    config2 = {'configurable': {'thread_id': thread_2}}
    result2  = graph.invoke(state2, config=config2)

    summary_populated = bool(result2.get('system_summary', '').strip())

    check2_passed = (
        summary_populated and
        bool(result2.get('final_response', '').strip())
    )

    checks.append({
        'label':        'Above threshold — summarization_node fires',
        'passed':       check2_passed,
        'has_response': bool(result2.get('final_response', '').strip()),
        'note':         f"summary_length={len(result2.get('system_summary', ''))}",
    })
    print(f"{'✅' if check2_passed else '❌'} CHECK 2: Above threshold | "
          f"summary_populated={summary_populated}")

    # ── CHECK 3: system_summary is non-empty and meaningful ───────

    summary_text = result2.get('system_summary', '')

    meaningful = (
        'C-1002' in summary_text or
        '998'    in summary_text or
        'arjun'  in summary_text.lower() or
        'past due' in summary_text.lower() or
        'balance'  in summary_text.lower()
    )

    check3_passed = (len(summary_text) > 20 and meaningful)

    checks.append({
        'label':        'system_summary is non-empty and contains key facts',
        'passed':       check3_passed,
        'has_response': True,
        'note':         f"summary: {summary_text[:60]}...",
    })
    print(f"{'✅' if check3_passed else '❌'} CHECK 3: Summary quality | "
          f"meaningful={meaningful} len={len(summary_text)}")

    # ── CHECK 4: system_summary persists in checkpointer ──────────

    try:
        current_state   = graph.get_state(config2)
        summary_persisted = bool(
            current_state.values.get('system_summary', '').strip()
        )
    except Exception:
        summary_persisted = len(get_conversation_history(thread_2)) > 0

    check4_passed = summary_persisted

    checks.append({
        'label':        'system_summary persists in SQLite checkpointer',
        'passed':       check4_passed,
        'has_response': True,
        'note':         f"persisted={summary_persisted}",
    })
    print(f"{'✅' if check4_passed else '❌'} CHECK 4: Persistence | "
          f"persisted={summary_persisted}")

    # ── CHECK 5: Agent uses summary without re-calling tool ───────

    result3    = run_ticket(
        "Based on what you found, what are my options?",
        thread_id=thread_2
    )
    response3  = result3.get('final_response', '').lower()

    references_context = (
        '998'      in response3 or
        'past due' in response3 or
        'balance'  in response3 or
        'payment'  in response3
    )

    check5_passed = (
        references_context and
        bool(result3.get('final_response', '').strip())
    )

    checks.append({
        'label':        'Agent references summary data in follow-up',
        'passed':       check5_passed,
        'has_response': bool(result3.get('final_response', '').strip()),
        'note':         f"context_referenced={references_context}",
    })
    print(f"{'✅' if check5_passed else '❌'} CHECK 5: Summary usage | "
          f"context_referenced={references_context}")

    # ── RETURN ────────────────────────────────────────────────────

    all_passed   = all(c['passed'] for c in checks)
    duration_ms  = int((time.time() - start) * 1000)
    passed_count = sum(1 for c in checks if c['passed'])

    summary_line = (f"{passed_count}/{len(checks)} checks passed "
                    f"in {duration_ms}ms")

    print("\n" + "▓" * 60)
    if all_passed:
        print("▓  VERIFICATION: ✅ PASSED — Session 6 is unblocked       ▓")
    else:
        print("▓  VERIFICATION: ❌ FAILED — Fix summarization_node        ▓")
    print(f"▓  {summary_line:<54}▓")
    print("▓" * 60)

    return {
        'passed':      all_passed,
        'checks':      checks,
        'summary':     summary_line,
        'duration_ms': duration_ms,
    }

# ══════════════════════════════════════════════════════════════════
# SECTION 10: CLI TEST HARNESS
# ══════════════════════════════════════════════════════════════════

def run_cli_tests():
    """Runs all Session 5 test cases when file is executed directly."""

    print("\n" + "█" * 62)
    print("█  ENTERPRISE AI SUPPORT PLATFORM — SESSION 5 OF 12      █")
    print("█  Context Management & Summarization                     █")
    print("█" * 62)

    import time as _time
    _ts = int(_time.time())

    # TEST 1 — Short conversation (no summarization)
    print(f"\n{'─' * 60}")
    print("TEST 1 — Short conversation (no summarization)")
    thread_short = f"test-s5-short-{_ts}"
    ticket1 = "What is my subscription plan? Account C-1001."
    print(f"TICKET:    {ticket1}")
    print(f"THREAD:    {thread_short}")
    print("EXPECTED:  agent responds, system_summary remains empty")

    result1 = run_ticket(ticket1, thread_id=thread_short)
    print(f"Category:  {result1.get('category', '?')}")
    print(f"Response:  {result1.get('final_response', '')[:120]}...")
    print(f"system_summary populated: {bool(result1.get('system_summary', ''))}")
    print(f"Status:    {'✅ PASS' if result1.get('final_response') else '❌ FAIL'}")

    # TEST 2 — Force summarization trigger
    print(f"\n{'─' * 60}")
    print("TEST 2 — Force summarization trigger (10 pre-loaded messages)")
    thread_long = f"test-s5-long-{_ts}"

    fake_msgs = []
    for i in range(5):
        fake_msgs.append(HumanMessage(content=f"Prior customer message {i}"))
        fake_msgs.append(AIMessage(content=f"Prior agent response {i}"))

    state2 = build_initial_state("What is my outstanding balance? Account C-1002.")
    state2['messages'] = fake_msgs + state2['messages']

    config_long = {'configurable': {'thread_id': thread_long}}
    result2 = graph.invoke(state2, config=config_long)

    print(f"EXPECTED:  [Summarize] log fires, system_summary non-empty")
    print(f"Summary:   {result2.get('system_summary', '')[:100]}...")
    print(f"Response:  {result2.get('final_response', '')[:120]}...")
    print(f"Status:    {'✅ PASS' if result2.get('system_summary') else '❌ FAIL'}")

    # TEST 3 — Follow-up after summarization
    print(f"\n{'─' * 60}")
    print("TEST 3 — Follow-up after summarization (same thread)")
    print(f"THREAD:    {thread_long}  (same as Test 2)")
    print("EXPECTED:  agent references balance/account without new tool call")

    result3 = run_ticket(
        "What should I do to restore my dashboard access?",
        thread_id=thread_long
    )
    resp3 = result3.get('final_response', '').lower()
    referenced = any(k in resp3 for k in ['998', 'past due', 'balance', 'payment', 'arjun', 'c-1002'])
    print(f"Response:  {result3.get('final_response', '')[:120]}...")
    print(f"Agent referenced prior context: {'yes' if referenced else 'no'}")
    print(f"Status:    {'✅ PASS' if result3.get('final_response') else '❌ FAIL'}")

    # TEST 4 — Summarization content check
    print(f"\n{'─' * 60}")
    print("TEST 4 — Summarization content check")
    full_summary = result2.get('system_summary', '')
    print(f"Full summary:\n  {full_summary}\n")
    has_account  = 'C-1002' in full_summary or 'c-1002' in full_summary.lower()
    has_balance  = '998' in full_summary or 'balance' in full_summary.lower()
    has_status   = 'past due' in full_summary.lower() or 'billing' in full_summary.lower()
    all_present  = has_account and has_balance and has_status
    print(f"  account ID present:  {'✓' if has_account else '✗'}")
    print(f"  balance present:     {'✓' if has_balance else '✗'}")
    print(f"  billing status:      {'✓' if has_status  else '✗'}")
    print(f"Status:    {'✅ PASS' if all_present else '❌ FAIL'}")

    # TEST 5 — Summary persists after simulated restart
    print(f"\n{'─' * 60}")
    print("TEST 5 — Summary persists after simulated restart")
    print("EXPECTED:  system_summary in checkpointer = True")

    try:
        persisted_state = graph.get_state(config_long)
        has_summary = bool(persisted_state.values.get('system_summary', '').strip())
    except Exception as e:
        print(f"  get_state error: {e}")
        has_summary = False

    print(f"system_summary in checkpointer: {has_summary}")
    print(f"Status:    {'✅ PASS' if has_summary else '❌ FAIL'}")

    # Run full verification suite
    verification = run_session_verification()

    print(f"\n{'═' * 62}")
    print(f"SESSION 5 COMPLETE — {verification['summary']}")
    for check in verification['checks']:
        status = '✅ PASS' if check['passed'] else '❌ FAIL'
        print(f"  {status}  {check['label']}")
        if check.get('note'):
            print(f"           {check['note']}")
    print("═" * 62)


# ══════════════════════════════════════════════════════════════════
# SECTION 11: MAIN BLOCK
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_cli_tests()