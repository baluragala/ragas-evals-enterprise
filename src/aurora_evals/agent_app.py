"""The second system under test: a tool-using Aurora Bank agent (LangChain 1.x `create_agent`).

The agent talks to a small, deterministic, in-memory "core banking" backend, so every run is
reproducible and side effects (a frozen card, an opened dispute) can be inspected afterwards.

    agent = AuroraAgent()
    run = await agent.arun("I'm C-1001, freeze my card ending 4242")
    run.messages        # full LangChain message trace (Human → AI(tool_calls) → Tool → AI)
    run.backend.frozen_cards   # what actually happened in the "bank"
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from .config import APP_MODEL
from .knowledge_base import build_vector_store

# --------------------------------------------------------------------------------------
# Mock core-banking data (fictional)
# --------------------------------------------------------------------------------------
SEED_CUSTOMERS: dict[str, dict] = {
    "C-1001": {
        "name": "Priya Raman",
        "accounts": {"checking": 2450.75, "savings": 15200.00},
        "cards": ["4242"],
        "transactions": [
            {"id": "T-8001", "merchant": "COFFEE-CORNER", "amount": 4.85, "date": "2026-10-20"},
            {"id": "T-8002", "merchant": "AIRLINE-SKYWAYS", "amount": 612.40, "date": "2026-10-18"},
        ],
    },
    "C-1002": {
        "name": "Daniel Okafor",
        "accounts": {"checking": 830.10, "savings": 0.00},
        "cards": ["5555"],
        "transactions": [
            {"id": "T-9001", "merchant": "GROCER-MART", "amount": 54.20, "date": "2026-10-21"},
            {"id": "T-9002", "merchant": "CITY-POWER", "amount": 120.00, "date": "2026-10-19"},
            {"id": "T-9003", "merchant": "GIZMO-ONLINE", "amount": 189.99, "date": "2026-10-17"},
        ],
    },
}

SYSTEM_PROMPT = """You are Aurora Bank's virtual assistant. You can look up account data and take
actions using your tools.

Rules:
- Only help with Aurora Bank banking topics. Politely decline anything else (poems, jokes, general chat).
- Never give personalised investment, tax, or legal advice; refer those to a licensed Aurora Wealth advisor.
- For questions about Aurora products, fees, limits, or procedures, call `search_policy` and answer from it.
- Account actions need the customer's ID (format C-1234). If it is missing, ask for it.
- To dispute a charge, first call `list_recent_transactions` to find the transaction ID.
- Never display full card or account numbers; at most the last 4 digits.
- Keep answers short and factual."""

# v2 = v1 + fixes for the failures found by goal_accuracy in notebook 04:
#   * after freezing a card the agent never told the customer how to get a replacement
#   * off-topic / advice requests were declined without a referral or an offer to help
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + """

Follow-up rules (always apply these after the main action):
- After freezing a card, tell the customer to call the 24/7 Fraud Line at 1-800-555-0199 to report it
  and order a replacement.
- After opening a dispute, give the case ID and say a provisional credit is issued within 10 business days.
- When declining an off-topic or advice request, say what you CAN help with (Aurora banking questions),
  and refer investment questions to a licensed Aurora Wealth advisor."""


class BankBackend:
    """Per-run copy of the bank's state, so runs never leak into each other."""

    def __init__(self) -> None:
        self.customers = copy.deepcopy(SEED_CUSTOMERS)
        self.frozen_cards: set[str] = set()
        self.disputes: list[dict] = []
        self.escalations: list[dict] = []


@dataclass
class AgentRun:
    user_input: str
    messages: list[BaseMessage]
    backend: BankBackend
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def final_response(self) -> str:
        for m in reversed(self.messages):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    @property
    def tool_calls(self) -> list[dict]:
        return [
            {"name": tc["name"], "args": tc["args"]}
            for m in self.messages
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
        ]


def build_tools(backend: BankBackend, policy_retriever):
    """Create tools bound to one backend instance (closures keep the state per run)."""

    def _customer(customer_id: str) -> dict | None:
        return backend.customers.get(customer_id.strip().upper())

    @tool
    def get_account_balance(customer_id: str, account_type: Literal["checking", "savings"]) -> str:
        """Get the current balance of a customer's checking or savings account."""
        c = _customer(customer_id)
        if c is None:
            return f"No customer found with ID {customer_id}."
        return f"{account_type.title()} balance for {customer_id}: ${c['accounts'][account_type]:,.2f}"

    @tool
    def list_recent_transactions(customer_id: str) -> str:
        """List a customer's recent card transactions (ID, merchant, amount, date)."""
        c = _customer(customer_id)
        if c is None:
            return f"No customer found with ID {customer_id}."
        rows = [f"{t['id']} | {t['merchant']} | ${t['amount']:.2f} | {t['date']}" for t in c["transactions"]]
        return "\n".join(rows)

    @tool
    def freeze_card(customer_id: str, card_last4: str) -> str:
        """Freeze a customer's card immediately, identified by its last 4 digits."""
        c = _customer(customer_id)
        if c is None:
            return f"No customer found with ID {customer_id}."
        if card_last4 not in c["cards"]:
            return f"No card ending {card_last4} on file for {customer_id}."
        backend.frozen_cards.add(card_last4)
        return f"Card ending {card_last4} is now frozen. New purchases will be declined."

    @tool
    def open_dispute(customer_id: str, transaction_id: str) -> str:
        """Open a dispute for one of the customer's transactions, by transaction ID (e.g. T-9003)."""
        c = _customer(customer_id)
        if c is None:
            return f"No customer found with ID {customer_id}."
        txn = next((t for t in c["transactions"] if t["id"] == transaction_id), None)
        if txn is None:
            return f"Transaction {transaction_id} not found for {customer_id}."
        case_id = f"D-{len(backend.disputes) + 1:04d}"
        backend.disputes.append({"case_id": case_id, "customer_id": customer_id, **txn})
        return (
            f"Dispute {case_id} opened for {transaction_id} ({txn['merchant']}, ${txn['amount']:.2f}). "
            "Provisional credit is issued within 10 business days."
        )

    @tool
    def search_policy(query: str) -> str:
        """Search Aurora Bank's official policy documents (fees, limits, rates, procedures)."""
        docs = policy_retriever.invoke(query)
        return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)

    @tool
    def escalate_to_human(customer_id: str, summary: str) -> str:
        """Hand the conversation to a human agent when you cannot resolve it."""
        backend.escalations.append({"customer_id": customer_id, "summary": summary})
        return "Escalated to a human agent. Expected response time: 15 minutes."

    return [
        get_account_balance,
        list_recent_transactions,
        freeze_card,
        open_dispute,
        search_policy,
        escalate_to_human,
    ]


class AuroraAgent:
    def __init__(self, model: str = APP_MODEL, system_prompt: str = SYSTEM_PROMPT):
        self.llm = ChatOpenAI(model=model, temperature=0, max_retries=5)
        self.system_prompt = system_prompt
        self.policy_retriever = build_vector_store().as_retriever(search_kwargs={"k": 3})

    async def arun(self, user_input: str) -> AgentRun:
        backend = BankBackend()
        agent = create_agent(
            model=self.llm,
            tools=build_tools(backend, self.policy_retriever),
            system_prompt=self.system_prompt,
        )
        start = time.perf_counter()
        state = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})
        latency = time.perf_counter() - start

        messages = state["messages"]
        usage = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
        return AgentRun(
            user_input=user_input,
            messages=messages,
            backend=backend,
            latency_s=round(latency, 3),
            input_tokens=sum(u.get("input_tokens", 0) for u in usage),
            output_tokens=sum(u.get("output_tokens", 0) for u in usage),
        )
