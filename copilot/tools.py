"""Reusable tools the agent can call.

Each tool is a LangChain tool (so an LLM can bind to its schema) whose body
calls the mock core banking API. The customer id is never a tool argument: it
is set by the agent from the authenticated session, so the model cannot be
talked into querying another customer's data.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Literal

import httpx
from fastapi.testclient import TestClient
from langchain_core.tools import tool

from .bank_api import app as bank_app
from .rag import get_index

current_customer: ContextVar[str] = ContextVar("current_customer")

_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    """In-process client by default; set BANK_API_URL to point at a running API."""
    global _client
    if _client is None:
        import os

        url = os.getenv("BANK_API_URL")
        _client = httpx.Client(base_url=url, timeout=10) if url else TestClient(bank_app)
    return _client


def _call(method: str, path: str, **kwargs) -> str:
    cid = current_customer.get()
    resp = get_client().request(method, f"/customers/{cid}{path}", **kwargs)
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.reason_phrase
        return json.dumps({"error": detail, "status": resp.status_code})
    return json.dumps(resp.json())


@tool
def get_balance(account_type: Literal["savings", "current", "all"] = "all") -> str:
    """Get the customer's account balances. Use for questions about how much money is in an account."""
    data = json.loads(_call("GET", "/accounts"))
    if "error" in data or account_type == "all":
        return json.dumps(data)
    data["accounts"] = [a for a in data["accounts"] if a["type"] == account_type]
    return json.dumps(data)


@tool
def list_transactions(limit: int = 5) -> str:
    """List the customer's most recent transactions, newest first. Use to find a transaction id before a dispute."""
    return _call("GET", "/transactions", params={"limit": limit})


@tool
def freeze_card(card_last4: str) -> str:
    """Freeze one of the customer's cards by its last 4 digits. Changes account state, so it needs customer approval."""
    return _call("POST", f"/cards/{card_last4}/freeze")


@tool
def open_dispute(transaction_id: str, reason: str) -> str:
    """Open a dispute on one of the customer's transactions (e.g. TXN-1003). Changes account state, so it needs customer approval."""
    return _call("POST", "/disputes", json={"transaction_id": transaction_id, "reason": reason})


@tool
def search_policy(query: str) -> str:
    """Search the bank's policy documents (fees, card security, disputes, accounts). Answer policy questions only from these results."""
    return json.dumps({"results": get_index().search(query, k=2)})


TOOLS = [get_balance, list_transactions, freeze_card, open_dispute, search_policy]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
