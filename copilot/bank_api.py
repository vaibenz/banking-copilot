"""Mock core banking API.

A small in-memory FastAPI service that stands in for a bank's core systems.
Every route is scoped to a customer id, so the copilot can never read or
change another customer's data through it.
"""

from __future__ import annotations

import copy
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_SEED = {
    "accounts": {
        "SAV-001": {"customer_id": "C001", "type": "savings", "currency": "SGD", "balance": 12450.30},
        "CUR-002": {"customer_id": "C001", "type": "current", "currency": "SGD", "balance": 2310.75},
        "SAV-101": {"customer_id": "C002", "type": "savings", "currency": "SGD", "balance": 880.00},
    },
    "cards": {
        "4417": {"customer_id": "C001", "type": "debit", "status": "active", "account_id": "CUR-002"},
        "9021": {"customer_id": "C001", "type": "credit", "status": "active", "account_id": None},
        "5530": {"customer_id": "C002", "type": "debit", "status": "active", "account_id": "SAV-101"},
    },
    "transactions": [
        {"id": "TXN-1001", "customer_id": "C001", "card": "4417", "date": "2026-10-18", "merchant": "FairPrice Finest", "amount": -86.40},
        {"id": "TXN-1002", "customer_id": "C001", "card": "9021", "date": "2026-10-17", "merchant": "Grab Transport", "amount": -14.20},
        {"id": "TXN-1003", "customer_id": "C001", "card": "9021", "date": "2026-10-16", "merchant": "ONLINE STORE XJ8 LTD", "amount": -499.00},
        {"id": "TXN-1004", "customer_id": "C001", "card": None, "date": "2026-10-15", "merchant": "Salary credit", "amount": 4200.00},
        {"id": "TXN-1005", "customer_id": "C001", "card": "4417", "date": "2026-10-12", "merchant": "Kopitiam", "amount": -6.50},
        {"id": "TXN-1006", "customer_id": "C001", "card": "9021", "date": "2026-07-01", "merchant": "Airline Booking", "amount": -1320.00},
        {"id": "TXN-2001", "customer_id": "C002", "card": "5530", "date": "2026-10-14", "merchant": "Cold Storage", "amount": -42.10},
    ],
    "disputes": [],
}

DISPUTE_WINDOW_DAYS = 60
TODAY = date(2026, 10, 21)

STORE: dict = {}


def reset_store() -> None:
    """Restore the seed data. Used by tests and evals so every run starts clean."""
    STORE.clear()
    STORE.update(copy.deepcopy(_SEED))


reset_store()

app = FastAPI(title="Mock Core Banking API", version="0.1.0")


class DisputeRequest(BaseModel):
    transaction_id: str
    reason: str = Field(min_length=3, max_length=500)


def _owned_card(customer_id: str, last4: str) -> dict:
    card = STORE["cards"].get(last4)
    if not card or card["customer_id"] != customer_id:
        raise HTTPException(404, f"No card ending {last4} on this profile")
    return card


@app.get("/customers/{customer_id}/accounts")
def get_accounts(customer_id: str):
    accounts = [
        {"account_id": aid, **{k: v for k, v in acc.items() if k != "customer_id"}}
        for aid, acc in STORE["accounts"].items()
        if acc["customer_id"] == customer_id
    ]
    if not accounts:
        raise HTTPException(404, "Customer not found")
    return {"accounts": accounts}


@app.get("/customers/{customer_id}/transactions")
def get_transactions(customer_id: str, limit: int = 5):
    limit = max(1, min(limit, 20))
    txns = [
        {k: v for k, v in t.items() if k != "customer_id"}
        for t in STORE["transactions"]
        if t["customer_id"] == customer_id
    ]
    txns.sort(key=lambda t: t["date"], reverse=True)
    return {"transactions": txns[:limit]}


@app.get("/customers/{customer_id}/cards")
def get_cards(customer_id: str):
    cards = [
        {"last4": last4, "type": c["type"], "status": c["status"]}
        for last4, c in STORE["cards"].items()
        if c["customer_id"] == customer_id
    ]
    return {"cards": cards}


@app.post("/customers/{customer_id}/cards/{last4}/freeze")
def freeze_card(customer_id: str, last4: str):
    card = _owned_card(customer_id, last4)
    already = card["status"] == "frozen"
    card["status"] = "frozen"
    return {"last4": last4, "status": "frozen", "already_frozen": already}


@app.post("/customers/{customer_id}/disputes", status_code=201)
def open_dispute(customer_id: str, req: DisputeRequest):
    txn = next((t for t in STORE["transactions"] if t["id"] == req.transaction_id), None)
    if not txn or txn["customer_id"] != customer_id:
        raise HTTPException(404, f"Transaction {req.transaction_id} not found on this profile")
    if txn["amount"] >= 0:
        raise HTTPException(422, "Only debit transactions can be disputed")
    age = (TODAY - date.fromisoformat(txn["date"])).days
    if age > DISPUTE_WINDOW_DAYS:
        raise HTTPException(422, f"Transaction is {age} days old, outside the {DISPUTE_WINDOW_DAYS} day dispute window")
    if any(d["transaction_id"] == txn["id"] for d in STORE["disputes"]):
        raise HTTPException(409, f"A dispute is already open for {txn['id']}")
    dispute = {
        "dispute_id": f"DSP-{len(STORE['disputes']) + 1:04d}",
        "transaction_id": txn["id"],
        "amount": txn["amount"],
        "reason": req.reason,
        "status": "under_review",
    }
    STORE["disputes"].append({**dispute, "customer_id": customer_id})
    return dispute
