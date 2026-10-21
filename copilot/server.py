"""HTTP interface for the copilot.

Run with:  uvicorn copilot.server:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import Copilot

app = FastAPI(title="Banking Support Copilot", version="0.1.0")
copilot = Copilot()


class ChatRequest(BaseModel):
    customer_id: str  # in production this comes from the authenticated session, not the body
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


@app.post("/chat")
def chat(req: ChatRequest):
    return copilot.chat(req.customer_id, req.message, req.thread_id)


@app.post("/approve")
def approve(req: ApprovalRequest):
    try:
        return copilot.respond_to_approval(req.thread_id, req.approved)
    except Exception as exc:  # no pending action on this thread
        raise HTTPException(400, str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}
