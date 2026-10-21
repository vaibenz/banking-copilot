"""LangGraph agent.

    guard_input ──blocked──▶ END
         │
         ▼
       plan ──no tool calls──▶ END
         │
         ├── read-only tools ─────────▶ tools ──▶ plan
         └── state-changing tools ──▶ approval (interrupt)
                                         ├── approved ──▶ tools
                                         └── declined ──▶ END
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from .guardrails import REQUIRES_APPROVAL, mask_pii, screen_input
from .planner import get_planner
from .tools import TOOLS_BY_NAME, current_customer

MAX_TOOL_ROUNDS = 4


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    blocked: str | None
    tool_rounds: int


def build_graph(planner=None, checkpointer=None):
    planner = planner or get_planner()

    def guard_input(state: State):
        human = state["messages"][-1]
        masked = mask_pii(human.content)
        screen = screen_input(masked.text)
        update = {"messages": [HumanMessage(content=masked.text, id=human.id)], "blocked": None, "tool_rounds": 0}
        if not screen.allowed:
            update["blocked"] = screen.category
            update["messages"].append(AIMessage(content=screen.message))
        return update

    def plan(state: State):
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(content="I wasn't able to finish that. Please try rephrasing your request.")]}
        msg = planner.plan(state["messages"])
        if not msg.tool_calls and isinstance(msg.content, str):
            msg = AIMessage(content=mask_pii(msg.content).text, id=msg.id)
        return {"messages": [msg]}

    def approval(state: State):
        calls = [c for c in state["messages"][-1].tool_calls if c["name"] in REQUIRES_APPROVAL]
        decision = interrupt({"actions": [{"tool": c["name"], "args": c["args"]} for c in calls]})
        if decision.get("approved"):
            return Command(goto="tools")
        declined = [
            ToolMessage(content=json.dumps({"declined": True}), name=c["name"], tool_call_id=c["id"])
            for c in state["messages"][-1].tool_calls
        ]
        return Command(
            goto=END,
            update={"messages": declined + [AIMessage(content="No problem, I haven't made any changes.")]},
        )

    def tools(state: State):
        token = current_customer.set(state["customer_id"])
        try:
            out = []
            for call in state["messages"][-1].tool_calls:
                fn = TOOLS_BY_NAME.get(call["name"])
                result = fn.invoke(call["args"]) if fn else json.dumps({"error": f"Unknown tool {call['name']}"})
                out.append(ToolMessage(content=result, name=call["name"], tool_call_id=call["id"]))
        finally:
            current_customer.reset(token)
        return {"messages": out, "tool_rounds": state.get("tool_rounds", 0) + 1}

    def after_guard(state: State):
        return END if state.get("blocked") else "plan"

    def after_plan(state: State):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if any(c["name"] in REQUIRES_APPROVAL for c in last.tool_calls):
            return "approval"
        return "tools"

    g = StateGraph(State)
    g.add_node("guard_input", guard_input)
    g.add_node("plan", plan)
    g.add_node("approval", approval, destinations=("tools", END))
    g.add_node("tools", tools)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, ["plan", END])
    g.add_conditional_edges("plan", after_plan, ["tools", "approval", END])
    g.add_edge("tools", "plan")
    return g.compile(checkpointer=checkpointer or MemorySaver())


class Copilot:
    """Thin session wrapper used by the API, the CLI and the eval harness."""

    def __init__(self, planner=None):
        self.graph = build_graph(planner)

    def _cfg(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _result(self, thread_id: str, start: int) -> dict:
        snap = self.graph.get_state(self._cfg(thread_id))
        new = snap.values["messages"][start:]
        pending = None
        for task in snap.tasks:
            if task.interrupts:
                pending = task.interrupts[0].value["actions"]
        replies = [m.content for m in new if isinstance(m, AIMessage) and m.content]
        return {
            "reply": replies[-1] if replies else None,
            "pending_approval": pending,
            "tools_called": [c["name"] for m in new if isinstance(m, AIMessage) for c in m.tool_calls],
            "blocked": snap.values.get("blocked"),
        }

    def _count(self, thread_id: str) -> int:
        snap = self.graph.get_state(self._cfg(thread_id))
        return len(snap.values.get("messages", [])) if snap.values else 0

    def chat(self, customer_id: str, text: str, thread_id: str | None = None) -> dict:
        thread_id = thread_id or uuid.uuid4().hex
        start = self._count(thread_id)
        self.graph.invoke({"messages": [HumanMessage(text)], "customer_id": customer_id}, self._cfg(thread_id))
        return {"thread_id": thread_id, **self._result(thread_id, start)}

    def respond_to_approval(self, thread_id: str, approved: bool) -> dict:
        start = self._count(thread_id)
        self.graph.invoke(Command(resume={"approved": approved}), self._cfg(thread_id))
        return {"thread_id": thread_id, **self._result(thread_id, start)}
