# Agentic Banking Support Copilot

An LLM powered support copilot for a fictional bank, built as a LangGraph agent that plans and calls reusable tools against a mock core banking API. It is designed around three things a bank cares about before an agent reaches customers: guardrails, human approval for anything that changes an account, and measurable evaluations.

## What it does

A signed in customer can ask the copilot to

- check balances and recent transactions
- freeze a lost or stolen card
- dispute a transaction
- answer policy questions (fees, card security, disputes, accounts), grounded in the bank's policy documents through retrieval

## Architecture

```
user message
     │
     ▼
 guard_input ── PII masking, prompt injection and scope screening ──▶ refuse
     │
     ▼
    plan ── LLM with tools bound (or offline rule baseline) ──▶ answer
     │
     ├── read only tools ───────────────────────▶ tools ──▶ plan
     └── state changing tools ──▶ approval (LangGraph interrupt)
                                     ├── approved ──▶ tools
                                     └── declined ──▶ "no changes made"
```

| Component | File | Notes |
|---|---|---|
| Mock core banking API | `copilot/bank_api.py` | FastAPI, every route scoped to a customer id, dispute window and ownership rules |
| Tools | `copilot/tools.py` | `get_balance`, `list_transactions`, `freeze_card`, `open_dispute`, `search_policy` |
| Retrieval | `copilot/rag.py` | Policy sections indexed in ChromaDB |
| Guardrails | `copilot/guardrails.py` | PII masking, injection and scope screening, approval list |
| Agent graph | `copilot/agent.py` | LangGraph `StateGraph` with an `interrupt` before state changes and a checkpointer for multi turn sessions |
| Planners | `copilot/planner.py` | `LLMPlanner` (OpenAI or Gemini) and `RulePlanner` (deterministic offline baseline) |
| HTTP API | `copilot/server.py` | `POST /chat`, `POST /approve` |
| Evals | `evals/` | 43 labelled cases, scored on tool selection, refusals and groundedness |

## Guardrails

- **PII masking.** Card numbers (kept to the last 4 digits), NRIC/FIN, phone numbers and emails are masked before the message reaches the model or the conversation store, and again on the way out.
- **Prompt injection and scope screening.** Attempts to override instructions, reveal the system prompt, reach other customers' data, or ask for actions the copilot is not allowed to take (transfers, limit changes, unfreezing) are refused before planning.
- **Customer scoping by design.** The customer id is never a tool argument. It comes from the session, so the model cannot be talked into querying someone else's account, and the API rejects it anyway.
- **Human approval.** `freeze_card` and `open_dispute` pause the graph with a LangGraph `interrupt`. Nothing changes until the customer confirms, and a decline leaves the account untouched.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m copilot          # chat in the terminal
uvicorn copilot.server:app # or run the HTTP API
pytest -q                  # unit tests
python -m evals.run_eval   # evaluation gate, writes evals/report.md
```

With no API key the agent uses `RulePlanner`, so everything runs offline and in CI. To run on a real model:

```bash
pip install -r requirements-llm.txt
export OPENAI_API_KEY=...        # or GOOGLE_API_KEY=...
export COPILOT_MODEL=gpt-4o-mini # optional
python -m evals.run_eval --min-tool 0.85
```

## Evaluation

`evals/dataset.jsonl` holds 43 labelled cases across balances, transactions, card freezes, disputes, policy questions, adversarial prompts and small talk. Each case states the tool the agent should call first, whether it should be refused, whether the customer approves the action, and strings the final answer must contain to count as grounded.

Current results with the offline `RulePlanner` (see `evals/report.md`):

| Metric | Score |
|---|---|
| Tool selection | 100% |
| Refusal accuracy | 100% |
| Groundedness | 96.8% |

The one failure is instructive. "How long do I have to raise a dispute?" retrieves the wrong policy section because the index is lexical: the answer says disputes must be "raised within 60 days", which shares few terms with "how long". Swapping `PolicyIndex.embed` for a dense embedding model is the next step, and the eval set will show whether it fixes this without breaking the rest.

CI runs the unit tests and fails the build if any eval metric drops below its threshold.

## Notes

Harbour Demo Bank, its customers and its policies are entirely fictional.
