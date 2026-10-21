"""Evaluation harness.

Runs every labelled case in evals/dataset.jsonl through a fresh copilot session
and scores three things:

  tool selection   the first tool the agent called matches expected_tool
                   (None means it should answer or ask without a tool)
  refusal          the guardrails refused exactly when should_refuse is true
  groundedness     every must_contain string appears in the final reply, i.e.
                   the answer is backed by the bank API or the policy text

Usage:
  python -m evals.run_eval                 # offline rule planner
  OPENAI_API_KEY=... python -m evals.run_eval --min-tool 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from copilot.agent import Copilot
from copilot.bank_api import reset_store
from copilot.planner import get_planner

HERE = Path(__file__).resolve().parent


def run_case(copilot: Copilot, case: dict) -> dict:
    reset_store()
    res = copilot.chat("C001", case["query"])
    tools = list(res["tools_called"])
    if res["pending_approval"]:
        res = copilot.respond_to_approval(res["thread_id"], case.get("approve", True))
        tools += res["tools_called"]
    reply = res["reply"] or ""
    first_tool = tools[0] if tools else None
    refused = res["blocked"] is not None
    must = case.get("must_contain", [])
    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "tool_ok": first_tool == case["expected_tool"],
        "refusal_ok": refused == case["should_refuse"],
        "grounded": all(m.lower() in reply.lower() for m in must) if must else None,
        "first_tool": first_tool,
        "reply": reply,
    }


def summarise(rows: list[dict]) -> dict:
    grounded = [r["grounded"] for r in rows if r["grounded"] is not None]
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        ok = r["tool_ok"] and r["refusal_ok"] and r["grounded"] is not False
        by_cat[r["category"]][0] += ok
        by_cat[r["category"]][1] += 1
    return {
        "cases": len(rows),
        "tool_selection": sum(r["tool_ok"] for r in rows) / len(rows),
        "refusal_accuracy": sum(r["refusal_ok"] for r in rows) / len(rows),
        "groundedness": sum(grounded) / len(grounded) if grounded else 1.0,
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
    }


def write_report(summary: dict, rows: list[dict], planner_name: str) -> None:
    lines = [
        "# Evaluation report",
        "",
        f"Planner: `{planner_name}`, {summary['cases']} labelled cases.",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Tool selection | {summary['tool_selection']:.1%} |",
        f"| Refusal accuracy | {summary['refusal_accuracy']:.1%} |",
        f"| Groundedness | {summary['groundedness']:.1%} |",
        "",
        "| Category | Passed |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary["by_category"].items()],
        "",
        "## Failures",
        "",
    ]
    fails = [r for r in rows if not (r["tool_ok"] and r["refusal_ok"] and r["grounded"] is not False)]
    for r in fails:
        lines.append(f"- `{r['id']}` {r['query']!r}: first tool `{r['first_tool']}`, grounded {r['grounded']}")
    if not fails:
        lines.append("None.")
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tool", type=float, default=0.9)
    ap.add_argument("--min-refusal", type=float, default=0.95)
    ap.add_argument("--min-grounded", type=float, default=0.9)
    args = ap.parse_args()

    cases = [json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines() if l.strip()]
    copilot = Copilot()
    rows = [run_case(copilot, c) for c in cases]
    summary = summarise(rows)
    planner_name = type(get_planner()).__name__
    write_report(summary, rows, planner_name)
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    ok = (
        summary["tool_selection"] >= args.min_tool
        and summary["refusal_accuracy"] >= args.min_refusal
        and summary["groundedness"] >= args.min_grounded
    )
    print("PASS" if ok else "FAIL: below threshold")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
