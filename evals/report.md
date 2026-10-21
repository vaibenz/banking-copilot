# Evaluation report

Planner: `RulePlanner`, 43 labelled cases.

| Metric | Score |
|---|---|
| Tool selection | 100.0% |
| Refusal accuracy | 100.0% |
| Groundedness | 96.8% |

| Category | Passed |
|---|---|
| balance | 5/5 |
| dispute | 5/5 |
| freeze | 6/6 |
| other | 1/1 |
| policy | 12/13 |
| refusal | 9/9 |
| transactions | 4/4 |

## Failures

- `policy-10` 'How long do I have to raise a dispute?': first tool `search_policy`, grounded False
