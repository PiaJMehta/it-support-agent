# IT Support Triage Agent

An LLM agent that reads an IT support ticket, searches a knowledge base, routes the ticket to the right team with a priority, and drafts a reply. It uses tool calling, a guardrail layer, and an evaluation set.

All data (knowledge base articles, tickets, history) is **fake** and written for this project.

## How it works

```
ticket -> redact secrets -> LLM <-> tools (search_kb, get_ticket_history, route_ticket)
                                   |
                                   +-> guardrails validate every route_ticket call
       -> TriageResult(team, priority, kb_article, summary, reply)
```

- **Agent loop** (`agent/agent.py`): the model chooses a tool, the code runs it, the result goes back to the model. It stops when the model replies without a tool call, or after 6 steps.
- **Retrieval** (`agent/kb.py`): BM25 keyword search written from scratch over the articles in `kb/`.
- **Guardrails** (`agent/guardrails.py`): the model proposes, rules decide.
  - Passwords and card numbers are redacted before the model sees the ticket.
  - Team and priority must come from fixed lists. Invalid calls return an error to the model so it can retry.
  - Suspected security incidents are always forced to the Security team at P2 or higher.
  - The ticket text is treated as untrusted data (prompt injection defence in the system prompt).
  - If the model never routes the ticket, it is escalated to a human instead of being dropped.
- **LLM backends** (`agent/llm.py`): `AnthropicLLM` (real) and `MockLLM` (fixed rules, for tests and CI).

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # 33 tests, no API key needed
python eval.py --llm mock                   # checks the code, offline

export ANTHROPIC_API_KEY=...                # real model
python -m agent.cli "VPN keeps dropping" --llm anthropic --user u1001
python eval.py --llm anthropic              # the real accuracy number
```

Set `AGENT_MODEL` to change the model (default `claude-haiku-5-5`).

Docker: `docker build -t it-agent . && docker run it-agent`

## Evaluation

`data/tickets.json` has 12 labelled tickets (expected team, priority and KB article).
`eval.py` reports team accuracy, priority accuracy and KB match.

**The mock model gets 100% by construction. That only proves the code works. Report only the numbers from `--llm anthropic`.**

## Ideas to extend

- Replace BM25 with embeddings (or combine both) and compare retrieval accuracy.
- Grow the eval set to 50+ tickets, including ambiguous and adversarial ones (prompt injection).
- Add a tool that creates a ticket in a mock ticketing API.
- Log every run (tool calls, latency, tokens) and add a small dashboard.
