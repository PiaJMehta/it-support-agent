import json
from pathlib import Path

import pytest

from agent.agent import TriageAgent
from agent.guardrails import is_security_incident, redact, validate_route
from agent.kb import KnowledgeBase, load_articles
from agent.llm import MockLLM, Turn, ToolCall

ROOT = Path(__file__).resolve().parent.parent
TICKETS = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(load_articles(ROOT / "kb"))


# ---------- retrieval ----------
@pytest.mark.parametrize("ticket", TICKETS, ids=lambda t: f"ticket{t['id']}")
def test_kb_top_hit_is_expected_article(kb, ticket):
    assert kb.search(ticket["text"], k=1)[0]["id"] == ticket["kb"]


def test_kb_returns_nothing_for_gibberish(kb):
    assert kb.search("zzzz qqqq xxxx") == []


# ---------- guardrails ----------
def test_redact_removes_passwords_and_cards():
    text = "my password is hunter2 and card 4111 1111 1111 1111"
    out = redact(text)
    assert "hunter2" not in out and "4111" not in out


def test_invalid_team_rejected(kb):
    route, notes = validate_route({"team": "Wizards", "priority": "P3"}, "x", kb.ids())
    assert route is None and "invalid team" in notes[0]


def test_invalid_priority_rejected(kb):
    route, _ = validate_route({"team": "Network", "priority": "P9"}, "x", kb.ids())
    assert route is None


def test_unknown_kb_article_dropped(kb):
    route, notes = validate_route({"team": "Network", "priority": "P3", "kb_article": "made-up"}, "x", kb.ids())
    assert route["kb_article"] is None and notes


def test_security_override_forces_team_and_priority(kb):
    text = "I entered my credentials on a suspicious link"
    assert is_security_incident(text)
    route, notes = validate_route({"team": "IT Helpdesk", "priority": "P4", "summary": "s"}, text, kb.ids())
    assert route["team"] == "Security" and route["priority"] == "P2" and len(notes) == 2


# ---------- agent loop ----------
@pytest.mark.parametrize("ticket", TICKETS, ids=lambda t: f"ticket{t['id']}")
def test_agent_with_mock_routes_all_tickets(ticket):
    result = TriageAgent(MockLLM()).run(ticket["text"], ticket["user_id"])
    assert result.team == ticket["team"]
    assert result.priority == ticket["priority"]
    assert not result.escalated_to_human


def test_secrets_never_reach_the_model():
    seen = []

    class Spy(MockLLM):
        def step(self, system, messages, tools):
            seen.append(messages[0]["content"])
            return super().step(system, messages, tools)

    TriageAgent(Spy()).run("VPN broken, my password is hunter2", "u1001")
    assert all("hunter2" not in s for s in seen)


def test_agent_escalates_when_model_never_routes():
    class Lazy:
        def step(self, system, messages, tools):
            return Turn("done", [], [{"type": "text", "text": "done"}])

    result = TriageAgent(Lazy()).run("printer offline", "u1")
    assert result.escalated_to_human and result.team == "IT Helpdesk"


def test_agent_recovers_from_invalid_tool_call():
    """First route_ticket call is invalid; the error goes back to the model, which retries correctly."""

    class Retry:
        def step(self, system, messages, tools):
            stage = sum(1 for m in messages if m["role"] == "assistant")
            if stage == 0:
                args = {"team": "Wizards", "priority": "P3", "summary": "x"}
            elif stage == 1:
                assert messages[-1]["content"][0]["is_error"] is True
                args = {"team": "Network", "priority": "P3", "summary": "x"}
            else:
                return Turn("ok", [], [{"type": "text", "text": "ok"}])
            return Turn("", [ToolCall(f"c{stage}", "route_ticket", args)],
                        [{"type": "tool_use", "id": f"c{stage}", "name": "route_ticket", "input": args}])

    result = TriageAgent(Retry()).run("wifi slow", "u1")
    assert result.team == "Network" and not result.escalated_to_human
