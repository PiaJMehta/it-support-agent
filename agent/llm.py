"""LLM backends. Both return the same Turn object so the agent loop does not care which is used."""
import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: list[dict] = field(default_factory=list)  # message content in Anthropic block format


class AnthropicLLM:
    """Real model. Needs ANTHROPIC_API_KEY. Set AGENT_MODEL to change the model."""

    def __init__(self, model: str | None = None):
        import anthropic  # imported here so the mock and tests work without the package

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("AGENT_MODEL", "claude-haiku-5-5")

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=system, tools=tools, messages=messages
        )
        text, calls, content = "", [], []
        for block in resp.content:
            if block.type == "text":
                text += block.text
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)})
        return Turn(text, calls, content)


class MockLLM:
    """Scripted stand-in with fixed rules. It tests the plumbing (loop, tools, guardrails),
    NOT model quality. Never report its accuracy as your agent's accuracy."""

    P1 = ("credentials", "phishing", "malware")
    P2 = ("locked out", "can't log in at all", "customer call")
    P4 = ("printer", "install", "access to", "not syncing", "fans")

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        stage = sum(1 for m in messages if m["role"] == "assistant")
        ticket = re.search(r"<ticket>\n(.*?)\n</ticket>", messages[0]["content"], re.S).group(1)

        if stage == 0:
            return self._call("search_kb", {"query": ticket}, stage)
        if stage == 1:
            results = json.loads(messages[-1]["content"][0]["content"])["results"]
            top = results[0] if results else {"team": "IT Helpdesk", "id": None}
            low = ticket.lower()
            if any(k in low for k in self.P1):
                priority = "P1"
            elif any(k in low for k in self.P2):
                priority = "P2"
            elif any(k in low for k in self.P4):
                priority = "P4"
            else:
                priority = "P3"
            args = {"team": top["team"], "priority": priority, "summary": ticket[:120]}
            if top["id"]:
                args["kb_article"] = top["id"]
            return self._call("route_ticket", args, stage)
        text = "Thanks, your ticket has been routed. Please follow the steps in the linked article while you wait."
        return Turn(text, [], [{"type": "text", "text": text}])

    @staticmethod
    def _call(name: str, args: dict, stage: int) -> Turn:
        call_id = f"mock_{stage}"
        return Turn("", [ToolCall(call_id, name, args)], [{"type": "tool_use", "id": call_id, "name": name, "input": args}])


def make_llm(kind: str):
    if kind == "mock":
        return MockLLM()
    if kind == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"unknown llm {kind!r}")
