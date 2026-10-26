"""Tools the agent can call, with JSON schemas in the Anthropic tool-use format."""
import json
from pathlib import Path

from .guardrails import PRIORITIES, TEAMS, validate_route
from .kb import KnowledgeBase

TOOL_SCHEMAS = [
    {
        "name": "search_kb",
        "description": "Search the IT knowledge base. Returns the best matching articles with steps and the owning team.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Short search query about the problem."}},
            "required": ["query"],
        },
    },
    {
        "name": "get_ticket_history",
        "description": "Get the user's previous tickets, to spot repeated issues.",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "route_ticket",
        "description": "Assign the ticket to a team with a priority. Call this exactly once when you have decided.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": sorted(TEAMS)},
                "priority": {"type": "string", "enum": PRIORITIES},
                "kb_article": {"type": "string", "description": "id of the most relevant KB article, if any"},
                "summary": {"type": "string", "description": "One-sentence summary for the assigned team."},
            },
            "required": ["team", "priority", "summary"],
        },
    },
]


class ToolBox:
    """Executes tool calls for one ticket. Errors are returned to the model so it can retry."""

    def __init__(self, kb: KnowledgeBase, history_path: str | Path, ticket_text: str):
        self.kb = kb
        self.history = json.loads(Path(history_path).read_text(encoding="utf-8"))
        self.ticket_text = ticket_text
        self.route: dict | None = None
        self.notes: list[str] = []

    def execute(self, name: str, args: dict) -> tuple[dict, bool]:
        """Return (output, is_error)."""
        try:
            if name == "search_kb":
                return {"results": self.kb.search(str(args["query"]), k=3)}, False
            if name == "get_ticket_history":
                return {"tickets": self.history.get(str(args["user_id"]), [])}, False
            if name == "route_ticket":
                route, notes = validate_route(args, self.ticket_text, self.kb.ids())
                if route is None:
                    return {"error": notes[0]}, True
                self.route = route
                self.notes.extend(notes)
                return {"routed": route, "notes": notes}, False
            return {"error": f"unknown tool {name!r}"}, True
        except KeyError as exc:
            return {"error": f"missing argument {exc}"}, True
