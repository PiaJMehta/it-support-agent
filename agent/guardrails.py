"""Deterministic checks applied around the LLM. The model proposes, these rules decide."""
import re

TEAMS = {"IT Helpdesk", "Network", "Identity & Access", "Security", "Enterprise Apps"}
PRIORITIES = ["P1", "P2", "P3", "P4"]  # P1 = most urgent

SECURITY_PATTERNS = [
    r"phishing", r"suspicious (email|link|attachment)", r"malware", r"ransomware", r"virus",
    r"compromised", r"entered my (credentials|password)", r"hacked",
]

_SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(password|passwd|pwd|passcode|pin)\b\s*(is|:|=)\s*\S+"), r"\1 [REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED CARD]"),
]


def redact(text: str) -> str:
    """Remove passwords and card numbers before the text reaches the model or the logs."""
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def is_security_incident(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in SECURITY_PATTERNS)


def validate_route(args: dict, ticket_text: str, valid_kb_ids: set[str]) -> tuple[dict | None, list[str]]:
    """Return (clean_route, notes). clean_route is None when the model's call is unusable."""
    notes: list[str] = []
    team = args.get("team")
    priority = args.get("priority")
    kb_article = args.get("kb_article")
    summary = str(args.get("summary", ""))[:300]

    if team not in TEAMS:
        return None, [f"invalid team: {team!r}; allowed: {sorted(TEAMS)}"]
    if priority not in PRIORITIES:
        return None, [f"invalid priority: {priority!r}; allowed: {PRIORITIES}"]
    if kb_article is not None and kb_article not in valid_kb_ids:
        notes.append(f"unknown kb_article {kb_article!r} removed")
        kb_article = None

    # Hard rule: suspected security incidents always go to Security at P2 or higher.
    if is_security_incident(ticket_text):
        if team != "Security":
            notes.append(f"security override: team {team} -> Security")
            team = "Security"
        if PRIORITIES.index(priority) > PRIORITIES.index("P2"):
            notes.append(f"security override: priority {priority} -> P2")
            priority = "P2"

    return {"team": team, "priority": priority, "kb_article": kb_article, "summary": summary}, notes
