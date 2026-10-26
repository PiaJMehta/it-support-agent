"""Run the agent on data/tickets.json and report accuracy.

    python eval.py --llm mock                 # plumbing check, runs offline
    python eval.py --llm anthropic            # real model, this is the number you can report
"""
import argparse
import json
import sys
from pathlib import Path

from agent.agent import TriageAgent
from agent.llm import make_llm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if team accuracy is below this")
    args = parser.parse_args()

    tickets = json.loads((Path(__file__).parent / "data" / "tickets.json").read_text(encoding="utf-8"))
    agent = TriageAgent(make_llm(args.llm))
    team_ok = prio_ok = kb_ok = 0

    print(f"{'id':>3}  {'team':<4} {'prio':<4} {'kb':<4}  ticket")
    for t in tickets:
        r = agent.run(t["text"], t["user_id"])
        a, b, c = r.team == t["team"], r.priority == t["priority"], r.kb_article == t["kb"]
        team_ok, prio_ok, kb_ok = team_ok + a, prio_ok + b, kb_ok + c
        print(f"{t['id']:>3}  {'ok' if a else 'FAIL':<4} {'ok' if b else 'FAIL':<4} {'ok' if c else 'FAIL':<4}  {t['text'][:60]}")

    n = len(tickets)
    print(f"\nLLM: {args.llm}   tickets: {n}")
    print(f"team accuracy:     {team_ok}/{n} = {team_ok / n:.0%}")
    print(f"priority accuracy: {prio_ok}/{n} = {prio_ok / n:.0%}")
    print(f"KB article match:  {kb_ok}/{n} = {kb_ok / n:.0%}")
    if args.llm == "mock":
        print("NOTE: mock results test the code only. They are not a model accuracy.")
    return 0 if team_ok / n >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
