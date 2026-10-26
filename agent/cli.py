"""Try the agent on one ticket:  python -m agent.cli "my VPN keeps dropping" --llm mock"""
import argparse
from dataclasses import asdict
import json

from .agent import TriageAgent
from .llm import make_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticket")
    parser.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    parser.add_argument("--user", default=None)
    args = parser.parse_args()
    result = TriageAgent(make_llm(args.llm)).run(args.ticket, args.user)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
