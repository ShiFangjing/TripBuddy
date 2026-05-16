#!/usr/bin/env python3
"""CLI entry for LangGraph rewrite phase-1."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph import ainvoke_once, build_graph, build_runtime, run_self_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph rewrite CLI (phase-1)")
    parser.add_argument("--self-check", action="store_true", help="Compile graph and run offline smoke check")
    parser.add_argument("--user-id", default=None, help="User ID (if omitted in interactive mode, will prompt)")
    parser.add_argument("--session-id", default=None, help="Session ID")
    parser.add_argument("--query", default=None, help="Run one-shot query and exit")
    return parser.parse_args()


def resolve_user_id(args: argparse.Namespace) -> str:
    """Resolve user id with interactive prompt for IDE run-button usage."""
    if args.user_id:
        return args.user_id

    # Keep one-shot/query mode non-blocking for scripts and regression.
    if args.query:
        return "default_user"

    try:
        user_id = input("Enter user_id (press Enter to use default_user): ").strip()
    except (EOFError, KeyboardInterrupt):
        user_id = ""
    return user_id or "default_user"


async def run_once(args: argparse.Namespace) -> int:
    user_id = resolve_user_id(args)
    session_id = args.session_id or str(uuid.uuid4())[:8]
    runtime = build_runtime(user_id=user_id, session_id=session_id)
    app = build_graph(runtime)

    if args.query:
        result = await ainvoke_once(app, user_id=user_id, session_id=session_id, user_input=args.query)
        print(result.get("final_response", ""))
        print("\n--- agent_results ---")
        print(json.dumps(result.get("agent_results", []), ensure_ascii=False, indent=2))
        return 0

    print("LangGraph rewrite CLI (phase-1). Type exit to quit.")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return 0

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Bye")
            return 0
        if user_input.lower() == "help":
            print("Commands: help / exit / enter a natural-language request")
            continue

        result = await ainvoke_once(app, user_id=user_id, session_id=session_id, user_input=user_input)
        print(result.get("final_response", ""))


def main() -> int:
    args = parse_args()

    if args.self_check:
        ok, msg = run_self_check()
        print(msg)
        return 0 if ok else 1

    return asyncio.run(run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())
