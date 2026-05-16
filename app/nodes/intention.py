"""Intention node that reuses current IntentionAgent protocol."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from agentscope.message import Msg

from app.state import GraphState


def _fallback_intention(user_input: str) -> Dict[str, Any]:
    return {
        "reasoning": "fallback",
        "intents": [{"type": "information_query", "confidence": 0.5}],
        "key_entities": {},
        "rewritten_query": user_input,
        "agent_schedule": [],
    }


async def intention_node(state: GraphState, intention_agent: Any) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    recent_context = state.get("recent_context", [])
    long_term_summary = state.get("long_term_summary", "")

    errors = list(state.get("errors", []))
    trace = list(state.get("execution_trace", []))

    if not user_input.strip():
        trace.append("intention(empty)")
        fallback = _fallback_intention(user_input)
        return {
            "intention_result": fallback,
            "agent_schedule": [],
            "errors": errors,
            "execution_trace": trace,
        }

    context_messages: List[Msg] = []
    if long_term_summary:
        context_messages.append(Msg(name="system", content=long_term_summary, role="system"))

    for msg in recent_context:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        context_messages.append(Msg(name=role, content=content, role=role))

    context_messages.append(Msg(name="user", content=user_input, role="user"))

    try:
        reply_msg = await intention_agent.reply(context_messages)
        raw = reply_msg.content
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError(f"intention payload must be dict, got {type(parsed).__name__}")
    except Exception as exc:
        errors.append(f"intention.failed: {exc}")
        parsed = _fallback_intention(user_input)

    schedule = parsed.get("agent_schedule", [])
    if not isinstance(schedule, list):
        errors.append("intention.invalid_schedule_type")
        schedule = []

    trace.append("intention")
    return {
        "intention_result": parsed,
        "agent_schedule": schedule,
        "errors": errors,
        "execution_trace": trace,
    }
