"""Agent execution node with explicit registry."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, List

from agentscope.message import Msg

from app.state import GraphState, NodeResult


def _normalize_schedule(schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for item in schedule:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "agent_name": item.get("agent_name"),
                "priority": item.get("priority", 999),
                "reason": item.get("reason", ""),
                "expected_output": item.get("expected_output", ""),
            }
        )
    return cleaned


async def _execute_one(
    task: Dict[str, Any],
    context: Dict[str, Any],
    previous_results: List[NodeResult],
    registry: Dict[str, Any],
) -> NodeResult:
    agent_name = task.get("agent_name")
    priority = int(task.get("priority", 999))

    if not agent_name:
        return {
            "status": "error",
            "agent_name": "",
            "priority": priority,
            "data": {"error": "missing agent_name"},
            "message": "missing agent_name",
        }

    agent = registry.get(agent_name)
    if agent is None:
        return {
            "status": "error",
            "agent_name": agent_name,
            "priority": priority,
            "data": {"error": f"unknown agent_name: {agent_name}"},
            "message": f"unknown agent_name: {agent_name}",
        }

    input_msg = Msg(
        name="LangGraphOrchestrator",
        content=json.dumps(
            {
                "context": context,
                "reason": task.get("reason", ""),
                "expected_output": task.get("expected_output", ""),
                "previous_results": previous_results,
            },
            ensure_ascii=False,
        ),
        role="user",
    )

    try:
        reply_msg = await agent.reply(input_msg)
        raw = reply_msg.content
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            data = {"output": data}

        status = data.get("status", "success")
        message = data.get("message", "")
        if status == "not_implemented":
            status = "not_implemented"

        return {
            "status": status,
            "agent_name": agent_name,
            "priority": priority,
            "data": data,
            "message": message,
        }
    except Exception as exc:
        return {
            "status": "error",
            "agent_name": agent_name,
            "priority": priority,
            "data": {"error": str(exc)},
            "message": str(exc),
        }


async def execute_agents_node(state: GraphState, agent_registry: Dict[str, Any]) -> Dict[str, Any]:
    schedule = _normalize_schedule(state.get("agent_schedule", []))
    intention = state.get("intention_result", {})

    errors = list(state.get("errors", []))
    trace = list(state.get("execution_trace", []))

    context = {
        "reasoning": intention.get("reasoning", ""),
        "intents": intention.get("intents", []),
        "key_entities": intention.get("key_entities", {}),
        "rewritten_query": intention.get("rewritten_query", state.get("user_input", "")),
        "recent_dialogue": state.get("recent_context", []),
    }

    if not schedule:
        trace.append("execute(empty)")
        return {
            "agent_results": [],
            "errors": errors,
            "execution_trace": trace,
        }

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for task in schedule:
        grouped[int(task.get("priority", 999))].append(task)

    all_results: List[NodeResult] = []
    for priority in sorted(grouped.keys()):
        batch = grouped[priority]
        batch_results = await asyncio.gather(
            *[_execute_one(task, context, all_results, agent_registry) for task in batch],
            return_exceptions=False,
        )
        all_results.extend(batch_results)

    for item in all_results:
        if item.get("status") == "error":
            errors.append(f"execute.{item.get('agent_name')}: {item.get('message')}")

    trace.append("execute")
    return {
        "agent_results": all_results,
        "errors": errors,
        "execution_trace": trace,
    }
