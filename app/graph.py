"""LangGraph assembly for phase-1 rewrite."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from langgraph.graph import END, START, StateGraph

from agents.intention_agent import IntentionAgent
from config import LLM_CONFIG, SYSTEM_CONFIG
from context.memory_manager import MemoryManager

from app.memory.adapters import MemoryManagerAdapter
from app.nodes.aggregate import aggregate_node, persist_node
from app.nodes.context import load_context_node
from app.nodes.execution import execute_agents_node
from app.nodes.intention import intention_node
from app.nodes.registry import build_explicit_registry
from app.nodes.route import route_node
from app.state import GraphState, make_initial_state


@dataclass
class GraphRuntime:
    model: Any
    memory_manager: Any
    memory_adapter: Any
    intention_agent: Any
    agent_registry: Dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runtime(
    user_id: str,
    session_id: str,
    model: Optional[Any] = None,
    memory_manager: Optional[MemoryManager] = None,
    intention_agent: Optional[Any] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
) -> GraphRuntime:
    if model is None:
        timeout_sec = SYSTEM_CONFIG.get("timeout", 60)
        model = OpenAIChatModel(
            model_name=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": float(timeout_sec)},
            temperature=LLM_CONFIG.get("temperature", 0.7),
            max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        )

    if memory_manager is None:
        memory_manager = MemoryManager(
            user_id=user_id,
            session_id=session_id,
            llm_model=model,
        )

    memory_adapter = MemoryManagerAdapter(memory_manager)

    if intention_agent is None:
        intention_agent = IntentionAgent(name="IntentionAgent", model=model)

    if agent_registry is None:
        agent_registry = build_explicit_registry(model=model, memory_manager=memory_manager, project_root=_project_root())

    return GraphRuntime(
        model=model,
        memory_manager=memory_manager,
        memory_adapter=memory_adapter,
        intention_agent=intention_agent,
        agent_registry=agent_registry,
    )


def build_graph(runtime: GraphRuntime):
    graph = StateGraph(GraphState)

    async def _load_context(state: GraphState):
        return await load_context_node(state, runtime.memory_adapter)

    async def _intention(state: GraphState):
        return await intention_node(state, runtime.intention_agent)

    def _route(state: GraphState):
        return route_node(state)

    async def _execute(state: GraphState):
        return await execute_agents_node(state, runtime.agent_registry)

    def _aggregate(state: GraphState):
        return aggregate_node(state)

    def _persist(state: GraphState):
        return persist_node(state, runtime.memory_adapter)

    graph.add_node("load_context", _load_context)
    graph.add_node("intention", _intention)
    graph.add_node("route", _route)
    graph.add_node("execute", _execute)
    graph.add_node("aggregate", _aggregate)
    graph.add_node("persist", _persist)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "intention")
    graph.add_edge("intention", "route")
    graph.add_edge("route", "execute")
    graph.add_edge("execute", "aggregate")
    graph.add_edge("aggregate", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


async def ainvoke_once(app, user_id: str, session_id: str, user_input: str) -> GraphState:
    state = make_initial_state(user_id=user_id, session_id=session_id, user_input=user_input)
    result = await app.ainvoke(state)
    return result


# -------- self-check runtime (offline, no network dependency) --------
class _DummyMemory:
    def __init__(self):
        self.short_term_messages = []
        self.long_term_messages = []

    async def get_long_term_summary(self, user_input: str, max_messages: int = 50) -> str:
        return "dummy summary"

    def get_recent_context(self, n_turns: int = 5):
        return self.short_term_messages[-n_turns * 2 :]

    def get_preferences(self):
        return {}

    def add_message(self, role: str, content: str, metadata=None):
        msg = {"role": role, "content": content, "metadata": metadata or {}}
        self.short_term_messages.append(msg)
        self.long_term_messages.append(msg)


class _DummyIntentionAgent:
    async def reply(self, x):
        payload = {
            "reasoning": "self-check",
            "intents": [{"type": "smoke", "confidence": 1.0}],
            "key_entities": {},
            "rewritten_query": "self-check",
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1, "reason": "self-check", "expected_output": ""},
                {"agent_name": "preference", "priority": 1, "reason": "self-check", "expected_output": ""},
            ],
        }
        return Msg(name="IntentionAgent", content=json.dumps(payload, ensure_ascii=False), role="assistant")


class _DummyAgent:
    def __init__(self, name: str, payload: Dict[str, Any]):
        self.name = name
        self.payload = payload

    async def reply(self, x):
        return Msg(name=self.name, content=json.dumps(self.payload, ensure_ascii=False), role="assistant")


def build_self_check_runtime() -> GraphRuntime:
    memory = _DummyMemory()
    registry = {
        "event_collection": _DummyAgent(
            "event_collection",
            {
                "origin": "\u4e0a\u6d77",
                "destination": "\u5317\u4eac",
                "start_date": "2026-04-30",
                "missing_info": [],
            },
        ),
        "preference": _DummyAgent(
            "preference",
            {
                "preferences": [
                    {"type": "seat_preference", "value": "\u9760\u7a97", "action": "replace"}
                ],
                "has_preferences": True,
            },
        ),
    }

    return GraphRuntime(
        model=None,
        memory_manager=None,
        memory_adapter=memory,
        intention_agent=_DummyIntentionAgent(),
        agent_registry=registry,
    )


def run_self_check() -> tuple[bool, str]:
    try:
        runtime = build_self_check_runtime()
        app = build_graph(runtime)
        result = asyncio.run(ainvoke_once(app, user_id="self_check", session_id="self_check", user_input="self check"))

        final_response = result.get("final_response", "")
        trace = result.get("execution_trace", [])
        if not final_response:
            return False, "final_response is empty"

        expected_steps = {"load_context", "intention", "route", "execute", "aggregate", "persist"}
        if not expected_steps.issubset(set(trace)):
            return False, f"missing execution steps: expected {expected_steps}, got {trace}"

        return True, "self-check passed"
    except Exception as exc:  # pragma: no cover - CLI safety net
        return False, str(exc)
