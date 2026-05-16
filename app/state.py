"""State contracts for LangGraph rewrite phase-1."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


class NodeResult(TypedDict, total=False):
    """Normalized node execution result."""

    status: Literal["success", "error", "skipped", "not_implemented"]
    agent_name: str
    data: Dict[str, Any]
    message: str
    priority: int


class GraphState(TypedDict, total=False):
    """Shared state contract across graph nodes."""

    user_id: str
    session_id: str
    user_input: str

    long_term_summary: str
    recent_context: List[Dict[str, Any]]

    intention_result: Dict[str, Any]
    agent_schedule: List[Dict[str, Any]]
    agent_results: List[NodeResult]

    final_response: str
    errors: List[str]
    execution_trace: List[str]


REQUIRED_KEYS = [
    "user_id",
    "session_id",
    "user_input",
    "long_term_summary",
    "recent_context",
    "intention_result",
    "agent_schedule",
    "agent_results",
    "final_response",
    "errors",
    "execution_trace",
]


def make_initial_state(user_id: str, session_id: str, user_input: str) -> GraphState:
    """Create a complete initial state with defaults."""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "user_input": user_input,
        "long_term_summary": "",
        "recent_context": [],
        "intention_result": {},
        "agent_schedule": [],
        "agent_results": [],
        "final_response": "",
        "errors": [],
        "execution_trace": [],
    }
