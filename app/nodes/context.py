"""Context-loading node."""
from __future__ import annotations

from typing import Any, Dict

from app.memory.ports import MemoryPort
from app.state import GraphState


async def load_context_node(state: GraphState, memory: MemoryPort) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    errors = list(state.get("errors", []))
    trace = list(state.get("execution_trace", []))

    try:
        long_term_summary = await memory.get_long_term_summary(user_input=user_input, max_messages=50)
    except Exception as exc:  # pragma: no cover - defensive branch
        long_term_summary = ""
        errors.append(f"load_context.long_term_summary_failed: {exc}")

    try:
        recent_context = memory.get_recent_context(n_turns=5)
    except Exception as exc:  # pragma: no cover - defensive branch
        recent_context = []
        errors.append(f"load_context.recent_context_failed: {exc}")

    trace.append("load_context")
    return {
        "long_term_summary": long_term_summary,
        "recent_context": recent_context,
        "errors": errors,
        "execution_trace": trace,
    }
