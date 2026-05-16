"""Phase-1 route normalizer: focus migrated nodes only."""
from __future__ import annotations

from typing import Any, Dict, List

from app.state import GraphState

MIGRATED_AGENTS = {
    "event_collection",
    "preference",
    "memory_query",
    "information_query",
    "itinerary_planning",
}
DISABLED_AGENTS = {"rag_knowledge"}
KNOWN_UNMIGRATED_AGENTS = set()


def _pref_cue(text: str) -> bool:
    cues = ["\u559c\u6b22", "\u504f\u597d", "\u642c\u5bb6", "\u9760\u7a97", "\u9152\u5e97", "\u5e38\u5750", "\u822a\u7a7a", "\u5ea7\u4f4d"]
    return any(c in text for c in cues)


def _event_cue(text: str) -> bool:
    cues = ["\u4ece", "\u53bb", "\u51fa\u53d1", "\u8fd4\u7a0b", "\u884c\u7a0b", "\u51fa\u5dee", "\u65c5\u6e38", "\u65e5\u671f", "\u5468", "\u6708", "\u5929"]
    return any(c in text for c in cues)


def _memory_query_cue(text: str) -> bool:
    cues = [
        "\u6211\u4f4f\u5728\u54ea\u91cc",
        "\u6211\u4f4f\u5728\u54ea",
        "\u6211\u5bb6\u4f4f\u5728\u54ea",
        "\u6211\u4e4b\u524d",
        "\u6211\u8fc7\u53bb",
        "\u6211\u8bf4\u8fc7",
        "\u53bb\u8fc7\u54ea\u4e9b",
        "\u504f\u597d\u6709\u54ea\u4e9b",
        "\u5386\u53f2",
        "\u8fd8\u8bb0\u5f97",
    ]
    return any(c in text for c in cues)


def route_node(state: GraphState) -> Dict[str, Any]:
    schedule = state.get("agent_schedule", [])
    user_input = (state.get("user_input", "") or "").strip()

    errors = list(state.get("errors", []))
    trace = list(state.get("execution_trace", []))

    filtered: List[Dict[str, Any]] = []
    for item in schedule:
        if not isinstance(item, dict):
            continue
        name = item.get("agent_name")
        if name in MIGRATED_AGENTS:
            filtered.append(item)
        elif name in DISABLED_AGENTS:
            errors.append(f"route.skipped_disabled_agent: {name}")
        elif name in KNOWN_UNMIGRATED_AGENTS:
            errors.append(f"route.skipped_unmigrated_agent: {name}")
        else:
            # Keep unknown names and let execution node return structured errors.
            filtered.append(item)

    # Phase-1 deterministic fallback: keep migrated agents testable even when intention drifts.
    if not filtered and user_input:
        if _memory_query_cue(user_input):
            filtered.append(
                {
                    "agent_name": "memory_query",
                    "priority": 1,
                    "reason": "phase1_fallback_memory_query",
                    "expected_output": "answer based on long-term memory and history",
                }
            )
        if _pref_cue(user_input) and not _memory_query_cue(user_input):
            filtered.append(
                {
                    "agent_name": "preference",
                    "priority": 1,
                    "reason": "phase1_fallback_preference",
                    "expected_output": "extract user preferences",
                }
            )
        if _event_cue(user_input):
            filtered.append(
                {
                    "agent_name": "event_collection",
                    "priority": 1,
                    "reason": "phase1_fallback_event_collection",
                    "expected_output": "extract origin/destination/date",
                }
            )

    trace.append("route")
    return {
        "agent_schedule": filtered,
        "errors": errors,
        "execution_trace": trace,
    }
