"""Aggregation and persistence nodes."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.memory.ports import MemoryPort
from app.state import GraphState, NodeResult


def _render_event_collection(data: Dict[str, Any]) -> List[str]:
    lines = ["[event_collection]"]
    for key in ["origin", "destination", "start_date", "end_date", "trip_purpose"]:
        value = data.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    missing = data.get("missing_info")
    if missing:
        lines.append(f"- missing_info: {missing}")
    return lines


def _render_preference(data: Dict[str, Any]) -> List[str]:
    lines = ["[preference]"]
    prefs = data.get("preferences")
    if isinstance(prefs, list) and prefs:
        for item in prefs:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('type')}: {item.get('value')} ({item.get('action', 'replace')})"
            )
    else:
        lines.append("- no preference extracted")
    return lines


def _render_memory_query(data: Dict[str, Any]) -> List[str]:
    lines = ["[memory_query]"]

    answer = data.get("answer") or data.get("result") or data.get("content") or data.get("message")
    if not answer and isinstance(data.get("data"), dict):
        nested = data["data"]
        answer = nested.get("answer") or nested.get("result") or nested.get("content") or nested.get("message")

    if answer:
        lines.append(f"- answer: {answer}")
    else:
        lines.append("- no memory answer")

    return lines




def _render_information_query(data: Dict[str, Any]) -> List[str]:
    lines = ["[information_query]"]
    payload = data.get("results") if isinstance(data.get("results"), dict) else data
    summary = ""
    if isinstance(payload, dict):
        summary = payload.get("summary") or payload.get("message") or payload.get("error") or ""
    if summary:
        lines.append(f"- summary: {summary}")
    else:
        lines.append("- no information summary")
    return lines


def _render_rag_knowledge(data: Dict[str, Any]) -> List[str]:
    lines = ["[rag_knowledge]"]
    answer = data.get("answer") or data.get("message") or data.get("content")
    if answer:
        lines.append(f"- answer: {answer}")
    else:
        lines.append("- no rag answer")
    return lines


def _render_itinerary_planning(data: Dict[str, Any]) -> List[str]:
    lines = ["[itinerary_planning]"]
    itinerary = data.get("itinerary", {}) if isinstance(data.get("itinerary"), dict) else {}
    title = itinerary.get("title")
    duration = itinerary.get("duration")
    if title:
        lines.append(f"- title: {title}")
    if duration:
        lines.append(f"- duration: {duration}")

    source_labels = itinerary.get("source_labels") if isinstance(itinerary.get("source_labels"), dict) else {}
    source_overall = source_labels.get("overall", "unknown")
    tool_verified = bool(source_labels.get("tool_verified", False))
    lines.append(f"- source_overall: {source_overall}")
    lines.append(f"- tool_verified: {str(tool_verified).lower()}")

    transport_plan = itinerary.get("transport_plan") if isinstance(itinerary.get("transport_plan"), dict) else {}
    if transport_plan:
        mode = transport_plan.get("mode")
        ticket = transport_plan.get("ticket_suggestion")
        seat = transport_plan.get("seat_suggestion")
        if mode:
            lines.append(f"- transport_mode: {mode}")
        if ticket:
            lines.append(f"- ticket_suggestion: {ticket}")
        if seat:
            lines.append(f"- seat_suggestion: {seat}")

    spots = itinerary.get("spot_suggestions")
    if isinstance(spots, list) and spots:
        for idx, spot in enumerate(spots, 1):
            lines.append(f"- spot_{idx}: {spot}")

    daily_plans = itinerary.get("daily_plans")
    if isinstance(daily_plans, list) and daily_plans:
        for day in daily_plans:
            if not isinstance(day, dict):
                continue
            day_no = day.get("day", "?")
            lines.append(f"- day_{day_no}:")
            activities = day.get("activities")
            if isinstance(activities, list):
                for act in activities:
                    if not isinstance(act, dict):
                        continue
                    t = act.get("time", "")
                    a = act.get("activity", "")
                    d = act.get("description", "")
                    lines.append(f"  {t} {a} - {d}".strip())

    notes = itinerary.get("notes")
    if isinstance(notes, list) and notes:
        for note in notes:
            if isinstance(note, dict):
                note_text = note.get("text") or note.get("value") or ""
                note_source = note.get("source", source_overall)
            else:
                note_text = str(note)
                note_source = source_overall
            lines.append(f"- note[{note_source}]: {note_text}")

    if len(lines) == 1:
        lines.append("- no itinerary details")
    return lines



def aggregate_node(state: GraphState) -> Dict[str, Any]:
    results: List[NodeResult] = state.get("agent_results", [])
    errors = list(state.get("errors", []))
    trace = list(state.get("execution_trace", []))

    if not results:
        final_response = "No sub-agent was triggered."
    else:
        lines: List[str] = ["LangGraph phase-1 result summary:"]
        for item in results:
            agent_name = item.get("agent_name", "unknown")
            status = item.get("status", "unknown")
            lines.append(f"- {agent_name}: {status}")
            data = item.get("data", {})
            if not isinstance(data, dict):
                continue

            if agent_name == "event_collection":
                lines.extend(_render_event_collection(data))
            elif agent_name == "preference":
                lines.extend(_render_preference(data))
            elif agent_name == "memory_query":
                lines.extend(_render_memory_query(data))
            elif agent_name == "information_query":
                lines.extend(_render_information_query(data))
            elif agent_name == "rag_knowledge":
                lines.extend(_render_rag_knowledge(data))
            elif agent_name == "itinerary_planning":
                lines.extend(_render_itinerary_planning(data))

        final_response = "\n".join(lines)

    trace.append("aggregate")
    return {
        "final_response": final_response,
        "errors": errors,
        "execution_trace": trace,
    }


def _merge_append_value(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, list):
        merged = list(existing)
        if incoming not in merged:
            merged.append(incoming)
        return merged
    if existing:
        if existing == incoming:
            return existing
        return [existing, incoming]
    return incoming


def _write_back_preference_results(state: GraphState, memory: MemoryPort, errors: List[str]) -> None:
    # MemoryManagerAdapter keeps a `memory_manager` attribute. For self-check fakes this branch is skipped.
    memory_manager = getattr(memory, "memory_manager", None)
    if memory_manager is None:
        return

    long_term = getattr(memory_manager, "long_term", None)
    if long_term is None or not hasattr(long_term, "save_preference"):
        return

    for item in state.get("agent_results", []):
        if item.get("agent_name") != "preference":
            continue
        if item.get("status") in {"error", "skipped", "not_implemented"}:
            continue

        data = item.get("data", {})
        if not isinstance(data, dict):
            continue

        preferences_data = data.get("preferences", [])
        pref_items: List[Dict[str, Any]] = []

        if isinstance(preferences_data, list):
            pref_items = [p for p in preferences_data if isinstance(p, dict)]
        elif isinstance(preferences_data, dict):
            for pref_type, pref_value in preferences_data.items():
                if pref_type in {"has_preferences", "error"}:
                    continue
                pref_items.append({"type": pref_type, "value": pref_value, "action": "replace"})

        for pref in pref_items:
            pref_type = pref.get("type")
            pref_value = pref.get("value")
            pref_action = pref.get("action", "replace")

            # Keep existing value when extraction is empty.
            if not pref_type or pref_value in (None, ""):
                continue

            try:
                if pref_action == "append":
                    current = {}
                    if hasattr(long_term, "get_preference"):
                        current = long_term.get_preference() or {}
                    merged_value = _merge_append_value(current.get(pref_type), pref_value)
                    long_term.save_preference(pref_type, merged_value)
                else:
                    long_term.save_preference(pref_type, pref_value)
            except Exception as exc:  # pragma: no cover - defensive branch
                errors.append(f"persist.preference_writeback_failed.{pref_type}: {exc}")




def _write_back_trip_results(state: GraphState, memory: MemoryPort, errors: List[str]) -> None:
    memory_manager = getattr(memory, "memory_manager", None)
    if memory_manager is None:
        return

    long_term = getattr(memory_manager, "long_term", None)
    if long_term is None or not hasattr(long_term, "save_trip_history"):
        return

    itinerary_success = False
    for item in state.get("agent_results", []):
        if item.get("agent_name") == "itinerary_planning" and item.get("status") == "success":
            itinerary = item.get("data", {}).get("itinerary") if isinstance(item.get("data"), dict) else None
            if itinerary:
                itinerary_success = True
                break
    if not itinerary_success:
        return

    event_data: Dict[str, Any] = {}
    for item in state.get("agent_results", []):
        if item.get("agent_name") == "event_collection" and isinstance(item.get("data"), dict):
            event_data = item.get("data", {})
            break

    destination = event_data.get("destination")
    if not destination:
        return

    trip_info = {
        "origin": event_data.get("origin"),
        "destination": destination,
        "start_date": event_data.get("start_date"),
        "end_date": event_data.get("end_date"),
        "purpose": event_data.get("trip_purpose", "travel"),
    }

    try:
        long_term.save_trip_history(trip_info)
    except Exception as exc:  # pragma: no cover - defensive branch
        errors.append(f"persist.trip_writeback_failed: {exc}")


def persist_node(state: GraphState, memory: MemoryPort) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    trace = list(state.get("execution_trace", []))

    try:
        _write_back_preference_results(state, memory, errors)
        _write_back_trip_results(state, memory, errors)

        user_input = state.get("user_input", "")
        if user_input:
            memory.add_message("user", user_input)

        assistant_payload = {
            "final_response": state.get("final_response", ""),
            "agent_results": state.get("agent_results", []),
        }
        memory.add_message("assistant", json.dumps(assistant_payload, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover - defensive branch
        errors.append(f"persist.failed: {exc}")

    trace.append("persist")
    return {"errors": errors, "execution_trace": trace}
