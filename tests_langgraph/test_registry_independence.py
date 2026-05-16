from pathlib import Path

from app.nodes.registry import build_explicit_registry


def test_registry_is_local_and_rag_disabled_in_flow():
    registry = build_explicit_registry(model=None, memory_manager=None, project_root=Path('.'))

    assert "TripBuddy2.skills.event_collection" in registry["event_collection"].__class__.__module__
    assert "TripBuddy2.skills.preference" in registry["preference"].__class__.__module__
    assert "TripBuddy2.skills.memory_query" in registry["memory_query"].__class__.__module__
    assert "TripBuddy2.skills.query_info" in registry["information_query"].__class__.__module__
    assert "TripBuddy2.skills.plan_trip" in registry["itinerary_planning"].__class__.__module__
    assert "rag_knowledge" not in registry
