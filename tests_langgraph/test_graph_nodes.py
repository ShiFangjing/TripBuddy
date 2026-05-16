import asyncio
import json

from agentscope.message import Msg

from app.graph import GraphRuntime, ainvoke_once, build_graph, build_self_check_runtime
from app.nodes.aggregate import aggregate_node, persist_node
from app.nodes.route import route_node


def test_graph_runs_and_returns_two_agent_results():
    runtime = build_self_check_runtime()
    app = build_graph(runtime)
    result = asyncio.run(ainvoke_once(app, user_id="u", session_id="s", user_input="test"))

    agent_results = result.get("agent_results", [])
    assert len(agent_results) == 2
    names = {item.get("agent_name") for item in agent_results}
    assert names == {"event_collection", "preference"}
    assert result.get("final_response")


class _UnknownIntention:
    async def reply(self, x):
        payload = {
            "reasoning": "unknown-agent-test",
            "intents": [{"type": "test", "confidence": 1.0}],
            "key_entities": {},
            "rewritten_query": "test",
            "agent_schedule": [{"agent_name": "non_existing_agent", "priority": 1}],
        }
        return Msg(name="IntentionAgent", content=json.dumps(payload, ensure_ascii=False), role="assistant")


class _MemoryNoop:
    async def get_long_term_summary(self, user_input: str, max_messages: int = 50):
        return ""

    def get_recent_context(self, n_turns: int = 5):
        return []

    def get_preferences(self):
        return {}

    def add_message(self, role: str, content: str, metadata=None):
        return None


def test_unknown_agent_returns_structured_error():
    runtime = GraphRuntime(
        model=None,
        memory_manager=None,
        memory_adapter=_MemoryNoop(),
        intention_agent=_UnknownIntention(),
        agent_registry={},
    )
    app = build_graph(runtime)
    result = asyncio.run(ainvoke_once(app, user_id="u", session_id="s", user_input="trigger"))

    assert result.get("agent_results"), result
    first = result["agent_results"][0]
    assert first.get("status") == "error"
    assert first.get("agent_name") == "non_existing_agent"
    assert "unknown agent_name" in (first.get("message") or "")


def test_route_keeps_migrated_schedule_entries():
    routed = route_node(
        {
            "user_input": "\u5e2e\u6211\u67e5\u5317\u4eac\u5929\u6c14",
            "agent_schedule": [
                {"agent_name": "information_query", "priority": 1},
            ],
            "errors": [],
            "execution_trace": [],
        }
    )

    names = [item.get("agent_name") for item in routed.get("agent_schedule", [])]
    assert names == ["information_query"]




def test_route_skips_disabled_rag_knowledge_schedule_entry():
    routed = route_node(
        {
            "user_input": "\u5dee\u65c5\u6807\u51c6\u662f\u591a\u5c11",
            "agent_schedule": [
                {"agent_name": "rag_knowledge", "priority": 1},
            ],
            "errors": [],
            "execution_trace": [],
        }
    )

    names = [item.get("agent_name") for item in routed.get("agent_schedule", [])]
    assert names == []
    assert "route.skipped_disabled_agent: rag_knowledge" in routed.get("errors", [])


def test_route_keeps_unknown_agent_for_structured_error_path():
    routed = route_node(
        {
            "user_input": "trigger",
            "agent_schedule": [
                {"agent_name": "non_existing_agent", "priority": 1},
            ],
            "errors": [],
            "execution_trace": [],
        }
    )
    names = [item.get("agent_name") for item in routed.get("agent_schedule", [])]
    assert names == ["non_existing_agent"]


def test_route_fallback_uses_memory_query_for_memory_questions():
    routed = route_node(
        {
            "user_input": "\u6211\u4f4f\u5728\u54ea\u91cc？",
            "agent_schedule": [],
            "errors": [],
            "execution_trace": [],
        }
    )
    names = [item.get("agent_name") for item in routed.get("agent_schedule", [])]
    assert names == ["memory_query"]


class _FakeLongTerm:
    def __init__(self):
        self.prefs = {"airlines": ["\u56fd\u822a"]}

    def save_preference(self, pref_type, value):
        self.prefs[pref_type] = value

    def get_preference(self):
        return dict(self.prefs)


class _FakeMemoryManager:
    def __init__(self):
        self.long_term = _FakeLongTerm()


class _FakeMemoryAdapter:
    def __init__(self):
        self.memory_manager = _FakeMemoryManager()
        self.messages = []

    def add_message(self, role, content, metadata=None):
        self.messages.append({"role": role, "content": content, "metadata": metadata or {}})


def test_persist_writes_back_preference_and_skips_empty():
    memory = _FakeMemoryAdapter()
    state = {
        "user_input": "\u6211\u4f4f\u5728\u5e7f\u5dde，\u4e5f\u5e38\u5750\u4e1c\u822a",
        "final_response": "ok",
        "agent_results": [
            {
                "status": "success",
                "agent_name": "preference",
                "data": {
                    "preferences": [
                        {"type": "home_location", "value": "\u5e7f\u5dde\u5e02\u5929\u6cb3\u533a", "action": "replace"},
                        {"type": "airlines", "value": "\u4e1c\u822a", "action": "append"},
                        {"type": "seat_preference", "value": "", "action": "replace"},
                    ]
                },
            }
        ],
        "errors": [],
        "execution_trace": [],
    }

    out = persist_node(state, memory)
    assert out.get("errors") == []
    assert len(memory.messages) == 2

    prefs = memory.memory_manager.long_term.get_preference()
    assert prefs.get("home_location") == "\u5e7f\u5dde\u5e02\u5929\u6cb3\u533a"
    assert prefs.get("airlines") == ["\u56fd\u822a", "\u4e1c\u822a"]
    assert "seat_preference" not in prefs


def test_aggregate_renders_memory_query_answer():
    state = {
        "agent_results": [
            {
                "status": "success",
                "agent_name": "memory_query",
                "data": {"answer": "\u4f60\u5e38\u4f4f\u5728\u5e7f\u5dde\u5e02\u5929\u6cb3\u533a。"},
            }
        ],
        "errors": [],
        "execution_trace": [],
    }
    out = aggregate_node(state)
    assert "\u4f60\u5e38\u4f4f\u5728\u5e7f\u5dde\u5e02\u5929\u6cb3\u533a" in out.get("final_response", "")



def test_persist_writes_trip_history_when_itinerary_exists():
    class _TripLongTerm(_FakeLongTerm):
        def __init__(self):
            super().__init__()
            self.trip_calls = []

        def save_trip_history(self, trip_info):
            self.trip_calls.append(trip_info)

    class _TripMemoryManager(_FakeMemoryManager):
        def __init__(self):
            self.long_term = _TripLongTerm()

    class _TripMemoryAdapter(_FakeMemoryAdapter):
        def __init__(self):
            self.memory_manager = _TripMemoryManager()
            self.messages = []

    memory = _TripMemoryAdapter()
    state = {
        "user_input": "\u5e2e\u6211\u89c4\u5212\u53bb\u5317\u4eac\u51fa\u5dee",
        "final_response": "ok",
        "agent_results": [
            {
                "status": "success",
                "agent_name": "event_collection",
                "data": {
                    "origin": "\u4e0a\u6d77",
                    "destination": "\u5317\u4eac",
                    "start_date": "2026-05-01",
                    "end_date": "2026-05-03",
                    "trip_purpose": "\u51fa\u5dee",
                },
            },
            {
                "status": "success",
                "agent_name": "itinerary_planning",
                "data": {"itinerary": {"title": "\u5317\u4eac\u51fa\u5dee\u8ba1\u5212"}},
            },
        ],
        "errors": [],
        "execution_trace": [],
    }

    persist_node(state, memory)
    calls = memory.memory_manager.long_term.trip_calls
    assert len(calls) == 1
    assert calls[0]["destination"] == "\u5317\u4eac"


def test_aggregate_renders_itinerary_details():
    state = {
        "agent_results": [
            {
                "status": "success",
                "agent_name": "itinerary_planning",
                "data": {
                    "itinerary": {
                        "title": "\u4e0a\u6d77\u4e00\u65e5\u884c\u7a0b",
                        "duration": "1\u65e5",
                        "transport_plan": {
                            "mode": "\u9ad8\u94c1",
                            "ticket_suggestion": "\u5efa\u8bae\u9009\u62e9G\u5b57\u5934",
                            "seat_suggestion": "\u4f18\u5148\u9760\u7a97",
                        },
                        "spot_suggestions": ["\u5916\u6ee9", "\u8c6b\u56ed", "\u6b66\u5eb7\u8def"],
                        "daily_plans": [
                            {
                                "day": 1,
                                "activities": [
                                    {"time": "09:00", "activity": "\u5916\u6ee9", "description": "\u89c2\u666f"}
                                ],
                            }
                        ],
                        "source_labels": {"overall": "llm_generated", "tool_verified": False},
                        "notes": ["\u63d0\u524d\u8d2d\u7968"],
                    }
                },
            }
        ],
        "errors": [],
        "execution_trace": [],
    }

    out = aggregate_node(state)
    summary = out.get("final_response", "")
    assert "ticket_suggestion" in summary
    assert "spot_1" in summary
    assert "day_1" in summary
    assert "source_overall: llm_generated" in summary
    assert "tool_verified: false" in summary
    assert "note[llm_generated]" in summary
