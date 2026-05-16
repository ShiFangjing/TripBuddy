"""Standalone ItineraryPlanning agent for LangGraph rewrite."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.agents._common import collect_model_text, extract_query, parse_json_object, parse_orchestrator_payload


class ItineraryPlanningAgent(AgentBase):
    def __init__(self, name: str = "itinerary_planning", model: Any = None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model

    def _fallback(self, query: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        destination = event_data.get("destination") or "\u76ee\u7684\u5730"
        origin = event_data.get("origin") or "\u51fa\u53d1\u5730"
        start_date = event_data.get("start_date") or "\u5f85\u5b9a\u65e5\u671f"
        is_one_day = any(k in query for k in ["\u660e\u5929", "\u4e00\u65e5", "\u4e00\u5929", "\u5f53\u65e5", "\u5f53\u5929"])

        if is_one_day:
            itinerary = {
                "title": f"{origin}\u81f3{destination}\u4e00\u65e5\u884c\u7a0b（\u793a\u4f8b）",
                "duration": "1\u65e5",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {"time": "07:30", "activity": "\u51fa\u53d1\u524d\u51c6\u5907", "description": "\u786e\u8ba4\u8bc1\u4ef6、\u884c\u674e\u4e0e\u76ee\u7684\u5730\u5929\u6c14"},
                            {"time": "08:30", "activity": "\u9ad8\u94c1\u524d\u5f80\u76ee\u7684\u5730", "description": f"\u4ece{origin}\u524d\u5f80{destination}，\u5efa\u8bae\u63d0\u524d30\u5206\u949f\u5230\u7ad9"},
                            {"time": "11:00", "activity": "\u666f\u70b91", "description": "\u57ce\u5e02\u5730\u6807/\u5386\u53f2\u6587\u5316\u666f\u70b9"},
                            {"time": "13:00", "activity": "\u666f\u70b92", "description": "\u5348\u9910\u540e\u524d\u5f80\u6838\u5fc3\u666f\u533a"},
                            {"time": "15:30", "activity": "\u666f\u70b93", "description": "\u4f11\u95f2\u6563\u6b65\u6216\u535a\u7269\u9986\u53c2\u89c2"},
                            {"time": "18:30", "activity": "\u8fd4\u7a0b", "description": f"\u8fd4\u56de{origin}"},
                        ],
                    }
                ],
                "transport_plan": {
                    "mode": "\u9ad8\u94c1",
                    "ticket_suggestion": "\u5efa\u8bae\u4f18\u5148\u9009\u62e9G\u5b57\u5934\u8f66\u6b21，\u53bb\u7a0b08:00-10:00\u65f6\u6bb5，\u8fd4\u7a0b18:00-20:00\u65f6\u6bb5",
                    "seat_suggestion": "\u5982\u6709\u504f\u597d，\u4f18\u5148\u9009\u62e9\u9760\u7a97\u5ea7\u4f4d；\u8d2d\u7968\u5e73\u53f0\u53ef\u8bbe\u7f6e\u5e2d\u4f4d\u504f\u597d",
                },
                "spot_suggestions": [
                    "\u666f\u70b91：\u57ce\u5e02\u5730\u6807（\u5efa\u8bae\u505c\u75591.5\u5c0f\u65f6）",
                    "\u666f\u70b92：\u6838\u5fc3\u666f\u533a（\u5efa\u8bae\u505c\u75592\u5c0f\u65f6）",
                    "\u666f\u70b93：\u6587\u5316\u4f11\u95f2\u70b9（\u5efa\u8bae\u505c\u75591\u5c0f\u65f6）",
                ],
                "notes": [
                    f"\u51fa\u53d1\u65e5\u671f\u53c2\u8003：{start_date}",
                    "\u8bf7\u572812306\u6216\u5e38\u7528\u8d2d\u7968\u5e73\u53f0\u786e\u8ba4\u5b9e\u65f6\u7968\u4ef7\u4e0e\u4f59\u7968",
                    "\u884c\u7a0b\u53ef\u6839\u636e\u5929\u6c14\u548c\u666f\u70b9\u6392\u961f\u60c5\u51b5\u5fae\u8c03",
                ],
            }
        else:
            itinerary = {
                "title": f"{origin}\u5230{destination}\u884c\u7a0b\u5efa\u8bae",
                "duration": "3\u59292\u665a",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {"time": "\u4e0a\u5348", "activity": "\u51fa\u53d1\u4e0e\u62b5\u8fbe", "description": f"\u4ece{origin}\u524d\u5f80{destination}"},
                            {"time": "\u4e0b\u5348", "activity": "\u5165\u4f4f\u4e0e\u5468\u8fb9\u719f\u6089", "description": "\u529e\u7406\u5165\u4f4f，\u4e86\u89e3\u5468\u8fb9\u4ea4\u901a"},
                        ],
                    },
                    {
                        "day": 2,
                        "activities": [
                            {"time": "\u5168\u5929", "activity": "\u6838\u5fc3\u884c\u7a0b", "description": "\u6839\u636e\u51fa\u5dee/\u6e38\u73a9\u76ee\u6807\u5b89\u6392\u6838\u5fc3\u6d3b\u52a8"},
                        ],
                    },
                    {
                        "day": 3,
                        "activities": [
                            {"time": "\u4e0a\u5348", "activity": "\u6536\u5c3e\u4e0e\u8fd4\u7a0b", "description": "\u9000\u623f\u5e76\u8fd4\u7a0b"},
                        ],
                    },
                ],
                "transport_plan": {
                    "mode": "\u9ad8\u94c1\u6216\u98de\u673a",
                    "ticket_suggestion": "\u53bb\u8fd4\u7a0b\u63d0\u524d3-7\u5929\u5173\u6ce8\u7968\u4ef7\u6ce2\u52a8",
                    "seat_suggestion": "\u53ef\u6309\u4e2a\u4eba\u504f\u597d\u9009\u62e9\u9760\u7a97/\u8fc7\u9053",
                },
                "notes": [f"\u51fa\u53d1\u65e5\u671f\u53c2\u8003：{start_date}", "\u8bf7\u7ed3\u5408\u5b9e\u65f6\u5929\u6c14\u4e0e\u4ea4\u901a\u518d\u786e\u8ba4"],
            }

        itinerary.setdefault("source_labels", {"overall": "fallback_template", "tool_verified": False})
        return {
            "itinerary": itinerary,
            "planning_complete": True,
        }

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content='{"status":"error","message":"no input"}', role="assistant")

        content = x.content if not isinstance(x, list) else x[-1].content
        _, context, previous_results = parse_orchestrator_payload(content)
        query = extract_query(context, fallback=str(content))

        event_data: Dict[str, Any] = {}
        pref_data: Dict[str, Any] = {}
        for item in previous_results:
            if not isinstance(item, dict):
                continue
            if item.get("agent_name") == "event_collection" and isinstance(item.get("data"), dict):
                event_data = item["data"]
            if item.get("agent_name") == "preference" and isinstance(item.get("data"), dict):
                pref_data = item["data"]

        if self.model is None:
            result = self._fallback(query, event_data)
            return Msg(name=self.name, content=__import__('json').dumps(result, ensure_ascii=False), role="assistant")

        prompt = f"""You are an expert travel itinerary planning assistant.
Generate the itinerary in the same language as the user input.

User request: {query}

Collected event data: {event_data}
Preference data: {pref_data}

Return a complete JSON object only. It must include these fields:
{{
  "itinerary": {{
    "title": "...",
    "duration": "...",
    "daily_plans": [
      {{
        "day": 1,
        "activities": [
          {{"time": "09:00", "activity": "...", "description": "..."}}
        ]
      }}
    ],
    "transport_plan": {{
      "mode": "rail/flight/metro/etc.",
      "ticket_suggestion": "ticket suggestion, including time window and booking advice",
      "seat_suggestion": "seat suggestion"
    }},
    "spot_suggestions": ["spot suggestion 1", "spot suggestion 2", "spot suggestion 3"],
    "notes": ["note 1", "note 2"]
  }},
  "planning_complete": true
}}
Output JSON only. Do not include explanatory text outside the JSON object."""

        try:
            text = await collect_model_text(self.model, [{"role": "user", "content": prompt}])
            parsed = parse_json_object(text)
            if "itinerary" not in parsed or not isinstance(parsed.get("itinerary"), dict):
                raise ValueError("missing itinerary")

            itinerary = parsed["itinerary"]
            if not isinstance(itinerary.get("daily_plans"), list):
                itinerary["daily_plans"] = []
            if "transport_plan" not in itinerary or not isinstance(itinerary.get("transport_plan"), dict):
                itinerary["transport_plan"] = {}
            if not isinstance(itinerary.get("spot_suggestions"), list):
                itinerary["spot_suggestions"] = []
            if not isinstance(itinerary.get("notes"), list):
                itinerary["notes"] = []

            itinerary.setdefault("source_labels", {"overall": "llm_generated", "tool_verified": False})
            parsed.setdefault("planning_complete", True)
            result = parsed
        except Exception:
            result = self._fallback(query, event_data)

        return Msg(name=self.name, content=__import__('json').dumps(result, ensure_ascii=False), role="assistant")
