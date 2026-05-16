"""Standalone Preference agent for LangGraph rewrite."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.agents._common import collect_model_text, extract_query, parse_json_object, parse_orchestrator_payload


class PreferenceAgent(AgentBase):
    def __init__(self, name: str = "preference", model: Any = None, memory_manager: Any = None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager

    def _action(self, query: str) -> str:
        if any(k in query for k in ["\u8fd8", "\u4e5f", "\u53e6\u5916", "\u4ee5\u53ca"]):
            return "append"
        if any(k in query for k in ["\u642c\u5bb6", "\u6539\u6210", "\u6362\u6210", "\u73b0\u5728"]):
            return "replace"
        return "replace"

    def _heuristic(self, query: str) -> Dict[str, Any]:
        action = self._action(query)
        preferences: List[Dict[str, Any]] = []

        home = ""
        m = re.search("(?:\u4f4f\u5728|\u5bb6\u5728|\u642c\u5bb6\u5230)([^，。\\s]+)", query)
        if m:
            home = m.group(1).strip()
        if home:
            preferences.append({"type": "home_location", "value": home, "action": "replace" if "\u642c\u5bb6" in query else action})

        airline_names = ["\u56fd\u822a", "\u4e1c\u822a", "\u5357\u822a", "\u6d77\u822a", "\u5ddd\u822a", "\u6df1\u822a"]
        for a in airline_names:
            if a in query:
                preferences.append({"type": "airlines", "value": a, "action": action})

        hotel_names = ["\u4e07\u8c6a", "\u5e0c\u5c14\u987f", "\u5982\u5bb6", "\u6c49\u5ead", "\u4e9a\u6735", "\u5168\u5b63"]
        for h in hotel_names:
            if h in query:
                preferences.append({"type": "hotel_brands", "value": h, "action": action})

        if "\u9760\u7a97" in query:
            preferences.append({"type": "seat_preference", "value": "\u9760\u7a97", "action": "replace"})
        if "\u9760\u8fc7\u9053" in query or "\u8fc7\u9053" in query:
            preferences.append({"type": "seat_preference", "value": "\u9760\u8fc7\u9053", "action": "replace"})

        # de-duplicate same (type,value,action)
        dedup = []
        seen = set()
        for p in preferences:
            key = (p.get("type"), p.get("value"), p.get("action"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)

        return {"preferences": dedup, "has_preferences": bool(dedup)}

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content="{}", role="assistant")

        content = x.content if not isinstance(x, list) else x[-1].content
        _, context, _ = parse_orchestrator_payload(content)
        query = extract_query(context, fallback=str(content))

        current_preferences = {}
        if self.memory_manager is not None:
            try:
                current_preferences = self.memory_manager.long_term.get_preference()
            except Exception:
                current_preferences = {}

        prompt = f"""You are a preference extraction assistant.
Extract durable travel preferences from the user input.

Current known preferences: {current_preferences}
User input: {query}

Return a JSON object only:
{{"preferences":[{{"type":"...","value":"...","action":"append|replace"}}],"has_preferences":true/false}}
Keep extracted preference values in the original language used by the user."""

        result: Dict[str, Any]
        try:
            text = await collect_model_text(self.model, [{"role": "user", "content": prompt}])
            parsed = parse_json_object(text)
            prefs = parsed.get("preferences", [])
            if not isinstance(prefs, list):
                prefs = []
            parsed["preferences"] = [p for p in prefs if isinstance(p, dict)]
            parsed["has_preferences"] = bool(parsed["preferences"])
            result = parsed
        except Exception:
            result = self._heuristic(query)

        return Msg(name=self.name, content=__import__('json').dumps(result, ensure_ascii=False), role="assistant")
