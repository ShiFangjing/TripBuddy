"""Standalone EventCollection agent for LangGraph rewrite."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.agents._common import collect_model_text, extract_query, parse_json_object, parse_orchestrator_payload


class EventCollectionAgent(AgentBase):
    def __init__(self, name: str = "event_collection", model: Any = None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model

    def _heuristic(self, query: str) -> Dict[str, Any]:
        origin = ""
        destination = ""
        start_date = ""
        end_date = ""
        trip_purpose = "\u51fa\u5dee" if "\u51fa\u5dee" in query or "\u5546\u52a1" in query else ("\u65c5\u6e38" if "\u65c5\u6e38" in query or "\u73a9" in query else "")

        m = re.search("\u4ece([^，。\\s]+?)(?:\u5230|\u53bb)([^，。\\s]+)", query)
        if m:
            origin, destination = m.group(1).strip(), m.group(2).strip()
        else:
            m2 = re.search("\u53bb([^，。\\s]+)", query)
            if m2:
                destination = m2.group(1).strip()

        date_tokens = re.findall("(\\d{4}-\\d{1,2}-\\d{1,2}|\\d{1,2}\u6708\\d{1,2}\u65e5|\u660e\u5929|\u540e\u5929|\u4e0b\u5468[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u65e5\u5929]?|\u5468[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u65e5\u5929])", query)
        if date_tokens:
            start_date = date_tokens[0]
            if len(date_tokens) > 1:
                end_date = date_tokens[1]

        missing: List[str] = []
        if not origin:
            missing.append("origin")
        if not destination:
            missing.append("destination")
        if not start_date:
            missing.append("start_date")

        return {
            "origin": origin,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "trip_purpose": trip_purpose,
            "missing_info": missing,
            "extracted_count": int(bool(origin)) + int(bool(destination)) + int(bool(start_date)) + int(bool(end_date)),
        }

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content="{}", role="assistant")

        content = x.content if not isinstance(x, list) else x[-1].content
        _, context, _ = parse_orchestrator_payload(content)
        query = extract_query(context, fallback=str(content))

        prompt = f"""You are a travel event extraction assistant.
Extract structured trip fields from the user input.

User input: {query}

Return a JSON object only. The JSON object must include:
origin, destination, start_date, end_date, trip_purpose, missing_info (array), extracted_count (integer).
If a value cannot be determined, use an empty string. Put missing field names in missing_info.
Keep extracted entity values in the original language used by the user."""

        result: Dict[str, Any]
        try:
            text = await collect_model_text(self.model, [{"role": "user", "content": prompt}])
            parsed = parse_json_object(text)
            if not isinstance(parsed.get("missing_info", []), list):
                parsed["missing_info"] = []
            if "extracted_count" not in parsed:
                parsed["extracted_count"] = 0
            result = parsed
        except Exception:
            result = self._heuristic(query)

        return Msg(name=self.name, content=__import__('json').dumps(result, ensure_ascii=False), role="assistant")
