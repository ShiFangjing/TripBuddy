"""Standalone MemoryQuery agent for LangGraph rewrite."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.agents._common import collect_model_text, extract_query, parse_orchestrator_payload


class MemoryQueryAgent(AgentBase):
    def __init__(self, name: str = "memory_query", model: Any = None, memory_manager: Any = None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager

    def _format_preferences(self, preferences: Dict[str, Any]) -> str:
        if not preferences:
            return "（\u6682\u65e0\u504f\u597d\u8bb0\u5f55）"
        lines: List[str] = []
        for k, v in preferences.items():
            if not v:
                continue
            if isinstance(v, list):
                value = "、".join(str(x) for x in v)
            else:
                value = str(v)
            lines.append(f"- {k}: {value}")
        return "\n".join(lines) if lines else "（\u6682\u65e0\u504f\u597d\u8bb0\u5f55）"

    def _direct_answer(self, query: str, preferences: Dict[str, Any], trips: List[Dict[str, Any]]) -> str:
        q = query.strip()
        home = preferences.get("home_location")
        if any(k in q for k in ["\u6211\u4f4f\u5728\u54ea\u91cc", "\u6211\u4f4f\u5728\u54ea", "\u6211\u5bb6\u4f4f\u5728\u54ea", "\u4f4f\u5740"]) and home:
            return f"\u4f60\u5e38\u4f4f\u5728{home}。"

        if any(k in q for k in ["\u504f\u597d", "\u559c\u6b22\u4ec0\u4e48"]) and preferences:
            return "\u4f60\u5f53\u524d\u8bb0\u5f55\u7684\u504f\u597d\u5982\u4e0b：\n" + self._format_preferences(preferences)

        if any(k in q for k in ["\u53bb\u8fc7", "\u53bb\u8fc7\u54ea\u4e9b", "\u5230\u8fc7"]) and trips:
            parts = []
            for t in trips[:5]:
                origin = t.get("origin") or "\u672a\u77e5"
                destination = t.get("destination") or "\u672a\u77e5"
                start_date = t.get("start_date") or ""
                if start_date:
                    parts.append(f"{origin}→{destination}（{start_date}）")
                else:
                    parts.append(f"{origin}→{destination}")
            return "\u4f60\u7684\u5386\u53f2\u884c\u7a0b\u5305\u62ec：" + "；".join(parts)

        return ""

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content='{"status":"error","message":"no input"}', role="assistant")

        content = x.content if not isinstance(x, list) else x[-1].content
        _, context, _ = parse_orchestrator_payload(content)
        query = extract_query(context, fallback=str(content))

        preferences: Dict[str, Any] = {}
        trips: List[Dict[str, Any]] = []
        summary = ""

        if self.memory_manager is not None:
            try:
                preferences = self.memory_manager.long_term.get_preference() or {}
            except Exception:
                preferences = {}
            try:
                trips = self.memory_manager.long_term.get_trip_history(limit=50) or []
            except Exception:
                trips = []
            try:
                summary = await self.memory_manager.get_long_term_summary_async(max_messages=30)
            except Exception:
                summary = ""

        direct = self._direct_answer(query, preferences, trips)
        if direct:
            result = {
                "status": "success",
                "query": query,
                "answer": direct,
                "memory_sources": {
                    "trip_count": len(trips),
                    "has_preferences": bool(preferences),
                    "has_chat_summary": bool(summary),
                },
            }
            return Msg(name=self.name, content=__import__('json').dumps(result, ensure_ascii=False), role="assistant")

        prompt = f"""You are a personal memory assistant.
Answer the user question using only the provided memory.
Respond in the same language as the user input.

User question: {query}

User preferences:
{self._format_preferences(preferences)}

Trip history: {trips[:10]}

Historical summary: {summary or 'none'}

If the provided memory is insufficient, clearly say that no relevant record is available in memory."""

        try:
            answer = await collect_model_text(self.model, [
                {"role": "system", "content": "You are a personal memory assistant. Respond in the same language as the user."},
                {"role": "user", "content": prompt},
            ])
            if not answer:
                answer = "No relevant record is available in memory."
            result = {
                "status": "success",
                "query": query,
                "answer": answer,
                "memory_sources": {
                    "trip_count": len(trips),
                    "has_preferences": bool(preferences),
                    "has_chat_summary": bool(summary),
                },
            }
        except Exception as exc:
            result = {"status": "error", "query": query, "message": f"Memory query failed: {exc}"}

        return Msg(name=self.name, content=__import__('json').dumps(result, ensure_ascii=False), role="assistant")
