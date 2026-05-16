"""Explicit local registry for LangGraph rewrite (standalone, no .claude dependency)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.skills.registry import SkillRegistry


class PlaceholderAgent(AgentBase):
    """Explicit placeholder for disabled or future agents."""

    def __init__(self, name: str, reason: str):
        super().__init__()
        self.name = name
        self.reason = reason

    async def reply(self, x=None) -> Msg:
        payload = {
            "status": "not_implemented",
            "agent_name": self.name,
            "error": self.reason,
        }
        return Msg(name=self.name, content=json.dumps(payload, ensure_ascii=False), role="assistant")


def build_explicit_registry(model: Any, memory_manager: Any, project_root=None) -> Dict[str, AgentBase]:
    """Create all local agent instances at startup (standalone registration)."""
    base_root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
    skill_registry = SkillRegistry(project_root=base_root, model=model, memory_manager=memory_manager)

    registry: Dict[str, AgentBase] = skill_registry.build(
        [
            "event_collection",
            "preference",
            "memory_query",
            "information_query",
            "itinerary_planning",
        ]
    )
    registry.update(
        {
        # Future extension slot.
        "future_agent_placeholder": PlaceholderAgent("future_agent_placeholder", "phase-3 placeholder"),
        }
    )
    return registry
