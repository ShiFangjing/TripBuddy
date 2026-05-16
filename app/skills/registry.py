"""Explicit skill registry for the LangGraph rewrite."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

from agentscope.agent import AgentBase


class SkillRegistry:
    """Load local skills from TripBuddy2/skills without depending on .claude."""

    LEGACY_MAPPING = {
        "event_collection": "event-collection",
        "preference": "preference",
        "memory_query": "memory-query",
        "information_query": "query-info",
        "rag_knowledge": "ask-question",
        "itinerary_planning": "plan-trip",
    }

    def __init__(self, project_root: Path, model: Any, memory_manager: Any = None):
        self.project_root = Path(project_root).resolve()
        self.skills_root = self.project_root / "TripBuddy2" / "skills"
        self.model = model
        self.memory_manager = memory_manager
        self.cache: Dict[str, AgentBase] = {}

    def _resolve_skill_name(self, agent_name: str) -> Optional[str]:
        skill_name = self.LEGACY_MAPPING.get(agent_name, agent_name)
        if (self.skills_root / skill_name / "script" / "agent.py").exists():
            return skill_name
        return None

    def _load_agent_class(self, agent_name: str, skill_name: str) -> Type[AgentBase]:
        script_path = self.skills_root / skill_name / "script" / "agent.py"
        module_name = f"TripBuddy2.skills.{skill_name.replace('-', '_')}.agent"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill script: {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, AgentBase) and obj is not AgentBase:
                return obj
        raise ValueError(f"No AgentBase subclass found for {agent_name} in {script_path}")

    def get(self, agent_name: str) -> Optional[AgentBase]:
        if agent_name in self.cache:
            return self.cache[agent_name]

        skill_name = self._resolve_skill_name(agent_name)
        if skill_name is None:
            return None

        agent_class = self._load_agent_class(agent_name, skill_name)
        kwargs = {
            "name": agent_name,
            "model": self.model,
        }

        signature = inspect.signature(agent_class.__init__)
        if "memory_manager" in signature.parameters:
            kwargs["memory_manager"] = self.memory_manager
        if "project_root" in signature.parameters:
            kwargs["project_root"] = self.project_root
        if "knowledge_base_path" in signature.parameters:
            kwargs["knowledge_base_path"] = self.project_root / "TripBuddy2" / "data" / "rag_knowledge"
        if "collection_name" in signature.parameters:
            kwargs["collection_name"] = "business_travel_knowledge"

        instance = agent_class(**kwargs)
        self.cache[agent_name] = instance
        return instance

    def build(self, agent_names: list[str]) -> Dict[str, AgentBase]:
        registry: Dict[str, AgentBase] = {}
        for name in agent_names:
            agent = self.get(name)
            if agent is not None:
                registry[name] = agent
        return registry

