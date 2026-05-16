"""Adapters to reuse existing memory manager semantics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from context.memory_manager import MemoryManager


class MemoryManagerAdapter:
    """Adapter around existing MemoryManager to keep phase-1 semantics."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    async def get_long_term_summary(self, user_input: str, max_messages: int = 50) -> str:
        # Keep behavior aligned with current implementation
        return await self.memory_manager.get_long_term_summary_async(max_messages=max_messages)

    def get_recent_context(self, n_turns: int = 5) -> List[Dict[str, Any]]:
        return self.memory_manager.short_term.get_recent_context(n_turns=n_turns)

    def get_preferences(self) -> Dict[str, Any]:
        return self.memory_manager.long_term.get_preference()

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.memory_manager.add_message(role, content, metadata)
