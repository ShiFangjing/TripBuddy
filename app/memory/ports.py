"""Memory ports for graph runtime."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class MemoryPort(Protocol):
    async def get_long_term_summary(self, user_input: str, max_messages: int = 50) -> str:
        ...

    def get_recent_context(self, n_turns: int = 5) -> List[Dict[str, Any]]:
        ...

    def get_preferences(self) -> Dict[str, Any]:
        ...

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        ...
