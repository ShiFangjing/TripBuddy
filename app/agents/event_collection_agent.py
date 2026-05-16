"""Compatibility wrapper for the event-collection skill."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "skills" / "event-collection" / "script" / "agent.py"
_SPEC = importlib.util.spec_from_file_location("langgraph_event_collection_skill", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load event-collection skill from {_SCRIPT}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

EventCollectionAgent = _MODULE.EventCollectionAgent

