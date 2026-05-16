"""Compatibility wrapper for the query-info skill."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "skills" / "query-info" / "script" / "agent.py"
_SPEC = importlib.util.spec_from_file_location("langgraph_query_info_skill", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load query-info skill from {_SCRIPT}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

InformationQueryAgent = _MODULE.InformationQueryAgent

