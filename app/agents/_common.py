"""Shared helpers for local standalone agents."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


def parse_orchestrator_payload(content: Any) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Parse orchestrator input payload into (root, context, previous_results)."""
    if isinstance(content, dict):
        payload = content
    elif isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {"context": {"rewritten_query": content}, "previous_results": []}
    else:
        payload = {"context": {"rewritten_query": str(content)}, "previous_results": []}

    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    previous_results = payload.get("previous_results") if isinstance(payload.get("previous_results"), list) else []
    return payload, context, previous_results


def extract_query(context: Dict[str, Any], fallback: str = "") -> str:
    q = context.get("rewritten_query", "")
    if isinstance(q, str) and q.strip():
        return q.strip()

    recent = context.get("recent_dialogue", [])
    if isinstance(recent, list):
        for msg in reversed(recent):
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"].strip()

    return fallback.strip()


async def collect_model_text(model: Any, messages: List[Dict[str, str]]) -> str:
    if model is None:
        return ""

    response = await model(messages)

    text = ""
    if hasattr(response, "__aiter__"):
        async for chunk in response:
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, "content"):
                content = chunk.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
    elif hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "content"):
        text = response.content
    elif isinstance(response, dict) and "content" in response:
        text = response["content"]
    elif response is not None:
        text = str(response)

    return str(text or "").strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty model output")

    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object start")

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(s[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not object")
    return obj


def maybe_json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def normalize_list_value(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value is None:
        return []
    s = str(value).strip()
    return [s] if s else []


def first_match(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""
