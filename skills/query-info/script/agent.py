"""Standalone InformationQuery agent for LangGraph rewrite with real tools."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.agents._common import collect_model_text, extract_query, parse_orchestrator_payload

logger = logging.getLogger(__name__)

try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS = None  # type: ignore

_SUSPICIOUS_DOMAIN_PATTERN = re.compile(
    r"\.(cc|tk|ml|ga|cf|gq|xyz|top|work|click|link|pw|buzz)(/|$)",
    re.I,
)
_RANDOM_DOMAIN_PATTERN = re.compile(r"^[a-z0-9]{10,}$", re.I)


def _is_suspicious_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return True
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).netloc or "").split(":")[0].lower()
        if not host:
            return True
        if _SUSPICIOUS_DOMAIN_PATTERN.search(host):
            return True
        name = host.rsplit(".", 2)[0] if "." in host else host
        if len(name) >= 10 and _RANDOM_DOMAIN_PATTERN.match(name):
            return True
        return False
    except Exception:
        return False


class InformationQueryAgent(AgentBase):
    def __init__(self, name: str = "information_query", model: Any = None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content='{"status":"error","message":"no input"}', role="assistant")

        content = x.content if not isinstance(x, list) else x[-1].content
        _, context, _ = parse_orchestrator_payload(content)
        query = extract_query(context, fallback=str(content))

        result: Dict[str, Any]
        if self._is_weather_query(query):
            try:
                result = await self._weather_query(query)
            except Exception as exc:
                logger.warning("weather query failed, fallback to web search: %s", exc)
                result = await self._web_search(query)
        else:
            result = await self._web_search(query)

        result.setdefault("status", "success")
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    def _is_weather_query(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        return any(token in q for token in ["\u5929\u6c14", "\u6c14\u6e29", "\u4e0b\u96e8", "\u9884\u62a5", "\u6e29\u5ea6", "\u6e7f\u5ea6"])

    def _extract_city_from_query(self, query: str) -> str:
        common_cities = [
            "\u5317\u4eac", "\u4e0a\u6d77", "\u5e7f\u5dde", "\u6df1\u5733", "\u676d\u5dde", "\u5357\u4eac", "\u6210\u90fd", "\u6b66\u6c49", "\u897f\u5b89", "\u82cf\u5dde",
            "\u5929\u6d25", "\u91cd\u5e86", "\u53a6\u95e8", "\u9752\u5c9b", "\u5927\u8fde", "\u5b81\u6ce2", "\u65e0\u9521", "\u957f\u6c99", "\u90d1\u5dde", "\u6d4e\u5357",
            "\u54c8\u5c14\u6ee8", "\u6c88\u9633", "\u6606\u660e", "\u5408\u80a5", "\u798f\u5dde", "\u77f3\u5bb6\u5e84", "\u5357\u660c", "\u8d35\u9633", "\u592a\u539f", "\u5357\u5b81",
        ]
        q = (query or "").strip()
        for city in common_cities:
            if city in q:
                return city
        match = re.search(r"[\u4e00-\u9fa5]{2,6}", q)
        return match.group(0).strip() if match else ""

    async def _weather_query(self, query: str) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError:
            return {
                "status": "success",
                "query_type": "weather",
                "query_success": False,
                "results": {"message": "httpx is required to run weather queries"},
            }

        city = self._extract_city_from_query(query)
        if not city:
            return {
                "status": "success",
                "query_type": "weather",
                "query_success": False,
                "results": {"message": "No city was detected. Please specify a city."},
            }

        url = f"https://wttr.in/{city}?format=j1"
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: httpx.get(url, timeout=10.0, headers={"User-Agent": "curl/7.64.1"}),
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return {
                "status": "success",
                "query_type": "weather",
                "query_success": False,
                "results": {
                    "message": f"The weather API is temporarily unavailable: {exc}",
                    "sources": [{"url": "https://wttr.in", "title": "wttr.in"}],
                },
            }

        try:
            current = data.get("current_condition", [{}])[0]
            temp_c = current.get("temp_C", "?")
            humidity = current.get("humidity", "?")
            desc_list = current.get("weatherDesc", [{}])
            desc = (desc_list[0].get("value") if desc_list else None) or "—"

            weather_text = f"Current weather in {city}: {desc}, temperature {temp_c}°C, humidity {humidity}%."
            forecasts: List[str] = []
            for day in data.get("weather", [])[:3]:
                date = day.get("date", "")
                maxtemp = day.get("maxtempC", "?")
                mintemp = day.get("mintempC", "?")
                hourly = (day.get("hourly") or [{}])[0]
                hourly_desc = (hourly.get("weatherDesc") or [{}])[0].get("value", "—")
                forecasts.append(f"{date}: {hourly_desc}, {mintemp}~{maxtemp}°C")
            if forecasts:
                weather_text += " Next few days: " + "; ".join(forecasts)

            return {
                "status": "success",
                "query_type": "weather",
                "query_success": True,
                "results": {
                    "summary": weather_text,
                    "sources": [{"url": "https://wttr.in", "title": "wttr.in"}],
                },
            }
        except Exception:
            return {
                "status": "success",
                "query_type": "weather",
                "query_success": False,
                "results": {
                    "message": "Failed to parse weather data",
                    "sources": [{"url": "https://wttr.in", "title": "wttr.in"}],
                },
            }

    async def _web_search(self, query: str) -> Dict[str, Any]:
        if not DDGS_AVAILABLE:
            return {
                "status": "success",
                "query_type": "web_search",
                "query_success": False,
                "results": {"message": "The search package is not installed. Please install ddgs."},
            }

        try:
            ddgs = DDGS()
            search_results: List[Dict[str, Any]] = []
            for backend in ("bing", "duckduckgo", "auto"):
                try:
                    raw = ddgs.text(
                        query,
                        max_results=10,
                        safesearch="on",
                        region="cn-zh",
                        backend=backend,
                    )
                    search_results = list(raw)
                    if search_results:
                        break
                except Exception as exc:
                    logger.debug("DDGS backend %s failed: %s", backend, exc)

            results: List[Dict[str, Any]] = []
            for result in search_results:
                href = result.get("href", "")
                if _is_suspicious_url(href):
                    continue
                results.append(
                    {
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": href,
                    }
                )
                if len(results) >= 5:
                    break

            if not results:
                return {
                    "status": "success",
                    "query_type": "web_search",
                    "query_success": False,
                    "results": {"message": "No relevant results were found"},
                }

            summary = await self._summarize_search_results(query, results)
            return {
                "status": "success",
                "query_type": "web_search",
                "query_success": True,
                "results": {
                    "summary": summary,
                    "sources": results,
                },
            }
        except Exception as exc:
            logger.error("web search failed: %s", exc)
            return {
                "status": "success",
                "query_type": "web_search",
                "query_success": False,
                "results": {"error": f"Search failed: {exc}"},
            }

    async def _summarize_search_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant information was found"

        if self.model is None:
            first = results[0]
            title = first.get("title", "")
            snippet = first.get("snippet", "")
            return f"{title}: {snippet}".strip(": ")

        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. {result.get('title', '')}\n{result.get('snippet', '')}\n"

        current_date = datetime.now().strftime("%Y-%m-%d")
        weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][datetime.now().weekday()]
        prompt = f"""Answer the user question concisely based on the following search results.
Respond in the same language as the user input.

Current time:
{current_date} {weekday}

User question:
{query}

Search results:
{results_text}

Requirements:
1. Prefer consistent information from higher-quality sources.
2. Do not fabricate anything that is not present in the search results.
3. Output concise natural language. Do not output JSON."""

        try:
            text = await collect_model_text(self.model, [{"role": "user", "content": prompt}])
            return text.strip() if text else "Unable to generate a summary"
        except Exception:
            return "Search succeeded, but summary generation failed"
