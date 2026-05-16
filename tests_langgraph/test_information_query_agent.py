import asyncio
import json

from agentscope.message import Msg

from app.agents.information_query_agent import InformationQueryAgent


def test_information_query_weather_path(monkeypatch):
    agent = InformationQueryAgent(model=None)

    async def fake_weather(query):
        return {
            "status": "success",
            "query_type": "weather",
            "query_success": True,
            "results": {
                "summary": f"{query}：\u6674 10-20°C",
                "sources": [{"url": "https://wttr.in", "title": "wttr.in"}],
            },
        }

    monkeypatch.setattr(agent, "_weather_query", fake_weather)

    msg = Msg(name="u", role="user", content=json.dumps({"context": {"rewritten_query": "\u5317\u4eac\u5929\u6c14\u600e\u4e48\u6837"}}))
    out = asyncio.run(agent.reply(msg))
    payload = json.loads(out.content)

    assert payload["query_type"] == "weather"
    assert payload["query_success"] is True
    assert payload["results"]["sources"][0]["url"] == "https://wttr.in"


def test_information_query_web_search_path(monkeypatch):
    agent = InformationQueryAgent(model=None)

    async def fake_web(query):
        return {
            "status": "success",
            "query_type": "web_search",
            "query_success": True,
            "results": {
                "summary": f"{query}：\u8fd9\u91cc\u662f\u641c\u7d22\u6458\u8981",
                "sources": [{"title": "example", "snippet": "demo", "url": "https://example.com"}],
            },
        }

    monkeypatch.setattr(agent, "_web_search", fake_web)

    msg = Msg(name="u", role="user", content=json.dumps({"context": {"rewritten_query": "\u67e5\u4e00\u4e0b\u65e0\u9521\u65c5\u6e38\u653b\u7565"}}))
    out = asyncio.run(agent.reply(msg))
    payload = json.loads(out.content)

    assert payload["query_type"] == "web_search"
    assert payload["query_success"] is True
    assert payload["results"]["sources"][0]["url"] == "https://example.com"


def test_information_query_summarize_without_model():
    agent = InformationQueryAgent(model=None)
    summary = asyncio.run(
        agent._summarize_search_results(
            "\u67e5\u4e00\u4e0b\u676d\u5dde\u666f\u70b9",
            [{"title": "\u676d\u5dde\u666f\u70b9\u63a8\u8350", "snippet": "\u897f\u6e56、\u7075\u9690\u5bfa、\u9f99\u4e95\u6751", "url": "https://example.com"}],
        )
    )
    assert "\u676d\u5dde\u666f\u70b9\u63a8\u8350" in summary

