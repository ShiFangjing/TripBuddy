from __future__ import annotations

from pathlib import Path

from app.agents.rag_knowledge_agent import RagKnowledgeAgent


class FakeEmbedder:
    def encode(self, query):
        return FakeVector([0.1, 0.2, 0.3])


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeMilvusClient:
    def __init__(self):
        self.search_calls = []

    def has_collection(self, collection_name):
        return collection_name == "business_travel_knowledge"

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [
            [
                {
                    "entity": {
                        "id": 1,
                        "content": "\u98de\u673a\u7968\u62a5\u9500\u901a\u5e38\u4ee5\u5408\u89c4\u884c\u7a0b\u5355\u548c\u516c\u53f8\u5dee\u65c5\u6807\u51c6\u4e3a\u51c6。",
                        "metadata": '{"category":"\u62a5\u9500\u89c4\u5b9a","title":"\u673a\u7968\u62a5\u9500"}',
                    },
                    "distance": 0.87,
                }
            ]
        ]

    def get_collection_stats(self, collection_name):
        return {"row_count": 1}


def test_search_knowledge_reads_chunk_text_from_milvus(monkeypatch):
    agent = RagKnowledgeAgent(model=None, project_root=Path("."))
    fake_client = FakeMilvusClient()

    monkeypatch.setattr(agent, "_ensure_backend", lambda: (True, ""))
    agent._embedder = FakeEmbedder()
    agent._milvus_client = fake_client

    docs, err = agent.search_knowledge("\u98de\u673a\u7968\u7684\u62a5\u9500\u4e0a\u9650\u662f\u4ec0\u4e48")

    assert err == ""
    assert docs[0]["content"] == "\u98de\u673a\u7968\u62a5\u9500\u901a\u5e38\u4ee5\u5408\u89c4\u884c\u7a0b\u5355\u548c\u516c\u53f8\u5dee\u65c5\u6807\u51c6\u4e3a\u51c6。"
    assert docs[0]["metadata"] == {"category": "\u62a5\u9500\u89c4\u5b9a", "title": "\u673a\u7968\u62a5\u9500"}
    assert fake_client.search_calls[0]["collection_name"] == "business_travel_knowledge"
    assert fake_client.search_calls[0]["output_fields"] == ["id", "content", "metadata"]


def test_missing_milvus_db_fails_before_loading_embedding(tmp_path):
    agent = RagKnowledgeAgent(model=None, knowledge_base_path=tmp_path, project_root=Path("."))

    ok, err = agent._ensure_backend()

    assert ok is False
    assert "knowledge base file not found" in err


def test_parse_hit_accepts_metadata_dict():
    hit = {
        "entity": {
            "id": 2,
            "content": "\u4f4f\u5bbf\u8d39\u6309\u57ce\u5e02\u7b49\u7ea7\u548c\u804c\u7ea7\u6807\u51c6\u6267\u884c。",
            "metadata": {"category": "\u5dee\u65c5\u89c4\u5b9a"},
        },
        "distance": 0.76,
    }

    parsed = RagKnowledgeAgent._parse_hit(hit)

    assert parsed["id"] == 2
    assert parsed["metadata"] == {"category": "\u5dee\u65c5\u89c4\u5b9a"}
    assert parsed["score"] == 0.76
