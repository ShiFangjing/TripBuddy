"""Standalone RAGKnowledge agent backed by the migrated Milvus Lite knowledge base."""
from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from agentscope.agent import AgentBase
from agentscope.message import Msg

from app.agents._common import collect_model_text, extract_query, parse_orchestrator_payload

try:
    from sentence_transformers import SentenceTransformer

    _EMBEDDING_READY = True
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]
    _EMBEDDING_READY = False

try:
    from pymilvus import MilvusClient

    _MILVUS_READY = True
except Exception:
    MilvusClient = None  # type: ignore[assignment]
    _MILVUS_READY = False


class RagKnowledgeAgent(AgentBase):
    """RAG agent that retrieves chunk text from Milvus, without reading raw txt files at runtime."""

    def __init__(
        self,
        name: str = "rag_knowledge",
        model: Any = None,
        knowledge_base_path: Optional[Union[str, Path]] = None,
        collection_name: str = "business_travel_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 3,
        project_root: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.top_k = max(int(top_k), 1)
        root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
        # PyCharm may run this script with TripBuddy2 as the working directory.
        # Normalize that case so data paths do not become TripBuddy2/TripBuddy2.
        self.project_root = root.parent if root.name == "TripBuddy2" else root

        if knowledge_base_path is None:
            knowledge_base_path = self.project_root / "TripBuddy2" / "data" / "rag_knowledge"
        self.knowledge_base_path = Path(knowledge_base_path).resolve()
        self.milvus_db_path = self.knowledge_base_path / "milvus_lite.db"
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._embedder: Any = None
        self._milvus_client: Any = None
        self._ready = False
        self._ready_error = ""

    def _resolve_embedding_path(self) -> str:
        configured = self.embedding_model
        try:
            from config import RAG_CONFIG

            configured = RAG_CONFIG.get("embedding_model", configured)
        except Exception:
            pass

        candidates: List[Path] = []
        cfg_path = Path(str(configured)).expanduser()
        if cfg_path.is_absolute():
            candidates.append(cfg_path)
        else:
            candidates.append((self.project_root / cfg_path).resolve())
            candidates.append((self.project_root / "TripBuddy2" / cfg_path).resolve())

        candidates.append(
            self.project_root / "TripBuddy2" / "data" / "models" / "models--BAAI--bge-small-zh-v1.5"
        )
        candidates.append(self.project_root / "data" / "models" / "models--BAAI--bge-small-zh-v1.5")

        for candidate in candidates:
            if not candidate.exists():
                continue
            snapshots = candidate / "snapshots"
            if snapshots.exists() and snapshots.is_dir():
                subdirs = sorted([p for p in snapshots.iterdir() if p.is_dir()], key=lambda p: p.name)
                if subdirs:
                    return str(subdirs[-1])
            return str(candidate)

        return "BAAI/bge-small-zh-v1.5"

    def _ensure_backend(self) -> Tuple[bool, str]:
        if self._ready:
            return True, ""
        if self._ready_error:
            return False, self._ready_error
        if not _EMBEDDING_READY:
            self._ready_error = "sentence-transformers is not available"
            return False, self._ready_error
        if not _MILVUS_READY:
            self._ready_error = "pymilvus is not available"
            return False, self._ready_error
        if not self.milvus_db_path.exists():
            self._ready_error = f"knowledge base file not found: {self.milvus_db_path}"
            return False, self._ready_error

        try:
            # Suppress noisy transformer loading reports in CLI/PyCharm output.
            logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("transformers.integrations.tensor_parallel").setLevel(logging.ERROR)

            model_path = self._resolve_embedding_path()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self._embedder = SentenceTransformer(model_path)
                self._milvus_client = MilvusClient(str(self.milvus_db_path))

            if not self._milvus_client.has_collection(self.collection_name):
                self._ready_error = f"collection not found: {self.collection_name}"
                return False, self._ready_error

            self._ready = True
            return True, ""
        except Exception as exc:
            self._ready_error = str(exc)
            return False, self._ready_error

    def _ensure_connection(self) -> None:
        ok, err = self._ensure_backend()
        if not ok:
            raise RuntimeError(err)
        try:
            self._milvus_client.has_collection(self.collection_name)
        except Exception:
            if hasattr(self._milvus_client, "close"):
                try:
                    self._milvus_client.close()
                except Exception:
                    pass
            self._milvus_client = MilvusClient(str(self.milvus_db_path))

    @staticmethod
    def _hit_get(hit: Any, key: str, default: Any = None) -> Any:
        if isinstance(hit, dict):
            return hit.get(key, default)
        try:
            return hit[key]
        except Exception:
            return getattr(hit, key, default)

    @classmethod
    def _parse_hit(cls, hit: Any) -> Dict[str, Any]:
        entity = cls._hit_get(hit, "entity", {}) or {}
        if not isinstance(entity, dict):
            entity = getattr(entity, "to_dict", lambda: {})()

        metadata_raw = entity.get("metadata", "{}")
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except Exception:
                metadata = {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            metadata = {}

        return {
            "id": entity.get("id", cls._hit_get(hit, "id", "")),
            "content": entity.get("content", ""),
            "metadata": metadata,
            "score": float(cls._hit_get(hit, "distance", 0.0) or 0.0),
        }

    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
        ok, err = self._ensure_backend()
        if not ok:
            return [], err

        try:
            self._ensure_connection()
            limit = int(top_k or self.top_k)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                query_embedding = self._embedder.encode(query).tolist()

            results = self._milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=["id", "content", "metadata"],
            )

            docs: List[Dict[str, Any]] = []
            if results:
                for hit in results[0]:
                    parsed = self._parse_hit(hit)
                    if parsed.get("content"):
                        docs.append(parsed)
            return docs, ""
        except Exception as exc:
            return [], str(exc)

    def get_stats(self) -> Dict[str, Any]:
        ok, err = self._ensure_backend()
        if not ok:
            return {"status": "error", "message": err}
        try:
            stats = self._milvus_client.get_collection_stats(self.collection_name)
            return {
                "status": "success",
                "collection_name": self.collection_name,
                "total_documents": stats.get("row_count", 0),
                "knowledge_base_path": str(self.knowledge_base_path),
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def close(self) -> None:
        if self._milvus_client is not None and hasattr(self._milvus_client, "close"):
            try:
                self._milvus_client.close()
            except Exception:
                pass

    def _knowledge_context(self, docs: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[Knowledge chunk {i}]\n{doc.get('content', '')}")
        return "\n\n".join(parts)

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content='{"status":"error","message":"no input"}', role="assistant")

        content = x.content if not isinstance(x, list) else x[-1].content
        _, context, _ = parse_orchestrator_payload(content)
        query = extract_query(context, fallback=str(content))

        if not query:
            return Msg(
                name=self.name,
                content=json.dumps({"status": "error", "message": "empty query"}, ensure_ascii=False),
                role="assistant",
            )

        docs, search_error = self.search_knowledge(query)
        if search_error:
            result = {
                "status": "error",
                "query": query,
                "answer": "Knowledge retrieval is unavailable. Please check the Milvus knowledge base and the local embedding model.",
                "message": search_error,
                "retrieved_documents": [],
            }
            return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

        if not docs:
            result = {
                "status": "no_knowledge",
                "query": query,
                "answer": "No relevant information was found in the knowledge base.",
                "retrieved_documents": [],
            }
            return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

        if self.model is None:
            answer = "Here is the relevant information from the knowledge base:\n\n" + self._knowledge_context(docs)
        else:
            prompt = f"""You are a business travel knowledge assistant.
Answer strictly based on the provided knowledge chunks. Do not fabricate.
Respond in the same language as the user input.

User question: {query}

Knowledge chunks:
{self._knowledge_context(docs)}

If the chunks are insufficient, say that no relevant information was found in the knowledge base."""
            try:
                answer = await collect_model_text(
                    self.model,
                    [
                        {
                            "role": "system",
                            "content": "You are a business travel knowledge assistant. Respond in the same language as the user.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                if not answer:
                    answer = "No relevant information was found in the knowledge base."
            except Exception as exc:
                answer = f"Knowledge was retrieved, but answer generation failed: {exc}"

        result = {
            "status": "success",
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "content": (doc.get("content", "")[:220] + "...")
                    if len(doc.get("content", "")) > 220
                    else doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.0),
                }
                for doc in docs
            ],
            "knowledge_base_path": str(self.knowledge_base_path),
        }
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")
