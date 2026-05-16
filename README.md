# LangGraph Rewrite (Phase-1)

This is a parallel rewrite project under the same repository.

## Scope
- Graph skeleton implemented with LangGraph
- Explicit registration (no lazy loading)
- Enabled execution nodes: `event_collection`, `preference`, `memory_query`, `information_query`, `itinerary_planning`
- Disabled in flow: `rag_knowledge` / `ask-question`
- Reuse existing memory semantics via adapters
- Standalone local skills under `langgraph_rewrite/skills/` (no `.claude/skills` runtime dependency)
- `app/agents/` is kept as a compatibility import layer for existing tests and callers
- Skill-backed execution nodes are loaded by `app.skills.registry.SkillRegistry`
- `rag_knowledge` code is retained but disabled in the active LangGraph flow. When re-enabled, it uses Milvus Lite vector retrieval:
  - knowledge DB: `langgraph_rewrite/data/rag_knowledge/milvus_lite.db`
  - collection: `business_travel_knowledge`
  - embedding model: `langgraph_rewrite/data/models/models--BAAI--bge-small-zh-v1.5/snapshots/<hash>`
  - runtime does not read `langgraph_rewrite/data/documents/*.txt`; chunk text is stored in Milvus

## Quick start
```bash
python cli_langgraph.py --self-check
python cli_langgraph.py --query "I live in Beijing Chaoyang and prefer window seats."
```

## Tests
```bash
python -m pytest tests_langgraph -q
```
