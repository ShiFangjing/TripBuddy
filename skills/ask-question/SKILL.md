---
name: ask-question
description: Answer travel policy, reimbursement, booking, and business travel knowledge questions using the local Milvus Lite RAG knowledge base.
---

# Ask Question

Use this skill for business travel policy and knowledge-base questions.

## Retrieval

- Runtime knowledge base: `langgraph_rewrite/data/rag_knowledge/milvus_lite.db`
- Collection: `business_travel_knowledge`
- Stored fields: chunk vector, chunk text, metadata
- Local query embedding model: `langgraph_rewrite/data/models`
- The runtime does not read raw txt documents from `langgraph_rewrite/data/documents`

## Output Contract

- `status`
- `query`
- `answer`
- `retrieved_documents`
