# DistributedRAG

> Industry-grade Retrieval-Augmented Generation pipeline built from algorithmic first principles.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit App](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://distributed-rag.streamlit.app/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🚀 **Live Interactive Demo:** [https://distributed-rag.streamlit.app/](https://distributed-rag.streamlit.app/)

---

## What Makes This Different

DistributedRAG contains zero third-party vector databases or wrapper frameworks: no FAISS, no Pinecone, and no LangChain. Every data structure and algorithm is implemented from scratch using Python standard libraries and NumPy. The codebase implements four core systems from algorithmic fundamentals: a distributed consistent hash cache engine, a hybrid sparse-dense retrieval engine, a space-optimized document diff engine, and an end-to-end generation pipeline with automatic local model fallback.

---

## Architecture Diagram

```
User Query
    │
    ▼
LRUCache (O(1) TTL lookup) ──── HIT ──── Cached Response
    │ MISS
    ▼
EmbeddingEngine (all-MiniLM-L6-v2, 128-dim)
    │
    ▼
RetrievalEngine
├── InvertedIndex (TF-IDF from scratch)
├── KDTree (variance-split, cosine similarity)
└── Hybrid fusion (α=0.5)
    │
    ▼
OllamaClient (qwen2.5-coder → codellama fallback)
    │
    ▼
Grounded Answer + Sources
```

---

## Core Algorithms

| Module | Algorithm | Complexity |
|---|---|---|
| `cache_engine.py` | `ConsistentHashRing` (MD5, 150 vnodes) | Lookup: $O(\log (N \cdot V))$ time, $O(N \cdot V)$ space |
| `cache_engine.py` | `LRUCache` (doubly linked list + hashmap) | Get / Put / Evict: $O(1)$ time, $O(C)$ space |
| `cache_engine.py` | `LeakyBucketRateLimiter` (`threading.Condition`) | Acquire: $O(1)$ time, $O(1)$ space |
| `retrieval_engine.py` | `InvertedIndex` TF-IDF | Build: $O(D \cdot L)$, Query: $O(\|Q\| \cdot \text{df}_{\text{avg}} + K \log K)$ |
| `retrieval_engine.py` | `KDTree` KNN (variance split, cosine metric) | Build: $O(D \cdot N \log N)$, Search: $O(\log N)$ avg / $O(N)$ worst |
| `diff_engine.py` | `Hirschberg LCS` (space-optimized divide-and-conquer) | Time: $O(M \cdot N)$, Space: $O(\min(M, N))$ |
| `diff_engine.py` | `DeltaExtractor` (chunk-level + char-level) | Time: $O(C_1 \cdot C_2 + U \cdot \Delta)$, Space: $O(C_1 + C_2)$ |

---

## Project Structure

```
rag-core/
├── .env.example
├── Dockerfile
├── README.md
├── README_DEPLOY.md
├── requirements.txt
├── run.py
├── __init__.py
├── api/
│   ├── __init__.py
│   └── main.py
├── cache/
│   ├── __init__.py
│   └── cache_engine.py
├── demo/
│   └── demo.webp
├── diff/
│   ├── __init__.py
│   └── diff_engine.py
├── frontend/
│   ├── __init__.py
│   ├── app.py
│   └── hf_app.py
├── pipeline/
│   ├── __init__.py
│   └── rag_pipeline.py
└── retrieval/
    ├── __init__.py
    └── retrieval_engine.py
```

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running

### 2. Clone repository
```bash
git clone https://github.com/Shashwat-911/Distributed-RAG-SYSTEM.git
cd Distributed-RAG-SYSTEM
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull LLM models
```bash
ollama pull qwen2.5-coder
ollama pull codellama
```

### 5. Launch DistributedRAG
```bash
python run.py
```

### 6. Open URLs
- **Live Cloud Web UI:** https://distributed-rag.streamlit.app/
- **Local Web UI:** http://localhost:8501
- **REST API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents/ingest` | Chunk, embed, and index a new document into keyword and vector indices. |
| `POST` | `/documents/update` | Incrementally diff and update an existing document using Hirschberg LCS. |
| `DELETE` | `/documents/{doc_id}` | Remove a document and all its indexed chunks from the retrieval engine. |
| `POST` | `/query` | Execute end-to-end RAG query using hybrid, vector, or keyword search and LLM generation. |
| `GET` | `/health` | Check backend availability, Ollama connectivity, indexed document count, and cache size. |
| `GET` | `/cache/stats` | Retrieve cache metrics including hit rate, hits, misses, and active entry count. |
| `POST` | `/cache/clear` | Evict all cached responses from memory and reset hit/miss counters. |

---

## Demo

[▶ Watch Demo](demo/demo.webp)

A live demo video showing end-to-end RAG query, delta document update, and retrieval inspection.

![DistributedRAG Live Demo](demo/demo.webp)

---

## What I Built from Scratch

- **ConsistentHashRing**: Built from scratch using MD5 hashing and virtual nodes to distribute cache keys uniformly across partitions while minimizing remapping during node additions or removals.
- **LRUCache**: Built with a custom doubly linked list and hash map to achieve strictly $O(1)$ lookups, insertions, and evictions alongside millisecond-level TTL expiration.
- **LeakyBucketRateLimiter**: Built using `threading.Condition` and continuous leak rate calculation to enforce smooth request rate limits across concurrent worker threads without external dependencies.
- **InvertedIndex TF-IDF**: Built from scratch to maintain exact term-frequency matrices, inverse-document frequencies, and sparse vector representations for sub-millisecond lexical search.
- **KDTree**: Built from scratch using variance-based dimension splitting and recursive pruning to perform exact $k$-nearest neighbor search across cosine embedding spaces.
- **Hirschberg LCS**: Built using linear-space divide-and-conquer dynamic programming to compute longest common subsequences in $O(\min(M, N))$ space rather than standard quadratic $O(M \cdot N)$ memory.
- **DeltaExtractor**: Built from scratch to identify granular chunk additions, modifications, and deletions so document updates only re-embed modified text rather than re-indexing entire corpora.

---

## License

MIT
