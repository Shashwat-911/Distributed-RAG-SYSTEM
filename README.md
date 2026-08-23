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
AeroCache Distributed Partition Engine
├── ConsistentHashRing (150 virtual nodes per partition)
├── Sharded LRUCache Partitions with O(1) Intrusive DLL
├── Predictive AI Cold-Key Eviction with LRU Fallback
└── Hit/Miss & Epoch Metrics Tracker
    │ (MISS)
    ▼
EmbeddingEngine (all-MiniLM-L6-v2, 128-dim)
    │
    ▼
NexusSearch MapReduce Retrieval Engine
├── Sharded Posting List InvertedIndex (Sublinear TF-IDF + Length Norm)
├── Variance-Split KD-Tree (Vectorized Cosine Distance + Branch Pruning)
├── MapReduce Parallel Retrieval (Concurrent Executor across shards)
└── Reciprocal Rank Fusion (RRF) + Alpha Dense-Sparse Hybrid Blending
    │
    ▼
Ollama Generator (qwen2.5-coder:1.5b → codellama fallback)
    │
    ▼
Grounded Answer + Ranked Sources
```

---

## Core Algorithms

| Module | Architecture | Algorithm | Complexity |
|---|---|---|---|
| `cache_engine.py` | **AeroCache** | `ConsistentHashRing` (MD5, 150 vnodes) | Lookup: $O(\log (N \cdot V))$ time, $O(N \cdot V)$ space |
| `cache_engine.py` | **AeroCache** | `ShardedAeroCache` (Multi-partition O(1) DLL) | Get / Put: $O(1)$ time, Zero lock contention |
| `cache_engine.py` | **AeroCache** | `PredictiveEvictionPolicy` (Frequency-Recency Decay) | Eval: $O(1)$ time, proactive cold-key purge |
| `cache_engine.py` | **AeroCache** | `LeakyBucketRateLimiter` (`threading.Condition`) | Acquire: $O(1)$ time, $O(1)$ space |
| `retrieval_engine.py` | **NexusSearch** | `NexusInvertedIndex` (Sublinear TF-IDF $1+\ln(\text{tf})$) | Build: $O(D \cdot L)$, Query: $O(|Q| \cdot \text{df}_{\text{avg}} + K \log K)$ |
| `retrieval_engine.py` | **NexusSearch** | `NexusKDTree` (Vectorized Cosine + Pruning) | Build: $O(D \cdot N \log N)$, Search: $O(\log N)$ avg |
| `retrieval_engine.py` | **NexusSearch** | MapReduce `RetrievalEngine` (Threaded RRF Fusion) | Parallel Map: $O(1)$ threads, RRF merge: $O(K \log K)$ |
| `diff_engine.py` | **DiffEngine** | `Hirschberg LCS` (Space-optimized divide-and-conquer) | Time: $O(M \cdot N)$, Space: $O(\min(M, N))$ |
| `diff_engine.py` | **DiffEngine** | `DeltaExtractor` (Chunk-level + char-level diff) | Time: $O(C_1 \cdot C_2 + U \cdot \Delta)$, Space: $O(C_1 + C_2)$ |

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

- **AeroCache Sharded Engine & ConsistentHashRing**: Built from scratch using MD5 hashing and 150 virtual nodes per partition to distribute cache keys across independent shards with zero lock contention.
- **Predictive AI Eviction Policy**: Built from scratch using frequency-recency exponential decay metrics to identify and purge cold keys before memory saturation, with $O(1)$ LRU fallback.
- **LeakyBucketRateLimiter**: Built using `threading.Condition` and continuous leak rate calculation to enforce smooth request rate limits across concurrent worker threads without external dependencies.
- **NexusSearch InvertedIndex**: Built from scratch with sublinear TF-IDF ($1 + \ln(\text{tf})$), term posting lists, and cosine document length normalization for sub-millisecond keyword retrieval.
- **NexusSearch KDTree**: Built from scratch using variance-based dimension splitting, NumPy vectorized cosine distance calculation, and bounding-box pruning for exact $k$-nearest neighbor search.
- **NexusSearch MapReduce Fusion**: Built from scratch using thread-level concurrent fan-out and Reciprocal Rank Fusion (RRF) to combine sparse and dense signals.
- **Hirschberg LCS & DeltaExtractor**: Built using linear-space divide-and-conquer dynamic programming to compute longest common subsequences in $O(\min(M, N))$ space and re-index only modified text.

---

## License

MIT
