"""
FastAPI REST API Layer for RAG-Core.

Provides endpoints for document ingestion, incremental delta updates,
document deletion, multi-mode querying (hybrid/vector/keyword), health diagnostics,
and LRU cache management.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure package root is resolvable across varied run contexts
_current_dir = Path(__file__).resolve().parent
_rag_core_dir = _current_dir.parent
_workspace_root = _rag_core_dir.parent

for _p in (str(_workspace_root), str(_rag_core_dir), str(_current_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

try:
    from rag_core.diff.diff_engine import DeltaExtractor
    from rag_core.pipeline.rag_pipeline import (
        OllamaUnavailableError,
        RAGPipeline,
        RAGResponse,
        SearchResult,
    )
except ImportError:
    from diff.diff_engine import DeltaExtractor
    from pipeline.rag_pipeline import (
        OllamaUnavailableError,
        RAGPipeline,
        RAGResponse,
        SearchResult,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan Management
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes a single singleton RAGPipeline instance upon application startup."""
    app.state.pipeline = RAGPipeline()
    app.state.cache_hits = 0
    app.state.cache_misses = 0
    yield


app = FastAPI(
    title="RAG-Core API",
    description="Algorithmic Retrieval-Augmented Generation REST API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Raw document content")


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int
    status: str


class UpdateRequest(BaseModel):
    doc_id: str = Field(..., description="Existing document identifier")
    new_text: str = Field(..., description="Updated document content")


class UpdateResponse(BaseModel):
    doc_id: str
    chunks_added: int
    chunks_modified: int
    chunks_deleted: int


class DeleteResponse(BaseModel):
    doc_id: str
    status: str


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, description="Number of context chunks to retrieve")
    mode: str = Field(default="hybrid", description="Retrieval mode: 'hybrid', 'vector', or 'keyword'")
    use_cache: bool = Field(default=True, description="Whether to check and update LRU response cache")


class SearchResultModel(BaseModel):
    doc_id: str
    chunk_text: str
    score: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResultModel]
    cached: bool
    model_used: str
    retrieval_time_ms: float
    generation_time_ms: float


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    docs_indexed: int
    cache_size: int


class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    hit_rate: float
    size: int


class CacheClearResponse(BaseModel):
    status: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _get_pipeline() -> RAGPipeline:
    pipeline: Optional[RAGPipeline] = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Pipeline is not initialized",
        )
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/documents/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(payload: IngestRequest) -> IngestResponse:
    if not payload.doc_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="doc_id cannot be empty",
        )
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text cannot be empty",
        )

    pipeline = _get_pipeline()
    try:
        chunks = pipeline.store._chunker.chunk_text(payload.text)
        pipeline.store.add_document(payload.doc_id, payload.text)
        return IngestResponse(
            doc_id=payload.doc_id,
            chunks_added=len(chunks),
            status="indexed",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {exc}",
        )


@app.post("/documents/update", response_model=UpdateResponse, status_code=status.HTTP_200_OK)
async def update_document(payload: UpdateRequest) -> UpdateResponse:
    if not payload.doc_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="doc_id cannot be empty",
        )

    pipeline = _get_pipeline()
    with pipeline.store._lock:
        old_text = pipeline.store._documents.get(payload.doc_id)
        if old_text is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{payload.doc_id}' not found",
            )

    try:
        extractor = DeltaExtractor()
        old_chunks = pipeline.store._chunker.chunk_text(old_text)
        new_chunks = pipeline.store._chunker.chunk_text(payload.new_text)
        deltas = extractor.extract(old_chunks, new_chunks)

        chunks_added = sum(1 for d in deltas if d.kind == "ADDED")
        chunks_modified = sum(1 for d in deltas if d.kind == "MODIFIED")
        chunks_deleted = sum(1 for d in deltas if d.kind == "DELETED")

        pipeline.store.update_document(payload.doc_id, payload.new_text)

        return UpdateResponse(
            doc_id=payload.doc_id,
            chunks_added=chunks_added,
            chunks_modified=chunks_modified,
            chunks_deleted=chunks_deleted,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {exc}",
        )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str) -> DeleteResponse:
    if not doc_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="doc_id cannot be empty",
        )

    pipeline = _get_pipeline()
    with pipeline.store._lock:
        if doc_id not in pipeline.store._documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found",
            )
        del pipeline.store._documents[doc_id]
        if doc_id in pipeline.store._chunk_map:
            del pipeline.store._chunk_map[doc_id]

    return DeleteResponse(doc_id=doc_id, status="deleted")


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_pipeline(payload: QueryRequest) -> QueryResponse:
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query cannot be empty",
        )
    if payload.top_k < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be at least 1",
        )

    mode = payload.mode.lower()
    if mode not in ("hybrid", "vector", "keyword"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{payload.mode}'. Must be 'hybrid', 'vector', or 'keyword'",
        )

    pipeline = _get_pipeline()

    try:
        if payload.use_cache:
            response = pipeline.query(
                user_query=payload.query,
                top_k=payload.top_k,
                mode=mode,
            )
        else:
            query_vector = pipeline.store._retrieval._tree and None
            t0 = time.perf_counter()
            sources = pipeline.store.search(payload.query, top_k=payload.top_k, mode=mode)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0

            context_parts: List[str] = [
                f"[{i}] (doc: {src.doc_id}, score: {src.score:.4f})\n{src.chunk_text}"
                for i, src in enumerate(sources, 1)
            ]
            context = "\n\n".join(context_parts) if context_parts else "(no context found)"
            prompt = pipeline._build_prompt(context, payload.query)

            t1 = time.perf_counter()
            answer = pipeline._llm.generate(prompt)
            generation_ms = (time.perf_counter() - t1) * 1000.0
            model_used = pipeline._llm.last_model_used

            response = RAGResponse(
                answer=answer,
                sources=sources,
                cached=False,
                model_used=model_used,
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=generation_ms,
            )

        return QueryResponse(
            answer=response.answer,
            sources=[
                SearchResultModel(
                    doc_id=s.doc_id,
                    chunk_text=s.chunk_text,
                    score=s.score,
                    source=s.source,
                )
                for s in response.sources
            ],
            cached=response.cached,
            model_used=response.model_used,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama generation service unavailable: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {exc}",
        )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    pipeline = _get_pipeline()
    ollama_connected = pipeline._llm.health_check()
    with pipeline.store._lock:
        docs_indexed = len(pipeline.store._documents)
    
    stats = pipeline.cache.stats() if hasattr(pipeline, "cache") else {"size": 0}
    cache_size = stats.get("size", 0)

    app_status = "healthy" if ollama_connected else "degraded"
    return HealthResponse(
        status=app_status,
        ollama_connected=ollama_connected,
        docs_indexed=docs_indexed,
        cache_size=cache_size,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse, status_code=status.HTTP_200_OK)
async def cache_stats() -> CacheStatsResponse:
    pipeline = _get_pipeline()
    if hasattr(pipeline, "cache"):
        stats = pipeline.cache.stats()
        return CacheStatsResponse(
            hits=stats["hits"],
            misses=stats["misses"],
            hit_rate=round(stats["hit_rate"], 4),
            size=stats["size"],
        )
    hits = getattr(app.state, "cache_hits", 0)
    misses = getattr(app.state, "cache_misses", 0)
    total = hits + misses
    hit_rate = (hits / total) if total > 0 else 0.0

    return CacheStatsResponse(
        hits=hits,
        misses=misses,
        hit_rate=round(hit_rate, 4),
        size=0,
    )


@app.post("/cache/clear", response_model=CacheClearResponse, status_code=status.HTTP_200_OK)
async def clear_cache() -> CacheClearResponse:
    pipeline = _get_pipeline()
    if hasattr(pipeline, "cache"):
        pipeline.cache.clear()
    app.state.cache_hits = 0
    app.state.cache_misses = 0

    return CacheClearResponse(status="cache cleared")
