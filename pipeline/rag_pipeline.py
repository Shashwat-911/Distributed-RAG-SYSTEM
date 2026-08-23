"""
RAG Pipeline — end-to-end Retrieval-Augmented Generation orchestrator.

Integrates the cache, retrieval, and diff engines with an embedding model
(sentence-transformers) and a local Ollama LLM to deliver a complete
query → context-retrieval → generation pipeline with caching, incremental
document updates, and automatic model fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rag_core.cache.cache_engine import LRUCache
    from rag_core.retrieval.retrieval_engine import RetrievalEngine
    from rag_core.diff.diff_engine import Chunker, DiffEngine
except ImportError:
    from cache.cache_engine import LRUCache
    from retrieval.retrieval_engine import RetrievalEngine
    from diff.diff_engine import Chunker, DiffEngine

import os

logger = logging.getLogger(__name__)

_EMBEDDING_DIM    = int(os.environ.get("RAG_EMBEDDING_DIM", "128"))
_OLLAMA_HOST      = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_PRIMARY_MODEL    = os.environ.get("OLLAMA_PRIMARY", "qwen2.5-coder")
_FALLBACK_MODEL   = os.environ.get("OLLAMA_FALLBACK", "codellama")
_OLLAMA_TIMEOUT   = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
_CACHE_TTL        = int(os.environ.get("RAG_CACHE_TTL", "3600"))


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class OllamaUnavailableError(Exception):
    """Raised when all configured Ollama models fail to produce a response."""


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchResult:
    """Single retrieval hit returned by :class:`DocumentStore`.

    Attributes:
        doc_id: Identifier of the source document.
        chunk_text: Retrieved chunk content.
        score: Retrieval relevance score.
        source: Retrieval mode that produced this result.
    """

    doc_id: str
    chunk_text: str
    score: float
    source: str


@dataclass(frozen=True)
class RAGResponse:
    """Structured response from :meth:`RAGPipeline.query`.

    Attributes:
        answer: Generated answer text from the LLM.
        sources: Retrieval results used as context.
        cached: Whether the response was served from cache.
        model_used: Ollama model that generated the answer.
        retrieval_time_ms: Wall-clock time for the retrieval phase.
        generation_time_ms: Wall-clock time for the generation phase.
    """

    answer: str
    sources: List[SearchResult]
    cached: bool
    model_used: str
    retrieval_time_ms: float
    generation_time_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Engine
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Singleton embedding engine backed by sentence-transformers.

    Loads the ``all-MiniLM-L6-v2`` model lazily on first invocation and
    reuses it for all subsequent calls. Embeddings are projected to
    128 dimensions and L2-normalised.

    Thread Safety:
        Model loading is guarded by a lock; inference itself is stateless
        and safe for concurrent use once the model is loaded.
    """

    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    _instance_lock: threading.Lock = threading.Lock()
    _model: Optional[Any] = None

    @classmethod
    def _load_model(cls) -> None:
        """Load the sentence-transformer model if not already resident."""
        if cls._model is not None:
            return
        with cls._instance_lock:
            if cls._model is not None:
                return
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(cls._MODEL_NAME)
            logger.info("EmbeddingEngine: loaded model '%s'", cls._MODEL_NAME)

    @classmethod
    def embed(cls, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Raw input text.

        Returns:
            L2-normalised numpy array of shape ``(128,)``.
        """
        cls._load_model()
        raw: np.ndarray = cls._model.encode(text, convert_to_numpy=True)
        truncated = raw[:_EMBEDDING_DIM].astype(np.float32)
        norm = np.linalg.norm(truncated)
        if norm > 0:
            truncated /= norm
        return truncated

    @classmethod
    def embed_batch(cls, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of text strings.

        Args:
            texts: List of raw input texts.

        Returns:
            List of L2-normalised numpy arrays, each of shape ``(128,)``.
        """
        if not texts:
            return []
        cls._load_model()
        raw_batch: np.ndarray = cls._model.encode(texts, convert_to_numpy=True)
        results: List[np.ndarray] = []
        for raw in raw_batch:
            truncated = raw[:_EMBEDDING_DIM].astype(np.float32)
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated /= norm
            results.append(truncated)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """HTTP client for the Ollama local LLM REST API.

    Communicates exclusively via :mod:`urllib.request` — no third-party
    HTTP libraries. Attempts generation with the primary model first and
    falls back to the secondary model on failure.

    Attributes:
        _base_url: Ollama API base URL.
        _primary_model: First-choice model identifier.
        _fallback_model: Second-choice model identifier.
        _timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _OLLAMA_HOST,
        primary_model: str = _PRIMARY_MODEL,
        fallback_model: str = _FALLBACK_MODEL,
        timeout: int = _OLLAMA_TIMEOUT,
    ) -> None:
        """
        Args:
            base_url: Ollama server base URL.
            primary_model: Primary model for generation.
            fallback_model: Fallback model if primary is unavailable.
            timeout: HTTP request timeout in seconds.
        """
        self._base_url: str = base_url.rstrip("/")
        self._primary_model: str = primary_model
        self._fallback_model: str = fallback_model
        self._timeout: int = timeout

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        """Send a JSON POST request and return the parsed response.

        Args:
            endpoint: API path (e.g., ``/api/generate``).
            payload: Request body as a dictionary.

        Returns:
            Parsed JSON response dictionary.

        Raises:
            urllib.error.URLError: On network-level failures.
            ValueError: On non-JSON or malformed responses.
        """
        url = f"{self._base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def _get_json(self, endpoint: str) -> dict:
        """Send a GET request and return the parsed response.

        Args:
            endpoint: API path.

        Returns:
            Parsed JSON response dictionary.
        """
        url = f"{self._base_url}{endpoint}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def _get_available_models(self) -> List[str]:
        """Fetch list of model tags currently installed in Ollama."""
        try:
            tags = self._get_json("/api/tags")
            return [m.get("name") for m in tags.get("models", []) if m.get("name")]
        except Exception:
            return []

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate a completion from the given prompt.

        Tries the specified (or primary) model first, then falls back to
        secondary or any locally installed models if earlier attempts fail.

        Args:
            prompt: Full prompt string including context and question.
            model: Override model name. Defaults to the primary model.

        Returns:
            Generated response text.

        Raises:
            OllamaUnavailableError: If all candidate models fail.
        """
        available = self._get_available_models()
        candidates: List[str] = []
        if model:
            candidates.append(model)
        for m in (self._primary_model, self._fallback_model):
            if m and m not in candidates:
                candidates.append(m)
        for am in available:
            if am not in candidates:
                candidates.append(am)

        last_error: Optional[Exception] = None

        for m in candidates:
            try:
                result = self._post_json("/api/generate", {
                    "model": m,
                    "prompt": prompt,
                    "stream": False,
                })
                self._last_model_used = m
                return result.get("response", "")
            except Exception as exc:
                logger.warning(
                    "OllamaClient: model '%s' failed: %s", m, exc
                )
                last_error = exc

        raise OllamaUnavailableError(
            f"All Ollama models failed ({candidates}). Last error: {last_error}"
        )

    @property
    def last_model_used(self) -> str:
        """Model identifier from the most recent successful generation."""
        return getattr(self, "_last_model_used", "")

    def health_check(self) -> bool:
        """Verify Ollama server reachability.

        Returns:
            ``True`` if the server responds, ``False`` otherwise.
        """
        try:
            self._get_json("/api/tags")
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Document Store
# ─────────────────────────────────────────────────────────────────────────────

class DocumentStore:
    """In-memory document store with incremental update support.

    Combines :class:`RetrievalEngine` for hybrid search with
    :class:`DiffEngine` for delta-aware re-embedding on updates.

    Chunk identifiers follow the format ``{doc_id}::chunk::{index}``
    to maintain positional addressability within the retrieval engine.

    Attributes:
        _retrieval: RetrievalEngine instance for search operations.
        _diff: DiffEngine instance for incremental diffing.
        _documents: In-memory store of current document texts.
        _chunk_map: Maps doc_id → list of (chunk_id, chunk_text) pairs.
    """

    def __init__(self, dimensions: int = _EMBEDDING_DIM) -> None:
        """
        Args:
            dimensions: Embedding vector dimensionality.
        """
        self._retrieval: RetrievalEngine = RetrievalEngine(dimensions=dimensions)
        self._diff: DiffEngine = DiffEngine()
        self._chunker: Chunker = Chunker()
        self._documents: Dict[str, str] = {}
        self._chunk_map: Dict[str, List[tuple]] = {}
        self._lock: threading.Lock = threading.Lock()

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        """Construct a deterministic chunk identifier."""
        return f"{doc_id}::chunk::{index}"

    def add_document(self, doc_id: str, text: str) -> None:
        """Chunk, embed, and index a new document.

        Args:
            doc_id: Unique document identifier.
            text: Raw document content.
        """
        chunks = self._chunker.chunk_text(text)
        if not chunks:
            return

        embeddings = EmbeddingEngine.embed_batch(chunks)

        with self._lock:
            self._documents[doc_id] = text
            chunk_entries: List[tuple] = []

            for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._make_chunk_id(doc_id, idx)
                self._retrieval.add_document(chunk_id, chunk_text, vector)
                chunk_entries.append((chunk_id, chunk_text))

            self._chunk_map[doc_id] = chunk_entries

    def update_document(self, doc_id: str, new_text: str) -> None:
        """Incrementally update an existing document using diff-based re-embedding.

        Only ADDED and MODIFIED chunks trigger embedding generation.
        DELETED chunks are noted for removal. The retrieval engine receives
        new entries for changed chunks while stale chunk IDs are effectively
        superseded.

        Args:
            doc_id: Document identifier (must have been previously added).
            new_text: Updated document content.

        Raises:
            KeyError: If doc_id has not been previously indexed.
        """
        with self._lock:
            old_text = self._documents.get(doc_id)
            if old_text is None:
                raise KeyError(
                    f"Document '{doc_id}' not found. Use add_document() first."
                )
            old_chunk_entries = list(self._chunk_map.get(doc_id, []))

        diff_output = self._diff.diff_texts(old_text, new_text)

        upsert_texts: List[str] = []
        upsert_indices: List[int] = []

        for delta in diff_output.to_upsert:
            target_text = delta.new_text if delta.new_text else ""
            target_idx = delta.new_index if delta.new_index is not None else 0
            if target_text:
                upsert_texts.append(target_text)
                upsert_indices.append(target_idx)

        upsert_embeddings = EmbeddingEngine.embed_batch(upsert_texts) if upsert_texts else []

        new_chunks = self._chunker.chunk_text(new_text)

        with self._lock:
            new_chunk_entries: List[tuple] = []

            deleted_set = set(diff_output.to_delete)

            for idx, chunk_text in enumerate(new_chunks):
                chunk_id = self._make_chunk_id(doc_id, idx)
                new_chunk_entries.append((chunk_id, chunk_text))

            upsert_map: Dict[int, np.ndarray] = {}
            for i, target_idx in enumerate(upsert_indices):
                upsert_map[target_idx] = upsert_embeddings[i]

            for idx, (chunk_id, chunk_text) in enumerate(new_chunk_entries):
                if idx in upsert_map:
                    self._retrieval.add_document(
                        chunk_id, chunk_text, upsert_map[idx]
                    )

            self._documents[doc_id] = new_text
            self._chunk_map[doc_id] = new_chunk_entries

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> List[SearchResult]:
        """Search indexed documents across all chunks.

        Args:
            query: Raw query text.
            top_k: Maximum number of results.
            mode: Retrieval mode — ``"keyword"``, ``"vector"``, or ``"hybrid"``.

        Returns:
            List of :class:`SearchResult` sorted by descending relevance.
        """
        query_vector = EmbeddingEngine.embed(query)

        raw_results = self._retrieval.search(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            mode=mode,
        )

        search_results: List[SearchResult] = []
        for chunk_id, score in raw_results:
            parts = str(chunk_id).split("::chunk::")
            doc_id = parts[0] if parts else str(chunk_id)

            chunk_text = ""
            with self._lock:
                for cid, ctxt in self._chunk_map.get(doc_id, []):
                    if cid == chunk_id:
                        chunk_text = ctxt
                        break

            search_results.append(SearchResult(
                doc_id=doc_id,
                chunk_text=chunk_text,
                score=score,
                source=mode,
            ))

        return search_results


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert coding assistant.\n"
    "Use ONLY the following context to answer.\n"
    "If the answer is not in the context, say "
    "'I don't have enough context to answer this.'\n"
)


class RAGPipeline:
    def __init__(
        self,
        cache_capacity: int = 256,
        ollama_base_url: str = _OLLAMA_HOST,
        primary_model: str = _PRIMARY_MODEL,
        fallback_model: str = _FALLBACK_MODEL,
    ) -> None:
        """
        Args:
            cache_capacity: Maximum cached responses before LRU eviction.
            ollama_base_url: Ollama REST API base URL.
            primary_model: Primary Ollama model for generation.
            fallback_model: Fallback Ollama model.
        """
        self._store: DocumentStore = DocumentStore()
        self._cache: LRUCache = LRUCache(capacity=cache_capacity)
        self._llm: OllamaClient = OllamaClient(
            base_url=ollama_base_url,
            primary_model=primary_model,
            fallback_model=fallback_model,
        )

    @property
    def store(self) -> DocumentStore:
        """Underlying document store for index management."""
        return self._store

    @staticmethod
    def _cache_key(query: str, mode: str, top_k: int) -> str:
        """Derive a deterministic cache key from query parameters."""
        raw = f"{query}|{mode}|{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        """Assemble the final LLM prompt from context and question."""
        return (
            f"{_SYSTEM_PROMPT}\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    def query(
        self,
        user_query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> RAGResponse:
        """Execute a full RAG query: retrieve → generate → cache.

        Args:
            user_query: Natural-language question from the user.
            top_k: Number of retrieval results to use as context.
            mode: Retrieval mode — ``"keyword"``, ``"vector"``, or ``"hybrid"``.

        Returns:
            :class:`RAGResponse` containing the answer, sources, timing,
            and cache metadata.

        Raises:
            OllamaUnavailableError: If all Ollama models are unreachable.
        """
        cache_key = self._cache_key(user_query, mode, top_k)

        cached_response = self._cache.get(cache_key)
        if cached_response is not None:
            return RAGResponse(
                answer=cached_response["answer"],
                sources=cached_response["sources"],
                cached=True,
                model_used=cached_response["model_used"],
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
            )

        t0 = time.perf_counter()
        sources = self._store.search(user_query, top_k=top_k, mode=mode)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        context_parts: List[str] = []
        for i, src in enumerate(sources, 1):
            context_parts.append(
                f"[{i}] (doc: {src.doc_id}, score: {src.score:.4f})\n{src.chunk_text}"
            )
        context = "\n\n".join(context_parts) if context_parts else "(no context found)"

        prompt = self._build_prompt(context, user_query)

        t1 = time.perf_counter()
        answer = self._llm.generate(prompt)
        generation_ms = (time.perf_counter() - t1) * 1000.0

        model_used = self._llm.last_model_used

        self._cache.put(
            cache_key,
            {"answer": answer, "sources": sources, "model_used": model_used},
            ttl=_CACHE_TTL,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            cached=False,
            model_used=model_used,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
        )
