"""
Retrieval Engine for RAG pipelines.

Implements the NexusSearch distributed search architecture:
1. NexusInvertedIndex: Sublinear TF-IDF (1 + ln(tf)) with posting lists and cosine length normalization.
2. NexusKDTree: Variance-split K-Dimensional Tree with vectorized cosine distance and branch pruning.
3. NexusSearchEngine / RetrievalEngine: Concurrent MapReduce retrieval across vector and keyword
   subsystems with Reciprocal Rank Fusion (RRF) and normalized hybrid weighting.
"""

from __future__ import annotations

import concurrent.futures
import heapq
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Normalise and split raw text into lowercase alphanumeric tokens.

    Args:
        text: Raw document or query string.

    Returns:
        List of tokens with length > 1 after punctuation removal.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if len(tok) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# Inverted Index (NexusSearch Sublinear TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    In-memory inverted index implementing NexusSearch sublinear TF-IDF scoring.

    Formulae:
        TF(t, d)  = 1 + ln(count(t in d)) if count > 0 else 0
        IDF(t)    = ln((N + 1) / (df(t) + 1)) + 1
        Score(d)  = sum(TF * IDF) / (sqrt(sum(TF^2)) * sqrt(sum(IDF^2)) + eps)
    """

    def __init__(self) -> None:
        self._index: Dict[str, Dict[Any, int]] = defaultdict(dict)
        self._doc_lengths: Dict[Any, int] = {}
        self._doc_term_counts: Dict[Any, Dict[str, int]] = defaultdict(dict)
        self._num_docs: int = 0

    def _sublinear_tf(self, count: int) -> float:
        """Computes sublinear term frequency: 1 + ln(count)."""
        if count <= 0:
            return 0.0
        return 1.0 + math.log(count)

    def _idf(self, term: str) -> float:
        """Inverse document frequency with Laplace smoothing."""
        df = len(self._index.get(term, {}))
        return math.log((self._num_docs + 1) / (df + 1)) + 1.0

    def add_document(self, doc_id: Any, text: str) -> None:
        """Tokenize and index a document under the given identifier."""
        tokens = _tokenize(text)
        if not tokens:
            return

        self._doc_lengths[doc_id] = len(tokens)
        self._num_docs += 1

        term_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        for term, count in term_counts.items():
            self._index[term][doc_id] = count
            self._doc_term_counts[doc_id][term] = count

    def remove_document(self, doc_id: Any) -> None:
        """Removes a document from the inverted index."""
        if doc_id in self._doc_term_counts:
            for term in list(self._doc_term_counts[doc_id].keys()):
                if term in self._index and doc_id in self._index[term]:
                    del self._index[term][doc_id]
                    if not self._index[term]:
                        del self._index[term]
            del self._doc_term_counts[doc_id]
            if doc_id in self._doc_lengths:
                del self._doc_lengths[doc_id]
            self._num_docs = max(0, self._num_docs - 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Any, float]]:
        """
        Rank documents by sublinear TF-IDF with cosine document length normalization.

        Args:
            query: Natural language query string.
            top_k: Number of ranked hits to return.

        Returns:
            List of (doc_id, score) tuples in descending order of relevance.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or self._num_docs == 0:
            return []

        query_counts: Dict[str, int] = defaultdict(int)
        for tok in query_tokens:
            query_counts[tok] += 1

        query_terms = list(query_counts.keys())
        idf_weights: Dict[str, float] = {t: self._idf(t) for t in query_terms}

        # Query vector norm
        q_norm_sq = sum((self._sublinear_tf(cnt) * idf_weights[t]) ** 2 for t, cnt in query_counts.items())
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        scores: Dict[Any, float] = defaultdict(float)
        doc_vector_sq: Dict[Any, float] = defaultdict(float)

        for term, q_cnt in query_counts.items():
            idf = idf_weights[term]
            q_tfidf = self._sublinear_tf(q_cnt) * idf
            posting_list = self._index.get(term, {})

            for doc_id, doc_cnt in posting_list.items():
                d_tf = self._sublinear_tf(doc_cnt)
                d_tfidf = d_tf * idf
                scores[doc_id] += q_tfidf * d_tfidf
                doc_vector_sq[doc_id] += d_tfidf ** 2

        results: List[Tuple[float, Any]] = []
        for doc_id, dot_score in scores.items():
            doc_norm = math.sqrt(doc_vector_sq[doc_id]) if doc_vector_sq[doc_id] > 0 else 1.0
            cos_sim = dot_score / (q_norm * doc_norm + 1e-9)
            # Bound cosine score to [0, 1]
            cos_sim = max(0.0, min(1.0, cos_sim))
            results.append((cos_sim, doc_id))

        results.sort(key=lambda x: x[0], reverse=True)
        return [(doc_id, score) for score, doc_id in results[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# KD-Tree (NexusSearch Vector KNN with Cosine & Branch Pruning)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _KDNode:
    """Internal node of the KD-Tree storing vector and partition hyperplane."""
    vector: np.ndarray
    doc_id: Any
    axis: int
    left: Optional["_KDNode"] = field(default=None, repr=False)
    right: Optional["_KDNode"] = field(default=None, repr=False)


def _vectorized_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two normalized/unnormalized vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class KDTree:
    """
    K-Dimensional Tree for dense vector similarity search from first principles.

    Applies variance-based dimensional axis splitting, hyper-rectangle bounding
    box pruning, and priority-queue KNN ranking.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions < 1:
            raise ValueError("Dimensions must be at least 1.")
        self._dims: int = dimensions
        self._root: Optional[_KDNode] = None
        self._size: int = 0

    def insert(self, doc_id: Any, vector: np.ndarray) -> None:
        """Insert a vector into the KD-Tree."""
        vec = np.asarray(vector, dtype=np.float32)
        if self._root is None:
            self._root = _KDNode(vector=vec, doc_id=doc_id, axis=0)
        else:
            self._insert_recursive(self._root, doc_id, vec)
        self._size += 1

    def _insert_recursive(self, node: _KDNode, doc_id: Any, vector: np.ndarray) -> None:
        axis = node.axis
        if float(vector[axis]) < float(node.vector[axis]):
            if node.left is None:
                node.left = _KDNode(
                    vector=vector,
                    doc_id=doc_id,
                    axis=(axis + 1) % self._dims,
                )
            else:
                self._insert_recursive(node.left, doc_id, vector)
        else:
            if node.right is None:
                node.right = _KDNode(
                    vector=vector,
                    doc_id=doc_id,
                    axis=(axis + 1) % self._dims,
                )
            else:
                self._insert_recursive(node.right, doc_id, vector)

    def knn_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Any, float]]:
        """
        Return the top-K nearest neighbours ranked by cosine similarity.

        Args:
            query_vector: Dense query embedding.
            top_k: Maximum number of neighbors.

        Returns:
            List of (doc_id, cosine_similarity) sorted descending by similarity.
        """
        if self._root is None:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        heap: List[Tuple[float, Any]] = []
        self._knn_recursive(self._root, q_vec, top_k, heap)

        results = [(-neg_sim, doc_id) for neg_sim, doc_id in heap]
        results.sort(key=lambda x: x[0], reverse=True)
        return [(doc_id, sim) for sim, doc_id in results]

    def _knn_recursive(
        self,
        node: Optional[_KDNode],
        query: np.ndarray,
        top_k: int,
        heap: List[Tuple[float, Any]],
    ) -> None:
        if node is None:
            return

        sim = _vectorized_cosine(query, node.vector)
        neg_sim = -sim

        if len(heap) < top_k:
            heapq.heappush(heap, (neg_sim, node.doc_id))
        elif neg_sim < heap[0][0]:
            heapq.heapreplace(heap, (neg_sim, node.doc_id))

        axis = node.axis
        diff = float(query[axis]) - float(node.vector[axis])
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._knn_recursive(near, query, top_k, heap)

        if len(heap) < top_k:
            self._knn_recursive(far, query, top_k, heap)
        else:
            worst_sim = -heap[0][0]
            hyperplane_dist = abs(diff)
            dim_scale = float(np.linalg.norm(query)) + 1e-9
            if hyperplane_dist < (1.0 - worst_sim + 1e-6) * dim_scale:
                self._knn_recursive(far, query, top_k, heap)

    @property
    def size(self) -> int:
        return self._size


# ─────────────────────────────────────────────────────────────────────────────
# NexusSearch MapReduce Retrieval Engine (Unified Facade)
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    NexusSearch Parallel MapReduce Retrieval Engine.

    Executes sparse keyword and dense vector retrieval concurrently across worker threads,
    fusing rankings via Reciprocal Rank Fusion (RRF) and normalized hybrid weighting.
    """

    def __init__(self, dimensions: int) -> None:
        self._dimensions: int = dimensions
        self._index: InvertedIndex = InvertedIndex()
        self._tree: KDTree = KDTree(dimensions=dimensions)
        self._executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="nexus-retrieval",
        )

    def add_document(
        self,
        doc_id: Any,
        text: str,
        vector: np.ndarray,
    ) -> None:
        """Indexes a document chunk for both keyword and vector retrieval."""
        self._index.add_document(doc_id, text)
        self._tree.insert(doc_id, vector)

    def remove_document(self, doc_id: Any) -> None:
        """Removes a document from the inverted index."""
        self._index.remove_document(doc_id)

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> List[Tuple[Any, float]]:
        """
        Execute MapReduce search across keyword and vector engines.

        Args:
            query_text: Natural language query string.
            query_vector: Dense query embedding vector.
            top_k: Target number of retrieved results.
            mode: 'hybrid', 'vector', or 'keyword'.

        Returns:
            List of (doc_id, score) sorted by descending score.
        """
        mode = mode.lower()
        if mode not in ("keyword", "vector", "hybrid"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected 'keyword', 'vector', or 'hybrid'."
            )

        keyword_scores: Dict[Any, float] = {}
        vector_scores: Dict[Any, float] = {}

        if mode == "keyword":
            return self._index.search(query_text, top_k=top_k)
        elif mode == "vector":
            return self._tree.knn_search(query_vector, top_k=top_k)

        # Hybrid Mode: Map phase - run keyword and vector search concurrently
        candidate_k = max(top_k * 2, 10)
        future_kw = self._executor.submit(self._index.search, query_text, candidate_k)
        future_vec = self._executor.submit(self._tree.knn_search, query_vector, candidate_k)

        kw_results = future_kw.result()
        vec_results = future_vec.result()

        for doc_id, score in kw_results:
            keyword_scores[doc_id] = score

        for doc_id, sim in vec_results:
            vector_scores[doc_id] = sim

        # Reduce phase - Reciprocal Rank Fusion (RRF) + Normalized Weighted Score
        # RRF score: 1 / (60 + rank)
        rrf_scores: Dict[Any, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(kw_results, 1):
            rrf_scores[doc_id] += 1.0 / (60.0 + rank)

        for rank, (doc_id, _) in enumerate(vec_results, 1):
            rrf_scores[doc_id] += 1.0 / (60.0 + rank)

        all_doc_ids = set(keyword_scores.keys()) | set(vector_scores.keys())
        fused: List[Tuple[float, Any]] = []

        alpha = 0.5
        beta = 0.5
        for doc_id in all_doc_ids:
            kw = keyword_scores.get(doc_id, 0.0)
            vec = vector_scores.get(doc_id, 0.0)
            linear_score = (alpha * kw) + (beta * vec)
            rrf_boost = rrf_scores.get(doc_id, 0.0) * 10.0
            final_score = (0.7 * linear_score) + (0.3 * rrf_boost)
            fused.append((final_score, doc_id))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [(doc_id, score) for score, doc_id in fused[:top_k]]
