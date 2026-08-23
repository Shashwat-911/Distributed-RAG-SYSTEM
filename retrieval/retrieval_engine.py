"""
Retrieval Engine for RAG pipelines.

Provides keyword retrieval via a from-scratch TF-IDF inverted index, dense vector
retrieval via a variance-split KD-Tree with cosine ranking, and a hybrid mode that
fuses both signal types through weighted score combination.

Scoring Formulae (InvertedIndex)
--------------------------------
    TF(t, d)  = count(t in d) / len(d)
    IDF(t)    = log((N + 1) / (df(t) + 1)) + 1
    Score(d)  = sum(TF * IDF for each query term) / L2_norm(d)

Distance Metrics (KDTree)
-------------------------
    Ranking:  Cosine similarity  (higher = more similar)
    Pruning:  Euclidean distance along the splitting hyperplane
"""

from __future__ import annotations

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
# Inverted Index (TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    In-memory inverted index with TF-IDF scoring and cosine-normalised ranking.

    Maintains term → {doc_id: raw_count} posting lists and computes TF-IDF
    weights at query time with L2-normalised document scores.

    Attributes:
        _index: Posting lists mapping term → {doc_id → raw term count}.
        _doc_lengths: Maps doc_id → total token count for TF normalisation.
        _num_docs: Total number of indexed documents (corpus size N).
    """

    def __init__(self) -> None:
        self._index: Dict[str, Dict[Any, int]] = defaultdict(dict)
        self._doc_lengths: Dict[Any, int] = {}
        self._num_docs: int = 0

    def _tf(self, raw_count: int, doc_length: int) -> float:
        """Normalised term frequency: count(t in d) / len(d)."""
        if doc_length == 0:
            return 0.0
        return raw_count / doc_length

    def _idf(self, term: str) -> float:
        """
        Inverse document frequency with Laplace smoothing.

        IDF(t) = log((N + 1) / (df(t) + 1)) + 1
        """
        df = len(self._index.get(term, {}))
        return math.log((self._num_docs + 1) / (df + 1)) + 1.0

    def _compute_doc_norm(
        self,
        doc_id: Any,
        query_terms: List[str],
        idf_weights: Dict[str, float],
    ) -> float:
        """
        L2 norm of a document's TF-IDF vector projected onto query term dimensions.

        Args:
            doc_id: Document identifier.
            query_terms: Unique terms from the query.
            idf_weights: Pre-computed IDF values keyed by term.

        Returns:
            Scalar L2 norm used for cosine normalisation.
        """
        norm_sq = 0.0
        doc_length = self._doc_lengths.get(doc_id, 1)
        for term in query_terms:
            raw_count = self._index.get(term, {}).get(doc_id, 0)
            tf = self._tf(raw_count, doc_length)
            tfidf = tf * idf_weights[term]
            norm_sq += tfidf ** 2
        return math.sqrt(norm_sq)

    def add_document(self, doc_id: Any, text: str) -> None:
        """
        Tokenize and index a document under the given identifier.

        Args:
            doc_id: Unique document identifier (hashable).
            text: Raw document content.
        """
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

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Any, float]]:
        """
        Rank documents by cosine-normalised TF-IDF relevance to the query.

        Args:
            query: Human-readable search query.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        query_terms = list(set(_tokenize(query)))
        if not query_terms or self._num_docs == 0:
            return []

        scores: Dict[Any, float] = defaultdict(float)
        idf_weights: Dict[str, float] = {t: self._idf(t) for t in query_terms}

        for term in query_terms:
            idf = idf_weights[term]
            posting_list = self._index.get(term, {})
            for doc_id, raw_count in posting_list.items():
                doc_length = self._doc_lengths.get(doc_id, 1)
                tf = self._tf(raw_count, doc_length)
                scores[doc_id] += tf * idf

        normalised: List[Tuple[float, Any]] = []
        for doc_id, score in scores.items():
            doc_norm = self._compute_doc_norm(doc_id, query_terms, idf_weights)
            final = score / doc_norm if doc_norm > 0 else score
            normalised.append((final, doc_id))

        normalised.sort(key=lambda x: x[0], reverse=True)

        return [(doc_id, score) for score, doc_id in normalised[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# KD-Tree (Vector KNN)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _KDNode:
    """Internal node of the KD-Tree storing a single vector and its metadata."""

    vector: np.ndarray
    doc_id: Any
    axis: int
    left: Optional[_KDNode] = field(default=None, repr=False)
    right: Optional[_KDNode] = field(default=None, repr=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors in [-1, 1].

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _euclidean_distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance for pruning comparisons (avoids sqrt)."""
    diff = a - b
    return float(np.dot(diff, diff))


class KDTree:
    """
    K-Dimensional Tree for nearest-neighbour retrieval over dense vectors.

    Construction splits on the axis of maximum variance at each level for
    balanced partitioning. Queries rank results by cosine similarity while
    using Euclidean hyperplane distance for branch pruning.

    Attributes:
        _dims: Dimensionality of stored vectors.
        _root: Root node of the tree, or None if empty.
        _size: Total number of vectors stored.
    """

    def __init__(self, dimensions: int) -> None:
        """
        Args:
            dimensions: Fixed dimensionality of all vectors in this tree.
        """
        if dimensions < 1:
            raise ValueError("Dimensions must be at least 1.")
        self._dims: int = dimensions
        self._root: Optional[_KDNode] = None
        self._size: int = 0

    def _build_recursive(
        self,
        records: List[Tuple[Any, np.ndarray]],
    ) -> Optional[_KDNode]:
        """
        Recursively partition records into a balanced subtree.

        Splitting axis is chosen as the dimension with maximum variance
        across the current record set.

        Args:
            records: List of (doc_id, vector) tuples.

        Returns:
            Root _KDNode of the constructed subtree.
        """
        if not records:
            return None

        vecs = np.stack([r[1] for r in records])
        axis = int(np.argmax(np.var(vecs, axis=0)))

        records_sorted = sorted(records, key=lambda r: float(r[1][axis]))
        median_idx = len(records_sorted) // 2
        pivot_id, pivot_vec = records_sorted[median_idx]

        node = _KDNode(vector=pivot_vec, doc_id=pivot_id, axis=axis)
        node.left = self._build_recursive(records_sorted[:median_idx])
        node.right = self._build_recursive(records_sorted[median_idx + 1:])

        return node

    def insert(self, doc_id: Any, vector: np.ndarray) -> None:
        """
        Insert a single vector into the tree without rebalancing.

        Args:
            doc_id: Document identifier associated with this vector.
            vector: Dense float vector of shape (dimensions,).
        """
        if self._root is None:
            self._root = _KDNode(vector=vector, doc_id=doc_id, axis=0)
        else:
            self._insert_recursive(self._root, doc_id, vector)
        self._size += 1

    def _insert_recursive(
        self,
        node: _KDNode,
        doc_id: Any,
        vector: np.ndarray,
    ) -> None:
        """Descend to the correct leaf position and attach a new node."""
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

        Uses depth-first traversal with Euclidean hyperplane pruning and
        a min-heap (of negated cosine similarities) to track the best candidates.

        Args:
            query_vector: Dense float vector of shape (dimensions,).
            top_k: Number of neighbours to return.

        Returns:
            List of (doc_id, cosine_similarity) sorted descending by similarity.
        """
        if self._root is None:
            return []

        heap: List[Tuple[float, Any]] = []
        self._knn_recursive(self._root, query_vector, top_k, heap)

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
        """
        Recursive KNN traversal with cosine ranking and Euclidean pruning.

        Args:
            node: Current tree node.
            query: Query vector.
            top_k: Desired result count.
            heap: Shared min-heap of (-cosine_sim, doc_id).
        """
        if node is None:
            return

        sim = _cosine_similarity(query, node.vector)
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


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Engine (Unified Facade)
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Unified retrieval facade combining keyword (TF-IDF) and vector (KD-Tree) search.

    Supports three retrieval modes:
        - ``"keyword"``:  TF-IDF inverted index search only.
        - ``"vector"``:   KD-Tree cosine similarity search only.
        - ``"hybrid"``:   Weighted linear combination of both scores
                          (alpha=0.5 keyword + beta=0.5 vector).

    Attributes:
        _index: InvertedIndex instance for keyword retrieval.
        _tree: KDTree instance for vector retrieval.
        _dimensions: Embedding dimensionality.
    """

    def __init__(self, dimensions: int) -> None:
        """
        Args:
            dimensions: Fixed dimensionality of document embedding vectors.
        """
        self._dimensions: int = dimensions
        self._index: InvertedIndex = InvertedIndex()
        self._tree: KDTree = KDTree(dimensions=dimensions)

    def add_document(
        self,
        doc_id: Any,
        text: str,
        vector: np.ndarray,
    ) -> None:
        """
        Index a document for both keyword and vector retrieval.

        Args:
            doc_id: Unique document identifier (hashable).
            text: Raw document text for TF-IDF indexing.
            vector: Dense embedding vector of shape (dimensions,).
        """
        self._index.add_document(doc_id, text)
        self._tree.insert(doc_id, vector)

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> List[Tuple[Any, float]]:
        """
        Execute a retrieval query in the specified mode.

        Args:
            query_text: Raw query string (used in keyword and hybrid modes).
            query_vector: Dense query embedding (used in vector and hybrid modes).
            top_k: Maximum number of results.
            mode: One of ``"keyword"``, ``"vector"``, or ``"hybrid"``.

        Returns:
            List of (doc_id, score) tuples sorted by descending relevance.

        Raises:
            ValueError: If mode is not one of the supported values.
        """
        mode = mode.lower()
        if mode not in ("keyword", "vector", "hybrid"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected 'keyword', 'vector', or 'hybrid'."
            )

        keyword_scores: Dict[Any, float] = {}
        vector_scores: Dict[Any, float] = {}

        if mode in ("keyword", "hybrid"):
            for doc_id, score in self._index.search(query_text, top_k=top_k * 2):
                keyword_scores[doc_id] = score

        if mode in ("vector", "hybrid"):
            for doc_id, sim in self._tree.knn_search(query_vector, top_k=top_k * 2):
                vector_scores[doc_id] = sim

        all_doc_ids = set(keyword_scores) | set(vector_scores)
        fused: List[Tuple[float, Any]] = []

        for doc_id in all_doc_ids:
            kw = keyword_scores.get(doc_id, 0.0)
            vec = vector_scores.get(doc_id, 0.0)

            if mode == "keyword":
                final = kw
            elif mode == "vector":
                final = vec
            else:
                alpha = 0.5
                beta = 0.5
                final = alpha * kw + beta * vec

            fused.append((final, doc_id))

        fused.sort(key=lambda x: x[0], reverse=True)

        return [(doc_id, score) for score, doc_id in fused[:top_k]]
