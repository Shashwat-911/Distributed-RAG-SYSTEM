"""
Diff Engine for RAG pipelines.

Provides chunk-aware text differencing optimised for incremental re-embedding
workflows. Documents are split into semantic chunks, aligned via a space-optimised
Hirschberg LCS algorithm, and classified into atomic delta operations.

No dependency on ``difflib`` or any external diff library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Delta:
    """Atomic chunk-level change between two document revisions.

    Attributes:
        kind: Classification of the change — one of
              ``"ADDED"``, ``"DELETED"``, ``"MODIFIED"``, or ``"UNCHANGED"``.
        old_index: Position in the old chunk list (None for ADDED deltas).
        new_index: Position in the new chunk list (None for DELETED deltas).
        old_text: Original chunk content (None for ADDED deltas).
        new_text: Replacement chunk content (None for DELETED deltas).
    """

    kind: str
    old_index: Optional[int] = None
    new_index: Optional[int] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

_MD_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


class Chunker:
    """Semantic text chunker for RAG document processing.

    Splitting strategy (applied in priority order):

    1. **Markdown headers** — if the document contains ATX-style headers
       (``#`` through ``######``), each header begins a new chunk.
    2. **Blank-line paragraphs** — otherwise the text is split on runs
       of two or more consecutive newlines.

    All chunks are CRLF-normalised and whitespace-stripped; empty chunks
    are discarded.
    """

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        """Split *text* into logical chunks.

        Args:
            text: Raw document content (may contain CRLF line endings).

        Returns:
            Ordered list of non-empty, whitespace-stripped chunk strings.
        """
        normalised = "\n".join(
            line.rstrip() for line in text.replace("\r\n", "\n").split("\n")
        )

        if _MD_HEADER_RE.search(normalised):
            segments = _MD_HEADER_RE.split(normalised)
            markers = _MD_HEADER_RE.findall(normalised)

            parts: List[str] = []
            if segments[0].strip():
                parts.append(segments[0].strip())

            for marker, body in zip(markers, segments[1:]):
                chunk = (marker + body).strip()
                if chunk:
                    parts.append(chunk)

            if parts:
                return parts

        paragraphs = re.split(r"\n{2,}", normalised)
        return [p.strip() for p in paragraphs if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Hirschberg LCS
# ─────────────────────────────────────────────────────────────────────────────

class HirschbergLCS:
    """Space-optimised Longest Common Subsequence via Hirschberg's algorithm.

    Recovers the full LCS alignment in O(m·n) time and O(min(m, n)) space
    by combining forward/backward 2-row DP passes with divide-and-conquer
    backtracking. No dependency on ``difflib``.
    """

    @staticmethod
    def _forward_row(a: Sequence, b: Sequence) -> List[int]:
        """Compute the last row of the LCS-length DP table using two rows.

        Args:
            a: First sequence (iterated row-wise).
            b: Second sequence (iterated column-wise).

        Returns:
            List of length ``len(b) + 1`` representing the final DP row.
        """
        n = len(b)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(len(a)):
            for j in range(n):
                if a[i] == b[j]:
                    curr[j + 1] = prev[j] + 1
                else:
                    curr[j + 1] = max(prev[j + 1], curr[j])
            prev, curr = curr, [0] * (n + 1)
        return prev

    @staticmethod
    def _lcs_backtrack(a: Sequence, b: Sequence) -> List[Tuple[int, int]]:
        """Recover aligned index pairs via Hirschberg divide-and-conquer.

        Args:
            a: First sequence (old chunks or characters).
            b: Second sequence (new chunks or characters).

        Returns:
            List of ``(a_index, b_index)`` pairs constituting the LCS.
        """

        def _hirschberg(
            a_seq: Sequence,
            a_off: int,
            b_seq: Sequence,
            b_off: int,
        ) -> List[Tuple[int, int]]:
            m, n = len(a_seq), len(b_seq)

            if m == 0 or n == 0:
                return []

            if m == 1:
                for j, item in enumerate(b_seq):
                    if a_seq[0] == item:
                        return [(a_off, b_off + j)]
                return []

            if n == 1:
                for i, item in enumerate(a_seq):
                    if b_seq[0] == item:
                        return [(a_off + i, b_off)]
                return []

            mid = m // 2
            top = HirschbergLCS._forward_row(a_seq[:mid], b_seq)
            bot = HirschbergLCS._forward_row(a_seq[mid:][::-1], b_seq[::-1])

            best_j = 0
            best_val = -1
            for j in range(n + 1):
                val = top[j] + bot[n - j]
                if val > best_val:
                    best_val = val
                    best_j = j

            left = _hirschberg(a_seq[:mid], a_off, b_seq[:best_j], b_off)
            right = _hirschberg(
                a_seq[mid:], a_off + mid, b_seq[best_j:], b_off + best_j
            )
            return left + right

        return _hirschberg(a, 0, b, 0)

    @staticmethod
    def lcs(a: Sequence, b: Sequence) -> List[Tuple[int, int]]:
        """Compute the LCS alignment between two sequences.

        Args:
            a: First sequence.
            b: Second sequence.

        Returns:
            Ordered list of ``(a_index, b_index)`` aligned pairs.
        """
        return HirschbergLCS._lcs_backtrack(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Delta Extractor
# ─────────────────────────────────────────────────────────────────────────────

class DeltaExtractor:
    """Classifies chunk-level differences using Hirschberg LCS alignment.

    Chunk pairs excluded from the LCS are tested for character-level
    similarity; pairs exceeding *similarity_threshold* are classified as
    ``MODIFIED`` rather than independent ``DELETED`` + ``ADDED`` deltas.

    Attributes:
        similarity_threshold: Minimum character-level LCS ratio (0.0–1.0)
            for two unmatched chunks to be considered a modification of
            the same logical unit.
    """

    def __init__(self, similarity_threshold: float = 0.55) -> None:
        """
        Args:
            similarity_threshold: Character-level similarity cutoff for
                MODIFIED classification. Must be in [0.0, 1.0].
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self.similarity_threshold: float = similarity_threshold

    def _char_similarity(self, a: str, b: str) -> float:
        """Character-level LCS ratio between two strings (case-insensitive).

        Args:
            a: First string.
            b: Second string.

        Returns:
            Ratio of character-level LCS length to the average string length.
        """
        a_low, b_low = a.lower(), b.lower()
        if a_low == b_low:
            return 1.0

        avg_len = (len(a_low) + len(b_low)) / 2.0
        if avg_len == 0:
            return 1.0

        n = len(b_low)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(len(a_low)):
            for j in range(n):
                if a_low[i] == b_low[j]:
                    curr[j + 1] = prev[j] + 1
                else:
                    curr[j + 1] = max(prev[j + 1], curr[j])
            prev, curr = curr, [0] * (n + 1)

        return prev[n] / avg_len

    def extract(
        self,
        old_chunks: List[str],
        new_chunks: List[str],
    ) -> List[Delta]:
        """Compare two ordered chunk lists and classify every difference.

        Args:
            old_chunks: Chunks from the previous document revision.
            new_chunks: Chunks from the current document revision.

        Returns:
            List of :class:`Delta` objects covering all ADDED, DELETED,
            MODIFIED, and UNCHANGED positions.
        """
        matched_pairs = HirschbergLCS.lcs(old_chunks, new_chunks)

        matched_old = {p[0] for p in matched_pairs}
        matched_new = {p[1] for p in matched_pairs}

        anchors = [(-1, -1)] + matched_pairs + [
            (len(old_chunks), len(new_chunks))
        ]

        deltas: List[Delta] = []

        for k in range(len(anchors) - 1):
            old_start = anchors[k][0] + 1
            old_end = anchors[k + 1][0]
            new_start = anchors[k][1] + 1
            new_end = anchors[k + 1][1]

            unmatched_old = list(range(old_start, old_end))
            unmatched_new = list(range(new_start, new_end))

            paired_old: set = set()
            paired_new: set = set()

            for oi in unmatched_old:
                for ni in unmatched_new:
                    if ni in paired_new:
                        continue
                    sim = self._char_similarity(old_chunks[oi], new_chunks[ni])
                    if sim >= self.similarity_threshold:
                        deltas.append(Delta(
                            kind="MODIFIED",
                            old_index=oi,
                            new_index=ni,
                            old_text=old_chunks[oi],
                            new_text=new_chunks[ni],
                        ))
                        paired_old.add(oi)
                        paired_new.add(ni)
                        break

            for oi in unmatched_old:
                if oi not in paired_old:
                    deltas.append(Delta(
                        kind="DELETED",
                        old_index=oi,
                        old_text=old_chunks[oi],
                    ))

            for ni in unmatched_new:
                if ni not in paired_new:
                    deltas.append(Delta(
                        kind="ADDED",
                        new_index=ni,
                        new_text=new_chunks[ni],
                    ))

            if k < len(anchors) - 2:
                match_oi, match_ni = anchors[k + 1]
                if match_oi < len(old_chunks) and match_ni < len(new_chunks):
                    deltas.append(Delta(
                        kind="UNCHANGED",
                        old_index=match_oi,
                        new_index=match_ni,
                        old_text=old_chunks[match_oi],
                        new_text=new_chunks[match_ni],
                    ))

        return deltas


# ─────────────────────────────────────────────────────────────────────────────
# Diff Engine (Unified Facade)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiffOutput:
    """Structured result from :meth:`DiffEngine.diff_texts`.

    Attributes:
        to_upsert: Deltas with kind ``ADDED`` or ``MODIFIED`` — chunks
            that require new embedding generation.
        to_delete: Old-chunk indices with kind ``DELETED`` — entries to
            remove from the vector index.
    """

    to_upsert: List[Delta]
    to_delete: List[int]


class DiffEngine:
    """End-to-end diff facade combining chunking and delta extraction.

    Wraps :class:`Chunker` and :class:`DeltaExtractor` to provide a single
    method that accepts raw document text and returns the minimal set of
    index mutations required for an incremental re-embedding pipeline.

    Attributes:
        _chunker: Chunker instance for text segmentation.
        _extractor: DeltaExtractor instance for chunk alignment and classification.
    """

    def __init__(self, similarity_threshold: float = 0.55) -> None:
        """
        Args:
            similarity_threshold: Passed through to :class:`DeltaExtractor`.
        """
        self._chunker: Chunker = Chunker()
        self._extractor: DeltaExtractor = DeltaExtractor(
            similarity_threshold=similarity_threshold
        )

    def diff_texts(self, old_text: str, new_text: str) -> DiffOutput:
        """Chunk both texts, compute deltas, and partition into upsert/delete sets.

        Args:
            old_text: Previous document revision (raw text).
            new_text: Current document revision (raw text).

        Returns:
            :class:`DiffOutput` containing the ADDED + MODIFIED deltas
            (chunks needing re-embedding) and DELETED old-chunk indices
            (entries to purge from the vector index).
        """
        old_chunks = self._chunker.chunk_text(old_text)
        new_chunks = self._chunker.chunk_text(new_text)

        all_deltas = self._extractor.extract(old_chunks, new_chunks)

        to_upsert = [
            d for d in all_deltas if d.kind in ("ADDED", "MODIFIED")
        ]
        to_delete = [
            d.old_index
            for d in all_deltas
            if d.kind == "DELETED" and d.old_index is not None
        ]

        return DiffOutput(to_upsert=to_upsert, to_delete=to_delete)
