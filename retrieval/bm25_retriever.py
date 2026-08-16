"""
BM25 Lexical Retrieval Module for GST Grounded RAG Assistant.

Implements a deterministic, pure-Python BM25Okapi inverted index with
domain-specialized tokenization for legal GST / CBIC circulars.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import BM25_INDEX_PATH, BM25_TOP_K
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Search Result Data Structure
# ==============================================================================

@dataclass
class SearchResult:
    """
    Represents a scored, ranked chunk retrieved from the BM25 index.
    """
    chunk: Chunk
    score: float
    rank: int  # 1-indexed rank position


# ==============================================================================
# 2. Domain-Specialized Legal Tokenizer
# ==============================================================================

def tokenize(text: str) -> List[str]:
    """
    Deterministic tokenizer tailored for legal and tax regulatory texts.

    Preserves domain-critical tokens:
      - Circular citation numbers: '172/04/2022-gst', '183/15/2022-gst'
      - Statutory sections / rules with parens: '17(5)', '36(4)', '16(2)(c)', '54(14)'
      - Form identifiers & hyphenated terms: 'gstr-3b', 'gstr-2a', '2018-19', 'inter-state'
      - Standard alphanumeric words & numbers: 'itc', 'rule', '42', 'refund'

    Also emits sub-parts for composite tokens so that queries like 'Rule 36' or 'Section 17'
    match composite targets '36(4)' and '17(5)'.
    """
    if not text:
        return []

    text_lower = text.lower()

    # Pattern for primary composite tokens
    pattern = r"[a-z0-9]+(?:/[a-z0-9]+)*(?:-[a-z0-9]+)*(?:\([a-z0-9]+\))*|[a-z0-9]+"
    primary_tokens = re.findall(pattern, text_lower)

    tokens: List[str] = []
    for token in primary_tokens:
        tokens.append(token)
        # For composite tokens with parens, slashes, or hyphens, also emit subparts
        if "(" in token or "-" in token or "/" in token:
            subparts = re.findall(r"[a-z0-9]+", token)
            for sp in subparts:
                if sp and sp != token:
                    tokens.append(sp)

    return tokens


# ==============================================================================
# 3. BM25 Retriever
# ==============================================================================

class BM25Retriever:
    """
    Inverted-index BM25Okapi retriever for Chunk objects.

    Maintains document frequencies, inverted postings, and length normalization
    parameters to provide fast, deterministic keyword search.
    """

    def __init__(
        self,
        chunks: Optional[List[Chunk]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Initialize the BM25 retriever.

        Args:
            chunks: List of finalized Chunk objects to index.
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = chunks or []
        self.corpus_size: int = len(self.chunks)
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        if self.chunks:
            self._build_index()

    def _build_index(self) -> None:
        """Construct the inverted index and compute IDFs across all chunks."""
        self.corpus_size = len(self.chunks)
        if self.corpus_size == 0:
            self.doc_lengths = []
            self.avgdl = 0.0
            self.idf = {}
            self.inverted_index = defaultdict(list)
            return

        # 1. Tokenize all document bodies
        doc_tokens_list: List[List[str]] = []
        total_tokens = 0

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            doc_tokens_list.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

        self.avgdl = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0

        # 2. Compute document frequencies and build inverted index postings
        df: Dict[str, int] = defaultdict(int)
        for doc_idx, tokens in enumerate(doc_tokens_list):
            counts = Counter(tokens)
            for term, tf in counts.items():
                df[term] += 1
                self.inverted_index[term].append((doc_idx, tf))

        # 3. Compute Robertson BM25 IDF with strictly positive floor:
        # IDF(q) = ln( 1 + (N - df + 0.5) / (df + 0.5) )
        for term, doc_freq in df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """
        Factory method to construct and build a BM25Retriever from Chunk objects.

        Args:
            chunks: List of Chunk objects to index.
            k1: BM25 k1 hyperparameter.
            b: BM25 b hyperparameter.

        Returns:
            Configured and indexed BM25Retriever instance.
        """
        return cls(chunks=list(chunks), k1=k1, b=b)

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> List[SearchResult]:
        """
        Retrieve top-k relevant chunks matching the query using BM25 scoring.

        Ranking is deterministic:
          - Highest BM25 score first.
          - Tie-breaker: chunk_id ascending.

        Args:
            query: User search query string.
            top_k: Maximum number of candidates to return (default: BM25_TOP_K).

        Returns:
            List of SearchResult objects sorted by rank.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if self.corpus_size == 0 or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Accumulate BM25 scores across matching terms
        scores: Dict[int, float] = defaultdict(float)
        query_counts = Counter(query_tokens)

        for term, q_tf in query_counts.items():
            if term not in self.idf:
                continue

            idf_val = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, doc_tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = doc_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0))
                term_score = idf_val * ((doc_tf * (self.k1 + 1.0)) / denom)
                scores[doc_idx] += term_score * q_tf

        if not scores:
            return []

        # Deterministic sorting: (-score, chunk_id, doc_idx)
        candidate_indices = [idx for idx, s in scores.items() if s > 0.0]
        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: (-scores[idx], self.chunks[idx].chunk_id, idx),
        )

        limit = min(top_k, len(ranked_indices))
        results: List[SearchResult] = []

        for rank, idx in enumerate(ranked_indices[:limit], start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results

    def save(self, path: Union[str, Path] = BM25_INDEX_PATH) -> None:
        """
        Serialize and persist the BM25 index and all chunk metadata to disk.

        Args:
            path: Destination file path (defaults to settings.BM25_INDEX_PATH).
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "idf": self.idf,
            "inverted_index": dict(self.inverted_index),
            "chunks": self.chunks,
        }

        with open(target_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved BM25 index with {self.corpus_size} chunks to {target_path}")

    @classmethod
    def load(cls, path: Union[str, Path] = BM25_INDEX_PATH) -> BM25Retriever:
        """
        Load a persisted BM25 index from disk.

        Args:
            path: Source file path (defaults to settings.BM25_INDEX_PATH).

        Returns:
            Restored BM25Retriever instance.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"BM25 index file not found at: {source_path}")

        with open(source_path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            chunks=payload.get("chunks", []),
            k1=payload.get("k1", 1.5),
            b=payload.get("b", 0.75),
        )
        instance.corpus_size = payload.get("corpus_size", len(instance.chunks))
        instance.avgdl = payload.get("avgdl", 0.0)
        instance.doc_lengths = payload.get("doc_lengths", [])
        instance.idf = payload.get("idf", {})
        instance.inverted_index = defaultdict(list, payload.get("inverted_index", {}))

        logger.info(f"Loaded BM25 index with {instance.corpus_size} chunks from {source_path}")
        return instance
