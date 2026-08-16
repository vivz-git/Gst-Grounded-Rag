"""
Hybrid Retrieval Module for GST Grounded RAG Assistant.

Combines BM25 lexical retrieval and dense semantic retrieval via min-max
score normalization and weighted linear combination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import (
    BM25_INDEX_PATH,
    BM25_TOP_K,
    BM25_WEIGHT,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    HYBRID_TOP_K,
    INDEX_DIR,
)
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import BM25Retriever, SearchResult
from retrieval.dense_retriever import DENSE_INDEX_PATH, DenseRetriever

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Hybrid Search Result Data Structure
# ==============================================================================

@dataclass
class HybridSearchResult:
    """
    Represents a ranked result produced by combining BM25 and dense retrievers.
    """
    chunk: Chunk
    score: float  # Combined hybrid score
    rank: int  # 1-indexed rank position
    retrieval_sources: List[str] = field(default_factory=list)  # e.g., ["bm25"], ["dense"], or ["bm25", "dense"]
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    normalized_bm25_score: float = 0.0
    normalized_dense_score: float = 0.0


# ==============================================================================
# 2. Score Normalization Helper
# ==============================================================================

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalize a dictionary of {chunk_id: score} to [0.0, 1.0].

    Edge cases:
      - Empty dictionary -> returns {}
      - All scores identical (or single candidate) -> assigns 1.0 if score > 0 else 0.0
        to prevent division-by-zero while preserving positive relevance.
    """
    if not scores:
        return {}

    vals = list(scores.values())
    min_val = min(vals)
    max_val = max(vals)

    if max_val == min_val:
        neutral_val = 1.0 if max_val > 0.0 else 0.0
        return {k: neutral_val for k in scores}

    denom = max_val - min_val
    return {k: (v - min_val) / denom for k, v in scores.items()}


# ==============================================================================
# 3. Hybrid Retriever
# ==============================================================================

class HybridRetriever:
    """
    Hybrid retriever combining BM25 lexical search and dense semantic search.

    Architecture:
      1. Retrieve top candidates independently from BM25 and Dense.
      2. Min-max normalize candidate scores for each retriever.
      3. Fuse scores via weighted linear combination:
           score = BM25_WEIGHT * norm_bm25 + DENSE_WEIGHT * norm_dense
      4. Sort descending with deterministic tie-breaking on chunk_id.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
    ) -> None:
        """
        Initialize HybridRetriever.

        Args:
            bm25_retriever: Configured BM25Retriever instance.
            dense_retriever: Configured DenseRetriever instance.
            bm25_weight: Weight assigned to normalized BM25 score (default: 0.5).
            dense_weight: Weight assigned to normalized dense score (default: 0.5).
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    @property
    def corpus_size(self) -> int:
        """Return the corpus size from the underlying retrievers."""
        return max(self.bm25_retriever.corpus_size, self.dense_retriever.corpus_size)

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
    ) -> HybridRetriever:
        """
        Construct and build both BM25 and Dense indices over a list of chunks.

        Args:
            chunks: List of Chunk objects.
            bm25_weight: Weight for BM25 score.
            dense_weight: Weight for dense score.

        Returns:
            Configured HybridRetriever instance.
        """
        bm25 = BM25Retriever.build(chunks)
        dense = DenseRetriever.build(chunks)
        return cls(
            bm25_retriever=bm25,
            dense_retriever=dense,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def search(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K,
        candidate_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid retrieval combining BM25 and dense semantic search.

        Args:
            query: Search query string.
            top_k: Maximum number of final hybrid results to return (default: HYBRID_TOP_K).
            candidate_k: Number of candidates to fetch from each retriever before fusion
                         (defaults to max(top_k, BM25_TOP_K, DENSE_TOP_K)).

        Returns:
            List of HybridSearchResult objects sorted by combined score descending.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if self.corpus_size == 0 or top_k <= 0:
            return []

        # Determine candidate retrieval depth
        if candidate_k is None:
            candidate_k = max(top_k, BM25_TOP_K, DENSE_TOP_K)

        # 1. Retrieve candidates from individual retrievers
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)

        if not bm25_results and not dense_results:
            return []

        # 2. Extract raw score maps and chunk references
        bm25_raw_scores: Dict[str, float] = {}
        dense_raw_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        for r in bm25_results:
            cid = r.chunk.chunk_id
            bm25_raw_scores[cid] = r.score
            chunk_map[cid] = r.chunk

        for r in dense_results:
            cid = r.chunk.chunk_id
            dense_raw_scores[cid] = r.score
            chunk_map[cid] = r.chunk

        # 3. Min-Max normalize scores per retriever
        bm25_norm_scores = normalize_scores(bm25_raw_scores)
        dense_norm_scores = normalize_scores(dense_raw_scores)

        # 4. Combine scores and build candidate results
        all_chunk_ids = sorted(chunk_map.keys())
        candidates: List[HybridSearchResult] = []

        for cid in all_chunk_ids:
            chunk = chunk_map[cid]
            sources: List[str] = []
            
            raw_bm25 = bm25_raw_scores.get(cid)
            norm_bm25 = bm25_norm_scores.get(cid, 0.0)
            if raw_bm25 is not None:
                sources.append("bm25")

            raw_dense = dense_raw_scores.get(cid)
            norm_dense = dense_norm_scores.get(cid, 0.0)
            if raw_dense is not None:
                sources.append("dense")

            combined_score = (
                self.bm25_weight * norm_bm25
                + self.dense_weight * norm_dense
            )

            candidates.append(
                HybridSearchResult(
                    chunk=chunk,
                    score=float(combined_score),
                    rank=0,  # assigned after sorting
                    retrieval_sources=sources,
                    bm25_score=raw_bm25,
                    dense_score=raw_dense,
                    normalized_bm25_score=float(norm_bm25),
                    normalized_dense_score=float(norm_dense),
                )
            )

        # 5. Deterministic sorting: combined score descending, chunk_id ascending
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (-item.score, item.chunk.chunk_id),
        )

        # 6. Apply top_k cutoff and assign 1-indexed ranks
        limit = min(top_k, len(ranked_candidates))
        final_results: List[HybridSearchResult] = []

        for rank_idx, item in enumerate(ranked_candidates[:limit], start=1):
            item.rank = rank_idx
            final_results.append(item)

        return final_results

    def save(
        self,
        bm25_path: Union[str, Path] = BM25_INDEX_PATH,
        dense_path: Union[str, Path] = DENSE_INDEX_PATH,
    ) -> None:
        """
        Persist both BM25 and dense indices to disk.

        Args:
            bm25_path: Path for BM25 index file.
            dense_path: Path for dense index file.
        """
        self.bm25_retriever.save(bm25_path)
        self.dense_retriever.save(dense_path)
        logger.info(f"Saved HybridRetriever indices to {bm25_path} and {dense_path}")

    @classmethod
    def load(
        cls,
        bm25_path: Union[str, Path] = BM25_INDEX_PATH,
        dense_path: Union[str, Path] = DENSE_INDEX_PATH,
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
    ) -> HybridRetriever:
        """
        Load persisted BM25 and dense indices from disk.

        Args:
            bm25_path: Path to BM25 index file.
            dense_path: Path to dense index file.
            bm25_weight: Weight for BM25 score.
            dense_weight: Weight for dense score.

        Returns:
            Restored HybridRetriever instance.
        """
        bm25 = BM25Retriever.load(bm25_path)
        dense = DenseRetriever.load(dense_path)
        return cls(
            bm25_retriever=bm25,
            dense_retriever=dense,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
