"""
Dense Semantic Retrieval Module for GST Grounded RAG Assistant.

Implements a dense vector retriever utilizing sentence-transformers
with the configured bi-encoder embedding model (BAAI/bge-small-en-v1.5).
Provides L2-normalized embeddings, exact cosine similarity search,
deterministic tie-breaking, and index persistence.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config.settings import (
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_DIR,
)
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import SearchResult

logger = logging.getLogger(__name__)

# Default persistence path for serialized dense vector index
DENSE_INDEX_PATH: Path = INDEX_DIR / "dense_index.pkl"


# ==============================================================================
# 1. Embedding Model Loader & Cache
# ==============================================================================

_MODEL_CACHE: Dict[str, Any] = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> Any:
    """
    Load or retrieve cached SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model identifier or local directory path.

    Returns:
        SentenceTransformer instance.
    """
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
        except ImportError as e:
            logger.error(
                "sentence_transformers is not installed. Please install sentence-transformers."
            )
            raise ImportError(
                "sentence_transformers is required for dense retrieval."
            ) from e
    return _MODEL_CACHE[model_name]


# ==============================================================================
# 2. Dense Vector Retriever
# ==============================================================================

class DenseRetriever:
    """
    Dense semantic retriever operating on Chunk embeddings.

    Maintains an L2-normalized numpy embedding matrix of shape (N, dim)
    and computes cosine similarity via matrix-vector multiplication.
    """

    def __init__(
        self,
        chunks: Optional[List[Chunk]] = None,
        embeddings: Optional[np.ndarray] = None,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        normalize: bool = True,
    ) -> None:
        """
        Initialize DenseRetriever.

        Args:
            chunks: List of Chunk objects.
            embeddings: Optional precomputed numpy embedding matrix (shape: (N, dim)).
            model_name: Name of the embedding model.
            dimension: Expected embedding dimensionality.
            normalize: Whether to L2-normalize vectors for cosine similarity.
        """
        self.model_name = model_name
        self.dimension = dimension
        self.normalize = normalize
        self.chunks: List[Chunk] = chunks or []
        self.corpus_size: int = len(self.chunks)

        if embeddings is not None:
            if len(embeddings) != self.corpus_size:
                raise ValueError(
                    f"Embeddings count ({len(embeddings)}) does not match chunks count ({self.corpus_size})"
                )
            self.embeddings: np.ndarray = np.asarray(embeddings, dtype=np.float32)
        elif self.corpus_size > 0:
            self.embeddings = self._encode_chunks(self.chunks)
        else:
            self.embeddings = np.empty((0, self.dimension), dtype=np.float32)

    def _encode_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Compute normalized embeddings for a list of Chunk objects.

        Normalization:
          All chunk embeddings are L2-normalized at indexing time so that
          inner product directly computes cosine similarity.
        """
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = get_embedding_model(self.model_name)
        texts = [c.text for c in chunks]

        # sentence-transformers encode with normalize_embeddings=True
        embeddings = model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Compute normalized embedding for a search query.

        Normalization:
          Query embeddings are L2-normalized at search time.
        """
        model = get_embedding_model(self.model_name)
        # BGE models benefit from standard text encoding
        query_embedding = model.encode(
            query,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(query_embedding, dtype=np.float32)

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        normalize: bool = True,
    ) -> DenseRetriever:
        """
        Factory method to construct, encode, and index a collection of Chunk objects.

        Args:
            chunks: List of Chunk objects.
            model_name: Bi-encoder embedding model name.
            dimension: Expected embedding dimensionality.
            normalize: Whether to apply L2 normalization.

        Returns:
            Configured DenseRetriever with populated embedding matrix.
        """
        return cls(
            chunks=list(chunks),
            model_name=model_name,
            dimension=dimension,
            normalize=normalize,
        )

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
    ) -> List[SearchResult]:
        """
        Search for the top-k most semantically similar chunks using cosine similarity.

        Ranking is deterministic:
          - Highest similarity score first.
          - Deterministic tie-breaker: chunk_id ascending, then document index.

        Args:
            query: Natural language query string.
            top_k: Maximum number of ranked results to return (default: DENSE_TOP_K).

        Returns:
            List of SearchResult objects sorted by descending similarity rank.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if self.corpus_size == 0 or top_k <= 0:
            return []

        # 1. Encode query
        query_vector = self._encode_query(query.strip())

        # 2. Compute cosine similarity via dot product on L2-normalized vectors
        if self.normalize:
            scores = np.dot(self.embeddings, query_vector)
        else:
            # Explicit cosine similarity fallback if unnormalized
            norm_chunks = np.linalg.norm(self.embeddings, axis=1)
            norm_query = np.linalg.norm(query_vector)
            denom = norm_chunks * norm_query
            denom = np.where(denom == 0, 1e-10, denom)
            scores = np.dot(self.embeddings, query_vector) / denom

        # 3. Deterministic sorting: (-score, chunk_id, idx)
        candidate_indices = list(range(self.corpus_size))
        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: (-float(scores[idx]), self.chunks[idx].chunk_id, idx),
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

    def save(self, path: Union[str, Path] = DENSE_INDEX_PATH) -> None:
        """
        Serialize and persist the dense retriever index and chunk metadata to disk.

        Args:
            path: Target file path (defaults to DENSE_INDEX_PATH in settings.INDEX_DIR).
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "1.0",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "corpus_size": self.corpus_size,
            "embeddings": self.embeddings,
            "chunks": self.chunks,
        }

        with open(target_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved DenseRetriever index with {self.corpus_size} vectors to {target_path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path] = DENSE_INDEX_PATH,
        model_name: Optional[str] = None,
    ) -> DenseRetriever:
        """
        Load a persisted DenseRetriever index from disk.

        Args:
            path: Source file path.
            model_name: Optional override for the embedding model.

        Returns:
            Restored DenseRetriever instance.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Dense index file not found at: {source_path}")

        with open(source_path, "rb") as f:
            payload = pickle.load(f)

        retriever = cls(
            chunks=payload.get("chunks", []),
            embeddings=payload.get("embeddings"),
            model_name=model_name or payload.get("model_name", EMBEDDING_MODEL),
            dimension=payload.get("dimension", EMBEDDING_DIM),
            normalize=payload.get("normalize", True),
        )

        logger.info(f"Loaded DenseRetriever index with {retriever.corpus_size} vectors from {source_path}")
        return retriever
