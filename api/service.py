"""
RAG Service Orchestration Layer for GST Grounded Assistant.

Coordinates Hybrid Retrieval and Gemini Grounded Answer Generation.
Maintains persistent retriever and generator instances across API requests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from config.settings import (
    BM25_INDEX_PATH,
    DENSE_INDEX_PATH,
    HYBRID_TOP_K,
)
from generation.gemini_generator import GeminiGenerator, GroundedAnswer
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGService:
    """
    Application service encapsulating GST RAG business logic.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[GeminiGenerator] = None,
        top_k: int = HYBRID_TOP_K,
    ) -> None:
        """
        Initialize RAGService with retriever and generator instances.

        Args:
            retriever: Configured HybridRetriever instance.
            generator: Configured GeminiGenerator instance.
            top_k: Number of hybrid candidates to retrieve for LLM context.
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    @classmethod
    def from_persisted_indices(
        cls,
        bm25_path: Union[str, Path] = BM25_INDEX_PATH,
        dense_path: Union[str, Path] = DENSE_INDEX_PATH,
        generator: Optional[GeminiGenerator] = None,
        top_k: int = HYBRID_TOP_K,
    ) -> RAGService:
        """
        Factory method to initialize RAGService from precomputed on-disk indices.

        Args:
            bm25_path: Path to serialized BM25 index.
            dense_path: Path to serialized Dense vector index.
            generator: Optional GeminiGenerator instance (defaults to standard GeminiGenerator).
            top_k: Number of hybrid candidates to retrieve.

        Returns:
            Configured RAGService instance.

        Raises:
            FileNotFoundError: If any required retrieval index file is missing.
        """
        b_path = Path(bm25_path)
        d_path = Path(dense_path)

        missing = []
        if not b_path.exists():
            missing.append(f"BM25 index at {b_path}")
        if not d_path.exists():
            missing.append(f"Dense index at {d_path}")

        if missing:
            err_msg = (
                f"Cannot initialize RAGService. Missing required index files: {', '.join(missing)}. "
                f"Please ensure ingestion and indexing are complete before launching the API."
            )
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"Loading HybridRetriever from {b_path} and {d_path}...")
        retriever = HybridRetriever.load(bm25_path=b_path, dense_path=d_path)
        gen = generator if generator is not None else GeminiGenerator()

        return cls(retriever=retriever, generator=gen, top_k=top_k)

    def ask(self, question: str) -> GroundedAnswer:
        """
        Process a user question through the Grounded RAG pipeline.

        Workflow:
          1. Retrieve top-K relevant chunks via HybridRetriever.
          2. Generate grounded citation-backed answer via GeminiGenerator.

        Args:
            question: Cleaned user question string.

        Returns:
            GroundedAnswer dataclass containing answer, confidence label, and source citations.

        Raises:
            RuntimeError: If service is not properly initialized or generation fails.
        """
        if self.retriever is None:
            raise RuntimeError("RAGService is not initialized with a retriever.")
        if self.generator is None:
            raise RuntimeError("RAGService is not initialized with a generator.")

        cleaned_question = question.strip() if question else ""
        if not cleaned_question:
            return GroundedAnswer(
                answer="No question provided.",
                sources=[],
                used_chunk_ids=[],
                confidence_label="insufficient_evidence",
            )

        # 1. Retrieve candidates
        candidates = self.retriever.search(cleaned_question, top_k=self.top_k)

        # 2. Generate grounded answer
        answer = self.generator.generate(cleaned_question, candidates)
        return answer
